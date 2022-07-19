import multiprocessing
import signal
import time
from collections import deque
from queue import Empty
import torch
import numpy as np
import psutil
import os

from torch.multiprocessing import Process as TorchProcess
from brain_agent.core.agents.agent_utils import create_agent
from brain_agent.utils.logger import log
from brain_agent.utils.utils import AttrDict
from brain_agent.utils.timing import Timing

from brain_agent.core.core_utils import TaskType, dict_of_lists_append, slice_mems, join_or_kill

class PolicyWorker:
    def __init__(self, cfg, obs_space, action_space, level_info, shared_buffer, policy_queue, actor_worker_queues,
                 policy_worker_queue, report_queue, policy_lock=None, resume_experience_collection_cv=None):
        log.info('Initializing policy worker %d', cfg.dist.world_rank)

        self.cfg = cfg

        self.obs_space = obs_space
        self.action_space = action_space
        self.level_info = level_info

        self.device = None
        self.actor_critic = None
        self.shared_model_weights = None

        self.policy_queue = policy_queue
        self.actor_worker_queues = actor_worker_queues
        self.report_queue = report_queue
        self.policy_worker_queue = policy_worker_queue

        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.initialized = False
        self.terminate = False
        self.initialized_event = multiprocessing.Event()
        self.initialized_event.clear()

        self.shared_buffer = shared_buffer

        self.tensors_individual_transitions = self.shared_buffer.tensors_individual_transitions
        self.policy_outputs = self.shared_buffer.policy_output_tensors

        self.latest_policy_version = -1
        self.num_policy_updates = 0

        self.requests = []

        self.total_num_samples = 0

        self.num_traj_buffers = shared_buffer.num_traj_buffers
        self.timing = Timing()

        self.initialized = False
        self.timing = Timing()

        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def init(self):
        self.policy_worker_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()

    def init_model(self, data):
        self.policy_worker_queue.put((TaskType.INIT_MODEL, data))

    def load_model(self):
        self.policy_worker_queue.put((TaskType.INIT_MODEL, None))

    def _init(self):
        if self.cfg.model.device == 'cuda':
            assert torch.cuda.device_count() == 1
            self.device = torch.device('cuda', index=0)
        else:
            self.device = torch.device('cpu')

        log.info('Policy worker %d initialized', self.cfg.dist.world_rank)
        self.initialized_event.set()

    def _init_model(self, init_model_data):

        self.actor_critic = create_agent(self.cfg, self.action_space, self.obs_space, self.level_info[
            'num_levels'], need_half=self.cfg.model.use_half_policy_worker)

        self.actor_critic.model_to_device(self.device)
        for p in self.actor_critic.parameters():
            p.requires_grad = False

        if self.cfg.model.core.core_type == 'trxl':
            max_batch_size = self.cfg.actor.num_workers * self.cfg.actor.num_envs_per_worker * 2
            mem_T_dim = self.cfg.model.core.mem_len
            mem_D_dim = self.shared_buffer.mems_dimensions[-1]
            mem_dones_D_dim = 1
            self.mems = torch.zeros([max_batch_size, mem_T_dim, mem_D_dim]).float().to(self.device)
            self.mems_dones = torch.zeros([max_batch_size, mem_T_dim, mem_dones_D_dim]).float().to(self.device)
            self.mems_actions = torch.zeros([max_batch_size, mem_T_dim, mem_dones_D_dim]).short().to(self.device)
            self.max_mems_buffer_len = self.shared_buffer.max_mems_buffer_len

            self.mems_buffer = None
            self.mems_dones_buffer = None

        if init_model_data is None:
            self._load_model()
        else:
            policy_version, state_dict, mems_buffer, mems_dones_buffer, mems_actions_buffer = init_model_data
            self.actor_critic.load_state_dict(state_dict)

            self.mems_buffer = mems_buffer
            self.mems_actions_buffer = mems_actions_buffer
            self.mems_dones_buffer = mems_dones_buffer

            self.shared_model_weights = state_dict
            self.latest_policy_version = policy_version

        log.info('Initialized model on the policy worker %d!', self.cfg.dist.world_rank)
        self.initialized = True

    def _load_model(self):
        ckpt = torch.load(self.cfg.test.checkpoint)
        env_step = ckpt['env_steps']
        policy_version = ckpt['train_step']
        state_dict = ckpt['model']

        self.actor_critic.load_state_dict(state_dict)
        self.shared_model_weights = state_dict
        self.latest_policy_version = policy_version

        self.mems_buffer = torch.zeros(self.shared_buffer.mems_dimensions).to(self.cfg.model.device)
        self.mems_dones_buffer = torch.zeros(self.shared_buffer.mems_dones_dimensions, dtype=torch.bool).to(
            self.cfg.model.device)
        self.mems_actions_buffer = torch.zeros(self.shared_buffer.mems_actions_dimensions).short().to(
            self.cfg.model.device)

        self.report_queue.put(dict(learner_env_steps=env_step))

    def _write_done_on_mems_dones_buffer(self, raw_index, actor_env_step):
        traj_tensors = self.shared_buffer.tensors_individual_transitions

        actor_idx, split_idx, env_idx, traj_buffer_idx, rollout_step = raw_index
        if rollout_step == 0:
            index_for_done = actor_idx, split_idx, env_idx, (traj_buffer_idx - 1) % self.num_traj_buffers, rollout_step - 1
        else:
            index_for_done = actor_idx, split_idx, env_idx, traj_buffer_idx, rollout_step - 1

        done = traj_tensors['dones'][index_for_done]
        index_for_mems_dones = actor_idx, split_idx, env_idx, (actor_env_step - 1) % self.max_mems_buffer_len
        self.mems_dones_buffer[index_for_mems_dones] = bool(done)

    def _handle_policy_steps(self):
        with torch.no_grad():
            with self.timing.timeit('deserialize'):
                observations = AttrDict()
                r_idx = 0
                first_rollout_list = []
                rollout_step_list = []
                actor_env_step_list = []
                actions = []
                rewards = []
                task_ids = []
                rnn_states = []
                dones = []

                traj_tensors = self.shared_buffer.tensors_individual_transitions

                for request in self.requests:
                    actor_idx, split_idx, request_data = request
                    if self.cfg.model.core.core_type == 'trxl':
                        for env_idx, traj_buffer_idx, rollout_step, first_rollout, actor_env_step in request_data:
                            index = actor_idx, split_idx, env_idx, traj_buffer_idx, rollout_step
                            with self.timing.timeit('write_done_on_mems_dones_buffer'):
                                self._write_done_on_mems_dones_buffer(index, actor_env_step)
                            self.timing['write_done_on_mems_dones_buffer'] *= len(self.requests) * len(request_data)
                            dict_of_lists_append(observations, traj_tensors['obs'], index)

                            s_idx = (actor_env_step - self.cfg.model.core.mem_len) % self.max_mems_buffer_len
                            e_idx = actor_env_step % self.max_mems_buffer_len
                            with self.timing.timeit('mems_copy'):
                                self.mems[r_idx], self.mems_dones[r_idx], self.mems_actions[r_idx] = slice_mems(
                                    self.mems_buffer, self.mems_dones_buffer, self.mems_actions_buffer, *index[:3], s_idx, e_idx)
                            self.timing['mems_copy'] *= len(self.requests) * len(request_data)
                            r_idx += 1
                            first_rollout_list.append(first_rollout)
                            rollout_step_list.append(rollout_step)
                            actor_env_step_list.append(actor_env_step)

                            actions.append(traj_tensors['prev_actions'][index])
                            rewards.append(traj_tensors['prev_rewards'][index])
                            task_ids.append(self.shared_buffer.task_ids[actor_idx][split_idx][env_idx].unsqueeze(0))
                            self.total_num_samples += 1
                    elif self.cfg.model.core.core_type == 'rnn':
                        for env_idx, traj_buffer_idx, rollout_step, first_rollout, actor_env_step in request_data:
                            index = actor_idx, split_idx, env_idx, traj_buffer_idx, rollout_step
                            rnn_states.append(traj_tensors['rnn_states'][index])
                            dones.append(traj_tensors['dones'][index])

                            first_rollout_list.append(first_rollout)
                            rollout_step_list.append(rollout_step)
                            actor_env_step_list.append(actor_env_step)
                            dict_of_lists_append(observations, traj_tensors['obs'], index)

                            actions.append(traj_tensors['prev_actions'][index])
                            rewards.append(traj_tensors['prev_rewards'][index])
                            task_ids.append(self.shared_buffer.task_ids[actor_idx][split_idx][env_idx].unsqueeze(0))
                            self.total_num_samples += 1

            with self.timing.timeit('reordering_mems'):
                if self.cfg.model.core.core_type == 'trxl':
                    n_batch = len(actor_env_step_list)
                    if self.cfg.model.core.mem_len > 0:
                        mems_dones = self.mems_dones[:n_batch]
                        actor_env_step = torch.tensor(actor_env_step_list)  # (n_batch)
                        mem_begin_index = self.actor_critic.core.get_mem_begin_index(mems_dones, actor_env_step)
                        self.actor_critic.actor_env_step = actor_env_step
                    else:
                        mem_begin_index = [0] * n_batch


            with self.timing.timeit('stack'):
                for key, x in observations.items():
                    observations[key] = torch.stack(x)
                actions = torch.stack(actions)
                rewards = torch.stack(rewards)
                task_ids = torch.stack(task_ids)
                if self.cfg.model.core.core_type == 'rnn':
                    dones = torch.stack(dones)
                    rnn_states = torch.stack(rnn_states)

            with self.timing.timeit('obs_to_device'):
                for key, x in observations.items():
                    device, dtype = self.actor_critic.device_and_type_for_input_tensor(key)
                    observations[key] = x.to(device).type(dtype)

                actions = actions.to(self.device).long()
                rewards = rewards.to(self.device).float()
                task_ids = task_ids.to(self.device).long()

                if self.cfg.model.core.core_type == 'rnn':
                    rnn_states = rnn_states.to(self.device).float()
                    dones = dones.to(self.device).float()
                    if self.cfg.model.use_half_policy_worker:
                        rnn_states = rnn_states.half()


            num_samples = actions.shape[0]
            if self.cfg.model.core.core_type == 'trxl':
                mems = self.mems[:num_samples]

            with self.timing.timeit('forward'):
                if self.cfg.model.use_half_policy_worker:
                    rewards = rewards.half()
                    if self.cfg.model.core.core_type == 'trxl':
                        mems = mems.half()
                    for key in observations:
                        obs = observations[key]
                        if obs.dtype == torch.float32:
                            observations[key] = obs.half()

                if self.cfg.model.core.core_type == 'trxl':
                    policy_outputs = self.actor_critic(observations, actions.squeeze(1), rewards.squeeze(1),
                                                       mems=mems.transpose(0, 1), mem_begin_index=mem_begin_index,
                                                       task_ids=task_ids,
                                                       with_action_distribution=False, from_learner=False)
                elif self.cfg.model.core.core_type == 'rnn':
                    policy_outputs = self.actor_critic(observations, actions.squeeze(1), rewards.squeeze(1),
                                                       rnn_states=rnn_states, dones=dones.squeeze(1), is_seq=False,
                                                       task_ids=task_ids, with_action_distribution=False)


            if self.cfg.model.core.core_type == 'trxl':
                midx = 0
                for request in self.requests:
                    actor_idx, split_idx, request_data = request
                    for env_idx, traj_buffer_idx, rollout_step, first_rollout, actor_env_step in request_data:
                        mem_index = actor_idx, split_idx, env_idx, actor_env_step % self.max_mems_buffer_len
                        self.mems_buffer[mem_index] = policy_outputs['mems'][midx]
                        midx += 1
                del policy_outputs['mems']


            for key, output_value in policy_outputs.items():
                policy_outputs[key] = output_value.cpu()

            policy_outputs.policy_version = torch.empty([num_samples]).fill_(self.latest_policy_version)

            # concat all tensors into a single tensor for performance
            output_tensors = []
            for policy_output in self.shared_buffer.policy_outputs:
                tensor_name = policy_output.name
                output_value = policy_outputs[tensor_name].float()
                if len(output_value.shape) == 1:
                    output_value.unsqueeze_(dim=1)
                output_tensors.append(output_value)

            output_tensors = torch.cat(output_tensors, dim=1)

            self._enqueue_policy_outputs(self.requests, output_tensors)

        self.requests = []

    def _enqueue_policy_outputs(self, requests, output_tensors):
        output_idx = 0

        outputs_ready = set()
        policy_outputs = self.shared_buffer.policy_output_tensors

        for request in requests:
            actor_idx, split_idx, request_data = request
            worker_outputs = policy_outputs[actor_idx, split_idx]
            for env_idx, traj_buffer_idx, rollout_step, _, _ in request_data: # writing at shared buffer
                worker_outputs[env_idx].copy_(output_tensors[output_idx])
                output_idx += 1

            outputs_ready.add((actor_idx, split_idx))

        for actor_idx, split_idx in outputs_ready:
            advance_rollout_request = dict(split_idx=split_idx)
            self.actor_worker_queues[actor_idx].put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _update_weights(self):
        learner_policy_version = self.shared_buffer.policy_versions[0].item()

        if self.latest_policy_version < learner_policy_version and self.shared_model_weights is not None:
            if self.policy_lock is not None:
                with self.timing.timeit('weight_update'):
                    with self.policy_lock:
                        self.actor_critic.load_state_dict(self.shared_model_weights)

            self.latest_policy_version = learner_policy_version

            if self.num_policy_updates % 10 == 0:
                log.info(
                    'Updated weights on worker %d, policy_version %d',
                    self.cfg.dist.world_rank, self.latest_policy_version,
                )

            self.num_policy_updates += 1

    def _run(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        psutil.Process().nice(2)

        torch.multiprocessing.set_sharing_strategy('file_system')
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cfg.dist.local_rank}'

        log.info('Initializing model on the policy worker %d...', self.cfg.dist.world_rank)
        log.info(f'POLICY worker {self.cfg.dist.world_rank}\tpid {os.getpid()}\tparent {os.getppid()}')

        torch.set_num_threads(1)

        last_report = last_cache_cleanup = time.time()
        last_report_samples = 0
        request_count = deque(maxlen=50)

        min_num_requests = self.cfg.actor.num_workers
        min_num_requests //= 3
        min_num_requests = max(1, min_num_requests)
        if self.cfg.test.is_test:
            min_num_requests = 1
        log.info('Min num requests: %d', min_num_requests)

        # Again, very conservative timer. Only wait a little bit, then continue operation.
        wait_for_min_requests = 0.025

        while not self.terminate:
            try:
                while self.shared_buffer.stop_experience_collection:
                    if self.resume_experience_collection_cv is not None:
                        with self.resume_experience_collection_cv:
                            self.resume_experience_collection_cv.wait(timeout=0.05)

                waiting_started = time.time()
                while len(self.requests) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                    try:
                        policy_requests = self.policy_queue.get_many(timeout=0.005)
                        self.requests.extend(policy_requests)
                    except Empty:
                        pass

                self._update_weights()

                with self.timing.timeit('one_step'):
                    if self.initialized:
                        if len(self.requests) > 0:
                            request_count.append(len(self.requests))
                            self._handle_policy_steps()


                try:
                    task_type, data = self.policy_worker_queue.get_nowait()
                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        self.terminate = True
                        break
                    elif task_type == TaskType.INIT_MODEL:
                        self._init_model(data)

                except Empty:
                    pass

                if time.time() - last_report > 3.0:
                    samples_since_last_report = self.total_num_samples - last_report_samples
                    stats = dict()
                    if len(request_count) > 0:
                        stats['avg_request_count'] = np.mean(request_count)

                    stats['times_policy_worker'] = {}
                    for key, value in self.timing.items():
                        stats['times_policy_worker'][key] = value

                    # self.report_queue.put(dict(
                    #     samples=samples_since_last_report, stats=stats,
                    # ))
                    self.report_queue.put(stats)

                    last_report = time.time()
                    last_report_samples = self.total_num_samples

                if time.time() - last_cache_cleanup > 300.0 or (
                        self.total_num_samples < 1000):
                    if self.cfg.model.device == 'cuda':
                        torch.cuda.empty_cache()
                    last_cache_cleanup = time.time()

            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on worker %d', self.cfg.dist.world_rank)
                self.terminate = True
                self.report_queue.put(('terminate', 'policy_worker'))

            except:
                log.exception('Unknown exception on policy worker')
                self.terminate = True
                self.report_queue.put(('terminate', 'policy_worker'))

        time.sleep(0.2)

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)