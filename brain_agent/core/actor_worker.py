import psutil
import random
import time
import os
import signal
import torch
from queue import Empty, Full
from threadpoolctl import threadpool_limits
from torch.multiprocessing import Process as TorchProcess
from brain_agent.utils.logger import log
from brain_agent.utils.utils import AttrDict
from brain_agent.core.core_utils import set_process_cpu_affinity, safe_put, safe_put_many, join_or_kill
from brain_agent.envs.env_utils import create_env
from brain_agent.core.core_utils import TaskType
from brain_agent.utils.timing import Timing


# TODO: actions -> action

class ActorWorker:
    """
    ActorWorker is responsible for running the environment(s) with the action(s) that may be provided by the policy workers.

    Args:
        cfg (['brain_agent.utils.utils.AttrDict'], 'AttrDict'):
            Global configuration in a form of AttrDict, a dictionary whose values can be accessed
        obs_space ('gym.spaces'):
            Observation space.
        action_space ('gym.spaces.discrete.Discrete'):
            Action space object. Currently only supports discrete action spaces.
        shared_buffer (['brain_agent.core.shared_buffer.SharedBuffer']):
            Shared buffer object that stores collected rollouts.
        actor_worker_queue ('faster_fifo.Queue'):
            Task queue for the actor worker.
        policy_queue ('faster_fifo.Queue'):
            Action request queue for the policy in the policy worker. Not to be confused with policy_worker_queue.
        report_queue ('faster_fifo.Queue'):
            Task queue for reporting. This is where various workers dump information to log.
        learner_worker_queue ('faster_fifo.Queue', *optional*):
            Task queue for the learner worker. This is where other processes dump tasks for the learner.
    """
    def __init__(self, cfg, obs_space, action_space, actor_idx, shared_buffer, actor_worker_queue, policy_queue,
                 report_queue, learner_worker_queue=None):
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor_idx = actor_idx
        self.shared_buffer = shared_buffer
        self.actor_worker_queue = actor_worker_queue
        self.policy_queue = policy_queue
        self.learner_worker_queue = learner_worker_queue
        self.report_queue = report_queue

        self.env_runners = None

        self.num_splits = self.cfg.actor.num_splits
        self.num_envs_per_worker = self.cfg.actor.num_envs_per_worker
        self.num_envs_per_vector = self.cfg.actor.num_envs_per_worker // self.num_splits
        assert self.num_envs_per_worker >= self.num_splits
        assert self.num_envs_per_worker % self.num_splits == 0, 'Vector size should be divisible by num_splits'

        self.terminate = False
        self.num_complete_rollouts = 0
        self.timing = Timing()


        self.process = TorchProcess(target=self._run, daemon=True)
        self.process.start()

    def _init(self):
        log.info('Initializing envs for env runner %d...', self.actor_idx)

        threadpool_limits(limits=1, user_api=None)

        if self.cfg.actor.set_workers_cpu_affinity:
            set_process_cpu_affinity(self.actor_idx, self.cfg.actor.num_workers, self.cfg.dist.local_rank,
                                 self.cfg.dist.nproc_per_node)
        psutil.Process().nice(10) # learner: 0

        self.env_runners = []
        for split_idx in range(self.num_splits):
            env_runner = VectorEnvRunner(
                self.cfg, self.num_envs_per_vector, self.actor_idx, split_idx,
                self.shared_buffer
            )
            env_runner.init()
            self.env_runners.append(env_runner)

    def _terminate(self):
        for env_runner in self.env_runners:
            env_runner.close()
        self.terminate = True

    def _enqueue_policy_request(self, split_idx, requests):
        """Distribute action requests to their corresponding queues."""
        policy_request = (self.actor_idx, split_idx, requests)
        self.policy_queue.put(policy_request)

    def _enqueue_complete_rollouts(self, split_idx, complete_rollouts):
        """Send complete rollouts from VectorEnv to the learner."""
        if self.cfg.test.is_test:
            return

        traj_buffer_idx = complete_rollouts['traj_buffer_idx']

        env_runner = self.env_runners[split_idx]
        env_runner.traj_tensors_available[:, traj_buffer_idx] = 0

        self.learner_worker_queue.put((TaskType.TRAIN, complete_rollouts))

    def _report_stats(self, stats):
        safe_put_many(self.report_queue, stats, queue_name='report')

    def _handle_reset(self):
        for split_idx, env_runner in enumerate(self.env_runners):
            policy_inputs = env_runner.reset(self.report_queue)
            self._enqueue_policy_request(split_idx, policy_inputs)

        log.info('Finished reset for worker %d', self.actor_idx)
        safe_put(self.report_queue, dict(finished_reset=self.actor_idx), queue_name='report')

    def _advance_rollouts(self, data):
        split_idx = data['split_idx']

        runner = self.env_runners[split_idx]
        policy_request, complete_rollouts, episodic_stats = runner.advance_rollouts(data, self.timing)

        if complete_rollouts:
            self._enqueue_complete_rollouts(split_idx, complete_rollouts)

            if self.num_complete_rollouts == 0:

                delay = (float(self.actor_idx) / self.cfg.actor.num_workers) * \
                        self.cfg.env.decorrelate_experience_max_seconds
                log.info(
                    'Worker %d, sleep for %.3f sec to decorrelate experience collection',
                    self.actor_idx, delay,
                )
                time.sleep(delay)
                log.info('Worker %d awakens!', self.actor_idx)

            self.num_complete_rollouts += len(complete_rollouts)

        if policy_request is not None:
            self._enqueue_policy_request(split_idx, policy_request)

        if episodic_stats:
            self._report_stats(episodic_stats)

    def _run(self):
        log.info('Initializing vector env runner %d...', self.actor_idx)
        log.info(f'ACTOR worker {self.actor_idx}\tpid {os.getpid()}\tparent {os.getppid()}')

        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        torch.multiprocessing.set_sharing_strategy('file_system')

        last_report = time.time()
        with torch.no_grad():
            while not self.terminate:
                try:
                    try:
                        with self.timing.timeit('wait_actor'):
                            tasks = self.actor_worker_queue.get_many(timeout=0.1)
                    except Empty:
                        tasks = []

                    for task in tasks:
                        task_type, data = task

                        if task_type == TaskType.INIT:
                            self._init()
                            continue

                        if task_type == TaskType.TERMINATE:
                            self._terminate()
                            break

                        # handling actual workload
                        if task_type == TaskType.ROLLOUT_STEP:
                            with self.timing.timeit('one_step'):
                                self._advance_rollouts(data)

                        elif task_type == TaskType.RESET:
                            self._handle_reset()

                    if time.time() - last_report > 5.0 and 'one_step' in self.timing:
                        stats = {}
                        stats['times_actor_worker'] = {}
                        for key, value in self.timing.items():
                            stats['times_actor_worker'][key] = value
                        safe_put(self.report_queue, stats)

                        last_report = time.time()

                except RuntimeError as exc:
                    log.warning('Error while processing data w: %d, exception: %s', self.actor_idx, exc)
                    log.warning('Terminate process...')
                    self.terminate = True
                    self.report_queue.put(('terminate', 'actor_worker'))
                except KeyboardInterrupt:
                    self.terminate = True
                except:
                    log.exception('Unknown exception in rollout worker')
                    self.report_queue.put(('terminate', 'actor_worker'))
                    self.terminate = True

        if self.actor_idx <= 1:
            time.sleep(0.1)
            log.info(
                'Env runner %d, CPU aff. %r, rollouts %d',
                self.actor_idx, psutil.Process().cpu_affinity(), self.num_complete_rollouts,
            )

    def init(self):
        self.actor_worker_queue.put((TaskType.INIT, None))

    def request_reset(self):
        self.actor_worker_queue.put((TaskType.RESET, None))

    def request_step(self, split, actions):
        data = (split, actions)
        self.actor_worker_queue.put((TaskType.ROLLOUT_STEP, data))

    def close(self):
        self.actor_worker_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)


class VectorEnvRunner:
    def __init__(self, cfg, num_envs, actor_idx, split_idx, shared_buffer):
        self.cfg = cfg

        self.num_envs = num_envs
        self.actor_idx = actor_idx
        self.actor_idx_node = actor_idx * self.cfg.dist.nproc_per_node + self.cfg.dist.local_rank
        self.actor_idx_world = actor_idx * self.cfg.dist.world_size + self.cfg.dist.world_rank
        self.split_idx = split_idx

        self.rollout_step = 0
        self.traj_buffer_idx = 0  # current shared trajectory buffer to use

        self.first_rollout = [True] * shared_buffer.num_traj_buffers
        self.shared_buffer = shared_buffer

        index = (actor_idx, split_idx)

        self.traj_tensors = shared_buffer.tensors_individual_transitions.index(index)
        self.traj_tensors_available = shared_buffer.is_traj_tensor_available[index]
        self.num_traj_buffers = shared_buffer.num_traj_buffers
        self.policy_outputs = shared_buffer.policy_outputs
        self.policy_output_tensors = shared_buffer.policy_output_tensors[index]
        self.task_id = shared_buffer.task_ids[index]

        self.envs, self.actor_states, self.episode_rewards = [], [], []

    def init(self):

        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            env_id = self.actor_idx_world * self.cfg.actor.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.actor_idx_world, vector_index=vector_idx, env_id=env_id,
            )

            env = create_env(self.cfg, env_config=env_config)

            env.seed(env_id + self.cfg.seed * self.cfg.actor.num_workers * self.cfg.actor.num_envs_per_worker *
                     self.cfg.dist.world_size)

            self.envs.append(env)
            self.task_id[env_i] = env.unwrapped.task_id

            traj_tensors = self.traj_tensors.index(env_i)
            actor_state = ActorState(
                self.cfg, env, self.actor_idx, self.split_idx, env_i, traj_tensors,
                self.num_traj_buffers, self.policy_outputs, self.policy_output_tensors[
                    env_i]
            )
            episode_rewards_env = 0.0

            self.actor_states.append(actor_state)
            self.episode_rewards.append(episode_rewards_env)

    def _process_policy_outputs(self):

        for env_i in range(self.num_envs):
            actor_state = self.actor_states[env_i]

            # via shared memory mechanism the new data should already be copied into the shared tensors

            policy_outputs = torch.split(
                actor_state.policy_output_tensors,
                split_size_or_sections=actor_state.policy_output_sizes,
                dim=0,
            )
            policy_outputs_dict = dict()
            for tensor_idx, name in enumerate(actor_state.policy_output_names):
                if name == 'rnn_states' and self.cfg.model.core.core_type == 'rnn':
                    new_rnn_state = policy_outputs[tensor_idx]
                else:
                    policy_outputs_dict[name] = policy_outputs[tensor_idx]

            actor_state.set_trajectory_data(policy_outputs_dict, self.traj_buffer_idx, self.rollout_step)
            actor_state.last_actions = policy_outputs_dict['actions']
            actor_state.prev_actions = policy_outputs_dict['actions']

            if self.cfg.model.core.core_type == 'rnn':
                actor_state.last_rnn_state = new_rnn_state

    def _process_env_step(self, new_obs, rewards, dones, infos, env_i):

        episodic_stats = []
        actor_state = self.actor_states[env_i]

        actor_state.record_env_step(
            rewards, dones, infos, self.traj_buffer_idx, self.rollout_step,
        )
        actor_state.last_obs = new_obs
        actor_state.prev_rewards = float(rewards)

        actor_state.actor_env_step += 1
        if self.cfg.model.core.core_type=='rnn':
            actor_state.update_rnn_state(dones)

        if dones:
            actor_state.prev_rewards = 0.
            actor_state.prev_actions = torch.Tensor([-1])
            episodic_stat = dict()
            episodic_stat['episodic_stats'] = infos['episodic_stats']
            episodic_stats.append(episodic_stat)

        return episodic_stats

    def _finalize_trajectories(self):

        rollouts = []
        for env_i in range(self.num_envs):
            actor_state = self.actor_states[env_i]
            rollout = actor_state.finalize_trajectory(self.rollout_step,
                                                      self.first_rollout[self.traj_buffer_idx])
            rollout['task_idx'] = self.task_id[env_i]
            rollouts.append(rollout)

        return dict(rollouts=rollouts, traj_buffer_idx=self.traj_buffer_idx)

    def _format_policy_request(self):

        policy_request = []

        for env_i in range(self.num_envs):
            actor_state = self.actor_states[env_i]
            data = (env_i, self.traj_buffer_idx, self.rollout_step, self.first_rollout[self.traj_buffer_idx],
                    actor_state.actor_env_step)
            policy_request.append(data)

        return policy_request

    def _prepare_next_step(self):

        for env_i in range(len(self.envs)):
            actor_state = self.actor_states[env_i]
            if self.cfg.model.core.core_type == 'trxl':
                policy_inputs = dict(obs=actor_state.last_obs)
            elif self.cfg.model.core.core_type == 'rnn':
                policy_inputs = dict(obs=actor_state.last_obs, rnn_states=actor_state.last_rnn_state)
            actor_state.traj_tensors['dones'][self.traj_buffer_idx, self.rollout_step].fill_(actor_state.done)
            actor_state.traj_tensors['prev_actions'][self.traj_buffer_idx, self.rollout_step].copy_(
                actor_state.prev_actions.type(
                    actor_state.traj_tensors['prev_actions'][self.traj_buffer_idx, self.rollout_step].type()))
            actor_state.traj_tensors['prev_rewards'][self.traj_buffer_idx, self.rollout_step].fill_(
                actor_state.prev_rewards)

            actor_state.set_trajectory_data(policy_inputs, self.traj_buffer_idx, self.rollout_step)

            if self.rollout_step == 0 and self.first_rollout[self.traj_buffer_idx]:  # start of the new trajectory,
                self.first_rollout[self.traj_buffer_idx] = False

    def reset(self, report_queue):

        for env_i, e in enumerate(self.envs):
            obs = e.reset()

            env_i_split = self.num_envs * self.split_idx + env_i
            if self.cfg.env.decorrelate_envs_on_one_worker and not self.cfg.test.is_test:
                decorrelate_steps = self.cfg.optim.rollout * env_i_split + self.cfg.optim.rollout * random.randint(0, 4)

                log.info('Decorrelating experience for %d frames...', decorrelate_steps)
                for decorrelate_step in range(decorrelate_steps):
                    action = e.action_space.sample()
                    obs, rew, done, info = e.step(action)

            actor_state = self.actor_states[env_i]
            actor_state.set_trajectory_data(dict(obs=obs), self.traj_buffer_idx, self.rollout_step)
            actor_state.traj_tensors['prev_actions'][self.traj_buffer_idx, self.rollout_step][0].fill_(-1)
            actor_state.traj_tensors['prev_rewards'][self.traj_buffer_idx, self.rollout_step][0] = 0.
            actor_state.traj_tensors['dones'][self.traj_buffer_idx, self.rollout_step][0].fill_(False)

            safe_put(report_queue, dict(initialized_env=(self.actor_idx, self.split_idx, env_i, self.task_id.tolist())),
                     queue_name='report')

        policy_request = self._format_policy_request()
        return policy_request

    def advance_rollouts(self, data, timing):

        self._process_policy_outputs()

        complete_rollouts, episodic_stats = [], []
        timing['env_step'], timing['overhead'] = 0, 0


        for env_i, e in enumerate(self.envs):
            with timing.timeit('_env_step'):
                actions = self.actor_states[env_i].last_actions.type(torch.int32).numpy().item()
                new_obs, rewards, dones, infos = e.step(actions)

            timing['env_step'] += timing['_env_step']

            with timing.timeit('_overhead'):
                stats = self._process_env_step(new_obs, rewards, dones, infos, env_i)
                episodic_stats.extend(stats)
            timing['overhead'] += timing['_overhead']

        self.rollout_step += 1
        if self.rollout_step == self.cfg.optim.rollout:
            # finalize and serialize the trajectory if we have a complete rollout
            complete_rollouts = self._finalize_trajectories()
            self.rollout_step = 0
            self.traj_buffer_idx = (self.traj_buffer_idx + 1) % self.num_traj_buffers

            if self.traj_tensors_available[:, self.traj_buffer_idx].min() == 0:
                with timing.timeit('wait_traj_buffer'):
                    self.wait_for_traj_buffers()

        self._prepare_next_step()
        policy_request = self._format_policy_request()

        return policy_request, complete_rollouts, episodic_stats

    def wait_for_traj_buffers(self):
        print_warning = True
        while self.traj_tensors_available[:, self.traj_buffer_idx].min() == 0:
            if print_warning:
                log.warning(
                    'Waiting for trajectory buffer %d on actor %d-%d',
                    self.traj_buffer_idx, self.actor_idx, self.split_idx,
                )
                print_warning = False
            time.sleep(0.002)

    def close(self):
        for e in self.envs:
            e.close()


class ActorState:
    def __init__(self, cfg, env, actor_idx, split_idx, env_idx, traj_tensors, num_traj_buffers, policy_outputs_info,
                 policy_output_tensors):

        self.cfg = cfg
        self.env = env
        self.actor_idx = actor_idx
        self.split_idx = split_idx
        self.env_idx = env_idx

        if not self.cfg.model.core.core_type == 'rnn':
            self.last_rnn_state = None

        self.traj_tensors = traj_tensors
        self.num_traj_buffers = num_traj_buffers

        self.policy_output_names = [p.name for p in policy_outputs_info]
        self.policy_output_sizes = [p.size for p in policy_outputs_info]
        self.policy_output_tensors = policy_output_tensors

        self.prev_actions = None
        self.prev_rewards = None

        self.last_actions = None
        self.last_policy_steps = None

        self.num_trajectories = 0
        self.rollout_env_steps = 0

        self.last_episode_reward = 0
        self.last_episode_duration = 0
        self.last_episode_true_reward = 0
        self.last_episode_extra_stats = dict()

        self.actor_env_step = 0

    def set_trajectory_data(self, data, traj_buffer_idx, rollout_step):

        index = (traj_buffer_idx, rollout_step)
        self.traj_tensors.set_data(index, data)

    def record_env_step(self, reward, done, info, traj_buffer_idx, rollout_step):

        self.traj_tensors['rewards'][traj_buffer_idx, rollout_step][0] = float(reward)
        self.traj_tensors['dones'][traj_buffer_idx, rollout_step][0] = done

        env_steps = info.get('num_frames', 1)
        self.rollout_env_steps += env_steps
        self.last_episode_duration += env_steps

        if done:
            self.done = True
            self.last_episode_extra_stats = info.get('episode_extra_stats', dict())
        else:
            self.done = False

    def finalize_trajectory(self, rollout_step, first_rollout=None):

        t_id = f'{self.actor_idx}_{self.split_idx}_{self.env_idx}_{self.num_trajectories}'
        mem_idx = (
            self.actor_idx,
            self.split_idx,
            self.env_idx,
            rollout_step,
            self.actor_env_step,
            first_rollout
        )
        traj_dict = dict(
            t_id=t_id, length=rollout_step, env_steps=self.rollout_env_steps, mem_idx=mem_idx,
            actor_idx=self.actor_idx, split_idx=self.split_idx, env_idx=self.env_idx
        )

        self.num_trajectories += 1
        self.rollout_env_steps = 0

        return traj_dict

    def episodic_stats(self):
        stats = dict(reward=self.last_episode_reward, len=self.last_episode_duration)

        stats['episode_extra_stats'] = self.last_episode_extra_stats

        report = dict(episodic=stats)
        self.last_episode_reward = self.last_episode_duration = self.last_episode_true_reward = 0
        self.last_episode_extra_stats = dict()
        return report

    def update_rnn_state(self, done):
        """If we encountered an episode boundary, reset rnn states to their default values."""
        if done:
            self.last_rnn_state.fill_(0.0)
