import os
import signal
import threading
import time
from collections import OrderedDict, deque
from os.path import join
from queue import Empty, Queue, Full
import shutil

import numpy as np
import psutil
import torch
from torch.multiprocessing import Process, Event as MultiprocessingEvent

from brain_agent.core.core_utils import TaskType, iter_dicts_recursively, copy_dict_structure, slice_mems, iterate_recursively
from brain_agent.core.models.action_distributions import CategoricalActionDistribution
from brain_agent.core.agents.agent_utils import create_agent
from brain_agent.core.models.model_utils import EPS, to_scalar
from brain_agent.core.algos.vtrace import calculate_vtrace
from brain_agent.utils.utils import list_of_dicts_to_dict_of_lists, log, AttrDict, get_checkpoint_dir, get_checkpoints
from brain_agent.utils.timing import Timing
from brain_agent.core.core_utils import join_or_kill, safe_get, safe_put
from brain_agent.utils.dist_utils import dist_init, dist_broadcast_model, dist_reduce_gradient, dist_all_reduce_buffers


class LearnerWorker:
    def __init__(
        self, cfg, obs_space, action_space, level_info, report_queue, learner_worker_queue, policy_worker_queue,
            shared_buffer,
        policy_lock, resume_experience_collection_cv,
    ):
        log.info('Initializing the learner %d', cfg.dist.world_rank)

        self.cfg = cfg

        self.should_save_model = True  # set to true if we need to save the model to disk on the next training iteration

        self.terminate = False
        self.num_batches_processed = 0

        self.obs_space = obs_space
        self.action_space = action_space
        self.level_info = level_info

        self.shared_buffer = shared_buffer

        # deferred initialization
        self.rollout_tensors = None
        self.traj_tensors_available = None
        self.policy_versions = None
        self.stop_experience_collection = None

        self.stop_experience_collection_num_msgs = self.resume_experience_collection_num_msgs = 0

        self.device = None
        self.actor_critic = None
        self.aux_loss_module = None
        self.optimizer = None
        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.report_queue = report_queue
        self.learner_worker_queue = learner_worker_queue

        self.initialized_event = MultiprocessingEvent()
        self.initialized_event.clear()

        self.model_saved_event = MultiprocessingEvent()
        self.model_saved_event.clear()

        # queues corresponding to policy workers using the same policy
        # we send weight updates via these queues
        self.policy_worker_queue = policy_worker_queue

        self.experience_buffer_queue = None  # deferred initialization

        self.tensor_batch_pool = self.tensor_batcher = None

        self.with_training = True  # set to False for debugging no-training regime

        self.training_thread = None
        self.train_thread_initialized = None

        self.is_training = False

        self.train_step = self.env_steps = 0

        self.last_summary_time = 0

        self.last_saved_time = self.last_milestone_time = self.optimizer_step_count = 0
        self.timing = Timing()

        self.process = Process(target=self._run, daemon=True)

        # deferred initialization
        self.exploration_loss_func = None

        if self.cfg.log.save_milestones_step > 0:
            self.next_milestone_step = self.cfg.log.save_milestones_step

    def start_process(self):
        self.process.start()

    def deferred_initialization(self):
        self.rollout_tensors = self.shared_buffer.tensor_trajectories
        self.policy_versions = self.shared_buffer.policy_versions
        self.traj_tensors_available = self.shared_buffer.is_traj_tensor_available
        self.stop_experience_collection = self.shared_buffer.stop_experience_collection

        self.experience_buffer_queue = Queue()

        self.tensor_batch_pool = ObjectPool()
        self.tensor_batcher = TensorBatcher(self.tensor_batch_pool)

        self.train_thread_initialized = threading.Event()

        self.exploration_loss_func = self._entropy_exploration_loss

        self.initialize()

    def _init(self):
        log.info('Waiting for the learner to initialize...')
        self.train_thread_initialized.wait()
        log.info('Learner %d initialized', self.cfg.dist.world_rank)
        self.initialized_event.set()

    def _terminate(self):
        self.terminate = True

    def _broadcast_model_weights(self):
        state_dict = self.actor_critic.state_dict()
        policy_version = self.train_step
        log.debug('Broadcast model weights for model version %d', policy_version)
        mems_buffer = self.mems_buffer
        mems_dones_buffer = self.mems_dones_buffer
        mems_actions_buffer = self.mems_actions_buffer
        model_state = (policy_version, state_dict, mems_buffer, mems_dones_buffer, mems_actions_buffer)
        self.policy_worker_queue.put((TaskType.INIT, None))
        self.policy_worker_queue.put((TaskType.INIT_MODEL, model_state))

    def _prepare_train_buffer(self, rollouts, macro_batch_size):
        trajectories = [AttrDict(r['t']) for r in rollouts]

        buffer = AttrDict()

        # by the end of this loop the buffer is a dictionary containing lists of numpy arrays
        buffer['task_idx'] = []
        for i, t in enumerate(trajectories):
            for key, x in t.items():
                if key not in buffer:
                    buffer[key] = []
                buffer[key].append(x)
            task_idx_t = torch.Tensor([rollouts[i]['task_idx']] * self.cfg.optim.rollout).unsqueeze(1)
            buffer['task_idx'].append(task_idx_t)

        # convert lists of dict observations to a single dictionary of lists
        for key, x in buffer.items():
            if isinstance(x[0], (dict, OrderedDict)):
                buffer[key] = list_of_dicts_to_dict_of_lists(x)

        # concatenate rollouts from different workers into a single batch efficiently
        # that is, if we already have memory for the buffers allocated, we can just copy the data into
        # existing cached tensors instead of creating new ones. This is a performance optimization.
        with self.timing.timeit('batching'):
            use_pinned_memory = self.cfg.model.device == 'cuda'
            buffer = self.tensor_batcher.cat(buffer, macro_batch_size, use_pinned_memory)

        # Mark buffer for used rollouts free
        with self.timing.timeit('buff_ready'):
            for r in rollouts:
                self.traj_tensors_available[r.actor_idx, r.split_idx][r.env_idx, r.traj_buffer_idx] = 1

        with self.timing.timeit('tensors_gpu_float'):
            device_buffer = self._copy_train_data_to_device(buffer)

        # will squeeze actions only in simple categorical case
        tensors_to_squeeze = [
            'actions', 'log_prob_actions', 'policy_version', 'values',
            'rewards', 'dones', 'rewards_cpu', 'dones_cpu',
        ]
        for tensor_name in tensors_to_squeeze:
            device_buffer[tensor_name].squeeze_()

        # we no longer need the cached buffer, and can put it back into the pool
        self.tensor_batch_pool.put(buffer)
        return device_buffer

    def _process_macro_batch(self, rollouts, batch_size):
        assert batch_size % self.cfg.optim.rollout == 0

        samples = env_steps = 0
        for rollout in rollouts:
            samples += rollout['length']
            env_steps += rollout['env_steps']

        with self.timing.timeit('prepare'):
            buffer = self._prepare_train_buffer(rollouts, batch_size)
            self.experience_buffer_queue.put((buffer, batch_size, samples, env_steps))

    def _process_rollouts(self, rollouts):
        batch_size = self.cfg.optim.batch_size
        rollouts_in_macro_batch = batch_size // self.cfg.optim.rollout

        if len(rollouts) < rollouts_in_macro_batch:
            return rollouts

        if len(rollouts) >= rollouts_in_macro_batch:
            # process newest rollouts
            rollouts_to_process = rollouts[:rollouts_in_macro_batch]
            rollouts = rollouts[rollouts_in_macro_batch:]

            self._process_macro_batch(rollouts_to_process, batch_size)
            # log.info('Unprocessed rollouts: %d (%d samples)', len(rollouts), len(rollouts) * self.cfg.rollout)
        return rollouts

    def _should_save_summaries(self):
        summaries_every_seconds = 60
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False
        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1
        if self.distenv.master:
            if self.cfg.log.save_milestones_step > 0:
                if self.env_steps >= self.next_milestone_step:
                    self.should_save_model = True
            self._maybe_save()

    def _maybe_save(self):
        if time.time() - self.last_saved_time >= self.cfg.log.save_every_sec or self.should_save_model:
            self._save()
            self.model_saved_event.set()
            self.should_save_model = False
            self.last_saved_time = time.time()

    def _get_checkpoint_dict(self):
        checkpoint = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.aux_loss_module is not None:
            checkpoint['aux_loss_module'] = self.aux_loss_module.state_dict()

        return checkpoint

    def _save(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = get_checkpoint_dir(self.cfg)
        tmp_filepath = join(checkpoint_dir, '.temp_checkpoint')
        checkpoint_name = f'checkpoint_{self.train_step:09d}_{self.env_steps}.pth'
        filepath = join(checkpoint_dir, checkpoint_name)
        log.info('Saving %s...', tmp_filepath)
        torch.save(checkpoint, tmp_filepath)
        log.info('Renaming %s to %s', tmp_filepath, filepath)
        os.rename(tmp_filepath, filepath)

        if self.cfg.log.save_milestones_step > 0:
            if self.env_steps > self.next_milestone_step:
                milestones_dir = (join(checkpoint_dir, 'milestones'))
                os.makedirs(milestones_dir, exist_ok=True)
                milestone_path = join(milestones_dir, f'{checkpoint_name}.milestone')
                log.debug('Saving a milestone %s', milestone_path)
                shutil.copy(filepath, milestone_path)
                self.next_milestone_step += self.cfg.log.save_milestones_step

        while len(get_checkpoints(checkpoint_dir)) > self.cfg.learner.keep_checkpoints:
            oldest_checkpoint = get_checkpoints(checkpoint_dir)[0]
            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

    # @staticmethod
    def _policy_loss(self, ratio, log_prob_actions, adv, clip_ratio_low, clip_ratio_high, ppo=False, exclude_last=False):
        if ppo:
            clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
            loss_unclipped = ratio * adv
            loss_clipped = clipped_ratio * adv
            loss = torch.min(loss_unclipped, loss_clipped)
            loss = -loss.mean()
        else:
            loss = log_prob_actions * adv
            if exclude_last:
                loss_mask = torch.ones_like(loss).view(-1, self.cfg.optim.rollout)
                loss_mask[:, -1] = 0
                loss = loss * loss_mask.contiguous().view(-1)
            loss = -loss.mean()

        return loss

    def _value_loss(self, new_values, target, exclude_last=False):
        value_loss = (new_values - target).pow(2)
        if exclude_last:
            loss_mask = torch.ones_like(value_loss).view(-1, self.cfg.optim.rollout)
            loss_mask[:, -1] = 0
            value_loss = value_loss * loss_mask.contiguous().view(-1)
        value_loss = value_loss.mean()

        value_loss *= self.cfg.learner.value_loss_coeff
        return value_loss

    def _kl_loss(self, action_space, action_logits, action_distribution):
        old_action_distribution = CategoricalActionDistribution(action_space, action_logits)
        kl_loss = action_distribution.kl_divergence(old_action_distribution)
        kl_loss = kl_loss.mean()

        kl_loss *= self.cfg.kl_loss_coeff

        return kl_loss

    def _entropy_exploration_loss(self, action_distribution, exclude_last=False):
        entropy = action_distribution.entropy()
        if exclude_last:
            entropy = entropy.view(-1, self.cfg.optim.rollout)[:, :-1].contiguous().view(-1)
        entropy_loss = -self.cfg.learner.exploration_loss_coeff * entropy.mean()
        return entropy_loss

    def _reconstruction_loss(self, obs_dict, target, mean, scale):
        obs_normalized = (obs_dict['obs'] - mean) / scale
        loss = self.cfg.learner.reconstruction_loss_coeff * torch.mean(
            torch.pow(obs_normalized - target, 2))
        return loss

    def _curr_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _update_lr(self, new_lr):
        if new_lr != self._curr_lr():
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def _prepare_observations(self, obs_tensors, gpu_buffer_obs):
        for d, gpu_d, k, v, _ in iter_dicts_recursively(obs_tensors, gpu_buffer_obs):
            device, dtype = self.actor_critic.device_and_type_for_input_tensor(k)
            tensor = v.detach().to(device, copy=True).type(dtype)
            gpu_d[k] = tensor

    def _copy_train_data_to_device(self, buffer):
        device_buffer = copy_dict_structure(buffer)

        for key, item in buffer.items():
            if key == 'obs':
                self._prepare_observations(item, device_buffer['obs'])
            else:
                device_tensor = item.detach().to(self.device, copy=True, non_blocking=True)
                device_buffer[key] = device_tensor.float()

        device_buffer['dones_cpu'] = buffer.dones.to('cpu', copy=True, non_blocking=True).float()
        device_buffer['rewards_cpu'] = buffer.rewards.to('cpu', copy=True, non_blocking=True).float()

        return device_buffer

    def _forward_model(self, mb):
        decode = self.cfg.learner.use_decoder
        recurrence = self.cfg.optim.rollout
        rollouts_in_batch = self.cfg.optim.batch_size // recurrence

        head_outputs = self.actor_critic.forward_head(
            mb.obs, mb.prev_actions.squeeze().long(), mb.prev_rewards.squeeze(), decode=decode)

        if self.cfg.model.core.core_type == 'trxl':
            mems_indices = mb.mems_indices.type(torch.long)
            for bidx in range(mems_indices.shape[0]):
                self.mems[:, bidx], self.mems_dones[:, bidx], self.mems_actions[:, bidx] = slice_mems(
                    self.mems_buffer, self.mems_dones_buffer, self.mems_actions_buffer, *mems_indices[bidx, :5])
            dones = mb.dones.view(rollouts_in_batch, recurrence).transpose(0, 1)

            actor_env_step_of_rollout_begin = mb.actor_env_step - recurrence
            mems_dones = self.mems_dones.transpose(0, 1)
            mem_begin_index = self.actor_critic.core.get_mem_begin_index(mems_dones,
                                                                         actor_env_step_of_rollout_begin)
            with self.timing.timeit('bptt'):
                core_outputs, _ = self.actor_critic.forward_core_transformer(
                    head_outputs, self.mems, mem_begin_index=mem_begin_index, dones=dones, from_learner=True)

        elif self.cfg.model.core.core_type == 'rnn':
            with self.timing.timeit('bptt'):
                head_output_seq, rnn_states, inverted_select_inds = self.actor_critic.core.build_rnn_inputs(
                    head_outputs, mb.dones_cpu, mb.rnn_states, recurrence)

                core_output_seq, _ = self.actor_critic.forward_core_rnn(head_output_seq, rnn_states,
                                                                                        mb.dones, is_seq=True)
                core_outputs = core_output_seq.data.index_select(0, inverted_select_inds)
        else:
            raise NotImplementedError

        # calculate policy tail outside of recurrent loop
        with self.timing.timeit('tail'):
            result = self.actor_critic.forward_tail(core_outputs, mb.task_idx.to(torch.long),
                                                with_action_distribution=True)

        if self.cfg.learner.use_aux_future_pred_loss:
            future_pred_loss = self.actor_critic.future_pred_module.calc_loss(mb,
                  core_outputs, self.mems, self.mems_actions, mems_dones, mem_begin_index, rollouts_in_batch,recurrence)
            result.future_pred_loss = future_pred_loss

        return result


    def _train(self, mb):
        # Initialize some variables
        policy_version_before_train = self.train_step
        epoch_actor_losses = []
        # recent mean KL-divergences per minibatch, this used by LR schedulers
        recent_kls = []
        # V-trace parameters
        rho_hat = torch.Tensor([self.cfg.learner.vtrace_rho])
        c_hat = torch.Tensor([self.cfg.learner.vtrace_c])
        clip_ratio_high = 1.0 + self.cfg.learner.ppo_clip_ratio  # e.g. 1.1
        clip_ratio_low = 1.0 / clip_ratio_high  # to handle large clip ratio e.g. 2

        clip_value = self.cfg.learner.ppo_clip_value
        recurrence = self.cfg.optim.rollout
        rollouts_in_batch = self.cfg.optim.batch_size // recurrence

        num_sgd_steps = 0

        stats_and_summaries = None
        if not self.with_training:
            return stats_and_summaries
        summary_this_epoch = force_summaries = False

        # Forward entire model using given minibatch
        with self.timing.timeit('forward'):
            result = self._forward_model(mb)

        # Predictions
        action_distribution = result.action_distribution
        values = result.values.squeeze()
        normalized_values = result.normalized_values.squeeze()
        log_prob_actions = action_distribution.log_prob(mb.actions)
        ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

        # super large/small values can cause numerical problems and are probably noise anyway
        ratio = torch.clamp(ratio, 0.01, 100.0)


        # Calculate learning targets
        with torch.no_grad(), self.timing.timeit('calc_targets'):  # these computations are not the part of the computation graph
            ratios_cpu = ratio.cpu()
            vtrace_rho = torch.min(rho_hat, ratios_cpu)
            vtrace_c = torch.min(c_hat, ratios_cpu)

            if self.cfg.learner.psychlab_gamma >= 0.0:
                zero_tensors = torch.zeros_like(mb.task_idx)
                x1 = torch.ge(mb.task_idx, zero_tensors + 18)
                x2 = torch.le(mb.task_idx, zero_tensors + 21)
                gamma_mask = torch.logical_and(x1, x2).float().squeeze(1).cpu()

                gamma = (1. - gamma_mask) * self.cfg.learner.gamma + gamma_mask * self.cfg.learner.psychlab_gamma
            else:
                gamma = self.cfg.learner.gamma

            exclude_last = self.cfg.learner.exclude_last
            with self.timing.timeit('calc_vtrace'):
                targets, adv = calculate_vtrace(values, mb.rewards, mb.dones,
                                                vtrace_rho, vtrace_c, rollouts_in_batch, recurrence, gamma,
                                                exclude_last=exclude_last)

            if self.cfg.model.use_popart:
                vs = targets.clone().detach()
                mus = result.mus.squeeze(1).cpu()
                sigmas = result.sigmas.squeeze(1).cpu()
                targets = (targets - mus) / sigmas
                adv = ((adv + values.cpu()) - mus) / sigmas - normalized_values.cpu()

            # Apply off-policy correction
            adv = adv * vtrace_rho

            # Normalize advantage
            if self.cfg.learner.use_adv_normalization:
                adv = (adv - adv.mean()) / max(1e-3, adv.std())
            adv = adv.to(self.device)

        # Get losses
        with self.timing.timeit('losses'):
            policy_loss = self._policy_loss(ratio, log_prob_actions, adv.detach(), clip_ratio_low, clip_ratio_high,
                                            ppo=self.cfg.learner.use_ppo, exclude_last=exclude_last)
            exploration_loss = self.exploration_loss_func(action_distribution)


            actor_loss = policy_loss + exploration_loss
            epoch_actor_losses.append(actor_loss.item())

            targets = targets.to(self.device).detach()
            old_values = mb.values.detach()
            old_normalized_values = mb.normalized_values.squeeze().detach()
            if self.cfg.model.use_popart:
                value_loss = self._value_loss(normalized_values, targets, exclude_last=exclude_last)
            else:
                value_loss = self._value_loss(values, targets, exclude_last=exclude_last)
            critic_loss = value_loss

            loss = actor_loss + critic_loss

            if self.cfg.learner.use_decoder:
                assert self.actor_critic.reconstruction is not None, 'reconstruction not made yet!'
                reconstruction_target = self.actor_critic.reconstruction
                reconstruction_loss = self._reconstruction_loss(mb.obs, reconstruction_target,
                                                                self.cfg.env.obs_subtract_mean, self.cfg.env.obs_scale)
                loss = loss + reconstruction_loss

            if self.cfg.learner.use_aux_future_pred_loss:
                future_pred_loss = result.future_pred_loss
                loss = loss + future_pred_loss


        # calculate KL-divergence with the behaviour policy action distribution
        old_action_distribution = CategoricalActionDistribution(mb.action_logits,)
        kl_old = action_distribution.kl_divergence(old_action_distribution)
        kl_old_mean = kl_old.mean().item()
        recent_kls.append(kl_old_mean)

        # Update the weights
        # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
        for p in self.actor_critic.parameters():
            p.grad = None

        with self.timing.timeit('backward'):
            loss.backward()


        if self.distenv.world_size > 1:
            with self.timing.timeit('dist_all_reduce_gradient'):
                dist_reduce_gradient(self.actor_critic)
                if self.aux_loss_module is not None:
                    dist_reduce_gradient(self.aux_loss_module)


        grad_norm_before_clip = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.actor_critic.parameters()
            if p.grad is not None
        ) ** 0.5

        if self.cfg.optim.max_grad_norm > 0.0:
            with self.timing.timeit('clip'):
                torch.nn.utils.clip_grad_norm_(list(self.actor_critic.parameters()), self.cfg.optim.max_grad_norm)

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.actor_critic.parameters()
            if p.grad is not None
        ) ** 0.5

        curr_policy_version = self.train_step  # policy version before the weight update

        if self.optimizer_step_count < self.cfg.optim.warmup_optimizer:
            lr_original = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = 0.0

        with self.policy_lock:
            if self.distenv.world_size > 1:
                if self.distenv.master:
                    self.optimizer.step()
                dist_broadcast_model(self.actor_critic)
            else:
                self.optimizer.step()

            # For popart
            if self.cfg.model.use_popart:
                # TODO: funtionalize popart
                mu, sigma, oldmu, oldsigma = self.actor_critic.update_mu_sigma(
                    vs.to(self.device), mb.task_idx.long(), cfg=self.cfg)
                if self.distenv.world_size > 1:
                    dist_all_reduce_buffers(self.actor_critic)
                self.actor_critic.update_parameters(mu, sigma, oldmu, oldsigma)

        if self.optimizer_step_count < self.cfg.optim.warmup_optimizer:
            self.optimizer.param_groups[0]['lr'] = lr_original
        self.optimizer_step_count += 1

        num_sgd_steps += 1

        with torch.no_grad():
            self._after_optimizer_step()

            # collect and report summaries
            with_summaries = self._should_save_summaries() or force_summaries
            if with_summaries and not summary_this_epoch:
                stats_and_summaries = self._record_summaries(AttrDict(locals()))
                summary_this_epoch = True
                force_summaries = False

        # this will force policy update on the inference worker (policy worker)
        self.policy_versions[0] = self.train_step

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars):
        var = train_loop_vars

        self.last_summary_time = time.time()
        stats = AttrDict()

        stats.lr = self._curr_lr()

        stats.grad_norm = var.grad_norm
        stats.grad_norm_before_clip = var.grad_norm_before_clip
        stats.loss = var.loss
        stats.value = var.result.values.mean()
        stats.entropy = var.action_distribution.entropy().mean()
        stats.policy_loss = var.policy_loss
        stats.value_loss = var.value_loss
        stats.exploration_loss = var.exploration_loss
        if self.cfg.learner.use_decoder:
            stats.reconstruction_loss = var.reconstruction_loss
        if self.cfg.learner.use_aux_future_pred_loss:
            stats.future_pred_loss = var.future_pred_loss

        # stats.ratio = var.ratio
        stats.adv_min = var.adv.min()
        stats.adv_max = var.adv.max()
        stats.adv_std = var.adv.std()
        stats.max_abs_logprob = torch.abs(var.action_distribution.log_probs).max()

        if hasattr(var.action_distribution, 'summaries'):
            stats.update(var.action_distribution.summaries())

        ratio_mean = torch.abs(1.0 - var.ratio).mean().detach()
        ratio_min = var.ratio.min().detach()
        ratio_max = var.ratio.max().detach()

        value_delta = torch.abs(var.values - var.old_values)
        value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

        stats.kl_divergence = var.kl_old_mean
        stats.kl_divergence_max = var.kl_old.max()
        stats.value_delta = value_delta_avg
        stats.value_delta_max = value_delta_max
        stats.fraction_clipped = ((var.ratio < var.clip_ratio_low).float() + (var.ratio > var.clip_ratio_high).float()).mean()
        stats.ratio_mean = ratio_mean
        stats.ratio_min = ratio_min
        stats.ratio_max = ratio_max

        # this caused numerical issues on some versions of PyTorch with second moment reaching infinity
        adam_max_second_moment = 0.0
        for key, tensor_state in self.optimizer.state.items():
            adam_max_second_moment = max(tensor_state['exp_avg_sq'].max().item(), adam_max_second_moment)
        stats.adam_max_second_moment = adam_max_second_moment

        version_diff = (var.curr_policy_version - var.mb.policy_version)
        stats.version_diff_avg = version_diff.mean()
        stats.version_diff_min = version_diff.min()
        stats.version_diff_max = version_diff.max()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def init_model(self):
        self.actor_critic = create_agent(self.cfg, self.action_space, self.obs_space,
                            self.level_info['num_levels'], need_half=False)

        self.actor_critic.model_to_device(self.device)
        self.actor_critic.share_memory()

        if self.aux_loss_module is not None:
            self.aux_loss_module.to(device=self.device)

        self.mems_buffer = torch.zeros(self.shared_buffer.mems_dimensions).to(self.cfg.model.device)
        self.mems_buffer.share_memory_()
        self.mems_dones_buffer = torch.zeros(self.shared_buffer.mems_dones_dimensions, dtype=torch.bool).to(
            self.cfg.model.device)
        self.mems_actions_buffer = torch.zeros(self.shared_buffer.mems_dones_dimensions).short().to(
            self.cfg.model.device)
        self.mems_dones_buffer.share_memory_()
        self.mems_actions_buffer.share_memory_()
        self.max_mems_buffer_len = self.shared_buffer.max_mems_buffer_len
        self.mems_dimensions = self.shared_buffer.mems_dimensions
        self.mems_dones_dimensions = self.shared_buffer.mems_dones_dimensions
        self.mems_actions_dimensions = self.shared_buffer.mems_actions_dimensions

        rollouts_in_batch = self.cfg.optim.batch_size // self.cfg.optim.rollout
        self.mems = torch.zeros(
            [self.cfg.model.core.mem_len, rollouts_in_batch, self.mems_dimensions[-1]], device=self.device)
        self.mems_dones = torch.zeros(
            [self.cfg.model.core.mem_len, rollouts_in_batch, 1], dtype=torch.bool, device=self.device)
        self.mems_actions = torch.zeros(
            [self.cfg.model.core.mem_len, rollouts_in_batch, 1], device=self.device).short()

        if self.distenv.world_size > 1:
            dist_broadcast_model(self.actor_critic)

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict['train_step']
            self.env_steps = checkpoint_dict['env_steps']
        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if self.aux_loss_module is not None:
            self.aux_loss_module.load_state_dict(checkpoint_dict['aux_loss_module'])
        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found')
            return None
        else:
            latest_checkpoint = checkpoints[-1]
            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                try:
                    log.warning('Loading state from checkpoint %s...', latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f'Could not load from checkpoint, attempt {attempt}')

    def load_from_checkpoint(self):
        checkpoints = get_checkpoints(get_checkpoint_dir(self.cfg))
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug('Did not load from checkpoint, starting from scratch!')
        else:
            log.debug('Loading model from checkpoint')

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            self._load_state(checkpoint_dict, load_progress=True)

    def initialize(self):
        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info('Starting seed is not provided')
        else:
            log.info('Setting fixed seed %d', self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # this does not help with a single experiment
        # but seems to do better when we're running more than one experiment in parallel
        torch.set_num_threads(1)

        if self.cfg.model.device == 'cuda':
            torch.backends.cudnn.benchmark = True

            # we should already see only one CUDA device, because of env vars
            # assert torch.cuda.device_count() == 1
            self.device = torch.device('cuda', index=0)
        else:
            self.device = torch.device('cpu')

        self.distenv = dist_init(self.cfg)

        self.init_model()

        params = list(self.actor_critic.parameters())

        if self.aux_loss_module is not None:
            params += list(self.aux_loss_module.parameters())

        if self.cfg.optim.type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.cfg.optim.learning_rate,
                betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
                eps=self.cfg.optim.adam_eps
            )
        elif self.cfg.optim.type == 'adam':
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.cfg.optim.learning_rate,
                betas=(self.cfg.optim.adam_beta1, self.cfg.optim.adam_beta2),
                eps=self.cfg.optim.adam_eps
            )

        else:
            raise NotImplementedError

        if self.cfg.learner.resume_training:
            self.load_from_checkpoint()

        self._broadcast_model_weights()  # sync the very first version of the weights

        self.train_thread_initialized.set()

    def _process_training_data(self, data, wait_stats=None):
        self.is_training = True

        buffer, batch_size, samples, env_steps = data
        assert samples == batch_size

        self.env_steps += env_steps * self.cfg.dist.world_size

        stats = dict(learner_env_steps=self.env_steps)

        with self.timing.timeit('train'):
            train_stats = self._train(buffer)

        if train_stats is not None:
            stats['train'] = train_stats

        stats['times_learner_worker'] = {}
        for key, value in self.timing.items():
            stats['times_learner_worker'][key] = value

        self.is_training = False

        try:
            safe_put(self.report_queue, stats, queue_name='report')
        except Full:
            log.warning('Could not report training stats, the report queue is full!')

    def _extract_rollouts(self, data):
        rollouts = []
        traj_buffer_idx = data['traj_buffer_idx']
        for rollout_data in data['rollouts']:
            actor_idx = rollout_data['actor_idx']
            split_idx = rollout_data['split_idx']
            env_idx = rollout_data['env_idx']
            tensors = self.rollout_tensors.index((actor_idx, split_idx, env_idx, traj_buffer_idx))

            if self.cfg.model.core.core_type == 'trxl':
                actor_idx, split_idx, env_idx, rollout_step, actor_env_step, first_rollout = rollout_data['mem_idx']
                s_idx = (actor_env_step - self.cfg.model.core.mem_len - self.cfg.optim.rollout) % self.max_mems_buffer_len
                e_idx = (actor_env_step - self.cfg.optim.rollout) % self.max_mems_buffer_len

                tensors['actor_env_step'] = torch.tensor([actor_env_step])
                tensors['mems_indices'] = torch.tensor([[actor_idx, split_idx, env_idx, s_idx, e_idx]],
                                                       dtype=torch.long)

            rollout_data['t'] = tensors
            rollout_data['traj_buffer_idx'] = traj_buffer_idx

            rollouts.append(AttrDict(rollout_data))

        return rollouts

    def _accumulated_too_much_experience(self, rollouts):
        max_minibatches_to_accumulate = 2

        # allow the max batches to accumulate, plus the minibatches we're currently training on
        max_minibatches_on_learner = max_minibatches_to_accumulate + 1

        minibatches_currently_training = int(self.is_training)

        rollouts_per_minibatch = self.cfg.optim.batch_size / self.cfg.optim.rollout

        # count contribution from unprocessed rollouts
        minibatches_currently_accumulated = len(rollouts) / rollouts_per_minibatch

        # count minibatches ready for training
        minibatches_currently_accumulated += self.experience_buffer_queue.qsize()

        total_minibatches_on_learner = minibatches_currently_training + minibatches_currently_accumulated

        return total_minibatches_on_learner >= max_minibatches_on_learner

    def _run(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cfg.dist.local_rank}'
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{self.cfg.dist.local_rank}'

        self.deferred_initialization()

        log.info(f'LEARNER\tpid {os.getpid()}\tparent {os.getppid()}')

        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            psutil.Process().nice(0)
        except psutil.AccessDenied:
            log.error('Low niceness requires sudo!')

        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)

        rollouts = []

        while not self.terminate:
            while True:
                try:
                    tasks = self.learner_worker_queue.get_many(timeout=0.005)

                    for task_type, data in tasks:
                        if task_type == TaskType.TRAIN:
                            with self.timing.timeit('extract'):
                                rollouts.extend(self._extract_rollouts(data))
                        elif task_type == TaskType.INIT:
                            self._init()
                        elif task_type == TaskType.TERMINATE:
                            time.sleep(0.3)
                            log.info('GPU learner timing: %s')
                            self._terminate()
                            break
                except Empty:
                    break
                except:
                    self.report_queue.put(('terminate', 'learner'))

            if self._accumulated_too_much_experience(rollouts):
                # if we accumulated too much experience, signal the policy workers to stop experience collection
                if not self.stop_experience_collection:
                    self.stop_experience_collection_num_msgs += 1
                    # TODO: add a logger function for this
                    if self.stop_experience_collection_num_msgs >= 50:
                        log.info(
                            'Learner %d accumulated too much experience, stop experience collection! '
                            'Learner is likely a bottleneck in your experiment (%d times)',
                            self.cfg.dist.world_rank, self.stop_experience_collection_num_msgs,
                        )
                        self.stop_experience_collection_num_msgs = 0

                self.stop_experience_collection.fill_(True)
            elif self.stop_experience_collection:
                # otherwise, resume the experience collection if it was stopped
                self.stop_experience_collection.fill_(False)
                with self.resume_experience_collection_cv:
                    self.resume_experience_collection_num_msgs += 1
                    if self.resume_experience_collection_num_msgs >= 50:
                        log.debug('Learner %d is resuming experience collection!', self.cfg.dist.world_rank)
                        self.resume_experience_collection_num_msgs = 0
                    self.resume_experience_collection_cv.notify_all()

            with torch.no_grad():
                rollouts = self._process_rollouts(rollouts)

            while not self.experience_buffer_queue.empty():
                training_data = self.experience_buffer_queue.get()
                self._process_training_data(training_data)

        self.report_queue.put(('terminate', 'learner'))

    def init(self):
        self.learner_worker_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()

    def save_model(self, timeout=None):
        self.model_saved_event.clear()
        log.debug('Wait while learner %d saves the model...', self.cfg.dist.world_rank)
        if self.model_saved_event.wait(timeout=timeout):
            log.debug('Learner %d saved the model!', self.cfg.dist.world_rank)
        else:
            log.warning('Model saving request timed out!')
        self.model_saved_event.clear()

    def close(self):
        self.learner_worker_queue.put((TaskType.TERMINATE, None))
        self.shared_buffer._stop_experience_collection.fill_(False)

    def join(self):
        join_or_kill(self.process)


class TensorBatcher:
    def __init__(self, batch_pool):
        self.batch_pool = batch_pool

    def cat(self, dict_of_tensor_arrays, macro_batch_size, use_pinned_memory):

        tensor_batch = self.batch_pool.get()

        if tensor_batch is not None:
            old_batch_size = tensor_batch_size(tensor_batch)
            if old_batch_size != macro_batch_size:
                # this can happen due to PBT changing batch size during the experiment
                log.warning('Tensor macro-batch size changed from %d to %d!', old_batch_size, macro_batch_size)
                log.warning('Discarding the cached tensor batch!')
                del tensor_batch
                tensor_batch = None

        if tensor_batch is None:
            tensor_batch = copy_dict_structure(dict_of_tensor_arrays)
            log.info('Allocating new CPU tensor batch (could not get from the pool)')

            for d1, cache_d, key, tensor_arr, _ in iter_dicts_recursively(dict_of_tensor_arrays, tensor_batch):
                cache_d[key] = torch.from_numpy(np.concatenate(tensor_arr, axis=0))
                if use_pinned_memory:
                    cache_d[key] = cache_d[key].pin_memory()
        else:
            for d1, cache_d, key, tensor_arr, cache_t in iter_dicts_recursively(dict_of_tensor_arrays, tensor_batch):
                offset = 0
                for t in tensor_arr:
                    first_dim = t.shape[0]
                    cache_t[offset:offset + first_dim].copy_(torch.as_tensor(t))
                    offset += first_dim

        return tensor_batch


class ObjectPool:
    def __init__(self, pool_size=10):
        self.pool_size = pool_size
        self.pool = deque([], maxlen=self.pool_size)
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            if len(self.pool) <= 0:
                return None

            obj = self.pool.pop()
            return obj

    def put(self, obj):
        with self.lock:
            self.pool.append(obj)

    def clear(self):
        with self.lock:
            self.pool = deque([], maxlen=self.pool_size)


def tensor_batch_size(tensor_batch):
    for _, _, v in iterate_recursively(tensor_batch):
        return v.shape[0]
