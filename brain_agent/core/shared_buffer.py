import math
import torch
import numpy as np
import gym

from brain_agent.utils.logger import log
from brain_agent.core.core_utils import iter_dicts_recursively, copy_dict_structure, iterate_recursively
from brain_agent.core.models.model_utils import get_hidden_size


def ensure_memory_shared(*tensors):
    for tensor_dict in tensors:
        for _, _, t in iterate_recursively(tensor_dict):
            assert t.is_shared()

def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype

def to_numpy(t, num_dimensions):
    arr_shape = t.shape[:num_dimensions]
    arr = np.ndarray(arr_shape, dtype=object)
    to_numpy_func(t, arr)
    return arr


def to_numpy_func(t, arr):
    if len(arr.shape) == 1:
        for i in range(t.shape[0]):
            arr[i] = t[i]
    else:
        for i in range(t.shape[0]):
            to_numpy_func(t[i], arr[i])


class SharedBuffer:
    def __init__(self, cfg, obs_space, action_space):

        self.cfg = cfg
        assert not cfg.actor.num_envs_per_worker % cfg.actor.num_splits, \
            f'actor.num_envs_per_worker ({cfg.actor.num_envs_per_worker}) ' \
            f'is not divided by actor.num_splits ({cfg.actor.num_splits})'

        self.obs_space = obs_space
        self.action_space = action_space

        self.envs_per_split = cfg.actor.num_envs_per_worker // cfg.actor.num_splits
        self.num_traj_buffers = self.calc_num_trajectory_buffers()

        core_hidden_size = get_hidden_size(self.cfg, action_space)

        log.debug('Allocating shared memory for trajectories')
        self.tensors = TensorDict()

        obs_dict = TensorDict()
        self.tensors['obs'] = obs_dict
        if isinstance(obs_space, gym.spaces.Dict):
            for name, space in obs_space.spaces.items():
                obs_dict[name] = self.init_tensor(space.dtype, space.shape)
        else:
            raise Exception('Only Dict observations spaces are supported')
        self.tensors['prev_actions'] = self.init_tensor(torch.int64, [1])
        self.tensors['prev_rewards'] = self.init_tensor(torch.float32, [1])

        self.tensors['rewards'] = self.init_tensor(torch.float32, [1])
        self.tensors['dones'] = self.init_tensor(torch.bool, [1])
        self.tensors['raw_rewards'] = self.init_tensor(torch.float32, [1])

        # policy outputs
        if self.cfg.model.core.core_type == 'trxl':
            policy_outputs = [
                ('actions', 1),
                ('action_logits', action_space.n),
                ('log_prob_actions', 1),
                ('values', 1),
                ('normalized_values', 1),
                ('policy_version', 1),
            ]
        elif self.cfg.model.core.core_type == 'rnn':
            policy_outputs = [
                ('actions', 1),
                ('action_logits', action_space.n),
                ('log_prob_actions', 1),
                ('values', 1),
                ('normalized_values', 1),
                ('policy_version', 1),
                ('rnn_states', core_hidden_size)
            ]

        policy_outputs = [PolicyOutput(*po) for po in policy_outputs]
        policy_outputs = sorted(policy_outputs, key=lambda policy_output: policy_output.name)

        for po in policy_outputs:
            self.tensors[po.name] = self.init_tensor(torch.float32, [po.size])

        ensure_memory_shared(self.tensors)

        self.tensors_individual_transitions = self.tensor_dict_to_numpy(len(self.tensor_dimensions()))

        self.tensor_trajectories = self.tensor_dict_to_numpy(len(self.tensor_dimensions()) - 1)

        traj_buffer_available_shape = [
            self.cfg.actor.num_workers,
            self.cfg.actor.num_splits,
            self.envs_per_split,
            self.num_traj_buffers,
        ]
        self.is_traj_tensor_available = torch.ones(traj_buffer_available_shape, dtype=torch.uint8)
        self.is_traj_tensor_available.share_memory_()
        self.is_traj_tensor_available = to_numpy(self.is_traj_tensor_available, 2)

        policy_outputs_combined_size = sum(po.size for po in policy_outputs)
        policy_outputs_shape = [
            self.cfg.actor.num_workers,
            self.cfg.actor.num_splits,
            self.envs_per_split,
            policy_outputs_combined_size,
        ]

        self.policy_outputs = policy_outputs
        self.policy_output_tensors = torch.zeros(policy_outputs_shape, dtype=torch.float32)
        self.policy_output_tensors.share_memory_()
        self.policy_output_tensors = to_numpy(self.policy_output_tensors, 3)

        self.policy_versions = torch.zeros([1], dtype=torch.int32)
        self.policy_versions.share_memory_()

        self.stop_experience_collection = torch.ones([1], dtype=torch.bool)
        self.stop_experience_collection.share_memory_()

        self.task_ids = torch.zeros([self.cfg.actor.num_workers, self.cfg.actor.num_splits, self.envs_per_split],
                                    dtype=torch.uint8)
        self.task_ids.share_memory_()

        self.max_mems_buffer_len = self.cfg.model.core.mem_len + self.cfg.optim.rollout * (self.num_traj_buffers + 1)
        self.mems_dimensions = [self.cfg.actor.num_workers, self.cfg.actor.num_splits, self.envs_per_split, self.max_mems_buffer_len]
        self.mems_dimensions.append(core_hidden_size)
        self.mems_dones_dimensions = [self.cfg.actor.num_workers, self.cfg.actor.num_splits,
                                      self.envs_per_split, self.max_mems_buffer_len]
        self.mems_dones_dimensions.append(1)
        self.mems_actions_dimensions = [self.cfg.actor.num_workers, self.cfg.actor.num_splits,
                                      self.envs_per_split, self.max_mems_buffer_len]
        self.mems_actions_dimensions.append(1)

    def calc_num_trajectory_buffers(self):
        num_traj_buffers = self.cfg.optim.batch_size / (
                    self.cfg.actor.num_workers * self.cfg.actor.num_envs_per_worker * self.cfg.optim.rollout)

        num_traj_buffers *= 3

        num_traj_buffers = math.ceil(max(num_traj_buffers, self.cfg.shared_buffer.min_traj_buffers_per_worker))
        log.info('Using %d sets of trajectory buffers', num_traj_buffers)
        return num_traj_buffers

    def tensor_dimensions(self):
        dimensions = [
            self.cfg.actor.num_workers,
            self.cfg.actor.num_splits,
            self.envs_per_split,
            self.num_traj_buffers,
            self.cfg.optim.rollout,
        ]
        return dimensions

    def init_tensor(self, tensor_type, tensor_shape):
        if not isinstance(tensor_type, torch.dtype):
            tensor_type = to_torch_dtype(tensor_type)

        dimensions = self.tensor_dimensions()
        final_shape = dimensions + list(tensor_shape)
        t = torch.zeros(final_shape, dtype=tensor_type)
        t.share_memory_()
        return t

    def tensor_dict_to_numpy(self, num_dimensions):
        numpy_dict = copy_dict_structure(self.tensors)
        for d1, d2, key, curr_t, value2 in iter_dicts_recursively(self.tensors, numpy_dict):
            assert isinstance(curr_t, torch.Tensor)
            assert value2 is None
            d2[key] = to_numpy(curr_t, num_dimensions)
            assert isinstance(d2[key], np.ndarray)
        return numpy_dict


class TensorDict(dict):
    def index(self, indices):
        return self.index_func(self, indices)

    def index_func(self, x, indices):
        if isinstance(x, (dict, TensorDict)):
            res = TensorDict()
            for key, value in x.items():
                res[key] = self.index_func(value, indices)
            return res
        else:
            t = x[indices]
            return t

    def set_data(self, index, new_data):
        self.set_data_func(self, index, new_data)

    def set_data_func(self, x, index, new_data):
        if isinstance(new_data, (dict, TensorDict)):
            for new_data_key, new_data_value in new_data.items():
                self.set_data_func(x[new_data_key], index, new_data_value)
        elif isinstance(new_data, torch.Tensor):
            x[index].copy_(new_data)
        elif isinstance(new_data, np.ndarray):
            t = torch.from_numpy(new_data)
            x[index].copy_(t)
        else:
            raise Exception(f'Type {type(new_data)} not supported in set_data_func')


class PolicyOutput:
    def __init__(self, name, size):
        self.name = name
        self.size = size
