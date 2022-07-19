import copy
import torch
from torch import nn
from brain_agent.utils.utils import AttrDict

EPS = 1e-8

def to_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.item()
    else:
        return value

def get_hidden_size(cfg, action_space):
    if cfg.model.core.core_type == 'trxl':
        size = cfg.model.core.hidden_size * (cfg.model.core.n_layer + 1)
        size += 64 * (cfg.model.core.n_layer + 1)
        if cfg.model.extended_input:
            size += (action_space.n + 1) * (cfg.model.core.n_layer + 1)
    elif cfg.model.core.core_type == 'rnn':
        size = cfg.model.core.hidden_size * cfg.model.core.n_rnn_layer * 2
    else:
        raise NotImplementedError
    return size


def nonlinearity(cfg):
    if cfg.model.encoder.nonlinearity == 'elu':
        return nn.ELU(inplace=cfg.model.encoder.nonlinear_inplace)
    elif cfg.model.encoder.nonlinearity == 'relu':
        return nn.ReLU(inplace=cfg.model.encoder.nonlinear_inplace)
    elif cfg.model.encoder.nonlinearity == 'tanh':
        return nn.Tanh()
    else:
        raise Exception('Unknown nonlinearity')


def get_obs_shape(obs_space):
    obs_shape = AttrDict()
    if hasattr(obs_space, 'spaces'):
        for key, space in obs_space.spaces.items():
            obs_shape[key] = space.shape
    else:
        obs_shape.obs = obs_space.shape

    return obs_shape


def calc_num_elements(module, module_input_shape):
    shape_with_batch_dim = (1,) + module_input_shape
    some_input = torch.rand(shape_with_batch_dim)
    num_elements = module(some_input).numel()
    return num_elements

def normalize_obs_return(obs_dict, cfg, half=False):
    with torch.no_grad():
        mean = cfg.env.obs_subtract_mean
        scale = cfg.env.obs_scale

        normalized_obs_dict = copy.deepcopy(obs_dict)

        if normalized_obs_dict['obs'].dtype != torch.float:
            normalized_obs_dict['obs'] = normalized_obs_dict['obs'].float()

        if abs(mean) > EPS:
            normalized_obs_dict['obs'].sub_(mean)

        if abs(scale - 1.0) > EPS:
            normalized_obs_dict['obs'].mul_(1.0 / scale)

        if half:
            normalized_obs_dict['obs'] = normalized_obs_dict['obs'].half()

    return normalized_obs_dict

