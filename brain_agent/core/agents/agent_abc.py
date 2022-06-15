import torch
from torch import nn
from brain_agent.utils.utils import AttrDict
from brain_agent.core.models.action_distributions import DiscreteActionsParameterization
from brain_agent.core.models.action_distributions import CategoricalActionDistribution

class ActorCriticBase(nn.Module):
    def __init__(self, action_space, cfg):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space

    def get_action_parameterization(self, core_output_size):
        action_parameterization = DiscreteActionsParameterization(self.cfg, core_output_size, self.action_space)
        return action_parameterization

    def get_extra_action_parameterization(self, core_output_size, num_actions):
        action_space = AttrDict(n=num_actions)
        action_parameterization = DiscreteActionsParameterization(self.cfg, core_output_size, action_space)
        return action_parameterization

    def model_to_device(self, device):
        self.to(device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        return self.encoder.device_and_type_for_input_tensor(input_tensor_name)

    def get_log_prob_action(self, result, mb):
        return result.action_distribution.log_prob(mb.actions)

    def get_exploration_loss(self, exploration_loss_func, result, mb, exclude_last):
        return exploration_loss_func(result.action_distribution, exclude_last=exclude_last)

    def get_kl_divergence(self, result, mb):
        old_action_distribution = CategoricalActionDistribution(mb.action_logits)
        kl_old = result.action_distribution.kl_divergence(old_action_distribution)
        kl_old_mean = kl_old.mean().item()
        return kl_old, kl_old_mean

    def get_tensors_to_squeeze(self):
        tensors_to_squeeze = [
            'actions', 'log_prob_actions', 'policy_version', 'values',
            'rewards', 'dones', 'rewards_cpu', 'dones_cpu',
        ]
        return tensors_to_squeeze

    def update_stats(self, var, stats):
        stats.entropy = var.result.action_distribution.entropy().mean()
        if hasattr(var.result.action_distribution, 'summaries'):
            stats.update(var.result.action_distribution.summaries())

        stats.max_abs_logprob = torch.abs(var.result.action_distribution.log_probs).max()
        stats.kl_divergence = var.kl_old_mean
        stats.kl_divergence_max = var.kl_old.max()
