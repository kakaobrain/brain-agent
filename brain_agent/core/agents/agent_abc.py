from torch import nn
from brain_agent.core.models.action_distributions import DiscreteActionsParameterization

class ActorCriticBase(nn.Module):
    def __init__(self, action_space, cfg):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space

    def get_action_parameterization(self, core_output_size):
        action_parameterization = DiscreteActionsParameterization(self.cfg, core_output_size, self.action_space)
        return action_parameterization

    def model_to_device(self, device):
        self.to(device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        return self.encoder.device_and_type_for_input_tensor(input_tensor_name)
