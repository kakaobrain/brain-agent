import torch
from torch import nn
from brain_agent.core.models.model_utils import nonlinearity

class ActionsParameterizationBase(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space

class EncoderBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.fc_after_enc = None
        self.encoder_out_size = -1  # to be initialized in the constuctor of derived class

    def get_encoder_out_size(self):
        return self.encoder_out_size

    def init_fc_blocks(self, input_size):
        layers = []
        fc_layer_size = self.cfg.model.core.hidden_size

        for i in range(self.cfg.model.encoder.encoder_extra_fc_layers):
            size = input_size if i == 0 else fc_layer_size

            layers.extend([
                nn.Linear(size, fc_layer_size),
                nonlinearity(self.cfg),
            ])

        if len(layers) > 0:
            self.fc_after_enc = nn.Sequential(*layers)
            self.encoder_out_size = fc_layer_size
        else:
            self.encoder_out_size = input_size

    def model_to_device(self, device):
        self.to(device)

    def device_and_type_for_input_tensor(self, _):
        return self.model_device(), torch.float32

    def model_device(self):
        return next(self.parameters()).device

    def forward_fc_blocks(self, x):
        if self.fc_after_enc is not None:
            x = self.fc_after_enc(x)

        return x