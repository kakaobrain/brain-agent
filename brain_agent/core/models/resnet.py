import torch
from torch import nn
from brain_agent.core.models.model_abc import EncoderBase
from brain_agent.core.models.model_utils import get_obs_shape, calc_num_elements, nonlinearity
from brain_agent.utils.logger import log

class ResnetEncoder(EncoderBase):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        if cfg.model.encoder.encoder_subtype == 'resnet_impala':
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        elif cfg.model.encoder.encoder_subtype == 'resnet_impala_large':
            resnet_conf = [[32, 2], [64, 2], [64, 2]]
        else:
            raise NotImplementedError(f'Unknown resnet subtype {cfg.model.encoder.encoder_subtype}')

        curr_input_channels = input_ch
        layers = []
        if cfg.learner.use_decoder:
            layers_decoder = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            if cfg.model.encoder.encoder_pooling == 'stride':
                enc_stride = 2
                pool = nn.Identity
            else:
                enc_stride = 1
                pool = nn.MaxPool2d if cfg.model.encoder.encoder_pooling == 'max' else nn.AvgPool2d
            layers.extend([
                nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=enc_stride, padding=1),
                pool(kernel_size=3, stride=2, padding=1),  # padding SAME
            ])

            for j in range(res_blocks):
                layers.append(ResBlock(cfg, out_channels, out_channels))

            if cfg.learner.use_decoder:
                for j in range(res_blocks):
                    layers_decoder.append(ResBlock(cfg, curr_input_channels, curr_input_channels))
                layers_decoder.append(
                    nn.ConvTranspose2d(out_channels, curr_input_channels, kernel_size=3, stride=2,
                                       padding=1, output_padding=1)
                )
            curr_input_channels = out_channels

        layers.append(nonlinearity(cfg))

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)
        log.debug('Convolutional layer output size: %r', self.conv_head_out_size)
        self.init_fc_blocks(self.conv_head_out_size)

        if cfg.learner.use_decoder:
            layers_decoder.reverse()
            self.deconv_head = nn.Sequential(*layers_decoder)

        self.apply(self.initialize)

    def initialize(self, layer):
        gain = 1.0
        if hasattr(layer, 'bias') and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if self.cfg.model.encoder.encoder_init == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
        elif self.cfg.model.encoder.encoder_init == 'xavier_uniform':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                nn.init.xavier_uniform_(layer.weight_ih, gain=gain)
                nn.init.xavier_uniform_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
        elif self.cfg.model.encoder.encoder_init == 'torch_default':
            pass
        else:
            raise NotImplementedError

    def forward(self, obs_dict, decode=False):
        x = self.conv_head(obs_dict['obs'])

        if decode:
            self.reconstruction = torch.tanh(self.deconv_head(x))

        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.forward_fc_blocks(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out
