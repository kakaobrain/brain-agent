import torch
from torch import nn

from brain_agent.core.models.model_utils import calc_num_elements, get_obs_shape, nonlinearity
from brain_agent.core.models.model_abc import EncoderBase
from brain_agent.utils.utils import log


class ConvEncoder(EncoderBase):
    class ConvEncoderImpl(nn.Module):
        """
        After we parse all the configuration and figure out the exact architecture of the model,
        we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
        fusion).
        """
        def __init__(self, activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape):
            super(ConvEncoder.ConvEncoderImpl, self).__init__()
            conv_layers = []
            for layer in conv_filters:
                if layer == 'maxpool_2x2':
                    conv_layers.append(nn.MaxPool2d((2, 2)))
                elif layer == 'avgpool_2x2':
                    conv_layers.append(nn.AvgPool2d((2, 2)))
                elif isinstance(layer, (list, tuple)):
                    if len(layer) == 4:
                        inp_ch, out_ch, filter_size, stride = layer
                        padding = 0
                    elif len(layer) == 5:
                        inp_ch, out_ch, filter_size, stride, padding = layer

                    conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride, padding=padding))
                    conv_layers.append(activation)
                else:
                    raise NotImplementedError(f'Layer {layer} not supported!')

            self.conv_head = nn.Sequential(*conv_layers)
            self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

            fc_layers = []
            for i in range(encoder_extra_fc_layers):
                size = self.conv_head_out_size if i == 0 else fc_layer_size
                fc_layers.extend([nn.Linear(size, fc_layer_size), activation])

            self.fc_layers = nn.Sequential(*fc_layers)

        def forward(self, obs):
            x = self.conv_head(obs)
            x = x.contiguous().view(-1, self.conv_head_out_size)
            x = self.fc_layers(x)
            return x

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        # inp_ch, out_ch, filter_size, stride
        if cfg.model.encoder.encoder_subtype == 'convnet_simple':
            conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.model.encoder.encoder_subtype == 'convnet_simple_chx2':
            conv_filters = [[input_ch, 64, 8, 4], [64, 128, 4, 2], [128, 256, 3, 2]]
        # doubled # layers except the first layer
        # extra layers are kener_size=3, padding=1
        # input_ch, output_ch, kernel_size, stride, padding (optional)
        elif cfg.model.encoder.encoder_subtype == 'convnet_simple_nlx2':
            conv_filters = [[input_ch, 32, 8, 4], [32, 32, 3, 1, 1], [32, 64, 4, 2], [64, 64, 3, 1, 1], [64, 128, 3, 2], [128, 128, 3, 1, 1]]
        elif cfg.model.encoder.encoder_subtype == 'convnet_impala':
            conv_filters = [[input_ch, 16, 8, 4], [16, 32, 4, 2]]
        elif cfg.model.encoder.encoder_subtype == 'minigrid_convnet_tiny':
            conv_filters = [[3, 16, 3, 1], [16, 32, 2, 1], [32, 64, 2, 1]]
        elif cfg.model.encoder.encoder_subtype == 'nethack_glyph':
            conv_filters = [[input_ch, 128, 3, 1], 'avgpool_2x2', [128, 64, 3, 1], 'avgpool_2x2', [64, 32, 3, 1]]
        else:
            raise NotImplementedError(f'Unknown encoder {cfg.model.encoder.encoder_subtype}')

        activation = nonlinearity(self.cfg)
        fc_layer_size = self.cfg.model.core.hidden_size
        encoder_extra_fc_layers = self.cfg.model.encoder.encoder_extra_fc_layers

        enc = self.ConvEncoderImpl(activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_shape.obs)
        log.debug('Encoder output size: %r', self.encoder_out_size)

    def forward(self, obs_dict):
        return self.enc(obs_dict['obs'])