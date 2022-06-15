import numpy as np
import torch
from brain_agent.utils.utils import log

# register custom encoders
ENCODER_REGISTRY = dict()


def create_standard_encoder(cfg, obs_space):
    if cfg.model.encoder.encoder_type == 'conv':
        from brain_agent.core.models.conv_encoder import ConvEncoder
        encoder = ConvEncoder(cfg, obs_space)
    elif cfg.model.encoder.encoder_type == 'resnet':
        from brain_agent.core.models.resnet import ResnetEncoder
        encoder = ResnetEncoder(cfg, obs_space)
    else:
        raise Exception('Encoder type not supported')

    return encoder


def register_custom_encoder(custom_encoder_name, encoder_cls):
    #assert issubclass(encoder_cls, EncoderBase), 'Custom encoders must be derived from EncoderBase'
    assert custom_encoder_name not in ENCODER_REGISTRY

    log.debug('Adding model class %r to registry (with name %s)', encoder_cls, custom_encoder_name)
    ENCODER_REGISTRY[custom_encoder_name] = encoder_cls


def create_nethack_encoder(cfg, obs_space):
    if cfg.model.encoder.encoder_custom:
        encoder_cls = ENCODER_REGISTRY[cfg.model.encoder.encoder_custom]
        encoder = encoder_cls(cfg, obs_space)
    else:
        encoder = create_standard_encoder(cfg, obs_space)

    return encoder


def get_action_space_mask(action_space, reduced_action_space):
    mask = np.array([int(a in reduced_action_space) for a in action_space])
    return torch.Tensor(mask)


def nethack_register_models():
    from .baselines.models import NethackEncoder
    register_custom_encoder('nethack_baseline', NethackEncoder)

    from .anssi.models import NLEMainEncoder
    register_custom_encoder('nle_obs_vector_encoder', NLEMainEncoder)

    from .replay.models import NethackReplayEncoder
    register_custom_encoder('nethack_replay', NethackReplayEncoder)

    from .kakaobrain.models import TrXLIEncoder
    register_custom_encoder('trxli_encoder', TrXLIEncoder)

    from .kakaobrain.models import AvgPoolEncoder
    register_custom_encoder('avgpool_encoder', AvgPoolEncoder)