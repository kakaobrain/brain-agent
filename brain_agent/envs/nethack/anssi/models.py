# https://github.com/Miffyli/nle-sample-factory-baseline/blob/main/models.py

import torch
from torch import nn
import numpy as np

import os
from brain_agent.core.models.model_utils import (
    get_obs_shape, nonlinearity
)
from brain_agent.envs.nethack.nethack_model import create_standard_encoder
from brain_agent.core.models.model_abc import EncoderBase

from brain_agent.utils.utils import log
from brain_agent.envs.nethack.baselines.models import MessageEncoder

from ..wrappers.message_vocab import _torchtext_vocab


class MessageEmbedding(nn.Module):
    """
    Thanks to Martin
    """
    def __init__(self, embedding_dim, hidden_dim, out_dim, mode='mean', use_pretrained=False, vocab_size=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mode = mode
        assert mode in ('mean', 'sum', 'max')

        if use_pretrained:
            assets_path = os.path.join(os.path.dirname(__file__), 'assets')
            embed_mat = np.load(os.path.join(assets_path, 'glove.840B.300d.npy'))
            embedding = nn.EmbeddingBag.from_pretrained(torch.from_numpy(embed_mat).float(),
                                                        freeze=False, mode=self.mode)
            self._embedding = embedding
        else:
            assert vocab_size is not None
            embedding = nn.EmbeddingBag(vocab_size, self.embedding_dim, mode=self.mode)
            self._embedding = embedding

        self._linear1 = nn.Linear(self.embedding_dim, self.hidden_dim)
        self._linear2 = nn.Linear(self.hidden_dim, self.out_dim)
        self._nonlinear = nn.ELU()

    def forward(self, text):
        out = self._embedding(text.long())
        out = self._nonlinear(self._linear1(out))
        out = self._nonlinear(self._linear2(out))
        return out


class NLEMainEncoder(EncoderBase):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        # Use standard CNN for the image observation in "obs"
        # See all arguments with "-h" to change this head to e.g. ResNet
        self.basic_encoder = create_standard_encoder(cfg, obs_space)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)

        self.vector_obs_head = None
        self.message_head = None
        if 'vector_obs' in obs_shape:
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_vector_obs = nn.LayerNorm(obs_shape.vector_obs[0])
            self.vector_obs_head = nn.Sequential(
                nn.Linear(obs_shape.vector_obs[0], self.cfg.model.encoder.vobs_hidden_dim),
                nonlinearity(cfg),
                nn.Linear(self.cfg.model.encoder.vobs_hidden_dim, self.cfg.model.encoder.vobs_hidden_dim),
                nonlinearity(cfg),
            )
            out_size = self.cfg.model.encoder.vobs_hidden_dim
            # Add vector_obs to NN output for more direct access by LSTM core
            self.encoder_out_size += out_size
            if self.cfg.model.encoder.cat_vobs:
                self.encoder_out_size += obs_shape.vector_obs[0]

        self.use_spell_feature_fc = False
        if 'spell_feature' in obs_shape:
            self.use_spell_feature_fc = True
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_spell_feature = nn.LayerNorm(obs_shape.spell_feature[1])
            self.fc1_spell_feature = nn.Linear(obs_shape.spell_feature[1], 64)
            self.fc2_spell_feature = nn.Linear(64, 64)
            self.encoder_out_size += 64

        self.use_item_feature_fc = False
        if 'item_feature' in obs_shape:
            self.use_item_feature_fc = True
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_item_feature = nn.LayerNorm(obs_shape.item_feature[1])
            self.fc1_item_feature = nn.Linear(obs_shape.item_feature[1], 128)
            self.fc2_item_feature = nn.Linear(128, 128)
            self.encoder_out_size += 128

        if 'message' in obs_shape or 'message_embedding' in obs_shape:
            # _Very_ poor for text understanding,
            # but it is simple and probably enough to overfit to specific sentences.
            if cfg.model.encoder.message_encoder == 'mlp':
                self.message_head = nn.Sequential(
                    nn.Linear(obs_shape.message[0], 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                )
                out_size = 128
            elif cfg.model.encoder.message_encoder == 'embedding':
                self.message_head = MessageEmbedding(80, 128, 128, use_pretrained=False, vocab_size=len(_torchtext_vocab))
                out_size = self.message_head.out_dim
            elif cfg.model.encoder.message_encoder == 'embedding_pretrained':
                self.message_head = MessageEmbedding(300, 128, 128, use_pretrained=True)
                out_size = self.message_head.out_dim
            elif cfg.model.encoder.message_encoder == 'baseline':
                self.message_head = MessageEncoder(self.cfg.model.encodermsg_hidden_dim, self.cfg.model.encoder.msg_embedding_dim)
                out_size = self.cfg.model.encoder.msg_hidden_dim
            else:
                raise NotImplementedError
            self.encoder_out_size += out_size

        # use hidden_size, encoder_extra_fc_layers
        # self.init_fc_blocks(self.encoder_out_size)

        if cfg.model.encoder.add_fc_layers_after_concat:
            self.fc_after_concat = nn.Sequential(
                nn.Linear(self.encoder_out_size, cfg.model.core.hidden_size),
                nonlinearity(cfg),
                nn.Linear(cfg.model.core.hidden_size, cfg.model.core.hidden_size),
                nonlinearity(cfg),
            )
            self.encoder_out_size = cfg.model.core.hidden_size
        else:
            self.fc_after_concat = None

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        # This one handles the "obs" key which contains the main image
        x = self.basic_encoder(obs_dict)

        cats = [x]
        if self.vector_obs_head is not None:
            vector_obs = obs_dict['vector_obs']
            if self.cfg.model.encoder.use_layer_norm:
                vector_obs = self.ln_vector_obs(vector_obs)
            if self.cfg.model.encoder.cat_vobs:
                vector_obs = self.vector_obs_head(vector_obs)
            else:
                vector_obs = self.vector_obs_head[1](self.vector_obs_head[0](vector_obs))
                vector_obs = self.vector_obs_head[3](self.vector_obs_head[2](vector_obs)) + vector_obs # residual
            cats.append(vector_obs)
            if self.cfg.model.encoder.cat_vobs:
                cats.append(obs_dict['vector_obs'])

        if self.message_head is not None:
            if isinstance(self.message_head, MessageEmbedding):
                message = self.message_head(obs_dict['message_embedding'])
            elif isinstance(self.message_head, nn.Sequential):
                message = self.message_head(obs_dict['message'] / 255)
            elif isinstance(self.message_head, MessageEncoder):
                message = self.message_head(obs_dict['message'].unsqueeze(1))
            else:
                raise NotImplementedError
            cats.append(message)

        if self.use_spell_feature_fc:
            h_spell = obs_dict['spell_feature']
            if self.cfg.model.encoder.use_layer_norm:
                h_spell = self.ln_spell_feature(h_spell)
            h_spell = nonlinearity(self.cfg)(self.fc1_spell_feature(h_spell)) # (**, n_spell, d_h_spell)
            h_spell = torch.mean(h_spell, dim=-2) # (**, d_h_spell)
            h_spell = nonlinearity(self.cfg)(self.fc2_spell_feature(h_spell)) # (**, d_h_spell)
            cats.append(h_spell)

        if self.use_item_feature_fc:
            h_item = obs_dict['item_feature']
            if self.cfg.model.encoder.use_layer_norm:
                h_item = self.ln_item_feature(h_item)
            h_item = nonlinearity(self.cfg)(self.fc1_item_feature(h_item)) # (**, n_item, d_h_item)
            h_item = torch.mean(h_item, dim=-2) # (**, d_h_item)
            h_item = nonlinearity(self.cfg)(self.fc2_item_feature(h_item))
            cats.append(h_item)

        if len(cats) > 1:
            x = torch.cat(cats, dim=1)

        # x = self.forward_fc_blocks(x)

        if self.fc_after_concat:
            x = self.fc_after_concat(x)

        return x


# register_custom_encoder('nle_obs_vector_encoder', NLEMainEncoder)
