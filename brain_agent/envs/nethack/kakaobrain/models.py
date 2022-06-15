# https://github.com/Miffyli/nle-sample-factory-baseline/blob/main/models.py

import torch
from torch import nn
import torch.nn.functional as F
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

class TrXLI(nn.Module):
    def __init__(self, cfg, dim_k, dim_v):
        super().__init__()
        self.cfg = cfg
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dim_q = cfg.model.core.hidden_size
        self.n_head = cfg.model.encoder.n_head_trxli

        assert self.dim_q % self.n_head == 0, f"dim_q ({self.dim_q}) sholud be dividable by n_head ({self.n_head})!"

        self.ln_q = nn.LayerNorm(self.dim_q)
        self.ln_k = nn.LayerNorm(self.dim_k)
        self.ln_v = nn.LayerNorm(self.dim_v)

        self.W_qs = nn.ModuleList([nn.Linear(self.dim_q, self.dim_q // self.n_head) for _ in range(self.n_head)])
        self.W_ks = nn.ModuleList([nn.Linear(self.dim_k, self.dim_q // self.n_head) for _ in range(self.n_head)])
        self.W_vs = nn.ModuleList([nn.Linear(self.dim_v, self.dim_q // self.n_head) for _ in range(self.n_head)])

        self.fc_mhdpa = nn.Linear(self.dim_q, self.dim_q)

        self.ln_fc = nn.LayerNorm(self.dim_q)
        self.fc = nn.Linear(self.dim_q, self.dim_q)

    def forward(self, q, k, v, mask=None):
        q = q.unsqueeze(-2)
        q, k, v = self.ln_q(q), self.ln_k(k), self.ln_v(v)

        h_heads = list()
        for idx_head in range(self.n_head):
            h_q = self.W_qs[idx_head](q) # (**, 1, dim / n_head)
            h_k = self.W_ks[idx_head](k) # (**, n_entity, dim / n_head)
            h_v = self.W_vs[idx_head](v) # (**, n_entity, dim / n_head)

            logit_attention = torch.matmul(h_q, h_k.transpose(-2, -1)) / (self.dim_q / self.n_head) ** 0.5 # (**, 1, n_entity)
            if mask is not None:
                logit_attention = mask * logit_attention - (1 - mask) ** 1e10
            attention = torch.softmax(logit_attention, dim=-1)

            h_head = torch.matmul(attention, h_v) # (**, 1, dim / n_head)
            h_heads.append(h_head)

        h_head = q + self.fc_mhdpa(torch.cat(h_heads, dim=-1)) # (**, dim)
        h_fc = self.ln_fc(h_head)
        h_fc = q + nonlinearity(self.cfg)(self.fc(h_fc))
        h_fc = h_fc.squeeze(-2)

        return h_fc

class MessageEmbedding(nn.Module):
    """
    Thanks to Martin
    """
    def __init__(self, cfg, embedding_dim, hidden_dim, out_dim, mode='mean', use_pretrained=False, vocab_size=None):
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


class TrXLIEncoder(EncoderBase):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.cfg = cfg
        self.obs_space = obs_space

        # Use standard CNN for the image observation in "obs"
        # See all arguments with "-h" to change this head to e.g. ResNet
        self.basic_encoder = create_standard_encoder(cfg, obs_space)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)

        self.fc_vector_obs = None
        if 'vector_obs' in obs_shape:
            self.fc_vector_obs = nn.Linear(obs_shape.vector_obs[0], self.cfg.model.encoder.vobs_hidden_dim)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_vector_obs = nn.LayerNorm(self.cfg.model.encoder.vobs_hidden_dim)
            out_size = self.cfg.model.encoder.vobs_hidden_dim
            # Add vector_obs to NN output for more direct access by LSTM core
            self.encoder_out_size += out_size

        self.fc_action_class = None
        if 'action_class' in obs_shape:
            self.fc_action_class = nn.Linear(5, 32)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_action_class = nn.LayerNorm(32)
            self.encoder_out_size += 32

        self.fc_last_atype = None
        if 'last_atype' in obs_shape:
            self.fc_last_atype = nn.Linear(113, 64)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_last_atype = nn.LayerNorm(64)
            self.encoder_out_size += 64

        self.fc_avail_atype = None
        if 'avail_atype' in obs_shape and self.cfg.model.encoder.embed_avail_atype:
            self.fc_avail_atype = nn.Linear(113, 64)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_avail_atype = nn.LayerNorm(64)
            self.encoder_out_size += 64

        self.message_head = None
        if 'message' in obs_shape or 'message_embedding' in obs_shape:
            # _Very_ poor for text understanding,
            # but it is simple and probably enough to overfit to specific sentences.
            if self.cfg.model.encoder.message_encoder == 'mlp':
                self.message_head = nn.Sequential(
                    nn.Linear(obs_shape.message[0], 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                )
                out_size = 128
            elif self.cfg.model.encoder.message_encoder == 'embedding':
                self.message_head = MessageEmbedding(cfg, 80, 128, 128, use_pretrained=False, vocab_size=len(_torchtext_vocab))
                out_size = self.message_head.out_dim
            elif self.cfg.model.encoder.message_encoder == 'embedding_pretrained':
                self.message_head = MessageEmbedding(cfg, 300, 128, 128, use_pretrained=True)
                out_size = self.message_head.out_dim
            elif self.cfg.model.encoder.message_encoder == 'baseline':
                self.message_head = MessageEncoder(self.cfg.model.encoder.msg_hidden_dim, self.cfg.model.encoder.msg_embedding_dim)
                out_size = self.cfg.model.encoder.msg_hidden_dim
            else:
                raise NotImplementedError

            if self.cfg.model.encoder.use_layer_norm:
                self.ln_message = nn.LayerNorm(out_size)
            self.encoder_out_size += out_size

        # use hidden_size, encoder_extra_fc_layers
        # self.init_fc_blocks(self.encoder_out_size)

        if self.cfg.model.encoder.add_fc_layers_after_concat:
            self.fc_after_concat = nn.Sequential(
                nn.Linear(self.encoder_out_size, cfg.model.core.hidden_size),
                nonlinearity(cfg),
            )
            self.encoder_out_size = self.cfg.model.core.hidden_size
        else:
            self.fc_after_concat = None

        self.trxli_spell = None
        if 'spell_feature' in obs_shape:
            self.trxli_spell = TrXLI(self.cfg, dim_k=obs_shape.spell_feature[1], dim_v=obs_shape.spell_feature[1])

        self.trxli_item = None
        if 'item_feature' in obs_shape:
            self.trxli_item = TrXLI(self.cfg, dim_k=obs_shape.item_feature[1], dim_v=obs_shape.item_feature[1])

        self.trxli_pick_item = None
        if 'pick_item_feature' in obs_shape and self.cfg.env.encode_pick_item_feature:
            self.trxli_pick_item = TrXLI(self.cfg, dim_k=obs_shape.pick_item_feature[1], dim_v=obs_shape.pick_item_feature[1])

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        # This one handles the "obs" key which contains the main image
        x = self.basic_encoder(obs_dict)

        cats = [x]
        if self.fc_vector_obs is not None:
            vector_obs = nonlinearity(self.cfg)(self.fc_vector_obs(obs_dict['vector_obs'].float()))
            if self.cfg.model.encoder.use_layer_norm:
                vector_obs = self.ln_vector_obs(vector_obs)
            cats.append(vector_obs)

        if self.message_head is not None:
            if isinstance(self.message_head, MessageEmbedding):
                message = self.message_head(obs_dict['message_embedding'])
            elif isinstance(self.message_head, nn.Sequential):
                message = self.message_head(obs_dict['message'].float() / 255)
            elif isinstance(self.message_head, MessageEncoder):
                message = self.message_head(obs_dict['message'].unsqueeze(1))
            else:
                raise NotImplementedError
            if self.cfg.model.encoder.use_layer_norm:
                message = self.ln_message(message)
            cats.append(message)

        if self.fc_action_class is not None:
            onehot_action_class = F.one_hot(obs_dict['action_class'].long(), num_classes=5).squeeze(-2)
            embed_action_class = nonlinearity(self.cfg)(self.fc_action_class(onehot_action_class.float()))
            if self.cfg.model.encoder.use_layer_norm:
                embed_action_class = self.ln_action_class(embed_action_class)
            cats.append(embed_action_class)

        if self.fc_last_atype is not None:
            embed_last_atype = nonlinearity(self.cfg)(self.fc_last_atype(obs_dict['last_atype'].float()))
            if self.cfg.model.encoder.use_layer_norm:
                embed_last_atype = self.ln_last_atype(embed_last_atype)
            cats.append(embed_last_atype)

        if self.fc_avail_atype is not None:
            embed_avail_atype = nonlinearity(self.cfg)(self.fc_avail_atype(obs_dict['avail_atype'].float()))
            if self.cfg.model.encoder.use_layer_norm:
                embed_avail_atype = self.ln_avail_atype(embed_avail_atype)
            cats.append(embed_avail_atype)

        if len(cats) > 1:
            x = torch.cat(cats, dim=-1)

        # x = self.forward_fc_blocks(x)

        if self.fc_after_concat:
            x = self.fc_after_concat(x)

        if not self.trxli_spell is None:
            x = self.trxli_spell(q=x, k=obs_dict['spell_feature'], v=obs_dict['spell_feature'], mask=obs_dict['avail_spell'].unsqueeze(-2))

        if not self.trxli_item is None:
            x = self.trxli_item(q=x, k=obs_dict['item_feature'], v=obs_dict['item_feature'], mask=obs_dict['avail_use_item'].unsqueeze(-2))

        if not self.trxli_pick_item is None:
            x = self.trxli_pick_item(q=x, k=obs_dict['pick_item_feature'], v=obs_dict['pick_item_feature'], mask=obs_dict['avail_pick_item'].unsqueeze(-2))

        return x

class AvgPoolEncoder(EncoderBase):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.cfg = cfg
        self.obs_space = obs_space

        # Use standard CNN for the image observation in "obs"
        # See all arguments with "-h" to change this head to e.g. ResNet
        self.basic_encoder = create_standard_encoder(cfg, obs_space)
        self.encoder_out_size = self.basic_encoder.encoder_out_size
        obs_shape = get_obs_shape(obs_space)

        self.fc_vector_obs = None
        if 'vector_obs' in obs_shape:
            self.fc_vector_obs = nn.Linear(obs_shape.vector_obs[0], self.cfg.model.encoder.vobs_hidden_dim)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_vector_obs = nn.LayerNorm(self.cfg.model.encoder.vobs_hidden_dim)
            out_size = self.cfg.model.encoder.vobs_hidden_dim
            # Add vector_obs to NN output for more direct access by LSTM core
            self.encoder_out_size += out_size

        self.fc_action_class = None
        if 'action_class' in obs_shape:
            self.fc_action_class = nn.Linear(5, 32)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_action_class = nn.LayerNorm(32)
            self.encoder_out_size += 32

        self.fc_last_atype = None
        if 'last_atype' in obs_shape:
            self.fc_last_atype = nn.Linear(113, 64)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_last_atype = nn.LayerNorm(64)
            self.encoder_out_size += 64

        self.fc_avail_atype = None
        if 'avail_atype' in obs_shape and self.cfg.model.encoder.embed_avail_atype:
            self.fc_avail_atype = nn.Linear(113, 64)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_avail_atype = nn.LayerNorm(64)
            self.encoder_out_size += 64

        self.message_head = None
        if 'message' in obs_shape or 'message_embedding' in obs_shape:
            # _Very_ poor for text understanding,
            # but it is simple and probably enough to overfit to specific sentences.
            if self.cfg.model.encoder.message_encoder == 'mlp':
                self.message_head = nn.Sequential(
                    nn.Linear(obs_shape.message[0], 128),
                    nonlinearity(cfg),
                    nn.Linear(128, 128),
                    nonlinearity(cfg),
                )
                out_size = 128
            elif self.cfg.model.encoder.message_encoder == 'embedding':
                self.message_head = MessageEmbedding(cfg, 80, 128, 128, use_pretrained=False, vocab_size=len(_torchtext_vocab))
                out_size = self.message_head.out_dim
            elif self.cfg.model.encoder.message_encoder == 'embedding_pretrained':
                self.message_head = MessageEmbedding(cfg, 300, 128, 128, use_pretrained=True)
                out_size = self.message_head.out_dim
            elif self.cfg.model.encoder.message_encoder == 'baseline':
                self.message_head = MessageEncoder(self.cfg.model.encoder.msg_hidden_dim, self.cfg.model.encoder.msg_embedding_dim)
                out_size = self.cfg.model.encoder.msg_hidden_dim
            else:
                raise NotImplementedError

            if self.cfg.model.encoder.use_layer_norm:
                self.ln_message = nn.LayerNorm(out_size)
            self.encoder_out_size += out_size

        self.fc_spell = None
        if 'spell_feature' in obs_shape:
            self.fc_spell = nn.Linear(obs_shape.spell_feature[1], 64)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_spell = nn.LayerNorm(64)
            self.encoder_out_size += 64

        self.fc_item = None
        if 'item_feature' in obs_shape:
            self.fc_item = nn.Linear(obs_shape.item_feature[1], 128)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_item = nn.LayerNorm(128)
            self.encoder_out_size += 128

        self.fc_pick_item = None
        if 'pick_item_feature' in obs_shape and self.cfg.env.encode_pick_item_feature:
            self.fc_pick_item = nn.Linear(obs_shape.pick_item_feature[1], 64)
            if self.cfg.model.encoder.use_layer_norm:
                self.ln_pick_item = nn.LayerNorm(64)
            self.encoder_out_size += 64

        if cfg.model.encoder.add_fc_layers_after_concat:
            self.fc_after_concat = nn.Sequential(
                nn.Linear(self.encoder_out_size, self.cfg.model.core.hidden_size),
                nonlinearity(cfg),
            )
            self.encoder_out_size = self.cfg.model.core.hidden_size
        else:
            self.fc_after_concat = None

        log.debug('Policy head output size: %r', self.get_encoder_out_size())

    def forward(self, obs_dict):
        # This one handles the "obs" key which contains the main image
        x = self.basic_encoder(obs_dict)

        cats = [x]
        if self.fc_vector_obs is not None:
            vector_obs = nonlinearity(self.cfg)(self.fc_vector_obs(obs_dict['vector_obs'].float()))
            if self.cfg.model.encoder.use_layer_norm:
                vector_obs = self.ln_vector_obs(vector_obs)
            cats.append(vector_obs)

        if self.message_head is not None:
            if isinstance(self.message_head, MessageEmbedding):
                message = self.message_head(obs_dict['message_embedding'])
            elif isinstance(self.message_head, nn.Sequential):
                message = self.message_head(obs_dict['message'].float() / 255)
            elif isinstance(self.message_head, MessageEncoder):
                message = self.message_head(obs_dict['message'].unsqueeze(1))
            else:
                raise NotImplementedError
            if self.cfg.model.encoder.use_layer_norm:
                message = self.ln_message(message)
            cats.append(message)

        if self.fc_action_class is not None:
            onehot_action_class = F.one_hot(obs_dict['action_class'].long(), num_classes=5).squeeze(-2)
            embed_action_class = nonlinearity(self.cfg)(self.fc_action_class(onehot_action_class.float()))
            if self.cfg.model.encoder.use_layer_norm:
                embed_action_class = self.ln_action_class(embed_action_class)
            cats.append(embed_action_class)

        if self.fc_last_atype is not None:
            embed_last_atype = nonlinearity(self.cfg)(self.fc_last_atype(obs_dict['last_atype'].float()))
            if self.cfg.model.encoder.use_layer_norm:
                embed_last_atype = self.ln_last_atype(embed_last_atype)
            cats.append(embed_last_atype)

        if self.fc_avail_atype is not None:
            embed_avail_atype = nonlinearity(self.cfg)(self.fc_avail_atype(obs_dict['avail_atype'].float()))
            if self.cfg.model.encoder.use_layer_norm:
                embed_avail_atype = self.ln_avail_atype(embed_avail_atype)
            cats.append(embed_avail_atype)

        if self.fc_spell:
            embed_spell = nonlinearity(self.cfg)(self.fc_spell(obs_dict['spell_feature'])) # (n_batch, n_spell, dim)
            if self.cfg.model.encoder.use_layer_norm:
                embed_spell = self.ln_spell(embed_spell)
            if 'avail_spell' in obs_dict:
                embed_spell = obs_dict['avail_spell'].unsqueeze(-1) * embed_spell # mask out empty spell
                embed_spell = torch.sum(embed_spell, dim=-2) / (torch.sum(obs_dict['avail_spell'], dim=-1, keepdim=True) + 1e-10) # (n_batch, dim)
            else:
                embed_spell = torch.mean(embed_spell, dim=-2)
            cats.append(embed_spell)

        if self.fc_item:
            embed_item = nonlinearity(self.cfg)(self.fc_item(obs_dict['item_feature']))
            if self.cfg.model.encoder.use_layer_norm:
                embed_item = self.ln_item(embed_item)
            if 'avail_use_item' in obs_dict:
                embed_item = obs_dict['avail_use_item'].unsqueeze(-1) * embed_item # mask out empty / non-proper item
                embed_item = torch.sum(embed_item, dim=-2) / (torch.sum(obs_dict['avail_use_item'], dim=-1, keepdim=True) + 1e-10)
            else:
                embed_item = torch.mean(embed_item, dim=-2)
            cats.append(embed_item)

        if self.fc_pick_item:
            embed_pick_item = nonlinearity(self.cfg)(self.fc_pick_item(obs_dict['pick_item_feature']))
            if self.cfg.model.encoder.use_layer_norm:
                embed_pick_item = self.ln_pick_item(embed_pick_item)
            if 'avail_pick_item' in obs_dict:
                embed_pick_item = obs_dict['avail_pick_item'].unsqueeze(-1) * embed_pick_item # mask out empty spell
                embed_pick_item = torch.sum(embed_pick_item, dim=-2) / (torch.sum(obs_dict['avail_pick_item'], dim=-1, keepdim=True) + 1e-10)
            else:
                embed_pick_item = torch.mean(embed_pick_item, dim=-2)
            cats.append(embed_pick_item)

        if len(cats) > 1:
            x = torch.cat(cats, dim=-1)

        # x = self.forward_fc_blocks(x)

        if self.fc_after_concat:
            x = self.fc_after_concat(x)

        return x

# register_custom_encoder('nle_obs_vector_encoder', NLEMainEncoder)
