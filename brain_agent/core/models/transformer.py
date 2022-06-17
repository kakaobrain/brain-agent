"""
Copyright 2019 kimiyoung
Copyright 2022 Kakao Brain Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Note:
    This specific file is modification of kimiyoung's version of TrXL
    https://github.com/kimiyoung/transformer-xl which implements TrXL-I in https://arxiv.org/pdf/1910.06764.pdf for
    RL environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Linear(d_inner, d_model),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = F.relu(core_out) + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError


class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)


    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if (mems is not None) and mems.size(0) > 0: # TODO: check
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            if mems.dtype == torch.float16:
                r = r.half() # TODO: should be handled with cfg
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)

            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float("inf")).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float("inf")).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)

        if self.pre_lnorm:
            ##### residual connection
            # modified. applying ReLU before residual connection
            output = w + F.relu(attn_out)
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class MemTransformerLM(nn.Module):
    def __init__(self, cfg, n_layer, n_head, d_model, d_head, d_inner,
                 mem_len=1, pre_lnorm=False):
        super(MemTransformerLM, self).__init__()
        self.cfg = cfg

        self.d_embed = d_model
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.n_layer = n_layer

        self.mem_len = mem_len

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head, d_model, d_head, d_inner,
                    mem_len=mem_len, pre_lnorm=pre_lnorm)
            )

        # create positional encoding-related parameters
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        self.apply(self.initialize)

    def initialize(self, layer):
        def init_weight(weight):
            nn.init.normal_(weight, 0.0, 0.02)  # args.init_std)

        def init_bias(bias):
            nn.init.constant_(bias, 0.0)

        classname = layer.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(layer, 'weight') and layer.weight is not None:
                init_weight(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                init_bias(layer.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(layer, 'emb_projs'):
                for i in range(len(layer.emb_projs)):
                    if layer.emb_projs[i] is not None:
                        nn.init.normal_(layer.emb_projs[i], 0.0, 0.01)  # args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(layer, 'weight'):
                init_weight(layer.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(layer, 'cluster_weight') and layer.cluster_weight is not None:
                init_weight(layer.cluster_weight)
            if hasattr(layer, 'cluster_bias') and layer.cluster_bias is not None:
                init_bias(layer.cluster_bias)
            if hasattr(layer, 'out_projs'):
                for i in range(len(layer.out_projs)):
                    if layer.out_projs[i] is not None:
                        nn.init.normal_(layer.out_projs[i], 0.0, 0.01)  # args.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(layer, 'weight'):
                nn.init.normal_(layer.weight, 1.0, 0.02)  # args.init_std)
            if hasattr(layer, 'bias') and layer.bias is not None:
                init_bias(layer.bias)
        elif classname.find('TransformerLM') != -1:
            if hasattr(layer, 'r_emb'):
                init_weight(layer.r_emb)
            if hasattr(layer, 'r_w_bias'):
                init_weight(layer.r_w_bias)
            if hasattr(layer, 'r_r_bias'):
                init_weight(layer.r_r_bias)
            if hasattr(layer, 'r_bias'):
                init_bias(layer.r_bias)

    def get_core_out_size(self):
        return self.d_model

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen)
            beg_idx = max(0, end_idx - self.mem_len)

            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        # only return last step mem
        new_mems = [m[-1] for m in new_mems]
        new_mems = torch.cat(new_mems, dim=-1)

        return new_mems

    def _forward(self, obs_emb, mems=None, mem_begin_index=None, dones=None, from_learner=False):
        qlen, bsz, _ = obs_emb.size()

        mlen = mems[0].size(0) if mems is not None else 0

        klen = mlen + qlen

        dec_attn_mask = (torch.triu(
             obs_emb.new_ones(qlen, klen), diagonal=1+mlen)
            + torch.tril(
             obs_emb.new_ones(qlen, klen), diagonal=-1)).bool().unsqueeze(-1).repeat(1, 1, bsz)

        for b in range(bsz):
            dec_attn_mask[:, :(mlen - max(0, mem_begin_index[b])), b] = True
            if dones is not None:
                query_done_index = torch.where(dones[:, b] > 0)
                for q in query_done_index[0]:
                    # Going to mask out elements before done for new episode
                    dec_attn_mask[q + 1:, :(mlen + q + 1), b] = True

        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=obs_emb.device,
                               dtype=obs_emb.dtype)

        pos_emb = self.pos_emb(pos_seq)
        core_out = obs_emb

        hids.append(core_out)

        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias,
                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        new_mems = self._update_mems(hids, mems, mlen, qlen) if not from_learner else None

        return core_out, new_mems

    def forward(self, data, mems, mem_begin_index=None, dones=None, from_learner=False):
        if mems is None:
            mems = self.init_mems()
        else:
            mems = torch.split(mems, self.d_model, dim=-1)

        if from_learner:
            data = data.reshape(
                int(self.cfg.optim.batch_size // self.cfg.optim.rollout),
                self.cfg.optim.rollout,
                -1).transpose(0, 1)
        else:
            data = data.unsqueeze(0)

        # input observation should be either (1 x B x dim) or (T x B x dim)
        hidden, new_mems = self._forward(data, mems=mems, mem_begin_index=mem_begin_index, dones=dones,
                                         from_learner=from_learner)

        # reshape hidden: T x B x dim -> TB x dim
        hidden = hidden.transpose(0, 1).reshape(hidden.size(0) * hidden.size(1), -1)

        return hidden, new_mems

    def get_mem_begin_index(self, mems_dones, actor_env_step):
        # mems_dones: (n_batch, n_seq, 1)
        # actor_env_step: (n_batch)
        assert mems_dones.shape[0] == actor_env_step.shape[0], (
            f'The number of batches should be same for mems_done ({mems_dones.shape[0]})'
            + f' and actor_env_step ({actor_env_step.shape[0]})'
        )
        mems_dones = mems_dones.squeeze(-1).cpu()
        actor_env_step = actor_env_step.cpu()

        arange = torch.arange(1, self.cfg.model.core.mem_len + 1, 1).unsqueeze(0)  # 0 ~ self.cfg.mem_len - 1, (1, n_seq)
        step_count_dones = mems_dones * arange  # (n_batch, n_seq)
        step_count_last_dones = step_count_dones.max(dim=-1).values  # (n_batch)
        numel_to_be_attentioned = self.cfg.model.core.mem_len - step_count_last_dones
        mem_begin_index = torch.min(numel_to_be_attentioned, actor_env_step)
        mem_begin_index = mem_begin_index.int().tolist()

        return mem_begin_index
