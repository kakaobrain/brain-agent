import torch
import torch.nn as nn
from torch.nn import functional as F
from brain_agent.core.models.transformer import PositionalEmbedding, RelPartialLearnableDecoderLayer


class CausalTransformer(nn.Module):
    def __init__(self, core_out_size, n_action, pre_lnorm=False):
        super().__init__()
        self.n_layer = 4
        self.n_head = 3
        self.d_head = 64
        self.d_inner = 512
        self.d_model = 196
        self.mem_len = 64

        self.blocks = nn.ModuleList()
        for i in range(self.n_layer):
            self.blocks.append(RelPartialLearnableDecoderLayer(
                n_head=self.n_head, d_model=self.d_model, d_head=self.d_head, d_inner=self.d_inner,
                pre_lnorm=pre_lnorm))
        # decoder head
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, core_out_size, bias=True)

        self.apply(self._init_weights)

        self.state_encoder = nn.Sequential(nn.Linear(core_out_size+n_action, self.d_model), nn.Tanh())

        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, states, mem_begin_index, num_traj, mems=None):
        state_embeddings = self.state_encoder(states)  # (batch * block_size, n_embd)
        x = state_embeddings

        qlen, bsz, _ = x.size()  # qlen is number of characters in input ex

        if mems is not None:
            mlen = mems[0].size(0)
            klen = mlen + qlen
            dec_attn_mask_triu = torch.triu(state_embeddings.new_ones(qlen, klen), diagonal=1 + mlen)
            dec_attn_mask = dec_attn_mask_triu.bool().unsqueeze(-1).repeat(1, 1, bsz)
        else:
            mlen = self.mem_len
            klen = self.mem_len
            dec_attn_mask_triu = torch.triu(state_embeddings.new_ones(qlen, klen), diagonal=1)
            dec_attn_mask = dec_attn_mask_triu.bool().unsqueeze(-1).repeat(1, 1, bsz)

        for b in range(bsz):
            if mlen-mem_begin_index[b] > 0:
                dec_attn_mask[:, :mlen-mem_begin_index[b], b] = True

        dec_attn_mask = dec_attn_mask.transpose(1,2)
        temp = torch.logical_not(dec_attn_mask)
        temp = torch.sum(temp, dim=2, keepdim=True)
        temp = torch.ge(temp, 0.1)

        dec_attn_mask = torch.logical_and(temp, dec_attn_mask)
        dec_attn_mask = dec_attn_mask.transpose(1, 2)

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=states.device, dtype=states.dtype)  # [99,...0]
        pos_emb = self.pos_emb(pos_seq)  # T x 1 x dim

        hids = [x]
        for i, layer in enumerate(self.blocks):
            if mems is not None:
                x = layer(x, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems[i])
            else:
                x = layer(x, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=None)
            hids.append(x)

        if mems is None:
            new_mems = hids
        else:
            new_mems = self._update_mems(hids, mems, mlen, qlen)
            mem_begin_index = mem_begin_index + 1
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, new_mems, mem_begin_index


    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        new_mems = []
        end_idx = mlen + max(0, qlen) # ext_len looks to usually be 0 (in their experiments anyways
        beg_idx = max(0, end_idx - self.mem_len) #if hids[0].shape[0] > 1 else 0
        for i in range(len(hids)):
            cat = torch.cat([mems[i], hids[i]], dim=0) # (m_len + q) x B x dim
            aa=1
            if beg_idx == end_idx: # cfg.mem_len=0
                new_mems.append(torch.zeros(cat[0:1].size()))
            else: # cfg.mem_len > 0
                new_mems.append(cat[beg_idx:end_idx])

        return new_mems