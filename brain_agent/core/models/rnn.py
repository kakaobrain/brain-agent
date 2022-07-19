import torch
import torch.nn as nn
from typing import Tuple
from torch.nn.utils.rnn import PackedSequence, invert_permutation

def _build_pack_info_from_dones(dones: torch.Tensor, T: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = len(dones)

    rollout_boundaries = dones.clone().detach()
    rollout_boundaries[T - 1::T] = 1
    rollout_boundaries = rollout_boundaries.nonzero().squeeze(dim=1) + 1

    first_len = rollout_boundaries[0].unsqueeze(0)

    if len(rollout_boundaries) <= 1:
        rollout_lengths = first_len
    else:
        rollout_lengths = rollout_boundaries[1:] - rollout_boundaries[:-1]
        rollout_lengths = torch.cat([first_len, rollout_lengths])

    rollout_starts_orig = rollout_boundaries - rollout_lengths

    is_new_episode = dones.clone().detach().view((-1, T))
    is_new_episode = is_new_episode.roll(1, 1)

    is_new_episode[:, 0] = 0
    is_new_episode = is_new_episode.view((-1, ))

    lengths, sorted_indices = torch.sort(rollout_lengths, descending=True)

    cpu_lengths = lengths.to(device='cpu', non_blocking=True)

    rollout_starts_sorted = rollout_starts_orig.index_select(0, sorted_indices)

    select_inds = torch.empty(num_samples, device=dones.device, dtype=torch.int64)

    max_length = int(cpu_lengths[0].item())

    batch_sizes = torch.empty((max_length,), device='cpu', dtype=torch.int64)

    offset = 0
    prev_len = 0
    num_valid_for_length = lengths.size(0)

    unique_lengths = torch.unique_consecutive(cpu_lengths)

    for i in range(len(unique_lengths) - 1, -1, -1):
        valids = lengths[0:num_valid_for_length] > prev_len
        num_valid_for_length = int(valids.float().sum().item())

        next_len = int(unique_lengths[i])

        batch_sizes[prev_len:next_len] = num_valid_for_length

        new_inds = (
            rollout_starts_sorted[0:num_valid_for_length].view(1, num_valid_for_length)
            + torch.arange(prev_len, next_len, device=rollout_starts_sorted.device).view(next_len - prev_len, 1)
        ).view(-1)

        select_inds[offset:offset + new_inds.numel()] = new_inds

        offset += new_inds.numel()

        prev_len = next_len

    assert offset == num_samples
    assert is_new_episode.shape[0] == num_samples

    return rollout_starts_orig, is_new_episode, select_inds, batch_sizes, sorted_indices

def _build_rnn_inputs(x, dones_cpu, rnn_states, T: int):
    rollout_starts, is_new_episode, select_inds, batch_sizes, sorted_indices = _build_pack_info_from_dones(
        dones_cpu, T)
    inverted_select_inds = invert_permutation(select_inds)

    def device(t):
        return t.to(device=x.device)

    select_inds = device(select_inds)
    inverted_select_inds = device(inverted_select_inds)
    sorted_indices = device(sorted_indices)
    rollout_starts = device(rollout_starts)
    is_new_episode = device(is_new_episode)

    x_seq = PackedSequence(x.index_select(0, select_inds), batch_sizes, sorted_indices)

    rnn_states = rnn_states.index_select(0, rollout_starts)
    is_same_episode = (1 - is_new_episode.view(-1, 1)).index_select(0, rollout_starts)
    rnn_states = rnn_states * is_same_episode

    return x_seq, rnn_states, inverted_select_inds


class LSTM(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()

        self.cfg = cfg
        self.core = nn.LSTM(input_size, cfg.model.core.hidden_size, cfg.model.core.n_rnn_layer)

        self.core_output_size = cfg.model.core.hidden_size
        self.n_rnn_layer = cfg.model.core.n_rnn_layer
        self.apply(self.initialize)

    def initialize(self, layer):
        gain = 1.0

        if self.cfg.model.core.core_init == 'tensorflow_default':
            if type(layer) == nn.LSTM:
                for n, p in layer.named_parameters():
                    if 'weight_ih' in n:
                        nn.init.xavier_uniform_(p.data, gain=gain)
                    elif 'weight_hh' in n:
                        nn.init.orthogonal_(p.data)
                    elif 'bias_ih' in n:
                        p.data.fill_(0)
                        # Set forget-gate bias to 1
                        n = p.size(0)
                        p.data[(n // 4):(n // 2)].fill_(1)
                    elif 'bias_hh' in n:
                        p.data.fill_(0)
        elif self.cfg.model.core.core_init == 'torch_default':
            pass
        else:
            raise NotImplementedError

    def forward(self, head_output, rnn_states, dones, is_seq):
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.n_rnn_layer > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.model.core.n_rnn_layer, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        h, c = torch.split(rnn_states, self.cfg.model.core.hidden_size, dim=2)

        x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
        new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.n_rnn_layer > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states

    def get_core_out_size(self):
        return self.core_output_size

    @classmethod
    def build_rnn_inputs(cls, x, dones_cpu, rnn_states, T: int):
        return _build_rnn_inputs(x, dones_cpu, rnn_states, T)
