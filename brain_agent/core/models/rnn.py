import torch
import torch.nn as nn


RNN_MODULES = ['lstm', 'gru']


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
        if hasattr(layer, 'bias') and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if self.cfg.model.core.core_init == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
        elif self.cfg.model.core.core_init == 'xavier_uniform':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                nn.init.xavier_uniform_(layer.weight_ih, gain=gain)
                nn.init.xavier_uniform_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
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
        if not is_seq:
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)
            h_stack, c_stack = h, c
        else:
            outputs = []
            hs, cs = [], []
            num_trajectories = head_output.size(0) // self.cfg.optim.rollout
            head_output = head_output.view(num_trajectories, self.cfg.optim.rollout, -1)  # B x T x D
            is_new_episode = dones.clone().detach().view((-1, self.cfg.optim.rollout))
            is_new_episode = is_new_episode.roll(1, 1)
            is_new_episode[:, 0] = 0
            for t in range(self.cfg.optim.rollout):
                h = (1.0 - is_new_episode[:, t]).view(1, -1, 1) * h
                c = (1.0 - is_new_episode[:, t]).view(1, -1, 1) * c
                output, (h, c) = self.core(head_output[:, t, :].unsqueeze(0), (h.contiguous(), c.contiguous()))
                outputs.append(output.squeeze(0))
                hs.append(h)
                cs.append(c)
            x = torch.stack(outputs)    # T x B x D
            x = x.permute(1, 0, 2).flatten(0,1) # (BT) x D
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


class GRU(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()

        self.cfg = cfg
        self.core = nn.GRU(input_size, cfg.model.core.hidden_size, cfg.model.core.n_rnn_layer)

        self.core_output_size = cfg.model.core.hidden_size
        self.n_rnn_layer = cfg.model.core.n_rnn_layer
        self.apply(self.initialize)

    def initialize(self, layer):
        gain = 1.0
        if hasattr(layer, 'bias') and isinstance(layer.bias, torch.nn.parameter.Parameter):
            layer.bias.data.fill_(0)

        if self.cfg.model.core.core_init == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
        elif self.cfg.model.core.core_init == 'xavier_uniform':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                nn.init.xavier_uniform_(layer.weight_ih, gain=gain)
                nn.init.xavier_uniform_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
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

        if not is_seq:
            x, new_rnn_states = self.core(head_output, rnn_states)
        else:
            outputs = []
            num_trajectories = head_output.size(0) // self.cfg.optim.rollout
            head_output = head_output.view(num_trajectories, self.cfg.optim.rollout, -1)  # B x T x D
            is_new_episode = dones.clone().detach().view((-1, self.cfg.optim.rollout))
            is_new_episode = is_new_episode.roll(1, 1)
            is_new_episode[:, 0] = 0
            for t in range(self.cfg.optim.rollout):
                rnn_states = (1.0 - is_new_episode[:, t]).view(1, -1, 1) * rnn_states
                output, rnn_states = self.core(head_output[:, t, :].unsqueeze(0), rnn_states)
                outputs.append(output.squeeze(0))
            x = torch.stack(outputs)    # T x B x D
            x = x.permute(1, 0, 2).flatten(0,1) # (BT) x D
            new_rnn_states = rnn_states

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
