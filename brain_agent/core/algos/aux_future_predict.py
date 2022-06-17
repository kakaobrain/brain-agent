import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_agent.core.models.model_utils import normalize_obs_return
from brain_agent.core.models.resnet import ResBlock
from brain_agent.core.models.causal_transformer import CausalTransformer

''' 
To learn the useful representations, this module tries to minimize the difference between the predicted future 
observations and the real observations for k(2~10) steps. It uses causal transformer in autoregressive manner 
to predict the state transitions when the actual action sequence and state embedding of current step are given. 

For the current state embedding, we use the outputs of TrXL core. These state embeddings are concatenated 
with actual actions taken, and then fed into the causal transformer. The causal transformer iteratively 
produces next states embedding in autoregressive manner. 

After we produce all the future state embeddings for K (2~10) steps, we apply transposed convolutional 
decoding layer to map the state embedding into the original image observation space. Finally, we calculate 
the L2 distance between decoded image outputs and real images.

With this auxiliary loss added, mean HNS(human normalized score) of 30 tasks increased from 123.6 to 128.0, 
although capped mean HNS decreased from 91.25 to 90.53.
You can add this module by changing the argument learner.use_aux_future_pred_loss to True.

'''

class FuturePredict(nn.Module):
    def __init__(self, cfg, encoder_input_ch, conv_head_out_size, core_out_size, action_space):
        super(FuturePredict, self).__init__()
        self.cfg = cfg
        self.action_space = action_space
        self.n_action = action_space.n
        self.horizon_k: int = 10
        self.time_subsample: int = 6
        self.forward_subsample: int = 2
        self.core_out_size = core_out_size
        self.conv_head_out_size = conv_head_out_size

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_sizes = [action_space.n]
        else:
            self.action_sizes = [space.n for space in action_space.spaces]

        self.g = nn.Sequential(
            nn.Linear(core_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, conv_head_out_size)
        )

        self.mse_loss = nn.MSELoss(reduction='none')
        self.causal_transformer = CausalTransformer(core_out_size, action_space.n, pre_lnorm=True)
        self.causal_transformer_window = self.causal_transformer.mem_len

        if cfg.model.encoder.encoder_subtype == 'resnet_impala':
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        elif cfg.model.encoder.encoder_subtype == 'resnet_impala_large':
            resnet_conf = [[32, 2], [64, 2], [64, 2]]
        self.conv_out_ch = resnet_conf[-1][0]
        layers_decoder = list()
        curr_input_channels = encoder_input_ch
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):

            for j in range(res_blocks):
                layers_decoder.append(ResBlock(cfg, curr_input_channels, curr_input_channels))
            layers_decoder.append(
                nn.ConvTranspose2d(out_channels, curr_input_channels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1)
            )
            curr_input_channels = out_channels

        layers_decoder.reverse()
        self.layers_decoder = nn.Sequential(*layers_decoder)

    def calc_loss(self, mb, b_t, mems, mems_actions, mems_dones, mem_begin_index, num_traj, recurrence):
        not_dones = (1.0 - mb.dones).view(num_traj, recurrence, 1)
        mask_res = self._build_mask_and_subsample(not_dones)
        (forward_mask, unroll_subsample, time_subsample, max_k) = mask_res

        actions_raw = mb.actions.view(num_traj, recurrence, -1).long()
        mems = torch.split(mems, self.core_out_size, dim=-1)[-1].transpose(0,1)
        mem_len = mems.size(1)
        mems_actions = mems_actions.transpose(0,1)
        b_t = b_t.reshape(num_traj, recurrence, -1)
        obs = normalize_obs_return(mb.obs, self.cfg)
        obs = obs['obs'].reshape(num_traj, recurrence, 3, 72, 96)

        cat_out = torch.cat([mems, b_t], dim=1)
        cat_action = torch.cat([mems_actions, actions_raw], dim=1)
        mems_not_dones = 1.0 - mems_dones.float()
        cat_not_dones = torch.cat([mems_not_dones, not_dones], dim=1)
        mem_begin_index = torch.tensor(mem_begin_index, device=cat_out.device)

        x, y, z, w, forward_targets = self._get_transfomrer_input(obs, cat_out, cat_action, cat_not_dones, time_subsample, max_k, mem_len, mem_begin_index)
        h_pred_stack, done_mask = self._make_transformer_pred(x,y,z,w, num_traj, max_k)
        h_pred_stack = h_pred_stack.index_select(0, unroll_subsample)
        final_pred = self.g(h_pred_stack)

        x_dec = final_pred.view(-1, self.conv_out_ch, 9, 12)
        for i in range(len(self.layers_decoder)):
            layer_decoder = self.layers_decoder[i]
            x_dec = layer_decoder(x_dec)
        x_dec = torch.tanh(x_dec)

        with torch.no_grad():
            forward_targets = forward_targets.flatten(0, 1)
            forward_targets = forward_targets.transpose(0, 1)
            forward_targets = forward_targets.index_select(0, unroll_subsample)
            forward_targets = forward_targets.flatten(0, 1)

        loss = self.mse_loss(x_dec, forward_targets)
        loss = loss.view(loss.size()[0], -1).mean(-1, keepdim=True)

        final_mask = torch.logical_and(forward_mask, done_mask)
        loss = torch.masked_select(loss, final_mask.flatten(0,1)).mean()
        loss = loss * self.cfg.learner.aux_future_pred_loss_coeff
        return loss

    def _make_transformer_pred(self, states, actions, not_dones, mem_begin_index, num_traj, max_k):
        actions = nn.functional.one_hot(actions.long(), num_classes=self.n_action).squeeze(3).float()
        actions = actions.view(self.time_subsample*num_traj, -1, self.n_action).transpose(0,1)
        states = states.view(self.time_subsample*num_traj, -1, self.core_out_size).transpose(0,1)
        not_dones = not_dones.view(self.time_subsample*num_traj, -1, 1).transpose(0,1)
        dones = 1.0 - not_dones
        tokens = torch.cat([states, actions], dim=2)
        input_tokens_past = tokens[:self.causal_transformer_window-1,:,:]
        input_token_current = tokens[self.causal_transformer_window-1,:,:].unsqueeze(0)
        input_token = torch.cat([input_tokens_past, input_token_current], dim=0)

        mem_begin_index = mem_begin_index.view(-1)

        y, mems, mem_begin_index = self.causal_transformer(input_token, mem_begin_index, num_traj, mems=None)

        lst = []
        for mem in mems:
            lst.append(mem)
        mems = lst

        y = y[-1].unsqueeze(0)
        out = [y]
        for i in range(max_k-1):
            new_input = torch.cat([y, actions[self.causal_transformer_window+i].unsqueeze(0)], dim=-1)
            y, mems, mem_begin_index = self.causal_transformer(new_input, mem_begin_index, num_traj, mems=mems)
            out.append(y)

        done_mask = torch.ge(dones.sum(dim=0), 1.0)
        done_mask = torch.logical_not(done_mask)  # False means masking.

        return torch.stack(out).squeeze(1), done_mask

    def _get_transfomrer_input(self, obs, cat_out, cat_action, cat_not_dones, time_subsample, max_k, mem_len, mem_begin_index):
        out_lst = []
        actions_lst = []
        not_dones_lst = []
        mem_begin_index_lst = []
        target_lst = []

        for i in range(self.time_subsample):
            first_idx = mem_len + time_subsample[i]
            max_idx = first_idx + max_k
            min_idx = first_idx - self.causal_transformer_window + 1
            x = cat_out[:, min_idx:max_idx, :]
            y = cat_action[:, min_idx:max_idx, :]
            z = cat_not_dones[:, min_idx:max_idx, :]
            w = mem_begin_index + time_subsample[i] + 1
            k = obs[:, time_subsample[i]+1:time_subsample[i]+1+max_k]
            out_lst.append(x)
            actions_lst.append(y)
            not_dones_lst.append(z)
            mem_begin_index_lst.append(w)
            target_lst.append(k)


        return torch.stack(out_lst), torch.stack(actions_lst), torch.stack(not_dones_lst), \
               torch.stack(mem_begin_index_lst), torch.stack(target_lst)

    def _build_mask_and_subsample(self, not_dones):
        t = not_dones.size(1) - self.horizon_k

        not_dones_unfolded = self._build_unfolded(not_dones[:, :-1].to(dtype=torch.bool), self.horizon_k) # 10, 32, 24, 1
        time_subsample = torch.randperm(t - 2, device=not_dones.device, dtype=torch.long)[0:self.time_subsample]

        forward_mask = torch.cumprod(not_dones_unfolded.index_select(2, time_subsample), dim=0).to(dtype=torch.bool) # 10, 32, 6, 1
        forward_mask = forward_mask.flatten(1, 2) # 10, 192, 1

        max_k = forward_mask.flatten(1).any(-1).nonzero().max().item() + 1

        unroll_subsample = torch.randperm(max_k, dtype=torch.long)[0:self.forward_subsample]

        max_k = unroll_subsample.max().item() + 1

        unroll_subsample = unroll_subsample.to(device=not_dones.device)
        forward_mask = forward_mask.index_select(0, unroll_subsample)

        return forward_mask, unroll_subsample, time_subsample, max_k

    def _build_unfolded(self, x, k: int):
        tobe_cat = x.new_zeros(x.size(0), k, x.size(2))
        cat = torch.cat((x, tobe_cat), 1)
        cat = cat.unfold(1, size=k, step=1)
        cat = cat.permute(3, 0, 1, 2)
        return cat

