import torch
from torch import nn
from brain_agent.core.agents.agent_abc import ActorCriticBase
from brain_agent.envs.dmlab.dmlab_model import DmlabEncoder
from brain_agent.core.models.transformer import MemTransformerLM
from brain_agent.core.models.rnn import LSTM
from brain_agent.core.algos.popart import update_parameters, update_mu_sigma
from brain_agent.core.models.model_utils import normalize_obs_return
from brain_agent.core.models.action_distributions import sample_actions_log_probs
from brain_agent.utils.utils import AttrDict

class DMLabMultiTaskAgent(ActorCriticBase):
    def __init__(self, cfg, action_space, obs_space, num_levels, need_half=False):
        super().__init__(action_space, cfg)

        self.encoder = DmlabEncoder(cfg, obs_space)

        core_input_size = self.encoder.get_encoder_out_size()
        if cfg.model.extended_input:
            core_input_size += action_space.n + 1

        if cfg.model.core.core_type == "trxl":
            self.core = MemTransformerLM(cfg, n_layer=cfg.model.core.n_layer, n_head=cfg.model.core.n_heads,
                                         d_head=cfg.model.core.d_head, d_model=core_input_size,
                                         d_inner=cfg.model.core.d_inner,
                                         mem_len=cfg.model.core.mem_len, pre_lnorm=True)

        elif cfg.model.core.core_type == "rnn":
            self.core = LSTM(cfg, core_input_size)
        else:
            raise Exception('Error: Not support given model core_type')

        core_out_size = self.core.get_core_out_size()

        if self.cfg.model.use_popart:
            self.register_buffer('mu', torch.zeros(num_levels, requires_grad=False))
            self.register_buffer('nu', torch.ones(num_levels, requires_grad=False))
            self.critic_linear = nn.Linear(core_out_size, num_levels)
            self.beta = self.cfg.model.popart_beta
        else:
            self.critic_linear = nn.Linear(core_out_size, 1)

        self.action_parameterization = self.get_action_parameterization(core_out_size)

        self.action_parameterization.apply(self.initialize)
        self.critic_linear.apply(self.initialize)
        self.train()

        self.need_half = need_half
        if self.need_half:
            self.half()

    def initialize(self, layer):
        def init_weight(weight):
            nn.init.normal_(weight, 0.0, 0.02)

        def init_bias(bias):
            nn.init.constant_(bias, 0.0)

        classname = layer.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(layer, 'weight') and layer.weight is not None:
                init_weight(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                init_bias(layer.bias)

    # TODO: functionalize popart
    # def update_parameters(self, mu, sigma, oldmu, oldsigma):
    #     self.critic_linear.weight.data, self.critic_linear.bias.data = \
    #         update_parameters(self.critic_linear.weight, self.critic_linear.bias, mu, sigma, oldmu, oldsigma)

    def update_parameters(self, mu, sigma, oldmu, oldsigma):
        self.critic_linear.weight.data = (self.critic_linear.weight.t() * oldsigma / sigma).t()
        self.critic_linear.bias.data = (oldsigma * self.critic_linear.bias + oldmu - mu) / sigma

    # def update_mu_sigma(self, vs, task_ids):
    #     vs = vs.reshape(-1, self.cfg.optim.rollout)
    #     task_ids = task_ids.reshape(-1, self.cfg.optim.rollout)[:,0]
    #     clamp_max = 1e4 if hasattr(self, 'need_half') and self.need_half else 1e6
    #
    #     mu, nu, sigma, oldmu, oldsigma = update_mu_sigma(self.nu, self.mu, vs, task_ids,
    #                                                      self.cfg.model.popart_clip_min, clamp_max, self.cfg.model.popart_beta)
    #     self.mu = mu
    #     self.nu = nu
    #
    #     return self.mu, sigma, oldmu, oldsigma
    def update_mu_sigma(self, vs, task_ids, cfg=None):
        oldnu = self.nu.clone()
        oldsigma = torch.sqrt(oldnu - self.mu ** 2)
        oldsigma[torch.isnan(oldsigma)] = self.cfg.model.popart_clip_min
        clamp_max = 1e4 if hasattr(self, 'need_half') and self.need_half else 1e6
        oldsigma = torch.clamp(oldsigma, min=cfg.model.popart_clip_min, max=clamp_max)
        oldmu = self.mu.clone()

        vs = vs.reshape(-1, self.cfg.optim.rollout)
        # same task ids over all time steps within a single episode
        task_ids_per_epi = task_ids.reshape(-1, self.cfg.optim.rollout)[:, 0]
        for i in range(len(task_ids_per_epi)):
            task_id = task_ids_per_epi[i]
            v = torch.mean(vs[i])
            self.mu[task_id] = (1 - self.beta) * self.mu[task_id] + self.beta * v
            self.nu[task_id] = (1 - self.beta) * self.nu[task_id] + self.beta * (v ** 2)

        sigma = torch.sqrt(self.nu - self.mu ** 2)
        sigma[torch.isnan(sigma)] = self.cfg.model.popart_clip_min
        sigma = torch.clamp(sigma, min=cfg.model.popart_clip_min, max=clamp_max)

        return self.mu, sigma, oldmu, oldsigma

    def forward_head(self, obs_dict, actions, rewards, decode=False):
        obs_dict = normalize_obs_return(obs_dict, self.cfg)  # before normalize, [0,255]
        if self.need_half:
            obs_dict['obs'] = obs_dict['obs'].half()

        x = self.encoder(obs_dict, decode=decode)
        if decode:
            self.reconstruction = self.encoder.basic_encoder.reconstruction
        else:
            self.reconstruction = None

        x_extended = []
        if self.cfg.model.extended_input:
            assert torch.min(actions) >= -1 and torch.max(actions) < self.action_space.n
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            prev_actions = nn.functional.one_hot(actions, self.action_space.n).float()
            prev_actions[done_ids] = 0.
            x_extended.append(prev_actions)
            x_extended.append(rewards.clamp(-1, 1).unsqueeze(1))

        x = torch.cat([x] + x_extended, dim=-1)

        if self.need_half:
            x = x.half()

        return x

    def forward_core_transformer(self, head_output, mems=None, mem_begin_index=None, dones=None, from_learner=False):
        x, new_mems = self.core(head_output, mems, mem_begin_index, dones=dones, from_learner=from_learner)
        return x, new_mems

    def forward_core_rnn(self, head_output, rnn_states, dones, is_seq=None):
        x, new_rnn_states = self.core(head_output, rnn_states, dones, is_seq)
        return x, new_rnn_states

    def forward_tail(self, core_output, task_ids, with_action_distribution=False):
        values = self.critic_linear(core_output)
        normalized_values = values.clone()
        sigmas = torch.ones((values.size(0), 1), requires_grad=False)
        mus = torch.zeros((values.size(0), 1), requires_grad=False)
        if self.cfg.model.use_popart:
            normalized_values = normalized_values.gather(dim=1, index=task_ids)
            with torch.no_grad():
                nus = self.nu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                mus = self.mu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                sigmas = torch.sqrt(nus - mus ** 2)
                sigmas[torch.isnan(sigmas)] = self.cfg.model.popart_clip_min
                clamp_max = 1e4 if self.need_half else 1e6
                sigmas = torch.clamp(sigmas, min=self.cfg.model.popart_clip_min, max=clamp_max)
                values = normalized_values * sigmas + mus

        action_distribution_params, action_distribution = self.action_parameterization(core_output)

        actions, log_prob_actions = sample_actions_log_probs(action_distribution)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_distribution_params,
            log_prob_actions=log_prob_actions,
            values=values,
            normalized_values=normalized_values,
            sigmas=sigmas,
            mus=mus
        ))

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    def forward(self, obs_dict, actions, rewards, mems=None, mem_begin_index=None, rnn_states=None, dones=None, is_seq=None,
                task_ids=None, with_action_distribution=False, from_learner=False):
        x = self.forward_head(obs_dict, actions, rewards)

        if self.cfg.model.core.core_type == 'trxl':
            x, new_mems = self.forward_core_transformer(x, mems, mem_begin_index, from_learner=from_learner)
        elif self.cfg.model.core.core_type == 'rnn':
            x, new_rnn_states = self.forward_core_rnn(x, rnn_states, dones, is_seq)

        assert not x.isnan().any()

        result = self.forward_tail(x, task_ids, with_action_distribution=with_action_distribution)

        if self.cfg.model.core.core_type == "trxl":
            result.mems = new_mems
        elif self.cfg.model.core.core_type == 'rnn':
            result.rnn_states = new_rnn_states

        return result
