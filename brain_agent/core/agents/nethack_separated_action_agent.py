import torch
from torch import nn
from brain_agent.core.agents.agent_abc import ActorCriticBase
from brain_agent.envs.nethack.nethack_model import create_nethack_encoder
from brain_agent.core.models.transformer import MemTransformerLM
from brain_agent.core.models.rnn import LSTM, GRU, RNN_MODULES
from brain_agent.core.algos.popart import update_parameters, update_mu_sigma
from brain_agent.core.models.model_utils import normalize_obs_return
from brain_agent.core.models.action_distributions import sample_actions_log_probs
from brain_agent.utils.utils import AttrDict
from brain_agent.envs.nethack.inven.models import ActionBaseModel
from brain_agent.core.models.action_distributions import CategoricalActionDistribution
from brain_agent.envs.nethack.ids.action_classes import ActionClasses


class NetHackSeparatedActionAgent(ActorCriticBase):
    def __init__(self, cfg, action_space, obs_space, num_levels, need_half=False):
        super().__init__(action_space, cfg)
        assert cfg.model.extended_input is False, 'extended_input is not supported for separated action agent'
        assert cfg.model.extended_input_action is False, 'extended_input_action is not supported for separated action agent'

        self.encoder = create_nethack_encoder(cfg, obs_space)

        core_input_size = self.encoder.get_encoder_out_size()
        if cfg.model.extended_input:
            core_input_size += action_space.n + 1

        if cfg.model.core.core_type == "trxl":
            self.core = MemTransformerLM(cfg, n_layer=cfg.model.core.n_layer, n_head=cfg.model.core.n_heads,
                                         d_head=cfg.model.core.d_head, d_model=core_input_size,
                                         d_inner=cfg.model.core.d_inner,
                                         mem_len=cfg.model.core.mem_len, pre_lnorm=True)
        elif cfg.model.core.core_type == "lstm":
            self.core = LSTM(cfg, core_input_size)
        elif cfg.model.core.core_type == "gru":
            self.core = GRU(cfg, core_input_size)
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

        if self.cfg.model.extended_input_action or self.cfg.model.extended_input:
            if self.cfg.model.use_prev_action_emb:
                self.prev_action_encoder = ActionBaseModel(
                    action_space['atype'].n, self.cfg.model.prev_action_emb_dim, self.cfg.model.prev_action_use_index_select)

        self.embedding_vector_obs = None
        self.embedding_vector_obs = nn.Linear(self.encoder.obs_space['vector_obs'].shape[0], core_out_size)

        self.atype_parameterization = self.get_extra_action_parameterization(core_out_size, num_actions=action_space['atype'].n)
        self.direction_parameterization = self.get_extra_action_parameterization(core_out_size, num_actions=action_space['direction'].n)
        self.fc_k_spell = nn.Linear(self.encoder.obs_space['spell_feature'].shape[1], core_out_size)
        self.fc_k_use_item = nn.Linear(self.encoder.obs_space['item_feature'].shape[1], core_out_size)
        self.fc_k_pick_item = nn.Linear(self.encoder.obs_space['pick_item_feature'].shape[1], core_out_size)

        self.atype_parameterization.apply(self.initialize)
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

    def update_parameters(self, mu, sigma, oldmu, oldsigma):
        self.critic_linear.weight.data = (self.critic_linear.weight.t() * oldsigma / sigma).t()
        self.critic_linear.bias.data = (oldsigma * self.critic_linear.bias + oldmu - mu) / sigma

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
        self.obs_dict = obs_dict
        if self.need_half:
            for key, var in obs_dict.items():
                obs_dict[key] = var.half()

        x = self.encoder(obs_dict)

        x_extended = []
        if self.cfg.model.extended_input or self.cfg.model.extended_input_action:
            # -1 is transformed into all zero vector
            assert torch.min(actions['atype']) >= -1 and torch.max(actions['atype']) < self.action_space['atype'].n
            done_ids = actions['atype'].eq(-1).nonzero(as_tuple=False)
            actions['atype'][done_ids] = 0
            prev_actions = torch.nn.functional.one_hot(actions['atype'], self.action_space['atype'].n).float()
            prev_actions['atype'][done_ids] = 0.

            if self.cfg.model.use_prev_action_emb:
                if self.cfg.model.extended_input:
                    x = torch.cat((x, self.prev_action_encoder(actions['atype']), rewards.clamp(-1, 1).unsqueeze(1)), dim=1)
                elif self.cfg.model.extended_input_action:
                    x = torch.cat((x, self.prev_action_encoder(actions['atype'])), dim=1)
            else:
                if self.cfg.model.extended_input:
                    x = torch.cat((x, prev_actions['atype'], rewards.clamp(-1, 1).unsqueeze(1)), dim=1)
                elif self.cfg.model.extended_input_action:
                    x = torch.cat((x, prev_actions['atype']), dim=1)

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

        embed_vector_obs = torch.tanh(self.embedding_vector_obs(self.obs_dict['vector_obs']))
        core_output = core_output * embed_vector_obs

        avail_atype = self.obs_dict['avail_atype'] if 'avail_atype' in self.obs_dict else None
        atype_distribution_params, atype_distribution = self.atype_parameterization(core_output, avail_atype,
                                                                                    nethack_action_restriction=True)
        atypes, log_prob_atypes = sample_actions_log_probs(atype_distribution)

        avail_direction = self.obs_dict['avail_direction'] if 'avail_direction' in self.obs_dict else None
        direction_distribution_params, direction_distribution = self.direction_parameterization(core_output, avail_direction)
        directions, log_prob_directions = sample_actions_log_probs(direction_distribution)

        q_spell = core_output.unsqueeze(-2)  # (**, 1, dim)
        k_spell = self.fc_k_spell(self.obs_dict['spell_feature'])  # (**, 5, dim)
        spell_distribution_params = torch.matmul(q_spell, k_spell.transpose(-2, -1)).squeeze(
            -2)  # / q_spell.shape[-1] ** 0.5 # (**, 5)
        if 'avail_spell' in self.obs_dict:
            avail_spell = self.obs_dict['avail_spell']
            spell_distribution_params = avail_spell * spell_distribution_params - (1 - avail_spell) * 1e10
        spell_distribution = CategoricalActionDistribution(raw_logits=spell_distribution_params)
        spells, log_prob_spells = sample_actions_log_probs(spell_distribution)

        q_use_item = core_output.unsqueeze(-2)  # (**, 1, dim)
        k_use_item = self.fc_k_use_item(self.obs_dict['item_feature'])  # (**, 55, dim)
        use_item_distribution_params = torch.matmul(q_use_item, k_use_item.transpose(-2, -1)).squeeze(
            -2)  # / q_use_item.shape[-1] ** 0.5  # (**, 5)
        if 'avail_use_item' in self.obs_dict:
            avail_use_item = self.obs_dict['avail_use_item']
            use_item_distribution_params = avail_use_item * use_item_distribution_params - (1 - avail_use_item) * 1e10
        use_item_distribution = CategoricalActionDistribution(raw_logits=use_item_distribution_params)
        use_items, log_prob_use_items = sample_actions_log_probs(use_item_distribution)

        q_pick_item = core_output.unsqueeze(-2)  # (**, 1, dim)
        k_pick_item = self.fc_k_pick_item(self.obs_dict['pick_item_feature'])  # (**, 55, dim)
        pick_item_distribution_params = torch.matmul(q_pick_item, k_pick_item.transpose(-2, -1)).squeeze(
            -2)  # / q_pick_item.shape[-1] ** 0.5 # (**, 5)
        if 'avail_pick_item' in self.obs_dict:
            avail_pick_item = self.obs_dict['avail_pick_item']
            pick_item_distribution_params = avail_pick_item * pick_item_distribution_params - (
                        1 - avail_pick_item) * 1e10
        pick_item_distribution = CategoricalActionDistribution(raw_logits=pick_item_distribution_params)
        pick_items, log_prob_pick_items = sample_actions_log_probs(pick_item_distribution)

        result = AttrDict(dict(
            atypes=atypes,
            atype_logits=atype_distribution_params,
            # perhaps `action_logits` is not the best name here since we now support continuous actions
            log_prob_atypes=log_prob_atypes,

            directions=directions,
            direction_logits=direction_distribution_params,
            log_prob_directions=log_prob_directions,

            spells=spells,
            spell_logits=spell_distribution_params,
            log_prob_spells=log_prob_spells,

            use_items=use_items,
            use_item_logits=use_item_distribution_params,
            log_prob_use_items=log_prob_use_items,

            pick_items=pick_items,
            pick_item_logits=pick_item_distribution_params,
            log_prob_pick_items=log_prob_pick_items,

            values=values,
            normalized_values=normalized_values,
            sigmas=sigmas,
            mus=mus
        ))

        if with_action_distribution:
            result.atype_distribution = atype_distribution
            result.direction_distribution = direction_distribution
            result.spell_distribution = spell_distribution
            result.use_item_distribution = use_item_distribution
            result.pick_item_distribution = pick_item_distribution

        return result

    def forward(self, obs_dict, actions, rewards, mems=None, mem_begin_index=None, rnn_states=None, dones=None,
                is_seq=None, task_ids=None, with_action_distribution=False, from_learner=False):
        x = self.forward_head(obs_dict, actions, rewards)

        if self.cfg.model.core.core_type == 'trxl':
            x, new_mems = self.forward_core_transformer(x, mems, mem_begin_index, from_learner=from_learner)
        elif self.cfg.model.core.core_type in RNN_MODULES:
            x, new_rnn_states = self.forward_core_rnn(x, rnn_states, dones, is_seq)

        assert not x.isnan().any()
        result = self.forward_tail(x, task_ids, with_action_distribution=with_action_distribution)

        if self.cfg.model.core.core_type == "trxl":
            result.mems = new_mems
        elif self.cfg.model.core.core_type in RNN_MODULES:
            result.rnn_states = new_rnn_states

        return result

    def get_tensors_to_squeeze(self):
        tensors_to_squeeze = [
            'atypes', 'log_prob_atypes',
            'directions', 'log_prob_directions',
            'spells', 'log_prob_spells',
            'use_items', 'log_prob_use_items',
            'pick_items', 'log_prob_pick_items',
            'policy_version', 'values',
            'rewards', 'dones', 'rewards_cpu', 'dones_cpu', 'raw_rewards',
        ]
        return tensors_to_squeeze

    def get_log_prob_action(self, result, mb):
        log_prob_atypes = result.atype_distribution.log_prob(mb.atypes)
        log_prob_directions = result.direction_distribution.log_prob(mb.directions)
        log_prob_spells = result.spell_distribution.log_prob(mb.spells)
        log_prob_use_items = result.use_item_distribution.log_prob(mb.use_items)
        log_prob_pick_items = result.pick_item_distribution.log_prob(mb.pick_items)

        action_class = mb.obs['action_class'].squeeze(-1)
        log_prob_actions = (
                log_prob_atypes * (action_class == ActionClasses.ATYPE)
                + log_prob_directions * (action_class == ActionClasses.DIRECTION)
                + log_prob_spells * (action_class == ActionClasses.SPELL)
                + log_prob_use_items * (action_class == ActionClasses.USE_ITEM)
                + log_prob_pick_items * (action_class == ActionClasses.PICK_ITEM)
        )
        mb.log_prob_actions = (
                mb.log_prob_atypes * (action_class == 0)
                + mb.log_prob_directions * (action_class == 1)
                + mb.log_prob_spells * (action_class == 2)
                + mb.log_prob_use_items * (action_class == 3)
                + mb.log_prob_pick_items * (action_class == 4)
        )
        return log_prob_actions

    def get_exploration_loss(self, exploration_loss_func, result, mb, exclude_last):
        action_class = mb.obs['action_class'].squeeze(-1)
        mask = dict(atype=action_class == ActionClasses.ATYPE,
                    direction=action_class == ActionClasses.DIRECTION,
                    spell=action_class == ActionClasses.SPELL,
                    use_item=action_class == ActionClasses.USE_ITEM,
                    pick_item=action_class == ActionClasses.PICK_ITEM)

        exploration_loss_atype = exploration_loss_func(result.atype_distribution, exclude_last, mask['atype'])
        exploration_loss_direction = exploration_loss_func(result.direction_distribution, exclude_last, mask['direction'])
        exploration_loss_spell = exploration_loss_func(result.spell_distribution, exclude_last, mask['spell'])
        exploration_loss_use_item = exploration_loss_func(result.spell_distribution, exclude_last, mask['use_item'])
        exploration_loss_pick_item = exploration_loss_func(result.spell_distribution, exclude_last, mask['pick_item'])
        exploration_loss = (exploration_loss_atype
                            + self.cfg.learner.relative_exploration_direction * exploration_loss_direction
                            + self.cfg.learner.relative_exploration_spell * exploration_loss_spell
                            + exploration_loss_use_item
                            + exploration_loss_pick_item)

        return exploration_loss

    def get_kl_divergence(self, result, mb):
        return None, -1 # SeparatedActionAgent does not use kl_divergence

    def update_stats(self, var, stats):
        action_class = var.mb.obs['action_class'].squeeze(-1)
        mask = dict(atype=action_class == ActionClasses.ATYPE,
                    direction=action_class == ActionClasses.DIRECTION,
                    spell=action_class == ActionClasses.SPELL,
                    use_item=action_class == ActionClasses.USE_ITEM,
                    pick_item=action_class == ActionClasses.PICK_ITEM)
        stats.entropy_atype = var.result.atype_distribution.entropy().mean().item()
        stats.entropy_direction = var.result.direction_distribution.entropy().mean().item()
        stats.entropy_spell = var.result.spell_distribution.entropy().mean().item()
        stats.entropy_use_item = var.result.use_item_distribution.entropy().mean().item()
        stats.entropy_pick_item = var.result.pick_item_distribution.entropy().mean().item()
        stats.entropy = (mask['atype'] * var.result.atype_distribution.entropy()
                         + mask['direction'] * var.result.direction_distribution.entropy()
                         + mask['spell'] * var.result.spell_distribution.entropy()
                         + mask['use_item'] * var.result.use_item_distribution.entropy()
                         + mask['pick_item'] * var.result.pick_item_distribution.entropy()).mean().item()