from math import tanh

import gym
import torch
import numpy as np
from nle.env import NLE
from nle import nethack

import copy

from brain_agent.envs.nethack.ids.action_ids import ActionIds
from brain_agent.envs.nethack.ids.blstat_ids import BLStatIds
from brain_agent.envs.nethack.ids.obj_ids import ObjClasses

RAW_SCORE_SUMMARY_KEY_SUFFIX = 'nethack_raw_score'

TRAP_MSGS = ['You feel your magical energy drain away',
             'You feel momentarily lethargic',
             'You feel sluggish',
             'You feel very lethargic',
             'An arrow shoots out at you!',
             'You are caught in a bear trap',
             'A bear trap closes on your foot!'
             'A little dart shoots out at you!',
             'A tower of flame erupts from the floor!',
             'A cascade of steamy bubbles erupts from the pool!',
             "There's a gaping hole under you!",
             'A trapdoor in the ceiling opens and a rock falls on your head!',
             'A trapdoor in the sky opens and a rock falls on your head!',
             'Click! You trigger a rolling boulder trap!',
             'A gush of water hits you on the',
             'A gush of water hits you!',
             'A cloud of gas puts you to sleep!',
             'You are enveloped in a cloud of gas!']

class RewardShapingWrapper(gym.Wrapper):


    def __init__(self, env, reward_shaping, task_id, level, cfg=None):
        super().__init__(env)
        self.reward_shaping = reward_shaping
        self.raw_episode_return = self.episode_length = 0
        self.task_id = task_id
        self.level = level

        self.coeffs = __import__(
            f"brain_agent.envs.nethack.wrappers.reward_coeffs.{self.cfg.env.reward_shaping}",
            fromlist=-1).coeffs

        self.visited_depth = []

    def _get_hp_diff(self, action, obs):
        if self.coeffs.HP_DIFF == 0:
            return 0
        # good for increment
        hp_diff = obs['blstats'][BLStatIds.HP] - self.obs_prev['blstats'][BLStatIds.HP]
        return hp_diff

    def _get_ac_diff(self, action, obs):
        if self.coeffs.AC_DIFF == 0:
            return 0
        # bad for increment
        ac_diff = obs['blstats'][BLStatIds.AC] - self.obs_prev['blstats'][BLStatIds.AC]
        return ac_diff

    def _get_hunger_diff(self, action, obs):
        if self.coeffs.HUNGER_DIFF == 0:
            return 0
        # bad for increment
        hunger_diff = obs['blstats'][BLStatIds.HUNGER] - self.obs_prev['blstats'][BLStatIds.HUNGER]
        return hunger_diff

    def _get_cap_diff(self, action, obs):
        if self.coeffs.CAP_DIFF == 0:
            return 0
        # bad for increment
        cap_diff = obs['blstats'][BLStatIds.CAP] - self.obs_prev['blstats'][BLStatIds.CAP]
        return cap_diff

    def _get_exp_diff(self, aciton, obs):
        if self.coeffs.EXP_DIFF == 0:
            return 0
        # good for increment
        exp_diff = obs['blstats'][BLStatIds.EXP] - self.obs_prev['blstats'][BLStatIds.EXP]
        return exp_diff

    def _get_level_diff(self, action, obs):
        if self.coeffs.LEVEL_DIFF == 0:
            return 0
        # good for increment
        level_diff = obs['blstats'][BLStatIds.LV] - self.obs_prev['blstats'][BLStatIds.LV]
        return level_diff

    def _get_food_diff(self, action, obs):
        if self.coeffs.FOOD_DIFF == 0:
            return 0
        # good for increment but should not be penaltied for decrement
        def _get_num_food(obs):
            mask_foods = obs['inv_oclasses'] == ObjClasses.FOOD
            str_foods = obs['inv_strs'][mask_foods]
            num_food = 0
            for i in range(str_foods.shape[0]):
                str_food = str_foods[i].tobytes().decode('latin_1')
                if not 'corpse' in str_food:
                    if 'a ' in str_food or 'an ' in str_food:
                        num_food += 1
                    else:
                        num_food += int(str_food[:2])
            return num_food

        num_food = _get_num_food(obs)
        num_food_prev = _get_num_food(self.obs_prev)
        return num_food - num_food_prev

    def _get_gold_diff(self, action, obs):
        if self.coeffs.GOLD_DIFF == 0:
            return 0
        # good for increment
        gold_diff = obs['blstats'][BLStatIds.GOLD] - self.obs_prev['blstats'][BLStatIds.GOLD]
        if ActionIds.Command_PAY in self.prev_actions:
            gold_diff = 0
        return gold_diff

    def _get_item_diff(self, action, obs):
        if self.coeffs.ITEM_DIFF == 0:
            return 0
        num_items = np.float32(obs['inv_glyphs'] < 5976).sum()
        num_items_prev = np.float32(self.obs_prev['inv_glyphs'] < 5976).sum()
        item_diff = num_items - num_items_prev
        return item_diff

    def _get_pet_kill(self, action, obs):
        if self.coeffs.PET_KILL == 0:
            return 0
        _get_pet_kill = 'You kill the poor' in obs['message'].tobytes().decode('latin_1')
        return float(_get_pet_kill)

    def _get_avoid(self, action, obs):
        if self.coeffs.AVOID == 0:
            return 0
        avoid = 'there will be no return!' in obs['message'].tobytes().decode('latin_1')
        return float(avoid)

    def _get_reveal(self, action, obs):
        if self.coeffs.REVEAL == 0:
            return 0
        if obs['blstats'][BLStatIds.TIME] == 0:
            return 0
        dark_space = ord(' ') # 32
        reveal = np.float32(obs['chars'] == dark_space).sum() - np.float32(self.obs_prev['chars'] == dark_space).sum()
        reveal = -reveal
        if reveal < 0:
            reveal = 0
        # if reveal > 0:
        #     print(f'reveal: {reveal}')
        #     from brain_agent.envs.nethack.wrappers.travel import print_chars
        #     print_chars(obs['chars'])
        #     print_chars(self.obs_prev['chars'])
        return reveal

    def _get_depth_diff(self, action, obs):
        if self.coeffs.MAX_DEPTH == 0:
            return 0
        # good for increment
        depth_diff = np.float32(obs['blstats'][BLStatIds.DEPTH]) - np.float32(self.obs_prev['blstats'][BLStatIds.DEPTH])
        depth = np.float32(obs['blstats'][BLStatIds.DEPTH])
        for d in self.visited_depth:
            if depth == d:
                return 0
        self.visited_depth.append(depth)

        if obs['blstats'][BLStatIds.DEPTH] >= 20:
            depth_diff *= 1000
        else:
            depth_diff *= 50
        # if depth_diff < 0: # dont penalty go upstairs
        #     depth_diff = 0
        # if depth_diff > 0:
        #     print(f'depth_diff: {depth_diff}')
        return depth_diff

    def _get_in_sokovan(self, action, obs):
        if self.coeffs.IN_SOKOVAN == 0:
            return 0
        in_sokovan = obs['blstats'][BLStatIds.DN] == 2
        return in_sokovan

    def _get_exp_bonus(self, action, obs):
        if self.coeffs.EXP_DIFF == 0:
            return 0
        if 'role' in dir(self.unwrapped):
            role = self.unwrapped.role
            if role == 'rogue':
                return 1.5 * float(ActionIds.Command_THROW in self.prev_actions)
            elif role == 'tourist':
                return 1.5 * float(ActionIds.Command_THROW in self.prev_actions)
            elif role == 'ranger':
                return 1.5 * float(ActionIds.Command_FIRE in self.prev_actions)
            elif role == 'wizard':
                return 1.5 * float(ActionIds.Command_CAST in self.prev_actions)
            else:
                return 0
        else:
            return 0

    def _get_trapped(self, action, obs):
        if self.coeffs.TRAP == 0:
            return 0
        obs_msg = obs['message'].tobytes().decode('latin_1')
        for msg in TRAP_MSGS:
            if msg in obs_msg:
                return 1.0
        return 0

    def _get_shaped_reward(self, action, obs, done):
        reward = (
                self.coeffs.HP_DIFF * self._get_hp_diff(action, obs)
                - self.coeffs.AC_DIFF * self._get_ac_diff(action, obs)
                - self.coeffs.HUNGER_DIFF * self._get_hunger_diff(action, obs)
                - self.coeffs.CAP_DIFF * self._get_cap_diff(action, obs)
                + self.coeffs.EXP_DIFF * self._get_exp_bonus(action, obs) * self._get_exp_diff(action, obs)
                + self.coeffs.LEVEL_DIFF * self._get_level_diff(action, obs)
                + self.coeffs.FOOD_DIFF * self._get_food_diff(action, obs)
                + self.coeffs.ITEM_DIFF * self._get_item_diff(action, obs)
                - self.coeffs.GOLD_DIFF * self._get_gold_diff(action, obs) * float(ActionIds.Command_PAY in self.prev_actions) # compensate gold loss since paying is important action
                - self.coeffs.PET_KILL * self._get_pet_kill(action, obs)
                - self.coeffs.DONE * float(done)
                - self.coeffs.AVOID * self._get_avoid(action, obs)
                + self.coeffs.REVEAL * self._get_reveal(action, obs)
                + self.coeffs.MAX_DEPTH * self._get_depth_diff(action, obs)
                + self.coeffs.IN_SOKOVAN * self._get_in_sokovan(action, obs)
                - self.coeffs.TRAP * self._get_trapped(action, obs)
        )
        return reward

    def reset(self):
        obs = self.env.reset()
        self.raw_episode_return = self.episode_length = 0
        self.obs_prev = copy.deepcopy(obs)
        self.prev_actions = list()
        self.visited_depth = []
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        game_score = 0 if self.env.penalty_mode == 'constant' and self.env.penalty_step == rew else rew
        self.raw_episode_return += game_score

        self.prev_actions.append(action)
        if len(self.prev_actions) > 3:
            self.prev_actions.pop(0)

        raw_rew = rew
        shaped_reward = self._get_shaped_reward(action, obs, done)
        rew = rew + shaped_reward

        self.episode_length += info.get('num_frames', 1)

        self.obs_prev = copy.deepcopy(obs)
        if done:
            score = self.raw_episode_return
            if 'episodic_stats' not in info:
                info['episodic_stats'] = dict()
            level_name = self.level

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f'z_{self.task_id:02d}_{level_name}'
            info['episodic_stats'][f'{level_name_key}_raw_score'] = score
            info['episodic_stats'][f'{level_name_key}_len'] = self.episode_length
            if 'role' in dir(self.env.unwrapped):
                info['episodic_stats'][f'role_{self.env.unwrapped.role}/true_reward'] = self.raw_episode_return
                info['episodic_stats'][f'role_{self.env.unwrapped.role}/len'] = self.episode_length

        return obs, rew, done, info, raw_rew


class NetHackRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, task_id, level):
        super().__init__(env)
        self.raw_episode_return = self.episode_length = 0
        self.task_id = task_id
        self.level = level

    def reset(self):
        obs = self.env.reset()
        self.raw_episode_return = self.episode_length = 0
        self.obs_prev = copy.deepcopy(obs)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        game_score = 0 if self.env.penalty_mode == 'constant' and self.env.penalty_step == rew else rew
        self.raw_episode_return += game_score

        raw_rew = rew

        self.episode_length += info.get('num_frames', 1)

        self.obs_prev = copy.deepcopy(obs)
        if done:
            score = self.raw_episode_return
            if 'episodic_stats' not in info:
                info['episodic_stats'] = dict()
            info['episodic_stats']['level_name'] = self.level
            info['episodic_stats']['task_id'] = self.task_id
            level_name = self.level

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f'z_{self.task_id:02d}_{level_name}'
            info['episodic_stats']['true_reward'] = score
            info['episodic_stats'][f'{level_name_key}_true_reward'] = score
            info['episodic_stats'][f'{level_name_key}_len'] = self.episode_length
            if 'role' in dir(self.env.unwrapped):
                info['episodic_stats'][f'role_{self.env.unwrapped.role}/true_reward'] = self.raw_episode_return
                info['episodic_stats'][f'role_{self.env.unwrapped.role}/len'] = self.episode_length

        return obs, rew, done, info