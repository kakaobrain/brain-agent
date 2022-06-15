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


class RewardShapingWrapper(gym.Wrapper):
    C_FINAL = 0.01
    C_HP_DIFF = 0.01
    C_AC_DIFF = 20.0
    C_HUNGER_DIFF = 5.0
    C_CAP_DIFF = 5.0
    C_EXP_DIFF = 1.0
    C_LEVEL_DIFF = 10.0
    C_FOOD_DIFF = 2.5
    C_GOLD_DIFF = 1.0
    C_ITEM_DIFF = 0.0
    C_PET_KILL = 30
    C_DONE = 0
    C_AVOID = 100

    def __init__(self, env, reward_shaping, task_id, level, cfg=None):
        super().__init__(env)
        self.raw_episode_return = self.episode_length = 0
        self.task_id = task_id
        self.level = level

    def _get_hp_diff(self, action, obs):
        # good for increment
        hp_diff = obs['blstats'][BLStatIds.HP] - self.obs_prev['blstats'][BLStatIds.HP]
        return hp_diff

    def _get_ac_diff(self, action, obs):
        # bad for increment
        ac_diff = obs['blstats'][BLStatIds.AC] - self.obs_prev['blstats'][BLStatIds.AC]
        return ac_diff

    def _get_hunger_diff(self, action, obs):
        # bad for increment
        hunger_diff = obs['blstats'][BLStatIds.HUNGER] - self.obs_prev['blstats'][BLStatIds.HUNGER]
        return hunger_diff

    def _get_cap_diff(self, action, obs):
        # bad for increment
        cap_diff = obs['blstats'][BLStatIds.CAP] - self.obs_prev['blstats'][BLStatIds.CAP]
        return cap_diff

    def _get_exp_diff(self, aciton, obs):
        # good for increment
        exp_diff = obs['blstats'][BLStatIds.EXP] - self.obs_prev['blstats'][BLStatIds.EXP]
        return exp_diff

    def _get_level_diff(self, action, obs):
        # good for increment
        level_diff = obs['blstats'][BLStatIds.LV] - self.obs_prev['blstats'][BLStatIds.LV]
        return level_diff

    def _get_food_diff(self, action, obs):
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
        # good for increment
        gold_diff = obs['blstats'][BLStatIds.GOLD] - self.obs_prev['blstats'][BLStatIds.GOLD]
        if ActionIds.Command_PAY in self.prev_actions:
            gold_diff = 0
        return gold_diff

    def _get_item_diff(self, action, obs):
        num_items = np.float32(obs['inv_glyphs'] < 5976).sum()
        num_items_prev = np.float32(self.obs_prev['inv_glyphs'] < 5976).sum()
        item_diff = num_items - num_items_prev
        return item_diff

    def _get_pet_kill(self, action, obs):
        _get_pet_kill = 'You kill the poor' in obs['message'].tobytes().decode('latin_1')
        return float(_get_pet_kill)

    def _get_avoid(self, action, obs):
        avoid = 'there will be no return!' in obs['message'].tobytes().decode('latin_1')
        return float(avoid)

    def _get_shaped_reward(self, action, obs, done):
        # reward = self.C_FINAL * (
        #     self.C_HP_DIFF * self._get_hp_diff(action, obs)
        #     - self.C_AC_DIFF * self._get_ac_diff(action, obs)
        #     - self.C_HUNGER_DIFF * self._get_hunger_diff(action, obs)
        #     - self.C_CAP_DIFF * self._get_cap_diff(action, obs)
        #     + self.C_EXP_DIFF * self._get_exp_diff(action, obs)
        #     + self.C_LEVEL_DIFF * self._get_level_diff(action, obs)
        #     + self.C_FOOD_DIFF * self._get_food_diff(action, obs)
        #     + self.C_GOLD_DIFF * self._get_gold_diff(action, obs)
        #     - self.C_DONE * float(done)
        # )
        not_done = float(not done)
        reward = (
                self.C_HP_DIFF * self._get_hp_diff(action, obs) * not_done
                - self.C_AC_DIFF * self._get_ac_diff(action, obs) * not_done
                - self.C_HUNGER_DIFF * self._get_hunger_diff(action, obs) * not_done
                - self.C_CAP_DIFF * self._get_cap_diff(action, obs) * not_done
                + self.C_EXP_DIFF * self._get_exp_diff(action, obs) * not_done
                + self.C_LEVEL_DIFF * self._get_level_diff(action, obs) * not_done
                + self.C_FOOD_DIFF * self._get_food_diff(action, obs) * not_done
                + self.C_ITEM_DIFF * self._get_item_diff(action, obs) * not_done
                - self.C_GOLD_DIFF * self._get_gold_diff(action, obs) * float(ActionIds.Command_PAY in self.prev_actions) * not_done # compensate gold loss since paying is important action
                #- self.C_PET_KILL * self._get_pet_kill(action, obs) * not_done
                #- self.C_DONE * float(done)
                #- self.C_AVOID * self._get_avoid(action, obs)
        )
        return reward

    def reset(self):
        obs = self.env.reset()
        self.raw_episode_return = self.episode_length = 0
        self.obs_prev = copy.deepcopy(obs)
        self.prev_actions = list()
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

        # # reward clipping
        #rew = np.clip(rew, 0, rew + 1)

        self.episode_length += info.get('num_frames', 1)

        self.obs_prev = copy.deepcopy(obs)
        if done:
            score = self.raw_episode_return
            if 'episode_extra_stats' not in info:
                info['episode_extra_stats'] = dict()
            level_name = self.level

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f'z_{self.task_id:02d}_{level_name}'
            info['episode_extra_stats'][f'{level_name_key}_{RAW_SCORE_SUMMARY_KEY_SUFFIX}'] = score
            info['episode_extra_stats'][f'{level_name_key}_len'] = self.episode_length
            if 'role' in dir(self.env.unwrapped):
                info['episode_extra_stats'][f'{self.env.unwrapped.role}/true_reward'] = self.raw_episode_return
                info['episode_extra_stats'][f'{self.env.unwrapped.role}/len'] = self.episode_length

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

        # # reward clipping
        #rew = np.clip(rew, 0, rew + 1)

        self.episode_length += info.get('num_frames', 1)

        self.obs_prev = copy.deepcopy(obs)
        if done:
            score = self.raw_episode_return
            if 'episode_extra_stats' not in info:
                info['episode_extra_stats'] = dict()
            level_name = self.level

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f'z_{self.task_id:02d}_{level_name}'
            info['episode_extra_stats'][f'{level_name_key}_{RAW_SCORE_SUMMARY_KEY_SUFFIX}'] = score
            info['episode_extra_stats'][f'{level_name_key}_len'] = self.episode_length
            if 'role' in dir(self.env.unwrapped):
                info['episode_extra_stats'][f'role_{self.env.unwrapped.role}/true_reward'] = self.raw_episode_return
                info['episode_extra_stats'][f'role_{self.env.unwrapped.role}/len'] = self.episode_length

        return obs, rew, done, info