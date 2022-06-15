import os
import copy

import gym
import logging
import enum

import numpy as np

import nle
from nle import nethack as nh
from brain_agent.envs.nethack.ids.action_ids import ActionIds
from brain_agent.envs.nethack.ids.misc_ids import MISCIds
from brain_agent.envs.nethack.utils.number_to_binary import number_to_binary
from brain_agent.envs.nethack.utils.glyph_to_feature import glyph_to_feature_table

from brain_agent.envs.nethack.ids.armor import ArmorData
from brain_agent.envs.nethack.ids.weapon import WeaponData
from brain_agent.envs.nethack.ids.obj_ids import INDEXED_ARMOR_DATA, INDEXED_WEAPON_DATA

class ItemFeatureWrapper(gym.Wrapper):
    SCALE_COUNT = 0.1

    def __init__(self, env, cfg):
        super().__init__(env)

        obs_spaces = dict(self.observation_space.spaces)

        self.cfg = cfg
        self.use_item_data = self.cfg.env.use_item_data
        self.item_data_extra_dim = self.cfg.env.item_data_extra_dim
        if self.use_item_data:
            obs_spaces['item_feature'] = gym.spaces.Box(
                low=-5,
                high=5,
                shape=(55, 32+self.item_data_extra_dim),
                dtype=np.float32
            )
        else:
            obs_spaces['item_feature'] = gym.spaces.Box(
                low=-5,
                high=5,
                shape=(55, 32),
                dtype=np.float32
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.glyph_to_feature_table = glyph_to_feature_table()


    def _create_item_feature(self, obs):
        n_item = obs['inv_glyphs'].shape[0]
        item_feature_glyph = self.glyph_to_feature_table[obs['inv_glyphs']][:, 13:]  # (55, 28)
        #item_feature_binary = -np.ones([n_item, 9], dtype=np.int8) # (55, 9)
        item_feature_cursed = np.zeros([n_item, 1], dtype=np.float32)  # (55, 1)
        item_feature_worn = np.zeros([n_item, 1], dtype = np.float32) # (55, 1)
        item_feature_enchant = np.zeros([n_item, 1], dtype = np.float32) # (55, 1)
        item_feature_count = np.zeros([n_item, 1], dtype=np.float32)  # (55, 1), this is for comestible (food)

        if self.use_item_data:
            item_feature_data = np.zeros([n_item, self.item_data_extra_dim], dtype=np.float32)

        for idx_item in range(n_item):
            item_glyph = obs['inv_glyphs'][idx_item]
            if item_glyph == nh.MAX_GLYPH: continue

            #item_feature_binary[idx_item] = number_to_binary(item_id, 9)

            obj_class_item = obs['inv_oclasses'][idx_item]
            str_item = obs['inv_strs'][idx_item].tostring().decode('latin_1').lower()
            # cursed/uncursed feature generation
            if 'uncursed' in str_item: cursed = -1
            elif 'cursed' in str_item:  cursed = 1
            else:                      cursed = 0
            item_feature_cursed[idx_item, 0] = cursed

            if self.use_item_data:
                item_feature_armor = np.zeros([10])
                item_feature_weapon = np.zeros([33])
                if obj_class_item == 3:  # armor
                    # print(f'item_glyph: {item_glyph}, str: {str_item}')
                    # print(f'indexed_data: {INDEXED_ARMOR_DATA[item_glyph].name}')
                    try:
                        item_feature_armor = INDEXED_ARMOR_DATA[item_glyph].feature.copy()
                    except:
                        f = open('../data/item_debug/item.txt', 'a')
                        f.write(f'item_glyph: {item_glyph}, str: {str_item}')
                        f.close()
                if obj_class_item == 2:  # weapon
                    # print(f'item_glyph: {item_glyph}, str: {str_item}')
                    # print(f'indexed_data: {INDEXED_WEAPON_DATA[item_glyph].name}')
                    try:
                        item_feature_weapon = INDEXED_WEAPON_DATA[item_glyph].feature.copy()
                    except:
                        f = open('../data/item_debug/item.txt', 'a')
                        f.write(f'item_glyph: {item_glyph}, str: {str_item}')
                        f.close()
                item_feature_data[idx_item, :] = np.concatenate([item_feature_armor, item_feature_weapon])
                # if obj_class_item == 2:
                #     print(item_feature_data[idx_item, :])

            # worn feature generation
            if 'worn' in str_item: worn = 1
            else:                  worn = 0
            item_feature_worn[idx_item, 0] = worn

            # enchant feature generation
            if '+' in str_item:
                try:
                    enchant = int(str_item[str_item.index('+') + 1])
                except ValueError:
                    # logging.exception(f'str_item={str_item}')
                    enchant = 0
            else:
                enchant = 0
            item_feature_enchant[idx_item, 0] = enchant

            # count feature generation for comestible (food)
            if obj_class_item==7:
                if str_item[0] == 'a': count = self.SCALE_COUNT
                else:                  count = self.SCALE_COUNT * int( str_item[:2] )
                item_feature_count[idx_item, 0] = count

        if self.use_item_data:
            item_feature = np.concatenate([
                #item_feature_binary,
                item_feature_glyph,
                item_feature_cursed,
                item_feature_worn,
                item_feature_enchant,
                item_feature_count,
                item_feature_data
            ], axis=-1)  # (55, 45)
        else:
            item_feature = np.concatenate([
                #item_feature_binary,
                item_feature_glyph,
                item_feature_cursed,
                item_feature_worn,
                item_feature_enchant,
                item_feature_count
            ], axis=-1)  # (55, 45)

        obs['item_feature'] = item_feature

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._create_item_feature(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._create_item_feature(obs)
        return obs

    ## for submission below here
    def on_reset_submission(self, env, obs, is_holding_aicrowd=False):
        self.is_holding_aicrowd = is_holding_aicrowd
        self.env_aicrowd = env
        #self.step = lambda action: self.step_submission(action, self.env_aicrowd)
        obs = self.env.on_reset_submission(env, obs)

        self._create_item_feature(obs)

        return obs