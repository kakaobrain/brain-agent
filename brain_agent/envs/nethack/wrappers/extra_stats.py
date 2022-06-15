import copy
from math import tanh

import gym
import torch
import numpy as np

from brain_agent.envs.nethack.ids.blstat_ids import BLStatIds
from brain_agent.envs.nethack.ids.action_ids import ActionIds
from brain_agent.envs.nethack.ids.obj_ids import ObjClasses
import nle.nethack as nh


class ExtraStatsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def _initialize_stats(self):
        self.stats = dict(
            level=0,
            avg_hp=0,
            avg_hp_ratio=0,
            ac_diff=0,
            hunger=0,
            gold_diff=0,
            capacity=0,
            dungeon_depth=0,
            dungeon_level=0,
            time=0,
            pet_kill=0,

            eat_try=0,
            spell_try=0,
            quaff_try=0,
            read_try=0,
            apply_try=0,
            pickup_try=0,
            fire_try=0,
            throw_try=0,
            wear_try=0,
            takeoff_try=0,
            engrave_try=0,
            drop_try=0,

            food_gain=0,
            armor_gain=0,
            weapon_gain=0,
            item_gain=0,

            visited_noum=0,
        )

    def _on_reset(self, obs):
        self._initialize_stats()
        self.blstats_init = obs['blstats'].copy()
        self.misc_prev = obs['misc'].copy()
        self.inv_oclasses_prev = obs['inv_oclasses'].copy()
        self.step_count = 0

    def _update_stats(self, action, obs):
        blstats = obs['blstats']
        self.step_count += 1

        self.stats['level'] = blstats[BLStatIds.LV]
        self.stats['avg_hp'] = (
            (self.step_count - 1) * self.stats['avg_hp'] + blstats[BLStatIds.HP]
        ) / self.step_count
        self.stats['avg_hp_ratio'] = (
            (self.step_count - 1) * self.stats['avg_hp_ratio']
            + float(blstats[BLStatIds.HP]) / float(blstats[BLStatIds.MAX_HP] + 1e-5)
        ) / self.step_count
        self.stats['ac_diff'] = blstats[BLStatIds.AC] - self.blstats_init[BLStatIds.AC]
        self.stats['hunger'] = blstats[BLStatIds.HUNGER]
        self.stats['gold_diff'] = blstats[BLStatIds.GOLD] - self.blstats_init[BLStatIds.GOLD]
        self.stats['capacity'] = blstats[BLStatIds.CAP]
        self.stats['dungeon_depth'] = blstats[BLStatIds.DEPTH]
        self.stats['dungeon_level'] = blstats[BLStatIds.LN]
        self.stats['time'] = blstats[BLStatIds.TIME]
        self.stats['max_hp'] = blstats[BLStatIds.MAX_HP]
        self.stats['pet_kill'] += float('You kill the poor' in obs['message'].tobytes().decode('latin_1')) * (1 - self.stats['pet_kill'])

        # Action statistics logging
        if not np.any(self.misc_prev):
            self.stats['eat_try'] += float(action == ActionIds.Command_EAT)
            self.stats['spell_try'] += float(action == ActionIds.Command_CAST)
            self.stats['quaff_try'] += float(action == ActionIds.Command_QUAFF)
            self.stats['read_try'] += float(action == ActionIds.Command_READ)
            self.stats['apply_try'] += float(action == ActionIds.Command_APPLY)
            self.stats['pickup_try'] += float(action == ActionIds.Command_PICKUP)
            self.stats['fire_try'] += float(action == ActionIds.Command_FIRE)
            self.stats['throw_try'] += float(action == ActionIds.Command_THROW)
            self.stats['wear_try'] += float(action == ActionIds.Command_WEAR)
            self.stats['takeoff_try'] += float(action == ActionIds.Command_TAKEOFF)
            self.stats['engrave_try'] += float(action == ActionIds.Command_ENGRAVE)
            self.stats['drop_try'] += float(action == ActionIds.Command_DROP)

        self.stats['food_gain'] += max(
            (obs['inv_oclasses']==ObjClasses.FOOD).sum() - (self.inv_oclasses_prev==ObjClasses.FOOD).sum(),
            0
        )
        self.stats['armor_gain'] += max(
            (obs['inv_oclasses']==ObjClasses.ARMOR).sum() - (self.inv_oclasses_prev==ObjClasses.ARMOR).sum(),
            0
        )
        self.stats['weapon_gain'] += max(
            (obs['inv_oclasses'] == ObjClasses.WEAPON).sum() - (self.inv_oclasses_prev == ObjClasses.WEAPON).sum(),
            0
        )
        self.stats['item_gain'] += max(
            (obs['inv_oclasses'] < ObjClasses.MAXOCLASSES).sum() - (self.inv_oclasses_prev < ObjClasses.MAXOCLASSES).sum(),
            0
        )
        DN_NOUM = 2
        self.stats['visited_noum'] = float(self.stats['visited_noum'] or obs['blstats'][BLStatIds.DN] == DN_NOUM)


    def _on_step(self, action, obs, rew, done, info):
        if not done:
            self._update_stats(action, obs)
            self.misc_prev = obs['misc'].copy()
            self.inv_oclasses_prev = obs['inv_oclasses'].copy()
        else:
            for key in list(self.stats.keys()):
                if 'role' in dir(self.env.unwrapped):
                    self.stats[f'role_{self.env.unwrapped.role}/extra_stat_{key}'] = self.stats.pop(key)
                else:
                    self.stats[f'extra_stat_{key}'] = self.stats.pop(key)
            if 'episodic_stats' in info:
                info['episodic_stats'].update(self.stats)
            else:
                info['episodic_stats'] = self.stats

    def reset(self):
        obs = self.env.reset()
        self._on_reset(obs)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._on_step(action, obs, rew, done, info)
        return obs, rew, done, info