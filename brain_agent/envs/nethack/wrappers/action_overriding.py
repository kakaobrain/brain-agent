import gym

import enum

import numpy as np
import copy

import nle
from nle import nethack as nh
from brain_agent.envs.nethack.utils.number_to_binary import number_to_binary
from brain_agent.envs.nethack.ids.action_ids import ActionIds

import logging

from brain_agent.utils.utils import log
import random
from brain_agent.envs.nethack.wrappers.travel import travel, print_chars
from brain_agent.envs.nethack.ids.action_ids import KeyActionMapping
from brain_agent.envs.nethack.ids.armor import ArmorData
from nle.nethack import ACTIONS, Command, MiscAction, TextCharacters, MiscDirection

ENGRAVE_INDEX = ACTIONS.index(Command.ENGRAVE)
MINUS_INDEX = ACTIONS.index(TextCharacters.MINUS)
MORE_INDEX = ACTIONS.index(MiscAction.MORE)
TRAVEL_INDEX = ACTIONS.index(Command.TRAVEL)
WAIT_INDEX = ACTIONS.index(MiscDirection.WAIT)

COMMNAD_WEAR = 91
COMMAND_TAKEOFF = 80
COMMAND_MORE = 19
COMMAND_ESC = 36
ARMOR_CLASS = 3

ACT2KEY = {
    value:key for key, value in KeyActionMapping.items()
}


def decode_msg(encoded_msg):
    return encoded_msg.tostring().decode(errors='ignore').rstrip('\x00')


def get_message(obs):
    return decode_msg(obs['tty_chars'][0]).lower().strip()


def get_armor_type(desc, debug=False):
    type = None
    for key, item_dict in ArmorData.items():
        if key in desc:
            type = item_dict['Type']
            break
    if debug:
        assert type is not None, f"desc: {desc} key: {key}"
    return type


def pass_more(env, obs):
    message_txt = get_message(obs)
    more_message = '--more--' in message_txt
    if more_message:
        env.step(COMMAND_MORE)


class TakeOffOverrider:
    def __init__(self, cfg):
        self.cfg = cfg
        self.protocol = [COMMAND_ESC, COMMAND_TAKEOFF, [], COMMNAD_WEAR, []]
        self.prev_obs = None
        self.protocol_idx = -1
        self.situation_occured = False
        self.debug = False if self.cfg is None else self.cfg.env.debug_action_overrider

    def get_inv_state(self, obs=None):
        if obs is None:
            obs = self.prev_obs
        armor_ids = np.where(obs['inv_oclasses'] == ARMOR_CLASS)[0]
        armor_codes = [decode_msg(obs['inv_letters'][idx]) for idx in armor_ids]
        armor_txt = [decode_msg(obs['inv_strs'][idx]) for idx in armor_ids]
        armor_type = [get_armor_type(txt, debug=self.debug) for txt in armor_txt]
        mask_worn = ['being worn' in txt for txt in armor_txt]

        return armor_ids, armor_codes, armor_txt, armor_type, mask_worn

    def reset(self):
        self.protocol[2] = [] # what to take-off
        self.protocol[4] = [] # what to wear
        self.protocol_idx = -1
        self.prev_obs = None
        self.situation_occured = False

    def should_override(self, action):
        if action == COMMNAD_WEAR:
            self.protocol_idx = 0
            return False

        if self.protocol_idx == 0:
            if not action in ACT2KEY:
                return False

            prev_msg = get_message(self.prev_obs)
            if not prev_msg.startswith('what do you want to wear'):
                return False

            armor_ids, armor_codes, armor_txt, armor_type, mask_worn = self.get_inv_state()
            act_key = ACT2KEY[action]

            n_worn = sum(mask_worn)
            if n_worn == len(mask_worn):
                return False

            if not act_key in armor_codes:
                return False

            idx = armor_codes.index(act_key)
            type = armor_type[idx]

            # 착용하려는 아이템 type을 모를 때
            if type is None:
                return False

            # wear 시도하는 아이템을 현재 착용중일 때
            if mask_worn[idx]:
                return False

            ids_takeoff = []
            itemkeys_wear = [action]
            for i, is_worn in enumerate(mask_worn):
                if is_worn:
                    if armor_type[i] == type:
                        ids_takeoff.append(i)
                    # wear 하려는 아이템이 Suits type인데 현재 Cloaks type을 착용하고 있는경우
                    # Cloaks 벗기 -> Suits 입기 -> Cloaks 입기
                    elif type == 'Suits' and armor_type[i] == 'Cloaks':
                        if self.debug:
                            print('\tSituation Occured!')
                        self.situation_occured = True
                        ids_takeoff.append(i)
                        itemkeys_wear.append(KeyActionMapping[armor_codes[i]])

            if len(ids_takeoff) == 0:
                return False
            else:
                self.protocol[2] = [KeyActionMapping[armor_codes[_id]] for _id in ids_takeoff]
                self.protocol[4] = itemkeys_wear
                return True

        return False

    def override(self, env, action):
        obs, reward, done, info = env.step(self.protocol[0])

        if not done:
            obs, reward, done, info = env.step(self.protocol[1])
        msg = get_message(obs)
        while not done and len(self.protocol[2]) > 0 and msg.startswith('what do you want to take off'):
            obs, reward, done, info = env.step(self.protocol[2].pop(0))
            if len(self.protocol[2]) > 0:
                obs, reward, done, info = env.step(self.protocol[1])
                msg = get_message(obs)

        while not done and len(self.protocol[4]) > 0:
            obs, reward, done, info = env.step(self.protocol[3])
            if not done:
                obs, reward, done, info = env.step(self.protocol[4].pop(0))
            if self.debug and self.situation_occured:
                print(get_message(obs))
                print(self.get_inv_state(obs))

        return obs, reward, done, info

    def set_obs(self, obs):
        self.prev_obs = dict(
                inv_oclasses=obs['inv_oclasses'].copy(),
                inv_letters=obs['inv_letters'].copy(),
                inv_strs=obs['inv_strs'].copy(),
                tty_chars=obs['tty_chars'].copy(),
            )


class EngravingOverrider:
    def __init__(self):
        self.protocol = [37, 1, 6, 35, 67, 35, 83, 3, MORE_INDEX]  # Elbereth + more
        self.prev_obs = None
        self.protocol_idx = -1

    def reset(self):
        self.prev_obs = None
        self.protocol_idx = -1

    def should_override(self, action):
        if action == ENGRAVE_INDEX:
            self.protocol_idx = 0
            return True
        return False

    def override(self, env, action):
        obs, rew, done, info = env.step(ENGRAVE_INDEX)
        if not done:
            obs, rew, done, info = env.step(MINUS_INDEX)
            message_txt = get_message(obs)
            # print(message_txt)
            if 'do you want to add to the current engraving' in message_txt:
                if not done:
                    obs, rew, done, info = env.step(5)  # 'n'
                if not done:
                    obs, rew, done, info = env.step(MORE_INDEX)
            if not done:
                obs, rew, done, info = env.step(MORE_INDEX)

            if not done:
                message_txt = get_message(obs)
                # print(message_txt)
                if 'what do you want to write' in message_txt:
                    for idx in range(len(self.protocol)):
                        obs, rew, done, info = env.step(self.protocol[idx])
                        # print('\t', get_message(obs))
                        self.protocol_idx = idx
                        if done:
                            break

                    # message_txt = get_message(obs)
                    # print("after engraving:", message_txt)
                    # success = ('flee' in message_txt) or ('killed' in message_txt)

        return obs, rew, done, info

    def set_obs(self, obs):
        pass



class Traveler:
    def __init__(self):
        self.prev_obs = None

    def reset(self):
        pass

    def should_override(self, action):
        if action == TRAVEL_INDEX:
            return True
        return False

    def escape_location(self, obs=None):
        if obs is None:
            obs = self.prev_obs

        try:
            y_list, x_list = np.where((obs['chars'] == 35) | (obs['chars'] == 46))

            # 0 < ty < 20, 0 < tx < 78
            y_list_temp = []
            x_list_temp = []
            for ty, tx in zip(y_list, x_list):
                if (0 < ty < 20) and (0 < tx < 78):
                    y_list_temp.append(ty)
                    x_list_temp.append(tx)
            y_list = y_list_temp
            x_list = x_list_temp

            x = obs['blstats'][0]
            y = obs['blstats'][1]
            dist = abs(x_list-x) + abs(y_list-y)
            idx = dist.argmax()
            idx = random.choice(np.where(dist == dist[idx])[0])

            return y_list[idx], x_list[idx]
        except:
            return None, None

    def override(self, env, action):
        ty, tx = self.escape_location()
        if ty is None or tx is None:
            obs, rew, done, info = env.step(action)
            if not done:
                obs, rew, done, info = env.step(WAIT_INDEX)
        else:
            obs, rew, done, info = travel(env, ty, tx)
        self.set_obs(obs)
        return obs, rew, done, info

    def set_obs(self, obs):
        self.prev_obs = dict(
                chars=obs['chars'].copy(),
                blstats=obs['blstats'].copy()
            )


class ActionOverridingWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self.prev_message = None
        self.cur_overrider = None
        self.prev_obs = None
        self.overriders = [TakeOffOverrider(self.cfg), EngravingOverrider(), Traveler()]
        self.recent_rewards = list()
        self.step_count = 0
        self.score = 0
        self.is_submission = False

    def step(self, action):
        has_overridden = False
        done = False

        for overrider in self.overriders:
            if overrider.should_override(action):
                obs, reward, done, info = overrider.override(self.env, action)
                overrider.reset()
                has_overridden = True
                if done:
                    break

        if not done and not has_overridden:
            obs, reward, done, info = self.env.step(action)
            self.analyze(obs, enforce=done)

        THRESHOLD_STOP = 300
        if len(self.recent_rewards) >= THRESHOLD_STOP:
            self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)
        self.score += reward
        self.step_count += 1

        if (
            self.is_submission
            and len(self.recent_rewards) >= THRESHOLD_STOP
            and self.role in ['healer', 'ranger', 'rogue', 'wizard']
        ):
            recent_rewards = np.array(self.recent_rewards)
            if (
                not np.any(recent_rewards > 0)
                or self.score > 2000
                or self.step_count > 2000
            ): # no positive reward during 1000 steps
                print(
                    f"### [{self.role}] score: {self.score} step_count: {self.step_count} Meaningless time abusing happened!! ###"
                )
                while np.any(obs['misc']) and not done:
                    obs, reward, done, info = self.env.step(ActionIds.TextCharacters_SPACE)

                if not done:
                    obs, reward, done, info = self.env.step(ActionIds.Command_QUIT)
                    obs, reward, done, info = self.env.step(KeyActionMapping['y'])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        for overrider in self.overriders:
            overrider.reset()
        return obs

    def analyze(self, obs, enforce=False):
        for overrider in self.overriders:
            overrider.set_obs(obs)

        # print(self.overriders[0].get_inv_state())

    ## for submission
    def on_reset_submission(self, env, obs):
        self.role = None
        self.recent_rewards = list()
        self.step_count = 0
        self.score = 0
        self.env_aicrowd = env
        self.is_submission = True
        for overrider in self.overriders:
            overrider.reset()

        return obs

