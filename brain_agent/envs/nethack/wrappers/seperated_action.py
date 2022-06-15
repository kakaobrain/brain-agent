import copy
from math import tanh

import gym
import torch
import numpy as np

from brain_agent.envs.nethack.ids.action_ids import ActionIds, KeyActionMapping
from brain_agent.envs.nethack.ids.action_classes import ActionClasses
from brain_agent.envs.nethack.ids.obj_ids import ObjClasses, ObjNames
from brain_agent.envs.nethack.utils.get_object import get_object
from brain_agent.envs.nethack.utils.number_to_binary import number_to_binary

from brain_agent.envs.nethack.wrappers.action_masking import ActionMaskingWrapper
from brain_agent.envs.nethack.wrappers.spell_feature import SpellFeatureWrapper

from nle import nethack as nh
from brain_agent.envs.nethack.ids.obj_ids import INDEXED_ARMOR_DATA, INDEXED_WEAPON_DATA


class SeperatedActionWrapper(ActionMaskingWrapper):
    '''
    This wrapper requires env wrapped by ObsDeepCopyWrapper!!!
    '''

    NUM_PICK_ITEM = 10
    def __init__(self, env, cfg):
        super().__init__(env)

        # update observation space
        self.cfg = cfg
        self.item_data_extra_dim = self.cfg.env.item_data_extra_dim if self.cfg.env.use_item_data else 0
        obs_spaces = {
            'action_class': gym.spaces.Box(
                low=0,
                high=4,
                shape=(1,),
                dtype=np.uint8
            ),
            'avail_spell': gym.spaces.Box(
                low=0,
                high=1,
                shape=(5,),
                dtype=np.uint8
            ),
            'avail_use_item':gym.spaces.Box(
                low=0,
                high=1,
                shape=(55,),
                dtype=np.uint8
            ),
            'avail_pick_item': gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.NUM_PICK_ITEM, ), # Maximally 20 items are available for picking
                dtype=np.int8
            ),
            'pick_item_feature':gym.spaces.Box(
                low=-1,
                high=1,
                shape=(self.NUM_PICK_ITEM, 30 + self.item_data_extra_dim), # ObjClass (18) + ObjId(9) + num(1) + cursed(1) + enchant(1)
                dtype=np.float32
            ),
            'last_atype':gym.spaces.Box(
                low=0,
                high=1,
                shape=(113,), #
                dtype=np.uint8
            ),
        }
        obs_spaces.update([
            (k, self.env.observation_space[k]) for k in self.env.observation_space
        ])
        self.observation_space = gym.spaces.Dict(obs_spaces)

        # update action space
        action_space = {
            'atype': gym.spaces.discrete.Discrete(113),
            'direction': gym.spaces.discrete.Discrete(8),
            'spell': gym.spaces.discrete.Discrete(5),
            'use_item': gym.spaces.discrete.Discrete(55),
            'pick_item': gym.spaces.discrete.Discrete(self.NUM_PICK_ITEM),
        }

        self.action_space = gym.spaces.Dict(action_space)
        self.current_action_class = ActionClasses.ATYPE # checks which class of action is required in current state
        self.obs = None # just used for referencing observation, it won't be modified in this code

    def _get_required_action_class(self, obs):
        '''
        Recognize which action class is required in current state.
        For example, ActionClasses.DIRECTION is required after you choose fire since it requires direction.
        ActionClasses.USE_ITEM is required after you choose to eat, wear, wield, etc which are actions that require a choice of item in your inventory.
        ActionClasses.PICK_ITEM is required when you choose among multiple items.
        '''
        tty_chars = obs['tty_chars'].copy()

        # multiple item pick check
        str_tty_chars = obs['tty_chars'].tostring().decode('latin_1')

        is_direction = 'In what direction?' in str_tty_chars
        if is_direction: return ActionClasses.DIRECTION

        is_spell = 'Choose which spell to cast' in str_tty_chars
        if is_spell: return ActionClasses.SPELL

        is_use_item = 'What do you want to' in str_tty_chars
        if is_use_item: return ActionClasses.USE_ITEM

        is_pick_item = 'Pick up what?' in str_tty_chars
        if is_pick_item: return ActionClasses.PICK_ITEM

        return ActionClasses.ATYPE

    def _get_real_action_from_direction(self, obs, direction):
        ''' Real Actions of directions are 0 ~ 7 which are same as the values of directions.
        CompassDirection_N = 0,
        CompassDirection_E = 1,
        CompassDirection_S = 2,
        CompassDirection_W = 3,
        CompassDirection_NE = 4,
        CompassDirection_SE = 5,
        CompassDirection_SW = 6,
        CompassDirection_NW = 7,
        '''
        assert direction < 8, f"direction must be an integer [0, 7] but got {direction}"
        real_action = direction
        return real_action

    def _get_real_action_from_spell(self, obs, spell):
        assert spell < 5, f"spell must be an integer [0, 4] but got {spell}"

        if spell == 0: # 'a'
            real_action = KeyActionMapping['a']
        elif spell == 1: # 'b'
            real_action = KeyActionMapping['b']
        elif spell == 2: # 'c'
            real_action = KeyActionMapping['c']
        elif spell == 3: # 'd'
            real_action = KeyActionMapping['d']
        elif spell == 4: # 'e'
            real_action = KeyActionMapping['e']

        return real_action

    def _get_real_action_from_use_item(self, obs, use_item):
        char_assigned_to_item = obs['inv_letters'][use_item].tobytes().decode('latin_1')
        if char_assigned_to_item == '\x00':
            real_action = ActionIds.TextCharacters_SPACE
        else:
            real_action = KeyActionMapping[char_assigned_to_item]

        return real_action

    def _get_real_action_from_pick_item(self, obs, pick_item):
        tty_chars = obs['tty_chars'].copy()

        pick_item_keys = list()
        start_item_check = False
        for idx_line in range(21):
            str_item = tty_chars[idx_line].tostring().decode('latin_1')

            # is_end_of_line = str_spell[20:25] == '(end)'
            is_end_of_line = '(end)' in str_item
            if is_end_of_line:
                break

            if not start_item_check:
                start_item_check = 'Pick up what?' in str_item  # start checking spell after we see a row of spell
                if start_item_check:
                    idx_column_item_key = str_item.rindex('Pick up what?')
                continue

            key_item = str_item[idx_column_item_key] # a - uncurse ..., b - an apple, ...
            must_be_space_for_key_item = str_item[idx_column_item_key + 1].isspace() # there are words like Weapons, Armors
            if key_item.isalpha() and must_be_space_for_key_item:
                pick_item_keys.append(str_item[idx_column_item_key])

        if pick_item < len(pick_item_keys):
            char_pick_item = pick_item_keys[pick_item]
            real_action = KeyActionMapping[char_pick_item]
            return real_action
        else:
            char_pick_item = pick_item_keys[0]
            real_action = KeyActionMapping[char_pick_item]
            return real_action

    def _convert_to_real_action(self, required_action_class, action):
        real_action = None
        if required_action_class == ActionClasses.ATYPE:
            real_action = action['atype']
        elif required_action_class == ActionClasses.DIRECTION:
            real_action = self._get_real_action_from_direction(self.obs, action['direction'])
        elif required_action_class == ActionClasses.SPELL:
            real_action = self._get_real_action_from_spell(self.obs, action['spell'])
        elif required_action_class == ActionClasses.USE_ITEM:
            real_action = self._get_real_action_from_use_item(self.obs, action['use_item'])
        elif required_action_class == ActionClasses.PICK_ITEM:
            real_action = self._get_real_action_from_pick_item(self.obs, action['pick_item'])

        assert not real_action is None, "real action shold not be None"

        return real_action

    def _get_extra_avail_atype(self, obs, required_action_class):
        # get more squeezed avail_atype from super class
        avail_atype = obs['avail_atype'] if 'avail_atype' in obs else None
        if (
            not avail_atype is None
            and required_action_class == ActionClasses.ATYPE
            and obs['misc'][2]
        ):
            avail_atype = np.zeros(avail_atype.shape, dtype=np.uint8)
            avail_atype[ActionIds.TextCharacters_SPACE] = 1 # Turn on space only when it is the only available action
        elif (
            not avail_atype is None
            and required_action_class == ActionClasses.ATYPE
            and 'Right or Left' in obs['message'].tobytes().decode('latin_1')
        ): # handling Right/Left choice
            avail_atype = np.zeros(avail_atype.shape, dtype=np.uint8)
            avail_atype[KeyActionMapping['r']] = 1
            avail_atype[KeyActionMapping['l']] = 1
        elif (
            not avail_atype is None
            and required_action_class == ActionClasses.ATYPE
            and obs['misc'][0]
        ): # handling y/n choice, always say yes
            avail_atype = np.zeros(avail_atype.shape, dtype=np.uint8)
            avail_atype[KeyActionMapping['y']] = 1 # Always say yes to question, if it wants say 'no', it shouldn't have tried that action


        return avail_atype

    def _get_avail_spell(self, obs):
        if 'spell_feature' in obs:
            retentioning_spell = obs['spell_feature'][:, SpellFeatureWrapper.OBS_OFFSET_RETENTION] # all available spells have retention > 0
            avail_spell = np.uint8(retentioning_spell > 0)
        else:
            avail_spell = np.ones([5], dtype=np.uint8)
        return avail_spell

    def _get_avail_use_item(self, obs, prev_action):
        # It returns available item based on given action.
        avail_use_item = obs['inv_glyphs'] < nh.MAX_GLYPH
        if prev_action is None:
            return np.uint8(avail_use_item)

        num_items = obs['inv_strs'].shape[0]
        is_not_worn = np.array([
            not 'worn' in obs['inv_strs'][idx].tobytes().decode('latin_1') for idx in range(num_items)
        ])
        prev_atype = prev_action['atype']

        if prev_atype == ActionIds.Command_EAT: # food
            avail_use_item = obs['inv_oclasses'] == ObjClasses.FOOD
        elif prev_atype == ActionIds.Command_QUAFF: # potion
            avail_use_item = obs['inv_oclasses'] == ObjClasses.POTION
        elif prev_atype == ActionIds.Command_WEAR: # armor
            is_armor = obs['inv_oclasses'] == ObjClasses.ARMOR
            avail_use_item = np.logical_and(is_armor, is_not_worn)
        elif prev_atype == ActionIds.Command_TAKEOFF: # armor
            is_armor = obs['inv_oclasses'] == ObjClasses.ARMOR
            avail_use_item = np.logical_and(is_armor, np.logical_not(is_not_worn))
        elif prev_atype == ActionIds.Command_PUTON: # accessory
            is_ring = obs['inv_oclasses'] == ObjClasses.RING
            is_amulet = obs['inv_oclasses'] == ObjClasses.AMULET
            is_accessory = np.logical_or(is_ring, is_amulet)
            avail_use_item = np.logical_and(is_accessory, is_not_worn)
        elif prev_atype == ActionIds.Command_WIELD: # wield
            avail_use_item = obs['inv_oclasses'] == ObjClasses.WEAPON
        elif prev_atype == ActionIds.Command_APPLY: # tools
            avail_use_item = obs['inv_oclasses'] == ObjClasses.TOOL
        elif prev_atype == ActionIds.Command_READ: # scroll, spbook
            is_scroll = obs['inv_oclasses'] == ObjClasses.SCROLL
            is_spbook = obs['inv_oclasses'] == ObjClasses.SPBOOK
            avail_use_item = np.logical_or(is_scroll, is_spbook)
        elif prev_atype == ActionIds.Command_ZAP: # wand
            avail_use_item = obs['inv_oclasses'] == ObjClasses.WAND

        return np.uint8(avail_use_item)

    def _extract_item_name(self, full_name_item):
        name_item = full_name_item.replace(
            'uncursed', '').replace(
            'cursed', '').replace(
            'blessed', '').replace(
            'rusted', '').replace(
            'corroded', '').replace(
            'rotten', '').replace(
            'partialy', '').replace(
            'eaten', '').replace(
            'a ', '').replace(
            'an ', '').replace(
            '+', '').replace(
            '(', '').replace(
            ')', '').replace(
            ':', '').strip(' ')
        for i in range(10): name_item = name_item.replace(f'{i}', '')
        name_item = name_item.strip(' ')
        return name_item

    def _fill_pick_item_feature(self, pick_item_feature, idx_pick_item, full_name_item):
        # This function fills the pick_item feature (n_pick_item, d_pick_item) which is initially initialized as -np.ones
        # name_item = self._extract_item_name(full_name_item)
        extracted_name = None
        for _name in ObjNames:
            if _name in full_name_item:
                extracted_name = _name
                break

        if extracted_name is None:
            pick_item_feature[idx_pick_item, :] = 0  # set 0 for unknown items
        else:
            obj, obj_idx = get_object(_name, return_index=True)
            obj_class_feature = np.zeros([18], np.int8)
            obj_class_feature[ord(obj.oc_class)] = 1
            obj_idx_feature = number_to_binary(obj.oc_name_idx, 9)
            num_item = np.array([
                int(full_name_item[:2]) if full_name_item[:2].isdecimal()
                else 1
            ], dtype=np.int8)

            if 'uncursed' in full_name_item:
                cursed = -1
            elif 'cursed' in full_name_item:
                cursed = 1
            else:
                cursed = 0
            cursed = np.array([cursed], dtype=np.int8)

            if '+' in full_name_item:
                enchant = int(full_name_item[full_name_item.index('+') + 1])
            else:
                enchant = 0
            enchant = np.array([enchant], dtype=np.int8)

            if self.cfg.env.use_item_data:
                armor = np.zeros([10])
                weapon = np.zeros([33])
                if ord(obj.oc_class) == 3:  # armor
                    # print(f'name: {extracted_name}, cidx: {ord(obj.oc_class)}')
                    armor = INDEXED_ARMOR_DATA[obj_idx+nh.GLYPH_OBJ_OFF].feature.copy()
                elif ord(obj.oc_class) == 2:  # armor
                    # print(f'name: {extracted_name}, cidx: {ord(obj.oc_class)}')
                    weapon = INDEXED_WEAPON_DATA[obj_idx+nh.GLYPH_OBJ_OFF].feature.copy()

                pick_item_feature[idx_pick_item, :] = np.concatenate([
                    obj_class_feature,
                    obj_idx_feature,
                    0.1 * num_item,
                    cursed,
                    enchant,
                    armor,
                    weapon
                ], axis=-1)

            else:
                pick_item_feature[idx_pick_item, :] = np.concatenate([
                    obj_class_feature,
                    obj_idx_feature,
                    0.1 * num_item,
                    cursed,
                    enchant
                ], axis=-1)

    def _get_pick_item_feature_and_avail(self, obs):
        tty_chars = obs['tty_chars'].copy()

        pick_item_feature = -np.ones([self.NUM_PICK_ITEM, 30+self.item_data_extra_dim], dtype=np.int8)
        avail_pick_item = np.ones([self.NUM_PICK_ITEM], dtype=np.uint8)

        need_encoding1 = obs['action_class'] == ActionClasses.PICK_ITEM # 'Pick up what?' in screen
        need_encoding2 = 'Things that are here' in obs['tty_chars'].tobytes().decode('latin_1')
        if not (need_encoding1 or need_encoding2):
            return pick_item_feature, avail_pick_item

        idx_pick_item = 0
        start_item_check = False
        for idx_line in range(21):
            str_item = tty_chars[idx_line].tostring().decode('latin_1')

            if need_encoding1: # make pick item feature when it is time to pick
                is_end_of_line = '(end)' in str_item
                if is_end_of_line:
                    break

                if not start_item_check:
                    start_item_check = 'Pick up what?' in str_item  # start checking spell after we see a row of spell
                    if start_item_check:
                        idx_column_item_key = str_item.rindex('Pick up what?')
                        idx_column_item_name = idx_column_item_key + 4
                    continue

                key_item = str_item[idx_column_item_key]  # a - uncurse ..., b - an apple, ...
                must_be_space_for_key_item = str_item[idx_column_item_key + 1].isspace()  # there are words like Weapons, Armors
                if key_item.isalpha() and must_be_space_for_key_item:
                    full_name_item = str_item[idx_column_item_name:]
                    self._fill_pick_item_feature(pick_item_feature, idx_pick_item, full_name_item)
                    idx_pick_item += 1

            elif need_encoding2: # make pick item feature when the agent see items below itself
                is_end_of_line = '--More--' in str_item
                if is_end_of_line:
                    break

                if not start_item_check:
                    start_item_check = 'Things that are here' in str_item  # start checking spell after we see a row of spell
                    if start_item_check:
                        idx_column_item_name = str_item.rindex('Things that are here')
                    continue

                full_name_item = str_item[idx_column_item_name:]
                self._fill_pick_item_feature(pick_item_feature, idx_pick_item, full_name_item)
                idx_pick_item += 1

            if idx_pick_item >= self.NUM_PICK_ITEM:
                break

        avail_pick_item = np.int8(pick_item_feature.sum(-1) >= 0)

        return pick_item_feature, avail_pick_item

    def reset(self):
        obs = self.env.reset()

        self.last_atype = np.zeros(113, dtype=np.uint8)
        obs['last_atype'] = self.last_atype

        self.obs = obs
        self.prev_action = None
        obs['action_class'] = np.array([ActionClasses.ATYPE], np.uint8)

        extra_avail_atype = self._get_extra_avail_atype(obs, obs['action_class'][0])
        if not (extra_avail_atype is None):
            obs['avail_atype'] = extra_avail_atype

        avail_spell = self._get_avail_spell(obs)
        avail_use_item = self._get_avail_use_item(obs, self.prev_action)
        pick_item_feature, avail_pick_item = self._get_pick_item_feature_and_avail(obs)

        obs['avail_spell'] = avail_spell
        obs['avail_use_item'] = avail_use_item
        obs['pick_item_feature'] = pick_item_feature
        obs['avail_pick_item'] = avail_pick_item

        return obs

    def step(self, action):
        required_action_class = self.obs['action_class'][0]
        real_action = self._convert_to_real_action(required_action_class, action)
        obs, rew, done, info = self.env.step(real_action)

        if required_action_class == ActionClasses.PICK_ITEM:
            while obs['misc'][2] and not done: # Skip item lists after choose it
                obs, rew, done, info = self.env.step(ActionIds.TextCharacters_SPACE)

        if required_action_class == ActionClasses.ATYPE:
            self.last_atype = np.zeros(113, dtype=np.uint8)
            self.last_atype[action['atype']] = 1
        obs['last_atype'] = self.last_atype

        self.obs = obs
        self.prev_action = action
        obs['action_class'] = np.array([self._get_required_action_class(obs)], dtype=np.uint8)

        extra_avail_atype = self._get_extra_avail_atype(obs, obs['action_class'][0])
        if not (extra_avail_atype is None):
            obs['avail_atype'] = extra_avail_atype

        avail_spell = self._get_avail_spell(obs)
        avail_use_item = self._get_avail_use_item(obs, self.prev_action)
        pick_item_feature, avail_pick_item = self._get_pick_item_feature_and_avail(obs)

        obs['avail_spell'] = avail_spell
        obs['avail_use_item'] = avail_use_item
        obs['pick_item_feature'] = pick_item_feature
        obs['avail_pick_item'] = avail_pick_item

        return obs, rew, done, info

    ## for submission below here
    def on_reset_submission(self, env, obs, is_holding_aicrowd=False):
        self.is_holding_aicrowd = is_holding_aicrowd
        self.env_aicrowd = env
        #self.step = lambda action: self.step_submission(action, self.env_aicrowd)

        self.last_atype = np.zeros(113, dtype=np.uint8)
        obs['last_atype'] = self.last_atype

        self.obs = obs
        self.prev_action = None
        obs['action_class'] = np.array([ActionClasses.ATYPE], np.uint8)

        extra_avail_atype = self._get_extra_avail_atype(obs, obs['action_class'][0])
        if not (extra_avail_atype is None):
            obs['avail_atype'] = extra_avail_atype

        avail_spell = self._get_avail_spell(obs)
        avail_use_item = self._get_avail_use_item(obs, self.prev_action)
        pick_item_feature, avail_pick_item = self._get_pick_item_feature_and_avail(obs)

        obs['avail_spell'] = avail_spell
        obs['avail_use_item'] = avail_use_item
        obs['pick_item_feature'] = pick_item_feature
        obs['avail_pick_item'] = avail_pick_item
        return obs