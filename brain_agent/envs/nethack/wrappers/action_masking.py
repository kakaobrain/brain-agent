import time
import enum
import gym
import torch
import numpy as np
import string
import re

from brain_agent.envs.nethack.ids.action_ids import ActionIds
from brain_agent.envs.nethack.ids.obj_ids import ObjClasses
from brain_agent.envs.nethack.ids.cmap_ids import CmapIds
from brain_agent.envs.nethack.ids.misc_ids import MISCIds
from brain_agent.envs.nethack.ids.blstat_ids import BLStatIds

from nle.nethack import ACTIONS, USEFUL_ACTIONS, GLYPH_CMAP_OFF, action_id_to_type, Command, MiscAction, TextCharacters

from brain_agent.utils.timing import Timing
timing = Timing()

import logging
logger = logging.getLogger(__name__)

import nle.nethack as nh


class Directions(enum.IntEnum):
    NW = 0
    N = 1
    NE = 2
    W = 3
    E = 4
    SW = 5
    S = 6
    SE = 7


CHAR_ACTION_IDX_DICT = dict()
ESC_ACTION_IDX = ACTIONS.index(Command.ESC)

for i, k in enumerate(ACTIONS):
    for ch in string.printable:
        if k.value == ord(ch):
            CHAR_ACTION_IDX_DICT[ch] = i
            break


class MessageActionMaskingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['avail_atype'] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(113,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def reset(self):
        obs = self.env.reset()

        avail_atype = np.array([int(action in USEFUL_ACTIONS) for action in ACTIONS], dtype=np.uint8)  # activate only useful actions
        obs['avail_atype'] = avail_atype
        return obs

    def _create_avail_atype(self, obs, done):
        avail_atype = np.array([int(action in USEFUL_ACTIONS) for action in ACTIONS], dtype=np.uint8)  # activate only useful actions
        message = obs['message'].tostring().decode(errors='ignore')
        m = re.search('\? \[(.+)\]', message)
        if m is not None:
            chars = m.group(1)
            if ' or ?*' in chars:
                avail_atype = np.zeros_like(avail_atype)
                chars = chars.replace(' or ?*', '')
                shorts = re.findall('([a-zA-Z0-9]-[a-zA-Z0-9])', chars)
                if shorts:
                    for short in shorts:
                        chars = chars.replace(short, ''.join([chr(c) for c in range(ord(short[0]), ord(short[2]) + 1)]))
                for x in chars:
                    if x not in '?*' and x in CHAR_ACTION_IDX_DICT:
                        avail_atype[CHAR_ACTION_IDX_DICT[x]] = 1
                avail_atype[ESC_ACTION_IDX] = 1  # enable ESC
                # print(action_id_to_type(ACTIONS[action].value), action, reward, message, '------', m.group(1), '->', chars)
        obs['avail_atype'] = avail_atype
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._create_avail_atype(obs, done)
        return obs, reward, done, info


class MiscActionWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        for i in range(10):
            if done:
                break
            misc = obs['misc']
            tty_message = obs['tty_chars'][0].tostring()
            if misc[2] and tty_message[0] != ' ':
                # print(i, ACTIONS[action], misc, obs['tty_chars'][0].tostring())
                # action = ACTIONS.index(TextCharacters.SPACE)
                action = ACTIONS.index(MiscAction.MORE)
                obs, reward, done, info = self.env.step(action)
            else:
                break
        return obs, reward, done, info


class RuleBasedActionMaskingWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg

        obs_spaces = {
            'avail_atype': gym.spaces.Box(
                low=0,
                high=1,
                shape=(113,),
                dtype=np.float32
            )
        }
        # Add other obs spaces other than blstats
        obs_spaces.update([
            (k, self.env.observation_space[k]) for k in self.env.observation_space
        ])
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.block_kill_pet = self.cfg.env.block_kill_pet
        self.hide_pet = self.cfg.env.hide_pet
        self.roles_to_block_kill_pet = ['wizard', 'cave', 'tourist', 'healer']

    def _set_block_on_reset(self, env=None):
        if env is None:
            env = self.env

        obs = env.step(25)  # get check attribute
        str_attribute = obs[0]['tty_chars'].reshape(-1).tostring().decode('latin_1').lower()
        env.step(99); env.step(99)  # pass 2 pages of explanation by entering 'space' action

        role = None

        # Check role
        roles = ['archeologist', 'barbarian', 'cave', 'healer', 'knight', 'monk', 'priest', 'ranger', 'rogue', 'samurai', 'tourist', 'valkyrie', 'wizard' ]
        for _role in roles:
            if _role in str_attribute:
                role = _role
        assert not (
                role is None
        ), f"role: {role}"

        if self.cfg.env.block_kill_pet and role in self.roles_to_block_kill_pet:
            self.block_kill_pet = True
        else:
            self.block_kill_pet = False

        if self.cfg.env.hide_pet and role in self.roles_to_block_kill_pet:
            self.hide_pet = True
        else:
            self.hide_pet = False

    def _block_kill_pet(self, obs):
        check1_pet = obs['glyphs'] > nh.GLYPH_PET_OFF
        check2_pet = obs['glyphs'] < nh.GLYPH_INVIS_OFF
        is_pet = np.logical_and(check1_pet, check2_pet)
        y_pet, x_pet = np.where(is_pet)
        avail_atype = obs['avail_atype']

        if x_pet.shape[0] == 1 and y_pet.shape[0] == 1: # is pet is visible
            x_pet, y_pet = x_pet[0], y_pet[0] # numpy array to int
            x_agent, y_agent = obs['blstats'][0:2]
            if abs(x_pet - x_agent) <= 1 and abs(y_pet - y_agent) <= 1:
                avail_atype[ActionIds.Command_FIGHT] = 0 # if pet is near the agent, block FIGHT since it is used to kill pet


    def _hide_pet(self, obs):
        # It makes pet feature invisible, pet becomes invisible helper
        y_pet, x_pet = np.where(obs['specials'])

        if x_pet.shape[0] == 1 and y_pet.shape[0] == 1:  # is pet is visible
            x_pet, y_pet = x_pet[0], y_pet[0]
            obs_glyphs_wall_pad = np.ones([23, 81]) * (CmapIds.WALL[0] + nh.GLYPH_CMAP_OFF)
            obs_glyphs_wall_pad[1:22,1:80] = obs['glyphs']
            x_pet_pad, y_pet_pad = x_pet + 1, y_pet + 1
            pet_surroundings = obs_glyphs_wall_pad[y_pet_pad - 1: y_pet_pad + 2 , x_pet_pad -1: x_pet_pad + 2] - nh.GLYPH_CMAP_OFF
            for dy_ in range(3):
                for dx_ in range(3):
                    dx, dy = dx_ - 1, dy_ -1 # convert to realy dx, dy not indexing
                    if (
                        pet_surroundings[dy_, dx_] in CmapIds.FLOOR_OF_A_ROOM # tty_char: 46, tty_color: 7
                        or pet_surroundings[dy_, dx_] in CmapIds.CORRIDOR # tty_char: 35, tty_color: 7
                    ):
                        x_pet_tty, y_pet_tty = x_pet, y_pet + 1 # convert x, y in ttys since they have differences in shape
                        obs['tty_chars'][y_pet_tty, x_pet_tty] = obs['tty_chars'][y_pet_tty + dy, x_pet_tty + dx] # override pet as a floor
                        obs['tty_colors'][y_pet_tty, x_pet_tty] = obs['tty_colors'][y_pet_tty + dy, x_pet_tty + dx]

    def _create_avail_atype(self, obs):
        avail_atype = np.array([int(action in USEFUL_ACTIONS) for action in ACTIONS], dtype=np.uint8)  # activate only useful actions
        obs['avail_atype'] = avail_atype

        if self.block_kill_pet:
            self._block_kill_pet(obs)

    def reset(self):
        obs = self.env.reset()
        self._set_block_on_reset()

        self._create_avail_atype(obs)
        if self.hide_pet:
            self._hide_pet(obs)
        return obs

    def step(self, action):
        obs, r, done,info = self.env.step(action)
        self._create_avail_atype(obs)
        if self.hide_pet:
            self._hide_pet(obs)
        return obs, r, done, info

    ## for submission below here
    def on_reset_submission(self, env, obs, is_holding_aicrowd=True):
        self._set_block_on_reset(env)

        self._create_avail_atype(obs)
        if self.hide_pet:
            self._hide_pet(obs)
        return obs


class ActionMaskingWrapper(gym.Wrapper):
    ASCII_UPSTAIR = 60
    ASCII_DOWNSTAIR = 62
    def __init__(self, env, cfg=None):
        super().__init__(env)
        self.cfg = cfg
        self.env = env

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['avail_atype'] = gym.spaces.Box(
            low=0,
            high=1,
            shape=(113,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.dict_dungeon_info = dict()
        self.use_travel_action = False

        if self.cfg is not None and self.cfg.env.use_action_overrider:
            self.use_travel_action = True

    def _update_dungeon_info(self, obs):
        number_and_level = (int(obs['blstats'][BLStatIds.DN]), int(obs['blstats'][BLStatIds.LN]))
        if not number_and_level in self.dict_dungeon_info:
            self.dict_dungeon_info[number_and_level] = {'pos_upstairs': set(), 'pos_downstairs': set()} # (y, x)

        upstair = ord('<')
        ys, xs = np.where(obs['chars']==upstair)
        for i in range(ys.shape[0]):
            self.dict_dungeon_info[number_and_level]['pos_upstairs'].add( (ys[i], xs[i]) )

        downstair = ord('>')
        ys, xs = np.where(obs['chars'] == downstair)
        for i in range(ys.shape[0]):
            self.dict_dungeon_info[number_and_level]['pos_downstairs'].add((ys[i], xs[i]))

    def _get_obs_look(self, obs, done):
        #if not (obs['misc'][MISCIds.YN] or obs['misc'][MISCIds.WAITSPACE]): # take a look if it is available
        if not np.any(obs['misc']) and not done:
            _obs, _, _, _ = self.env.step(ActionIds.Command_LOOK)
            obs_look = _obs['message'].tostring().decode('latin_1')
            for _ in range(10): # skip messages after look
                if _obs['misc'][MISCIds.WAITSPACE]:
                    _obs, _, _, _ = self.env.step(ActionIds.TextCharacters_SPACE)
                else: break
            return obs_look
        else:
            return None

    def _create_avail_atype(self, obs, done):
        def get_inv_strs_of_obj_class(obs, obj_class, encoding='latin_1'):
            # Get strs related to specific object class in inventory
            idx_obj_classes = obs['inv_oclasses'] == obj_class
            _str_obj_classes = obs['inv_strs'][idx_obj_classes]
            str_obj_classes = list()
            for idx in range(_str_obj_classes.shape[0]):
                str_decoded = _str_obj_classes[idx].tostring().decode(encoding)
                str_obj_classes.append(str_decoded)
            return str_obj_classes\

        with timing.timeit('create_np'):
            avail_atype = np.array([int(action in USEFUL_ACTIONS) for action in ACTIONS],
                                   dtype=np.uint8)  # activate only useful actions
            avail_direction = np.zeros(8, dtype=np.uint8)

        if done:
            return avail_atype, avail_direction
        with timing.timeit('inv_oclasses'):
            num_atypes_before = np.sum(avail_atype)
            inv_oclasses = obs['inv_oclasses'].tolist()

        if obs['blstats'][BLStatIds.ENERGY] < 5:
            avail_atype[ActionIds.Command_CAST] = 0

        # check surrounding
        x, y = obs['blstats'][0:2]
        glyphs_padded = np.ones([23, 81], dtype=np.uint16) * (CmapIds.WALL[0] + GLYPH_CMAP_OFF)
        glyphs_padded[1:22, 1:80] = obs['glyphs']

        surroundings = glyphs_padded[y + 1 - 1:y + 1 + 2, x + 1 - 1:x + 1 + 2] # (3,3)
        surroundings_cmap = surroundings - GLYPH_CMAP_OFF
        surroundings_cmap = surroundings_cmap.reshape(-1)
        surroundings_cmap = np.delete(surroundings_cmap, 4, axis=0).tolist()  # delte center glyph since it is need not to be checked (it is agent)

        # check wall / empty space for ActionIds.Command_KICK
        with timing.timeit('kick'):
            wall_surroundings = np.array([s_c in CmapIds.WALL for s_c in surroundings_cmap])
            floor_surroundings = np.array([s_c in CmapIds.FLOOR_OF_A_ROOM for s_c in surroundings_cmap])
            wall_or_floor = np.logical_or(wall_surroundings, floor_surroundings)
            if np.all(wall_or_floor): # every pixel around agent is wall or empty
                avail_atype[ActionIds.Command_KICK] = 0
                logger.debug("[block] kick")

        # check wall for move direction
        with timing.timeit('direction'):
            if wall_surroundings[Directions.NW]:
                avail_atype[ActionIds.CompassDirection_NW] = 0
                avail_atype[ActionIds.CompassDirectionLonger_NW] = 0
                avail_direction[ActionIds.CompassDirection_NW] = 0
                logger.debug("[block] direction_NW")
            if wall_surroundings[Directions.N]:
                avail_atype[ActionIds.CompassDirection_N] = 0
                avail_atype[ActionIds.CompassDirectionLonger_N] = 0 # Long move is also deactivated in this case
                avail_direction[ActionIds.CompassDirection_N] = 0
                logger.debug("[block] direction_N")
            if wall_surroundings[Directions.NE]:
                avail_atype[ActionIds.CompassDirection_NE] = 0
                avail_atype[ActionIds.CompassDirectionLonger_NE] = 0
                avail_direction[ActionIds.CompassDirection_NE] = 0
                logger.debug("[block] direction_NE")
            if wall_surroundings[Directions.W]:
                avail_atype[ActionIds.CompassDirection_W] = 0
                avail_atype[ActionIds.CompassDirectionLonger_W] = 0
                avail_direction[ActionIds.CompassDirection_W] = 0
                logger.debug("[block] direction_W")
            if wall_surroundings[Directions.E]:
                avail_atype[ActionIds.CompassDirection_E] = 0
                avail_atype[ActionIds.CompassDirectionLonger_E] = 0
                avail_direction[ActionIds.CompassDirection_E] = 0
                logger.debug("[block] direction_E")
            if wall_surroundings[Directions.SW]:
                avail_atype[ActionIds.CompassDirection_SW] = 0
                avail_atype[ActionIds.CompassDirectionLonger_SW] = 0
                avail_direction[ActionIds.CompassDirection_SW] = 0
                logger.debug("[block] direction_SW")
            if wall_surroundings[Directions.S]:
                avail_atype[ActionIds.CompassDirection_S] = 0
                avail_atype[ActionIds.CompassDirectionLonger_S] = 0
                avail_direction[ActionIds.CompassDirection_S] = 0
                logger.debug("[block] direction_S")
            if wall_surroundings[Directions.SE]:
                avail_atype[ActionIds.CompassDirection_SE] = 0
                avail_atype[ActionIds.CompassDirectionLonger_SE] = 0
                avail_direction[ActionIds.CompassDirection_SE] = 0
                logger.debug("[block] direction_SE")

        # check closed_door for ActionIdss.Command_OPEN
        with timing.timeit('open_close'):
            closed_door_surroundings = np.array([s_c in CmapIds.CLOSED_DOOR for s_c in surroundings_cmap])
            if not np.any(closed_door_surroundings):
                avail_atype[ActionIds.Command_OPEN] = 0
                logger.debug("[block] OPEN")

            # check opened_door for ActionIdss.Command_CLOSE
            opened_door_surroundings = np.array([s_c in CmapIds.OPENED_DOOR for s_c in surroundings_cmap])
            if not np.any(opened_door_surroundings):
                avail_atype[ActionIds.Command_CLOSE] = 0
                logger.debug("[block] CLOSE")

        ## check upstairs
        number_and_level = (int(obs['blstats'][BLStatIds.DN]), int(obs['blstats'][BLStatIds.LN]))
        avail_atype[ActionIds.MiscDirection_UP] = 0
        for y_up, x_up in self.dict_dungeon_info[number_and_level]['pos_upstairs']:
            if (
                y_up == y and x_up == x
                and not (obs['blstats'][BLStatIds.DN] == 0 and obs['blstats'][BLStatIds.LN] == 1)
            ): # block going up at beggining
                avail_atype[ActionIds.MiscDirection_UP] = 1
                logger.debug("[on] UP")

        ## check downstairs
        LV_TO_LN_RATIO = 1.0
        avail_atype[ActionIds.MiscDirection_DOWN] = 0
        for y_down, x_down in self.dict_dungeon_info[number_and_level]['pos_downstairs']:
            if (
                y_down == y and x_down == x
                and obs['blstats'][BLStatIds.LV] >= LV_TO_LN_RATIO * obs['blstats'][BLStatIds.LN] # Agent's level should be higher than dungeon level
            ):
                avail_atype[ActionIds.MiscDirection_DOWN] = 1
                logger.debug("[on] DOWN")

        # check comestible  for ActionIds.Command_EAT
        # if not ObjClasses.FOOD in inv_oclasses:
        #     avail_atype[ActionIds.Command_EAT] = 0
        #     logger.debug("[block] EAT")

        # check tool check  for ActionIds.Command_APPLY
        if not ObjClasses.TOOL in inv_oclasses:
            avail_atype[ActionIds.Command_APPLY] = 0
            logger.debug("[block] APPLY")

        # check scroll / spell book for ActionIds.Command_READ
        if not (
                ObjClasses.SCROLL in inv_oclasses
                or ObjClasses.SCROLL in inv_oclasses
        ):
            avail_atype[ActionIds.Command_READ] = 0
            logger.debug("[block] READ")

        # check weapon
        with timing.timeit('weapon'):
            str_weapons = get_inv_strs_of_obj_class(obs, ObjClasses.WEAPON)
            if len(str_weapons) == 0: # Not wielding weapon
                avail_atype[ActionIds.Command_FORCE] = 0 # Force is not available if not wielding a weapon

        # check armor
        with timing.timeit('armor'):
            str_armors = get_inv_strs_of_obj_class(obs, ObjClasses.ARMOR)
            worn_armors = ['worn' in str_armor for str_armor in str_armors]
            worn_all_armors = np.all(worn_armors)  # np.all([]) = True
            worn_any_armor = np.any([worn_armors])  # np.any([]) = False
            if worn_all_armors:
                avail_atype[ActionIds.Command_WEAR] = 0
                logger.debug("[block] WEAR")
            if not worn_any_armor:  # need not to take-off when wearing nothing
                avail_atype[ActionIds.Command_TAKEOFF] = 0
                avail_atype[ActionIds.Command_TAKEOFFALL] = 0
                logger.debug("[block] TAKEOFF, TAKEOFFALL")

        # check ring / amulet in inventory
        with timing.timeit('accessory'):
            str_rings = get_inv_strs_of_obj_class(obs, ObjClasses.RING)
            worn_rings = ['worn' in str_ring for str_ring in str_rings]
            worn_all_rings = np.all(worn_rings)  # np.all([]) = True
            worn_any_ring = np.any(worn_rings)  # np.any([]) = False

            str_amulets = get_inv_strs_of_obj_class(obs, ObjClasses.AMULET)
            worn_amulets = ['worn' in str_amulet for str_amulet in str_amulets]
            worn_all_amulets = np.all(worn_amulets)  # np.all([]) = True
            worn_any_amulet = np.any(worn_amulets)  # np.any([]) = False

            if worn_all_rings and worn_all_amulets:
                avail_atype[ActionIds.Command_PUTON] = 0
                logger.debug("[block] PUTON")
            if not (worn_any_ring or worn_any_amulet):
                avail_atype[ActionIds.Command_REMOVE] = 0
                logger.debug("[block] REMOVE")

        # check wands
        if not ObjClasses.WAND in inv_oclasses:
            avail_atype[ActionIds.Command_ZAP] = 0
            logger.debug("[block] ZAP")
        with timing.timeit('etc'):
            # check potion / fountain check for ActionIds.Command_QUAFF
            on_fountain = 'fountain' in obs['look'] if 'look' in obs else True
            if not (
                    ObjClasses.POTION in inv_oclasses
                    or on_fountain
            ):
                avail_atype[ActionIds.Command_QUAFF] = 0
                logger.debug("[block] QUAFF")

            # check no objects
            no_objects_below = 'no objects' in obs['look'] if 'look' in obs else False
            if no_objects_below:
                avail_atype[ActionIds.Command_PICKUP] = 0
                logger.debug("[block] PICKUP")

            # check go down
            on_staircase_down = 'staircase down' in obs['look'] if 'look' in obs else True
            on_ladder_down = 'ladder down' in obs['look'] if 'look' in obs else True
            if not (on_staircase_down or on_ladder_down):
                avail_atype[ActionIds.MiscDirection_DOWN] = 0
                logger.debug("[block] DOWN")

            # check go up
            on_staircase_up = 'staircase up' in obs['look'] if 'look' in obs else True
            on_ladder_up = 'ladder up' in obs['look'] if 'look' in obs else True
            if not (on_staircase_up or on_ladder_up):
                avail_atype[ActionIds.MiscDirection_UP] = 0
                logger.debug("[block] UP")

            # deactivate inventory action
            avail_atype[ActionIds.Command_INVENTORY] = 0
            avail_atype[ActionIds.Command_INVENTTYPE] = 0
            logger.debug("[block] INVENTORY, INVENTTYPE")

            # check chest / box under agent
            on_chset = 'chest' in obs['look'] if 'look' in obs else True
            on_box = 'box' in obs['look'] if 'look' in obs else True
            if not (on_chset or on_box):  # Force is unlocking box action
                avail_atype[ActionIds.Command_FORCE] = 0
                avail_atype[ActionIds.Command_LOOT] = 0
                logger.debug("[block] FORCE, LOOT")

            # deactivate unnecessary actions
            avail_atype[ActionIds.Command_CALL] = 0
            avail_atype[ActionIds.Command_ADJUST] = 0
            avail_atype[ActionIds.Command_ATTRIBUTES] = 0
            avail_atype[ActionIds.Command_DROPTYPE] = 0  # drop is enough
            avail_atype[ActionIds.Command_RIDE] = 0
            avail_atype[ActionIds.Command_VERSION] = 0  # checking nethack version is not necessary
            avail_atype[ActionIds.Command_VERSIONSHORT] = 0  # checking nethack version is not necessary
            avail_atype[ActionIds.Command_CHAT] = 0  # Chat is not useful for RL Agent
            if (
                self.cfg is not None
                and not self.cfg.env.use_action_overrider
            ): # ActionOverrider allows the agent to use engrave usefully with "Elbereth"
                avail_atype[ActionIds.Command_ENGRAVE] = 0  # It may need to activate engraving if we are going to engrace 'Elbereth' which blocks the attack of some monster
            avail_atype[ActionIds.Command_RUSH] = 0
            avail_atype[ActionIds.Command_RUSH2] = 0
            avail_atype[ActionIds.Command_MOVE] = 0
            avail_atype[ActionIds.Command_MOVEFAR] = 0
            avail_atype[ActionIds.Command_OFFER] = 0
            avail_atype[ActionIds.Command_JUMP] = 0

            # if self.use_travel_action:
            #     avail_atype[ActionIds.Command_TRAVEL] = 1

            logger.debug("[block] ADJUST, ATTRIBUTES, DROP_TYPE, RIDE, VERSION, VERSIONSHORT, ENGRAVE")

            num_atypes_after = np.sum(avail_atype)
        logger.debug(
            f"num_blocked_actions: {num_atypes_before - num_atypes_after}"
        )
        obs['avail_atype'] = avail_atype

    def reset(self):
        obs = self.env.reset()

        # obs_look = self._get_obs_look(obs, False)
        # if not obs_look is None:
        #     self.obs_look = obs_look
        #     obs['look'] = self.obs_look # it should be set when it is available on reset

        self._update_dungeon_info(obs)
        self._create_avail_atype(obs, done=False)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # obs_look = self._get_obs_look(obs, done)
        # if not obs_look is None:
        #     self.obs_look = obs_look
        # obs['look'] = self.obs_look

        self._update_dungeon_info(obs)
        self._create_avail_atype(obs, done)

        return obs, reward, done, info

    ## for submission below here
    def on_reset_submission(self, env, obs, is_holding_aicrowd=False):
        self.is_holding_aicrowd = is_holding_aicrowd
        self.env_aicrowd = env
        #self.step = lambda action: self.step_submission(action, self.env_aicrowd)  # override step function

        self._update_dungeon_info(obs)
        self._create_avail_atype(obs, done=False)

        return obs
