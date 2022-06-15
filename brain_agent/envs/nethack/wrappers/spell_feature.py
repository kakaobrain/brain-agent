import copy

import gym

import enum

import numpy as np

import nle
from nle import nethack as nh
from brain_agent.envs.nethack.ids.action_ids import ActionIds, KeyActionMapping
from brain_agent.envs.nethack.ids.misc_ids import MISCIds
from brain_agent.envs.nethack.utils.number_to_binary import number_to_binary
from brain_agent.envs.nethack.utils.get_object import get_object
from brain_agent.envs.nethack.ids.blstat_ids import BLStatIds
from brain_agent.envs.nethack.ids.obj_ids import ObjClasses

class SpellFeatureWrapper(gym.Wrapper):

    OBS_OFFSET_ID = 0
    OBS_OFFSET_LEVEL = 9
    OBS_OFFSET_FAIL = 10
    OBS_OFFSET_RETENTION = 11
    OBS_OFFSET_MAX = 12

    STR_OFFSET_NAME = 24
    STR_OFFSET_LEVEL = 45
    STR_OFFSET_CATEGORY = 50
    STR_OFFSET_FAIL = 64
    STR_OFFSET_RETENTION = 68
    STR_OFFSET_MAX = 79

    SCALE_LEVEL = 0.1
    SCALE_FAIL = 0.01
    SCALE_RETENTION = 0.01

    MAX_NUM_SPELL = 5

    def __init__(self, env):
        super().__init__(env)

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['spell_feature'] = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.MAX_NUM_SPELL, self.OBS_OFFSET_MAX), # id (9), level (1), fail(1), retention(1)
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self._steps_for_spell = 501

    def _get_binary_spell_id(self, str_spell):
        name_spell = str_spell[self.STR_OFFSET_NAME:self.STR_OFFSET_LEVEL] # such as " healing "
        name_spell = name_spell.strip(' ') # delete space
        obj_spell = get_object(name_spell)
        idx_obj_spell = obj_spell.oc_name_idx
        binary_spell_id = number_to_binary(idx_obj_spell, self.OBS_OFFSET_LEVEL)

        assert not obj_spell is None, f"The skill name {name_spell} is not in object group!!!"

        return binary_spell_id, name_spell

    def _get_spell_level(self, str_spell):
        level_spell = str_spell[self.STR_OFFSET_LEVEL:self.STR_OFFSET_CATEGORY] # such as " 2 "
        level_spell = level_spell.strip(' ') # delete space
        level_spell = int(level_spell)
        level_spell = self.SCALE_LEVEL * np.array([level_spell], dtype=np.float32)

        return level_spell

    def _get_spell_fail(self, str_spell):
        fail_spell = str_spell[self.STR_OFFSET_FAIL:self.STR_OFFSET_RETENTION] # such as " 54% "
        fail_spell = fail_spell.strip(' ') # delete space
        fail_spell = fail_spell.strip('%') # delete % char
        fail_spell = int(fail_spell)
        fail_spell = self.SCALE_FAIL * np.array([fail_spell], dtype=np.float32)

        return fail_spell

    def _get_spell_retention(self, str_spell):
        retention_spell = str_spell[self.STR_OFFSET_RETENTION:self.STR_OFFSET_MAX] # two possible cases: " 100% " or " 75% ~ 100% "
        retention_spell = retention_spell.strip(' ') # delete space
        if retention_spell == '100%':
            retention_spell = 100
        else:
            retention_spell = retention_spell[0:2].strip('%') # still str, not converted to int
            retention_spell = int(retention_spell) if retention_spell.isdecimal() else 0
        retention_spell = self.SCALE_RETENTION * np.array([retention_spell], dtype=np.float32)

        return retention_spell

    def _get_spell_feature(self, obs, env):
        _obs, _, _done, _ = env.step(ActionIds.Command_CAST)
        spell_feature = -np.ones([5, 12], dtype=np.float32)
        tty_chars = _obs['tty_chars'].copy()

        # no spell is learned
        str_msg = _obs['message'].tostring().decode('latin_1')
        if "You don't know any spells" in str_msg:
            return spell_feature, _done
        else:
            if _done: # Interestingly, if you try CAST too often, the environment returns done. We need to handle this situation
                return self.spell_feature, _done
            else:
                _, _, _done, _ = env.step(ActionIds.TextCharacters_SPACE)

        n_spell = 0
        start_spell_check = False
        for idx_line in range(21):
            str_spell = tty_chars[idx_line].tostring().decode('latin_1')

            #is_end_of_line = str_spell[20:25] == '(end)'
            is_end_of_line = '(end)' in str_spell or 'sort spells' in str_spell
            if is_end_of_line:
                break

            if not start_spell_check:
                start_spell_check = 'Name' in str_spell # start checking spell after we see a row of spell
                continue

            binary_spell_id, name_spell = self._get_binary_spell_id(str_spell)
            level_spell = self._get_spell_level(str_spell)
            fail_spell = self._get_spell_fail(str_spell)
            retention_spell = self._get_spell_retention(str_spell)

            # Update spell infos
            if not name_spell in self.spell_infos:
                self.spell_infos[name_spell] = dict(retention=retention_spell, last_read_step=0)

            spell_feature[n_spell] = np.concatenate([
                binary_spell_id,
                level_spell,
                fail_spell,
                retention_spell
            ], axis = 0)

            n_spell += 1

            # We do not consider spells after 5th
            if n_spell == self.MAX_NUM_SPELL:
                break

        return spell_feature, _done

    def _create_spell_feature(self, obs, env, done):
        # if obs['misc'][MISCIds.YN] or obs['misc'][MISCIds.WAITSPACE]:
        if (
                not np.any(obs['misc'])
                and not done
                and self._steps_for_spell > 500 # update spell feature for every 500 steps
        ):
            self.spell_feature, done = self._get_spell_feature(obs, env)
            self._steps_for_spell = 0

        obs['spell_feature'] = self.spell_feature
        self._steps_for_spell += 1
        return done

    def _force_read_when_forget(self, obs, rew, done, info, env=None):
        # force the agent to read the spell book when it forgets
        if env is None:
            env = self.env
        if done:
            return obs, rew, done, info

        def get_spellbook_key(obs, name_spell):
            idx = None
            mask_spbook = obs['inv_oclasses'] == ObjClasses.SPBOOK
            num_spbooks = mask_spbook.sum()
            idx_spbooks = np.where(mask_spbook)[0]
            for i in range(num_spbooks):
                idx_spbook = idx_spbooks[i]  # idx in item inventory
                str_spbook = obs['inv_strs'][idx_spbook].tobytes().decode('latin_1')
                if name_spell in str_spbook:
                    return obs['inv_letters'][idx_spbook].tobytes().decode('latin_1')
            return None

        _obs = obs
        for name_spell, spell_info in self.spell_infos.items():
            if (
                spell_info['retention'] == 0
                or (_obs['blstats'][BLStatIds.TIME] - spell_info['last_read_step']) > 20000
            ):
                x, y = _obs['blstats'][:2]
                readable = not np.any(_obs['misc'])
                monster_existance_in_7 = (_obs['glyphs'][y - 7:y + 7, x - 7:x + 7] < nh.GLYPH_PET_OFF).sum() > 1

                if readable and not monster_existance_in_7:
                    key = get_spellbook_key(_obs, name_spell)
                    if (
                        key is not None
                        and key in KeyActionMapping
                    ): # key is None when the spell book is dropped or missed
                        _ = env.step(ActionIds.Command_READ) # Do not have any reward
                        _obs, _rew, _done, _info = env.step(KeyActionMapping[key])
                        obs, rew, done, info = _obs, rew + _rew, _done, _info
                        spell_info['last_read_step'] = _obs['blstats'][BLStatIds.TIME]
                        count_while = 0
                        while np.any(_obs['misc']):
                            _obs, _rew, _done, _info = env.step(ActionIds.TextCharacters_SPACE)
                            obs, rew, done, info = _obs, rew + _rew, _done, _info
                            count_while += 1
                            if done:
                                break
                            if count_while > 100:
                                print("########################## WARNING from _force_read_when_forget in spell_feature.py #############################")
                    done = self._create_spell_feature(obs, env, done)

        return obs, rew, done, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        done = self._create_spell_feature(obs, self.env, done)
        obs, reward, done, info = self._force_read_when_forget(obs, reward, done, info)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._steps_for_spell = 501
        self.spell_infos = dict()
        self._create_spell_feature(obs, self.env, done=False)

        return obs

    ## for submission below here
    def on_reset_submission(self, env, obs, is_holding_aicrowd=False):
        self._steps_for_spell = 501
        self.spell_infos = dict()
        self._create_spell_feature(obs, env, done=False)
        return obs