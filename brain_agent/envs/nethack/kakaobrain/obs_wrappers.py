import os
import copy

import gym
import numpy as np

from brain_agent.envs.nethack.utils.glyph_to_feature import glyph_to_feature_table

class VectorFeaturesWrapper(gym.Wrapper):
    """Create network-friendly vector features from the stuff nethack has"""

    # Hand-chosen scaling values for each blstat entry. Aims to limit them in [0, 1] range.
    BLSTAT_NORMALIZATION_STATS = np.array([
        1.0 / 79.0,     # [0] hero col
        1.0 / 21,       # [1] hero row
        0.0,            # [2] strength pct
        1.0 / 10,       # [3] strength
        1.0 / 10,       # [4] dexterity
        1.0 / 10,       # [5] constitution
        1.0 / 10,       # [6] intelligence
        1.0 / 10,       # [7] wisdom
        1.0 / 10,       # [8] charisma
        0.0,            # [9] score
        1.0 / 10,       # [10] hitpoints
        1.0 / 10,       # [11] max hitpoints
        0.0,            # [12] depth
        1.0 / 1000,     # [13] gold
        1.0 / 10,       # [14] energy
        1.0 / 10,       # [15] max energy
        1.0 / 10,       # [16] armor class
        0.0,            # [17] monster level
        1.0 / 10,       # [18] experience level
        1.0 / 100,      # [19] experience points
        1.0 / 1000,            # [20] time
        1.0,            # [21] hunger_state
        1.0 / 10,            # [22] carrying capacity
        0.0,            # [23] dungeon number
        0.0,            # [24] level number
        0.0,            # [25] condition bits  # nle v0.7.3+ only.
    ])
    BLSTATS_NORM_MEAN = np.array([38.143184, 11.350447, 19.385929, 48.681689, 16.364037, 17.475396, 13.386432, 15.515700, 11.217532, 0,
                         140.676504, 142.441755, 0, 246.591319, 64.147795, 74.791243, -14.114734, 0, 13.747976, 21255.588020,
                         31590.399317, 0, 0, 17.505681, 0, 0])
    BLSTATS_NORM_STD = np.array([19.516993, 4.555499, 3.127983, 36.879129, 2.781533, 1.698254, 4.207449, 4.062547, 3.753857, 0,
                        87.590922, 86.641331, 0, 1077.061845, 53.786700, 56.686335, 12.114323, 0, 4.810798, 27597.589867,
                        19627.790005, 0, 0, 13.921685, 0, 0])
    BLSTATS_NORM_MIN = np.array([0, 0, 8, 8, 7, 8, 4, 7, 3, 0,
                        0, 10, 0, 0, 0, 1, -42, 0, 1, 0.0,
                        1, 0, 0, 1, 0, 0])
    BLSTATS_NORM_MAX = np.array([78, 21, 25, 118, 23, 22, 24, 25, 20, 0,
                        442, 442, 0, 64228, 347, 347, 10, 0, 30, 405504.0,
                        89228, 0, 0, 53, 0, 0])

    TTYREC_BLSTATS_INDICES = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18, 19]

    CROP_CENTER_NORMALIZATION_STATS = np.array([
        1.0 / 20,
        1.0 / 80
    ])

    # Make sure we do not spook the network
    BLSTAT_CLIP_RANGE = (-5, 5)

    def __init__(self, env, cfg):
        super().__init__(env)
        self.normalize_blstats = cfg.env.normalize_blstats
        self.cfg = cfg

        num_item_features = 0
        if self.normalize_blstats == 'default':
            num_items = VectorFeaturesWrapper.BLSTAT_NORMALIZATION_STATS.shape[0]
        else:
            num_items = len(VectorFeaturesWrapper.TTYREC_BLSTATS_INDICES)

        self.use_character_feature = 'vector_obs' in self.observation_space.spaces # if there is already 'vector_feature' in env.observation_space, it supposed to use character_feature
        if self.use_character_feature:
            num_items = self.observation_space['vector_obs'].shape[0]

        self.use_item_feature = 'item_feature' in self.observation_space.spaces and self.cfg.model.encoder.serialize_to_vobs # going to concat serialized item_feature to vector_obs
        if self.use_item_feature:
            num_items += np.prod(self.observation_space['item_feature'].shape)
            self.observation_space.spaces.pop('item_feature')

        self.use_spell_feature = 'spell_feature' in self.observation_space.spaces and self.cfg.model.encoder.serialize_to_vobs  # going to concat serialized item_feature to vector_obs
        if self.use_spell_feature:
            num_items += np.prod(self.observation_space['spell_feature'].shape)
            self.observation_space.spaces.pop('spell_feature')

        if 'last_atype' in self.observation_space.spaces and self.cfg.model.encoder.cat_last_atype:
            num_items += self.observation_space['last_atype'].shape[0]
            self.observation_space.spaces.pop('last_atype')

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['vector_obs'] = gym.spaces.Box(
            low=VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[0],
            high=VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[1],
            shape=(num_items,),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _create_vector_obs(self, obs):
        if not self.use_character_feature and not ('vector_obs' in obs):
            if self.normalize_blstats == 'zscore':
                blstats = obs["blstats"][VectorFeaturesWrapper.TTYREC_BLSTATS_INDICES]
                obs_vector = (blstats - self.BLSTATS_NORM_MEAN[self.TTYREC_BLSTATS_INDICES]) / self.BLSTATS_NORM_STD[self.TTYREC_BLSTATS_INDICES]
            elif self.normalize_blstats == 'minmax':
                blstats = obs["blstats"][VectorFeaturesWrapper.TTYREC_BLSTATS_INDICES]
                obs_vector = (blstats - self.BLSTATS_NORM_MIN[self.TTYREC_BLSTATS_INDICES]) / (self.BLSTATS_NORM_MAX[self.TTYREC_BLSTATS_INDICES] - self.BLSTATS_NORM_MIN[self.TTYREC_BLSTATS_INDICES])
            elif self.normalize_blstats == 'default':
                blstats = obs["blstats"]
                norm_stats = VectorFeaturesWrapper.BLSTAT_NORMALIZATION_STATS
                obs_vector = blstats * norm_stats
            else:
                raise NotImplementedError
            np.clip(
                obs_vector,
                VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[0],
                VectorFeaturesWrapper.BLSTAT_CLIP_RANGE[1],
                out=obs_vector
            )
        else:
            obs_vector = obs['vector_obs']

        if self.use_item_feature or 'item_feature' in obs and self.cfg.model.encoder.serialize_to_vobs:
            obs_vector = np.concatenate([obs_vector, obs['item_feature'].reshape(-1)], axis=-1)
            obs.pop('item_feature')

        if self.use_spell_feature or 'spell_feature' in obs and self.cfg.model.encoder.serialize_to_vobs:
            obs_vector = np.concatenate([obs_vector, obs['spell_feature'].reshape(-1)], axis=-1)
            obs.pop('spell_feature')

        if 'last_atype' in obs and self.cfg.model.encoder.cat_last_atype:
            obs_vector = np.concatenate([obs_vector, obs['last_atype']], axis=-1)
            obs.pop('last_atype')

        obs["vector_obs"] = obs_vector

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._create_vector_obs(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._create_vector_obs(obs)
        return obs

    ## for submission below here
    def on_reset_submission(self, env, obs):
        obs = self._create_vector_obs(obs)
        return obs

class GlyphEncoderWrapper(gym.Wrapper):
    SCALE_COUNT = 0.1

    def __init__(self, env):
        super().__init__(env)
        self.glyph_to_feature_table = glyph_to_feature_table()

        # sample-factory expects at least one observation named "screen_feature"
        obs_spaces = {
            'obs': gym.spaces.Box(
                low=0,
                high=1,
                shape=(42, 21, 79),
                dtype=np.bool
            ),
        }

        # Add other obs spaces other than blstats
        obs_spaces.update([
            (k, self.env.observation_space[k]) for k in self.env.observation_space
        ])
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _create_screen_feature(self, obs):
        screen_feature_glyph = self.glyph_to_feature_table[obs['glyphs']] # (21, 79, 41)
        screen_feature_pos = np.zeros(obs['glyphs'].shape, dtype=np.bool) # (21, 79)
        x, y = obs['blstats'][0], obs['blstats'][1]
        screen_feature_pos[y, x] = True
        screen_feature_pos = np.expand_dims(screen_feature_pos, axis=2) # (21, 79, 1)

        screen_feature = np.concatenate([screen_feature_glyph, screen_feature_pos], axis=2).transpose(2, 0, 1)
        obs['obs'] = screen_feature

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._create_screen_feature(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        self._create_screen_feature(obs)
        return obs

    ## for submission below here
    def on_reset_submission(self, env, obs):
        self._create_screen_feature(obs)
        return obs