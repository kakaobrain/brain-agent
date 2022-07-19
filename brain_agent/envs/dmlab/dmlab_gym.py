import os
import random
import shutil
import hashlib
import deepmind_lab
import gym
import numpy as np

from brain_agent.envs.dmlab import dmlab_level_cache
from brain_agent.envs.dmlab.dmlab30 import DMLAB_INSTRUCTIONS, DMLAB_MAX_INSTRUCTION_LEN, DMLAB_VOCABULARY_SIZE, \
    IMPALA_ACTION_SET, EXTENDED_ACTION_SET, EXTENDED_ACTION_SET_LARGE
from brain_agent.utils.logger import log


def string_to_hash_bucket(s, vocabulary_size):
    return (int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % (vocabulary_size - 1)) + 1


def dmlab_level_to_level_name(level):
    level_name = level.split('/')[-1]
    return level_name


class DmlabGymEnv(gym.Env):
    def __init__(
            self, task_id, level, action_repeat, res_w, res_h, dataset_path,
            action_set, use_level_cache, level_cache_path, extra_cfg=None,
    ):
        self.width = res_w
        self.height = res_h

        self.main_observation = 'RGB_INTERLEAVED'
        self.instructions_observation = DMLAB_INSTRUCTIONS

        self.action_repeat = action_repeat

        self.random_state = None

        self.task_id = task_id
        self.level = level
        self.level_name = dmlab_level_to_level_name(self.level)

        self.cache = dmlab_level_cache.DMLAB_GLOBAL_LEVEL_CACHE

        self.instructions = np.zeros([DMLAB_MAX_INSTRUCTION_LEN], dtype=np.int32)

        observation_format = [self.main_observation]
        observation_format += [self.instructions_observation]

        config = {
            'width': self.width,
            'height': self.height,
            'datasetPath': dataset_path,
            'logLevel': 'error',
        }

        if extra_cfg is not None:
            config.update(extra_cfg)
        config = {k: str(v) for k, v in config.items()}

        self.use_level_cache = use_level_cache
        self.level_cache_path = level_cache_path

        env_level_cache = self if use_level_cache else None
        self.env_uses_level_cache = False  # will be set to True when this env instance queries the cache
        self.last_reset_seed = None

        if env_level_cache is not None:
            if not isinstance(self.cache, dmlab_level_cache.DmlabLevelCacheGlobal):
                raise Exception(
                    'DMLab global level cache object is not initialized! Make sure to call'
                    'dmlab_ensure_global_cache_initialized() in the main thread before you fork any child processes'
                    'or create any DMLab envs'
                )

        self.dmlab = deepmind_lab.Lab(
            level, observation_format, config=config, renderer='software', level_cache=env_level_cache,
        )

        if action_set == 'impala_action_set':
            self.action_set = IMPALA_ACTION_SET
        elif action_set == 'extended_action_set':
            self.action_set = EXTENDED_ACTION_SET
        elif action_set == 'extended_action_set_large':
            self.action_set = EXTENDED_ACTION_SET_LARGE
        self.action_list = np.array(self.action_set, dtype=np.intc)  # DMLAB requires intc type for actions

        self.last_observation = None

        self.action_space = gym.spaces.Discrete(len(self.action_set))

        self.observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8)
        )
        self.observation_space.spaces[self.instructions_observation] = gym.spaces.Box(
            low=0, high=DMLAB_VOCABULARY_SIZE, shape=[DMLAB_MAX_INSTRUCTION_LEN], dtype=np.int32,
        )

        self.seed()

    def seed(self, seed=None):
        if seed is None:
            initial_seed = random.randint(0, int(1e9))
        else:
            initial_seed = seed

        self.random_state = np.random.RandomState(seed=initial_seed)
        return [initial_seed]

    def format_obs_dict(self, env_obs_dict):
        """We traditionally uses 'obs' key for the 'main' observation."""

        env_obs_dict['obs'] = env_obs_dict.pop(self.main_observation)

        instr = env_obs_dict.get(self.instructions_observation)
        self.instructions[:] = 0
        if instr is not None:
            instr_words = instr.split()
            for i, word in enumerate(instr_words):
                self.instructions[i] = string_to_hash_bucket(word, DMLAB_VOCABULARY_SIZE)

            env_obs_dict[self.instructions_observation] = self.instructions

        return env_obs_dict

    def reset(self):
        if self.use_level_cache:
            self.last_reset_seed = self.cache.get_unused_seed(self.level, self.random_state)
        else:
            self.last_reset_seed = self.random_state.randint(0, 2 ** 31 - 1)

        self.dmlab.reset(seed=self.last_reset_seed)
        self.last_observation = self.format_obs_dict(self.dmlab.observations())
        self.episodic_reward = 0
        return self.last_observation

    def step(self, action):

        reward = self.dmlab.step(self.action_list[action], num_steps=self.action_repeat)
        done = not self.dmlab.is_running()

        self.episodic_reward += reward
        info = {'num_frames': self.action_repeat}

        if not done:
            obs_dict = self.format_obs_dict(self.dmlab.observations())
            self.last_observation = obs_dict
        if done:
            self.reset()

        return self.last_observation, reward, done, info


    def close(self):
        self.dmlab.close()


    def fetch(self, key, pk3_path):
        if not self.env_uses_level_cache:
            self.env_uses_level_cache = True

        path = os.path.join(self.level_cache_path, key)

        if os.path.isfile(path):
            shutil.copyfile(path, pk3_path)
            return True
        else:
            log.warning('Cache miss in environment %s key: %s!', self.level_name, key)
            return False

    def write(self, key, pk3_path):
        log.debug('Add new level to cache! Level %s seed %r key %s', self.level_name, self.last_reset_seed, key)
        self.cache.add_new_level(self.level, self.last_reset_seed, key, pk3_path)
