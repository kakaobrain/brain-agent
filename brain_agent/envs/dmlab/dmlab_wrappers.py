import gym
import numpy as np
from gym import spaces, ObservationWrapper
from math import tanh
from brain_agent.envs.dmlab.dmlab30 import RANDOM_SCORES, HUMAN_SCORES


def has_image_observations(observation_space):
    return len(observation_space.shape) >= 2

def compute_hns(r, h, s):
    return (s-r) / (h-r) * 100

class PixelFormatChwWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if isinstance(env.observation_space, gym.spaces.Dict):
            img_obs_space = env.observation_space['obs']
            self.dict_obs_space = True
        else:
            img_obs_space = env.observation_space
            self.dict_obs_space = False

        if not has_image_observations(img_obs_space):
            raise Exception('Pixel format wrapper only works with image-based envs')

        obs_shape = img_obs_space.shape
        max_num_img_channels = 4

        if len(obs_shape) <= 2:
            raise Exception('Env obs do not have channel dimension?')

        if obs_shape[0] <= max_num_img_channels:
            raise Exception('Env obs already in CHW format?')

        h, w, c = obs_shape
        low, high = img_obs_space.low.flat[0], img_obs_space.high.flat[0]
        new_shape = [c, h, w]

        if self.dict_obs_space:
            dtype = env.observation_space.spaces['obs'].dtype if env.observation_space.spaces['obs'].dtype is not None else np.float32
        else:
            dtype = env.observation_space.dtype if env.observation_space.dtype is not None else np.float32

        new_img_obs_space = spaces.Box(low, high, shape=new_shape, dtype=dtype)

        if self.dict_obs_space:
            self.observation_space = env.observation_space
            self.observation_space.spaces['obs'] = new_img_obs_space
        else:
            self.observation_space = new_img_obs_space

        self.action_space = env.action_space

    @staticmethod
    def _transpose(obs):
        return np.transpose(obs, (2, 0, 1))

    def observation(self, observation):
        if observation is None:
            return observation

        if self.dict_obs_space:
            observation['obs'] = self._transpose(observation['obs'])
        else:
            observation = self._transpose(observation)
        return observation

class EpisodicStatWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.raw_episode_return = self.episode_return = self.episode_length = 0

    def reset(self):
        obs = self.env.reset()
        self.raw_episode_return = self.episode_return = self.episode_length = 0
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.episode_return += rew
        self.raw_episode_return += info.get('raw_rew', rew)
        self.episode_length += info.get('num_frames', 1)

        if done:
            level_name = self.unwrapped.level_name
            hns = compute_hns(
                RANDOM_SCORES[level_name.replace('train', 'test')],
                HUMAN_SCORES[level_name.replace('train', 'test')],
                self.raw_episode_return
            )

            info['episodic_stats'] = {
                'level_name': self.unwrapped.level_name,
                'task_id': self.unwrapped.task_id,
                'episode_return': self.episode_return,
                'episode_length': self.episode_length,
                'raw_episode_return': self.raw_episode_return,
                'hns': hns,
            }
            self.episode_return = 0
            self.raw_episode_return = 0
            self.episode_length = 0

        return obs, rew, done, info

class RewardShapingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        info['raw_rew'] = rew
        squeezed = tanh(rew / 5.0)
        clipped = (1.5 * squeezed) if rew < 0.0 else (5.0 * squeezed)
        rew = clipped

        return obs, rew, done, info
