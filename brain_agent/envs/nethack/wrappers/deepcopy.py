import copy
from math import tanh

import gym
import torch
import numpy as np


class ObsDeepCopyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs = copy.deepcopy(obs)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = copy.deepcopy(obs)

        return obs, rew, done, info