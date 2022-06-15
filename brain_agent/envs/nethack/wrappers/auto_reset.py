import copy
from math import tanh

import gym
import torch
import numpy as np


class AutoResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done:
            obs = self.reset()

        return obs, rew, done, info