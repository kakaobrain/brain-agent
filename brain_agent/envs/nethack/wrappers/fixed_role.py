import gym

import enum

import numpy as np

import nle
from nle import nethack as nh
from brain_agent.envs.nethack.utils.number_to_binary import number_to_binary

class FixedRoleWrapper(gym.Wrapper):
    """Fix the role as wizard"""

    def __init__(self, env, fixed_role):
        super().__init__(env)
        if fixed_role is None:
            self.fixed_role = 'random'
        else:
            self.fixed_role = fixed_role


    def _get_role_on_reset(self):
        obs = self.env.step(25)  # get check attribute
        str_attribute = obs[0]['tty_chars'].reshape(-1).tostring().decode('latin_1').lower()
        self.env.step(99); self.env.step(99)  # pass 2 pages of explanation by entering 'space' action

        role = None

        # Check role
        roles = [ 'archeologist', 'barbarian', 'cave', 'healer', 'knight', 'monk', 'priest', 'ranger', 'rogue', 'samurai', 'tourist', 'valkyrie', 'wizard' ]
        for _role in roles:
            if _role in str_attribute:
                role = _role
        assert not (
                role is None
        ), f"role: {role}"

        return role

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        obs = self.env.reset()
        role = self._get_role_on_reset()

        if self.fixed_role != 'random':
            while not role == self.fixed_role:
                obs = self.env.reset()
                role = self._get_role_on_reset()

        self.role = role
        self.env.unwrapped.role = role

        return obs