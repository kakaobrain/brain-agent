import numpy as np
import nle.nethack as nh

from brain_agent.envs.nethack.wrappers.reward_shapings.base import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.envs.nethack.wrappers.reward_shapings.base_2 import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.utils.utils import AttrDict

class RewardShapingWrapper(BaseRewardShapingWrapper):
    '''
        Implement your own reward function or coeffs in here.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs.update(
            AttrDict(
                HP_RATIO_DIFF = 20.0,
                MAX_DEPTH = 0.2,
                MISSING_PET = 0.1,
            )
        )

    def _get_missing_pet(self, action, obs):
        RANGE_PET = 5
        check1_pet = obs['glyphs'] > nh.GLYPH_PET_OFF
        check2_pet = obs['glyphs'] < nh.GLYPH_INVIS_OFF
        is_pet = np.logical_and(check1_pet, check2_pet)
        y_pet, x_pet = np.where(is_pet)
        x_agent, y_agent = obs['blstats'][:2]

        missing_pet = True
        if x_pet.shape[0] == 1 and y_pet.shape[0] == 1:  # if pet is visible
            missing_pet = max(abs(x_agent - x_pet), abs(y_agent - y_pet)) > RANGE_PET

        return float(missing_pet)

    def _get_shaped_reward(self, action, obs, done):
        shaped_reward = super()._get_shaped_reward(action, obs, done)
        shaped_reward = (
            shaped_reward
            - self.coeffs.MISSING_PET * self._get_missing_pet(action, obs)
        )
        return shaped_reward

