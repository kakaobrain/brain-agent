from brain_agent.envs.nethack.wrappers.reward_shapings.base import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.utils.utils import AttrDict

class RewardShapingWrapper(BaseRewardShapingWrapper):
    '''
        Implement your own reward function or coeffs in here.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs.update(
            AttrDict(
                HP_DIFF = 10.0,
                AC_DIFF = 20.0,
                HUNGER_DIFF = 10.0,
                CAP_DIFF = 10.0,
                EXP_DIFF = 1.0,
                EXP_BONUS=1.5,
                LEVEL_DIFF = 10.0,
                FOOD_DIFF = 5.0,
                GOLD_DIFF = 0.0,
                ITEM_DIFF = 1.0,
                PET_KILL = 30.0,
                DONE = 100.0,
                AVOID = 100.0,
                REVEAL = 0.0025,
                MAX_DEPTH = 0.0,
                IN_SOKOVAN = 0.1,
            )
        )

    def _get_diff_something(self, action, obs):
        return 0

    def _get_shaped_reward(self, action, obs, done):
        my_additional_reward = self._get_diff_something(action, obs)
        shaped_reward = super()._get_shaped_reward(action, obs, done) + my_additional_reward
        return shaped_reward

