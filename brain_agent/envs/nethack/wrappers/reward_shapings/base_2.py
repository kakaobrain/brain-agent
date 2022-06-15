from brain_agent.envs.nethack.wrappers.reward_shapings.base import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.utils.utils import AttrDict
from brain_agent.envs.nethack.ids.blstat_ids import BLStatIds

class RewardShapingWrapper(BaseRewardShapingWrapper):
    '''
        Implement your own reward function or coeffs in here.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs.update(
            AttrDict(
                HP_DIFF = 0.5,
                HP_RATIO_DIFF = 0.0,
                AC_DIFF = 20.0,
                HUNGER_DIFF = 10.0,
                CAP_DIFF = 20.0,
                EXP_DIFF = 0.0,
                EXP_BONUS=1.5,
                LEVEL_DIFF = 20.0,
                FOOD_DIFF = 5.0,
                GOLD_DIFF = 0.0,
                ITEM_DIFF = 1.0,
                PET_KILL = 30.0,
                DONE = 100.0,
                AVOID = 100.0,
                REVEAL = 0.005,
                MAX_DEPTH = 0.0,
                IN_SOKOVAN = 0.0,
                TICK_PENALTY = 0.00,
            )
        )

    def _get_hp_ratio_diff(self, action, obs):
        # good for increment
        hp_ratio_diff = (obs['blstats'][BLStatIds.HP] - self.obs_prev['blstats'][BLStatIds.HP]) / self.obs_prev['blstats'][BLStatIds.MAX_HP]

        return hp_ratio_diff

    def _get_shaped_reward(self, action, obs, done):
        shaped_reward = super()._get_shaped_reward(action, obs, done)
        shaped_reward = (
                shaped_reward
                - self.coeffs.HP_RATIO_DIFF * self._get_hp_ratio_diff(action, obs)
                - self.coeffs.TICK_PENALTY
        )
        return shaped_reward
