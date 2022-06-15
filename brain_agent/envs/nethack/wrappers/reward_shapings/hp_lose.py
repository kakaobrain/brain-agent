from brain_agent.envs.nethack.wrappers.reward_shapings.base import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.utils.utils import AttrDict
from brain_agent.envs.nethack.ids.action_ids import ActionIds
from brain_agent.envs.nethack.ids.blstat_ids import BLStatIds
from brain_agent.envs.nethack.ids.obj_ids import ObjClasses

class RewardShapingWrapper(BaseRewardShapingWrapper):
    '''
        Implement your own reward function or coeffs in here.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs  = AttrDict(
            HP_DIFF = 0.5,
            AC_DIFF = 20.0,
            HUNGER_DIFF = 10.0,
            CAP_DIFF = 10.0,
            EXP_DIFF = 1.0,
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
            HP_LOSE = 0.1,
        )


    def _get_hp_lose(self, action, obs):

        if self.coeffs.HP_LOSE == 0:
            return 0

        hp_lose = 1.0 - obs['blstats'][BLStatIds.HP] / obs['blstats'][BLStatIds.MAX_HP]

        return hp_lose

    def _get_shaped_reward(self, action, obs, done):
        hp_loss_reward = self.coeffs.HP_LOSE * self._get_hp_lose(action, obs)
        # print(hp_loss_reward)
        shaped_reward = super()._get_shaped_reward(action, obs, done) - hp_loss_reward
        return shaped_reward

