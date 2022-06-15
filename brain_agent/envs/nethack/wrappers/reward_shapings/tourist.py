from brain_agent.envs.nethack.wrappers.reward_shapings.base import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.envs.nethack.wrappers.reward_shapings.base_2 import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.utils.utils import AttrDict
from brain_agent.envs.nethack.ids.blstat_ids import BLStatIds
from brain_agent.envs.nethack.ids.obj_ids import ObjClasses

class RewardShapingWrapper(BaseRewardShapingWrapper):
    '''
        Implement your own reward function or coeffs in here.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coeffs.update(
            AttrDict(
                WIELD_WEAPON = 20.0,
            )
        )

    def _get_wield_weapn(self, action, obs):
        str_weapon_prev = self.obs_prev['inv_strs'][self.obs_prev['inv_oclasses'] == ObjClasses.WEAPON].tobytes().decode('latin_1')
        str_weapon = obs['inv_strs'][obs['inv_oclasses'] == ObjClasses.WEAPON].tobytes().decode('latin_1')
        wield_weapon = (
            'weapon in hand' not in str_weapon_prev
            and 'weapon in hand' in str_weapon
        )
        return float(wield_weapon)

    def _get_shaped_reward(self, action, obs, done):
        shaped_reward = super()._get_shaped_reward(action, obs, done)
        shaped_reward = (
                shaped_reward
                + self.coeffs.WIELD_WEAPON * self._get_wield_weapn(action, obs)
        )
        return shaped_reward
