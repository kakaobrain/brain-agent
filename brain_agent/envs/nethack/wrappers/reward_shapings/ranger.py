from brain_agent.envs.nethack.wrappers.reward_shapings.base import RewardShapingWrapper as BaseRewardShapingWrapper
from brain_agent.utils.utils import AttrDict

class RewardShapingWrapper(BaseRewardShapingWrapper):
    '''
        Implement your own reward function or coeffs in here.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
