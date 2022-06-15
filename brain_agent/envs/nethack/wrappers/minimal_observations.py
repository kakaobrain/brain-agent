
from collections import OrderedDict
import gym
from brain_agent.utils.utils import log


class MinimalObservationsWrapper(gym.Wrapper):
    def __init__(self, env, obs_keys, **kwargs):
        super().__init__(env)
        self._obs_keys = obs_keys if isinstance(obs_keys, list) else list(obs_keys)

        minimal_obs_spaces = OrderedDict()
        for key in self._obs_keys:
            if key in self.env.observation_space.spaces:
                minimal_obs_spaces[key] = self.env.observation_space[key]

        # log.info(f'minimalize observation_space {self.env.observation_space.spaces.keys()} -> {minimal_obs_spaces.keys()}')
        self.observation_space = gym.spaces.Dict(minimal_obs_spaces)

    def _minimalize_observation(self, observation):
        obs_keys = list(observation.keys())
        for key in obs_keys:
            if key not in self._obs_keys:
                del observation[key]
        return observation

    def step(self, action: int):
        try:
            result = self.env.step(action)
            observation = self._minimalize_observation(result[0])
        except Exception as e:
            log.exception(e)
            raise
        return result

    def reset(self, **kwargs):
        try:
            observation = self.env.reset(**kwargs)
            observation = self._minimalize_observation(observation)
        except Exception as e:
            log.exception(e)
            raise
        return observation

    ## for submission
    def on_reset_submission(self, env, obs, is_holding_aicrowd=False):
        self.env_aicrowd = env
        obs = self._minimalize_observation(obs)

        return obs