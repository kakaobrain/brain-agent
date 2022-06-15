import gym
from nle.env.tasks import NetHackChallenge, NetHackScore
from nle import nethack
from brain_agent.utils.utils import log
from typing import Union
import nle


class TerminateByChallengeRuleWrapper(gym.Wrapper):
    """
    https://www.aicrowd.com/challenges/neurips-2021-the-nethack-challenge/challenge_rules

    8-A. TIME AND TURN LIMITS
    1. During the development phase, the agent must complete 512-episode rollouts within 2hrs,
       with no single action lasting more than 300 seconds, and no single episode lasting more than 30 minutes.
       Episodes that do not reach a score of 1000 within 50,000 steps will be terminated.
       These restrictions are intended to be generous bounds to prevent abuse of the evaluation system resources,
       rather than to enforce particular efficiency constraints on agents.
    2. During the test phase, the agent must complete 4096 episode rollouts within 24hrs,
       with no single action lasting more than 300 seconds, and no single episode lasting more than 30 minutes.
       Episodes that do not reach a score of 1000 within 50,000 steps will be terminated.

    """
    def __init__(self,
                 env: Union[NetHackChallenge, NetHackScore],
                 step: int = 50000,
                 min_score: int = 1000,
                 **kwargs):
        super().__init__(env)

        nle_env = self.env
        while not isinstance(nle_env, nle.env.NLE) and hasattr(nle_env, 'env'):
            nle_env = nle_env.env
        assert isinstance(nle_env, nle.env.NLE)
        self._nle_env = nle_env

        self._step_for_rule = step
        self._min_score_for_rule = min_score

    def step(self, action: int):
        observation, reward, done, info = self.env.step(action)

        if self._nle_env._steps == self._step_for_rule and observation['blstats'][nethack.NLE_BL_SCORE] < self._min_score_for_rule:
            log.warning(f'Terminated by rule (steps: {self._nle_env._steps}, score: {observation["blstats"][nethack.NLE_BL_SCORE]})')
            info['end_status'] = 'TERMINATED_BY_RULE'
            done = True

        return observation, reward, done, info

    def reset(self, **kwargs):
        observations = self.env.reset(**kwargs)
        return observations
