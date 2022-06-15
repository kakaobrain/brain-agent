from collections import deque
import enum
import numpy as np

from brain_agent.utils.utils import log

from brain_agent.utils.utils import log, static_vars
from brain_agent.envs.nethack.wrappers import (
    NetHackRewardShapingWrapper,
    NetHackDashboardWrapper,
    TerminateByChallengeRuleWrapper,
    MessageVocabEmbeddingWrapper,
    MinimalObservationsWrapper,
    RAW_SCORE_SUMMARY_KEY_SUFFIX
)

from gym.wrappers import TimeLimit
import gym
import aicrowd_gym

# from nle.nethack import *  # noqa: F403
from brain_agent.envs.nethack.nethack_model import nethack_register_models

# flake8: noqa: F405

# TODO: import this from NLE again
NUM_OBJECTS = 453
MAXEXPCHARS = 9

NETHACK_INITIALIZED = False

NETHACK_ROLES = ['archeologist', 'barbarian', 'cave', 'healer', 'knight', 'monk', 'priest', 'ranger', 'rogue', 'samurai', 'tourist', 'valkyrie', 'wizard']

EXTRA_EPISODIC_STATS_PROCESSING = []
EXTRA_PER_POLICY_SUMMARIES = []
EXTRA_EPISODIC_TEST_STATS_PROCESSING = []
EXTRA_TEST_SUMMARIES = []


class NetHackSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.level = env_id
        # self.default_timeout = default_timeout
        self.has_timer = False


NetHack_ENVS = []

NetHack_ENVS.append(NetHackSpec('nethack_challenge', 'NetHackChallenge-v0'))
NetHack_ENVS.append(NetHackSpec('nethack_score', 'NetHackScore-v0'))


def nethack_env_by_name(name):
    for cfg in NetHack_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown NetHack env')


# noinspection PyUnusedLocal
def make_nethack_env(cfg, env_config, is_submission=False, **kwargs):
    ensure_initialized(cfg, cfg.env.name)
    nethack_spec = nethack_env_by_name(cfg.env.name)

    num_envs = len([nethack_spec.level])
    cfg.num_envs = num_envs

    task_id = get_task_id(env_config, nethack_spec, cfg)
    level = task_id_to_level(task_id, nethack_spec)
    if is_submission:
        env = aicrowd_gym.make("NetHackChallenge-v0", savedir=cfg.get('record_to'))
    else:
        env = gym.make("NetHackChallenge-v0", savedir=cfg.get('record_to'))
    env.task_id = task_id
    env.level_name = level

    if not cfg.test.is_test:
        env._max_episode_steps = cfg.env.max_num_steps
        env.character = '@'
        env._observation_keys = (
            "glyphs",
            "chars",
            "colors",
            "specials",
            "blstats",
            "message",
            "tty_chars",
            "tty_colors",
            "tty_cursor",
            "inv_glyphs",
            "inv_strs",
            "inv_letters",
            "inv_oclasses",
        )
        env.penalty_step = cfg.env.penalty_step
        env.penalty_time = cfg.env.penalty_time
        env.penalty_mode = cfg.env.fn_penalty_step

    # DeepCopy is used since tty_chars, tty_colors has static id: saved tty_chars in previous step may change as step goes on
    from brain_agent.envs.nethack.wrappers.deepcopy import ObsDeepCopyWrapper
    env = ObsDeepCopyWrapper(env)

    if not is_submission:
        from brain_agent.envs.nethack.wrappers.fixed_role import FixedRoleWrapper
        env = FixedRoleWrapper(env, cfg.env.fixed_role)

    from brain_agent.envs.nethack.wrappers.extra_stats import ExtraStatsWrapper
    env = ExtraStatsWrapper(env)
    
    if cfg.env.get('block_kill_pet', False) or cfg.get('hide_pet', False):
        from brain_agent.envs.nethack.wrappers.action_masking import RuleBasedActionMaskingWrapper
        env = RuleBasedActionMaskingWrapper(env, cfg)

    if cfg.env.get('use_action_overrider', None) is True:
        from brain_agent.envs.nethack.wrappers.action_overriding import ActionOverridingWrapper
        env = ActionOverridingWrapper(env, cfg)

    if cfg.env.get('use_character_feature', None):
        from brain_agent.envs.nethack.wrappers.character_feature import CharacterFeatureWrapper
        env = CharacterFeatureWrapper(env)

    if cfg.env.get('use_avail_atype', None):
        from brain_agent.envs.nethack.wrappers.action_masking import ActionMaskingWrapper
        env = ActionMaskingWrapper(env, cfg)

    if cfg.env.get('use_item_feature', None):
        from brain_agent.envs.nethack.wrappers.item_feature import ItemFeatureWrapper
        env = ItemFeatureWrapper(env, cfg)

    if cfg.env.get('use_spell_feature', None):
        from brain_agent.envs.nethack.wrappers.spell_feature import SpellFeatureWrapper
        env = SpellFeatureWrapper(env)

    if not is_submission and cfg.test.is_test:
        env = TerminateByChallengeRuleWrapper(env)

        if cfg.get('max_num_steps', None):
            env = TimeLimit(env, max_episode_steps=cfg.max_num_steps)

        if cfg.get('record_to', None):
            env = NetHackDashboardWrapper(env)

    if cfg.env.use_separated_action:
        from brain_agent.envs.nethack.wrappers.seperated_action import SeperatedActionWrapper
        env = SeperatedActionWrapper(env, cfg=cfg)

    if (cfg.model.encoder.get('encoder_custom', None) == 'nle_obs_vector_encoder'
        or cfg.model.encoder.get('encoder_custom', None) == 'trxli_encoder'
        or cfg.model.encoder.get('encoder_custom', None) == 'avgpool_encoder'):
        if cfg.model.encoder.encoder_subtype == 'nethack_glyph':
            from brain_agent.envs.nethack.kakaobrain.obs_wrappers import VectorFeaturesWrapper, GlyphEncoderWrapper
            env = GlyphEncoderWrapper(env)
        else:
            # https://github.com/Miffyli/nle-sample-factory-baseline/blob/49213ec2e3c713c1ba81ab290ec7e65f852a7443/env.py#L40-L42
            from brain_agent.envs.nethack.anssi.obs_wrappers import (VectorFeaturesWrapper,
                                                                     RenderCharImagesWithNumpyWrapper)
            crop_size = cfg.env.crop_size
            font_size = cfg.env.font_size
            rescale_font_size = (cfg.env.rescale_font_size, cfg.env.rescale_font_size) if cfg.env.rescale_font_size else (6, 6)
            env = RenderCharImagesWithNumpyWrapper(env, font_size=font_size, crop_size=crop_size,
                                                   rescale_font_size=rescale_font_size,
                                                   use_tty_chars_colors=cfg.env.get('use_tty_chars_colors', True))

        env = VectorFeaturesWrapper(env, cfg)

    if cfg.model.encoder.message_encoder.startswith('embedding'):
        encoder_tokenizer = {
            'embedding': 'torchtext',
            'embedding_pretrained': 'nltk',
        }
        env = MessageVocabEmbeddingWrapper(env, encoder_tokenizer[cfg.model.encoder.message_encoder])

    if cfg.env.reward_shaping is not None:
        RewardShapingWrapper = __import__(f"brain_agent.envs.nethack.wrappers.reward_shapings.{cfg.reward_shaping}", fromlist=-1).RewardShapingWrapper
        env = RewardShapingWrapper(env, cfg.reward_shaping, task_id, level)
    else:
        env = NetHackRewardShapingWrapper(env, task_id, level)

    if cfg.env.minimal_obs:
        required_obs_keys = ['obs', 'vector_obs', 'spell_feature', 'item_feature', 'action_class', 'avail_spell', 'avail_use_item', 'avail_pick_item', 'pick_item_feature', 'last_atype',
                             'message_embedding' if cfg.model.encoder.message_encoder.startswith('embedding') else 'message']
        if cfg.env.use_separated_action:
            required_obs_keys.extend(['inv_letters', 'tty_chars'])

        if cfg.env.get('use_avail_atype', False) or cfg.env.get('block_kill_pet', False):
            required_obs_keys.append('avail_atype')
        env = MinimalObservationsWrapper(env, required_obs_keys)

    from brain_agent.envs.nethack.wrappers.auto_reset import AutoResetWrapper
    env = AutoResetWrapper(env)

    env.level_info = dict(
        num_levels=1,
        all_levels=['nethack']
    )

    return env


def get_task_id(env_config, spec, cfg):
    if env_config is None:
        return 0
    elif isinstance(spec.level, str):
        return 0
    elif isinstance(spec.level, (list, tuple)):
        num_envs = len(spec.level)

        if cfg.nethack_one_task_per_worker:
            return env_config['worker_index'] % num_envs
        else:
            return env_config['env_id'] % num_envs
    else:
        raise Exception('spec level is either string or a list/tuple')


def task_id_to_level(task_id, spec):
    if isinstance(spec.level, str):
        return spec.level
    elif isinstance(spec.level, (list, tuple)):
        levels = spec.level
        level = levels[task_id]
        return level
    else:
        raise Exception('spec level is either string or a list/tuple')


def list_all_levels_for_experiment(env_name):
    spec = nethack_env_by_name(env_name)
    if isinstance(spec.level, str):
        return [spec.level]
    elif isinstance(spec.level, (list, tuple)):
        levels = spec.level
        return levels
    else:
        raise Exception('spec level is either string or a list/tuple')

@static_vars(new_level_returns=dict(), env_spec=None)
def nethack_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    if RAW_SCORE_SUMMARY_KEY_SUFFIX not in stat_key:
        return

    new_level_returns = nethack_extra_episodic_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        new_level_returns[policy_id] = dict()

    if nethack_extra_episodic_stats_processing.env_spec is None:
        nethack_extra_episodic_stats_processing.env_spec = nethack_env_by_name(cfg.env)

    task_id = int(stat_key.split('_')[1])  # this is a bit hacky but should do the job
    level = task_id_to_level(task_id, nethack_extra_episodic_stats_processing.env_spec)
    #level_name = atari_level_to_level_name(level)
    level_name = level

    if level_name not in new_level_returns[policy_id]:
        new_level_returns[policy_id][level_name] = []

    new_level_returns[policy_id][level_name].append(stat_value)


@static_vars(all_levels=None)
def nethack_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    """
    We precisely follow IMPALA repo (scalable_agent) here for the reward calculation.

    The procedure is:
    1. Calculate mean raw episode score for the last few episodes for each level
    2. Calculate human-normalized score using this mean value
    3. Calculate capped score

    The key point is that human-normalization and capping is done AFTER mean, which can lead to slighly higher capped
    scores for levels that exceed the human baseline.

    Another important point: we write the avg score summary only when we have at least one episode result for every
    level. Again, we try to precisely follow IMPALA implementation here.

    """
    new_level_returns = nethack_extra_episodic_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        return

    # exit if we don't have at least one episode for all levels
    if nethack_extra_summaries.all_levels is None:
        nethack_levels = list_all_levels_for_experiment(cfg.env)
        level_names = nethack_levels
        nethack_extra_summaries.all_levels = level_names

    all_levels = nethack_extra_summaries.all_levels
    for level in all_levels:
        if len(new_level_returns[policy_id].get(level, [])) < 256:
            return

    # level_mean_scores = []
    mean_score = 0
    median_score = 0

    for level_idx, level in enumerate(all_levels):
        level_score = new_level_returns[policy_id][level]
        assert len(level_score) > 0
        assert level_idx == 0

        # score = np.mean(level_score)
        mean_score = np.mean(level_score)
        median_score = np.median(level_score)

        # level_mean_scores.append(score)

        level_key = f'{level_idx:02d}_{level}'
        # summary_writer.add_scalar(f'_nethack/{level_key}_raw_score', score, env_steps)

    # assert len(level_mean_scores) == len(all_levels)

    # mean_score = np.mean(level_mean_scores)
    # median_score = np.median(level_mean_scores)

    # use 000 here to put these summaries on top in tensorboard (it sorts by ASCII)
    summary_writer.add_scalar(f'_nethack/000_mean_raw_score', mean_score, env_steps)
    summary_writer.add_scalar(f'_nethack/000_median_raw_score', median_score, env_steps)

    # clear the scores and start anew (this is exactly what IMPALA does)
    nethack_extra_episodic_stats_processing.new_level_returns[policy_id] = dict()

    # add a new stat that PBT can track
    target_objective_stat = 'dmlab_target_objective'
    if target_objective_stat not in policy_avg_stats:
        policy_avg_stats[target_objective_stat] = [deque(maxlen=1) for _ in range(cfg.num_policies)]

    policy_avg_stats[target_objective_stat][policy_id].append(median_score)

    return median_score

@static_vars(new_level_returns=dict())
def nethack_extra_test_stats_processing(policy_id, stat_key, stat_value, cfg):
    if RAW_SCORE_SUMMARY_KEY_SUFFIX not in stat_key:
        return

    new_level_returns = nethack_extra_test_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        new_level_returns[policy_id] = dict()

    task_id = int(stat_key.split('_')[1])  # this is a bit hacky but should do the job
    level_name = 'random_character'

    if level_name not in new_level_returns[policy_id]:
        new_level_returns[policy_id][level_name] = []

    new_level_returns[policy_id][level_name].append(stat_value)
    return task_id


@static_vars(all_levels=None)
def nethack_test_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    new_level_returns = nethack_extra_test_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        return

    # exit if we don't have at least one episode for all levels
    if nethack_test_summaries.all_levels is None:
        level_names = ['random_character']
        nethack_test_summaries.all_levels = level_names

    all_levels = nethack_test_summaries.all_levels
    for level in all_levels:
        if len(new_level_returns[policy_id].get(level, [])) < 1:
            return

    for level_idx, level in enumerate(all_levels):
        level_score = new_level_returns[policy_id][level]
        assert len(level_score) > 0
        if len(level_score) < cfg.test_num_episodes:
            return

    for level_idx, level in enumerate(all_levels):
        level_score = new_level_returns[policy_id][level]
        level_score = np.array(level_score[:cfg.test_num_episodes])

        mean_score = np.mean(level_score)
        std_score = np.std(level_score)
        median_score = np.median(level_score)

        level_key = f'{level_idx:02d}_{level}'
        summary_writer.add_scalar(f'_nethack/{level_key}_mean_score', mean_score, env_steps)
        summary_writer.add_scalar(f'_nethack/{level_key}_std_score', std_score, env_steps)
        summary_writer.add_scalar(f'_nethack/{level_key}_median_score', median_score, env_steps)
        log.debug('Policy %d %s mean_score: %f', cfg.test_policy_id[policy_id], level, mean_score)
        log.debug('Policy %d %s std_score: %f', cfg.test_policy_id[policy_id], level, std_score)
        log.debug('Policy %d %s median_score: %f', cfg.test_policy_id[policy_id], level, median_score)
        log.debug('Policy %d %s scores: %r', cfg.test_policy_id[policy_id], level, level_score)

    return policy_id


def ensure_initialized(cfg, env_name):
    global NETHACK_INITIALIZED
    if NETHACK_INITIALIZED:
        return

    nethack_register_models()

    if 'nethack' in env_name:
        EXTRA_EPISODIC_STATS_PROCESSING.append(nethack_extra_episodic_stats_processing)
        EXTRA_PER_POLICY_SUMMARIES.append(nethack_extra_summaries)

    if cfg.test.is_test:
        EXTRA_EPISODIC_TEST_STATS_PROCESSING.append(nethack_extra_test_stats_processing)
        EXTRA_TEST_SUMMARIES.append(nethack_test_summaries)

    NETHACK_INITIALIZED = True
