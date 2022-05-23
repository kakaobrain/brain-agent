import os
from brain_agent.envs.dmlab.dmlab30 import DMLAB_LEVELS_BY_ENVNAME
from brain_agent.envs.dmlab.dmlab_gym import DmlabGymEnv
from brain_agent.envs.dmlab.dmlab_level_cache import dmlab_ensure_global_cache_initialized
from brain_agent.envs.dmlab.dmlab_wrappers import PixelFormatChwWrapper, EpisodicStatWrapper, RewardShapingWrapper
from brain_agent.utils.utils import get_experiment_dir
from brain_agent.utils.logger import log

DMLAB_INITIALIZED = False

def get_task_id(env_config, levels, cfg):
    if env_config is None:
        return 0

    num_envs = len(levels)

    if cfg.env.one_task_per_worker:
        return env_config['worker_index'] % num_envs
    else:
        return env_config['env_id'] % num_envs


def make_dmlab_env_impl(levels, cfg, env_config, extra_cfg=None):
    skip_frames = cfg.env.frameskip

    task_id = get_task_id(env_config, levels, cfg)
    level = levels[task_id]
    log.debug('%r level %s task id %d', env_config, level, task_id)

    env = DmlabGymEnv(
        task_id, level, skip_frames, cfg.env.res_w, cfg.env.res_h,
        cfg.env.dataset_path, cfg.env.action_set,
        cfg.env.use_level_cache, cfg.env.level_cache_path, extra_cfg,
    )
    all_levels = []
    for l in levels:
        all_levels.append(l.replace('contributed/dmlab30/', ''))

    env.level_info = dict(
        num_levels=len(levels),
        all_levels=all_levels
    )

    env = PixelFormatChwWrapper(env)

    env = EpisodicStatWrapper(env)

    env = RewardShapingWrapper(env)

    return env


def make_dmlab_env(cfg, env_config=None):
    levels = DMLAB_LEVELS_BY_ENVNAME[cfg.env.name]
    extra_cfg = None
    if cfg.test.is_test and 'test' in cfg.env.name:
        extra_cfg = dict(allowHoldOutLevels='true')
    ensure_initialized(cfg, levels)
    return make_dmlab_env_impl(levels, cfg, env_config, extra_cfg=extra_cfg)


def ensure_initialized(cfg, levels):
    global DMLAB_INITIALIZED
    if DMLAB_INITIALIZED:
        return

    level_cache_dir = cfg.env.level_cache_path
    os.makedirs(level_cache_dir, exist_ok=True)

    dmlab_ensure_global_cache_initialized(get_experiment_dir(cfg=cfg), levels, level_cache_dir)
    DMLAB_INITIALIZED = True
