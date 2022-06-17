from brain_agent.envs.dmlab.dmlab_env import make_dmlab_env

def create_env(cfg=None, env_config=None):
    if 'dmlab' in cfg.env.name:
        env = make_dmlab_env(cfg, env_config)
    else:
        raise NotImplementedError
    return env
