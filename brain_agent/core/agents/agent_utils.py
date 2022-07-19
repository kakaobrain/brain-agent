from brain_agent.core.agents.dmlab_multitask_agent import DMLabMultiTaskAgent

def create_agent(cfg, action_space, obs_space, num_levels=1, need_half=False):
    if cfg.model.agent == 'dmlab_multitask_agent':
        agent = DMLabMultiTaskAgent(cfg, action_space, obs_space, num_levels, need_half)
    else:
        raise NotImplementedError
    return agent
