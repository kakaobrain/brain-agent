import sys
import time
import numpy as np
from faster_fifo import Queue, Empty
from tensorboardX import SummaryWriter
from brain_agent.core.actor_worker import ActorWorker
from brain_agent.core.policy_worker import PolicyWorker
from brain_agent.core.shared_buffer import SharedBuffer
from brain_agent.utils.cfg import Configs
from brain_agent.utils.utils import get_log_path, dict_of_list_put, AttrDict, get_summary_dir
from brain_agent.core.core_utils import TaskType
from brain_agent.utils.logger import log, init_logger
from brain_agent.envs.env_utils import create_env

def main():

    cfg = Configs.get_defaults()
    cfg = Configs.override_from_file_name(cfg)
    cfg = Configs.override_from_cli(cfg)

    cfg_str = Configs.to_yaml(cfg)
    cfg = Configs.to_attr_dict(cfg)

    init_logger(cfg.log.log_level, get_log_path(cfg))

    log.info(f'Experiment configuration:\n{cfg_str}')

    tmp_env = create_env(cfg, env_config=None)
    action_space = tmp_env.action_space
    obs_space = tmp_env.observation_space
    level_info = tmp_env.level_info
    num_levels = level_info['num_levels']
    tmp_env.close()

    assert cfg.test.is_test
    assert cfg.actor.num_workers >= level_info['num_levels']

    shared_buffer = SharedBuffer(cfg, obs_space, action_space)
    shared_buffer.stop_experience_collection.fill_(False)

    policy_worker_queue = Queue()
    actor_worker_queues = [Queue(2 * 1000 * 1000) for _ in range(cfg.actor.num_workers)]
    policy_queue = Queue()
    report_queue = Queue(40 * 1000 * 1000)

    policy_worker = PolicyWorker(cfg, obs_space, action_space, tmp_env.level_info, shared_buffer,
        policy_queue, actor_worker_queues, policy_worker_queue, report_queue)
    policy_worker.start_process()
    policy_worker.init()
    policy_worker.load_model()

    actor_workers = []
    for i in range(cfg.actor.num_workers):
        w = ActorWorker(cfg, obs_space, action_space, i, shared_buffer, actor_worker_queues[i], policy_queue,
                        report_queue)
        w.init()
        w.request_reset()
        actor_workers.append(w)

    writer = SummaryWriter(get_summary_dir(cfg, postfix='test'))

    stats = AttrDict()
    stats['episodic_stats'] = AttrDict()
    actor_worker_task_id = AttrDict()

    env_steps = 0
    num_collected = 0
    terminate = False

    while not terminate:
        try:
            reports = report_queue.get_many(timeout=0.1)
            for report in reports:
                if 'terminate' in report:
                    terminate = True
                if 'learner_env_steps' in report:
                    env_steps = report['learner_env_steps']
                if 'initialized_env' in report:
                    actor_idx, split_idx, _, task_id = report['initialized_env']
                    actor_worker_task_id[actor_idx] = task_id[0]
                if 'episodic_stats' in report:
                    s = report['episodic_stats']
                    level_name = s.pop('level_name')
                    level_id = s.pop('task_id')
                    for k, v in s.items():
                        tag = f'_nethack/{k}' if not 'role' in k else k
                        dict_of_list_put(stats.episodic_stats, tag, v, cfg.test.test_num_episodes)

                    true_reward = s['true_reward']
                    log.info(f'[{num_collected + 1} / {num_levels * cfg.test.test_num_episodes}] '
                             f'ture_reward: {true_reward} '
                             f'avg_true_reward: {np.mean(stats.episodic_stats["_nethack/true_reward"]):.1f}')

                    if len(stats.episodic_stats[tag]) >= cfg.test.test_num_episodes:
                        for i, w in enumerate(actor_workers):
                            if actor_worker_task_id[i] == level_id and w.process.is_alive:
                                actor_worker_queues[i].put((TaskType.TERMINATE, None))

                    num_collected = 0
                    for i, l in enumerate(level_info['all_levels']):
                        tag = '_nethack/true_reward'
                        h = stats.episodic_stats.get(tag, None)
                        if h is not None:
                            num_collected += len(h)

                    if num_collected >= num_levels * cfg.test.test_num_episodes:
                        for k, v in stats.items():
                            if k == 'episodic_stats':
                                for kk, vv in v.items():
                                    writer.add_scalar(kk, np.array(vv).mean(), env_steps)
                            else:
                                writer.add_scalar(k, np.array(v).mean(), env_steps)

                        terminate = True

        except Empty:
            time.sleep(1.0)
            pass

if __name__ == '__main__':
    sys.exit(main())