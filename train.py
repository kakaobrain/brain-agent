import sys
import time
from faster_fifo import Queue, Empty
import multiprocessing
from tensorboardX import SummaryWriter
import numpy as np

from brain_agent.core.actor_worker import ActorWorker
from brain_agent.core.policy_worker import PolicyWorker
from brain_agent.core.shared_buffer import SharedBuffer
from brain_agent.core.learner_worker import LearnerWorker
from brain_agent.utils.cfg import Configs
from brain_agent.utils.logger import log, init_logger
from brain_agent.utils.utils import get_log_path, dict_of_list_put, get_summary_dir, AttrDict
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
    tmp_env.close()

    shared_buffer = SharedBuffer(cfg, obs_space, action_space)

    learner_worker_queue = Queue()
    policy_worker_queue = Queue()
    actor_worker_queues = [Queue(2 * 1000 * 1000) for _ in range(cfg.actor.num_workers)]
    policy_queue = Queue()
    report_queue = Queue(40 * 1000 * 1000)

    policy_lock = multiprocessing.Lock()
    resume_experience_collection_cv = multiprocessing.Condition()

    learner_worker = LearnerWorker(cfg, obs_space, action_space, level_info, report_queue, learner_worker_queue,
                                   policy_worker_queue,
                                   shared_buffer, policy_lock, resume_experience_collection_cv)
    learner_worker.start_process()
    learner_worker.init()

    policy_worker = PolicyWorker(cfg, obs_space, action_space, level_info, shared_buffer,
        policy_queue, actor_worker_queues, policy_worker_queue, report_queue, policy_lock, resume_experience_collection_cv)
    policy_worker.start_process() # init(), init_model() will be triggered from learner worker

    actor_workers = []
    for i in range(cfg.actor.num_workers):
        w = ActorWorker(cfg, obs_space, action_space, i, shared_buffer, actor_worker_queues[i], policy_queue,
                        report_queue, learner_worker_queue)
        w.init()
        w.request_reset()
        actor_workers.append(w)

    summary_dir = get_summary_dir(cfg=cfg)
    writer = SummaryWriter(summary_dir) if cfg.dist.world_rank == 0 else None

    # Add configuration in tensorboard
    if cfg.dist.world_rank == 0:
        cfg_str = cfg_str.replace(' ', '&nbsp;').replace('\n', '  \n')
        writer.add_text('cfg', cfg_str, 0)

    stats = AttrDict()
    stats['episodic_stats'] = AttrDict()

    last_report = time.time()
    last_env_steps = 0
    terminate = False
    reports = []

    while not terminate:
        try:
            reports.extend(report_queue.get_many(timeout=0.1))
            if time.time() - last_report > cfg.log.report_interval:
                interval = time.time() - last_report
                last_report = time.time()
                terminate, last_env_steps = process_report(cfg, reports, writer, stats, last_env_steps, level_info,
                                                           interval)
                reports = []
        except Empty:
            time.sleep(1.0)
            pass
        except Exception as e:
            log.warning(f"Exception '{e}' occured in train.py!")


def process_report(cfg, reports, writer, stats, last_env_steps, level_info, interval):
    terminate = False
    env_steps = last_env_steps

    for report in reports:
        if report is not None:
            if 'terminate' in report:
                terminate = True
            if 'learner_env_steps' in report:
                env_steps = report['learner_env_steps']
            if 'train' in report:
                s = report['train']
                for k, v in s.items():
                    dict_of_list_put(stats, f'train/{k}', v, cfg.log.num_stats_average)
            if 'episodic_stats' in report:
                s = report['episodic_stats']
                level_name = s.pop('level_name')
                level_id = s.pop('task_id')
                for k, v in s.items():
                    tag = f'_nethack/{level_id:02d}_{level_name}_{k}' if not 'role' in k else k
                    dict_of_list_put(stats.episodic_stats, tag, v, cfg.log.num_stats_average)

            fps = (env_steps - last_env_steps) / interval
            dict_of_list_put(stats, f'train/_fps', fps, cfg.log.num_stats_average)

            key_timings = ['times_learner_worker', 'times_actor_worker', 'times_policy_worker']
            for key in key_timings:
                if key in report:
                    for k, v in report[key].items():
                        tag = key+'/'+k
                        dict_of_list_put(stats, tag, v, cfg.log.num_stats_average)


    if writer is not None:
        for k, v in stats.items():
            if k == 'episodic_stats':
                for kk, vv in v.items():
                    writer.add_scalar(kk, np.array(vv).mean(), env_steps)
            else:
                writer.add_scalar(k, np.array(v).mean(), env_steps)

    if env_steps >= cfg.optim.train_for_env_steps:
        terminate = True
    return terminate, env_steps


if __name__ == '__main__':
    sys.exit(main())
