import os
from datetime import datetime
from brain_agent.utils.logger import log
import glob

def dict_of_list_put(d, k, v, max_len=100):
    if d.get(k) is None:
        d[k] = []
    d[k].append(v)
    if len(d[k]) > max_len:
        d[k].pop(0)

def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = dict()

    for d in list_of_dicts:
        for key, x in d.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []

            dict_of_lists[key].append(x)

    return dict_of_lists

def get_checkpoint_dir(cfg):
    checkpoint_dir = os.path.join(get_experiment_dir(cfg=cfg), f'checkpoint')
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_checkpoints(checkpoints_dir):
    checkpoints = glob.glob(os.path.join(checkpoints_dir, 'checkpoint_*'))
    return sorted(checkpoints)


def get_experiment_dir(cfg):
    exp_dir = os.path.join(cfg.train_dir, cfg.experiment)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def get_log_path(cfg):
    exp_dir = os.path.join(cfg.train_dir, cfg.experiment)
    os.makedirs(exp_dir, exist_ok=True)
    log_dir = os.path.join(cfg.train_dir, cfg.experiment, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    date = datetime.now().strftime("%Y%m%d_%I%M%S%P")
    return os.path.join(log_dir, f'log-r{cfg.dist.world_rank:02d}-{date}.txt')

def get_summary_dir(cfg, postfix=None):
    summary_dir = os.path.join(cfg.train_dir, cfg.experiment, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    if postfix is not None:
        summary_dir = os.path.join(summary_dir, postfix)
        os.makedirs(summary_dir, exist_ok=True)
    return summary_dir

class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattribute__(self, item):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    @classmethod
    def from_nested_dicts(cls, data):
        if not isinstance(data, dict):
            return data
        else:
            return cls({key: cls.from_nested_dicts(data[key]) for key in data})
