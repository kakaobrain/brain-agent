import torch
import psutil
from collections import OrderedDict
from brain_agent.utils.logger import log
from faster_fifo import Full, Empty

class TaskType:
    INIT, TERMINATE, RESET, ROLLOUT_STEP, POLICY_STEP, TRAIN, INIT_MODEL, EMPTY = range(8)

def dict_of_lists_append(dict_of_lists, new_data, index):
    for key, x in new_data.items():
        if key in dict_of_lists:
            dict_of_lists[key].append(x[index])
        else:
            dict_of_lists[key] = [x[index]]

def copy_dict_structure(d):
    d_copy = type(d)()
    _copy_dict_structure_func(d, d_copy)
    return d_copy

def _copy_dict_structure_func(d, d_copy):
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            d_copy[key] = type(value)()
            _copy_dict_structure_func(value, d_copy[key])
        else:
            d_copy[key] = None

def iterate_recursively(d):
    for k, v in d.items():
        if isinstance(v, (dict, OrderedDict)):
            yield from iterate_recursively(v)
        else:
            yield d, k, v

def iter_dicts_recursively(d1, d2):
    for k, v in d1.items():
        assert k in d2

        if isinstance(v, (dict, OrderedDict)):
            yield from iter_dicts_recursively(d1[k], d2[k])
        else:
            yield d1, d2, k, d1[k], d2[k]


def slice_mems(mems_buffer, mems_dones_buffer, mems_actions_buffer, actor_idx, split_idx, env_idx, s_idx, e_idx):
    # Slice given mems buffers in a cyclic queue manner
    if s_idx > e_idx:
        mems = torch.cat(
            [mems_buffer[actor_idx, split_idx, env_idx, s_idx:],
             mems_buffer[actor_idx, split_idx, env_idx, :e_idx]])
        mems_dones = torch.cat(
            [mems_dones_buffer[actor_idx, split_idx, env_idx, s_idx:],
             mems_dones_buffer[actor_idx, split_idx, env_idx, :e_idx]])
        mems_actions = torch.cat(
            [mems_actions_buffer[actor_idx, split_idx, env_idx, s_idx:],
             mems_actions_buffer[actor_idx, split_idx, env_idx, :e_idx]])
    else:
        mems = mems_buffer[actor_idx, split_idx, env_idx, s_idx:e_idx]
        mems_dones = mems_dones_buffer[actor_idx, split_idx, env_idx, s_idx:e_idx]
        mems_actions = mems_actions_buffer[actor_idx, split_idx, env_idx, s_idx:e_idx]
    return mems, mems_dones, mems_actions

def join_or_kill(process, timeout=1.0):
    process.join(timeout)
    if process.is_alive():
        log.warning('Process %r could not join, kill it with fire!', process)
        process.kill()
        log.warning('Process %r is dead (%r)', process, process.is_alive())

def set_process_cpu_affinity(worker_idx, num_workers, local_rank=0, nproc_per_node=0):
    curr_process = psutil.Process()
    available_cores = curr_process.cpu_affinity()
    cpu_count = len(available_cores)
    if nproc_per_node > 1:
        worker_idx = worker_idx * nproc_per_node + local_rank
        num_workers = num_workers * nproc_per_node
    core_indices = cores_for_worker_process(worker_idx, num_workers, cpu_count)
    if core_indices is not None:
        curr_process_cores = [available_cores[c] for c in core_indices]
        curr_process.cpu_affinity(curr_process_cores)

    log.debug('Worker %d uses CPU cores %r', worker_idx, curr_process.cpu_affinity())

def cores_for_worker_process(worker_idx, num_workers, cpu_count):
    """
    Returns core indices, assuming available cores are [0, ..., cpu_count).
    If this is not the case (e.g. SLURM) use these as indices in the array of actual available cores.
    """

    worker_idx_modulo = worker_idx % cpu_count

    cores = None
    whole_workers_per_core = num_workers // cpu_count
    if worker_idx < whole_workers_per_core * cpu_count:
        cores = [worker_idx_modulo]
    else:
        remaining_workers = num_workers % cpu_count
        if cpu_count % remaining_workers == 0:
            cores_to_use = cpu_count // remaining_workers
            cores = list(range(worker_idx_modulo * cores_to_use, (worker_idx_modulo + 1) * cores_to_use, 1))

    return cores

def safe_put(q, msg, attempts=3, queue_name=''):
    safe_put_many(q, [msg], attempts, queue_name)


def safe_put_many(q, msgs, attempts=3, queue_name=''):
    for attempt in range(attempts):
        try:
            q.put_many(msgs)
            return
        except Full:
            log.warning('Could not put msgs to queue, the queue %s is full! Attempt %d', queue_name, attempt)

    log.error('Failed to put msgs to queue %s after %d attempts. Messages are lost!', queue_name, attempts)

def safe_get(q, timeout=1e6, msg='Queue timeout'):
    """Using queue.get() with timeout is necessary, otherwise KeyboardInterrupt is not handled."""
    while True:
        try:
            return q.get(timeout=timeout)
        except Empty:
            log.info('Queue timed out (%s), timeout %.3f', msg, timeout)