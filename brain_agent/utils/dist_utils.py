import os
import collections

import torch
from brain_agent.utils.logger import log

DistEnv = collections.namedtuple('DistEnv', ['world_size', 'world_rank', 'local_rank', 'num_gpus', 'master'])


def dist_init(cfg):
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1:
        log.debug('[dist] Distributed: wait dist process group:%d', cfg.dist.local_rank)
        torch.distributed.init_process_group(backend=cfg.dist.dist_backend, init_method='env://',
                                world_size=int(os.environ['WORLD_SIZE']))
        assert (int(os.environ['WORLD_SIZE']) == torch.distributed.get_world_size())
        log.debug('[dist] Distributed: success device:%d (%d/%d)',
                    cfg.dist.local_rank, torch.distributed.get_rank(), torch.distributed.get_world_size())
        distenv = DistEnv(torch.distributed.get_world_size(), torch.distributed.get_rank(), cfg.dist.local_rank, 1, torch.distributed.get_rank() == 0)
    else:
        log.debug('[dist] Single processed')
        distenv = DistEnv(1, 0, 0, torch.cuda.device_count(), True)
    log.debug('[dist] %s', distenv)
    return distenv


def dist_all_reduce_gradient(model):
    torch.distributed.barrier()
    world_size = float(torch.distributed.get_world_size())
    for p in model.parameters():
        if type(p.grad) is not type(None):
            torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
            p.grad.data /= world_size


def dist_reduce_gradient(model, grads=None):
    torch.distributed.barrier()
    world_size = float(torch.distributed.get_world_size())
    if grads is None:
        for p in model.parameters():
            if type(p.grad) is not type(None):
                torch.distributed.reduce(p.grad.data, 0, op=torch.distributed.ReduceOp.SUM)
                p.grad.data /= world_size
    else:
        for grad in grads:
            if type(grad) is not type(None):
                torch.distributed.reduce(grad.data, 0, op=torch.distributed.ReduceOp.SUM)
                grad.data /= world_size


def dist_all_reduce_buffers(model):
    torch.distributed.barrier()
    world_size = float(torch.distributed.get_world_size())
    for n, b in model.named_buffers():
        torch.distributed.all_reduce(b.data, op=torch.distributed.ReduceOp.SUM)
        b.data /= world_size


def dist_broadcast_model(model):
    torch.distributed.barrier()
    for _, param in model.state_dict().items():
        torch.distributed.broadcast(param, 0)
    torch.distributed.barrier()
    torch.cuda.synchronize()