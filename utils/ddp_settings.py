"""Initialize DDP settings"""
import builtins
import os

import torch
import torch.distributed as dist
from omegaconf import DictConfig


def ddp_settings(cfg: DictConfig) -> DictConfig:
    """Initialize DDP settings and return the updated config"""
    if cfg.distributed:
        import utils.idr_torch as idr_torch
        if "SLURM_PROCID" in os.environ:
            cfg.world_size = idr_torch.size
            cfg.rank = idr_torch.rank
            cfg.local_rank = idr_torch.local_rank
            cfg.optimizer.lr = cfg.optimizer.lr * cfg.world_size if not cfg.fixed_lr \
                else cfg.optimizer.lr
        elif "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) != -1:
            # for torch.distributed.launch
            cfg.rank = int(os.environ["LOCAL_RANK"])
            cfg.local_rank = cfg.rank
            cfg.world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=cfg.world_size, rank=cfg.rank,
        )
        torch.cuda.set_device(cfg.local_rank)
    else:
        cfg.world_size, cfg.rank, cfg.local_rank = 1, 0, 0
    if cfg.rank != 0:
        print(f"Rank {cfg.rank} is muted")
        def print_pass(*args):
            pass
        builtins.print = print_pass
    return cfg
