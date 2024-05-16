"""Functions to instantiate dataloaders with Hydra"""
from functools import partial
from multiprocessing import Value

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, get_worker_info
from tqdm import tqdm

from dataset.sentence import collate_fn_padd
from utils.instantiate_augmentations import text_augmentations, vid_augmentations


def worker_init_fn(skip_mode: bool, worker_id) -> None:
    """Worker init function."""
    info = get_worker_info()
    try:
        info.dataset.dataset.skip_mode = skip_mode
    except AttributeError:
        info.dataset.skip_mode = skip_mode


def instantiate_dataloaders(cfg: DictConfig):
    """DataLoader instantiation with Hydra config."""
    train_dataset = hydra.utils.instantiate(
        cfg.dataset,
        setname="train",
        text_augmentations=text_augmentations(cfg),
        video_augmentations=vid_augmentations(cfg),
    )
    val_dataset = hydra.utils.instantiate(
        cfg.dataset,
        setname="val",
    )

    # shuffle data if train from checkpoint
    # to avoid getting the same batches (since seeded runs are used for reproducibility)
    if cfg.checkpoint is not None:
        train_dataset.subtitles.shuffle()
        val_dataset.subtitles.shuffle()

    # if we want to train on a fraction of the data
    sampler = None
    if cfg.dataloader.train_data_fraction < 1:
        train_dataset = Subset(
            train_dataset,
            torch.randperm(
                len(train_dataset)
            )[:int(len(train_dataset) * cfg.dataloader.train_data_fraction)],
        )
    if cfg.distributed:
        assert cfg.world_size is not None and cfg.rank is not None
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=cfg.world_size, rank=cfg.rank,
        )
        cfg.dataloader.dataloader.shuffle = False
    train_skip_mode = Value("i", False)
    train_loader = hydra.utils.instantiate(
        cfg.dataloader.dataloader,
        dataset=train_dataset,
        collate_fn=collate_fn_padd,
        sampler=sampler,
        worker_init_fn=partial(worker_init_fn, train_skip_mode),
    )

    # if we want to validate on a fraction of the data
    if cfg.dataloader.val_data_fraction < 1:
        val_dataset = Subset(
            val_dataset,
            torch.randperm(
                len(val_dataset)
            )[:int(len(val_dataset) * cfg.dataloader.val_data_fraction)],
        )
    if cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=cfg.world_size, rank=cfg.rank,
        )
    val_skip_mode = Value("i", False)
    val_loader = hydra.utils.instantiate(
        cfg.dataloader.dataloader,
        dataset=val_dataset,
        collate_fn=collate_fn_padd,
        sampler=sampler,
        worker_init_fn=partial(worker_init_fn, val_skip_mode),
    )
    cfg.dataloader.N = len(train_loader)
    return train_loader, val_loader, train_skip_mode, val_skip_mode


def instantiate_vis_dataloaders(cfg: DictConfig):
    """DataLoader instantiation for visualization."""
    # save visualisation on weakly aligned subtitles
    sub_paths = cfg.paths.subtitles_path
    skip_mode = Value("i", False)
    train_dataset = hydra.utils.instantiate(
        cfg.dataset,
        setname="train",
        subtitles_path=sub_paths,
        text_augmentations=text_augmentations(cfg),
        video_augmentations=vid_augmentations(cfg),
        load_pl=False,
        load_word_embds=False,
    )
    val_dataset = hydra.utils.instantiate(
        cfg.dataset,
        setname="val",
        subtitles_path=sub_paths,
        load_pl=False,
        load_word_embds=False,
    )
    gallery_size = min(len(val_dataset), 2000)
    train_dataset = Subset(
        train_dataset,
        torch.randperm(len(train_dataset))[:gallery_size],
    )
    train_loader = hydra.utils.instantiate(
        cfg.dataloader.dataloader,
        dataset=train_dataset,
        collate_fn=collate_fn_padd,
        worker_init_fn=partial(worker_init_fn, skip_mode),
    )
    if gallery_size < len(val_dataset):
        val_dataset = Subset(
            val_dataset,
            torch.randperm(len(val_dataset))[:gallery_size],
        )
    val_loader = hydra.utils.instantiate(
        cfg.dataloader.dataloader,
        dataset=val_dataset,
        collate_fn=collate_fn_padd,
        worker_init_fn=partial(worker_init_fn, skip_mode),
    )
    return train_loader, val_loader


def instantiate_test_dataloader(cfg: DictConfig):
    """DataLoader instantiation for test."""
    sub_paths = cfg.paths.aligned_subtitles_path
    test_dataset = hydra.utils.instantiate(
        cfg.dataset,
        setname="public_test",
        subtitles_path=sub_paths,
    )
    skip_mode = Value("i", False)
    test_loader = hydra.utils.instantiate(
        cfg.dataloader.dataloader,
        dataset=test_dataset,
        collate_fn=collate_fn_padd,
        worker_init_fn=partial(worker_init_fn, skip_mode),
    )
    return test_loader

def skip_epochs(
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_skip_mode: Value,
    val_skip_mode: Value,
):
    """Skip epochs to avoid getting the same batches."""
    epoch_start = 0 if cfg.trainer.epoch_start is None else cfg.trainer.epoch_start
    if epoch_start != 0:
        # assert not cfg.dataloader.persistent_workers, \
        # "When resuming training, persistent_workers must be set to False"
        # need to go through train and val datasets for RNG
        print(f"Skipping {epoch_start} epochs (for RNG)")
        train_skip_mode.value = True
        val_skip_mode.value = True
        for _ in range(0, epoch_start):
            pbar = tqdm(iter(train_loader)) if cfg.do_print else iter(
                train_loader)
            for _, _ in enumerate(pbar):
                pass
            pbar = tqdm(iter(val_loader)) if cfg.do_print else iter(
                val_loader)
            for _, _ in enumerate(pbar):
                pass
        train_skip_mode.value = False
        val_skip_mode.value = False
    return train_loader, val_loader
