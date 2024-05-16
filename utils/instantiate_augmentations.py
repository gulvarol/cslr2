"""Functions to instantiate video and text augmentations."""
import hydra
from omegaconf import DictConfig
from torchvision.transforms import Compose


def text_augmentations(cfg: DictConfig):
    """Instantiate text augmentations"""
    augmentations = None
    if cfg.augmentation.do_swap:
        swap_words = hydra.utils.instantiate(cfg.augmentation.swap_words)
        if not cfg.augmentation.do_drop or not cfg.augmentation.do_shuffle:
            augmentations = swap_words
    if cfg.augmentation.do_drop:
        drop_words = hydra.utils.instantiate(cfg.augmentation.drop_words)
        if cfg.augmentation.do_swap and not cfg.augmentation.do_shuffle:
            augmentations = Compose([swap_words, drop_words])
        else:
            augmentations = drop_words
    if cfg.augmentation.do_shuffle:
        shuffle_words = hydra.utils.instantiate(cfg.augmentation.shuffle_words)
        if cfg.augmentation.do_swap and not cfg.augmentation.do_drop:
            augmentations = Compose([swap_words, shuffle_words])
        elif cfg.augmentation.do_drop and not cfg.augmentation.do_swap:
            augmentations = Compose([drop_words, shuffle_words])
        elif cfg.augmentation.do_drop and cfg.augmentation.do_swap:
            augmentations = Compose([swap_words, drop_words, shuffle_words])
        else:
            augmentations = shuffle_words
    return augmentations


def vid_augmentations(cfg: DictConfig):
    """Instantiate vid augmentations"""
    augmentations = None
    if cfg.augmentation.do_frame_drop:
        augmentations = hydra.utils.instantiate(cfg.augmentation.frame_drop)
    return augmentations
