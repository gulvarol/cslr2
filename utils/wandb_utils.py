"""
Utility functions for wandb
"""
from typing import Union

import wandb
from numpy import ndarray
from omegaconf import DictConfig, OmegaConf
from torch import Tensor


def wandb_run_name(cfg: DictConfig) -> str:
    """
    Setup wandb run name with some hyperparameters
    """
    run_name = cfg.run_name
    return run_name


def wandb_setup(cfg: DictConfig, setname: str="cslr2") -> None:
    """
    Initialize wandb
    """
    if cfg.do_print:
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            project=setname,
            dir=cfg.paths.log_dir,
            name=wandb_run_name(cfg),
        )


def log_retrieval_performances(
    train_v2t: Union[Tensor, ndarray],
    train_t2v: Union[Tensor, ndarray],
    val_v2t: Union[Tensor, ndarray],
    val_t2v: Union[Tensor, ndarray],
    epoch: int,
    pl_as_subtitles: bool = False,
):
    """
    Log retrieval performances
    """
    prefix = "pl_as_subtitles_" if pl_as_subtitles else ""
    wandb.log({
        f"{prefix}train_v2t_R1": train_v2t["R1"],
        f"{prefix}train_v2t_R5": train_v2t["R5"],
        f"{prefix}train_v2t_R10": train_v2t["R10"],
        f"{prefix}train_v2t_R50": train_v2t["R50"],
        f"{prefix}train_v2t_medr": train_v2t["MedR"],
        f"{prefix}train_v2t_meanr": train_v2t["MeanR"],
        f"{prefix}train_v2t_geometric_mean_R1-R5-R10": train_v2t["geometric_mean_R1-R5-R10"],
        f"{prefix}train_t2v_R1": train_t2v["R1"],
        f"{prefix}train_t2v_R5": train_t2v["R5"],
        f"{prefix}train_t2v_R10": train_t2v["R10"],
        f"{prefix}train_t2v_R50": train_t2v["R50"],
        f"{prefix}train_t2v_medr": train_t2v["MedR"],
        f"{prefix}train_t2v_meanr": train_t2v["MeanR"],
        f"{prefix}train_t2v_geometric_mean_R1-R5-R10": train_t2v["geometric_mean_R1-R5-R10"],
        f"{prefix}val_v2t_R1": val_v2t["R1"],
        f"{prefix}val_v2t_R5": val_v2t["R5"],
        f"{prefix}val_v2t_R10": val_v2t["R10"],
        f"{prefix}val_v2t_R50": val_v2t["R50"],
        f"{prefix}val_v2t_medr": val_v2t["MedR"],
        f"{prefix}val_v2t_meanr": val_v2t["MeanR"],
        f"{prefix}val_v2t_geometric_mean_R1-R5-R10": val_v2t["geometric_mean_R1-R5-R10"],
        f"{prefix}val_t2v_R1": val_t2v["R1"],
        f"{prefix}val_t2v_R5": val_t2v["R5"],
        f"{prefix}val_t2v_R10": val_t2v["R10"],
        f"{prefix}val_t2v_R50": val_t2v["R50"],
        f"{prefix}val_t2v_medr": val_t2v["MedR"],
        f"{prefix}val_t2v_meanr": val_t2v["MeanR"],
        f"{prefix}val_t2v_geometric_mean_R1-R5-R10": val_t2v["geometric_mean_R1-R5-R10"],
        "epoch": epoch,
    })


def log_test_retrieval_performances(
    test_v2t: Union[Tensor, ndarray],
    test_t2v: Union[Tensor, ndarray],
    epoch: int,
):
    """
    Log Retrieval Performances (for the test set)
    """
    wandb.log({
        "test_v2t_R1": test_v2t["R1"],
        "test_v2t_R5": test_v2t["R5"],
        "test_v2t_R10": test_v2t["R10"],
        "test_v2t_R50": test_v2t["R50"],
        "test_v2t_medr": test_v2t["MedR"],
        "test_v2t_meanr": test_v2t["MeanR"],
        "test_v2t_geometric_mean_R1-R5-R10": test_v2t["geometric_mean_R1-R5-R10"],
        "test_t2v_R1": test_t2v["R1"],
        "test_t2v_R5": test_t2v["R5"],
        "test_t2v_R10": test_t2v["R10"],
        "test_t2v_R50": test_t2v["R50"],
        "test_t2v_medr": test_t2v["MedR"],
        "test_t2v_meanr": test_t2v["MeanR"],
        "test_t2v_geometric_mean_R1-R5-R10": test_t2v["geometric_mean_R1-R5-R10"],
        "epoch": epoch,
    })
