"""Functions to instantiate models with Hydra"""
import hydra
import torch
import torch.nn as nn

from omegaconf import DictConfig


def instantiate_model(cfg: DictConfig) -> nn.Module:
    """Instantiate model from config."""
    model = hydra.utils.instantiate(cfg.model.cslr2)
    return model


def handle_model_freeze(
    model: nn.Module,
    cfg: DictConfig,
) -> nn.Module:
    """Freeze model parameters according to options in the config."""
    # freeze generator if not training with SignCls (avoid unused parameters error)
    if cfg.loss.lda_sign_cls == 0:
        for name, param in model.named_parameters():
            if "generator" in name and "text_encoder" not in name:
                param.requires_grad = False
    # freeze transformer if specified
    if cfg.model.freeze_transformer:
        for name, param in model.named_parameters():
            if "generator" in name:
                param.requires_grad = False
    # handling same text and video ll
    if cfg.model.cslr2.same_text_ll:
        for name, param in model.named_parameters():
            if "text_word_ll" in name:
                param.requires_grad = False
    if cfg.model.cslr2.same_video_ll:
        for name, param in model.named_parameters():
            if "video_token_ll" in name:
                param.requires_grad = False
    # freeze text encoder
    for name, param in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad = False
    return model


def load_checkpoint(
    cfg: DictConfig,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    device: torch.device,
):
    """Load checkpoint if specified in the config."""
    if cfg.checkpoint is not None:
        # load checkpoint
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        # remove module. from checkpoint keys
        model_state_dict = checkpoint["model_state_dict"]
        model_state_dict = {
            k.replace("module.", ""): v for k, v in model_state_dict.items()}
        # to prevent errors when text encoder is not in chkpt
        model.load_state_dict(model_state_dict, strict=False)
        if "optimizer_state_dict" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if "epoch" in checkpoint:
            cfg.trainer.epoch_start = checkpoint["epoch"]
        print(f"Loaded checkpoint from {cfg.checkpoint}")
    return model, opt
