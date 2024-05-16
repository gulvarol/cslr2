"""
Main file to train the CSLR2 model
"""
import os
import shutil
from typing import  Optional

import humanize
import hydra
import lmdb
import pandas as pd
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as DDP

from loops.train_loop import train_loop
from loops.val_loop import val_loop
from loops.retrieval_loop import retrieval_loop
from utils.ddp_settings import ddp_settings
from utils.instantiate_dataloaders import (
    instantiate_dataloaders, instantiate_test_dataloader,
    instantiate_vis_dataloaders, skip_epochs
)
from utils.instantiate_model import (
    handle_model_freeze, instantiate_model, load_checkpoint
)
from utils.seed import setup_seed
from utils.wandb_utils import log_retrieval_performances, wandb_setup


def log_and_save(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float,
    train_sent_ret: float,
    val_sent_ret: float,
    train_sign_ret: float,
    val_sign_ret: float,
    train_sign_cls: float,
    val_sign_cls: float,
    vis_train_loader: torch.utils.data.DataLoader,
    vis_val_loader: torch.utils.data.DataLoader,
    rgb_lmdb_env: lmdb.Environment,
    cfg: DictConfig,
    epoch: int,
    best_t2v: int,
):
    """Log and save checkpoint."""
    if cfg.do_print:
        epoch_log = {
            "Split": ["Train", "Val"],
            "Loss": [train_loss, val_loss],
            "SentRet": [train_sent_ret, val_sent_ret],
            "SignRet": [train_sign_ret, val_sign_ret],
            "SignCls": [train_sign_cls, val_sign_cls],
        }
        train_v2t, train_t2v = retrieval_loop(
            model, vis_train_loader, rgb_lmdb_env, cfg, "train", epoch,
        )
        val_v2t, val_t2v = retrieval_loop(
            model, vis_val_loader, rgb_lmdb_env, cfg, "val", epoch
        )
        epoch_log["T2V R@1"] = [train_t2v["R1"], val_t2v["R1"]]
        epoch_log["T2V R@5"] = [train_t2v["R5"], val_t2v["R5"]]
        epoch_log["T2V MedR"] = [train_t2v["MedR"], val_t2v["MedR"]]
        epoch_log["V2T R@1"] = [train_v2t["R1"], val_v2t["R1"]]
        epoch_log["V2T R@5"] = [train_v2t["R5"], val_v2t["R5"]]
        epoch_log["V2T MedR"] = [train_v2t["MedR"], val_v2t["MedR"]]
        log_df = pd.DataFrame(epoch_log)
        # display as table
        print("")
        print(
            tabulate(
                log_df,
                headers="keys",
                tablefmt="presto",
                showindex="never",
                floatfmt=".2f",
            )
        )
        print("")

        model_path = cfg.paths.log_dir + \
            "/models/model_" + str(epoch + 1) + ".pth"
        if not os.path.exists(cfg.paths.log_dir + "/models/"):
            os.makedirs(cfg.paths.log_dir + "/models/")
        print(f"Saving model to {model_path}")
        model_state_dict = model.state_dict()
        torch.save(
            {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch + 1,
                "loss": train_loss,
            },
            model_path
        )
        if best_t2v <= val_t2v["R1"]:
            best_t2v = val_t2v["R1"]
            model_path = cfg.paths.log_dir + "/models/model_best.pth"
            print(f"Saving new best model to {model_path}")
            torch.save(
                {
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch + 1,
                    "loss": train_loss,
                },
                model_path
            )
        # log in wandb
        log_retrieval_performances(
            train_v2t, train_t2v, val_v2t, val_t2v, epoch
        )
        wandb.log(
            {
                "train_loss_epoch": train_loss,
                "train_sent_ret_epoch": train_sent_ret,
                "train_sign_ret_epoch": train_sign_ret,
                "train_sign_cls_epoch": train_sign_cls,
                "val_loss_epoch": val_loss,
                "val_sent_ret_epoch": val_sent_ret,
                "val_sign_ret_epoch": val_sign_ret,
                "val_sign_cls_epoch": val_sign_cls,
                "epoch": epoch + 1,
            }
        )


def log_test_retrieval(
    test_t2v: dict,
    test_v2t: dict,
    cfg: DictConfig,
):
    """Log test retrieval performances."""
    if cfg.do_print:
        log_dict = {
            "T2V R@1": test_t2v["R1"],
            "T2V R@5": test_t2v["R5"],
            "T2V R@10": test_t2v["R10"],
            "T2V R@50": test_t2v["R50"],
            "T2V MedR": test_t2v["MedR"],
            "T2V MeanR": test_t2v["MeanR"],
            "T2V geometric_mean_R1-R5-R10": test_t2v["geometric_mean_R1-R5-R10"],
            "V2T R@1": test_v2t["R1"],
            "V2T R@5": test_v2t["R5"],
            "V2T R@10": test_v2t["R10"],
            "V2T R@50": test_v2t["R50"],
            "V2T MedR": test_v2t["MedR"],
            "V2T MeanR": test_v2t["MeanR"],
            "V2T geometric_mean_R1-R5-R10": test_v2t["geometric_mean_R1-R5-R10"],
        }
        log_df = pd.DataFrame(log_dict, index=[0])
        # display as table
        print("")
        print(
            tabulate(
                log_df,
                headers="keys",
                tablefmt="presto",
                showindex="never",
                floatfmt=".2f",
            )
        )
        print("")
        wandb.log(log_dict)


@hydra.main(version_base=None, config_path="config", config_name="cslr2")
def main(cfg: Optional[DictConfig] = None) -> None:
    """Main function"""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # setup configuration
    cfg = ddp_settings(cfg)
    if cfg.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    cfg.do_print = not cfg.distributed or cfg.rank == 0
    if not os.path.isdir(cfg.paths.log_dir):
        os.makedirs(cfg.paths.log_dir)
        print(f"Created {cfg.paths.log_dir}")
    cfg.paths.log_dir += f"{cfg.run_name}"
    if not os.path.isdir(cfg.paths.log_dir):
        os.mkdir(cfg.paths.log_dir)
        print(f"Created {cfg.paths.log_dir}")
    print(f"Logging to {cfg.paths.log_dir}")

    columns = shutil.get_terminal_size().columns
    # avoid deadlock in dataloader + too many open files errors
    torch.multiprocessing.set_sharing_strategy("file_system")

    # setup seed
    setup_seed(seed=cfg.seed)

    # setup logging
    wandb_setup(cfg, setname="CSLR2_train")

    # create model
    model = instantiate_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = handle_model_freeze(model, cfg)

    # optimiser
    opt = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # load checkpoint
    model, opt = load_checkpoint(cfg, model, opt, device)

    # parameter count
    param_count = humanize.intword(sum(p.numel() for p in model.parameters()))
    trainable_param_count = humanize.intword(
        sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    )
    print(
        f"Model has {param_count} parameters ({trainable_param_count} trainable)")

    if cfg.distributed:
        model = DDP(model, device_ids=[cfg.local_rank])

    if cfg.vis and cfg.do_print:
        rgb_frames = cfg.paths.rgb_frames
        rgb_lmdb_env = lmdb.open(
            rgb_frames, readonly=True, lock=False, max_readers=512
        )

    if cfg.test:
        # perform retrieval on the manually aligned test set with the loaded checkpoint
        test_loader = instantiate_test_dataloader(cfg)
        test_v2t, test_t2v = retrieval_loop(
            model=model,
            vis_loader=test_loader,
            rgb_lmdb_env=rgb_lmdb_env,
            setname="test",
            epoch=0,
            cfg=cfg,
        )
        log_test_retrieval(test_t2v, test_v2t, cfg)

    else:
        best_t2v = -1
        # create dataset + dataloader
        train_loader, val_loader, train_skip_mode, val_skip_mode = instantiate_dataloaders(
            cfg)
        print(f"Train dataloader size: {len(train_loader)}")
        print(f"Val dataloader size: {len(val_loader)}")
        vis_train_loader, vis_val_loader = instantiate_vis_dataloaders(cfg)
        print(f"Train vis dataloader size: {len(vis_train_loader)}")
        print(f"Val vis dataloader size: {len(vis_val_loader)}")

        # eventually skip epochs
        train_loader, val_loader = skip_epochs(
            cfg, train_loader, val_loader,
            train_skip_mode, val_skip_mode,
        )

        # loss function
        sent_ret_loss_fn = hydra.utils.instantiate(cfg.loss.sent_ret)
        # small hack to avoid getting unused_parameters
        # in ddp mode when not using SignRet and SignCls
        remove_all = (
            cfg.loss.lda_sign_ret == 0 and cfg.loss.lda_sign_cls == 0
        )
        sign_ret_loss_fn = hydra.utils.instantiate(cfg.loss.sign_ret) \
            if (cfg.loss.lda_sign_ret > 0 or remove_all) else None
        sign_cls_loss_fn = hydra.utils.instantiate(cfg.loss.sign_cls) \
            if (cfg.loss.lda_sign_cls > 0 or remove_all) else None

        for epoch in range(cfg.trainer.epoch_start, cfg.trainer.epochs):
            print("")
            print("-" * columns)
            print(
                f"Epoch {epoch + 1}/{cfg.trainer.epochs}".center(columns)
            )
            train_loss, train_sent_ret, train_sign_ret, train_sign_cls = train_loop(
                model=model,
                opt=opt,
                sent_ret_loss_fn=sent_ret_loss_fn,
                sign_ret_loss_fn=sign_ret_loss_fn,
                sign_cls_loss_fn=sign_cls_loss_fn,
                train_loader=train_loader,
                epoch=epoch,
                cfg=cfg,
            )
            val_loss, val_sent_ret, val_sign_ret, val_sign_cls = val_loop(
                model=model,
                sent_ret_loss_fn=sent_ret_loss_fn,
                sign_ret_loss_fn=sign_ret_loss_fn,
                sign_cls_loss_fn=sign_cls_loss_fn,
                val_loader=val_loader,
                epoch=epoch,
                cfg=cfg,
            )
            log_and_save(
                model, opt,
                train_loss, val_loss,
                train_sent_ret, val_sent_ret,
                train_sign_ret, val_sign_ret,
                train_sign_cls, val_sign_cls,
                vis_train_loader, vis_val_loader,
                rgb_lmdb_env, cfg, epoch, best_t2v,
            )
        print("Training complete!")


if __name__ == "__main__":
    main()
