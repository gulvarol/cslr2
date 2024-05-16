"""
Python file defining the retrieval loop of the model.
"""
import lmdb
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from loops.retrieval import t2v_metrics, v2t_metrics
from utils.matplotlib_utils import save_retrieval_vis

def retrieval_loop(
    model: nn.Module,
    vis_loader: DataLoader,
    rgb_lmdb_env: lmdb.Environment,
    cfg: DictConfig,
    setname: str,
    epoch: int,
):
    """
    T2V and V2T Retrieval Loop + eventual video visualisation saving.

    Args:
        model (nn.Module): model to train
        vis_loader (DataLoader): visualisation DataLoader
        rgb_lmdb_env (lmdb.Environment): lmdb environment for video retrieval
        cfg (DictConfig): config file
        setname (str): name of the dataset
        epoch (int): current epoch
    """
    device = torch.device(
        f"cuda:{cfg.local_rank}" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_text, all_cls_tokens = [], []
    all_sentences = []
    video_names, sub_starts, sub_ends = [], [], []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(iter(vis_loader))):
            # unpack the batch
            subs, feats, _, _, _, _, names, starts, ends = batch
            feats = feats.to(device)
            # retrieval forward pass
            cls_tokens, sentence_embds = model.forward_sentret(feats, subs) if not cfg.distributed \
                else model.module.forward_sentret(feats, subs)
            all_text.append(sentence_embds)
            all_cls_tokens.append(cls_tokens)
            all_sentences.extend(subs)
            video_names.extend(names)
            sub_starts.extend(starts)
            sub_ends.extend(ends)
    all_text = torch.cat(all_text, dim=0)
    all_cls_tokens = torch.cat(all_cls_tokens, dim=0)
    # compute similarities st sims[i, j] = <text_i, cls_j>
    sims = all_text @ all_cls_tokens.T
    sims = sims.detach().cpu().numpy()
    v2t, ranks = v2t_metrics(sims)
    t2v, _ = t2v_metrics(sims)

    if cfg.worst_retrieval:
        # get the worst retrieval cases
        indices = np.argsort(-ranks)
        # reorder sims
        sims = sims[:, indices]  # first order columns
        sims = sims[indices, :]  # then order rows
        all_sentences = np.array(all_sentences)[indices]
        sub_starts = np.array(sub_starts)[indices]
        sub_ends = np.array(sub_ends)[indices]
        video_names = np.array(video_names)[indices]
    if cfg.nb_vis > 0:
        save_retrieval_vis(
            cfg=cfg,
            sim_matrix=sims,
            all_sentences=all_sentences,
            video_names=video_names,
            sub_starts=sub_starts,
            sub_ends=sub_ends,
            rgb_lmdb_env=rgb_lmdb_env,
            setname=setname,
            epoch=epoch,
            k=5 if not cfg.worst_retrieval else 20,
            text_only=False,
        )
    return v2t, t2v
