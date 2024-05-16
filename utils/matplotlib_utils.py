"""Functions to save predictions (retrieval) with matplotlib."""
import os
import pickle
from pathlib import Path
from typing import List, Union

import cv2
import lmdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from matplotlib.animation import ArtistAnimation
from numpy import ndarray
from omegaconf import DictConfig
from torch import Tensor
from tqdm import tqdm


def lmdb_key_list(episode_name: str, begin_frame: int, end_frame: int) -> List:
    """
    Returns list of keys for RGB videos

    Args:
        episode_name (str): Episode name.
        begin_frame (int): Begin frame.
        end_frame (int): End frame.

    Returns:
        List: List of keys mapping to RGB frames in lmdb environment.
    """
    return [f"{Path(episode_name.split('.')[0])}/{frame_idx + 1:07d}.jpg".encode('ascii') \
            for frame_idx in range(begin_frame, end_frame + 1)]


def get_rgb_frames(lmdb_keys: List[str], lmdb_env: lmdb.Environment) -> List:
    """
    Returns list of RGB frames

    Args:
        lmdb_keys (List[str]): List of keys mapping to RGB frames in lmdb environment.
        lmdb_env (lmdb.Environment): lmdb environment.

    Returns:
        frames (List): List of RGB frames.
    """
    frames = []
    for key in lmdb_keys:
        with lmdb_env.begin() as txn:
            frame = txn.get(key)
        frame = cv2.imdecode(
            np.frombuffer(frame, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
    return frames

def save_retrieval_vis(
    cfg: DictConfig,
    sim_matrix: Union[ndarray, Tensor],
    all_sentences: List[str],
    video_names: List[str],
    sub_starts: List[float],
    sub_ends: List[float],
    rgb_lmdb_env: lmdb.Environment,
    setname: str,
    epoch: int,
    pl_as_subtitles: bool = False,
    k: int = 5,
    text_only: bool = False,
) -> None:
    """
    Save retrieval visualization.

    Args:
        cfg (DictConfig): Config file.
        sim_matrix (Union[ndarray, Tensor]): Similarity matrix. sim[i, j] = <text_i, vid_j>.
        all_sentences (List[str]): List of all text sentences.
        video_names (List[str]): List of all video names.
        sub_starts (List[float]): List of all subtitle start times.
        sub_ends (List[float]): List of all subtitle end times.
        rgb_lmdb_env (lmdb.Environment): lmdb environment.
        setname (str): Name of the set (train, val, test).
        epoch (int): Current epoch.
        pl_as_subtitles (bool, optional): Whether to use the PL as subtitles. Defaults to False.
        k (int): Number of subtitles to retrieve for visualisation. Defaults to 5.
        text_only (bool, optional): Whether to only save the text. Defaults to False.
    """
    if text_only:
        v2t_retrieval_results = {
            "gt": [], "retrieved_subs": [], "gt_start": [], "gt_end": [],
            "retrieved_starts": [], "retrieved_ends": [], "gt_name": [], "retrieved_names": [],
            "sims": [],
        }
        t2v_retrieval_results = {
            "gt": [], "retrieved_videos": [], "gt_start": [], "gt_end": [],
            "retrieved_starts": [], "retrieved_ends": [], "gt_name": [], "retrieved_names": [],
            "sims": [],
        }
    for vis_idx in tqdm(range(cfg.nb_vis)):
        # get the similarity scores for the current video
        sim_scores = sim_matrix[:, vis_idx]
        # get the indices of the top-k most similar videos
        topk = torch.topk(torch.tensor(sim_scores), k)
        topk_indices, topk_values = topk.indices, topk.values
        # get the subtitles corresponding to the indices in question
        topk_subtitles = [all_sentences[idx] for idx in topk_indices]
        topk_subtitles_str = ""
        video_name = video_names[vis_idx]
        start, end = sub_starts[vis_idx], sub_ends[vis_idx]
        subtitle = all_sentences[vis_idx]
        if text_only:
            v2t_retrieval_results["gt"].append(subtitle)
            v2t_retrieval_results["retrieved_subs"].append(topk_subtitles)
            v2t_retrieval_results["gt_start"].append(start)
            v2t_retrieval_results["gt_end"].append(end)
            v2t_retrieval_results["gt_name"].append(video_name)
            v2t_retrieval_results["retrieved_starts"].append(
                [sub_starts[idx] for idx in topk_indices]
            )
            v2t_retrieval_results["retrieved_ends"].append(
                [sub_ends[idx] for idx in topk_indices]
            )
            v2t_retrieval_results["retrieved_names"].append(
                [video_names[idx] for idx in topk_indices]
            )
            v2t_retrieval_results["sims"].append(topk_values)
        else:
            for idx, topk_subtitle in enumerate(topk_subtitles):
                topk_subtitles_str += topk_subtitle
                topk_subtitles_str += f"   ({topk_values[idx]:.2f}) \n"
            # get the corresponding video frames
            lmdb_keys = lmdb_key_list(
                episode_name=video_name,
                begin_frame=int(start * 25),
                end_frame=int(end * 25),
            )
            frames = get_rgb_frames(lmdb_keys, rgb_lmdb_env)
            # assemble into matplotlib figure
            fig = plt.figure()
            ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)
            ax1.set_xlim(0, 255)
            ax1.set_ylim(0, 255)
            ax1.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
            ax1.set_title(subtitle)
            ax2.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False,
            )
            ax2.text(
                .5, .5,
                topk_subtitles_str,
                fontsize=7,
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax2.transAxes,
                wrap=True,
            )
            animated_frames = []
            for idx, frame in enumerate(frames):
                animated_frame = []
                animated_frame.append(
                    ax1.imshow(
                        np.flipud(frame),
                        animated=True,
                        interpolation="nearest",
                    )
                )
                animated_frames.append(animated_frame)
            fig.tight_layout()
            anim = ArtistAnimation(
                fig,
                animated_frames,
                interval=50,
                blit=True,
                repeat=False,
            )
            # save the video locally
            video_prefix = f"pl_as_subtitles_{setname}" if pl_as_subtitles else setname
            video_path = cfg.paths.log_dir + \
                f"/videos/{video_prefix}_E_{str(epoch + 1)}_I_{str(vis_idx + 1)}.mp4"
            if not os.path.exists(cfg.paths.log_dir + "/videos/"):
                os.makedirs(cfg.paths.log_dir + "/videos/")
            anim.save(video_path, writer="ffmpeg", fps=25)
            # upload to wandb
            wandb.log(
                {f"{video_prefix}--video": wandb.Video(video_path, fps=25, format="mp4")},
            )
    for vis_idx in tqdm(range(cfg.nb_vis)):
        # get the similarity scores for the current subtitle
        sim_scores = sim_matrix[vis_idx, :]
        # get the indices of the top-k most similar videos
        topk = torch.topk(torch.tensor(sim_scores), k)
        topk_indices, topk_values = topk.indices, topk.values
        # get the subtitles corresponding to the indices in question
        topk_subtitles = [all_sentences[idx] for idx in topk_indices]
        topk_subtitles_str = ""
        video_name = video_names[vis_idx]
        start, end = sub_starts[vis_idx], sub_ends[vis_idx]
        subtitle = all_sentences[vis_idx]
        if text_only:
            t2v_retrieval_results["gt"].append(subtitle)
            t2v_retrieval_results["retrieved_videos"].append(topk_subtitles)
            t2v_retrieval_results["gt_start"].append(start)
            t2v_retrieval_results["gt_end"].append(end)
            t2v_retrieval_results["gt_name"].append(video_name)
            t2v_retrieval_results["retrieved_starts"].append(
                [sub_starts[idx] for idx in topk_indices]
            )
            t2v_retrieval_results["retrieved_ends"].append(
                [sub_ends[idx] for idx in topk_indices]
            )
            t2v_retrieval_results["retrieved_names"].append(
                [video_names[idx] for idx in topk_indices]
            )
            t2v_retrieval_results["sims"].append(topk_values)
    if text_only:
        # save the dictionary
        f_path = cfg.paths.log_dir + f"/retrieval_results_{setname}_E_{str(epoch + 1)}.pkl"
        saved_pickle = {"t2v": t2v_retrieval_results, "v2t": v2t_retrieval_results}
        pickle.dump(
            saved_pickle,
            open(f_path, "wb")
        )
