"""
Functions for augmenting videos.
"""
from typing import Union

import numpy as np
import torch


class DropFrames(object):
    """
    Drop frames from a video or a sequence of video frames/features.

    Args:
        p_sequence (float): probability of augmenting a video sequence
        p_frame (float): probability of dropping a frame
    """
    def __init__(
        self,
        p_sequence: float = 0.5,
        p_frame: float = 0.3,
    ) -> None:
        self.p_sequence = p_sequence
        self.p_frame = p_frame

    def _drop_frames(
        self,
        video: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        # keep at least 25 frames
        if len(video) <= 25:
            return video, np.arange(len(video))
        kept_frames = 25
        while kept_frames <= 25:
            # assign a random probability to each frame
            frames_probs = np.random.rand(len(video))
            # get indices of frames to keep
            kept_indices = np.where(frames_probs >= self.p_frame)[0]
            kept_frames = len(kept_indices)
        return video[kept_indices], kept_indices

    def __call__(
        self,
        video: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.p_sequence:
            return self._drop_frames(video)
        return video, np.arange(len(video))
