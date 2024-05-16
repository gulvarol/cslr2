"""
Function to set the seed for random number generators.
"""
import numpy as np
import torch


def setup_seed(seed: int) -> None:
    """Set seed for numpy and torch"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
