"""
Random seed utilities for ensuring reproducibility across all libraries.
"""

import random

import numpy as np
import pytorch_lightning as pl
import torch


def set_random_seeds(seed: int) -> None:
    """
    Align random state across libraries used in training.

    Args:
        seed: Integer seed applied everywhere.

    Returns:
        None.

    Note:
        ``workers=False`` avoids per-worker re-seeding in Lightning DataLoaders; set
        seeds in dataset code if you need stricter worker-level reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=False)
    print(f"Random seeds set to {seed} for reproducibility")
