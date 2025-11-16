"""
Random seed utilities for ensuring reproducibility across all libraries.
"""

import random
import numpy as np
import torch
import pytorch_lightning as pl

def set_random_seeds(seed: int):
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value to use for all libraries
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Set PyTorch Lightning seed
    pl.seed_everything(seed, workers=True)

    print(f"Random seeds set to {seed} for reproducibility")
