"""Reproducibility utilities — seed setting for numpy, torch, lightning."""

from __future__ import annotations

import os
import random

import lightning as L
import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set common RNG seeds and optional deterministic mode."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    L.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
