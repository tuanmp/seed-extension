"""Feather-based exploded-data cache for ColliderML — dataset and DataModule."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import lightning as L
import pyarrow.feather as feather
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from torch.utils.data import DataLoader, default_collate

from colliderml_dataloader import ColliderMLDataModule

from seed_extension.data.dataset import SeedExtensionDataset

logger = logging.getLogger(__name__)


class CachedColliderMLDataset(SeedExtensionDataset):
    """Dataset that reads pre-exploded events from feather cache files.

    Inherits all downstream processing (subsampling, filtering, seed
    building, target assembly) from SeedExtensionDataset.  Only the
    data-loading step is changed — feather files instead of Parquet.

    Parameters
    ----------
    cache_root : Path
        Root directory containing ``hits/`` and ``parts/`` subdirectories
        with per-event ``.feather`` files.
    event_ids : list[int]
        Ordered list of event IDs this dataset serves.
    stage : str
        Lightning stage.
    **kwargs
        Forwarded to SeedExtensionDataset._process_event via self._kwargs.
    """

    def __init__(
        self,
        cache_root: str | Path,
        event_ids: list[int],
        stage: str,
        **kwargs: Any,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.hits_dir = self.cache_root / "hits"
        self.parts_dir = self.cache_root / "parts"
        self.event_ids = list(event_ids)
        self.stage = stage
        self._kwargs = kwargs
        self._raw_cache = {}   # disabled — see notes below

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int):
        event_id = self.event_ids[idx]
        hits_raw = feather.read_feather(self.hits_dir / f"{event_id}.feather")
        parts_raw = feather.read_feather(self.parts_dir / f"{event_id}.feather")
        return self._process_event(hits_raw, parts_raw, idx)
