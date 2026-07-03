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
        # Skip ColliderMLDataset.__init__ — it would build Parquet shard
        # indices and scan files, which is unnecessary for feather I/O.
        # We only depend on SeedExtensionDataset._process_event and its
        # helper methods, which require: self.event_ids, self.stage,
        # self._kwargs, self._raw_cache.
        self.cache_root = Path(cache_root)
        self.hits_dir = self.cache_root / "hits"
        self.parts_dir = self.cache_root / "parts"
        self.event_ids = list(event_ids)
        self.stage = stage
        self._kwargs = kwargs
        self._raw_cache = {}   # disabled — feather reads are fast enough

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int):
        event_id = self.event_ids[idx]
        hits_raw = feather.read_feather(self.hits_dir / f"{event_id}.feather")
        parts_raw = feather.read_feather(self.parts_dir / f"{event_id}.feather")
        return self._process_event(hits_raw, parts_raw, idx)


class CachedColliderMLDataModule(L.LightningDataModule):
    """Lightning DataModule that reads from a pre-built feather cache.

    This replaces the Parquet-scanning ``setup()`` of ColliderMLDataModule
    with a cache-scanning version.  If the cache is missing or incomplete,
    it falls back to the parent-class behaviour (Parquet).

    Parameters
    ----------
    data_dir : str
        Source ColliderML root (used for fallback only).
    cache_dir : str | None
        Root of the feather cache.  If None, delegates entirely to
        ColliderMLDataModule.
    All other parameters are forwarded to ColliderMLDataModule.
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: str | None = None,
        process: str = "ttbar",
        pileup: str = "pu200",
        max_train_events: int = 50000,
        max_val_events: int = 5000,
        max_test_events: int = 5000,
        batch_size: int = 1,
        num_workers: int = 8,
        **dataset_kwargs: Any,
    ) -> None:
        super().__init__()
        dataset_kwargs = dict(dataset_kwargs)
        dataset_kwargs.setdefault("dataset_cls", SeedExtensionDataset)
        self._parent = ColliderMLDataModule(
            data_dir=data_dir,
            process=process,
            pileup=pileup,
            max_train_events=max_train_events,
            max_val_events=max_val_events,
            max_test_events=max_test_events,
            batch_size=batch_size,
            num_workers=num_workers,
            **dataset_kwargs,
        )
        self.cache_dir = cache_dir
        self.data_dir = self._parent.data_dir
        self.batch_size = self._parent.batch_size
        self.num_workers = self._parent.num_workers

    @property
    def cache_mode(self) -> str:
        return "feather" if self.cache_dir is not None else "none"

    def _cache_exists(self) -> bool:
        if self.cache_dir is None:
            return False
        meta = Path(self.cache_dir) / "meta.json"
        return meta.exists()

    def setup(self, stage: str | None = None) -> None:
        if self.cache_dir is None or not self._cache_exists():
            if self.cache_dir is not None:
                rank_zero_warn(
                    f"Feather cache not found at {self.cache_dir}, falling back to Parquet.",
                )
            self._parent.setup(stage)
            self.trainset = self._parent.trainset
            self.train_dataset = self.trainset
            self.valset = self._parent.valset
            self.val_dataset = self.valset
            self.testset = self._parent.testset
            self.test_dataset = self.testset
            return

        cache_root = Path(self.cache_dir)

        with (cache_root / "meta.json").open("r") as fh:
            meta = json.load(fh)

        import colliderml_dataloader.shard_index as _si
        cached_hit_cols = meta.get("hit_cols", [])
        cached_part_cols = meta.get("part_cols", [])
        if cached_hit_cols != _si.TRACKER_HIT_FEATURES or cached_part_cols != _si.PARTICLE_FEATURES:
            raise RuntimeError(
                "Cached column lists do not match current feature lists. "
                f"Rebuild the cache or adjust TRACKER_HIT_FEATURES / PARTICLE_FEATURES.\n"
                f"  Cache: hits={cached_hit_cols}  parts={cached_part_cols}\n"
                f"  Current: hits={_si.TRACKER_HIT_FEATURES}  parts={_si.PARTICLE_FEATURES}"
            )

        hits_dir = cache_root / "hits"
        parts_dir = cache_root / "parts"
        hit_ids = {int(p.stem) for p in hits_dir.glob("*.feather")}
        part_ids = {int(p.stem) for p in parts_dir.glob("*.feather")}
        all_event_ids = sorted(hit_ids & part_ids)

        total_needed = self._parent.max_train_events + self._parent.max_val_events + self._parent.max_test_events
        if len(all_event_ids) < total_needed:
            rank_zero_warn(
                f"Feather cache has {len(all_event_ids)} events but {total_needed} requested — using all available.",
            )
            total_needed = len(all_event_ids)

        all_event_ids = all_event_ids[:total_needed]

        dw = self._parent.dataset_kwargs

        train_ids = all_event_ids[: self._parent.max_train_events]
        self.trainset = CachedColliderMLDataset(
            cache_root=cache_root, event_ids=train_ids, stage="fit", **dw,
        )
        self.train_dataset = self.trainset

        v0 = self._parent.max_train_events
        v1 = min(v0 + self._parent.max_val_events, total_needed)
        val_ids = all_event_ids[v0:v1]
        self.valset = CachedColliderMLDataset(
            cache_root=cache_root, event_ids=val_ids, stage="validate", **dw,
        )
        self.val_dataset = self.valset

        s0 = v1
        s1 = min(s0 + self._parent.max_test_events, total_needed)
        test_ids = all_event_ids[s0:s1]
        self.testset = CachedColliderMLDataset(
            cache_root=cache_root, event_ids=test_ids, stage="test", **dw,
        )
        self.test_dataset = self.testset

        rank_zero_info(
            f"Loaded feather cache from {cache_root} — {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test events.",
        )

    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size,
            num_workers=self._parent.num_workers, drop_last=True,
            shuffle=True, collate_fn=default_collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.batch_size,
            num_workers=self._parent.num_workers, shuffle=False,
            collate_fn=default_collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset, batch_size=self.batch_size,
            num_workers=self._parent.num_workers, shuffle=False,
            collate_fn=default_collate,
        )

    def predict_dataloader(self):
        return [
            DataLoader(
                self.trainset, batch_size=self.batch_size,
                num_workers=self._parent.num_workers, shuffle=False,
                collate_fn=default_collate,
            ),
            DataLoader(
                self.valset, batch_size=self.batch_size,
                num_workers=self._parent.num_workers, shuffle=False,
                collate_fn=default_collate,
            ),
            DataLoader(
                self.testset, batch_size=self.batch_size,
                num_workers=self._parent.num_workers, shuffle=False,
                collate_fn=default_collate,
            ),
        ]
