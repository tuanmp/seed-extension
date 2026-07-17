"""Feather-based exploded-data cache for ColliderML — dataset and DataModule."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import pyarrow.feather as feather
from lightning.pytorch.utilities import rank_zero_info, rank_zero_warn
from torch.utils.data import DataLoader, default_collate

from colliderml_dataloader import ColliderMLDataModule
import polars as pl

from seed_extension.data.dataset import SeedExtensionDataset
from seed_extension.data.seed_utils import PARTICLE_FEATURES, TRACKER_HIT_FEATURES

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
        # self._kwargs.
        self.cache_root = Path(cache_root)
        self.hits_dir = self.cache_root / "hits"
        self.parts_dir = self.cache_root / "parts"
        self.event_ids = list(event_ids)
        self.stage = stage
        self._kwargs = kwargs

    def __len__(self) -> int:
        return len(self.event_ids)

    def __getitem__(self, idx: int):
        event_id = self.event_ids[idx]
        hits_raw = feather.read_feather(self.hits_dir / f"{event_id}.feather")
        parts_raw = feather.read_feather(self.parts_dir / f"{event_id}.feather")
        hits_raw = pl.from_pandas(hits_raw)
        parts_raw = pl.from_pandas(parts_raw)
        return self._process_event(hits_raw, parts_raw, idx)


class CachedColliderMLDataModule(L.LightningDataModule):
    """Lightning DataModule that reads from a pre-built feather cache.

    If the cache is missing or ``cache_dir`` is ``None``, it transparently
    delegates to :class:`ColliderMLDataModule` (Parquet path).
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
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        min_track_hits: int = 5,
        min_pT: float = 0.0,
        max_abs_eta: float = 4.0,
        seed_strategy: str = "random_consecutive",
        target_vertices: int = 200,
        primary_only: bool = False,
        predict_seed_hits: bool = False,
        n_seed_hits: int = 3,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        kwargs = dict(dataset_kwargs) if dataset_kwargs else {}
        kwargs.setdefault("min_track_hits", min_track_hits)
        kwargs.setdefault("min_pT", min_pT)
        kwargs.setdefault("max_abs_eta", max_abs_eta)
        kwargs.setdefault("seed_strategy", seed_strategy)
        kwargs.setdefault("target_vertices", target_vertices)
        kwargs.setdefault("primary_only", primary_only)
        kwargs.setdefault("predict_seed_hits", predict_seed_hits)
        kwargs.setdefault("n_seed_hits", n_seed_hits)
        kwargs.setdefault("dataset_cls", SeedExtensionDataset)
        self._parent = ColliderMLDataModule(
            data_dir=data_dir,
            process=process,
            pileup=pileup,
            max_train_events=max_train_events,
            max_val_events=max_val_events,
            max_test_events=max_test_events,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs,
        )
        self.cache_dir = cache_dir
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.data_dir = self._parent.data_dir
        self.batch_size = self._parent.batch_size
        self.num_workers = self._parent.num_workers

    @property
    def cache_mode(self) -> str:
        return "feather" if self.cache_dir is not None else "none"

    def _cache_exists(self) -> bool:
        if self.cache_dir is None:
            return False
        return Path(self.cache_dir, "meta.json").exists()

    def setup(self, stage: str | None = None) -> None:
        if self.cache_dir is None or not self._cache_exists():
            if self.cache_dir is not None:
                rank_zero_warn(
                    f"Feather cache not found at {self.cache_dir}, falling back to Parquet."
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

        cached_hit_cols = meta.get("hit_cols", [])
        cached_part_cols = meta.get("part_cols", [])
        if cached_hit_cols != TRACKER_HIT_FEATURES or cached_part_cols != PARTICLE_FEATURES:
            raise RuntimeError(
                "Cached column lists do not match current feature lists. "
                "Rebuild the cache or adjust seed_extension.data.seed_utils "
                "TRACKER_HIT_FEATURES / PARTICLE_FEATURES.\n"
                f"  Cache: hits={cached_hit_cols}  parts={cached_part_cols}\n"
                f"  Current: hits={TRACKER_HIT_FEATURES}  parts={PARTICLE_FEATURES}"
            )

        hits_dir = cache_root / "hits"
        parts_dir = cache_root / "parts"
        hit_ids = {int(p.stem) for p in hits_dir.glob("*.feather")}
        part_ids = {int(p.stem) for p in parts_dir.glob("*.feather")}
        all_event_ids = sorted(hit_ids & part_ids)

        total_needed = (
            self._parent.max_train_events
            + self._parent.max_val_events
            + self._parent.max_test_events
        )
        if len(all_event_ids) < total_needed:
            rank_zero_warn(
                f"Feather cache has {len(all_event_ids)} events "
                f"but {total_needed} requested — using all available."
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
            f"Loaded feather cache from {cache_root} — "
            f"{len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test events."
        )

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def _dataloader_kwargs(self, shuffle: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self._parent.num_workers,
            "collate_fn": default_collate,
        }
        if self._parent.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["persistent_workers"] = self.persistent_workers
        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.trainset, drop_last=True,
            **self._dataloader_kwargs(shuffle=True),
        )

    def val_dataloader(self):
        return DataLoader(self.valset, **self._dataloader_kwargs())

    def test_dataloader(self):
        return DataLoader(self.testset, **self._dataloader_kwargs())

    def predict_dataloader(self):
        kwargs = self._dataloader_kwargs()
        return [
            DataLoader(self.trainset, **kwargs),
            DataLoader(self.valset, **kwargs),
            DataLoader(self.testset, **kwargs),
        ]
