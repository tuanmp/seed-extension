# Feather Cache for ColliderML — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feather-based exploded-data cache for ColliderML that eliminates Parquet I/O + polars explode (1,629ms → 15.6ms per event, 104× speedup).

**Architecture:** Three new files: `cache.py` (torch dataset + Lightning DataModule), `build_feather_cache.py` (one-time builder script), and `test_feather_cache.py`. The CachedColliderMLDataModule scans feather cache directories instead of Parquet shards. CachedColliderMLDataset reads pre-exploded pandas DataFrames directly from feather files. All downstream processing (subsampling, filtering, seed building) is identical to SeedExtensionDataset.

**Tech Stack:** PyArrow Feather (zstd compression), polars (build phase only), Lightning DataModule.

---

## File Map

| File | Responsibility |
|------|---------------|
| `src/seed_extension/data/cache.py` (new) | `CachedColliderMLDataset`, `CachedColliderMLDataModule` |
| `scripts/build_feather_cache.py` (new) | CLI script to build the feather cache from source Parquet |
| `tests/test_feather_cache.py` (new) | Tests for cache build, read, and integration |
| `configs/cast_default.yaml` (modify) | Add `cache_dir: null` |
| `configs/cast_pu10.yaml` (modify) | Add `cache_dir: null` |
| `src/seed_extension/train.py` (modify) | Wire `CachedColliderMLDataModule` when `cache_dir` is set |

---

### Task 1: CachedColliderMLDataset

**Files:**
- Create: `src/seed_extension/data/cache.py`

- [ ] **Step 1: Create the module with CachedColliderMLDataset**

```python
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
```

- [ ] **Step 2: Verify it imports**

```bash
source setup.sh && uv run python -c "from seed_extension.data.cache import CachedColliderMLDataset; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/seed_extension/data/cache.py
git commit -m "feat: add CachedColliderMLDataset — reads exploded events from feather cache"
```

---

### Task 2: CachedColliderMLDataModule

**Files:**
- Modify: `src/seed_extension/data/cache.py`

- [ ] **Step 1: Add CachedColliderMLDataModule to cache.py**

Append after `CachedColliderMLDataset`:

```python
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
        # Mirror parent attributes for DataLoader transparency.
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
                    "Feather cache not found at %s, falling back to Parquet.",
                    self.cache_dir,
                )
            return self._parent.setup(stage)

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
                "Feather cache has %d events but %d requested — using all available.",
                len(all_event_ids), total_needed,
            )
            total_needed = len(all_event_ids)

        all_event_ids = all_event_ids[:total_needed]

        dw = self._parent.dataset_kwargs
        batch_size = self._parent.batch_size

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
            "Loaded feather cache from %s — %d train / %d val / %d test events.",
            cache_root, len(train_ids), len(val_ids), len(test_ids),
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
```

- [ ] **Step 2: Write the test for CachedColliderMLDataModule**

Create `tests/test_feather_cache.py`:

```python
"""Tests for the feather cache dataset and DataModule."""

import json
import os
import tempfile
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.feather as feather
import torch
import pytest

from colliderml_dataloader import ColliderMLDataModule
from colliderml.polars import explode_tracker_hits, explode_particles
from colliderml_dataloader.shard_index import (
    TRACKER_HIT_FEATURES,
    PARTICLE_FEATURES,
    shard_dir_for,
)

import seed_extension.data  # noqa: F401 — monkey-patch feature lists
from seed_extension.data.dataset import SeedExtensionDataset
from seed_extension.data.cache import CachedColliderMLDataset, CachedColliderMLDataModule

DATA_DIR = os.environ.get("COLLIDERML_DATA_DIR", "/pscratch/sd/p/pmtuan/.cache/colliderml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs_data() -> bool:
    return Path(DATA_DIR).exists()


requires_data = pytest.mark.skipif(
    not _needs_data(),
    reason="COLLIDERML_DATA_DIR not available on this machine",
)


def build_mini_cache(n_events: int = 2) -> Path:
    """Build a tiny feather cache with *n_events* events and return cache_root."""
    cache_root = Path(tempfile.mkdtemp(prefix="feather_cache_test_"))
    hits_dir = cache_root / "hits"
    parts_dir = cache_root / "parts"
    hits_dir.mkdir(parents=True)
    parts_dir.mkdir(parents=True)

    hit_shard_dir = shard_dir_for(DATA_DIR, "ttbar", "pu200", "tracker_hits")
    part_shard_dir = shard_dir_for(DATA_DIR, "ttbar", "pu200", "particles")

    from colliderml_dataloader.shard_index import build_shard_index
    idx = build_shard_index(hit_shard_dir, n_events)
    event_ids = sorted(idx.keys())[:n_events]

    for eid in event_ids:
        hit_file = idx[eid]
        part_file = build_shard_index(part_shard_dir, n_events)[eid]

        hits_raw = (
            pl.scan_parquet(hit_file)
            .filter(pl.col("event_id") == eid)
            .select(TRACKER_HIT_FEATURES)
            .collect()
        )
        parts_raw = (
            pl.scan_parquet(part_file)
            .filter(pl.col("event_id") == eid)
            .select(PARTICLE_FEATURES)
            .collect()
        )

        hits_expl = explode_tracker_hits(hits_raw).drop(columns=["event_id"])
        parts_expl = explode_particles(parts_raw).drop(columns=["event_id"])

        feather.write_feather(hits_dir / f"{eid}.feather", hits_expl, compression="zstd")
        feather.write_feather(parts_dir / f"{eid}.feather", parts_expl, compression="zstd")

    meta = {
        "version": 1,
        "process": "ttbar",
        "pileup": "pu200",
        "hit_cols": TRACKER_HIT_FEATURES,
        "part_cols": PARTICLE_FEATURES,
        "n_events": len(event_ids),
    }
    with (cache_root / "meta.json").open("w") as f:
        json.dump(meta, f)

    return cache_root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCachedDataset:
    @requires_data
    def test_returns_same_keys_as_seed_extension(self):
        cache_root = build_mini_cache(2)
        # Cached dataset
        cached_ds = CachedColliderMLDataset(
            cache_root=cache_root,
            event_ids=[0, 1],
            stage="fit",
            min_track_hits=5,
            seed_strategy="fixed_innermost",
            target_vertices=200,
            primary_only=False,
            predict_seed_hits=True,
        )
        sample = cached_ds[0]
        expected_keys = {
            "hits", "seeds", "targets", "seed_particle_ids",
            "hit_particle_ids", "kinematics", "event_idx",
        }
        assert set(sample.keys()) == expected_keys

        hits = sample["hits"]
        seeds = sample["seeds"]
        targets = sample["targets"]
        assert hits.ndim == 2, f"hits shape: {hits.shape}"
        assert seeds.ndim == 2, f"seeds shape: {seeds.shape}"
        assert targets.ndim == 2, f"targets shape: {targets.shape}"
        N_s, N_h = seeds.shape[0], hits.shape[0]
        assert targets.shape == (N_s, N_h)

        shutil.rmtree(cache_root)

    @requires_data
    def test_same_output_as_seed_extension(self):
        """Cached dataset must produce identical output to SeedExtensionDataset
        when run on the same event with same seed/rand config."""
        cache_root = build_mini_cache(1)

        # Ensure deterministic seed building (fixed_innermost, same event_idx).
        cached_ds = CachedColliderMLDataset(
            cache_root=cache_root, event_ids=[0], stage="validate",
            min_track_hits=5, seed_strategy="fixed_innermost",
            target_vertices=200, primary_only=False, predict_seed_hits=True,
        )
        cached_sample = cached_ds[0]

        # Re-read the same event via the Parquet path.
        dm = ColliderMLDataModule(
            data_dir=DATA_DIR, process="ttbar", pileup="pu200",
            max_train_events=1, max_val_events=0, max_test_events=0,
            batch_size=1, num_workers=0,
            dataset_cls=SeedExtensionDataset,
            min_track_hits=5, seed_strategy="fixed_innermost",
            target_vertices=200, primary_only=False, predict_seed_hits=True,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        pq_sample = next(iter(loader))
        # Remove batch dim
        pq_sample = {k: v[0] for k, v in pq_sample.items()}

        for key in cached_sample:
            assert torch.allclose(cached_sample[key], pq_sample[key]), (
                f"Mismatch in key='{key}': "
                f"cached={cached_sample[key].shape} vs pq={pq_sample[key].shape}"
            )

        shutil.rmtree(cache_root)


class TestCachedDataModule:
    @requires_data
    def test_cache_exists_uses_cached_path(self):
        cache_root = build_mini_cache(2)
        dm = CachedColliderMLDataModule(
            data_dir=DATA_DIR, cache_dir=str(cache_root),
            process="ttbar", pileup="pu200",
            max_train_events=2, max_val_events=0, max_test_events=0,
            batch_size=1, num_workers=0,
            min_track_hits=5, seed_strategy="fixed_innermost",
            target_vertices=200, primary_only=False, predict_seed_hits=True,
        )
        dm.setup("fit")
        assert len(dm.trainset) == 2
        assert isinstance(dm.trainset, CachedColliderMLDataset)

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, dict)
        assert "hits" in batch
        shutil.rmtree(cache_root)

    @requires_data
    def test_cache_missing_falls_back_to_parquet(self):
        dm = CachedColliderMLDataModule(
            data_dir=DATA_DIR, cache_dir="/tmp/nonexistent_cache_xyz",
            process="ttbar", pileup="pu200",
            max_train_events=1, max_val_events=0, max_test_events=0,
            batch_size=1, num_workers=0,
            min_track_hits=5, seed_strategy="fixed_innermost",
            target_vertices=200, primary_only=False, predict_seed_hits=True,
        )
        dm.setup("fit")
        assert len(dm.trainset) == 1
        assert isinstance(dm.trainset, SeedExtensionDataset)

    @requires_data
    def test_cache_dir_none_delegates_to_parent(self):
        dm = CachedColliderMLDataModule(
            data_dir=DATA_DIR, cache_dir=None,
            process="ttbar", pileup="pu200",
            max_train_events=1, max_val_events=0, max_test_events=0,
            batch_size=1, num_workers=0,
            min_track_hits=5, seed_strategy="fixed_innermost",
            target_vertices=200, primary_only=False, predict_seed_hits=True,
        )
        dm.setup("fit")
        assert len(dm.trainset) == 1
        assert isinstance(dm.trainset, SeedExtensionDataset)


class TestCacheMeta:
    @requires_data
    def test_column_mismatch_raises(self):
        cache_root = build_mini_cache(1)

        # Corrupt meta.json
        meta_path = cache_root / "meta.json"
        mid = json.loads(meta_path.read_text())
        mid["hit_cols"] = ["x", "wrong_column"]
        meta_path.write_text(json.dumps(mid))

        import colliderml_dataloader.shard_index as _si
        dm = CachedColliderMLDataModule(
            data_dir=DATA_DIR, cache_dir=str(cache_root),
            process="ttbar", pileup="pu200",
            max_train_events=1, max_val_events=0, max_test_events=0,
            batch_size=1, num_workers=0,
            min_track_hits=5, seed_strategy="fixed_innermost",
            target_vertices=200, primary_only=False, predict_seed_hits=True,
        )
        with pytest.raises(RuntimeError, match="Cached column lists"):
            dm.setup("fit")

        shutil.rmtree(cache_root)
```

- [ ] **Step 3: Run the tests**

```bash
source setup.sh && uv run pytest tests/test_feather_cache.py -v --timeout=300
```

Expected: 0 new failures (5 tests pass)

- [ ] **Step 4: Commit**

```bash
git add src/seed_extension/data/cache.py tests/test_feather_cache.py
git commit -m "feat: add CachedColliderMLDataModule with tests"
```

---

### Task 3: Build script

**Files:**
- Create: `scripts/build_feather_cache.py`

- [ ] **Step 1: Create the build script**

```python
#!/usr/bin/env python3
"""Build a feather cache of exploded ColliderML events from source Parquet.

Usage:
  python scripts/build_feather_cache.py \\
    --process ttbar --pileup pu200 \\
    --data-dir /pscratch/sd/p/pmtuan/.cache/colliderml \\
    --cache-dir /pscratch/sd/p/pmtuan/.cache/colliderml-feather \\
    --max-events 100000 \\
    --num-workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.feather as feather
from colliderml.polars import explode_particles, explode_tracker_hits
from colliderml_dataloader.shard_index import (
    PARTICLE_FEATURES,
    TRACKER_HIT_FEATURES,
    build_shard_index,
    shard_dir_for,
)

import seed_extension.data  # noqa: F401 — monkey-patch feature lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_cache(
    data_dir: str,
    process: str,
    pileup: str,
    cache_root: str,
    max_events: int,
    num_workers: int = 8,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    cache_path = Path(cache_root)
    hits_dir = cache_path / "hits"
    parts_dir = cache_path / "parts"
    meta_path = cache_path / "meta.json"

    hits_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Discover source shards.
    hit_shard_dir = shard_dir_for(data_dir, process, pileup, "tracker_hits")
    part_shard_dir = shard_dir_for(data_dir, process, pileup, "particles")

    logger.info("Building event index from %s …", hit_shard_dir)
    hit_index = build_shard_index(hit_shard_dir, max_events)
    part_index = build_shard_index(part_shard_dir, max_events)
    all_event_ids = sorted(set(hit_index) & set(part_index))[:max_events]
    logger.info("Found %d events.", len(all_event_ids))

    # Group events by shard file — each shard is one work unit.
    shard_to_events: dict[str, list[int]] = {}
    for eid in all_event_ids:
        shard = hit_index[eid]
        shard_to_events.setdefault(shard, []).append(eid)

    shard_files = sorted(shard_to_events.keys())
    logger.info("Processing %d shard files with %d workers.", len(shard_files), num_workers)

    # Write meta.json early (so partial builds are identifiable).
    meta = {
        "version": 1,
        "process": process,
        "pileup": pileup,
        "hit_cols": TRACKER_HIT_FEATURES,
        "part_cols": PARTICLE_FEATURES,
        "n_events": len(all_event_ids),
        "feather_compression": compression,
        "feather_compression_level": compression_level,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Build work items: (hit_file, part_file, [(event_id, event_id_in_parquet), …]).
    work_items = []
    for shard_path in shard_files:
        part_path = part_index[shard_to_events[shard_path][0]]
        events_in_shard = shard_to_events[shard_path]
        work_items.append((shard_path, part_path, events_in_shard))

    t0 = time.perf_counter()
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(
            _process_shard,
            [
                (item, str(hits_dir), str(parts_dir), compression, compression_level)
                for item in work_items
            ],
        )

    total_events = sum(results)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Built cache for %d events in %.1f minutes (%.2f s/event).",
        total_events, elapsed / 60, elapsed / max(total_events, 1),
    )


def _process_shard(
    work_item: tuple[str, str, list[int]],
    hits_dir: str,
    parts_dir: str,
    compression: str,
    compression_level: int,
) -> int:
    hit_file, part_file, event_ids = work_item
    n_written = 0

    for eid in event_ids:
        out_hit = Path(hits_dir) / f"{eid}.feather"
        out_part = Path(parts_dir) / f"{eid}.feather"
        if out_hit.exists() and out_part.exists():
            n_written += 1
            continue

        hits_raw = (
            pl.scan_parquet(hit_file)
            .filter(pl.col("event_id") == eid)
            .select(TRACKER_HIT_FEATURES)
            .collect()
        )
        parts_raw = (
            pl.scan_parquet(part_file)
            .filter(pl.col("event_id") == eid)
            .select(PARTICLE_FEATURES)
            .collect()
        )

        hits_expl = explode_tracker_hits(hits_raw).drop(columns=["event_id"])
        parts_expl = explode_particles(parts_raw).drop(columns=["event_id"])

        feather.write_feather(
            out_hit, hits_expl, compression=compression,
            compression_level=compression_level,
        )
        feather.write_feather(
            out_part, parts_expl, compression=compression,
            compression_level=compression_level,
        )
        n_written += 1

    return n_written


def main() -> None:
    p = argparse.ArgumentParser(description="Build feather cache for ColliderML.")
    p.add_argument("--process", default="ttbar")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-events", type=int, default=100000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--compression-level", type=int, default=3)
    args = p.parse_args()

    build_cache(
        data_dir=args.data_dir,
        process=args.process,
        pileup=args.pileup,
        cache_root=args.cache_dir,
        max_events=args.max_events,
        num_workers=args.num_workers,
        compression=args.compression,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the build script with 2 events**

```bash
source setup.sh && uv run python scripts/build_feather_cache.py \
  --data-dir /pscratch/sd/p/pmtuan/.cache/colliderml \
  --cache-dir /tmp/test_feather_cache \
  --max-events 2 --num-workers 1
```

Verify output:
```bash
echo "=== Cache contents ===" && \
ls /tmp/test_feather_cache/hits/ && \
ls /tmp/test_feather_cache/parts/ && \
python -c "import json; print(json.load(open('/tmp/test_feather_cache/meta.json')))" && \
echo "=== Reading back ===" && \
PYTHONPATH=src uv run python -c "
from seed_extension.data.cache import CachedColliderMLDataset
ds = CachedColliderMLDataset(cache_root='/tmp/test_feather_cache', event_ids=[0,1], stage='fit', min_track_hits=5, seed_strategy='fixed_innermost', target_vertices=200, predict_seed_hits=True)
print(f'len={len(ds)}')
s = ds[0]
print(f'hits shape: {s[\"hits\"].shape}, seeds shape: {s[\"seeds\"].shape}')
" && \
rm -rf /tmp/test_feather_cache
```

Expected: 2 `.feather` files in each directory, meta.json with correct columns, dataset can read both events.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_feather_cache.py
git commit -m "feat: add build_feather_cache.py — one-time cache builder from source Parquet"
```

---

### Task 4: Wire into training pipeline

**Files:**
- Modify: `src/seed_extension/train.py:62-99`
- Modify: `configs/cast_default.yaml`
- Modify: `configs/cast_pu10.yaml`

- [ ] **Step 1: Add cache_dir to both configs**

Add `cache_dir: null` to the `data` section:

In `configs/cast_default.yaml` after `pileup: pu200`:
```yaml
  cache_dir: null
```

In `configs/cast_pu10.yaml` after `pileup: pu200`:
```yaml
  cache_dir: null
```

- [ ] **Step 2: Wire CachedColliderMLDataModule in train.py**

Replace the `ColliderMLDataModule(...)` block (lines 62-99) with:

```python
    cache_dir = data_cfg.get("cache_dir")
    if cache_dir is not None:
        cache_dir = os.environ.get(
            "COLLIDERML_FEATHER_CACHE_DIR",
            cache_dir,
        )

    from seed_extension.data.cache import CachedColliderMLDataModule

    datamodule = CachedColliderMLDataModule(
        data_dir=default_datadir,
        cache_dir=cache_dir,
        process=data_cfg["process"],
        pileup=data_cfg["pileup"],
        max_train_events=data_cfg["max_train_events"],
        max_val_events=data_cfg["max_val_events"],
        max_test_events=data_cfg["max_test_events"],
        batch_size=1,
        num_workers=data_cfg["num_workers"],
        min_track_hits=data_cfg.get("min_track_hits", 5),
        min_pT=data_cfg.get("min_pT", 0.0),
        max_abs_eta=data_cfg.get("max_abs_eta", 4.0),
        seed_strategy=data_cfg.get("seed_strategy", "random_consecutive"),
        target_vertices=data_cfg.get("target_vertices", 200),
        primary_only=data_cfg.get("primary_only", False),
        predict_seed_hits=data_cfg.get("predict_seed_hits", False),
    )

    # Patch in prefetch_factor and persistent_workers when using Parquet
    # fallback (CachedColliderMLDataModule also wraps ColliderMLDataModule's
    # DataLoader methods).
    prefetch = int(data_cfg.get("prefetch_factor", 2))
    workers = data_cfg["num_workers"]
    from torch.utils.data import DataLoader

    _setup_orig = datamodule.setup

    def _setup_patched(stage=None):
        _setup_orig(stage)
        dl_kw = dict(batch_size=1, num_workers=workers,
                     prefetch_factor=prefetch, persistent_workers=True)
        datamodule.train_dataloader = lambda: DataLoader(
            datamodule.trainset, shuffle=True, drop_last=True, **dl_kw)
        datamodule.val_dataloader = lambda: DataLoader(
            datamodule.valset, shuffle=False, **dl_kw)

    datamodule.setup = _setup_patched
```

- [ ] **Step 3: Verify fallback still works (cache_dir=null)**

```bash
source setup.sh && uv run python -c "
import os, sys; sys.path.insert(0, 'src')
os.environ['COLLIDERML_DATA_DIR'] = '/pscratch/sd/p/pmtuan/.cache/colliderml'
import seed_extension.data
# Simulate config load
data_cfg = {'process': 'ttbar', 'pileup': 'pu200', 'max_train_events': 1, 'max_val_events': 0,
            'max_test_events': 0, 'num_workers': 0, 'cache_dir': None,
            'min_track_hits': 5, 'seed_strategy': 'fixed_innermost',
            'target_vertices': 200, 'predict_seed_hits': True}
from seed_extension.data.cache import CachedColliderMLDataModule
dm = CachedColliderMLDataModule(
    data_dir=os.environ['COLLIDERML_DATA_DIR'], cache_dir=None,
    process='ttbar', pileup='pu200', max_train_events=1,
    batch_size=1, num_workers=0,
    min_track_hits=5, seed_strategy='fixed_innermost',
    target_vertices=200, predict_seed_hits=True,
)
dm.setup('fit')
print(f'Fallback OK: {type(dm.trainset).__name__}, len={len(dm.trainset)}')
"
```
Expected: `Fallback OK: SeedExtensionDataset, len=1`

- [ ] **Step 4: Verify cached path works**

```bash
source setup.sh && PYTHONPATH=src uv run python scripts/build_feather_cache.py \
  --data-dir /pscratch/sd/p/pmtuan/.cache/colliderml \
  --cache-dir /tmp/test_feather_cache2 \
  --max-events 2 --num-workers 1 && \
PYTHONPATH=src uv run python -c "
import os, sys; sys.path.insert(0, 'src')
os.environ['COLLIDERML_DATA_DIR'] = '/pscratch/sd/p/pmtuan/.cache/colliderml'
import seed_extension.data
from seed_extension.data.cache import CachedColliderMLDataModule
dm = CachedColliderMLDataModule(
    data_dir=os.environ['COLLIDERML_DATA_DIR'], cache_dir='/tmp/test_feather_cache2',
    process='ttbar', pileup='pu200', max_train_events=2,
    batch_size=1, num_workers=0,
    min_track_hits=5, seed_strategy='fixed_innermost',
    target_vertices=200, predict_seed_hits=True,
)
dm.setup('fit')
print(f'Cached OK: {type(dm.trainset).__name__}, len={len(dm.trainset)}')
batch = next(iter(dm.train_dataloader()))
print(f'Batch keys: {list(batch.keys())}')
" && rm -rf /tmp/test_feather_cache2
```

Expected: `Cached OK: CachedColliderMLDataset, len=2`

- [ ] **Step 5: Commit**

```bash
git add src/seed_extension/train.py configs/cast_default.yaml configs/cast_pu10.yaml
git commit -m "feat: wire CachedColliderMLDataModule into training pipeline via cache_dir config"
```

---

### Task 5: Full test suite and lint

- [ ] **Step 1: Run the full test suite**

```bash
source setup.sh && uv run pytest tests/ -v --timeout=300 -x
```

Expected: all tests pass.

- [ ] **Step 2: Run lint**

```bash
make lint
```

Expected: clean compilation of all `.py` files.

- [ ] **Step 3: Commit any fixes (if needed)**

---

### Task 6: Prefatch factors + persistent_workers in CachedColliderMLDataModule DataLoaders

**Files:**
- Modify: `src/seed_extension/data/cache.py`

- [ ] **Step 1: Add prefetch_factor and persistent_workers to CachedColliderMLDataModule's DataLoaders**

Since `CachedColliderMLDataModule` has its own `train_dataloader` / `val_dataloader`, they don't get the `train.py` patch. Add `prefetch_factor` and `persistent_workers` support:

```python
    def __init__(
        self,
        data_dir: str,
        cache_dir: str | None = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        # ... existing init ...
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
```

Then update the DataLoader methods to include:
```python
    def train_dataloader(self):
        return DataLoader(
            self.trainset, batch_size=self.batch_size,
            num_workers=self._parent.num_workers, drop_last=True,
            shuffle=True, collate_fn=default_collate,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers,
        )
```

And similarly for `val_dataloader`. And update `train.py` to pass these from config:

```python
    datamodule = CachedColliderMLDataModule(
        ...
        prefetch_factor=prefetch,
        persistent_workers=(prefetch > 0),
        ...
    )
```

- [ ] **Step 2: Test**

```bash
source setup.sh && PYTHONPATH=src uv run python -c "
from seed_extension.data.cache import CachedColliderMLDataModule
# Test that parameters are stored
dm = CachedColliderMLDataModule(data_dir='/tmp', cache_dir=None, max_train_events=1, prefetch_factor=8, persistent_workers=True)
print(f'prefetch_factor={dm.prefetch_factor}, persistent_workers={dm.persistent_workers}')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/seed_extension/data/cache.py src/seed_extension/train.py
git commit -m "feat: add prefetch_factor and persistent_workers to CachedColliderMLDataModule DataLoaders"
```
