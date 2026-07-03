"""Tests for the feather cache dataset and DataModule."""

import json
import os
import tempfile
import shutil
from pathlib import Path

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
    build_shard_index,
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


def build_mini_cache(n_events: int = 2) -> tuple[Path, list[int]]:
    """Build a tiny feather cache with *n_events* events.

    Returns ``(cache_root, event_ids)`` — the actual event IDs written.
    """
    cache_root = Path(tempfile.mkdtemp(prefix="feather_cache_test_"))
    hits_dir = cache_root / "hits"
    parts_dir = cache_root / "parts"
    hits_dir.mkdir(parents=True)
    parts_dir.mkdir(parents=True)

    hit_shard_dir = shard_dir_for(DATA_DIR, "ttbar", "pu200", "tracker_hits")
    part_shard_dir = shard_dir_for(DATA_DIR, "ttbar", "pu200", "particles")

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

        feather.write_feather(
            hits_expl, hits_dir / f"{eid}.feather", compression="zstd",
        )
        feather.write_feather(
            parts_expl, parts_dir / f"{eid}.feather", compression="zstd",
        )

    meta = {
        "version": 1,
        "process": "ttbar",
        "pileup": "pu200",
        "hit_cols": TRACKER_HIT_FEATURES[:],
        "part_cols": PARTICLE_FEATURES[:],
        "n_events": len(event_ids),
    }
    with (cache_root / "meta.json").open("w") as f:
        json.dump(meta, f)

    return cache_root, event_ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCachedDataset:
    @requires_data
    def test_returns_same_keys_as_seed_extension(self):
        cache_root, event_ids = build_mini_cache(2)
        try:
            cached_ds = CachedColliderMLDataset(
                cache_root=cache_root,
                event_ids=event_ids,
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
        finally:
            shutil.rmtree(cache_root)

    @requires_data
    def test_same_output_as_seed_extension(self):
        """Cached dataset must produce identical output to SeedExtensionDataset
        when run on the same event with same seed/rand config."""
        cache_root, (eid,) = build_mini_cache(1)
        try:
            # Ensure deterministic seed building (fixed_innermost, same event_idx).
            cached_ds = CachedColliderMLDataset(
                cache_root=cache_root, event_ids=[eid], stage="validate",
                min_track_hits=5, seed_strategy="fixed_innermost",
                target_vertices=200, primary_only=False, predict_seed_hits=True,
            )
            cached_sample = cached_ds[0]

            # Re-read the same event via the Parquet path (same event_id as index 0).
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
        finally:
            shutil.rmtree(cache_root)


class TestCachedDataModule:
    @requires_data
    def test_cache_exists_uses_cached_path(self):
        cache_root, event_ids = build_mini_cache(2)
        try:
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
        finally:
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
        cache_root, _ = build_mini_cache(1)
        try:
            # Corrupt meta.json
            meta_path = cache_root / "meta.json"
            mid = json.loads(meta_path.read_text())
            mid["hit_cols"] = ["x", "wrong_column"]
            meta_path.write_text(json.dumps(mid))

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
        finally:
            shutil.rmtree(cache_root)
