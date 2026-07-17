"""SeedExtensionDataset — CAST seed-extension data processing."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch
import pandas as pd
import time

from colliderml_dataloader.dataset import ColliderMLDataset
from colliderml.polars import explode_particles, explode_tracker_hits

from seed_extension.data.seed_utils import (
    PARTICLE_FEATURES,
    TRACKER_HIT_FEATURES,
    build_seeds_fixed,
    build_seeds_random_consecutive,
    compute_pT_eta,
)


class SeedExtensionDataset(ColliderMLDataset):
    """CAST seed-extension dataset — seed construction, target matrices, kinematics."""

    _HIT_DETECTOR_COLS = ["layer_id", "detector"]
    _HIT_COORDS_COLS = ["x", "y", "z", "r", "sin_phi", "cos_phi", "phi", "theta"]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Disable the unbounded per-worker _raw_cache inherited from
        # ColliderMLDataset.  At 50k events each cached entry is ~20 MB
        # of exploded DataFrames, which would OOM every worker.
        # Parquet data stays in the kernel page cache after first read;
        # the polars explode overhead (~150 ms/event) is acceptable and
        # can be absorbed by DataLoader pre-fetching.
        self._raw_cache = {}

    # ------------------------------------------------------------------
    # I/O — overridden to use seed_extension feature lists
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):
        event_id = self.event_ids[idx]
        hits_raw, parts_raw = self._load_event(event_id)
        return self._process_event(hits_raw, parts_raw, idx)

    def _load_event(self, event_id: int):
        """Override parent — uses seed_extension feature lists."""
        hit_file = self.hit_file_map[event_id]
        hits_raw = pl.scan_parquet(hit_file) \
        .filter(pl.col("event_id") == event_id) \
        .select(TRACKER_HIT_FEATURES) \
        .collect()

        list_cols = [c for c, dt in hits_raw.collect_schema().items() if c != "event_id" and isinstance(dt, pl.List)]
        hits_raw = hits_raw.explode(list_cols, empty_as_null=True)

        part_file = self.part_file_map[event_id]
        parts_raw = pl.scan_parquet(part_file) \
        .filter(pl.col("event_id") == event_id) \
        .select(PARTICLE_FEATURES) \
        .collect()

        list_cols = [c for c, dt in parts_raw.collect_schema().items() if c != "event_id" and isinstance(dt, pl.List)]

        parts_raw = parts_raw.explode(list_cols, empty_as_null=True)

        return hits_raw, parts_raw

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def _process_event(
        self, hits_raw: pl.DataFrame, parts_raw: pl.DataFrame, event_idx: int,
    ) -> dict[str, Any]:
        kwargs = self._kwargs
        min_track_hits: int = kwargs.get("min_track_hits", 5)
        seed_strategy: str = kwargs.get("seed_strategy", "random_consecutive")
        target_vertices: int = kwargs.get("target_vertices", 200)
        primary_only: bool = kwargs.get("primary_only", False)
        predict_seed_hits: bool = kwargs.get("predict_seed_hits", False)
        n_seed_hits: int = kwargs.get("n_seed_hits", 3)
        min_pT: float = kwargs.get("min_pT", 0.0)
        max_abs_eta: float = kwargs.get("max_abs_eta", 4.0)

        part_df = parts_raw
        hit_df = hits_raw

        part_df, hit_df = self._subsample_pileup(
            part_df, hit_df, target_vertices, event_idx,
        )

        # preprocess hits: scale and add cylindrical coordinates
        hit_df = self._scale_hits(hit_df)
        hit_df = self._add_cylindrical_coords(hit_df)

        # add pt, eta to particle
        part_df = self._add_particle_kinematics(part_df)

        # filter out untargeted particles
        part_df, hit_df = self._filter_particles(
            part_df, hit_df, primary_only, min_track_hits, min_pT, max_abs_eta,
        )

        # add hit_id 
        hit_df = hit_df \
            .with_columns(pl.arange(0, pl.len()).alias("hit_id")) \

        seed_hit_ids, kinematics, seed_pids = self._build_seeds(
            part_df, hit_df, seed_strategy, n_seed_hits, event_idx,
        )

        hit_coords, hit_pids, hit_detector_info = self._build_hit_features(hit_df)

        return self._assemble_sample(
            hit_coords, hit_pids, hit_detector_info, seed_hit_ids, kinematics, seed_pids,
            predict_seed_hits, event_idx,
        )

    # ------------------------------------------------------------------
    # Step 0: scale hits
    # ------------------------------------------------------------------
    @staticmethod
    def _scale_hits(hit_df: pl.DataFrame) -> pl.DataFrame:  # noqa: F821
        """Scale hit coordinates to ~[-1, 1] range."""
        # hit_df = hit_df.copy()
        hit_df = hit_df.with_columns([
            (pl.col("x") / 1000.0).alias("x"),
            (pl.col("y") / 1000.0).alias("y"),
            (pl.col("z") / 1000.0).alias("z"),
        ])
        return hit_df

    @staticmethod
    def _unscale_hits(hit_df: pl.DataFrame) -> pl.DataFrame:  # noqa: F821
        """Unscale hit coordinates to original units."""
        # hit_df = hit_df.copy()
        hit_df = hit_df.with_columns([
            (pl.col("x") * 1000.0).alias("x"),
            (pl.col("y") * 1000.0).alias("y"),
            (pl.col("z") * 1000.0).alias("z"),
        ])
        return hit_df

    @staticmethod
    def _add_cylindrical_coords(hit_df: pl.DataFrame) -> pl.DataFrame:  # noqa: F821
        """Add cylindrical coordinates (r, phi) to hit DataFrame."""
        # hit_df = hit_df.copy()
        hit_df = hit_df.with_columns([
            (pl.col("x") ** 2 + pl.col("y") ** 2).sqrt().alias("r"),
        ])
        hit_df = hit_df.with_columns([
            (pl.col("x") / pl.col("r")).alias("sin_phi"),
            (pl.col("y") / pl.col("r")).alias("cos_phi"),
        ])
        phi = np.arctan2(hit_df["y"].to_numpy(), hit_df["x"].to_numpy())
        theta = np.arctan2(hit_df["r"].to_numpy(), hit_df["z"].to_numpy())
        hit_df = hit_df.with_columns([
            pl.Series("phi", phi.astype(np.float32)),
            pl.Series("theta", theta.astype(np.float32)),
        ])
        return hit_df

    @staticmethod
    def _add_particle_kinematics(part_df: pl.DataFrame) -> pl.DataFrame:  # noqa: F821
        """Add pT and eta to particle DataFrame."""
        # part_df = part_df.copy()
        px = part_df["px"].to_numpy().astype(np.float64)
        py = part_df["py"].to_numpy().astype(np.float64)
        pz = part_df["pz"].to_numpy().astype(np.float64)
        pT, eta = compute_pT_eta(px, py, pz)
        part_df = part_df.with_columns([
            pl.Series("pT", pT.astype(np.float32)),
            pl.Series("eta", eta.astype(np.float32)),
        ])
        return part_df

    # ------------------------------------------------------------------
    # Step 1: pileup subsampling
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample_pileup(
        part_df: pl.DataFrame, hit_df: pl.DataFrame,  # noqa: F821
        target_vertices: int, event_idx: int,
    ) -> tuple:
        all_vertices = np.unique(part_df["vertex_primary"].to_numpy())
        n_available = len(all_vertices)

        if target_vertices <= 0 or target_vertices >= n_available:
            return part_df, hit_df
        
        rng = np.random.RandomState(event_idx * 10007 + time.time_ns() % 1000000)
        non_hs_vertices = all_vertices[all_vertices != 1]
        kept_vertices = rng.choice(non_hs_vertices, size=target_vertices-1, replace=False)
        kept_vertices = [1] + list(kept_vertices)
        part_df = part_df.filter(pl.col("vertex_primary").is_in(kept_vertices))
        kept_pids = np.unique(part_df["particle_id"].to_numpy())
        hit_df = hit_df.filter(pl.col("particle_id").is_in(kept_pids))
        return part_df, hit_df

    # ------------------------------------------------------------------
    # Step 2: particle-level filters
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_particles(
        part_df: pl.DataFrame, hit_df: pl.DataFrame,  # noqa: F821
        primary_only: bool, min_track_hits: int,
        min_pT: float, max_abs_eta: float,
    ) -> tuple:
        if primary_only:
            part_df = part_df.filter(pl.col("primary") == 1)

        # Use compute_pT_eta (cheaper than full kinematics).
        if min_pT > 0.0 or max_abs_eta < 10.0:
            part_df = part_df.filter((pl.col("eta") >= -max_abs_eta) & (pl.col("eta") <= max_abs_eta) & (pl.col("pT") >= min_pT))

        # Min hits per particle.
        # hit_counts = hit_df.groupby("particle_id").len()
        hit_df = hit_df.with_columns(pl.len().over("particle_id").alias("hit_count"))
        valid_pids = hit_df.filter(pl.col("hit_count") >= min_track_hits)["particle_id"].unique().implode()
        part_df = part_df.filter(pl.col("particle_id").is_in(valid_pids))
        hit_df.drop_in_place("hit_count")
        # Filter hits to surviving particles only.
        # Actually no. Must keep all hits, even if the particle is filtered out,
        # because otherwise it would be cheating. In reality we don't get to which hits belongs to a target particle.
        # kept_pids = part_df["particle_id"].unique()
        # hit_df = hit_df[hit_df["particle_id"].isin(kept_pids)]
        return part_df, hit_df

    # ------------------------------------------------------------------
    # Step 3: seed construction
    # ------------------------------------------------------------------

    def _build_seeds(
        self, part_df: pl.DataFrame, hit_df: pl.DataFrame,  # noqa: F821
        seed_strategy: str, n_seed_hits: int, event_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        use_random = (seed_strategy == "random_consecutive") 
        if use_random:
            rng = np.random.RandomState(event_idx * 10007 + time.time_ns() % 1000000)
            return build_seeds_random_consecutive(
                part_df, hit_df, n_seed_hits=n_seed_hits, rng=rng,
            )
        return build_seeds_fixed(part_df, hit_df, n_seed_hits=n_seed_hits)

    # ------------------------------------------------------------------
    # Step 4: hit feature tensor
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hit_features(
        hit_df: pl.DataFrame,  # noqa: F821
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = hit_df.select(SeedExtensionDataset._HIT_COORDS_COLS).to_numpy().astype(np.float32)
        detector_info = hit_df.select(SeedExtensionDataset._HIT_DETECTOR_COLS).to_numpy().astype(np.float32)
        pids = hit_df.select("particle_id").to_numpy().astype(np.int64).flatten()
        return coords, pids, detector_info

    # ------------------------------------------------------------------
    # Step 5: assemble sample dict
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_sample(
        hit_features: np.ndarray, hit_pids: np.ndarray, hit_detector_info: np.ndarray, 
        seed_hit_ids: np.ndarray, kinematics: np.ndarray, seed_pids: np.ndarray,
        predict_seed_hits: bool,
        event_idx: int,
    ) -> dict[str, Any]:
        N_s = len(seed_pids)
        N_h = len(hit_pids)
        n_hit_dim = len(SeedExtensionDataset._HIT_COORDS_COLS)

        if N_s == 0 or N_h == 0:
            return {
                "hits": torch.zeros(N_h, n_hit_dim, dtype=torch.float32),
                "seeds": torch.zeros(N_s, seed_hit_ids.shape[1], dtype=torch.int64),
                "targets": torch.zeros(N_s, N_h, dtype=torch.float32),
                "seed_particle_ids": torch.zeros(N_s, dtype=torch.int64),
                "hit_particle_ids": torch.from_numpy(hit_pids),
                "hit_detector_info": torch.from_numpy(hit_detector_info),
                "kinematics": torch.zeros(N_s, 6, dtype=torch.float32),
                "event_idx": event_idx,
            }

        targets = (seed_pids[:, None] == hit_pids[None, :]).astype(np.float32)

        if not predict_seed_hits:
            targets = SeedExtensionDataset._mask_seed_hits(
                targets, seed_hit_ids, hit_features,
            )

        return {
            "hits": torch.from_numpy(hit_features),
            "seeds": torch.from_numpy(seed_hit_ids),
            "targets": torch.from_numpy(targets),
            "seed_particle_ids": torch.from_numpy(seed_pids),
            "hit_particle_ids": torch.from_numpy(hit_pids),
            "hit_detector_info": torch.from_numpy(hit_detector_info),
            "kinematics": torch.from_numpy(kinematics),
            "event_idx": event_idx,
        }

    # ------------------------------------------------------------------
    # Mask seed hit positions from the target matrix.
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_seed_hits(
        targets: np.ndarray, seed_hit_ids: np.ndarray, hit_features: np.ndarray,
    ) -> np.ndarray:
        
        # seed_hit_ids: (N_s, 3) indices of the seed hits in the hit_features tensor
        # targets: (N_s, N_h) binary matrix indicating which hits belong to the same particle as the seed hits
        # hit_features: (N_h, F_hit) features of all hits
        # For each seed, set the target values for the seed hits to 0
        # so that the model does not learn to predict the seed hits as targets.
        
        targets[
            np.arange(targets.shape[0], device=targets.device)[:, None],
            seed_hit_ids,
        ] = 0

        return targets
