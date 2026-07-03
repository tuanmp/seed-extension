"""SeedExtensionDataset — CAST seed-extension data processing."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import torch

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

    _HIT_FEATURE_COLS = ["x", "y", "z", "layer_id", "volume_id", "detector"]

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
        hits_raw = explode_tracker_hits(
            pl.scan_parquet(hit_file)
            .filter(pl.col("event_id") == event_id)
            .select(TRACKER_HIT_FEATURES)
            .collect()
        )
        part_file = self.part_file_map[event_id]
        parts_raw = explode_particles(
            pl.scan_parquet(part_file)
            .filter(pl.col("event_id") == event_id)
            .select(PARTICLE_FEATURES)
            .collect()
        )
        return hits_raw, parts_raw

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def _process_event(
        self, hits_raw: Any, parts_raw: Any, event_idx: int,
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

        part_df = parts_raw.copy()
        hit_df = hits_raw.copy()

        part_df, hit_df = self._subsample_pileup(
            part_df, hit_df, target_vertices, event_idx,
        )
        part_df, hit_df = self._filter_particles(
            part_df, hit_df, primary_only, min_track_hits, min_pT, max_abs_eta,
        )

        seed_coords, seed_pids, kinematics = self._build_seeds(
            part_df, hit_df, seed_strategy, n_seed_hits, event_idx,
        )

        hit_features, hit_pids = self._build_hit_features(hit_df)

        return self._assemble_sample(
            hit_features, hit_pids, seed_coords, seed_pids,
            kinematics, predict_seed_hits, event_idx,
        )

    # ------------------------------------------------------------------
    # Step 1: pileup subsampling
    # ------------------------------------------------------------------

    @staticmethod
    def _subsample_pileup(
        part_df: "pd.DataFrame", hit_df: "pd.DataFrame",  # noqa: F821
        target_vertices: int, event_idx: int,
    ) -> tuple:
        all_vertices = np.unique(part_df["vertex_primary"].values)
        n_available = len(all_vertices)

        if target_vertices <= 0 or target_vertices >= n_available:
            return part_df, hit_df

        rng = np.random.RandomState(event_idx * 10007 + 17)
        kept_vertices = set(rng.choice(all_vertices, size=target_vertices, replace=False))
        part_df = part_df[part_df["vertex_primary"].isin(kept_vertices)]
        kept_pids = part_df["particle_id"].unique()
        hit_df = hit_df[hit_df["particle_id"].isin(kept_pids)]
        return part_df, hit_df

    # ------------------------------------------------------------------
    # Step 2: particle-level filters
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_particles(
        part_df: "pd.DataFrame", hit_df: "pd.DataFrame",  # noqa: F821
        primary_only: bool, min_track_hits: int,
        min_pT: float, max_abs_eta: float,
    ) -> tuple:
        if primary_only:
            part_df = part_df[part_df["primary"] == 1]

        # Use compute_pT_eta (cheaper than full kinematics).
        if min_pT > 0.0 or max_abs_eta < 10.0:
            pT, eta = compute_pT_eta(
                part_df["px"].values, part_df["py"].values,
                part_df["pz"].values,
            )
            part_df = part_df[(eta >= -max_abs_eta) & (eta <= max_abs_eta) & (pT >= min_pT)]

        # Min hits per particle.
        hit_counts = hit_df.groupby("particle_id").size()
        valid_pids = hit_counts[hit_counts >= min_track_hits].index
        part_df = part_df[part_df["particle_id"].isin(valid_pids)].reset_index(drop=True)
        # Filter hits to surviving particles only.
        kept_pids = part_df["particle_id"].unique()
        hit_df = hit_df[hit_df["particle_id"].isin(kept_pids)]
        return part_df, hit_df

    # ------------------------------------------------------------------
    # Step 3: seed construction
    # ------------------------------------------------------------------

    def _build_seeds(
        self, part_df: "pd.DataFrame", hit_df: "pd.DataFrame",  # noqa: F821
        seed_strategy: str, n_seed_hits: int, event_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        use_random = seed_strategy == "random_consecutive" and self.stage in ("fit",)
        if use_random:
            rng = np.random.RandomState(event_idx * 10007 + 42)
            return build_seeds_random_consecutive(
                part_df, hit_df, n_seed_hits=n_seed_hits, rng=rng,
            )
        return build_seeds_fixed(part_df, hit_df, n_seed_hits=n_seed_hits)

    # ------------------------------------------------------------------
    # Step 4: hit feature tensor
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hit_features(
        hit_df: "pd.DataFrame",  # noqa: F821
    ) -> tuple[np.ndarray, np.ndarray]:
        features = hit_df[SeedExtensionDataset._HIT_FEATURE_COLS].to_numpy(dtype=np.float32)
        pids = hit_df["particle_id"].to_numpy(dtype=np.int64)
        return features, pids

    # ------------------------------------------------------------------
    # Step 5: assemble sample dict
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_sample(
        hit_features: np.ndarray, hit_pids: np.ndarray,
        seed_coords: np.ndarray, seed_pids: np.ndarray,
        kinematics: np.ndarray, predict_seed_hits: bool,
        event_idx: int,
    ) -> dict[str, Any]:
        N_s = len(seed_pids)
        N_h = len(hit_pids)
        n_feat = hit_features.shape[1]

        if N_s == 0 or N_h == 0:
            return {
                "hits": torch.zeros(N_h, n_feat, dtype=torch.float32),
                "seeds": torch.zeros(N_s, seed_coords.shape[1], dtype=torch.float32),
                "targets": torch.zeros(N_s, N_h, dtype=torch.float32),
                "seed_particle_ids": torch.zeros(N_s, dtype=torch.int64),
                "hit_particle_ids": torch.from_numpy(hit_pids),
                "kinematics": torch.zeros(N_s, 6, dtype=torch.float32),
                "event_idx": event_idx,
            }

        targets = (seed_pids[:, None] == hit_pids[None, :]).astype(np.float32)

        if not predict_seed_hits:
            targets = SeedExtensionDataset._mask_seed_hits(
                targets, seed_coords, hit_features,
            )

        return {
            "hits": torch.from_numpy(hit_features),
            "seeds": torch.from_numpy(seed_coords),
            "targets": torch.from_numpy(targets),
            "seed_particle_ids": torch.from_numpy(seed_pids),
            "hit_particle_ids": torch.from_numpy(hit_pids),
            "kinematics": torch.from_numpy(kinematics),
            "event_idx": event_idx,
        }

    # ------------------------------------------------------------------
    # Mask seed hit positions from the target matrix.
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_seed_hits(
        targets: np.ndarray, seed_coords: np.ndarray, hit_features: np.ndarray,
    ) -> np.ndarray:
        hit_xyz = hit_features[:, :3]
        n_seed_hits = seed_coords.shape[1] // 3

        for k in range(n_seed_hits):
            sx = seed_coords[:, [k * 3, k * 3 + 1, k * 3 + 2]]  # (N_s, 3)
            diff = sx[:, None, :] - hit_xyz[None, :, :]           # (N_s, N_h, 3)
            dist = np.sqrt(np.sum(diff * diff, axis=-1))          # (N_s, N_h)
            closest = dist.argmin(axis=1)
            matched = np.where(np.min(dist, axis=1) < 1e-4)[0]
            for s in matched:
                targets[s, closest[s]] = 0.0

        return targets
