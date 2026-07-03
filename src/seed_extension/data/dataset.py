"""SeedExtensionDataset — CAST seed-extension data processing."""

from typing import Any

import numpy as np
import torch

from colliderml_dataloader.dataset import ColliderMLDataset


def compute_kinematics(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray,
    d0: np.ndarray, z0: np.ndarray,
) -> np.ndarray:
    """Compute particle kinematics: [eta, pT, d0, z0, theta, phi].

    Returns array of shape (N, 6).
    """
    pT = np.sqrt(px**2 + py**2)
    p = np.sqrt(pT**2 + pz**2)
    # eta = arsinh(pz / pT).  Use this form which is stable for all pT > 0.
    # When pT == 0, eta is sign(pz) * inf; we clip to a large finite value.
    eta = np.where(
        pT > 1e-9,
        np.arcsinh(pz / np.clip(pT, 1e-9, None)),
        np.sign(pz) * 10.0,
    )
    theta = np.arctan2(pT, pz)
    phi = np.arctan2(py, px)
    return np.stack([eta, pT, d0, z0, theta, phi], axis=1)


def build_seeds_fixed(
    part_df: "pd.DataFrame", hit_df: "pd.DataFrame", n_seed_hits: int = 3,  # noqa: F821
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build seeds using the innermost (lowest layer_id) hits per particle.

    Uses polars group-by aggregation instead of pandas iterrows.
    Returns (seed_coords, seed_pids, kinematics).
    """
    import polars as pl

    h_pl = pl.from_pandas(hit_df)
    p_pl = pl.from_pandas(part_df)

    joined = (
        h_pl
        .join(
            p_pl.select(["particle_id", "px", "py", "pz", "perigee_d0", "perigee_z0"]),
            on="particle_id", how="inner",
        )
        .sort(["particle_id", "layer_id"])
        .group_by("particle_id", maintain_order=True)
        .agg([
            pl.col("x", "y", "z"),
            pl.col("px", "py", "pz", "perigee_d0", "perigee_z0").first(),
        ])
        .filter(pl.col("x").list.len() >= n_seed_hits)
    )

    N = len(joined)
    if N == 0:
        return (
            np.empty((0, n_seed_hits * 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
        )

    xl, yl, zl = joined["x"].to_list(), joined["y"].to_list(), joined["z"].to_list()
    coords = np.empty((N, n_seed_hits * 3), dtype=np.float32)
    for k in range(n_seed_hits):
        coords[:, k * 3] = np.array([v[k] for v in xl], dtype=np.float32)
        coords[:, k * 3 + 1] = np.array([v[k] for v in yl], dtype=np.float32)
        coords[:, k * 3 + 2] = np.array([v[k] for v in zl], dtype=np.float32)

    px = joined["px"].to_numpy().astype(np.float64)
    py = joined["py"].to_numpy().astype(np.float64)
    pz = joined["pz"].to_numpy().astype(np.float64)
    d0 = joined["perigee_d0"].to_numpy().astype(np.float64)
    z0 = joined["perigee_z0"].to_numpy().astype(np.float64)
    kinematics = compute_kinematics(px, py, pz, d0, z0).astype(np.float32)
    pids = joined["particle_id"].to_numpy().astype(np.int64)

    return coords, pids, kinematics


def build_seeds_random_consecutive(
    part_df: "pd.DataFrame", hit_df: "pd.DataFrame",  # noqa: F821
    n_seed_hits: int = 3,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build seeds from randomly-chosen consecutive-layer hit groups.

    Uses polars group-by for the grouping, then numpy for per-particle
    consecutive-layer logic and random selection.
    """
    import polars as pl

    if rng is None:
        rng = np.random.RandomState(0)

    h_pl = pl.from_pandas(hit_df)
    p_pl = pl.from_pandas(part_df)

    joined = (
        h_pl
        .join(
            p_pl.select(["particle_id", "px", "py", "pz", "perigee_d0", "perigee_z0"]),
            on="particle_id", how="inner",
        )
        .sort(["particle_id", "layer_id"])
        .group_by("particle_id", maintain_order=True)
        .agg([
            pl.col("x", "y", "z", "layer_id"),
            pl.col("px", "py", "pz", "perigee_d0", "perigee_z0").first(),
        ])
        .filter(pl.col("x").list.len() >= n_seed_hits)
    )

    N = len(joined)
    if N == 0:
        return (
            np.empty((0, n_seed_hits * 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
        )

    coords = np.empty((N, n_seed_hits * 3), dtype=np.float32)
    for i in range(N):
        x = np.array(joined["x"][i].to_list(), dtype=np.float32)
        y = np.array(joined["y"][i].to_list(), dtype=np.float32)
        z = np.array(joined["z"][i].to_list(), dtype=np.float32)
        layers = np.array(joined["layer_id"][i].to_list(), dtype=np.int32)

        # Find consecutive groups
        starts = []
        for s in range(len(layers) - n_seed_hits + 1):
            if np.all(np.diff(layers[s:s + n_seed_hits]) == 1):
                starts.append(s)

        start = int(rng.choice(starts)) if starts else 0
        for k in range(n_seed_hits):
            coords[i, k * 3] = x[start + k]
            coords[i, k * 3 + 1] = y[start + k]
            coords[i, k * 3 + 2] = z[start + k]

    px = joined["px"].to_numpy().astype(np.float64)
    py = joined["py"].to_numpy().astype(np.float64)
    pz = joined["pz"].to_numpy().astype(np.float64)
    d0 = joined["perigee_d0"].to_numpy().astype(np.float64)
    z0 = joined["perigee_z0"].to_numpy().astype(np.float64)
    kinematics = compute_kinematics(px, py, pz, d0, z0).astype(np.float32)
    pids = joined["particle_id"].to_numpy().astype(np.int64)

    return coords, pids, kinematics


class SeedExtensionDataset(ColliderMLDataset):
    """CAST seed-extension dataset — seed construction, target matrices, kinematics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable the unbounded per-worker _raw_cache inherited from
        # ColliderMLDataset.  At 50k events each cached entry is ~20 MB
        # of exploded DataFrames, which would OOM every worker.
        # Parquet data stays in the kernel page cache after first read;
        # the polars explode overhead (~150 ms/event) is acceptable and
        # can be absorbed by DataLoader pre-fetching.
        self._raw_cache = {}

    def __getitem__(self, idx):
        # Bypass parent's caching — always load from Parquet.
        # Kernel page cache amortises disk I/O after epoch 0.
        event_id = self.event_ids[idx]
        hits_raw, parts_raw = self._load_event(event_id)
        return self._process_event(hits_raw, parts_raw, idx)

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

        hit_features, hit_pids = self._build_hit_features(hit_df, n_seed_hits)

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
        """Keep a random subset of vertices, discarding all others."""
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

        # Kinematic cuts — compute kinematics once for filtering
        if min_pT > 0.0 or max_abs_eta < 10.0:
            kin = compute_kinematics(
                part_df["px"].values, part_df["py"].values,
                part_df["pz"].values, part_df["perigee_d0"].values,
                part_df["perigee_z0"].values,
            )
            eta = kin[:, 0]
            pT = kin[:, 1]
            eta_mask = (np.abs(eta) <= max_abs_eta)
            pT_mask = (pT >= min_pT)
            part_df = part_df[eta_mask & pT_mask]

        # Min hits per particle
        hit_counts = hit_df.groupby("particle_id").size()
        valid_pids = hit_counts[hit_counts >= min_track_hits].index
        part_df = part_df[part_df["particle_id"].isin(valid_pids)].reset_index(drop=True)
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

    _HIT_FEATURE_COLS = ["x", "y", "z", "layer_id", "volume_id", "detector"]

    @staticmethod
    def _build_hit_features(
        hit_df: "pd.DataFrame", n_seed_hits: int,  # noqa: F821
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

    @staticmethod
    def _mask_seed_hits(
        targets: np.ndarray, seed_coords: np.ndarray, hit_features: np.ndarray,
    ) -> np.ndarray:
        hit_xy = hit_features[:, :3]
        expected_zero = len(seed_coords) * 3
        zeroed = 0
        for s, sc in enumerate(seed_coords):
            for k in range(3):
                sx, sy, sz = sc[k*3], sc[k*3+1], sc[k*3+2]
                dist = np.sqrt(
                    (hit_xy[:, 0] - sx)**2 +
                    (hit_xy[:, 1] - sy)**2 +
                    (hit_xy[:, 2] - sz)**2
                )
                closest = np.argmin(dist)
                if dist[closest] < 1e-4:
                    targets[s, closest] = 0.0
                    zeroed += 1
        if zeroed < expected_zero:
            print(
                f"WARNING: Only {zeroed}/{expected_zero} seed hits matched for exclusion. "
                f"Some seed hits may remain in targets."
            )
        return targets
