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
    cos_theta = np.clip(pz / np.clip(p, 1e-12, None), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    eta = -np.log(np.tan(np.clip(theta, 1e-12, np.pi - 1e-12) / 2.0))
    phi = np.arctan2(py, px)
    return np.stack([eta, pT, d0, z0, theta, phi], axis=1)


def build_seeds_fixed(
    part_df: "pd.DataFrame", hit_df: "pd.DataFrame", n_seed_hits: int = 3,  # noqa: F821
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build seeds using the innermost (lowest layer_id) hits per particle.

    Returns (seed_coords, seed_pids, kinematics).
    """
    import pandas as pd

    seed_coords_list = []
    seed_pids_list = []
    kine_list = []

    for _, row in part_df.iterrows():
        pid = row["particle_id"]
        hits = hit_df[hit_df["particle_id"] == pid].sort_values("layer_id")
        if len(hits) < n_seed_hits:
            continue
        innermost = hits.iloc[:n_seed_hits]
        coords = innermost[["x", "y", "z"]].to_numpy(dtype=np.float32).ravel()
        seed_coords_list.append(coords)
        seed_pids_list.append(pid)
        kine_list.append(compute_kinematics(
            np.array([row["px"]]), np.array([row["py"]]),
            np.array([row["pz"]]), np.array([row["perigee_d0"]]),
            np.array([row["perigee_z0"]]),
        )[0])

    N = len(seed_coords_list)
    if N == 0:
        return (
            np.empty((0, n_seed_hits * 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
        )

    return (
        np.stack(seed_coords_list).astype(np.float32),
        np.array(seed_pids_list, dtype=np.int64),
        np.stack(kine_list).astype(np.float32),
    )


def build_seeds_random_consecutive(
    part_df: "pd.DataFrame", hit_df: "pd.DataFrame",  # noqa: F821
    n_seed_hits: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build seeds from randomly-chosen consecutive-layer hit groups.

    For each particle, finds groups of n_seed_hits consecutive-layer hits,
    randomly picks one group. Falls back to first n_seed_hits if no
    consecutive group exists.

    Returns (seed_coords, seed_pids, kinematics).
    """
    import pandas as pd

    if rng is None:
        rng = np.random.default_rng()

    seed_coords_list = []
    seed_pids_list = []
    kine_list = []

    for _, row in part_df.iterrows():
        pid = row["particle_id"]
        hits = hit_df[hit_df["particle_id"] == pid].sort_values("layer_id")
        if len(hits) < n_seed_hits:
            continue

        layer_ids = hits["layer_id"].to_numpy()
        coords_all = hits[["x", "y", "z"]].to_numpy(dtype=np.float32)

        # Find all groups of n_seed_hits consecutive layers.
        groups = []
        for i in range(len(layer_ids) - n_seed_hits + 1):
            window = layer_ids[i:i + n_seed_hits]
            if np.all(np.diff(window) == 1):
                groups.append(i)

        if groups:
            start = int(rng.choice(groups))
        else:
            start = 0

        chosen = coords_all[start:start + n_seed_hits]
        seed_coords_list.append(chosen.ravel())
        seed_pids_list.append(pid)
        kine_list.append(compute_kinematics(
            np.array([row["px"]]), np.array([row["py"]]),
            np.array([row["pz"]]), np.array([row["perigee_d0"]]),
            np.array([row["perigee_z0"]]),
        )[0])

    N = len(seed_coords_list)
    if N == 0:
        return (
            np.empty((0, n_seed_hits * 3), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0, 6), dtype=np.float32),
        )

    return (
        np.stack(seed_coords_list).astype(np.float32),
        np.array(seed_pids_list, dtype=np.int64),
        np.stack(kine_list).astype(np.float32),
    )


class SeedExtensionDataset(ColliderMLDataset):
    """CAST seed-extension dataset — seed construction, target matrices, kinematics."""

    def _process_event(
        self, hits_raw: Any, parts_raw: Any, event_idx: int,
    ) -> dict[str, Any]:
        import pandas as pd

        kwargs = self._kwargs
        min_track_hits: int = kwargs.get("min_track_hits", 5)
        seed_strategy: str = kwargs.get("seed_strategy", "fixed_innermost")
        target_vertices: int | None = kwargs.get("target_vertices", None)
        primary_only: bool = kwargs.get("primary_only", False)
        predict_seed_hits: bool = kwargs.get("predict_seed_hits", False)
        n_seed_hits: int = kwargs.get("n_seed_hits", 3)

        # ---- 1. Pileup subsampling & primary filter ----
        part_df = parts_raw.copy()
        hit_df = hits_raw.copy()

        if target_vertices is not None and target_vertices < 200:
            part_df = part_df[part_df["vertex_primary"] <= target_vertices]
            kept_pids = part_df["particle_id"].unique()
            hit_df = hit_df[hit_df["particle_id"].isin(kept_pids)]

        if primary_only:
            part_df = part_df[part_df["primary"] == 1]

        # ---- 2. Filter particles with >= min_track_hits hits ----
        hit_counts = hit_df.groupby("particle_id").size()
        valid_pids = hit_counts[hit_counts >= min_track_hits].index
        part_df = part_df[part_df["particle_id"].isin(valid_pids)].reset_index(drop=True)

        # ---- 3. Build seeds ----
        use_consecutive = (
            seed_strategy == "random_consecutive"
            and self.stage in ("fit",)
        )
        if use_consecutive:
            seed_coords, seed_pids, kinematics = build_seeds_random_consecutive(
                part_df, hit_df, n_seed_hits=n_seed_hits,
            )
        else:
            seed_coords, seed_pids, kinematics = build_seeds_fixed(
                part_df, hit_df, n_seed_hits=n_seed_hits,
            )

        # ---- 4. Build hit feature tensor ----
        hit_feat_cols = ["x", "y", "z", "layer_id", "volume_id", "detector"]
        hit_features = hit_df[hit_feat_cols].to_numpy(dtype=np.float32)
        hit_pids = hit_df["particle_id"].to_numpy(dtype=np.int64)

        # ---- 5. Build target matrix ----
        N_s = len(seed_pids)
        N_h = len(hit_pids)

        if N_s == 0 or N_h == 0:
            return {
                "hits": torch.zeros(0, len(hit_feat_cols), dtype=torch.float32),
                "seeds": torch.zeros(0, n_seed_hits * 3, dtype=torch.float32),
                "targets": torch.zeros(0, 0, dtype=torch.float32),
                "seed_particle_ids": torch.zeros(0, dtype=torch.int64),
                "hit_particle_ids": torch.zeros(N_h, dtype=torch.int64),
                "kinematics": torch.zeros(0, 6, dtype=torch.float32),
                "event_idx": event_idx,
            }

        targets = (seed_pids[:, None] == hit_pids[None, :]).astype(np.float32)

        # ---- 6. Zero out seed hits from targets if predict_seed_hits=False ----
        if not predict_seed_hits:
            # For each seed, find the matching hit indices by xyz proximity.
            seed_xyz = seed_coords  # (N_s, 9) — flatten of 3 hits
            for s in range(N_s):
                for h in range(3):
                    sx = seed_xyz[s, h * 3]
                    sy = seed_xyz[s, h * 3 + 1]
                    sz = seed_xyz[s, h * 3 + 2]
                    dist = np.sqrt(
                        (hit_features[:, 0] - sx) ** 2
                        + (hit_features[:, 1] - sy) ** 2
                        + (hit_features[:, 2] - sz) ** 2
                    )
                    match_idx = np.where(dist < 1e-4)[0]
                    targets[s, match_idx] = 0.0

        return {
            "hits": torch.from_numpy(hit_features),
            "seeds": torch.from_numpy(seed_coords),
            "targets": torch.from_numpy(targets),
            "seed_particle_ids": torch.from_numpy(seed_pids),
            "hit_particle_ids": torch.from_numpy(hit_pids),
            "kinematics": torch.from_numpy(kinematics),
            "event_idx": event_idx,
        }
