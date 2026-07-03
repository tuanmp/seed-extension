"""Seed-extension data utilities — feature lists, kinematics, seed builders."""

from __future__ import annotations

import numpy as np
import polars as pl
from colliderml.polars import explode_particles, explode_tracker_hits

# Feature lists used for Parquet column selection and feather cache column
# validation.  Kept as module-level constants so consumers import them
# directly instead of relying on global monkey-patching.
TRACKER_HIT_FEATURES = [
    "x", "y", "z", "particle_id", "event_id",
    "layer_id", "volume_id", "detector",
]
PARTICLE_FEATURES = [
    "particle_id", "px", "py", "pz", "primary", "pdg_id", "event_id",
    "perigee_d0", "perigee_z0", "vertex_primary",
]


def compute_kinematics(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray,
    d0: np.ndarray, z0: np.ndarray,
) -> np.ndarray:
    """Compute particle kinematics: [eta, pT, d0, z0, theta, phi].

    Returns array of shape (N, 6).
    """
    pT = np.sqrt(px**2 + py**2)
    eta = np.where(
        pT > 1e-9,
        np.arcsinh(pz / np.clip(pT, 1e-9, None)),
        np.sign(pz) * 10.0,
    )
    theta = np.arctan2(pT, pz)
    phi = np.arctan2(py, px)
    return np.stack([eta, pT, d0, z0, theta, phi], axis=1)


# ------------------------------------------------------------------
# Shared seed-building helpers
# ------------------------------------------------------------------

def compute_pT_eta(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (pT, eta) for filtering — cheaper than full compute_kinematics."""
    pT = np.sqrt(px**2 + py**2)
    eta = np.where(
        pT > 1e-9,
        np.arcsinh(pz / np.clip(pT, 1e-9, None)),
        np.sign(pz) * 10.0,
    )
    return pT, eta


def _extract_kinematics_and_pids(joined: pl.DataFrame):
    """Extract (kinematics, pids) from a grouped polars DataFrame."""
    px = joined["px"].to_numpy().astype(np.float64)
    py = joined["py"].to_numpy().astype(np.float64)
    pz = joined["pz"].to_numpy().astype(np.float64)
    d0 = joined["perigee_d0"].to_numpy().astype(np.float64)
    z0 = joined["perigee_z0"].to_numpy().astype(np.float64)
    pids = joined["particle_id"].to_numpy().astype(np.int64)
    kinematics = compute_kinematics(px, py, pz, d0, z0).astype(np.float32)
    return kinematics, pids


def build_seeds_fixed(
    part_df: "pd.DataFrame", hit_df: "pd.DataFrame", n_seed_hits: int = 3,  # noqa: F821
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build seeds using the innermost (lowest layer_id) hits per particle.

    Returns ``(seed_coords, seed_pids, kinematics)``.
    """
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

    kinematics, pids = _extract_kinematics_and_pids(joined)
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

        # Find consecutive layer groups of length n_seed_hits.
        starts = [
            s for s in range(len(layers) - n_seed_hits + 1)
            if np.all(np.diff(layers[s:s + n_seed_hits]) == 1)
        ]
        start = int(rng.choice(starts)) if starts else 0
        for k in range(n_seed_hits):
            coords[i, k * 3] = x[start + k]
            coords[i, k * 3 + 1] = y[start + k]
            coords[i, k * 3 + 2] = z[start + k]

    kinematics, pids = _extract_kinematics_and_pids(joined)
    return coords, pids, kinematics
