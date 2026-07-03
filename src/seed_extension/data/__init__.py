"""Seed extension data modules — CAST dataset, seed utilities, and feather cache."""

from seed_extension.data.seed_utils import (
    PARTICLE_FEATURES,
    TRACKER_HIT_FEATURES,
    build_seeds_fixed,
    build_seeds_random_consecutive,
    compute_kinematics,
    compute_pT_eta,
)
from seed_extension.data.dataset import SeedExtensionDataset
from seed_extension.data.cache import CachedColliderMLDataset, CachedColliderMLDataModule

__all__ = [
    "PARTICLE_FEATURES",
    "TRACKER_HIT_FEATURES",
    "SeedExtensionDataset",
    "CachedColliderMLDataset",
    "CachedColliderMLDataModule",
    "build_seeds_fixed",
    "build_seeds_random_consecutive",
    "compute_kinematics",
    "compute_pT_eta",
]
