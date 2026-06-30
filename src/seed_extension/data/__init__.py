"""Seed extension data modules — CAST dataset and utilities."""

import colliderml_dataloader.shard_index as _si

# Extend feature lists in-place so ColliderMLDataset loads extra columns.
_si.TRACKER_HIT_FEATURES[:] = [
    "x", "y", "z", "particle_id", "event_id",
    "layer_id", "volume_id", "detector",
]
_si.PARTICLE_FEATURES[:] = [
    "particle_id", "px", "py", "pz", "primary", "pdg_id", "event_id",
    "perigee_d0", "perigee_z0", "vertex_primary",
]
