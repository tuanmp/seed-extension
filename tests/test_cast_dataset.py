"""Smoke test for SeedExtensionDataset."""

import os

import torch
from colliderml_dataloader import ColliderMLDataModule

from seed_extension.data.dataset import SeedExtensionDataset

DATA_DIR = os.environ.get("COLLIDERML_DATA_DIR", "/pscratch/sd/p/pmtuan/.cache/colliderml")


def test_dataset_returns_correct_keys():
    dm = ColliderMLDataModule(
        data_dir=DATA_DIR,
        process="ttbar",
        pileup="pu200",
        max_train_events=2,
        max_val_events=0,
        max_test_events=0,
        batch_size=1,
        num_workers=0,
        dataset_cls=SeedExtensionDataset,
        min_track_hits=5,
        seed_strategy="fixed_innermost",
        target_vertices=200,
        primary_only=False,
        predict_seed_hits=True,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))

    # batch_size=1 + default_collate gives a dict with batched tensors.
    assert isinstance(batch, dict)

    expected_keys = {
        "hits", "seeds", "targets", "seed_particle_ids",
        "hit_particle_ids", "kinematics", "event_idx",
    }
    assert set(batch.keys()) == expected_keys

    # Remove batch dimension (size 1) for shape assertions.
    hits = batch["hits"][0]       # (N_h, F_hit)
    seeds = batch["seeds"][0]     # (N_s, 9)
    targets = batch["targets"][0]  # (N_s, N_h)

    assert hits.ndim == 2
    assert seeds.ndim == 2
    assert targets.ndim == 2

    N_s = seeds.shape[0]
    N_h = hits.shape[0]
    assert N_s == targets.shape[0]
    assert N_h == targets.shape[1]

    assert torch.all((targets == 0) | (targets == 1))

    if N_s > 0:
        positives_per_seed = targets.sum(dim=1)
        assert torch.all(positives_per_seed >= 5), f"Some seeds have <5 positives: {positives_per_seed}"
    else:
        print("WARNING: No seeds found in event — testing empty tensors")
