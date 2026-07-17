"""Smoke test for SeedExtensionDataset."""

import os

import torch
from colliderml_dataloader import ColliderMLDataModule

from seed_extension.data.dataset import SeedExtensionDataset

DATA_DIR = os.environ.get("COLLIDERML_DATA_DIR", "/pscratch/sd/p/pmtuan/.cache/colliderml")


config_string = """
data_dir: /pscratch/sd/p/pmtuan/.cache/colliderml
cache_dir: /pscratch/sd/p/pmtuan/.cache/colliderml-feather
process: ttbar
pileup: pu200
max_train_events: 96000
max_val_events: 2000
max_test_events: 2000
batch_size: 1
num_workers: 8
prefetch_factor: 4
persistent_workers: true
n_seed_hits: 3
min_track_hits: 5
min_pT: 0.9
max_abs_eta: 4.0
seed_strategy: random_consecutive
target_vertices: 50
primary_only: false
predict_seed_hits: false
"""

import yaml
config = yaml.safe_load(config_string)

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
        target_vertices=20,
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
        "hit_particle_ids", "kinematics", "event_idx", "hit_detector_info"
    }
    assert set(batch.keys()) == expected_keys

    # Remove batch dimension (size 1) for shape assertions.
    hits = batch["hits"][0]       # (N_h, F_hit)
    seeds = batch["seeds"][0]     # (N_s, 3) should contain the hit indices of the seed hits
    targets = batch["targets"][0]  # (N_s, N_h)

    assert seeds.shape[1] == 3

    hit_pids = batch["hit_particle_ids"][0]  # (N_h,)

    # all seeds must share the same particle_id as the first seed hit
    assert torch.all(hit_pids[seeds] == hit_pids[seeds[:, :1]])
    assert hit_pids.ndim == 1

    assert hit_pids.shape[0] == hits.shape[0]
    assert hit_pids.shape[0] == targets.shape[1]

    target_pids = targets.clone().detach().to(torch.int64) @ hit_pids.unsqueeze(1).to(torch.int64)  # (N_s, 1)
    num_hits = targets.sum(dim=1, keepdim=True).to(torch.int64)  # (N_s, 1)
    mean_pids = target_pids / num_hits.clamp(min=1)  # (N_s, 1) 

    first_seed_hit_id = targets.argmax(dim=1)  # (N_s,)
    first_seed_hit_pid = hit_pids[first_seed_hit_id]  # (N_s,)
    assert torch.all(mean_pids.squeeze() == first_seed_hit_pid.to(torch.float64))

    # all seed pid must be the same as truth pid
    assert torch.all(hit_pids[seeds].to(torch.long) == mean_pids.to(torch.long))

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
