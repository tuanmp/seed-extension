# CAST Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the CAST (Cross-Attention Seed Transformer) model for seed-extension hit-to-track assignment on ColliderML data.

**Architecture:** A LightningModule that encodes hits (xyz + detector geometry via Fourier features + MLP) and seeds (triplet coordinates → MLP), applies cross-attention (seeds as queries, hits as keys/values), and trains with multi-positive InfoNCE loss. Pluggable hit context encoder for future ablation.

**Tech Stack:** PyTorch 2.2+, Lightning 2.2+, colliderml-dataloader, colliderml (pileup)

---

### Task 1: Directory structure and feature monkey-patch

**Files:**
- Create: `src/ml_cookbook/data/seed_extension/__init__.py`
- Create: `src/ml_cookbook/models/cast/__init__.py`
- Modify: `src/ml_cookbook/data/seed_extension/__init__.py`

**Goal:** Create the package directories and extend the colliderml-dataloader feature lists to include detector geometry and particle kinematics columns.

- [ ] **Step 1: Create directories and module files**

Run:
```bash
mkdir -p src/ml_cookbook/data/seed_extension
mkdir -p src/ml_cookbook/models/cast
touch src/ml_cookbook/data/seed_extension/__init__.py
touch src/ml_cookbook/models/cast/__init__.py
```

- [ ] **Step 2: Write the feature extension in `src/ml_cookbook/data/seed_extension/__init__.py`**

Write:
```python
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
```

- [ ] **Step 3: Verify monkey-patch takes effect**

Run:
```bash
uv run python -c "
import src.ml_cookbook.data.seed_extension
import colliderml_dataloader.shard_index as si
print('TRACKER_HIT_FEATURES:', si.TRACKER_HIT_FEATURES)
print('PARTICLE_FEATURES:', si.PARTICLE_FEATURES)
"
```

Expected output: both lists have the extended columns (layer_id, volume_id, detector for hits; perigee_d0, perigee_z0, vertex_primary for particles).

- [ ] **Step 4: Commit**

```bash
git add src/ml_cookbook/data/seed_extension/__init__.py src/ml_cookbook/models/cast/__init__.py
git commit -m "feat: add seed_extension and cast package scaffolding with feature monkey-patch"
```

---

### Task 2: SeedExtensionDataset

**Files:**
- Create: `src/ml_cookbook/data/seed_extension/dataset.py`
- Test: `tests/test_cast_dataset.py`

**Goal:** Subclass ColliderMLDataset to construct seeds, targets, and particle kinematics from raw ColliderML data.

- [ ] **Step 1: Write the dataset class skeleton with a smoke test**

Create `tests/test_cast_dataset.py`:
```python
"""Smoke tests for SeedExtensionDataset."""
import pytest
import torch
from colliderml_dataloader import ColliderMLDataModule
from ml_cookbook.data.seed_extension.dataset import SeedExtensionDataset


@pytest.mark.slow
def test_dataset_returns_correct_keys():
    """One event load — verify output dict keys and tensor shapes."""
    import os
    datadir = os.environ.get("COLLIDERML_DATA_DIR", "/pscratch/sd/p/pmtuan/.cache/colliderml")
    dm = ColliderMLDataModule(
        data_dir=datadir,
        process="ttbar",
        pileup="pu200",
        max_train_events=2,
        max_val_events=2,
        max_test_events=0,
        batch_size=1,
        num_workers=0,
        dataset_cls=SeedExtensionDataset,
        min_track_hits=5,
        seed_strategy="fixed_innermost",
        target_vertices=200,
        primary_only=False,
        predict_seed_hits=False,
    )
    dm.setup("fit")
    loader = dm.train_dataloader()
    batch = next(iter(loader))
    # batch is a list of len 1 (batch_size=1 with default_collate for dict)
    sample = batch[0] if isinstance(batch, list) else batch

    assert "hits" in sample
    assert "seeds" in sample
    assert "targets" in sample
    assert "seed_particle_ids" in sample
    assert "hit_particle_ids" in sample
    assert "kinematics" in sample

    hits = sample["hits"]
    seeds = sample["seeds"]
    targets = sample["targets"]

    assert hits.ndim == 2  # (N_h, F_hit)
    assert seeds.ndim == 2  # (N_s, 9)
    assert targets.ndim == 2  # (N_s, N_h)

    assert seeds.shape[0] == targets.shape[0]
    assert hits.shape[0] == targets.shape[1]
    # targets values should be 0 or 1
    assert torch.all((targets == 0) | (targets == 1))
    # Each seed should have at least min_track_hits positives
    n_pos = targets.sum(dim=1)
    assert torch.all(n_pos >= 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_cast_dataset.py::test_dataset_returns_correct_keys -v
```

Expected: FAIL — `SeedExtensionDataset` not defined.

- [ ] **Step 3: Implement SeedExtensionDataset**

Create `src/ml_cookbook/data/seed_extension/dataset.py`:
```python
"""SeedExtensionDataset — constructs seeds and targets from ColliderML events."""

import numpy as np
import torch
from colliderml_dataloader.dataset import ColliderMLDataset


def compute_kinematics(px, py, pz, d0, z0):
    """Compute eta, pT, theta, phi from particle momentum."""
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(pt**2 + pz**2)
    theta = np.arctan2(pt, pz)
    # eta = -ln(tan(theta/2)), clamp to avoid inf
    theta_clipped = np.clip(theta, 1e-7, np.pi - 1e-7)
    eta = -np.log(np.tan(theta_clipped / 2.0))
    phi = np.arctan2(py, px)
    return np.stack([eta, pt, d0, z0, theta, phi], axis=1)  # (N, 6)


def build_seeds_fixed(part_df, hit_df, n_seed_hits=3):
    """Fixed-innermost strategy: take 3 hits from lowest layer_id per particle."""
    particle_ids = part_df["particle_id"].values
    all_seed_coords = []
    all_seed_pids = []
    all_kinematics = []

    for i, pid in enumerate(particle_ids):
        # Hits for this particle, sorted by layer
        mask = hit_df["particle_id"].values == pid
        p_hits = hit_df[mask].sort_values("layer_id")
        if len(p_hits) < n_seed_hits:
            continue
        # Take 3 innermost hits (lowest layer_id)
        seed_hits = p_hits.iloc[:n_seed_hits]
        coords = seed_hits[["x", "y", "z"]].values.flatten()  # (9,)
        all_seed_coords.append(coords)
        all_seed_pids.append(pid)
        # Kinematics from the particle row
        kin = compute_kinematics(
            part_df["px"].values[i],
            part_df["py"].values[i],
            part_df["pz"].values[i],
            part_df["perigee_d0"].values[i],
            part_df["perigee_z0"].values[i],
        )
        all_kinematics.append(kin[0])

    if len(all_seed_coords) == 0:
        return None, None, None

    return (
        np.stack(all_seed_coords).astype(np.float32),
        np.array(all_seed_pids, dtype=np.int64),
        np.stack(all_kinematics).astype(np.float32),
    )


def build_seeds_random_consecutive(part_df, hit_df, n_seed_hits=3, rng=None):
    """Random consecutive strategy: pick 3 consecutive-layer hits per particle."""
    if rng is None:
        rng = np.random.default_rng()
    particle_ids = part_df["particle_id"].values
    all_seed_coords = []
    all_seed_pids = []
    all_kinematics = []

    for i, pid in enumerate(particle_ids):
        mask = hit_df["particle_id"].values == pid
        p_hits = hit_df[mask].sort_values("layer_id")
        if len(p_hits) < n_seed_hits:
            continue
        # Find groups of consecutive layers
        layers = p_hits["layer_id"].values
        # Pick a random start index where there are n_seed_hits consecutive
        valid_starts = []
        for s in range(len(layers) - n_seed_hits + 1):
            if np.all(np.diff(layers[s:s + n_seed_hits]) == 1):
                valid_starts.append(s)
        if not valid_starts:
            # Fallback: just take first n_seed_hits
            start = 0
        else:
            start = rng.choice(valid_starts)
        seed_hits = p_hits.iloc[start:start + n_seed_hits]
        coords = seed_hits[["x", "y", "z"]].values.flatten()
        all_seed_coords.append(coords)
        all_seed_pids.append(pid)
        kin = compute_kinematics(
            part_df["px"].values[i],
            part_df["py"].values[i],
            part_df["pz"].values[i],
            part_df["perigee_d0"].values[i],
            part_df["perigee_z0"].values[i],
        )
        all_kinematics.append(kin[0])

    if len(all_seed_coords) == 0:
        return None, None, None

    return (
        np.stack(all_seed_coords).astype(np.float32),
        np.array(all_seed_pids, dtype=np.int64),
        np.stack(all_kinematics).astype(np.float32),
    )


class SeedExtensionDataset(ColliderMLDataset):
    """Dataset that constructs seeds and hit-assignment targets.

    Extra kwargs:
        min_track_hits: int — minimum non-seed hits per particle (default 5)
        seed_strategy: str — "fixed_innermost" or "random_consecutive"
        target_vertices: int — pileup vertices to keep (default 200 = full)
        primary_only: bool — keep only primary particles
        predict_seed_hits: bool — include seed hits in assignment targets
    """

    def _process_event(self, hits_raw, parts_raw, event_idx: int):
        stage = self.stage
        kwargs = self._kwargs
        min_track_hits = kwargs.get("min_track_hits", 5)
        seed_strategy = kwargs.get("seed_strategy", "random_consecutive")
        target_vertices = kwargs.get("target_vertices", 200)
        primary_only = kwargs.get("primary_only", False)
        predict_seed_hits = kwargs.get("predict_seed_hits", False)

        # --- Pileup subsampling ---
        if target_vertices < 200:
            kept_mask = parts_raw["vertex_primary"].values <= target_vertices
            parts_raw = parts_raw[kept_mask].copy()
            kept_pids = set(parts_raw["particle_id"].values.tolist())
            hits_raw = hits_raw[hits_raw["particle_id"].isin(kept_pids)].copy()
        # Ensure particle_id is int64 for matching
        hits_raw = hits_raw.copy()
        hits_raw["particle_id"] = hits_raw["particle_id"].astype(np.int64)

        # --- Particle filtering ---
        if primary_only:
            parts_raw = parts_raw[parts_raw["primary"] == True].copy()

        # Filter particles with enough hits
        hit_counts = hits_raw.groupby("particle_id").size()
        valid_pids = hit_counts[hit_counts >= min_track_hits].index
        parts_raw = parts_raw[parts_raw["particle_id"].isin(valid_pids)].copy()

        # --- Seed construction ---
        if stage in ("fit", "train") and seed_strategy == "random_consecutive":
            rng = np.random.default_rng()
            seed_coords, seed_pids, kinematics = build_seeds_random_consecutive(
                parts_raw, hits_raw, n_seed_hits=3, rng=rng
            )
        else:
            seed_coords, seed_pids, kinematics = build_seeds_fixed(
                parts_raw, hits_raw, n_seed_hits=3
            )

        if seed_coords is None:
            # Edge case: no valid seeds. Return empty tensors.
            return {
                "hits": torch.zeros(0, 3),
                "seeds": torch.zeros(0, 9),
                "targets": torch.zeros(0, 0, dtype=torch.long),
                "seed_particle_ids": torch.zeros(0, dtype=torch.long),
                "hit_particle_ids": torch.zeros(0, dtype=torch.long),
                "kinematics": torch.zeros(0, 6),
                "event_idx": event_idx,
            }

        # --- Hit features ---
        hit_xy = hits_raw[["x", "y", "z"]].values.astype(np.float32)
        hit_feats = [hit_xy]
        if "layer_id" in hits_raw.columns:
            hit_feats.append(hits_raw["layer_id"].values.astype(np.float32)[:, None])
        if "volume_id" in hits_raw.columns:
            hit_feats.append(hits_raw["volume_id"].values.astype(np.float32)[:, None])
        if "detector" in hits_raw.columns:
            hit_feats.append(hits_raw["detector"].values.astype(np.float32)[:, None])
        hit_features = np.concatenate(hit_feats, axis=1)  # (N_h, F_hit)
        hit_pids = hits_raw["particle_id"].values.astype(np.int64)

        # --- Target matrix ---
        # targets[i,j] = 1 if seed_pids[i] == hit_pids[j]
        N_s = len(seed_pids)
        N_h = len(hit_pids)
        targets = (seed_pids[:, None] == hit_pids[None, :]).astype(np.float32)

        # Optionally mask out seed hits from targets
        if not predict_seed_hits:
            seed_hit_mask = np.zeros(N_h, dtype=bool)
            # Mark hits that are used as seed coords (approximate by matching xyz)
            for sc in seed_coords:
                # each seed is 9 floats (3 hits × 3 coords)
                for k in range(3):
                    sx, sy, sz = sc[k*3], sc[k*3+1], sc[k*3+2]
                    dist = np.sqrt(
                        (hit_xy[:, 0] - sx)**2 +
                        (hit_xy[:, 1] - sy)**2 +
                        (hit_xy[:, 2] - sz)**2
                    )
                    closest = np.argmin(dist)
                    if dist[closest] < 1e-4:  # exact match
                        seed_hit_mask[closest] = True
            targets[:, seed_hit_mask] = 0.0

        return {
            "hits": torch.from_numpy(hit_features),
            "seeds": torch.from_numpy(seed_coords),
            "targets": torch.from_numpy(targets),
            "seed_particle_ids": torch.from_numpy(seed_pids),
            "hit_particle_ids": torch.from_numpy(hit_pids),
            "kinematics": torch.from_numpy(kinematics),
            "event_idx": event_idx,
        }
```

- [ ] **Step 4: Run the test**

Run:
```bash
uv run pytest tests/test_cast_dataset.py::test_dataset_returns_correct_keys -v -s
```

Expected: PASS (or SKIP if data not available). The test may take ~5-10 seconds per event on first load.

- [ ] **Step 5: Commit**

```bash
git add src/ml_cookbook/data/seed_extension/dataset.py tests/test_cast_dataset.py
git commit -m "feat: implement SeedExtensionDataset with seed construction and target generation"
```

---

### Task 3: Config file

**Files:**
- Create: `configs/cast_default.yaml`

**Goal:** Create the CAST experiment configuration.

- [ ] **Step 1: Write config**

Create `configs/cast_default.yaml`:
```yaml
experiment_name: cast_baseline
seed: 42

data:
  process: ttbar
  pileup: pu200
  max_train_events: 50000
  max_val_events: 5000
  max_test_events: 5000
  num_workers: 8
  target_vertices: 200
  seed_strategy: random_consecutive
  seed_n_samples_val: 1
  min_track_hits: 5
  primary_only: false
  predict_seed_hits: false
  use_detector_features: true
  normalize_coords: true

model:
  d_model: 256
  n_cross_attn_layers: 3
  n_heads: 8
  ff_dim: 1024
  dropout: 0.1
  temperature: 0.1
  fourier_L: 8
  use_cylindrical_pe: true
  use_seed_self_attn: true
  hit_encoder: identity
  learning_rate: 1e-4
  weight_decay: 1e-4
  lr_scheduler: cosine
  warmup_steps: 1000

trainer:
  max_epochs: 50
  accelerator: auto
  devices: 1
  precision: bf16-mixed
  deterministic: true
  accumulate_grad_batches: 4
  gradient_clip_val: 1.0
  log_every_n_steps: 1
```

- [ ] **Step 2: Commit**

```bash
git add configs/cast_default.yaml
git commit -m "feat: add CAST experiment config"
```

---

### Task 4: Embedders (FourierEncode, HitEmbedder, SeedEmbedder)

**Files:**
- Create: `src/ml_cookbook/models/cast/embedders.py`
- Test: `tests/test_cast_shapes.py` (first test)

**Goal:** Implement Fourier feature encoding, hit embedding, and seed embedding modules.

- [ ] **Step 1: Write the first shape test**

Create `tests/test_cast_shapes.py`:
```python
"""Shape tests for CAST model components."""
import pytest
import torch
from ml_cookbook.models.cast.embedders import (
    FourierEncode,
    HitEmbedder,
    SeedEmbedder,
)


class TestFourierEncode:
    def test_output_shape(self):
        L = 8
        fe = FourierEncode(L=L, input_dim=3)
        x = torch.randn(10, 3)
        out = fe(x)
        assert out.shape == (10, 3 * 2 * L)  # 3 * 2 * 8 = 48

    def test_normalization(self):
        fe = FourierEncode(L=4, input_dim=3)
        x = torch.randn(50, 3) * 100  # large values
        out = fe(x)
        # Should be finite
        assert torch.isfinite(out).all()
        # Values in [-1, 1] since sin/cos
        assert out.min() >= -1.0
        assert out.max() <= 1.0


class TestHitEmbedder:
    def test_output_shape(self):
        emb = HitEmbedder(
            d_model=256,
            fourier_L=8,
            use_detector_features=True,
            use_cylindrical_pe=False,
        )
        x = torch.randn(4, 100, 6)  # (B, N_h, 6): xyz + layer + volume + detector
        out = emb(x)
        assert out.shape == (4, 100, 256)

    def test_no_detector_features(self):
        emb = HitEmbedder(
            d_model=128,
            fourier_L=4,
            use_detector_features=False,
            use_cylindrical_pe=False,
        )
        x = torch.randn(2, 50, 3)  # xyz only
        out = emb(x)
        assert out.shape == (2, 50, 128)


class TestSeedEmbedder:
    def test_output_shape(self):
        emb = SeedEmbedder(d_model=256)
        x = torch.randn(4, 200, 9)  # (B, N_s, 9): 3 hits × xyz
        out = emb(x)
        assert out.shape == (4, 200, 256)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
uv run pytest tests/test_cast_shapes.py -v
```

Expected: FAIL — modules not defined.

- [ ] **Step 3: Implement embedders**

Create `src/ml_cookbook/models/cast/embedders.py`:
```python
"""Embedding modules for CAST model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncode(nn.Module):
    """NeRF-style multi-scale Fourier feature encoding.

    For each input coordinate x in [0,1], encodes as:
        [sin(2⁰πx), cos(2⁰πx), ..., sin(2^{L-1}πx), cos(2^{L-1}πx)]
    """

    def __init__(self, L: int = 8, input_dim: int = 3):
        super().__init__()
        self.L = L
        self.input_dim = input_dim
        self.output_dim = input_dim * 2 * L

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim) in [0, 1] range
        enc = []
        for i in range(self.L):
            freq = (2.0 ** i) * math.pi * x
            enc.append(torch.sin(freq))
            enc.append(torch.cos(freq))
        return torch.cat(enc, dim=-1)


class HitEmbedder(nn.Module):
    """Encodes hit features (xyz + optional detector geometry) into embeddings."""

    def __init__(
        self,
        d_model: int = 256,
        fourier_L: int = 8,
        use_detector_features: bool = True,
        use_cylindrical_pe: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_detector_features = use_detector_features
        self.use_cylindrical_pe = use_cylindrical_pe

        # Fourier encode for xyz coordinates
        self.fourier_xyz = FourierEncode(L=fourier_L, input_dim=3)
        xyz_enc_dim = self.fourier_xyz.output_dim  # 3 * 2 * L

        in_dim = xyz_enc_dim

        if use_cylindrical_pe:
            self.fourier_cyl = FourierEncode(L=fourier_L, input_dim=3)
            in_dim += self.fourier_cyl.output_dim

        if use_detector_features:
            # layer_id, volume_id, detector as scalar features
            # Small learned embeddings for discrete detector/volume values
            self.detector_emb = nn.Embedding(20, 16)
            self.volume_emb = nn.Embedding(30, 16)
            self.layer_proj = nn.Linear(1, 16)
            in_dim += 48

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _to_cylindrical(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert xyz to (r, phi, z) in [0, 1] range."""
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x) / (2 * math.pi) + 0.5  # normalize to [0,1]
        # r and z already approximately normalized
        return torch.stack([r, phi, z], dim=-1)

    def forward(self, hits: torch.Tensor) -> torch.Tensor:
        # hits: (..., N_h, F_hit)
        # First 3 features are always x, y, z
        xyz = hits[..., :3]
        xyz_norm = self._normalize_coords(xyz)
        xyz_enc = self.fourier_xyz(xyz_norm)

        features = [xyz_enc]

        if self.use_cylindrical_pe:
            cyl = self._to_cylindrical(xyz_norm)
            cyl_enc = self.fourier_cyl(cyl)
            features.append(cyl_enc)

        if self.use_detector_features and hits.shape[-1] >= 6:
            layer_id = hits[..., 3:4].long().clamp(0, 29)
            volume_id = hits[..., 4:5].long().clamp(0, 29)
            detector = hits[..., 5:6].long().clamp(0, 19)

            det_emb = self.detector_emb(detector.squeeze(-1))
            vol_emb = self.volume_emb(volume_id.squeeze(-1))
            lay_emb = self.layer_proj(layer_id.float())

            features.append(torch.cat([det_emb, vol_emb, lay_emb], dim=-1))

        combined = torch.cat(features, dim=-1)
        return self.mlp(combined)

    def _normalize_coords(self, xyz: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [0,1] range using approximate detector bounds."""
        # Tracker extends roughly ±1100 mm in x/y, ±3000 mm in z
        xyz_norm = xyz.clone()
        xyz_norm[..., 0] = xyz[..., 0] / 1100.0 * 0.5 + 0.5   # x
        xyz_norm[..., 1] = xyz[..., 1] / 1100.0 * 0.5 + 0.5   # y
        xyz_norm[..., 2] = xyz[..., 2] / 3000.0 * 0.5 + 0.5   # z
        return xyz_norm.clamp(0.0, 1.0)


class SeedEmbedder(nn.Module):
    """Encodes seed triplets (3 hits × 3 coords = 9 dims) into embeddings."""

    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(9, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, seeds: torch.Tensor) -> torch.Tensor:
        # seeds: (..., N_s, 9)
        return self.mlp(seeds)
```

- [ ] **Step 4: Run shape tests**

Run:
```bash
uv run pytest tests/test_cast_shapes.py -v
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ml_cookbook/models/cast/embedders.py tests/test_cast_shapes.py
git commit -m "feat: implement FourierEncode, HitEmbedder, SeedEmbedder"
```

---

### Task 5: Attention modules

**Files:**
- Create: `src/ml_cookbook/models/cast/attention.py`
- Modify: `tests/test_cast_shapes.py` (add attention tests)

**Goal:** Implement cross-attention decoder and optional seed self-attention.

- [ ] **Step 1: Add attention shape tests**

Append to `tests/test_cast_shapes.py`:
```python
from ml_cookbook.models.cast.attention import (
    CrossAttentionBlock,
    CrossAttentionDecoder,
    SeedSelfAttention,
)


class TestCrossAttentionBlock:
    def test_output_shape(self):
        block = CrossAttentionBlock(d_model=256, n_heads=8, ff_dim=1024, dropout=0.1)
        queries = torch.randn(4, 100, 256)   # (B, N_s, d)
        kv = torch.randn(4, 1000, 256)        # (B, N_h, d)
        out = block(queries, kv)
        assert out.shape == queries.shape

    def test_attention_scores_shape(self):
        block = CrossAttentionBlock(d_model=256, n_heads=8)
        queries = torch.randn(2, 50, 256)
        kv = torch.randn(2, 500, 256)
        out, attn = block(queries, kv, return_attention=True)
        assert attn.shape == (2, 50, 500)  # average over heads


class TestCrossAttentionDecoder:
    def test_output_shape(self):
        decoder = CrossAttentionDecoder(
            d_model=256, n_heads=8, ff_dim=1024,
            n_layers=3, dropout=0.1,
        )
        queries = torch.randn(4, 100, 256)
        kv = torch.randn(4, 1000, 256)
        out = decoder(queries, kv)
        assert out.shape == queries.shape

    def test_return_attention(self):
        decoder = CrossAttentionDecoder(
            d_model=128, n_heads=4, ff_dim=512,
            n_layers=2, dropout=0.0,
        )
        queries = torch.randn(2, 10, 128)
        kv = torch.randn(2, 100, 128)
        out, attn = decoder(queries, kv, return_attention=True)
        assert out.shape == queries.shape
        assert attn.shape == (2, 10, 100)  # final layer attention


class TestSeedSelfAttention:
    def test_output_shape(self):
        sa = SeedSelfAttention(d_model=256, n_heads=8, dropout=0.1)
        x = torch.randn(4, 100, 256)
        out = sa(x)
        assert out.shape == x.shape
```

- [ ] **Step 2: Run tests to verify failure**

Run:
```bash
uv run pytest tests/test_cast_shapes.py::TestCrossAttentionBlock tests/test_cast_shapes.py::TestCrossAttentionDecoder tests/test_cast_shapes.py::TestSeedSelfAttention -v
```

Expected: FAIL.

- [ ] **Step 3: Implement attention modules**

Create `src/ml_cookbook/models/cast/attention.py`:
```python
"""Cross-attention and self-attention modules for CAST."""

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: MHA + residual + FFN + residual."""

    def __init__(
        self, d_model: int = 256, n_heads: int = 8,
        ff_dim: int = 1024, dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries, kv, return_attention=False):
        # queries: (B, N_q, d), kv: (B, N_kv, d)
        attn_out, attn_weights = self.attn(
            query=queries, key=kv, value=kv,
            need_weights=return_attention,
            average_attn_weights=True,
        )
        queries = self.norm1(queries + attn_out)
        queries = self.norm2(queries + self.ffn(queries))

        if return_attention:
            return queries, attn_weights
        return queries


class CrossAttentionDecoder(nn.Module):
    """Stack of cross-attention blocks. Seeds attend to all hits."""

    def __init__(
        self, d_model: int = 256, n_heads: int = 8,
        ff_dim: int = 1024, n_layers: int = 3, dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, seed_emb, hit_emb, return_attention=False):
        # seed_emb: (B, N_s, d), hit_emb: (B, N_h, d)
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            if return_attention and is_last:
                seed_emb, attn = layer(seed_emb, hit_emb, return_attention=True)
                return seed_emb, attn
            else:
                seed_emb = layer(seed_emb, hit_emb)
        return seed_emb


class SeedSelfAttention(nn.Module):
    """Self-attention among seeds for track disambiguation."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seed_emb):
        # seed_emb: (B, N_s, d)
        attn_out, _ = self.attn(seed_emb, seed_emb, seed_emb)
        return self.norm(seed_emb + attn_out)
```

- [ ] **Step 4: Run tests**

Run:
```bash
uv run pytest tests/test_cast_shapes.py -v
```

Expected: 9 PASS (4 embedders + 5 attention).

- [ ] **Step 5: Commit**

```bash
git add src/ml_cookbook/models/cast/attention.py tests/test_cast_shapes.py
git commit -m "feat: implement CrossAttentionDecoder and SeedSelfAttention"
```

---

### Task 6: Identity hit encoder

**Files:**
- Create: `src/ml_cookbook/models/cast/encoders.py`
- Modify: `tests/test_cast_shapes.py` (add encoder test)

**Goal:** Implement the pluggable hit context encoder interface with IdentityEncoder as default.

- [ ] **Step 1: Add encoder test**

Append to `tests/test_cast_shapes.py`:
```python
from ml_cookbook.models.cast.encoders import IdentityEncoder


class TestIdentityEncoder:
    def test_passthrough(self):
        enc = IdentityEncoder()
        x = torch.randn(4, 1000, 256)
        out = enc(x)
        assert out.shape == x.shape
        assert torch.equal(out, x)
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
uv run pytest tests/test_cast_shapes.py::TestIdentityEncoder -v
```

Expected: FAIL.

- [ ] **Step 3: Implement encoders**

Create `src/ml_cookbook/models/cast/encoders.py`:
```python
"""Pluggable hit context encoders."""

import torch.nn as nn


class IdentityEncoder(nn.Module):
    """Pass-through encoder — no contextual processing of hits."""

    def forward(self, hit_emb: "torch.Tensor") -> "torch.Tensor":
        return hit_emb


class BinnedSelfAttentionEncoder(nn.Module):
    """Self-attention within spatial bins. Placeholder for future ablation."""

    def __init__(self):
        super().__init__()
        raise NotImplementedError("BinnedSelfAttentionEncoder not yet implemented")
```

- [ ] **Step 4: Run test**

Run:
```bash
uv run pytest tests/test_cast_shapes.py::TestIdentityEncoder -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ml_cookbook/models/cast/encoders.py tests/test_cast_shapes.py
git commit -m "feat: add pluggable hit context encoders (IdentityEncoder)"
```

---

### Task 7: Loss function

**Files:**
- Create: `src/ml_cookbook/models/cast/loss.py`
- Test: `tests/test_cast_loss.py`

**Goal:** Implement multi-positive InfoNCE loss.

- [ ] **Step 1: Write loss tests**

Create `tests/test_cast_loss.py`:
```python
"""Tests for CAST loss functions."""
import pytest
import torch
from ml_cookbook.models.cast.loss import info_nce_loss


class TestInfoNCELoss:
    def test_perfect_prediction_zero_loss(self):
        """Loss should be ~0 when all positives get max score."""
        scores = torch.zeros(4, 3, 5)  # (B, N_s, N_h)
        # seed 0 matches hit 0, seed 1 matches hit 1, seed 2 matches hit 2
        targets = torch.zeros(4, 3, 5)
        for b in range(4):
            for i in range(3):
                scores[b, i, i] = 10.0       # high score for true match
                targets[b, i, i] = 1.0       # positive pair

        loss = info_nce_loss(scores, targets, temperature=1.0)
        # Loss should be close to 0 since positives dominate
        assert loss.item() < 0.1

    def test_random_prediction_high_loss(self):
        """Random scores should give high loss."""
        scores = torch.randn(4, 10, 100)  # random logits
        targets = torch.zeros(4, 10, 100)
        for b in range(4):
            for i in range(10):
                # each seed matches exactly 5 hits
                matches = torch.randperm(100)[:5]
                targets[b, i, matches] = 1.0
                # boost those scores slightly but keep randomness
                scores[b, i, matches] += 2.0

        loss = info_nce_loss(scores, targets, temperature=0.1)
        # Random-ish should be reasonably high
        assert loss.item() > 0.5

    def test_no_positives_returns_zero(self):
        """Edge case: seed with no positives should not contribute to loss."""
        scores = torch.randn(1, 1, 10)
        targets = torch.zeros(1, 1, 10)  # no positives
        loss = info_nce_loss(scores, targets, temperature=1.0)
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_multiple_positives(self):
        """Each seed can have multiple positive hits."""
        scores = torch.randn(2, 3, 10)
        targets = torch.zeros(2, 3, 10)
        # Give high scores to all positives
        for b in range(2):
            for i in range(3):
                pos_indices = [i, i + 3, i + 6]  # 3 positives per seed
                targets[b, i, pos_indices] = 1.0
                scores[b, i, pos_indices] = 5.0 - i  # varying confidence

        loss = info_nce_loss(scores, targets, temperature=1.0)
        assert loss.item() >= 0.0

    def test_temperature_effect(self):
        """Lower temperature should amplify differences."""
        scores = torch.tensor([[[0.0, 1.0, 0.0]]])  # (1, 1, 3)
        targets = torch.tensor([[[0.0, 1.0, 0.0]]])

        loss_high_t = info_nce_loss(scores, targets, temperature=10.0)
        loss_low_t = info_nce_loss(scores, targets, temperature=0.1)

        # Low temperature = more peaked softmax = lower loss
        # (because correct hit gets higher relative prob)
        assert loss_low_t < loss_high_t

    def test_batch_independence(self):
        """Batch items should be computed independently."""
        scores1 = torch.randn(2, 5, 20)
        targets1 = torch.zeros(2, 5, 20)
        for b in range(2):
            for i in range(5):
                targets1[b, i, i] = 1.0

        loss_batch = info_nce_loss(scores1, targets1)

        loss0 = info_nce_loss(scores1[0:1], targets1[0:1])
        loss1 = info_nce_loss(scores1[1:2], targets1[1:2])
        loss_manual = (loss0 + loss1) / 2.0

        assert abs(loss_batch.item() - loss_manual.item()) < 1e-4
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
uv run pytest tests/test_cast_loss.py -v
```

Expected: FAIL — `info_nce_loss` not defined.

- [ ] **Step 3: Implement loss function**

Create `src/ml_cookbook/models/cast/loss.py`:
```python
"""Loss functions for CAST model."""

import torch
import torch.nn.functional as F


def info_nce_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Multi-positive InfoNCE loss.

    Args:
        scores: (B, N_s, N_h) similarity scores between seeds and hits.
        targets: (B, N_s, N_h) binary matrix, 1 if seed_i matches hit_j.
        temperature: softmax temperature.

    Returns:
        Scalar loss, averaged over batch and seeds that have positives.
    """
    B, N_s, N_h = scores.shape
    # Scale by temperature
    scores = scores / temperature

    # For numerical stability, subtract max per seed
    scores_max = scores.max(dim=-1, keepdim=True).values  # (B, N_s, 1)
    scores_stable = scores - scores_max

    # exp(scores)
    exp_scores = torch.exp(scores_stable)  # (B, N_s, N_h)

    # Denominator: sum over ALL hits
    denom = exp_scores.sum(dim=-1)  # (B, N_s)

    # Numerator: sum of exp(scores) over POSITIVE hits only
    num = (exp_scores * targets).sum(dim=-1)  # (B, N_s)

    # Per-seed loss: -log(num / denom)
    # Only compute for seeds that have at least one positive
    has_positives = targets.sum(dim=-1) > 0  # (B, N_s)

    loss_per_seed = -torch.log(num / denom.clamp(min=1e-8))  # (B, N_s)
    loss_per_seed = loss_per_seed * has_positives.float()

    # Average over seeds with positives
    n_valid = has_positives.float().sum()
    if n_valid == 0:
        return torch.tensor(0.0, device=scores.device)

    return loss_per_seed.sum() / n_valid
```

- [ ] **Step 4: Run tests**

Run:
```bash
uv run pytest tests/test_cast_loss.py -v
```

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ml_cookbook/models/cast/loss.py tests/test_cast_loss.py
git commit -m "feat: implement multi-positive InfoNCE loss"
```

---

### Task 8: Metrics module

**Files:**
- Create: `src/ml_cookbook/models/cast/metrics.py`
- Test: `tests/test_cast_metrics.py`

**Goal:** Implement binned efficiency and purity metrics for validation.

- [ ] **Step 1: Write metrics tests**

Create `tests/test_cast_metrics.py`:
```python
"""Tests for CAST metrics."""
import pytest
import torch
from ml_cookbook.models.cast.metrics import (
    compute_efficiency,
    compute_purity,
    compute_binned_metrics,
)


class TestComputeEfficiency:
    def test_perfect(self):
        targets = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])  # (2 seeds, 4 hits)
        preds = torch.tensor([
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ])
        eff = compute_efficiency(preds, targets)
        assert eff == 1.0

    def test_half(self):
        targets = torch.tensor([[1, 1, 0, 0]])
        preds = torch.tensor([[1, 0, 0, 0]])
        eff = compute_efficiency(preds, targets)
        assert eff == 0.5

    def test_empty_seed(self):
        targets = torch.tensor([[0, 0, 0, 0]])
        preds = torch.tensor([[0, 0, 0, 0]])
        eff = compute_efficiency(preds, targets)
        assert eff == 0.0  # no positives, no denominator


class TestComputePurity:
    def test_perfect(self):
        targets = torch.tensor([[1, 1, 0, 0]])
        preds = torch.tensor([[1, 1, 0, 0]])
        pur = compute_purity(preds, targets)
        assert pur == 1.0

    def test_false_positive(self):
        targets = torch.tensor([[1, 1, 0, 0]])
        preds = torch.tensor([[1, 1, 1, 0]])
        pur = compute_purity(preds, targets)
        assert pur == 2.0 / 3.0


class TestBinnedMetrics:
    def test_output_structure(self):
        targets = torch.zeros(3, 100)
        preds = torch.zeros(3, 100)
        kinematics = torch.randn(3, 6)  # (eta, pt, d0, z0, theta, phi)
        # Set some positives
        for i in range(3):
            targets[i, i*10:(i+1)*10] = 1
            preds[i, i*10:(i+1)*10] = 1

        result = compute_binned_metrics(preds, targets, kinematics)
        assert "eff_mean" in result
        assert "pur_mean" in result
        assert "eff_eta_bin0" in result  # binned keys exist
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
uv run pytest tests/test_cast_metrics.py -v
```

Expected: FAIL.

- [ ] **Step 3: Implement metrics**

Create `src/ml_cookbook/models/cast/metrics.py`:
```python
"""Evaluation metrics for CAST model — global and binned by kinematics."""

import torch


def _bin_values(values: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """Return bin index (0..n_bins) for each value. -1 means out of range."""
    bin_idx = torch.bucketize(values, bin_edges, right=False) - 1
    return bin_idx.clamp(-1, len(bin_edges) - 2)


def compute_efficiency(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-seed efficiency: fraction of true hits correctly predicted.

    Args:
        preds: (N_s, N_h) binary predictions
        targets: (N_s, N_h) binary ground truth
    Returns:
        Mean efficiency across seeds with ≥1 positive hit.
    """
    n_true = targets.sum(dim=-1)  # (N_s,)
    n_correct = (preds * targets).sum(dim=-1)  # (N_s,)
    has_pos = n_true > 0
    if has_pos.sum() == 0:
        return 0.0
    eff = n_correct[has_pos].float() / n_true[has_pos].float()
    return eff.mean().item()


def compute_purity(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Per-seed purity: fraction of predicted hits that are true positives.

    Args:
        preds: (N_s, N_h) binary predictions
        targets: (N_s, N_h) binary ground truth
    Returns:
        Mean purity across seeds with ≥1 prediction.
    """
    n_pred = preds.sum(dim=-1)  # (N_s,)
    n_correct = (preds * targets).sum(dim=-1)  # (N_s,)
    has_pred = n_pred > 0
    if has_pred.sum() == 0:
        return 0.0
    pur = n_correct[has_pred].float() / n_pred[has_pred].float()
    return pur.mean().item()


def compute_binned_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    kinematics: torch.Tensor,
) -> dict:
    """Compute metrics binned by eta, pT, d0, z0.

    Args:
        preds: (N_s, N_h) binary predictions
        targets: (N_s, N_h) binary ground truth
        kinematics: (N_s, 6) [eta, pt, d0, z0, theta, phi]

    Returns:
        Dict with keys like 'eff_mean', 'pur_mean', 'eff_eta_bin0', etc.
    """
    metrics = {}

    # Global
    metrics["eff_mean"] = compute_efficiency(preds, targets)
    metrics["pur_mean"] = compute_purity(preds, targets)

    # Bin definitions
    bin_configs = {
        "eta": (0, torch.tensor([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])),
        "pt": (1, torch.tensor([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])),
        "d0": (2, torch.tensor([0.0, 0.1, 0.5, 1.0, 5.0, 10.0])),
        "z0": (3, torch.tensor([0.0, 0.5, 1.0, 5.0, 10.0, 20.0])),
    }

    for name, (col_idx, bin_edges) in bin_configs.items():
        vals = kinematics[:, col_idx]
        bin_idx = _bin_values(vals, bin_edges)

        for b in range(len(bin_edges) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            bin_eff = compute_efficiency(preds[mask], targets[mask])
            metrics[f"eff_{name}_bin{b}"] = bin_eff

    return metrics
```

- [ ] **Step 4: Run tests**

Run:
```bash
uv run pytest tests/test_cast_metrics.py -v
```

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ml_cookbook/models/cast/metrics.py tests/test_cast_metrics.py
git commit -m "feat: implement binned efficiency and purity metrics"
```

---

### Task 9: CASTModel LightningModule

**Files:**
- Create: `src/ml_cookbook/models/cast/model.py`
- Modify: `tests/test_cast_shapes.py` (add model forward test)

**Goal:** Implement the full CAST LightningModule with training/validation steps.

- [ ] **Step 1: Add model forward shape test**

Append to `tests/test_cast_shapes.py`:
```python
from ml_cookbook.models.cast.model import CASTModel


class TestCASTModel:
    def test_forward_shape(self):
        model = CASTModel(
            d_model=128,
            n_cross_attn_layers=2,
            n_heads=4,
            ff_dim=512,
            dropout=0.0,
            temperature=0.1,
            fourier_L=4,
            use_cylindrical_pe=False,
            use_seed_self_attn=False,
            hit_encoder="identity",
            learning_rate=1e-3,
        )
        hits = torch.randn(2, 500, 6)   # (B, N_h, F_hit)
        seeds = torch.randn(2, 50, 9)   # (B, N_s, 9)
        scores = model(hits, seeds)
        assert scores.shape == (2, 50, 500)

    def test_training_step_runs(self):
        model = CASTModel(
            d_model=64,
            n_cross_attn_layers=1,
            n_heads=2,
            ff_dim=256,
            dropout=0.0,
            temperature=0.1,
            fourier_L=2,
            use_cylindrical_pe=False,
            use_seed_self_attn=False,
            hit_encoder="identity",
            learning_rate=1e-3,
        )
        batch = {
            "hits": torch.randn(1, 100, 3),
            "seeds": torch.randn(1, 10, 9),
            "targets": torch.zeros(1, 10, 100),
            "hit_particle_ids": torch.zeros(1, 100, dtype=torch.long),
            "seed_particle_ids": torch.zeros(1, 10, dtype=torch.long),
            "kinematics": torch.randn(1, 10, 6),
            "event_idx": 0,
        }
        batch["targets"][0, :5, :50] = 1.0  # some positives

        loss = model.training_step(batch, batch_idx=0)
        assert loss is not None
        assert loss.requires_grad
        assert loss.item() >= 0.0

    def test_configure_optimizers(self):
        model = CASTModel(
            d_model=64, n_cross_attn_layers=1, n_heads=2, ff_dim=256,
            dropout=0.0, temperature=0.1, fourier_L=2,
            use_cylindrical_pe=False, use_seed_self_attn=False,
            hit_encoder="identity", learning_rate=1e-3,
        )
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
```

- [ ] **Step 2: Run test to verify failure**

Run:
```bash
uv run pytest tests/test_cast_shapes.py::TestCASTModel -v
```

Expected: FAIL.

- [ ] **Step 3: Implement CASTModel**

Create `src/ml_cookbook/models/cast/model.py`:
```python
"""CAST LightningModule — Cross-Attention Seed Transformer."""

import torch
import torch.nn as nn
import lightning as L

from .embedders import HitEmbedder, SeedEmbedder
from .attention import CrossAttentionDecoder, SeedSelfAttention
from .encoders import IdentityEncoder
from .loss import info_nce_loss
from .metrics import compute_efficiency, compute_purity, compute_binned_metrics


class CASTModel(L.LightningModule):
    """Cross-Attention Seed Transformer for hit-to-track assignment."""

    def __init__(
        self,
        d_model: int = 256,
        n_cross_attn_layers: int = 3,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        temperature: float = 0.1,
        fourier_L: int = 8,
        use_cylindrical_pe: bool = True,
        use_seed_self_attn: bool = True,
        hit_encoder: str = "identity",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hit_embedder = HitEmbedder(
            d_model=d_model,
            fourier_L=fourier_L,
            use_detector_features=True,
            use_cylindrical_pe=use_cylindrical_pe,
            dropout=dropout,
        )
        self.seed_embedder = SeedEmbedder(
            d_model=d_model, dropout=dropout,
        )

        # Pluggable hit context encoder
        if hit_encoder == "identity":
            self.hit_encoder = IdentityEncoder()
        else:
            raise ValueError(f"Unknown hit_encoder: {hit_encoder}")

        self.cross_attn = CrossAttentionDecoder(
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            n_layers=n_cross_attn_layers,
            dropout=dropout,
        )

        if use_seed_self_attn:
            self.seed_self_attn = SeedSelfAttention(
                d_model=d_model, n_heads=n_heads, dropout=dropout,
            )
        else:
            self.seed_self_attn = None

        self.temperature = temperature

    def forward(self, hits: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between seeds and hits.

        Args:
            hits: (B, N_h, F_hit)
            seeds: (B, N_s, 9)
        Returns:
            scores: (B, N_s, N_h) unnormalized similarity scores
        """
        # Encode
        h_emb = self.hit_embedder(hits)       # (B, N_h, d)
        s_emb = self.seed_embedder(seeds)      # (B, N_s, d)

        # Contextual encoding of hits (identity or self-attention)
        h_emb = self.hit_encoder(h_emb)        # (B, N_h, d)

        # Cross-attention: seeds attend to hits
        s_emb = self.cross_attn(s_emb, h_emb)  # (B, N_s, d)

        # Optional seed self-attention
        if self.seed_self_attn is not None:
            s_emb = self.seed_self_attn(s_emb)

        # Similarity: dot product with temperature scaling
        scores = torch.bmm(s_emb, h_emb.transpose(1, 2))  # (B, N_s, N_h)
        scores = scores / self.temperature

        return scores

    def training_step(self, batch, batch_idx):
        # batch is a list of 1 dict when batch_size=1 with default_collate
        if isinstance(batch, list):
            sample = batch[0]
        else:
            sample = batch

        hits = sample["hits"]
        seeds = sample["seeds"]
        targets = sample["targets"]

        # hits: (1, N_h, F_hit), add batch dim if missing
        if hits.dim() == 2:
            hits = hits.unsqueeze(0)
            seeds = seeds.unsqueeze(0)
            targets = targets.unsqueeze(0)

        scores = self(hits, seeds)
        loss = info_nce_loss(scores, targets, temperature=1.0)  # scores already scaled

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list):
            sample = batch[0]
        else:
            sample = batch

        hits = sample["hits"]
        seeds = sample["seeds"]
        targets = sample["targets"]
        kinematics = sample.get("kinematics")

        if hits.dim() == 2:
            hits = hits.unsqueeze(0)
            seeds = seeds.unsqueeze(0)
            targets = targets.unsqueeze(0)

        scores = self(hits, seeds)
        loss = info_nce_loss(scores, targets, temperature=1.0)

        # Binary predictions: argmax per hit over seeds
        # scores: (1, N_s, N_h) -> best seed per hit
        best_seed = scores.squeeze(0).argmax(dim=0)  # (N_h,)
        N_s = scores.shape[1]
        preds = torch.zeros(N_s, scores.shape[2], device=scores.device)
        preds[best_seed, torch.arange(scores.shape[2])] = 1.0

        targets_2d = targets.squeeze(0)  # (N_s, N_h)

        # Global metrics
        eff = compute_efficiency(preds, targets_2d)
        pur = compute_purity(preds, targets_2d)

        metrics = {
            "val_loss": loss,
            "val_eff": eff,
            "val_pur": pur,
        }

        # Binned metrics (if kinematics available)
        if kinematics is not None and kinematics.numel() > 0:
            kin = kinematics if kinematics.dim() == 2 else kinematics.squeeze(0)
            binned = compute_binned_metrics(preds, targets_2d, kin)
            metrics.update({f"val_{k}": v for k, v in binned.items()})

        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 50,
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=5,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
```

- [ ] **Step 4: Run model tests**

Run:
```bash
uv run pytest tests/test_cast_shapes.py::TestCASTModel -v
```

Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ml_cookbook/models/cast/model.py tests/test_cast_shapes.py
git commit -m "feat: implement CASTModel LightningModule"
```

---

### Task 10: Update training entrypoint

**Files:**
- Modify: `src/ml_cookbook/train.py` (lines 15-17 for imports, lines 38-68 for main)
- Modify: `configs/cast_default.yaml` (add `type: cast`)

**Goal:** Extend the training entrypoint to support CAST model and ColliderML data when config specifies `model.type: cast`.

- [ ] **Step 1: Write the exact updated train.py**

Replace `src/ml_cookbook/train.py` with:
```python
from __future__ import annotations

import argparse
import os
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from ml_cookbook.data.dummy_datamodule import DummyDataModule
from ml_cookbook.models.prototype_model import PrototypeModel
from ml_cookbook.utils.repro import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a minimal Lightning prototype."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    deterministic = bool(cfg.get("trainer", {}).get("deterministic", True))
    seed_everything(seed=seed, deterministic=deterministic)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]

    model_type = model_cfg.get("type", "prototype")

    if model_type == "cast":
        from ml_cookbook.models.cast.model import CASTModel
        from ml_cookbook.data.seed_extension.dataset import SeedExtensionDataset
        from colliderml_dataloader import ColliderMLDataModule

        import ml_cookbook.data.seed_extension  # noqa: F401 — monkey-patches features

        datamodule = ColliderMLDataModule(
            data_dir=os.environ.get(
                "COLLIDERML_DATA_DIR",
                "/pscratch/sd/p/pmtuan/.cache/colliderml",
            ),
            process=data_cfg["process"],
            pileup=data_cfg["pileup"],
            max_train_events=data_cfg["max_train_events"],
            max_val_events=data_cfg["max_val_events"],
            max_test_events=data_cfg["max_test_events"],
            batch_size=1,
            num_workers=data_cfg["num_workers"],
            dataset_cls=SeedExtensionDataset,
            min_track_hits=data_cfg.get("min_track_hits", 5),
            seed_strategy=data_cfg.get("seed_strategy", "random_consecutive"),
            target_vertices=data_cfg.get("target_vertices", 200),
            primary_only=data_cfg.get("primary_only", False),
            predict_seed_hits=data_cfg.get("predict_seed_hits", False),
        )

        model = CASTModel(
            d_model=int(model_cfg["d_model"]),
            n_cross_attn_layers=int(model_cfg["n_cross_attn_layers"]),
            n_heads=int(model_cfg["n_heads"]),
            ff_dim=int(model_cfg["ff_dim"]),
            dropout=float(model_cfg["dropout"]),
            temperature=float(model_cfg["temperature"]),
            fourier_L=int(model_cfg["fourier_L"]),
            use_cylindrical_pe=bool(model_cfg["use_cylindrical_pe"]),
            use_seed_self_attn=bool(model_cfg["use_seed_self_attn"]),
            hit_encoder=str(model_cfg["hit_encoder"]),
            learning_rate=float(model_cfg["learning_rate"]),
            weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
            lr_scheduler=str(model_cfg.get("lr_scheduler", "cosine")),
            warmup_steps=int(model_cfg.get("warmup_steps", 1000)),
        )
    else:
        input_shape = tuple(data_cfg["input_shape"])
        num_classes = int(data_cfg["num_classes"])

        datamodule = DummyDataModule(
            batch_size=int(data_cfg["batch_size"]),
            input_shape=input_shape,
            num_classes=num_classes,
            train_samples=int(data_cfg["train_samples"]),
            val_samples=int(data_cfg["val_samples"]),
            test_samples=int(data_cfg["test_samples"]),
            num_workers=int(data_cfg["num_workers"]),
        )

        model = PrototypeModel(
            input_shape=input_shape,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_classes=num_classes,
            learning_rate=float(model_cfg["learning_rate"]),
        )

    exp_name = cfg.get("experiment_name", "baseline")
    logger = CSVLogger(save_dir="logs", name=exp_name)

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best",
        ),
        EarlyStopping(monitor="val_loss", mode="min", patience=3),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = L.Trainer(
        max_epochs=int(trainer_cfg["max_epochs"]),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", 1),
        deterministic=bool(trainer_cfg.get("deterministic", True)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 1)),
        enable_progress_bar=bool(trainer_cfg.get("enable_progress_bar", True)),
        limit_train_batches=trainer_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_cfg.get("limit_val_batches", 1.0),
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `type: cast` to config**

Add this line under the `model:` section in `configs/cast_default.yaml`:
```yaml
model:
  type: cast
  d_model: 256
  ...
```

- [ ] **Step 3: Run existing tests to verify no regression**

Run:
```bash
uv run pytest tests/test_shapes.py tests/test_train_smoke.py -v
```

Expected: Existing tests still PASS.

- [ ] **Step 4: Commit**

```bash
git add src/ml_cookbook/train.py configs/cast_default.yaml
git commit -m "feat: integrate CAST model into training entrypoint"
```

---

### Task 11: Smoke test

**Files:**
- Create: `tests/test_cast_smoke.py`

**Goal:** End-to-end smoke test: 1 epoch of training on a few events.

- [ ] **Step 1: Write smoke test**

Create `tests/test_cast_smoke.py`:
```python
"""Smoke test: 1-epoch CAST training on a few ColliderML events."""
import os
import pytest
import torch
import lightning as L

from ml_cookbook.models.cast.model import CASTModel
from ml_cookbook.data.seed_extension.dataset import SeedExtensionDataset
from colliderml_dataloader import ColliderMLDataModule


@pytest.mark.slow
def test_cast_smoke_train():
    """Train CAST for 1 epoch on 4 events, verify loss decreases."""
    datadir = os.environ.get("COLLIDERML_DATA_DIR", "/pscratch/sd/p/pmtuan/.cache/colliderml")

    dm = ColliderMLDataModule(
        data_dir=datadir,
        process="ttbar",
        pileup="pu200",
        max_train_events=4,
        max_val_events=2,
        max_test_events=0,
        batch_size=1,
        num_workers=0,
        dataset_cls=SeedExtensionDataset,
        min_track_hits=5,
        seed_strategy="fixed_innermost",
        target_vertices=1,            # Hard scatter only — fewer hits, faster
        primary_only=True,            # Only primary particles
        predict_seed_hits=False,
    )

    model = CASTModel(
        d_model=64,
        n_cross_attn_layers=1,
        n_heads=2,
        ff_dim=256,
        dropout=0.0,
        temperature=0.1,
        fourier_L=2,
        use_cylindrical_pe=False,
        use_seed_self_attn=False,
        hit_encoder="identity",
        learning_rate=1e-3,
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_train_batches=2,
        limit_val_batches=1,
    )

    trainer.fit(model, dm)
    assert trainer.state.finished
```

- [ ] **Step 2: Run smoke test**

Run:
```bash
uv run pytest tests/test_cast_smoke.py::test_cast_smoke_train -v -s
```

Expected: PASS. Training should complete in < 1 minute on 4 events with minimal model.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cast_smoke.py
git commit -m "test: add CAST 1-epoch smoke training test"
```

---

### Task 12: Final verification

**Files:** (none new)

**Goal:** Run full test suite to verify everything works together.

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS (smoke tests may be skipped if data not available).

- [ ] **Step 2: Run lint**

```bash
make lint
```

Expected: No syntax errors.

- [ ] **Step 3: Verify training can start**

```bash
uv run python train.py --config configs/cast_default.yaml
```

Let it run for a few steps, then Ctrl-C. Confirm no import errors, data loads correctly, loss is finite.

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final fixes for CAST integration"
```
