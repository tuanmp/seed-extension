# CAST: Cross-Attention Seed Transformer for Charged Particle Tracking

**Date**: 2026-06-30
**Status**: Design approved, pending implementation

## 1. Motivation

Particle tracking at HL-LHC involves associating ~2.5×10⁵ detector hits per event into ~10⁴ tracks. State-of-the-art approaches use self-attention over all hits (O(N²) ≈ 6×10¹⁰), or GNNs with explicit edge-building. Both are expensive.

**Key insight**: A small set of seeds (~10³ triplets from innermost pixel layers) can serve as queries that cross-attend to all hits. Cross-attention complexity is O(N_seeds × N_hits) ≈ 2.5×10⁸ — a ~250× reduction over self-attention. This idea has no direct precedent in the literature; the closest works are DETR/MaskFormer-style track queries (Van Stroud et al., PRX 2025) and HEPTv2's point transformer (Miao et al., 2026).

## 2. Dataset

**Source**: ColliderML Release 1 (`ttbar_pu200`, tracker hits + particles configs) on CERN HuggingFace. Locally cached at `/pscratch/sd/p/pmtuan/.cache/colliderml`.

**Scale per event**:
- Hits: 183k–342k (mean ~248k), each with x, y, z, particle_id, layer_id, volume_id, detector
- Particles: 134k–242k (mean ~178k), with px, py, pz, perigee_d0, perigee_z0, particle_id, primary

## 3. Data Pipeline

### 3.1 Seed Construction

| Phase | Strategy | Description |
|-------|----------|-------------|
| Training | Random consecutive | Each epoch, randomly sample 3 hits from consecutive layers among the particle's pixel hits |
| Validation | Fixed innermost | Take 3 innermost consecutive-layer hits per particle |
| Test | Fixed innermost | Same as validation |

Controlled by `data.seed_strategy` (`"random_consecutive"` or `"fixed_innermost"`).

### 3.2 Filtering

- `min_track_hits`: minimum hits per particle beyond seed triplet (default: 5)
- `primary_only`: if true, only particles with `primary=True` (default: false, configurable)
- `predict_seed_hits`: whether to include seed hits in assignment target (default: false)

### 3.3 Features

- **Hit features**: x, y, z + layer_id, volume_id, detector (one-hot or learned embedding)
- **Particle features**: px, py, pz, perigee_d0, perigee_z0 (used for metric computation, not model input)

### 3.4 Output per event

| Tensor | Shape | Description |
|--------|-------|-------------|
| `hits` | (N_h, F_hit) | Hit features |
| `seeds` | (N_s, 9) | 3 seed hits × (x, y, z) |
| `hit_particle_ids` | (N_h,) | Truth particle_id per hit |
| `seed_particle_ids` | (N_s,) | Truth particle_id per seed |
| `targets` | (N_s, N_h) | Sparse binary matrix: 1 if seed and hit share particle_id |
| `particle_kinematics` | (N_s, 6) | η, pT, d0, z0, θ, φ per seed (for metric binning) |

### 3.5 Implementation

Subclass `ColliderMLDataset`, override `_process_event(hits_raw, parts_raw, event_idx)`. Must extend `TRACKER_HIT_FEATURES` in colliderml-dataloader to include `layer_id`, `volume_id`, `detector`.

## 4. Model Architecture

### 4.1 Overview

```
Hits (N_h, 3)                     Seeds (N_s, 9)
     │                                  │
FourierEncode(L=8)                HitCoordEmbedder (3 hits)
→ (N_h, 48)                       + MLP = (N_s, d)
     │                                  │
HitEncoder (MLP): (N_h, d)              │
+ DetectorEmb (layer,vol,det)           │
     │                                  │
Pluggable HitContextEncoder ────────────┤
(Identity or BinnedSelfAttn)            │
     │                                  │
     └──── Cross-Attention (×L) ←──────┘
     │            seeds=Q, hits=K,V
     │
Optional Seed Self-Attention
     │
Similarity: s_ij = seed_i · hit_j / τ
     │
InfoNCE Loss / Threshold Inference
```

### 4.2 Components

#### Fourier Encoding
NeRF-style multi-scale encoding (Tancik et al., NeurIPS 2020) applied to normalized (x, y, z):
```
fourier_encode(x, L=8) = [sin(2⁰πx), cos(2⁰πx), ..., sin(2⁷πx), cos(2⁷πx)]
```
Output: 3 × 2L = 48 dims. Also applies to cylindrical (r, φ, z) for an additional 48 dims. Configuration: `model.fourier_L`, `model.use_cylindrical_pe`.

#### Hit Embedder
MLP: Fourier features + detector embeddings → d_model (default 256). 2 hidden layers with GELU.

#### Seed Embedder
Concatenate 3 seed hit coordinates (9 dims) → MLP → d_model. Alternatively, embed each seed hit through HitEmbedder and mean-pool (configurable).

#### Pluggable Hit Context Encoder
Interface: `forward(hit_embeddings) → contextualized_hit_embeddings`. Implementations:

| Encoder | Description | When used |
|---------|-------------|-----------|
| `IdentityEncoder` | Pass-through | Default — establishes baseline |
| `BinnedSelfAttention` | Self-attention within spatial bins (η-φ grid or LSH) | Ablation experiment |

#### Cross-Attention Decoder
L layers (default 3) of standard Transformer decoder layers, but **without** causal masking. Seeds attend to all hits:
```
Q = W_q(seed_emb), K = W_k(hit_emb), V = W_v(hit_emb)
A = softmax(QK^T / √d_head)
output = LayerNorm(seed_emb + Dropout(A · V))
output = LayerNorm(output + FFN(output))
```

#### Seed Self-Attention (optional)
Allows seeds to disambiguate overlapping tracks. Standard multi-head self-attention among seed embeddings. O(N_s²) ≈ 10⁶ — negligible.

### 4.3 Loss: Multi-Positive InfoNCE

For each seed i with positives P_i = {j : particle_id(seed_i) == particle_id(hit_j)}:
```
s_ij = (seed_i · hit_j) / τ
L_i = -log( Σ_{j∈P_i} exp(s_ij) / Σ_{all j} exp(s_ij) )
L = mean_i L_i
```

Properties:
- No separate classifier head
- Naturally handles variable track lengths
- Softmax creates competition among hits per seed
- Temperature τ controls sharpness (default 0.1)

### 4.4 Inference

Per hit j, assign to seed i* = argmax_i s_ij. Apply threshold θ on s_i*j to reject unassigned hits (from untracked particles). Threshold tuned on validation set to maximize F1.

## 5. Evaluation Metrics

### 5.1 Global Metrics

| Metric | Definition |
|--------|-----------|
| Seed efficiency | Per particle: fraction of non-seed hits correctly assigned. Mean across particles. |
| Hit purity | Per seed: fraction of assigned hits that belong to the seed's true particle. Mean across seeds. |
| Track efficiency | Fraction of particles with ≥50% hits correctly assigned (HEP standard). |
| Fake seed rate | Seeds accumulating hits from other particles above threshold. |
| InfoNCE loss | Validation/test loss. |

### 5.2 Binned Metrics

All efficiency metrics computed in bins of:

| Variable | Bins |
|----------|------|
| η (pseudorapidity) | [-2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5] |
| pT (transverse momentum, GeV) | [0, 1, 2, 5, 10, 20, 50, 100] |
| d0 (transverse impact parameter, mm) | [0, 0.1, 0.5, 1.0, 5.0, 10.0] |
| z0 (longitudinal impact parameter, mm) | [0, 0.5, 1.0, 5.0, 10.0, 20.0] |

For each bin: mean ± std across events. Global standard deviation of each metric also logged.

η and pT computed from particle momentum (px, py, pz) at dataset construction. d0 and z0 from particles table (`perigee_d0`, `perigee_z0`).

### 5.3 Logging

Logged per validation epoch via Lightning `self.log_dict()`. Binned metrics stored in dict with keys like `eff_eta_bin0`, `eff_eta_bin0_std`, etc.

## 6. Training Configuration

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
  hit_encoder: identity          # "identity" | "binned_self_attn" | "knn_self_attn"
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

## 7. File Structure

```
src/ml_cookbook/
├── data/
│   ├── dummy_datamodule.py          (existing)
│   ├── dummy_dataset.py             (existing)
│   └── seed_extension/
│       ├── __init__.py
│       └── dataset.py               # SeedExtensionDataset
├── models/
│   ├── prototype_model.py           (existing)
│   └── cast/
│       ├── __init__.py
│       ├── embedders.py             # FourierEncode, HitEmbedder, SeedEmbedder
│       ├── attention.py             # CrossAttentionBlock, CrossAttentionDecoder
│       ├── encoders.py              # IdentityEncoder, BinnedSelfAttentionEncoder
│       ├── model.py                 # CASTModel (LightningModule)
│       ├── loss.py                  # info_nce_loss, compute_metrics
│       └── metrics.py               # Binned metrics, logging helpers
├── utils/
│   └── repro.py                     (existing)
configs/
├── default.yaml                     (existing)
└── cast_default.yaml                # CAST experiment config
tests/
├── test_shapes.py                   (existing)
├── test_train_smoke.py              (existing)
├── test_cast_shapes.py              # Shape tests
└── test_cast_smoke.py               # 1-epoch smoke test
```

## 8. Scaling Path

| Technique | When | Expected gain |
|-----------|------|---------------|
| bf16 mixed precision | Always | 2× memory, ~1.5× speed |
| Gradient checkpointing | If OOM at BS=1 | ~2× memory, ~20% slower |
| FlashAttention | If A100/H100 available | ~5-10× memory, ~2× speed |
| LSH-bucketed cross-attention | If 80GB still OOMs | O(N_s × N_bucket) vs O(N_s × N_h) |
| Coarse→fine two-level attention | Production scaling | 10-100× for coarse stage |

## 9. Implementation Order

| Phase | Files | Validation |
|-------|-------|-----------|
| 1. Dataset | `dataset.py` | Manual inspection of seed/target shapes |
| 2. Embedders + model stub | `embedders.py`, `model.py` | `test_cast_shapes.py` |
| 3. Attention + loss | `attention.py`, `loss.py`, `metrics.py` | Forward/backward passes, metric shapes |
| 4. Config + smoke test | `cast_default.yaml`, `test_cast_smoke.py` | 1-epoch on 100 events |
| 5. Full training | — | Baseline metrics, binned efficiencies |
| 6. Hit encoder ablation | `encoders.py`, config swap | Compare `identity` vs `binned_self_attn` |
