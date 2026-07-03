# CAST Training Pipeline Profiling

**Date**: 2026-07-03
**GPU**: NVIDIA A100-PCIE-40GB

## Profiling Results

Per-event timing with various pileup levels:

### target_vertices=1 (hard scatter only)

| Stage | Time (ms) | % |
|-------|----------|---|
| Data loading (CPU) | 1,326 | 99.4% |
| GPU Transfer | 0.4 | 0.0% |
| Forward Pass | 2.8 | 0.2% |
| Loss (InfoNCE) | 0.4 | 0.0% |
| Backward | 4.8 | 0.4% |
| **TOTAL** | **1,335** | |

GPU forward sub-stages: CrossAttn 75%, HitEmbedder 15%, Scores 5%, SeedEmbedder 8%.

Event: 4,310 hits, 118 seeds.

### target_vertices=50

| Stage | Time (ms) | % GPU |
|-------|----------|-------|
| Data loading (CPU) | **7,444** | — |
| GPU Transfer | 66 | — |
| Forward Pass | 92 | 100% |
| └ CrossAttn (2×) | 84.8 | 91.7% |
| └ HitEmbedder | 2.5 | 2.7% |
| Loss (InfoNCE) | 372 | — |
| Backward | 548 | — |
| **TOTAL** | **~8,500** | |

Event: 58,533 hits, 3,902 seeds.
Target matrix: 229M elements, 914 MB fp32.
Attention per layer: 229M elements.

### GPU utilization

At pu=50, GPU is idle **88% of the time** waiting for the CPU data pipeline.
At pu=1, GPU is idle **99% of the time**.

## Bottlenecks

| Priority | Component | CPU/GPU | Root cause |
|----------|-----------|---------|------------|
| 1 | Data loading | CPU | `build_seeds_*()` uses pandas `iterrows()` over O(10^3-10^4) particles per event |
| 2 | Cross-attention | GPU | O(N_s × N_h) attention matrix: 229M elements at pu=50, projects to ~1B at pu=200 |
| 3 | InfoNCE loss | GPU | Dense (N_s × N_h) target matrix transferred CPU→GPU each event |
| 4 | Dense target matrix | CPU+GPU | 914 MB fp32 materialized per event; transferred and stored |

## Optimization Proposals

### 1. Vectorize seed construction (HIGH, CPU) — **IMPLEMENTED**

~~Replace pandas `iterrows()` with polars `group_by + agg` in `build_seeds_fixed` and `build_seeds_random_consecutive`. Polars executes group-by operations in native code (Rust/C), eliminating the Python loop.~~

~~**Expected**: 7,400ms → ~50-200ms per event (37-148× improvement for pu=50).~~

**Implemented**: Replaced `build_seeds_fixed` and `build_seeds_random_consecutive` with polars group-by + aggregate. Measured at pu=50 (58,533 hits, 3,905 seeds):

| Method | Time | Improvement |
|--------|------|-------------|
| pandas iterrows | 3,994 ms | — |
| polars groupby | 13 ms | **313×** |

The kinematics are now computed in bulk (vectorized numpy `compute_kinematics`) instead of one call per particle inside the loop.

### 2. Bulk kinematics computation (MEDIUM, CPU)

Currently `compute_kinematics` is called once per particle inside the loop. Pre-compute eta/pT for all particles at once using numpy vectorized ops.

**Expected**: Eliminates O(N_s) redundant function call overhead.

### 3. Sparse target representation (HIGH, CPU+GPU)

Replace the dense (N_s × N_h) binary target matrix with a list of positive hit indices per seed. Adapt `info_nce_loss` to use `torch.nn.functional.cross_entropy` or custom sparse gather.

**Expected**: 914 MB → ~100 KB per event. Eliminates O(N_s × N_h) transfer and memory.

### 4. Spatial hit filtering (HIGH, GPU)

Pre-select hits within Δη/Δφ of each seed before cross-attention. Reduces N_h from 58k to ~2-10k relevant hits per seed.

**Expected**: 10-50× cross-attention speedup.

### 5. Mixed precision (bf16) (LOW, GPU)

Enable `torch.cuda.amp` autograd or bf16-mixed precision. A100 has native bf16 support.

**Expected**: 1.3-1.8× GPU speedup, ~2× memory reduction.
