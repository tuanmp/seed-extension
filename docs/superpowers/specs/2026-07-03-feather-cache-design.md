## Feather Cache Design for ColliderMLDataset

### Problem

Training on ColliderML spends 1.6s per event in Parquet I/O + polars explode (83% of event load time). At 50k events × 8 workers, this is the dominant bottleneck. The kernel page cache helps after epoch 0, but the polars explode still costs 1.0s per event. A cache that stores already-exploded data in a fast-read format eliminates this cost once and for all — a one-time build investment that pays off across all future training runs.

### Constraint: variable-length events

ColliderML events have different numbers of hits and particles (5k–215k rows). This rules out dense memmap arrays (caloxtreme's approach), which require fixed-shape tensors. Variable-length events need a format that stores DataFrames of arbitrary row counts per file — Feather (Arrow IPC) is ideal.

### Storage Format: Feather + zstd

Perf benchmarks on one pu200 event (214k hits, 153k particles, 12.3 MB pandas):

| Format | Read → pandas DF | Disk per event | Notes |
|--------|-------------------|----------------|-------|
| Parquet list-col + explode (current) | 1,629 ms | ~15 MB source | polars scan + filter + explode |
| Feather zstd | 15.6 ms | 6.2 MB | direct pandas reconstruction |
| NumPy npz (compressed) | 47 ms (decompress) | 5.0 MB | loses column names and dtypes |
| Raw memmap (float32 only) | 2.0 ms (forced read) | 8.2 MB | loses int/bool columns, no compression |

Feather is 104× faster than current, returns proper pandas DFs with native dtypes, and is 2.5× smaller on disk than source Parquet.

### Cache Layout

```
{cache_dir}/
├── meta.json
│     { "version": 1,
│       "process": "ttbar",
│       "pileup": "pu200",
│       "hit_cols": ["x","y","z","particle_id","layer_id","volume_id","detector"],
│       "part_cols": ["particle_id","px","py","pz","primary","pdg_id",
│                     "perigee_d0","perigee_z0","vertex_primary"],
│       "n_events": 100000,
│       "created_at": "2026-07-03T12:00:00Z",
│       "feather_compression": "zstd",
│       "feather_compression_level": 3 }
├── hits/
│   ├── {event_id}.feather           # (N_h, 7) — exploded tracker hits
│   └── ...
├── parts/
│   ├── {event_id}.feather           # (N_p, 9) — exploded particles
│   └── ...
├── train_ids.json                    # list of event_ids assigned to train split
├── val_ids.json                      # list of event_ids assigned to val split
└── test_ids.json                     # list of event_ids assigned to test split
```

Two files per event (hits/parts in separate subdirectories to avoid 200k files in one directory) and split IDs stored separately so split boundaries can change without rebuilding the cache.

### On-Disk Size Extrapolation

| Scenario | Events | Per event | Total disk |
|----------|--------|-----------|------------|
| pu200, full exploded | 100,000 | 6.2 MB | **607 GB** |
| pu200, typical 50k training | 50,000 | 6.2 MB | 310 GB |

### Components

#### 1. `build_feather_cache.py` — one-time build script

Walks all source Parquet shards for a given process/pileup, reads each shard's 100 events, explodes them, writes feather files.

- Reuses `colliderml.polars.explode_tracker_hits` / `explode_particles`
- Processes shards in parallel with `multiprocessing` (one worker per shard group)
- Each worker reads a full Parquet shard (1.5 GB, 100 events), explodes all 100 events, writes 200 feather files
- Handles resume: skips events where feather files already exist
- Column selection matches `TRACKER_HIT_FEATURES` / `PARTICLE_FEATURES` from `src/seed_extension/data/__init__.py`
- Writes `meta.json` on completion, deduces splits from `max_train_events` / `max_val_events`
- Estimated build time: 2.5s/event × 100,000 ÷ N_workers. With 32 workers: ~2.2 hours.

#### 2. `CachedColliderMLDataset(ColliderMLDataset)` — torch Dataset

Replaces `__getitem__` to read from feather cache instead of Parquet:

```python
def __getitem__(self, idx: int) -> dict:
    event_id = self.event_ids[idx]
    hits = feather.read_feather(f"{self.cache_root}/hits/{event_id}.feather")
    parts = feather.read_feather(f"{self.cache_root}/parts/{event_id}.feather")
    return self._process_event(hits, parts, idx)
```

All post-load processing (`_subsample_pileup`, `_filter_particles`, `_build_seeds`, `_assemble_sample`) is identical to `SeedExtensionDataset`. The only difference is the source of `hits_raw` / `parts_raw`.

Disabled `_raw_cache` as before — feather reads are fast enough (15.6 ms) that per-worker caching is unnecessary and would OOM at scale.

#### 3. `CachedColliderMLDataModule` — Lightning DataModule

Extends `ColliderMLDataModule` to add `cache_dir` and `cache_mode`.

```python
class CachedColliderMLDataModule(ColliderMLDataModule):
    def __init__(self, cache_dir: str | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir

    def setup(self, stage: str | None = None) -> None:
        if self.cache_dir is None:
            return super().setup(stage)
        cache_root = Path(self.cache_dir)
        if not (cache_root / "meta.json").exists():
            rank_zero_warn(f"Cache not found at {cache_root}, falling back to Parquet")
            return super().setup(stage)
        # Build file maps from cache instead of Parquet
        self._setup_from_cache(cache_root, stage)
```

In `_setup_from_cache`: reads `meta.json` for column list verification, reads `train_ids.json` / `val_ids.json` for split event IDs, creates `CachedColliderMLDataset` instances.

When `cache_dir` is set but cache doesn't exist → emit a warning and fall back to Parquet (so training still works, just slow).

### Config Changes

Add to `configs/cast_default.yaml` and per-experiment configs:

```yaml
data:
  cache_dir: null              # or /pscratch/sd/p/pmtuan/.cache/colliderml-feather
```

### Non-Goals

- **Cross-process shared cache**: Each worker reads feather files independently. Feather reads are fast enough (15.6 ms) that the added complexity of shared-memory caching is not worth it.
- **Incremental cache updates**: If the ColliderML dataset grows, rebuild the cache. Rebuilding from scratch is fast enough (a few hours).
- **Compressed targets matrix**: The targets matrix is recomputed per event from particle IDs (cheap `==` broadcast). Caching it would cost 914 MB/event × 50k ≈ 46 TB — not feasible.
- **Pre-subsampled cache**: Cache stores full exploded data so any `target_vertices` config works. Subsampling is cheap (a few ms).

### Cache Validation

`meta.json` records column lists. On load, `CachedColliderMLDataset` verifies the cached columns match the current `TRACKER_HIT_FEATURES` and `PARTICLE_FEATURES`. Mismatch → warn and fall back to Parquet (or raise). The `version` field allows forced invalidation.
