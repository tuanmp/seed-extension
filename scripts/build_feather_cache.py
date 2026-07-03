#!/usr/bin/env python3
"""Build a feather cache of exploded ColliderML events from source Parquet.

Usage:
  python scripts/build_feather_cache.py \
    --process ttbar --pileup pu200 \
    --data-dir /pscratch/sd/p/pmtuan/.cache/colliderml \
    --cache-dir /pscratch/sd/p/pmtuan/.cache/colliderml-feather \
    --max-events 100000 \
    --num-workers 16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.feather as feather
from colliderml.polars import explode_particles, explode_tracker_hits
from colliderml_dataloader.shard_index import (
    PARTICLE_FEATURES,
    TRACKER_HIT_FEATURES,
    build_shard_index,
    shard_dir_for,
)

import seed_extension.data  # noqa: F401 — monkey-patch feature lists

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_cache(
    data_dir: str,
    process: str,
    pileup: str,
    cache_root: str,
    max_events: int,
    num_workers: int = 8,
    compression: str = "zstd",
    compression_level: int = 3,
) -> None:
    cache_path = Path(cache_root)
    hits_dir = cache_path / "hits"
    parts_dir = cache_path / "parts"
    meta_path = cache_path / "meta.json"

    hits_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Discover source shards.
    hit_shard_dir = shard_dir_for(data_dir, process, pileup, "tracker_hits")
    part_shard_dir = shard_dir_for(data_dir, process, pileup, "particles")

    logger.info("Building event index from %s ...", hit_shard_dir)
    hit_index = build_shard_index(hit_shard_dir, max_events)
    part_index = build_shard_index(part_shard_dir, max_events)
    all_event_ids = sorted(set(hit_index) & set(part_index))[:max_events]
    logger.info("Found %d events.", len(all_event_ids))

    # Group events by shard file — each shard is one work unit.
    shard_to_events: dict[str, list[int]] = {}
    for eid in all_event_ids:
        shard = hit_index[eid]
        shard_to_events.setdefault(shard, []).append(eid)

    shard_files = sorted(shard_to_events.keys())
    logger.info("Processing %d shard files with %d workers.", len(shard_files), num_workers)

    # Write meta.json early (so partial builds are identifiable).
    meta = {
        "version": 1,
        "process": process,
        "pileup": pileup,
        "hit_cols": TRACKER_HIT_FEATURES,
        "part_cols": PARTICLE_FEATURES,
        "n_events": len(all_event_ids),
        "feather_compression": compression,
        "feather_compression_level": compression_level,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Build work items: (hit_file, part_file, [event_ids]).
    work_items = []
    for shard_path in shard_files:
        part_path = part_index[shard_to_events[shard_path][0]]
        events_in_shard = shard_to_events[shard_path]
        work_items.append((shard_path, part_path, events_in_shard))

    t0 = time.perf_counter()
    with Pool(processes=num_workers) as pool:
        results = pool.starmap(
            _process_shard,
            [
                (item, str(hits_dir), str(parts_dir), compression, compression_level)
                for item in work_items
            ],
        )

    total_events = sum(results)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Built cache for %d events in %.1f minutes (%.2f s/event).",
        total_events, elapsed / 60, elapsed / max(total_events, 1),
    )


def _process_shard(
    work_item: tuple[str, str, list[int]],
    hits_dir: str,
    parts_dir: str,
    compression: str,
    compression_level: int,
) -> int:
    hit_file, part_file, event_ids = work_item
    n_written = 0

    for eid in event_ids:
        out_hit = Path(hits_dir) / f"{eid}.feather"
        out_part = Path(parts_dir) / f"{eid}.feather"
        if out_hit.exists() and out_part.exists():
            n_written += 1
            continue

        hits_raw = (
            pl.scan_parquet(hit_file)
            .filter(pl.col("event_id") == eid)
            .select(TRACKER_HIT_FEATURES)
            .collect()
        )
        parts_raw = (
            pl.scan_parquet(part_file)
            .filter(pl.col("event_id") == eid)
            .select(PARTICLE_FEATURES)
            .collect()
        )

        hits_expl = explode_tracker_hits(hits_raw).drop(columns=["event_id"])
        parts_expl = explode_particles(parts_raw).drop(columns=["event_id"])

        feather.write_feather(
            hits_expl, out_hit, compression=compression,
            compression_level=compression_level,
        )
        feather.write_feather(
            parts_expl, out_part, compression=compression,
            compression_level=compression_level,
        )
        n_written += 1

    return n_written


def main() -> None:
    p = argparse.ArgumentParser(description="Build feather cache for ColliderML.")
    p.add_argument("--process", default="ttbar")
    p.add_argument("--pileup", default="pu200")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--max-events", type=int, default=100000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--compression", default="zstd")
    p.add_argument("--compression-level", type=int, default=3)
    args = p.parse_args()

    build_cache(
        data_dir=args.data_dir,
        process=args.process,
        pileup=args.pileup,
        cache_root=args.cache_dir,
        max_events=args.max_events,
        num_workers=args.num_workers,
        compression=args.compression,
        compression_level=args.compression_level,
    )


if __name__ == "__main__":
    main()
