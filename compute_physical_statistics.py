#!/usr/bin/env python3
"""Compute contact/force normalization statistics from train clips only."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np

from data_common import load_collection, resolve_episode_path, write_json


def compute_statistics(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    _, clips = load_collection(args.clips, "clips")
    episodes = sorted({str(clip["episode"]) for clip in clips})

    entries: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for clip in clips:
        contact_paths = clip.get("contact_paths") or []
        force_paths = clip.get("force_paths") or []
        if len(contact_paths) != len(force_paths):
            raise ValueError(f"{clip.get('episode')}: contact/force path count mismatch")
        for contact_path, force_path in zip(contact_paths, force_paths):
            entry = (str(clip["episode"]), str(contact_path), str(force_path))
            if args.deduplicate and entry in seen:
                continue
            seen.add(entry)
            entries.append(entry)

    entries.sort()
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    available_frames = len(entries)
    if args.n_sample_frames > 0:
        entries = entries[: args.n_sample_frames]

    contact_max = np.zeros(2, dtype=np.float64)
    force_sum = np.zeros(6, dtype=np.float64)
    force_sq_sum = np.zeros(6, dtype=np.float64)
    force_count = np.zeros(6, dtype=np.int64)
    force_abs_max = np.zeros(6, dtype=np.float64)
    force_min = np.full(6, np.inf, dtype=np.float64)
    force_max = np.full(6, -np.inf, dtype=np.float64)
    spatial_shape: tuple[int, int] | None = None

    for episode, contact_relative, force_relative in entries:
        contact_file = resolve_episode_path(source_root, episode, contact_relative)
        force_file = resolve_episode_path(source_root, episode, force_relative)
        if not contact_file.is_file() or not force_file.is_file():
            raise FileNotFoundError(f"missing pair: {contact_file}, {force_file}")
        contact = np.load(contact_file).astype(np.float64, copy=False)
        force = np.load(force_file).astype(np.float64, copy=False)
        if contact.ndim != 3 or contact.shape[0] != 2:
            raise ValueError(f"{contact_file}: expected (2,H,W), got {contact.shape}")
        if force.ndim != 3 or force.shape[0] != 6:
            raise ValueError(f"{force_file}: expected (6,H,W), got {force.shape}")
        if contact.shape[1:] != force.shape[1:]:
            raise ValueError(f"spatial mismatch: {contact_file} {contact.shape}, {force_file} {force.shape}")
        if spatial_shape is None:
            spatial_shape = tuple(int(value) for value in contact.shape[1:])
        elif tuple(contact.shape[1:]) != spatial_shape:
            raise ValueError(f"inconsistent spatial shape at {contact_file}: {contact.shape[1:]}")
        if not np.isfinite(contact).all() or not np.isfinite(force).all():
            raise ValueError(f"NaN/Inf in {contact_file} or {force_file}")
        if float(contact.min()) < -args.contact_negative_tolerance:
            raise ValueError(f"negative contact value in {contact_file}: min={contact.min()}")

        contact_max = np.maximum(contact_max, contact.reshape(2, -1).max(axis=1))
        force_flat = force.reshape(6, -1)
        for channel in range(6):
            values = force_flat[channel][force_flat[channel] != 0]
            if values.size:
                force_sum[channel] += values.sum()
                force_sq_sum[channel] += np.square(values).sum()
                force_count[channel] += values.size
                force_abs_max[channel] = max(force_abs_max[channel], float(np.abs(values).max()))
                force_min[channel] = min(force_min[channel], float(values.min()))
                force_max[channel] = max(force_max[channel], float(values.max()))

    if not entries:
        raise RuntimeError("no train physical frames available for statistics")
    safe_count = np.maximum(force_count, 1)
    force_mean = force_sum / safe_count
    force_variance = np.maximum(force_sq_sum / safe_count - np.square(force_mean), 0.0)
    force_std = np.sqrt(force_variance)
    force_mean[force_count == 0] = 0.0
    force_std[force_count == 0] = 1.0
    force_min[~np.isfinite(force_min)] = 0.0
    force_max[~np.isfinite(force_max)] = 0.0

    result = {
        "schema_version": "physical_statistics_v1",
        "dataset_source": args.dataset_source,
        "source_clip_index": str(args.clips),
        "split": "train",
        "seed": args.seed,
        "deduplicated": args.deduplicate,
        "num_episodes": len(episodes),
        "episode_ids": episodes,
        "num_clips": len(clips),
        "num_available_unique_frames": available_frames,
        "num_frames": len(entries),
        "spatial_shape": list(spatial_shape or ()),
        "contact_ch_max": contact_max.tolist(),
        "force_ch_active_mean": force_mean.tolist(),
        "force_ch_active_std": force_std.tolist(),
        "force_ch_active_count": force_count.tolist(),
        "force_ch_abs_max": force_abs_max.tolist(),
        "force_ch_min": force_min.tolist(),
        "force_ch_max": force_max.tolist(),
    }
    write_json(args.out, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True, help="train clips JSON only")
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset_source", default="omnivitac")
    parser.add_argument("--n_sample_frames", type=int, default=3000, help="0 uses all unique frames")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deduplicate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--contact_negative_tolerance", type=float, default=1e-8)
    return parser.parse_args()


def main() -> None:
    stats = compute_statistics(parse_args())
    print(f"wrote train-only physical statistics from {stats['num_frames']} frames")


if __name__ == "__main__":
    main()

