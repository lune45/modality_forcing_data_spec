#!/usr/bin/env python3
"""Split a VAE clip index by episode while preserving active physical clips."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data_common import load_collection, resolve_episode_path, setting_of, write_json


def normalize_fraction(value: float) -> float:
    fraction = value / 100.0 if value > 1 else value
    if not 0 < fraction <= 1:
        raise ValueError("active_frac must be in (0,1] or (0,100]")
    return fraction


def clip_has_activity(clip: dict[str, Any], source_root: Path, modality: str) -> bool:
    key = f"{modality}_paths"
    paths = clip.get(key)
    if not isinstance(paths, list) or not paths:
        raise ValueError(f"missing {key}")
    for relative_path in paths:
        path = resolve_episode_path(source_root, clip["episode"], relative_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        array = np.load(path, mmap_mode="r")
        expected_channels = 2 if modality == "contact" else 6
        if array.ndim != 3 or array.shape[0] != expected_channels:
            raise ValueError(f"{path}: invalid shape {tuple(array.shape)}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: contains NaN or Inf")
        if float(np.max(np.abs(array))) > 1e-6:
            return True
    return False


def episode_weight(items: list[tuple[dict[str, Any], bool]], use_active: bool) -> int:
    return sum(int(active) for _, active in items) if use_active else len(items)


def split_episodes(
    episode_map: dict[str, list[tuple[dict[str, Any], bool]]],
    train_weight: float,
    test_weight: float,
    rng: random.Random,
    search_trials: int,
) -> tuple[list[str], list[str]]:
    episodes = sorted(episode_map)
    if len(episodes) < 2:
        return episodes, []
    test_fraction = test_weight / (train_weight + test_weight)
    use_active = any(active for items in episode_map.values() for _, active in items)
    weights = {episode: episode_weight(episode_map[episode], use_active) for episode in episodes}
    if sum(weights.values()) == 0:
        weights = {episode: 1 for episode in episodes}
    target_weight = sum(weights.values()) * test_fraction
    target_episodes = len(episodes) * test_fraction
    best: set[str] | None = None
    best_score: tuple[float, float, tuple[str, ...]] | None = None
    for _ in range(max(search_trials, len(episodes) * 50)):
        shuffled = episodes.copy()
        rng.shuffle(shuffled)
        current_weight = 0
        for size in range(1, len(shuffled)):
            current_weight += weights[shuffled[size - 1]]
            chosen = tuple(sorted(shuffled[:size]))
            score = (abs(current_weight - target_weight), abs(size - target_episodes), chosen)
            if best_score is None or score < best_score:
                best_score = score
                best = set(chosen)
    if not best:
        best = {episodes[0]}
    train = [episode for episode in episodes if episode not in best]
    test = [episode for episode in episodes if episode in best]
    return train, test


def select_active_and_inactive(
    episodes: list[str],
    episode_map: dict[str, list[tuple[dict[str, Any], bool]]],
    active_frac: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    active = [clip for episode in episodes for clip, flag in episode_map[episode] if flag]
    inactive = [clip for episode in episodes for clip, flag in episode_map[episode] if not flag]
    rng.shuffle(inactive)
    requested = round(len(active) * (1 - active_frac) / active_frac) if active else int(bool(inactive))
    chosen_inactive = inactive[: min(requested, len(inactive))]
    selected = active + chosen_inactive
    rng.shuffle(selected)
    actual = len(active) / len(selected) if selected else 0.0
    return selected, {
        "available_active": len(active),
        "available_inactive": len(inactive),
        "selected_active": len(active),
        "selected_inactive": len(chosen_inactive),
        "requested_inactive": requested,
        "actual_active_frac": actual,
    }


def split_index(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = Path(args.source_root).resolve()
    blob, clips = load_collection(args.clips, "clips")
    config = dict(blob.get("config", {}))
    active_frac = normalize_fraction(args.active_frac)
    rng = random.Random(args.seed)

    grouped: dict[str, dict[str, list[tuple[dict[str, Any], bool]]]] = defaultdict(lambda: defaultdict(list))
    rejected: list[dict[str, Any]] = []
    for clip in clips:
        try:
            active = clip_has_activity(clip, source_root, args.modality)
        except Exception as exc:
            rejected.append({
                "stage": "split",
                "episode": clip.get("episode"),
                "obs_frame_idx": (clip.get("frame_indices") or [None])[0],
                "reason_code": "activity_check_failed",
                "detail": str(exc),
            })
            continue
        grouped[setting_of(clip["episode"])][clip["episode"]].append((clip, active))
    if rejected and args.strict:
        write_json(args.report, {"status": "fail", "rejected": rejected})
        raise RuntimeError(f"refusing to split: {len(rejected)} clips failed activity validation")

    train_clips: list[dict[str, Any]] = []
    test_clips: list[dict[str, Any]] = []
    train_episodes: set[str] = set()
    test_episodes: set[str] = set()
    setting_reports: list[dict[str, Any]] = []
    missing_test_settings: list[str] = []

    for setting in sorted(grouped):
        episode_map = grouped[setting]
        train_eps, test_eps = split_episodes(
            episode_map, args.n_train, args.n_test, rng, args.search_trials
        )
        if not test_eps:
            missing_test_settings.append(setting)
        train_selected, train_stats = select_active_and_inactive(train_eps, episode_map, active_frac, rng)
        test_selected, test_stats = select_active_and_inactive(test_eps, episode_map, active_frac, rng)
        train_clips.extend(train_selected)
        test_clips.extend(test_selected)
        train_episodes.update(train_eps)
        test_episodes.update(test_eps)
        setting_reports.append({
            "setting": setting,
            "train_episodes": len(train_eps),
            "test_episodes": len(test_eps),
            "train": train_stats,
            "test": test_stats,
        })

    overlap = train_episodes & test_episodes
    if overlap:
        raise RuntimeError(f"episode leakage: {sorted(overlap)[:10]}")
    rng.shuffle(train_clips)
    rng.shuffle(test_clips)
    split_config = {
        "method": "all_active_plus_sampled_inactive_episode_level",
        "modality": args.modality,
        "train_weight": args.n_train,
        "test_weight": args.n_test,
        "requested_train_fraction": args.n_train / (args.n_train + args.n_test),
        "requested_test_fraction": args.n_test / (args.n_train + args.n_test),
        "target_active_frac": active_frac,
        "seed": args.seed,
        "strict": args.strict,
    }
    output_config = {**config, "split": split_config}
    train_output = {"config": output_config, "clips": train_clips}
    test_output = {"config": output_config, "clips": test_clips}
    write_json(args.out_train, train_output)
    write_json(args.out_test, test_output)
    total = len(train_clips) + len(test_clips)
    report = {
        "status": "pass" if not rejected else "warning",
        "input_clips": len(clips),
        "rejected_clips": len(rejected),
        "selected_train_clips": len(train_clips),
        "selected_test_clips": len(test_clips),
        "actual_train_fraction": len(train_clips) / total if total else 0.0,
        "actual_test_fraction": len(test_clips) / total if total else 0.0,
        "train_episodes": len(train_episodes),
        "test_episodes": len(test_episodes),
        "episode_overlap": 0,
        "settings_missing_from_test": missing_test_settings,
        "settings": setting_reports,
        "rejected": rejected,
    }
    write_json(args.report, report)
    return train_output, test_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", required=True)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--modality", choices=("contact", "force"), default="contact")
    parser.add_argument("--n_train", type=float, default=1000)
    parser.add_argument("--n_test", type=float, default=200)
    parser.add_argument("--active_frac", type=float, default=0.85)
    parser.add_argument("--out_train", required=True)
    parser.add_argument("--out_test", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--search_trials", type=int, default=1000)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.n_train <= 0 or args.n_test <= 0:
        parser.error("n_train and n_test must be positive")
    return args


def main() -> None:
    train, test = split_index(parse_args())
    print(f"wrote train={len(train['clips'])} clips, test={len(test['clips'])} clips")


if __name__ == "__main__":
    main()

