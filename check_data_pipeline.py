#!/usr/bin/env python3
"""Validate VAE index, split, pipeline data, and modality-forcing manifests."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from data_common import (
    TASKS,
    clip_base_sample_id,
    frame_number,
    largest_remainder_counts,
    load_collection,
    pipeline_base_sample_id,
    read_json,
    resolve_episode_path,
    task_masks,
    validate_array,
    validate_png_image,
    write_json,
)


class Validation:
    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []

    def error(self, stage: str, code: str, detail: Any, **context: Any) -> None:
        self.errors.append({"stage": stage, "reason_code": code, "detail": str(detail), **context})

    def warning(self, stage: str, code: str, detail: Any, **context: Any) -> None:
        self.warnings.append({"stage": stage, "reason_code": code, "detail": str(detail), **context})


def validate_clip(
    clip: dict[str, Any],
    source_root: Path,
    dataset_source: str,
    check_arrays: bool,
    validation: Validation,
) -> str | None:
    episode = str(clip.get("episode", ""))
    try:
        indices = [int(value) for value in clip["frame_indices"]]
        contacts = list(clip["contact_paths"])
        forces = list(clip["force_paths"])
        if len(indices) != 17 or len(contacts) != 17 or len(forces) != 17:
            raise ValueError(f"expected 17 entries, got indices={len(indices)}, contact={len(contacts)}, force={len(forces)}")
        for index, contact, force in zip(indices, contacts, forces):
            if frame_number(contact) != index or frame_number(force) != index:
                raise ValueError(f"frame mismatch at {index}: {contact}, {force}")
            contact_file = resolve_episode_path(source_root, episode, contact)
            force_file = resolve_episode_path(source_root, episode, force)
            if not contact_file.is_file() or not force_file.is_file():
                raise FileNotFoundError(f"{contact_file} or {force_file}")
            if check_arrays:
                contact_shape = validate_array(contact_file, 2)
                force_shape = validate_array(force_file, 6)
                if contact_shape[1:] != force_shape[1:]:
                    raise ValueError(f"spatial mismatch {contact_shape}, {force_shape}")
        return clip_base_sample_id(clip, dataset_source)
    except Exception as exc:
        validation.error("vae_index", "invalid_clip", exc, episode=episode)
        return None


def validate_pipeline_sample(
    sample: dict[str, Any],
    source_root: Path,
    dataset_source: str,
    check_arrays: bool,
    validation: Validation,
) -> str | None:
    episode = str(sample.get("episode", ""))
    try:
        rgb = [sample["observation_frame"], *sample["frames"]]
        contact = [sample["observation_contact_path"], *sample["contact_path"]]
        force = [sample["observation_force_path"], *sample["force_path"]]
        actions = np.asarray(sample["actions"], dtype=np.float64)
        if len(rgb) != 17 or len(contact) != 17 or len(force) != 17:
            raise ValueError("pipeline modalities must each have 17 frames")
        if actions.shape != (16, 7) or not np.isfinite(actions).all():
            raise ValueError(f"actions must be finite 16x7, got {actions.shape}")
        expected = sample.get("frame_indices")
        if expected is None:
            expected = [frame_number(path) for path in rgb]
        expected = [int(value) for value in expected]
        if len(expected) != 17:
            raise ValueError("frame_indices must contain 17 entries")
        for index, rgb_path, contact_path, force_path in zip(expected, rgb, contact, force):
            numbers = (frame_number(rgb_path), frame_number(contact_path), frame_number(force_path))
            if numbers != (index, index, index):
                raise ValueError(f"frame mismatch expected={index}, actual={numbers}")
            for path in (rgb_path, contact_path, force_path):
                full_path = Path(path) if Path(path).is_absolute() else source_root / path
                if not full_path.is_file():
                    raise FileNotFoundError(full_path)
            if check_arrays:
                image_shape = validate_png_image(Path(rgb_path) if Path(rgb_path).is_absolute() else source_root / rgb_path)
                contact_shape = validate_array(source_root / contact_path, 2)
                force_shape = validate_array(source_root / force_path, 6)
                if contact_shape[1:] != force_shape[1:]:
                    raise ValueError("contact/force spatial mismatch")
                if image_shape != contact_shape[1:]:
                    raise ValueError(f"RGB/physical spatial mismatch: {image_shape}, {contact_shape}")
        base_id = pipeline_base_sample_id(sample, dataset_source)
        if sample.get("base_sample_id") and sample["base_sample_id"] != base_id:
            raise ValueError("unstable base_sample_id")
        return base_id
    except Exception as exc:
        validation.error("pipeline", "invalid_sample", exc, episode=episode, obs_frame_idx=sample.get("obs_frame_idx"))
        return None


def validate_task_sample(sample: dict[str, Any], expected_task: str, validation: Validation) -> None:
    try:
        if sample.get("task_type") != expected_task:
            raise ValueError(f"expected {expected_task}, got {sample.get('task_type')}")
        expected_masks = task_masks(expected_task, 17)
        for key, expected in expected_masks.items():
            if sample.get(key) != expected:
                raise ValueError(f"{key} mismatch: expected {expected}, got {sample.get(key)}")
        expected_conditions = {
            "T0": ["action", "observation_rgb"],
            "T1": ["action", "video"],
            "T2": ["action", "video", "contact"],
            "T3": ["action", "contact"],
        }[expected_task]
        if sample.get("condition_modalities") != expected_conditions:
            raise ValueError(
                f"condition_modalities mismatch: expected {expected_conditions}, "
                f"got {sample.get('condition_modalities')}"
            )
        if sample.get("sample_id") != f"{sample.get('base_sample_id')}:{expected_task}":
            raise ValueError("sample_id does not equal base_sample_id:task")
    except Exception as exc:
        validation.error(
            "mixture",
            "invalid_task_sample",
            exc,
            sample_id=sample.get("sample_id"),
            task_type=sample.get("task_type"),
        )


def validate_all(args: argparse.Namespace) -> dict[str, Any]:
    validation = Validation()
    source_root = Path(args.source_root).resolve()
    _, all_clips = load_collection(args.all_clips, "clips")
    _, train_clips = load_collection(args.train_clips, "clips")
    _, test_clips = load_collection(args.test_clips, "clips")
    train_pipeline_blob, train_pipeline = load_collection(args.train_pipeline, "samples")
    test_pipeline_blob, test_pipeline = load_collection(args.test_pipeline, "samples")

    all_ids = {value for clip in all_clips if (value := validate_clip(clip, source_root, args.dataset_source, args.check_arrays, validation))}
    train_clip_ids = {clip_base_sample_id(clip, args.dataset_source) for clip in train_clips}
    test_clip_ids = {clip_base_sample_id(clip, args.dataset_source) for clip in test_clips}
    if not train_clip_ids <= all_ids or not test_clip_ids <= all_ids:
        validation.error("split", "unknown_clip", "train/test contains a clip absent from validated all_clips")
    clip_overlap = train_clip_ids & test_clip_ids
    if clip_overlap:
        validation.error("split", "clip_overlap", sorted(clip_overlap)[:10])
    train_episodes = {str(clip["episode"]) for clip in train_clips}
    test_episodes = {str(clip["episode"]) for clip in test_clips}
    episode_overlap = train_episodes & test_episodes
    if episode_overlap:
        validation.error("split", "episode_overlap", sorted(episode_overlap)[:10])

    train_pipeline_ids = {
        value for sample in train_pipeline
        if (value := validate_pipeline_sample(sample, source_root, args.dataset_source, args.check_arrays, validation))
    }
    test_pipeline_ids = {
        value for sample in test_pipeline
        if (value := validate_pipeline_sample(sample, source_root, args.dataset_source, args.check_arrays, validation))
    }
    if train_pipeline_ids != train_clip_ids:
        validation.error("pipeline", "train_mapping_mismatch", f"missing={len(train_clip_ids-train_pipeline_ids)}, extra={len(train_pipeline_ids-train_clip_ids)}")
    if test_pipeline_ids != test_clip_ids:
        validation.error("pipeline", "test_mapping_mismatch", f"missing={len(test_clip_ids-test_pipeline_ids)}, extra={len(test_pipeline_ids-test_clip_ids)}")
    if train_pipeline_blob.get("split") not in (None, "train"):
        validation.error("pipeline", "wrong_split", "train pipeline is not labelled train")
    if test_pipeline_blob.get("split") not in (None, "test"):
        validation.error("pipeline", "wrong_split", "test pipeline is not labelled test")

    physical_stats = read_json(args.physical_stats)
    if physical_stats.get("split") != "train":
        validation.error("statistics", "wrong_split", "physical statistics is not labelled train")
    stat_episodes = set(physical_stats.get("episode_ids") or [])
    if not stat_episodes <= train_episodes:
        validation.error("statistics", "non_train_episode", sorted(stat_episodes - train_episodes)[:10])
    if stat_episodes & test_episodes:
        validation.error("statistics", "test_leakage", sorted(stat_episodes & test_episodes)[:10])
    for key, length in (("contact_ch_max", 2), ("force_ch_active_std", 6)):
        values = np.asarray(physical_stats.get(key, []), dtype=np.float64)
        if values.shape != (length,) or not np.isfinite(values).all():
            validation.error("statistics", "invalid_values", f"{key}: {values}")

    action_stats = read_json(args.action_stats)
    if action_stats.get("split") != "train":
        validation.error("statistics", "wrong_action_split", "action statistics is not labelled train")
    action_stat_episodes = set(action_stats.get("episode_ids") or [])
    if action_stat_episodes & test_episodes:
        validation.error("statistics", "action_test_leakage", sorted(action_stat_episodes & test_episodes)[:10])

    mixture_dir = Path(args.mixture_dir)
    train_mixture_blob, train_mixture = load_collection(mixture_dir / "train_mixture.json", "samples")
    mixture_ids = [str(sample.get("base_sample_id")) for sample in train_mixture]
    if len(mixture_ids) != len(set(mixture_ids)):
        validation.error("mixture", "duplicate_train_base_id", "a train base sample appears more than once")
    if set(mixture_ids) != train_pipeline_ids:
        validation.error("mixture", "train_mapping_mismatch", f"mixture={len(set(mixture_ids))}, pipeline={len(train_pipeline_ids)}")
    counts = Counter(str(sample.get("task_type")) for sample in train_mixture)
    ratios = train_mixture_blob.get("task_ratios") or {"T0": 0.4, "T1": 0.2, "T2": 0.2, "T3": 0.2}
    expected_counts = largest_remainder_counts(len(train_mixture), ratios)
    if {task: counts[task] for task in TASKS} != expected_counts:
        validation.error("mixture", "wrong_ratio", f"expected={expected_counts}, actual={dict(counts)}")
    for sample in train_mixture:
        validate_task_sample(sample, str(sample.get("task_type")), validation)

    expected_test_order = [pipeline_base_sample_id(sample, args.dataset_source) for sample in test_pipeline]
    for task in TASKS:
        _, samples = load_collection(mixture_dir / f"test_{task.lower()}.json", "samples")
        ids = [str(sample.get("base_sample_id")) for sample in samples]
        if ids != expected_test_order:
            validation.error("mixture", "test_alignment", f"{task} base_sample_id order differs from test pipeline")
        for sample in samples:
            validate_task_sample(sample, task, validation)

    report = {
        "status": "pass" if not validation.errors else "fail",
        "vae_index": {"clips": len(all_clips), "valid_clip_ids": len(all_ids)},
        "split": {
            "train_clips": len(train_clips),
            "test_clips": len(test_clips),
            "train_episodes": len(train_episodes),
            "test_episodes": len(test_episodes),
            "episode_overlap": len(episode_overlap),
        },
        "pipeline": {"train_samples": len(train_pipeline), "test_samples": len(test_pipeline)},
        "mixture": {
            "train_samples": len(train_mixture),
            "task_counts": {task: counts[task] for task in TASKS},
            "test_base_samples": len(expected_test_order),
        },
        "errors": validation.errors,
        "warnings": validation.warnings,
    }
    write_json(args.report, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--all_clips", required=True)
    parser.add_argument("--train_clips", required=True)
    parser.add_argument("--test_clips", required=True)
    parser.add_argument("--physical_stats", required=True)
    parser.add_argument("--action_stats", required=True)
    parser.add_argument("--train_pipeline", required=True)
    parser.add_argument("--test_pipeline", required=True)
    parser.add_argument("--mixture_dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--dataset_source", default="omnivitac")
    parser.add_argument("--check_arrays", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    report = validate_all(parse_args())
    print(f"validation {report['status']}: {len(report['errors'])} errors, {len(report['warnings'])} warnings")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
