#!/usr/bin/env python3
"""Run the complete VAE-index -> split -> pipeline -> mixture workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from data_common import read_json


def resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def boolean_flag(enabled: bool, positive: str, negative: str) -> str:
    return positive if enabled else negative


def run_command(label: str, command: list[str], dry_run: bool) -> None:
    print(f"\n[{label}] {' '.join(command)}")
    if not dry_run:
        subprocess.run(command, check=True)


def build_commands(config_path: Path, config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    source_root = resolve_from_config(config_path, config["source_root"])
    output_root = resolve_from_config(config_path, config.get("output_dir", "../outputs"))
    vae_dir = output_root / "vae_index"
    split_dir = output_root / "split"
    pipeline_train_dir = output_root / "pipeline" / "train"
    pipeline_test_dir = output_root / "pipeline" / "test"
    mixture_dir = output_root / "mixture"
    reports_dir = output_root / "reports"
    vae = config["vae_index"]
    split = config["split"]
    physical = config["physical_statistics"]
    pipeline = config["pipeline"]
    mixture = config["mixture"]
    validate_arrays = bool(config.get("check_array_content", True))

    all_clips = vae_dir / "clips.json"
    train_clips = split_dir / "clips_train.json"
    test_clips = split_dir / "clips_test.json"
    physical_stats = vae_dir / "physical_statistics.json"
    train_pipeline = pipeline_train_dir / "data.json"
    test_pipeline = pipeline_test_dir / "data.json"
    action_stats = pipeline_train_dir / "statistics.json"

    return [
        ("1/7 build VAE clip index", [
            python, str(script_dir / "build_vae_index.py"),
            "--source_root", str(source_root),
            "--out_dir", str(vae_dir),
            "--camera", str(vae.get("camera", "camera2")),
            "--frame_stride", str(vae.get("frame_stride", 3)),
            "--n_frames", str(vae.get("n_frames", 17)),
            "--clips_per_episode", str(vae.get("clips_per_episode", 25)),
            "--trans_thresh", str(vae.get("trans_thresh_mm", 1.0)),
            "--rot_thresh", str(vae.get("rot_thresh_rad", 0.01)),
            boolean_flag(bool(vae.get("strict_camera", True)), "--strict_camera", "--no-strict_camera"),
            boolean_flag(validate_arrays, "--validate_arrays", "--no-validate_arrays"),
        ]),
        ("2/7 split train/test", [
            python, str(script_dir / "split_train_test.py"),
            "--clips", str(all_clips),
            "--source_root", str(source_root),
            "--modality", str(split.get("activity_modality", "contact")),
            "--n_train", str(split.get("train_weight", 1000)),
            "--n_test", str(split.get("test_weight", 200)),
            "--active_frac", str(split.get("active_frac", 0.85)),
            "--out_train", str(train_clips),
            "--out_test", str(test_clips),
            "--report", str(reports_dir / "split_report.json"),
            "--seed", str(split.get("seed", 0)),
            "--strict",
        ]),
        ("3/7 compute train-only physical statistics", [
            python, str(script_dir / "compute_physical_statistics.py"),
            "--clips", str(train_clips),
            "--source_root", str(source_root),
            "--out", str(physical_stats),
            "--dataset_source", str(config.get("dataset_source", "omnivitac")),
            "--n_sample_frames", str(physical.get("n_sample_frames", 3000)),
            "--seed", str(physical.get("seed", 0)),
            boolean_flag(bool(physical.get("deduplicate", True)), "--deduplicate", "--no-deduplicate"),
        ]),
        ("4/7 build train pipeline", [
            python, str(script_dir / "build_pipeline_data.py"),
            "--source_root", str(source_root),
            "--clip_index", str(train_clips),
            "--out_dir", str(pipeline_train_dir),
            "--dataset_source", str(config.get("dataset_source", "omnivitac")),
            "--split", "train",
            "--camera", str(pipeline.get("camera", "camera2")),
            "--action_stride", str(pipeline.get("action_stride", 3)),
            "--n_frames", str(vae.get("n_frames", 17)),
            "--trans_thresh", str(vae.get("trans_thresh_mm", 1.0)),
            "--rot_thresh", str(vae.get("rot_thresh_rad", 0.01)),
            "--strict", "--compute_action_stats",
        ]),
        ("5/7 build test pipeline", [
            python, str(script_dir / "build_pipeline_data.py"),
            "--source_root", str(source_root),
            "--clip_index", str(test_clips),
            "--out_dir", str(pipeline_test_dir),
            "--dataset_source", str(config.get("dataset_source", "omnivitac")),
            "--split", "test",
            "--camera", str(pipeline.get("camera", "camera2")),
            "--action_stride", str(pipeline.get("action_stride", 3)),
            "--n_frames", str(vae.get("n_frames", 17)),
            "--trans_thresh", str(vae.get("trans_thresh_mm", 1.0)),
            "--rot_thresh", str(vae.get("rot_thresh_rad", 0.01)),
            "--strict", "--skip_stats",
        ]),
        ("6/7 build modality mixture", [
            python, str(script_dir / "build_modality_mixture.py"),
            "--train_input", str(train_pipeline),
            "--test_input", str(test_pipeline),
            "--out_dir", str(mixture_dir),
            "--dataset_source", str(config.get("dataset_source", "omnivitac")),
            "--seed", str(mixture.get("random_seed", 42)),
            "--t0", str(mixture["task_ratios"]["T0"]),
            "--t1", str(mixture["task_ratios"]["T1"]),
            "--t2", str(mixture["task_ratios"]["T2"]),
            "--t3", str(mixture["task_ratios"]["T3"]),
        ]),
        ("7/7 validate complete workflow", [
            python, str(script_dir / "check_data_pipeline.py"),
            "--source_root", str(source_root),
            "--all_clips", str(all_clips),
            "--train_clips", str(train_clips),
            "--test_clips", str(test_clips),
            "--physical_stats", str(physical_stats),
            "--action_stats", str(action_stats),
            "--train_pipeline", str(train_pipeline),
            "--test_pipeline", str(test_pipeline),
            "--mixture_dir", str(mixture_dir),
            "--report", str(reports_dir / "validation_report.json"),
            "--dataset_source", str(config.get("dataset_source", "omnivitac")),
            boolean_flag(validate_arrays, "--check_arrays", "--no-check_arrays"),
        ]),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    if config.get("vae_index", {}).get("with_stats_before_split"):
        raise ValueError("with_stats_before_split must be false to avoid test leakage")
    ratios = config.get("mixture", {}).get("task_ratios", {})
    if ratios != {"T0": 0.4, "T1": 0.2, "T2": 0.2, "T3": 0.2}:
        raise ValueError("this project specification requires T0/T1/T2/T3 = 0.4/0.2/0.2/0.2")
    for label, command in build_commands(config_path, config):
        run_command(label, command, args.dry_run)
    if args.dry_run:
        print("\ndry run complete; no data was written")
    else:
        print("\ncomplete data workflow passed")


if __name__ == "__main__":
    main()
