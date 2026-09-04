#!/usr/bin/env python3
"""Shared helpers for the modality-forcing data preparation tools."""

from __future__ import annotations

import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


MODALITIES = ("video", "contact", "force")
TASKS = ("T0", "T1", "T2", "T3")
SCHEMA_VERSION = "modality_forcing_v1"
FRAME_NUMBER_RE = re.compile(r"(\d+)$")


def read_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: os.PathLike[str] | str, value: Any) -> None:
    """Write JSON atomically so an interrupted run does not leave half a file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, output)


def load_collection(path: os.PathLike[str] | str, key: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    blob = read_json(path)
    if isinstance(blob, list):
        return {}, blob
    if not isinstance(blob, dict) or key not in blob or not isinstance(blob[key], list):
        raise ValueError(f"{path}: expected a list or an object containing list key {key!r}")
    return blob, blob[key]


def setting_of(episode: str) -> str:
    parts = Path(episode).parts
    return parts[0] if parts else ""


def stable_base_sample_id(dataset_source: str, episode: str, obs_frame_idx: int) -> str:
    return f"{dataset_source}:{Path(episode).as_posix()}:{int(obs_frame_idx)}"


def clip_base_sample_id(clip: dict[str, Any], dataset_source: str) -> str:
    indices = clip.get("frame_indices") or []
    if not indices:
        raise ValueError("clip has no frame_indices")
    return stable_base_sample_id(dataset_source, clip["episode"], int(indices[0]))


def pipeline_base_sample_id(sample: dict[str, Any], dataset_source: str) -> str:
    if sample.get("base_sample_id"):
        return str(sample["base_sample_id"])
    return stable_base_sample_id(dataset_source, sample["episode"], int(sample["obs_frame_idx"]))


def resolve_episode_path(
    source_root: os.PathLike[str] | str,
    episode: str,
    path: os.PathLike[str] | str,
) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    episode_path = Path(episode)
    if candidate.parts[: len(episode_path.parts)] == episode_path.parts:
        return Path(source_root) / candidate
    return Path(source_root) / episode_path / candidate


def to_source_relative_path(
    source_root: os.PathLike[str] | str,
    episode: str,
    path: os.PathLike[str] | str,
) -> str:
    """Normalize a path to POSIX form relative to source_root."""
    root = Path(source_root).resolve()
    resolved = resolve_episode_path(root, episode, path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes source_root: {resolved}") from exc


def frame_number(path: os.PathLike[str] | str) -> int | None:
    stem = Path(path).stem
    match = FRAME_NUMBER_RE.search(stem)
    return int(match.group(1)) if match else None


def _metadata_entries(blob: Any) -> list[dict[str, Any]]:
    if isinstance(blob, list):
        return blob
    if isinstance(blob, dict):
        for key in ("frames", "metadata", "entries"):
            value = blob.get(key)
            if isinstance(value, list):
                return value
    raise ValueError("metadata.json must be a list or contain frames/metadata/entries")


def load_metadata(
    episode_dir: os.PathLike[str] | str,
    camera: str = "camera2",
    strict_camera: bool = True,
) -> list[dict[str, Any]]:
    entries = _metadata_entries(read_json(Path(episode_dir) / "metadata.json"))
    camera_entries = [entry for entry in entries if entry.get("camera") == camera]
    has_camera_labels = any("camera" in entry for entry in entries)
    if camera_entries:
        entries = camera_entries
    elif strict_camera and has_camera_labels:
        raise ValueError(f"requested camera {camera!r} is absent")

    entries = sorted(entries, key=lambda entry: int(entry.get("frame_idx", -1)))
    frame_indices = [int(entry["frame_idx"]) for entry in entries]
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError("duplicate frame_idx values after camera filtering")
    return entries


def get_pose(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    state = entry["eef_state"]
    translation = np.asarray([state["x"], state["y"], state["z"]], dtype=np.float64)
    rotation = np.asarray([state["r1"], state["r2"], state["r3"]], dtype=np.float64)
    gripper = float(state.get("gripper", 0.0))
    if not np.isfinite(translation).all() or not np.isfinite(rotation).all() or not math.isfinite(gripper):
        raise ValueError("non-finite eef_state value")
    return translation, rotation, gripper


def wrapped_rpy_delta_norm(first: np.ndarray, second: np.ndarray) -> float:
    delta = second - first
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return float(np.linalg.norm(delta))


def compute_valid_range(
    metadata: Sequence[dict[str, Any]],
    trans_thresh: float,
    rot_thresh: float,
) -> tuple[int, int] | None:
    if len(metadata) < 2:
        return None
    moving = np.zeros(len(metadata), dtype=bool)
    for index in range(1, len(metadata)):
        t0, r0, _ = get_pose(metadata[index - 1])
        t1, r1, _ = get_pose(metadata[index])
        moving[index] = (
            float(np.linalg.norm(t1 - t0)) > trans_thresh
            or wrapped_rpy_delta_norm(r0, r1) > rot_thresh
        )
    if not moving.any():
        return None
    first = int(np.argmax(moving))
    last = len(metadata) - 1 - int(np.argmax(moving[::-1]))
    return first, last


def image_path_for_entry(entry: dict[str, Any], frame_idx: int) -> str:
    return str(entry.get("image_path", f"images/frame_{frame_idx:06d}.png"))


def validate_array(path: Path, channels: int) -> tuple[int, int, int]:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 3 or array.shape[0] != channels:
        raise ValueError(f"expected ({channels},H,W), got {tuple(array.shape)}")
    if not np.isfinite(array).all():
        raise ValueError("array contains NaN or Inf")
    return tuple(int(value) for value in array.shape)


def validate_png_image(path: Path) -> tuple[int, int]:
    """Return (height, width) after validating the PNG signature and IHDR."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"{path}: invalid or unsupported PNG header")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"{path}: invalid image size {width}x{height}")
    return int(height), int(width)


def task_masks(task: str, n_frames: int = 17) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"unknown task: {task}")
    condition = {
        "T0": {"video": 0, "contact": 0, "force": 0},
        "T1": {"video": 1, "contact": 0, "force": 0},
        "T2": {"video": 1, "contact": 1, "force": 0},
        "T3": {"video": 0, "contact": 1, "force": 0},
    }[task]
    noise = {name: 1 - condition[name] for name in MODALITIES}
    if task == "T0":
        video_frame_mask = [0] + [1] * (n_frames - 1)
        observation_is_condition = True
    elif task in ("T1", "T2"):
        video_frame_mask = [0] * n_frames
        observation_is_condition = True
    else:
        video_frame_mask = [1] * n_frames
        observation_is_condition = False
    return {
        "action_is_condition": True,
        "observation_rgb_is_condition": observation_is_condition,
        "condition_mask": condition,
        "noise_mask": noise,
        "loss_mask": dict(noise),
        "video_noise_frame_mask": video_frame_mask,
        "video_loss_frame_mask": list(video_frame_mask),
    }


def largest_remainder_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if set(ratios) != set(TASKS):
        raise ValueError(f"ratios must have exactly these keys: {TASKS}")
    ratio_sum = sum(float(ratios[task]) for task in TASKS)
    if any(float(ratios[task]) < 0 for task in TASKS) or not math.isclose(ratio_sum, 1.0, abs_tol=1e-9):
        raise ValueError(f"ratios must be non-negative and sum to 1, got {ratio_sum}")
    raw = {task: total * float(ratios[task]) for task in TASKS}
    counts = {task: int(math.floor(raw[task])) for task in TASKS}
    remaining = total - sum(counts.values())
    ranked = sorted(TASKS, key=lambda task: (-(raw[task] - counts[task]), TASKS.index(task)))
    for task in ranked[:remaining]:
        counts[task] += 1
    return counts


def ensure_unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {label}: {duplicates[:5]}")
