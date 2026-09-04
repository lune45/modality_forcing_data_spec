#!/usr/bin/env python3
"""Attach RGB, explicit physical paths, and 7D actions to a split clip index."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from data_common import (
    compute_valid_range,
    image_path_for_entry,
    load_collection,
    load_metadata,
    resolve_episode_path,
    stable_base_sample_id,
    to_source_relative_path,
    validate_array,
    write_json,
)


ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_abs"]


def get_pose(entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    state = entry["eef_state"]
    translation = np.asarray([state["x"], state["y"], state["z"]], dtype=np.float64)
    rpy = np.asarray([state["r1"], state["r2"], state["r3"]], dtype=np.float64)
    gripper = float(state.get("gripper", 0.0))
    if not np.isfinite(translation).all() or not np.isfinite(rpy).all() or not np.isfinite(gripper):
        raise ValueError("non-finite pose or gripper")
    return translation, rpy, gripper


def action_delta(entry_from: dict[str, Any], entry_to: dict[str, Any]) -> np.ndarray:
    translation_from, rotation_from, _ = get_pose(entry_from)
    translation_to, rotation_to, gripper_to = get_pose(entry_to)
    delta_translation = translation_to - translation_from
    relative_rotation = (
        Rotation.from_euler("xyz", rotation_from).inv()
        * Rotation.from_euler("xyz", rotation_to)
    ).as_euler("xyz")
    return np.concatenate([delta_translation, relative_rotation, [gripper_to]]).astype(np.float64)


class MetadataCache:
    def __init__(self, source_root: Path, camera: str, strict_camera: bool):
        self.source_root = source_root
        self.camera = camera
        self.strict_camera = strict_camera
        self.cache: dict[str, tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]] = {}

    def get(self, episode: str) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        if episode not in self.cache:
            metadata = load_metadata(self.source_root / episode, self.camera, self.strict_camera)
            self.cache[episode] = (metadata, {int(entry["frame_idx"]): entry for entry in metadata})
        return self.cache[episode]


def compute_action_statistics(
    episodes: list[str],
    cache: MetadataCache,
    action_stride: int,
    trans_thresh: float,
    rot_thresh: float,
    source_clip_index: str,
) -> dict[str, Any]:
    actions: list[np.ndarray] = []
    used_episodes: list[str] = []
    for episode in episodes:
        metadata, _ = cache.get(episode)
        valid_range = compute_valid_range(metadata, trans_thresh, rot_thresh)
        if valid_range is None:
            continue
        first, last = valid_range
        episode_actions = [
            action_delta(metadata[position], metadata[position + action_stride])
            for position in range(first, last - action_stride + 1)
        ]
        if episode_actions:
            actions.extend(episode_actions)
            used_episodes.append(episode)
    if not actions:
        raise RuntimeError("no actions available for statistics")
    array = np.stack(actions)
    return {
        "schema_version": "action_statistics_v1",
        "source_clip_index": source_clip_index,
        "split": "train",
        "dim_names": ACTION_NAMES,
        "n_samples": int(array.shape[0]),
        "n_episodes_used": len(used_episodes),
        "episode_ids": used_episodes,
        "action_stride": action_stride,
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "p1": np.percentile(array, 1, axis=0).tolist(),
        "p99": np.percentile(array, 99, axis=0).tolist(),
        "normalization_note": (
            "Use train statistics for every split. Translation is millimetres; rotation is "
            "relative SO(3) Euler xyz in radians. gripper_abs is absolute and must not be "
            "blindly z-scored with the first six dimensions."
        ),
    }


def build_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob, clips = load_collection(args.clip_index, "clips")
    clip_config = blob.get("config", {})
    frame_stride = clip_config.get("frame_stride")
    action_stride = args.action_stride if args.action_stride is not None else frame_stride
    if action_stride is None or int(action_stride) <= 0:
        raise ValueError("action_stride is absent or invalid")
    action_stride = int(action_stride)
    cache = MetadataCache(source_root, args.camera, args.strict_camera)
    samples: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for clip in clips:
        episode = str(clip.get("episode", ""))
        indices = [int(value) for value in (clip.get("frame_indices") or [])]
        obs_frame_idx = indices[0] if indices else None
        try:
            if len(indices) != args.n_frames:
                raise ValueError(f"expected {args.n_frames} frame_indices, got {len(indices)}")
            if any(second - first != action_stride for first, second in zip(indices, indices[1:])):
                raise ValueError(f"frame spacing does not match action_stride={action_stride}")
            contact_paths = list(clip.get("contact_paths") or [])
            force_paths = list(clip.get("force_paths") or [])
            if len(contact_paths) != args.n_frames or len(force_paths) != args.n_frames:
                raise ValueError("contact_paths and force_paths must both have n_frames entries")

            _, frame_map = cache.get(episode)
            entries = [frame_map.get(index) for index in indices]
            if any(entry is None for entry in entries):
                missing = [index for index, entry in zip(indices, entries) if entry is None]
                raise ValueError(f"frame_idx absent from metadata: {missing}")
            typed_entries = [entry for entry in entries if entry is not None]
            images = [
                to_source_relative_path(source_root, episode, image_path_for_entry(entry, index))
                for entry, index in zip(typed_entries, indices)
            ]
            contacts = [to_source_relative_path(source_root, episode, path) for path in contact_paths]
            forces = [to_source_relative_path(source_root, episode, path) for path in force_paths]
            for path in [*images, *contacts, *forces]:
                if not (source_root / path).is_file():
                    raise FileNotFoundError(source_root / path)
            if args.validate_arrays:
                for contact, force in zip(contacts, forces):
                    contact_shape = validate_array(source_root / contact, 2)
                    force_shape = validate_array(source_root / force, 6)
                    if contact_shape[1:] != force_shape[1:]:
                        raise ValueError(f"physical spatial mismatch: {contact_shape}, {force_shape}")

            actions = np.stack([
                action_delta(typed_entries[index], typed_entries[index + 1])
                for index in range(args.n_frames - 1)
            ])
            if actions.shape != (args.n_frames - 1, 7) or not np.isfinite(actions).all():
                raise ValueError(f"invalid action tensor {actions.shape}")
            base_id = stable_base_sample_id(args.dataset_source, episode, indices[0])
            samples.append({
                "base_sample_id": base_id,
                "dataset_source": args.dataset_source,
                "episode": episode,
                "obs_frame_idx": indices[0],
                "frame_indices": indices,
                "observation_frame": images[0],
                "frames": images[1:],
                "observation_contact_path": contacts[0],
                "contact_path": contacts[1:],
                "observation_force_path": forces[0],
                "force_path": forces[1:],
                "actions": actions.tolist(),
            })
        except Exception as exc:
            rejected.append({
                "stage": "pipeline",
                "episode": episode,
                "obs_frame_idx": obs_frame_idx,
                "reason_code": "invalid_pipeline_sample",
                "detail": str(exc),
            })

    write_json(out_dir / "rejected_pipeline.json", {"n_rejected": len(rejected), "items": rejected})
    if rejected and args.strict:
        raise RuntimeError(f"refusing partial pipeline output: {len(rejected)} clips were rejected")

    output = {
        "schema_version": "pipeline_multimodal_v1",
        "source": str(args.clip_index),
        "dataset_source": args.dataset_source,
        "split": args.split,
        "config": {
            "camera": args.camera,
            "strict_camera": args.strict_camera,
            "action_stride": action_stride,
            "n_frames": args.n_frames,
            "n_future": args.n_frames - 1,
            "action_dim_names": ACTION_NAMES,
            "trans_thresh": args.trans_thresh,
            "rot_thresh": args.rot_thresh,
            "path_root": "all media paths are relative to source_root",
            "clip_index_config": clip_config,
        },
        "n_samples": len(samples),
        "samples": samples,
    }
    write_json(out_dir / "data.json", output)

    if args.compute_action_stats:
        if args.split != "train":
            raise ValueError("action statistics may only be computed for split=train")
        episodes = sorted({sample["episode"] for sample in samples})
        stats = compute_action_statistics(
            episodes,
            cache,
            action_stride,
            args.trans_thresh,
            args.rot_thresh,
            str(args.clip_index),
        )
        write_json(out_dir / "statistics.json", stats)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--clip_index", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_source", default="omnivitac")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--camera", default="camera2")
    parser.add_argument("--action_stride", type=int, default=None)
    parser.add_argument("--n_frames", type=int, default=17)
    parser.add_argument("--trans_thresh", type=float, default=1.0)
    parser.add_argument("--rot_thresh", type=float, default=0.01)
    parser.add_argument("--strict_camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate_arrays", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compute_action_stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_stats", dest="compute_action_stats", action="store_false")
    return parser.parse_args()


def main() -> None:
    result = build_pipeline(parse_args())
    print(f"wrote {result['n_samples']} {result['split']} pipeline samples")


if __name__ == "__main__":
    main()

