#!/usr/bin/env python3
"""Build the validated 17-frame clip index used by physical VAEs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from data_common import (
    compute_valid_range,
    image_path_for_entry,
    load_metadata,
    resolve_episode_path,
    validate_array,
    validate_png_image,
    write_json,
)


FLAG_NAME = "double_checked.flag"


def is_episode(path: Path) -> bool:
    return (
        (path / "metadata.json").is_file()
        and (path / "masks.json").is_file()
        and (path / FLAG_NAME).is_file()
        and (path / "modalities" / "contact").is_dir()
        and (path / "modalities" / "force").is_dir()
    )


def sample_clip_starts(
    first_pos: int,
    last_pos: int,
    frame_stride: int,
    n_frames: int,
    clips_per_episode: int,
) -> list[int]:
    span = (n_frames - 1) * frame_stride
    max_start = last_pos - span
    if max_start < first_pos:
        return []
    candidates = list(range(first_pos, max_start + 1))
    target_count = min(clips_per_episode, len(candidates))
    selected: list[int] = []
    used_frames: set[int] = set()
    while candidates and len(selected) < target_count:
        best_start = candidates[0]
        best_score = (-1, -1.0)
        for start in candidates:
            positions = {start + offset * frame_stride for offset in range(n_frames)}
            gain = len(positions - used_frames)
            spacing = min((abs(start - old) for old in selected), default=float("inf"))
            score = (gain, spacing)
            if score > best_score:
                best_start = start
                best_score = score
        selected.append(best_start)
        used_frames.update(best_start + offset * frame_stride for offset in range(n_frames))
        candidates.remove(best_start)
    return sorted(selected)


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    directories = [source_root, *(path for path in source_root.rglob("*") if path.is_dir())]
    episodes = sorted(path for path in directories if is_episode(path))
    clips: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    accepted_episodes = 0

    for episode_dir in episodes:
        episode = episode_dir.relative_to(source_root).as_posix()
        try:
            metadata = load_metadata(episode_dir, args.camera, args.strict_camera)
        except Exception as exc:  # rejection report is more useful than a partial crash
            rejected.append({"stage": "vae_index", "episode": episode, "reason_code": "metadata", "detail": str(exc)})
            continue
        if len(metadata) < args.n_frames:
            rejected.append({"stage": "vae_index", "episode": episode, "reason_code": "too_short", "detail": f"metadata entries={len(metadata)}"})
            continue
        try:
            valid_range = compute_valid_range(metadata, args.trans_thresh, args.rot_thresh)
        except Exception as exc:
            rejected.append({"stage": "vae_index", "episode": episode, "reason_code": "invalid_pose", "detail": str(exc)})
            continue
        if valid_range is None:
            rejected.append({"stage": "vae_index", "episode": episode, "reason_code": "no_motion", "detail": "no frame exceeds movement threshold"})
            continue

        starts = sample_clip_starts(
            valid_range[0],
            valid_range[1],
            args.frame_stride,
            args.n_frames,
            args.clips_per_episode,
        )
        if not starts:
            rejected.append({"stage": "vae_index", "episode": episode, "reason_code": "moving_range_too_short", "detail": f"valid_range={valid_range}"})
            continue

        episode_clip_count = 0
        for start in starts:
            positions = [start + index * args.frame_stride for index in range(args.n_frames)]
            entries = [metadata[position] for position in positions]
            frame_indices = [int(entry["frame_idx"]) for entry in entries]
            image_paths = [image_path_for_entry(entry, frame_idx) for entry, frame_idx in zip(entries, frame_indices)]
            contact_paths = [f"modalities/contact/{frame_idx:06d}.npy" for frame_idx in frame_indices]
            force_paths = [f"modalities/force/{frame_idx:06d}.npy" for frame_idx in frame_indices]

            missing = []
            for kind, paths in (("image", image_paths), ("contact", contact_paths), ("force", force_paths)):
                for path in paths:
                    if not resolve_episode_path(source_root, episode, path).is_file():
                        missing.append(f"{kind}:{path}")
            if missing:
                rejected.append({
                    "stage": "vae_index",
                    "episode": episode,
                    "obs_frame_idx": frame_indices[0],
                    "reason_code": "missing_file",
                    "detail": missing[:10],
                })
                continue

            if args.validate_arrays:
                try:
                    image_shapes = {
                        validate_png_image(resolve_episode_path(source_root, episode, path))
                        for path in image_paths
                    }
                    contact_shapes = {
                        validate_array(resolve_episode_path(source_root, episode, path), 2)
                        for path in contact_paths
                    }
                    force_shapes = {
                        validate_array(resolve_episode_path(source_root, episode, path), 6)
                        for path in force_paths
                    }
                    if len(image_shapes) != 1 or len(contact_shapes) != 1 or len(force_shapes) != 1:
                        raise ValueError(
                            f"inconsistent shapes: image={image_shapes}, contact={contact_shapes}, force={force_shapes}"
                        )
                    image_shape = next(iter(image_shapes))
                    contact_shape = next(iter(contact_shapes))
                    force_shape = next(iter(force_shapes))
                    if contact_shape[1:] != force_shape[1:]:
                        raise ValueError(f"spatial mismatch: contact={contact_shape}, force={force_shape}")
                    if image_shape != contact_shape[1:]:
                        raise ValueError(f"RGB/physical spatial mismatch: image={image_shape}, contact={contact_shape}")
                except Exception as exc:
                    rejected.append({
                        "stage": "vae_index",
                        "episode": episode,
                        "obs_frame_idx": frame_indices[0],
                        "reason_code": "invalid_array",
                        "detail": str(exc),
                    })
                    continue

            clips.append({
                "episode": episode,
                "frame_indices": frame_indices,
                "contact_paths": contact_paths,
                "force_paths": force_paths,
            })
            episode_clip_count += 1
        accepted_episodes += int(episode_clip_count > 0)

    output = {
        "config": {
            "frame_stride": args.frame_stride,
            "n_frames": args.n_frames,
            "clips_per_episode": args.clips_per_episode,
            "trans_thresh": args.trans_thresh,
            "rot_thresh": args.rot_thresh,
            "camera": args.camera,
            "strict_camera": args.strict_camera,
            "validate_arrays": args.validate_arrays,
            "n_discovered_episodes": len(episodes),
            "n_episodes": accepted_episodes,
            "n_clips": len(clips),
            "n_rejected": len(rejected),
        },
        "clips": clips,
    }
    write_json(out_dir / "clips.json", output)
    write_json(out_dir / "rejected_vae_index.json", {"n_rejected": len(rejected), "items": rejected})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--out_dir", default="./outputs/vae_index")
    parser.add_argument("--camera", default="camera2")
    parser.add_argument("--frame_stride", type=int, default=3)
    parser.add_argument("--n_frames", type=int, default=17)
    parser.add_argument("--clips_per_episode", type=int, default=25)
    parser.add_argument("--trans_thresh", type=float, default=1.0)
    parser.add_argument("--rot_thresh", type=float, default=0.01)
    parser.add_argument("--strict_camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate_arrays", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.frame_stride <= 0 or args.n_frames < 2 or args.clips_per_episode <= 0:
        parser.error("frame_stride and clips_per_episode must be positive; n_frames must be >= 2")
    return args


def main() -> None:
    result = build_index(parse_args())
    print(f"wrote {result['config']['n_clips']} clips from {result['config']['n_episodes']} episodes")


if __name__ == "__main__":
    main()
