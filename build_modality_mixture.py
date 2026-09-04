#!/usr/bin/env python3
"""Create the 40/20/20/20 T0-T3 train mixture and four aligned test views."""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data_common import (
    MODALITIES,
    SCHEMA_VERSION,
    TASKS,
    ensure_unique,
    largest_remainder_counts,
    load_collection,
    pipeline_base_sample_id,
    setting_of,
    task_masks,
    write_json,
)


def validate_pipeline_sample(sample: dict[str, Any], dataset_source: str) -> str:
    required = (
        "episode",
        "obs_frame_idx",
        "observation_frame",
        "frames",
        "observation_contact_path",
        "contact_path",
        "observation_force_path",
        "force_path",
        "actions",
    )
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"sample missing fields: {missing}")
    if len(sample["frames"]) != 16 or len(sample["contact_path"]) != 16 or len(sample["force_path"]) != 16:
        raise ValueError("future RGB/contact/force lists must each contain 16 items")
    if len(sample["actions"]) != 16 or any(len(action) != 7 for action in sample["actions"]):
        raise ValueError("actions must have shape 16x7")
    return pipeline_base_sample_id(sample, dataset_source)


def assign_tasks_stratified(
    samples: list[dict[str, Any]],
    ratios: dict[str, float],
    seed: int,
) -> list[tuple[dict[str, Any], str]]:
    """Keep exact global counts while making each setting locally close to the ratios."""
    global_targets = largest_remainder_counts(len(samples), ratios)
    remaining = dict(global_targets)
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[setting_of(str(sample["episode"]))].append(sample)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    assignments: list[tuple[dict[str, Any], str]] = []

    for group_name in group_names:
        group = sorted(groups[group_name], key=lambda item: str(item["base_sample_id"]))
        rng.shuffle(group)
        local_counts = Counter()
        for local_index, sample in enumerate(group, start=1):
            candidates = [task for task in TASKS if remaining[task] > 0]
            if not candidates:
                raise RuntimeError("task quotas were exhausted before all samples were assigned")

            def score(task: str) -> tuple[float, float, int]:
                local_deficit = ratios[task] * local_index - local_counts[task]
                global_urgency = remaining[task] / max(global_targets[task], 1)
                return local_deficit, global_urgency, -TASKS.index(task)

            chosen = max(candidates, key=score)
            local_counts[chosen] += 1
            remaining[chosen] -= 1
            assignments.append((sample, chosen))

    if any(remaining.values()):
        raise RuntimeError(f"unfilled task quotas: {remaining}")
    rng.shuffle(assignments)
    return assignments


def decorate_sample(sample: dict[str, Any], task: str, split: str, dataset_source: str) -> dict[str, Any]:
    output = dict(sample)
    base_id = pipeline_base_sample_id(sample, dataset_source)
    output.update({
        "sample_id": f"{base_id}:{task}",
        "base_sample_id": base_id,
        "dataset_source": dataset_source,
        "split": split,
        "task_type": task,
        "available_modalities": {"video": 1, "contact": 1, "force": 1, "action": 1},
        **task_masks(task, n_frames=17),
    })
    output["condition_modalities"] = ["action", *[
        modality for modality in MODALITIES if output["condition_mask"][modality] == 1
    ]]
    if output["observation_rgb_is_condition"] and "video" not in output["condition_modalities"]:
        output["condition_modalities"].append("observation_rgb")
    output["target_modalities"] = [
        modality for modality in MODALITIES if output["loss_mask"][modality] == 1
    ]
    return output


def build_mixture(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_blob, train_samples = load_collection(args.train_input, "samples")
    test_blob, test_samples = load_collection(args.test_input, "samples")
    dataset_source = args.dataset_source
    for split_name, samples in (("train", train_samples), ("test", test_samples)):
        ids = []
        for sample in samples:
            try:
                sample["base_sample_id"] = validate_pipeline_sample(sample, dataset_source)
                ids.append(sample["base_sample_id"])
            except Exception as exc:
                raise ValueError(f"invalid {split_name} pipeline sample {sample.get('episode')}: {exc}") from exc
        ensure_unique(ids, f"{split_name} base_sample_id")

    train_ids = {sample["base_sample_id"] for sample in train_samples}
    test_ids = {sample["base_sample_id"] for sample in test_samples}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"train/test base_sample_id overlap: {sorted(overlap)[:5]}")
    train_episodes = {str(sample["episode"]) for sample in train_samples}
    test_episodes = {str(sample["episode"]) for sample in test_samples}
    episode_overlap = train_episodes & test_episodes
    if episode_overlap:
        raise ValueError(f"train/test episode overlap: {sorted(episode_overlap)[:5]}")

    ratios = {"T0": args.t0, "T1": args.t1, "T2": args.t2, "T3": args.t3}
    assignments = assign_tasks_stratified(train_samples, ratios, args.seed)
    train_mixture = [decorate_sample(sample, task, "train", dataset_source) for sample, task in assignments]
    train_counts = Counter(sample["task_type"] for sample in train_mixture)
    train_output = {
        "schema_version": SCHEMA_VERSION,
        "source": str(args.train_input),
        "dataset_source": dataset_source,
        "split": "train",
        "random_seed": args.seed,
        "task_ratios": ratios,
        "n_samples": len(train_mixture),
        "samples": train_mixture,
    }
    write_json(out_dir / "train_mixture.json", train_output)

    test_outputs: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        samples = [decorate_sample(sample, task, "test", dataset_source) for sample in test_samples]
        output = {
            "schema_version": SCHEMA_VERSION,
            "source": str(args.test_input),
            "dataset_source": dataset_source,
            "split": "test",
            "task_type": task,
            "n_samples": len(samples),
            "samples": samples,
        }
        write_json(out_dir / f"test_{task.lower()}.json", output)
        test_outputs[task] = output

    by_setting: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in train_mixture:
        by_setting[setting_of(sample["episode"])][sample["task_type"]] += 1
    statistics = {
        "schema_version": SCHEMA_VERSION,
        "random_seed": args.seed,
        "train_input": str(args.train_input),
        "test_input": str(args.test_input),
        "total_samples": len(train_mixture),
        "test_base_samples": len(test_samples),
        "task_counts": {task: train_counts[task] for task in TASKS},
        "task_ratios_requested": ratios,
        "task_ratios_actual": {
            task: train_counts[task] / len(train_mixture) if train_mixture else 0.0 for task in TASKS
        },
        "episode_count": len(train_episodes),
        "train_test_episode_overlap": 0,
        "by_setting": {
            setting: {task: counts[task] for task in TASKS}
            for setting, counts in sorted(by_setting.items())
        },
        "upstream": {
            "train_pipeline_schema": train_blob.get("schema_version"),
            "test_pipeline_schema": test_blob.get("schema_version"),
        },
    }
    write_json(out_dir / "mixture_statistics.json", statistics)
    return {"train": train_output, "tests": test_outputs, "statistics": statistics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_input", required=True)
    parser.add_argument("--test_input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_source", default="omnivitac")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0", type=float, default=0.4)
    parser.add_argument("--t1", type=float, default=0.2)
    parser.add_argument("--t2", type=float, default=0.2)
    parser.add_argument("--t3", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    result = build_mixture(parse_args())
    print(f"wrote {result['train']['n_samples']} train samples with {result['statistics']['task_counts']}")


if __name__ == "__main__":
    main()
