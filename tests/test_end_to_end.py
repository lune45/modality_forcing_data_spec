from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1]


class EndToEndDataPipelineTest(unittest.TestCase):
    def make_episode(self, root: Path, setting_index: int, episode_index: int) -> None:
        episode = root / f"setting_{setting_index}" / "task" / f"{episode_index:04d}"
        image_dir = episode / "images"
        contact_dir = episode / "modalities" / "contact"
        force_dir = episode / "modalities" / "force"
        image_dir.mkdir(parents=True)
        contact_dir.mkdir(parents=True)
        force_dir.mkdir(parents=True)
        (episode / "masks.json").write_text("{}\n", encoding="utf-8")
        (episode / "double_checked.flag").write_text("ok\n", encoding="utf-8")

        metadata = []
        for frame_idx in range(20):
            metadata.append({
                "camera": "camera2",
                "frame_idx": frame_idx,
                "image_path": f"images/frame_{frame_idx:06d}.png",
                "eef_state": {
                    "x": float(frame_idx * 2),
                    "y": float(setting_index),
                    "z": float(episode_index),
                    "r1": 0.0,
                    "r2": 0.0,
                    "r3": frame_idx * 0.002,
                    "gripper": float(episode_index % 2),
                },
            })
            # PNG signature + IHDR prefix for a 5x4 image. The data tools only
            # inspect the header because they do not decode or transform RGB.
            (image_dir / f"frame_{frame_idx:06d}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00\x00\x00\x05\x00\x00\x00\x04"
            )
            contact = np.zeros((2, 4, 5), dtype=np.float32)
            contact[:, 1, 2] = 1.0 + 0.1 * episode_index
            force = np.zeros((6, 4, 5), dtype=np.float32)
            force[:, 1, 2] = np.arange(1, 7, dtype=np.float32) * (episode_index + 1)
            np.save(contact_dir / f"{frame_idx:06d}.npy", contact)
            np.save(force_dir / f"{frame_idx:06d}.npy", force)
        (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    def test_complete_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "dataset"
            for setting_index in range(5):
                for episode_index in range(2):
                    self.make_episode(source, setting_index, episode_index)

            output = root / "outputs"
            config = {
                "schema_version": "modality_forcing_v1",
                "dataset_source": "omnivitac",
                "source_root": str(source),
                "output_dir": str(output),
                "vae_index": {
                    "camera": "camera2",
                    "strict_camera": True,
                    "frame_stride": 1,
                    "n_frames": 17,
                    "clips_per_episode": 2,
                    "trans_thresh_mm": 1.0,
                    "rot_thresh_rad": 0.01,
                    "with_stats_before_split": False,
                    "seed": 0,
                },
                "split": {
                    "activity_modality": "contact",
                    "train_weight": 1,
                    "test_weight": 1,
                    "active_frac": 0.85,
                    "seed": 0,
                },
                "physical_statistics": {
                    "source": "train_clip_index_only",
                    "n_sample_frames": 0,
                    "deduplicate": True,
                    "seed": 0,
                },
                "pipeline": {
                    "camera": "camera2",
                    "action_stride": 1,
                    "train_statistics_only": True,
                    "require_explicit_physical_paths": True,
                },
                "mixture": {
                    "random_seed": 42,
                    "task_ratios": {"T0": 0.4, "T1": 0.2, "T2": 0.2, "T3": 0.2},
                },
                "check_array_content": True,
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "run_data_pipeline.py"), "--config", str(config_path)],
                check=True,
                cwd=SCRIPT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            report = json.loads((output / "reports" / "validation_report.json").read_text())
            self.assertEqual(report["status"], "pass")
            train = json.loads((output / "mixture" / "train_mixture.json").read_text())
            self.assertEqual(train["n_samples"], 10)
            self.assertEqual(report["mixture"]["task_counts"], {"T0": 4, "T1": 2, "T2": 2, "T3": 2})

            test_t3 = json.loads((output / "mixture" / "test_t3.json").read_text())
            self.assertTrue(test_t3["samples"])
            for sample in test_t3["samples"]:
                self.assertFalse(sample["observation_rgb_is_condition"])
                self.assertTrue(sample["action_is_condition"])
                self.assertEqual(sample["condition_modalities"], ["action", "contact"])
                self.assertEqual(sample["video_noise_frame_mask"], [1] * 17)
                self.assertEqual(sample["video_loss_frame_mask"], [1] * 17)

            physical = json.loads((output / "vae_index" / "physical_statistics.json").read_text())
            train_clips = json.loads((output / "split" / "clips_train.json").read_text())["clips"]
            test_clips = json.loads((output / "split" / "clips_test.json").read_text())["clips"]
            train_episodes = {clip["episode"] for clip in train_clips}
            test_episodes = {clip["episode"] for clip in test_clips}
            self.assertTrue(set(physical["episode_ids"]) <= train_episodes)
            self.assertFalse(set(physical["episode_ids"]) & test_episodes)


if __name__ == "__main__":
    unittest.main()
