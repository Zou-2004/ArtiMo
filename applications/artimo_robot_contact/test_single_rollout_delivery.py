#!/usr/bin/env python3
"""Regression tests for the single-rollout delivery contract."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import finalize_artimo_delivery as delivery


HEX = "1" * 64


class SingleRolloutDeliveryTest(unittest.TestCase):
    def test_finalizer_consumes_one_rollout_and_emits_no_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            task_spec = root / "task.json"
            rollout_dir = root / "rollout"
            output_dir = root / "output"
            visual_qa = root / "visual-qa.json"
            rollout_dir.mkdir()

            task_spec.write_text(
                json.dumps({"task_id": "single-rollout"}), encoding="utf-8"
            )
            grasp = {"seeds": {"search": 0, "physics": 0}}
            rollout = {
                "passed": True,
                "inputs": {},
                "physical": {
                    "robot_command_schedule_sha256": "2" * 64,
                    "object_joint_resets_after_initialization": 0,
                    "maximum_runtime_constraint_count": 0,
                    "contacts": [{"target_contact_observations": 1}],
                    "joint_motion": {},
                },
                "negative_control": {
                    "robot_command_schedule_sha256": "2" * 64,
                    "contacts": [{"target_contact_observations": 0}],
                    "causal_triggers": 0,
                    "requested_joint_motion_remained_initial": True,
                },
            }
            (rollout_dir / "grasp.json").write_text(
                json.dumps(grasp), encoding="utf-8"
            )
            (rollout_dir / "result.json").write_text(
                json.dumps(rollout), encoding="utf-8"
            )
            (rollout_dir / "video.mp4").write_bytes(b"video")
            visual_qa.write_text(
                json.dumps(
                    {
                        "sample_rate_fps": 5.0,
                        "no_visible_interpenetration": True,
                        "physical_contact_visible": True,
                        "requested_motion_visible": True,
                        "no_rendering_artifacts": True,
                    }
                ),
                encoding="utf-8",
            )

            result = delivery.finalize(
                task_spec,
                rollout_dir,
                visual_qa,
                output_dir,
                HEX,
                HEX,
            )

            self.assertTrue(result["passed"])
            self.assertIn("evidence", result)
            self.assertIn("native_rollout", result)
            self.assertNotIn("reproducibility", result)
            self.assertNotIn("native_first", result)
            self.assertNotIn("native_second", result)
            self.assertNotIn("second_run", result["evidence"])
            self.assertEqual(
                {path.name for path in output_dir.iterdir()}, delivery.FINAL_NAMES
            )


if __name__ == "__main__":
    unittest.main()
