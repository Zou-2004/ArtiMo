#!/usr/bin/env python3
"""Regression tests for visual priority deferred to full placement."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from apply_artimo_grasp_orientation_decisions import apply_decisions
from solve_artimo_placement import _gated_orientation_options


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OrientationPriorityTest(unittest.TestCase):
    def _case(self, priorities: dict[str, int]) -> tuple[Path, Path, Path, tempfile.TemporaryDirectory[str]]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        candidates = []
        decisions = []
        for index, candidate_id in enumerate(("roll_a", "roll_b", "roll_c")):
            execution = root / f"{candidate_id}.json"
            execution.write_text(
                json.dumps(
                    {
                        "candidate": candidate_id,
                        "stages": [
                            {
                                "id": "press",
                                "contact_pose_link": {
                                    "rotation_xyzw": [float(index), 0.0, 0.0, 1.0]
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            images: dict[str, str] = {}
            image_hashes: dict[str, str] = {}
            for view in ("oblique", "top", "surface_normal", "tangent"):
                image = root / f"{candidate_id}-{view}.png"
                image.write_bytes(f"{candidate_id}-{view}".encode())
                images[view] = str(image)
                image_hashes[view] = _sha256(image)
            candidates.append(
                {
                    "id": candidate_id,
                    "roll_degrees": float(index * 45),
                    "execution": str(execution),
                    "execution_sha256": _sha256(execution),
                    "orientation_images": images,
                    "orientation_image_sha256": image_hashes,
                }
            )
            decisions.append(
                {
                    "id": candidate_id,
                    "visual_status": "valid",
                    "visual_priority": priorities[candidate_id],
                    "reason": f"Reviewed visual choice {candidate_id}",
                    "reviewed_images": list(images.values()),
                }
            )
        report = {
            "schema_version": 4,
            "visual_render_ik_was_run": False,
            "stage_id": "press",
            "stage_index": 0,
            "interaction": "physical_contact",
            "candidates": candidates,
        }
        report_path = root / "report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        decisions_path = root / "decisions.json"
        decisions_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "report_sha256": _sha256(report_path),
                    "stage_id": "press",
                    "decisions": decisions,
                }
            ),
            encoding="utf-8",
        )
        return report_path, decisions_path, root / "gate", temporary

    def test_first_visual_choice_is_preferred_without_ik(self) -> None:
        report, decisions, output, temporary = self._case(
            {"roll_a": 2, "roll_b": 1, "roll_c": 3}
        )
        self.addCleanup(temporary.cleanup)
        gate = apply_decisions(report, decisions, output)
        self.assertEqual(gate["stages"][0]["selected_candidate_id"], "roll_b")
        self.assertEqual(
            gate["stages"][0]["numerically_evaluated_candidate_ids"], []
        )

    def test_all_visual_candidates_reach_placement_in_priority_order(self) -> None:
        report, decisions, output, temporary = self._case(
            {"roll_a": 2, "roll_b": 1, "roll_c": 3}
        )
        self.addCleanup(temporary.cleanup)
        gate = apply_decisions(report, decisions, output)
        candidates = gate["placement_candidate_groups"][0]["candidates"]
        self.assertEqual(
            [candidate["id"] for candidate in candidates],
            ["roll_b", "roll_a", "roll_c"],
        )
        self.assertEqual(
            [candidate["visual_priority"] for candidate in candidates], [1, 2, 3]
        )

        template = {
            "stages": [
                {
                    "id": "press",
                    "contact_pose_link": {
                        "rotation_xyzw": [99.0, 0.0, 0.0, 1.0]
                    },
                }
            ]
        }
        options = _gated_orientation_options(template, gate)
        self.assertEqual(
            [option["candidate_ids"] for option in options],
            [["roll_b"], ["roll_a"], ["roll_c"]],
        )
        self.assertEqual(
            [option["execution"]["stages"][0]["contact_pose_link"]["rotation_xyzw"][0]
             for option in options],
            [1.0, 0.0, 2.0],
        )


if __name__ == "__main__":
    unittest.main()
