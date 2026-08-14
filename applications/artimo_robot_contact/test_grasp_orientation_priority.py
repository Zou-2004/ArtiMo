#!/usr/bin/env python3
"""Regressions for visual hard-gating plus whole-task placement."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from apply_artimo_grasp_orientation_decisions import apply_decisions
from solve_artimo_placement import (
    _adaptive_driver_path,
    _candidate_rank,
    _contact_facing_sparse_matrix,
    _dense_shortlist,
    _expand_rule_based_grasp_depths,
    _feasible_region_summary,
    _gated_orientation_options,
    _manipulation_block_stage_ids,
    _sparse_survivors,
    _tier_ik_budget,
    _transit_route_repair,
)
from render_artimo_grasp_orientation_candidates import DEFAULT_ROLL_DEGREES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OrientationPriorityTest(unittest.TestCase):
    def test_prior_moved_link_transit_collision_requires_route_solver(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        urdf = Path(temporary.name) / "object.urdf"
        urdf.write_text(
            """<robot name='fixture'>
            <link name='base'/><link name='door'/><link name='handle'/>
            <joint name='door_joint' type='revolute'>
              <parent link='base'/><child link='door'/><axis xyz='0 0 1'/>
              <limit lower='0' upper='1.57' effort='1' velocity='1'/>
            </joint>
            <joint name='handle_fixed' type='fixed'>
              <parent link='door'/><child link='handle'/>
            </joint></robot>""",
            encoding="utf-8",
        )
        stages = [
            {"id": "open", "driver_joint": "door_joint", "contact_sequence": "a"},
            {
                "id": "pull_tray",
                "driver_joint": "tray_joint",
                "contact_sequence": "b",
                "minimum_swept_clearance_m": 0.0,
            },
        ]
        plan = SimpleNamespace(swept_clearance_violations=[{
            "phase": "transit_in",
            "object_link": "handle",
            "robot_link": "panda_link4",
            "distance_m": -0.012,
        }])
        repair = _transit_route_repair(urdf, stages, 1, plan)
        self.assertIsNotNone(repair)
        self.assertEqual(repair["primary_obstacle_link"], "handle")
        self.assertEqual(
            repair["classification"], "prior_plan_moved_link_blocks_transit"
        )

        plan.swept_clearance_violations[0]["phase"] = "manipulate"
        self.assertIsNone(_transit_route_repair(urdf, stages, 1, plan))

    def test_default_rolls_keep_symmetric_jaws_as_distinct_robot_ik_branches(self) -> None:
        self.assertEqual(
            DEFAULT_ROLL_DEGREES, (0.0, 45.0, 90.0, 135.0, 180.0)
        )
        self.assertEqual(len(DEFAULT_ROLL_DEGREES), 5)
        self.assertEqual(0.0 % 180.0, 180.0 % 180.0)

    def test_screening_ik_budgets_do_not_inherit_dense_restarts(self) -> None:
        execution_ik = {"random_restarts": 96, "max_iterations": 2000}
        self.assertEqual(_tier_ik_budget(execution_ik, "coarse"), (4, 500))
        self.assertEqual(_tier_ik_budget(execution_ik, "sparse"), (12, 1000))
        self.assertEqual(_tier_ik_budget(execution_ik, "dense"), (96, 2000))
        self.assertEqual(
            _tier_ik_budget(
                execution_ik,
                "coarse",
                random_restarts_override=6,
                max_iterations_override=700,
            ),
            (6, 700),
        )

    def test_adaptive_dense_path_refines_only_large_pose_steps(self) -> None:
        constant = _adaptive_driver_path(
            0.0,
            1.0,
            lambda value: ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            initial_samples=5,
            maximum_samples=33,
        )
        self.assertEqual(len(constant), 5)

        changing = _adaptive_driver_path(
            0.0,
            1.0,
            lambda value: ([value, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
            initial_samples=5,
            maximum_samples=33,
            maximum_position_step_m=0.08,
        )
        self.assertGreater(len(changing), 5)
        self.assertLessEqual(len(changing), 33)
        self.assertAlmostEqual(float(changing[0]), 0.0)
        self.assertAlmostEqual(float(changing[-1]), 1.0)

    def test_only_sparse_passes_reach_dense_top_k(self) -> None:
        rejected = self._placement_attempt(0.45, -0.1)
        rejected["sparse_screening_passed"] = False
        survivors = []
        for distance, residual in ((0.5, 0.003), (0.6, 0.001), (0.7, 0.002)):
            row = self._placement_attempt(distance, 0.0)
            row["sparse_screening_passed"] = True
            row["stages"][0]["maximum_ik_position_error_m"] = residual
            survivors.append(row)
        sparse_rows = _sparse_survivors([rejected, *survivors])
        self.assertNotIn(rejected, sparse_rows)
        self.assertEqual(len(sparse_rows), 3)
        dense_rows = _dense_shortlist(sparse_rows, 2)
        self.assertEqual(len(dense_rows), 2)
        self.assertNotIn(rejected, dense_rows)

    def test_dense_top_k_prefers_larger_gpu_path_clearance(self) -> None:
        tight = self._placement_attempt(0.55, 0.0, lateral=0.0)
        roomy = self._placement_attempt(0.65, 0.0, lateral=0.2)
        for row, clearance in ((tight, 0.002), (roomy, 0.035)):
            row["sparse_screening_passed"] = True
            row["manipulation_blocks"] = [{
                "feasible": True,
                "target_actually_gripped": True,
                "minimum_sample_completion_ratio": 1.0,
                "minimum_gpu_environment_clearance_m": clearance,
                "deepest_body_penetration_m": 0.0,
                "maximum_target_link_gap_m": 0.0,
                "maximum_ik_position_error_m": 0.001,
            }]
        self.assertIs(_dense_shortlist([tight, roomy], 1)[0], roomy)

    @staticmethod
    def _placement_attempt(
        distance: float, penetration: float, *, lateral: float = 0.0
    ) -> dict:
        return {
            "placement_mode": "contact_facing",
            "object_yaw_deg": 0.0,
            "robot_base_m": [distance, 0.0, 0.0],
            "contact_facing_distance_m": distance,
            "contact_facing_lateral_offset_m": lateral,
            "contact_facing_yaw_offset_deg": 0.0,
            "orientation_candidate_ids": ["roll_best"],
            "stages": [
                {
                    "samples_solved": 13,
                    "samples_required": 13,
                    "deepest_body_penetration_m": penetration,
                    "maximum_ik_position_error_m": 0.001,
                    "maximum_target_link_gap_m": 0.001,
                    "target_actually_gripped": True,
                }
            ],
        }

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
            "maximum_target_gap_m": 0.006,
            "contact_offset_under_review": {
                "contact_translation_m": [0.0, 0.0, 0.0],
                "grasp_depth_m": 0.0,
                "robot_tool_contact_offset_eef_m": None,
                "finger_opening_m": 0.01,
            },
            "maximum_target_gap_m": 0.006,
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
                    "schema_version": 4,
                    "report_sha256": _sha256(report_path),
                    "stage_id": "press",
                    "decisions": [
                        {**decision, "angle_status": "valid"}
                        for decision in decisions
                    ],
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

    def test_angle_decision_does_not_require_agent_depth_judgment(self) -> None:
        report, decisions, output, temporary = self._case(
            {"roll_a": 1, "roll_b": 2, "roll_c": 3}
        )
        self.addCleanup(temporary.cleanup)
        gate = apply_decisions(report, decisions, output)
        self.assertEqual(gate["stages"][0]["agent_decision_scope"], "wrist_angle_only")

    def test_gate_assigns_depth_to_rule_based_search(self) -> None:
        report, decisions, output, temporary = self._case(
            {"roll_a": 1, "roll_b": 2, "roll_c": 3}
        )
        self.addCleanup(temporary.cleanup)
        gate = apply_decisions(report, decisions, output)
        self.assertEqual(
            gate["stages"][0]["grasp_depth_owner"],
            "application_rule_based_dense_search",
        )

    def test_gate_records_angle_only_evidence(self) -> None:
        report, decisions, output, temporary = self._case(
            {"roll_a": 1, "roll_b": 2, "roll_c": 3}
        )
        self.addCleanup(temporary.cleanup)
        gate = apply_decisions(report, decisions, output)
        row = gate["stages"][0]
        self.assertFalse(row["nominal_contact_geometry_frozen_by_visual_gate"])
        self.assertEqual(row["selected_angle_status"], "valid")

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

    def test_sparse_matrix_contains_every_distance_lateral_cell(self) -> None:
        matrix = _contact_facing_sparse_matrix(
            [0.0], [0.0], [0.45, 0.55, 0.65], [0.0], [0.0, -0.1, 0.1]
        )
        self.assertEqual(len(matrix), 9)
        self.assertEqual(
            {(row[2], row[4]) for row in matrix},
            {
                (distance, lateral)
                for distance in (0.45, 0.55, 0.65)
                for lateral in (0.0, -0.1, 0.1)
            },
        )

    def test_rule_based_depth_search_preserves_one_depth_per_sequence(self) -> None:
        record = {
            "_candidate_execution": {
                "stages": [
                    {
                        "id": "turn",
                        "interaction": "explicit_ideal_feasibility",
                        "contact_sequence": "door",
                        "grasp_depth_m": 0.0,
                    },
                    {
                        "id": "pull",
                        "interaction": "explicit_ideal_feasibility",
                        "contact_sequence": "door",
                        "grasp_depth_m": 0.0,
                    },
                ]
            }
        }
        expanded = _expand_rule_based_grasp_depths([record])
        self.assertEqual(len(expanded), 6)
        self.assertEqual(
            expanded[0]["rule_based_grasp_depth_m_by_group"],
            {"sequence:door": 0.015},
        )
        self.assertTrue(
            all(
                row["_candidate_execution"]["stages"][0]["grasp_depth_m"]
                == row["_candidate_execution"]["stages"][1]["grasp_depth_m"]
                for row in expanded
            )
        )

    def test_adjacent_continuous_stages_form_one_manipulation_block(self) -> None:
        stages = [
            {"id": "handle_turn", "contact_sequence": "door_grasp"},
            {"id": "door_open", "contact_sequence": "door_grasp"},
            {"id": "tray_grasp", "contact_sequence": "tray_grasp"},
            {"id": "tray_pull", "contact_sequence": "tray_grasp"},
        ]
        self.assertEqual(
            _manipulation_block_stage_ids(stages),
            [["handle_turn", "door_open"], ["tray_grasp", "tray_pull"]],
        )

    def test_independent_blocks_keep_cartesian_contact_combinations(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        groups = []
        for stage_id, prefix in (("door", "d"), ("tray", "t")):
            candidates = []
            for priority in (1, 2):
                path = root / f"{prefix}{priority}.json"
                path.write_text(
                    json.dumps(
                        {
                            "stages": [
                                {
                                    "id": stage_id,
                                    "contact_pose_link": {
                                        "rotation_xyzw": [
                                            float(priority), 0.0, 0.0, 1.0
                                        ]
                                    },
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                candidates.append(
                    {
                        "id": f"{prefix}{priority}",
                        "visual_priority": priority,
                        "execution": str(path),
                        "execution_sha256": _sha256(path),
                    }
                )
            groups.append({"stage_ids": [stage_id], "candidates": candidates})
        template = {
            "stages": [
                {"id": "door", "contact_pose_link": {"rotation_xyzw": [0, 0, 0, 1]}},
                {"id": "tray", "contact_pose_link": {"rotation_xyzw": [0, 0, 0, 1]}},
            ]
        }
        options = _gated_orientation_options(
            template, {"placement_candidate_groups": groups}
        )
        self.assertEqual(len(options), 4)
        self.assertEqual(
            {tuple(option["candidate_ids"]) for option in options},
            {("d1", "t1"), ("d1", "t2"), ("d2", "t1"), ("d2", "t2")},
        )

    def test_worst_block_beats_greedy_first_block_score(self) -> None:
        def candidate(ratios: tuple[float, float], priority: int) -> dict:
            return {
                "stages": [],
                "orientation_visual_priorities": [priority],
                "manipulation_blocks": [
                    {
                        "feasible": False,
                        "target_actually_gripped": True,
                        "minimum_sample_completion_ratio": ratio,
                        "deepest_body_penetration_m": 0.0,
                        "maximum_target_link_gap_m": 0.001,
                        "maximum_ik_position_error_m": 0.001,
                    }
                    for ratio in ratios
                ],
            }

        greedy = candidate((0.98, 0.12), 1)
        balanced = candidate((0.78, 0.76), 2)
        self.assertLess(_candidate_rank(balanced), _candidate_rank(greedy))

    def test_lateral_seed_is_joint_over_contact_choices(self) -> None:
        greedy = self._placement_attempt(0.45, -0.001)
        greedy["orientation_candidate_ids"] = ["door_visual_best", "tray_a"]
        greedy["orientation_visual_priorities"] = [1, 1]
        greedy["manipulation_blocks"] = [
            {
                "feasible": False,
                "target_actually_gripped": True,
                "minimum_sample_completion_ratio": 0.98,
                "deepest_body_penetration_m": 0.0,
                "maximum_target_link_gap_m": 0.001,
                "maximum_ik_position_error_m": 0.001,
            },
            {
                "feasible": False,
                "target_actually_gripped": False,
                "minimum_sample_completion_ratio": 0.12,
                "deepest_body_penetration_m": -0.05,
                "maximum_target_link_gap_m": 0.04,
                "maximum_ik_position_error_m": 0.02,
            },
        ]
        balanced = self._placement_attempt(0.60, -0.003)
        balanced["orientation_candidate_ids"] = ["door_b", "tray_b"]
        balanced["orientation_visual_priorities"] = [2, 2]
        balanced["manipulation_blocks"] = [
            {
                "feasible": False,
                "target_actually_gripped": True,
                "minimum_sample_completion_ratio": 0.80,
                "deepest_body_penetration_m": -0.003,
                "maximum_target_link_gap_m": 0.002,
                "maximum_ik_position_error_m": 0.002,
            },
            {
                "feasible": False,
                "target_actually_gripped": True,
                "minimum_sample_completion_ratio": 0.77,
                "deepest_body_penetration_m": -0.003,
                "maximum_target_link_gap_m": 0.002,
                "maximum_ik_position_error_m": 0.002,
            },
        ]
        self.assertLess(_candidate_rank(balanced), _candidate_rank(greedy))

    def test_feasible_region_is_intersection_of_all_blocks(self) -> None:
        def attempt(x: float, block_ok: tuple[bool, bool]) -> dict:
            return {
                "robot_base_m": [x, 0.0, 0.0],
                "robot_yaw_deg": 180.0,
                "orientation_candidate_ids": ["door", "tray"],
                "feasible": all(block_ok),
                "manipulation_blocks": [
                    {"block_id": f"block_{index}", "feasible": value}
                    for index, value in enumerate(block_ok)
                ],
            }

        summary = _feasible_region_summary(
            [attempt(0.5, (True, False)), attempt(0.6, (True, True))]
        )
        self.assertEqual(
            [row["robot_base_m"] for row in summary["whole_task_feasible_base_region"]],
            [[0.6, 0.0, 0.0]],
        )


if __name__ == "__main__":
    unittest.main()
