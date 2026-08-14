#!/usr/bin/env python3
"""Regression tests for release-retreat ordering and nonzero settle time."""
from __future__ import annotations

import unittest

import numpy as np

import run_artimo_physics as physics


def _plan() -> dict:
    return {
        "timeline": [
            {
                "name": "contact",
                "controls": [
                    {"joint": "driver", "mode": "joint_position", "target": 0.01}
                ],
            },
            {
                "name": "latched_hold",
                "controls": [{"joint": "driver", "mode": "hold_position"}],
            },
            {
                "name": "mechanism_motion",
                "controls": [
                    {"joint": "effect", "mode": "joint_position", "target": 1.0}
                ],
            },
            {
                "name": "control_return",
                "controls": [
                    {"joint": "driver", "mode": "spring_return", "target": 0.0}
                ],
            },
        ]
    }


def _execution(release_before_phase: str) -> dict:
    return {
        "robot": {"home_joint_positions": [0.0] * 7},
        "settle_s": 0.0,
        "stages": [
            {
                "id": "contact-stage",
                "source_phase": "contact",
                "source_control_index": 0,
                "driver_joint": "driver",
                "interaction": "explicit_ideal_feasibility",
                "release_before_phase": release_before_phase,
                "finger_opening_m": 0.02,
                "hold_s": 0.0,
                "contact_acquisition": {
                    "mode": "maintain_width",
                    "approach_finger_opening_m": 0.02,
                    "close_s": 0.0,
                    "settle_s": 0.0,
                    "release_s": 0.0,
                },
            }
        ],
        "control_execution": [
            {
                "source_phase": "contact",
                "source_control_index": 0,
                "motion_owner": "robot_contact",
                "stage_id": "contact-stage",
            },
            {
                "source_phase": "latched_hold",
                "source_control_index": 0,
                "motion_owner": "hold",
            },
            {
                "source_phase": "mechanism_motion",
                "source_control_index": 0,
                "motion_owner": "internal_mechanism",
                "causal_rule_id": "triggered-motion",
            },
            {
                "source_phase": "control_return",
                "source_control_index": 0,
                "motion_owner": "passive_return",
            },
        ],
        "causal_rules": [
            {
                "id": "triggered-motion",
                "trigger_stage": "contact-stage",
                "source_effect_phase": "mechanism_motion",
                "effects": [],
            }
        ],
    }


def _stage_plan(execution: dict) -> physics.StagePlan:
    zeros = np.zeros(7, dtype=np.float64)
    safe = np.full(7, 0.1, dtype=np.float64)
    return physics.StagePlan(
        stage=execution["stages"][0],
        approach=np.asarray([zeros]),
        manipulation=np.asarray([zeros]),
        retreat=np.asarray([zeros, safe]),
        object_path=np.asarray([0.0]),
        maximum_position_error_m=0.0,
        maximum_orientation_error_rad=0.0,
        minimum_swept_clearance_m=0.1,
        swept_clearance_violations=[],
    )


class ReleaseTimingTest(unittest.TestCase):
    def test_terminal_plan_hold_omits_release_retreat_and_home(self) -> None:
        object_plan = {
            "timeline": [
                {
                    "name": "contact",
                    "controls": [
                        {
                            "joint": "driver",
                            "mode": "joint_position",
                            "q_target_rad": 1.0,
                        }
                    ],
                },
                {
                    "name": "terminal_hold",
                    "controls": [
                        {"joint": "driver", "mode": "hold_position"}
                    ],
                },
            ]
        }
        execution = _execution("mechanism_motion")
        stage = execution["stages"][0]
        stage.pop("release_before_phase")
        execution["control_execution"] = [
            {
                "source_phase": "contact",
                "source_control_index": 0,
                "motion_owner": "robot_contact",
                "stage_id": "contact-stage",
            },
            {
                "source_phase": "terminal_hold",
                "source_control_index": 0,
                "motion_owner": "hold",
            },
        ]
        execution["causal_rules"] = []
        self.assertEqual(
            physics._terminal_plan_hold_phase_index(
                object_plan, execution, stage
            ),
            1,
        )
        commands = physics._schedule(
            [_stage_plan(execution)], execution, object_plan
        )
        phases = {command["phase"] for command in commands}
        self.assertNotIn("contact_release", phases)
        self.assertNotIn("retreat", phases)
        phase_sequence = [command["phase"] for command in commands]
        self.assertLess(
            max(
                index
                for index, phase in enumerate(phase_sequence)
                if phase == "transit"
            ),
            phase_sequence.index("manipulate"),
        )
        self.assertEqual(commands[-1]["finger"], stage["finger_opening_m"])

    def test_grasp_is_closed_and_settled_before_attach_and_manipulate(self) -> None:
        execution = _execution("mechanism_motion")
        acquisition = execution["stages"][0]["contact_acquisition"]
        acquisition.update(
            {
                "mode": "open_then_close",
                "approach_finger_opening_m": 0.04,
                "close_s": 0.05,
                "settle_s": 0.05,
                "release_s": 0.05,
            }
        )
        commands = physics._schedule([_stage_plan(execution)], execution, _plan())
        phases = [command["phase"] for command in commands]
        last_close = max(i for i, phase in enumerate(phases) if phase == "contact_acquire")
        last_unattached_settle = max(
            i for i, phase in enumerate(phases) if phase == "contact_settle"
        )
        attach = phases.index("contact_attach")
        first_stabilize = phases.index("grasp_stabilize")
        first_manipulate = phases.index("manipulate")
        self.assertLess(last_close, last_unattached_settle)
        self.assertLess(last_unattached_settle, attach)
        self.assertLess(attach, first_stabilize)
        self.assertLess(first_stabilize, first_manipulate)

    def test_planning_can_defer_release_route_but_physics_cannot(self) -> None:
        stages = _execution("mechanism_motion")["stages"]
        with self.assertRaisesRegex(ValueError, "release_retreat_waypoints_world"):
            physics._validate_contact_sequences(stages, _plan())
        physics._validate_contact_sequences(
            stages, _plan(), require_release_route=False
        )

    def test_rejects_release_after_earlier_mechanism_motion(self) -> None:
        execution = _execution("control_return")
        with self.assertRaisesRegex(ValueError, "too late.*mechanism_motion"):
            physics._validate_release_boundaries(_plan(), execution)

    def test_retreat_settles_before_mechanism_phase_is_entered(self) -> None:
        execution = _execution("mechanism_motion")
        physics._validate_release_boundaries(_plan(), execution)
        commands = physics._schedule([_stage_plan(execution)], execution, _plan())

        settle_indices = [
            index
            for index, command in enumerate(commands)
            if command["phase"] == "release_retreat_settle"
        ]
        self.assertEqual(
            len(settle_indices),
            int(round(physics.RELEASE_RETREAT_SETTLE_S / physics.DT)),
        )
        self.assertGreater(len(settle_indices), 0)
        first_mechanism_index = next(
            index
            for index, command in enumerate(commands)
            if command["timeline_phase_index"] == 2
        )
        self.assertLess(settle_indices[-1], first_mechanism_index)
        self.assertTrue(
            all(
                command["timeline_phase_index"] == 1
                and not command["active_passive_joints"]
                for command in (commands[index] for index in settle_indices)
            )
        )


if __name__ == "__main__":
    unittest.main()
