#!/usr/bin/env python3
"""Plan-authority regressions for robot contact continuity."""
from __future__ import annotations

import unittest

import run_artimo_physics as physics


def _stage(stage_id: str, phase: str, driver: str, link: str, sequence: str | None) -> dict:
    return {
        "id": stage_id,
        "source_phase": phase,
        "source_control_index": 0,
        "driver_joint": driver,
        "contact_link": link,
        "contact_sequence": sequence,
        "interaction": "explicit_ideal_feasibility",
        "contact_pose_link": {
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "allowed_robot_contact_links": ["finger_a", "finger_b"],
        "finger_opening_m": 0.01,
        "grasp_depth_m": 0.0,
        "contact_acquisition": {
            "mode": "open_then_close",
            "approach_finger_opening_m": 0.04,
            "close_s": 0.4,
            "settle_s": 0.2,
            "release_s": 0.3,
        },
    }


def _plan(with_release: bool = False) -> dict:
    timeline = [
        {
            "name": "turn_handle",
            "phase_type": "control_actuation",
            "controls": [{"joint": "handle_joint", "mode": "joint_position", "target": 1.0}],
        },
        {
            "name": "hold_handle",
            "phase_type": "causal_latency",
            "controls": [{"joint": "handle_joint", "mode": "hold_position"}],
        },
    ]
    if with_release:
        timeline.append(
            {
                "name": "release_handle",
                "phase_type": "control_release",
                "controls": [{"joint": "handle_joint", "mode": "spring_return", "target": 0.0}],
            }
        )
    timeline.append(
        {
            "name": "open_door",
            "phase_type": "effect_motion",
            "controls": [{"joint": "door_joint", "mode": "joint_position", "target": 1.0}],
        }
    )
    return {"timeline": timeline}


class PlanContactAuthorityTest(unittest.TestCase):
    def test_rejects_contact_link_change_without_plan_release(self) -> None:
        stages = [
            _stage("turn", "turn_handle", "handle_joint", "handle", "grasp"),
            _stage("open", "open_door", "door_joint", "panel", None),
        ]
        with self.assertRaisesRegex(ValueError, "no control_release.*same contact_link"):
            physics._validate_contact_sequences(stages, _plan(), require_release_route=False)

    def test_accepts_continued_handle_grasp_across_changed_driver_joint(self) -> None:
        stages = [
            _stage("turn", "turn_handle", "handle_joint", "handle", "grasp"),
            _stage("open", "open_door", "door_joint", "handle", "grasp"),
        ]
        physics._validate_contact_sequences(stages, _plan(), require_release_route=False)

    def test_plan_control_release_allows_new_contact(self) -> None:
        stages = [
            _stage("turn", "turn_handle", "handle_joint", "handle", "first"),
            _stage("open", "open_door", "door_joint", "panel", "second"),
        ]
        physics._validate_contact_sequences(stages, _plan(with_release=True), require_release_route=False)


if __name__ == "__main__":
    unittest.main()
