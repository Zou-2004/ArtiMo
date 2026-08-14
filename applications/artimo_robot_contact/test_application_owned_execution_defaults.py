from __future__ import annotations

import copy
import unittest
from unittest import mock

import run_artimo_physics as ph


class ApplicationOwnedExecutionDefaultsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = {
            "schema_version": 2,
            "task_id": "door-defaults-test",
            "inputs": {
                "urdf": "data/causal_data/door1/mobility.urdf",
                "plan": "data/casual_output/door1/person_open_door/door1/plan.json",
                "robot_urdf": "applications/artimo_robot_contact/assets/panda/panda.urdf",
            },
        }
        self.execution = {
            "schema_version": 2,
            "control_execution": [
                {
                    "source_phase": "handle_turn",
                    "source_control_index": 0,
                    "motion_owner": "robot_contact",
                    "stage_id": "handle",
                },
                {
                    "source_phase": "unlatch_latency",
                    "source_control_index": 0,
                    "motion_owner": "hold",
                },
                {
                    "source_phase": "door_throw_open",
                    "source_control_index": 0,
                    "motion_owner": "robot_contact",
                    "stage_id": "door",
                },
                {
                    "source_phase": "handle_return",
                    "source_control_index": 0,
                    "motion_owner": "passive_return",
                },
                {
                    "source_phase": "settle_open",
                    "source_control_index": 0,
                    "motion_owner": "hold",
                },
            ],
            "stages": [
                {
                    "id": "handle",
                    "interaction": "explicit_ideal_feasibility",
                    "contact_link": "link_2",
                    "contact_pose_link": {
                        "translation_m": [-0.041, 0.0, -0.14098],
                        "rotation_xyzw": [0.0, 1.0, 0.0, 0.0],
                    },
                    "target_joint_position": -999.0,
                },
                {
                    "id": "door",
                    "interaction": "explicit_ideal_feasibility",
                    "contact_link": "link_2",
                    "contact_pose_link": {
                        "translation_m": [-0.041, 0.0, -0.14098],
                        "rotation_xyzw": [0.0, 1.0, 0.0, 0.0],
                    },
                    "target_joint_position": -999.0,
                },
            ],
        }

    def test_agent_quaternion_and_plan_projection_are_ignored(self) -> None:
        self.execution["physics_urdf"] = ".artimo-runs/agent-authored-proxy.urdf"
        left = ph.materialize_execution_defaults(self.task, self.execution)
        alternate = copy.deepcopy(self.execution)
        for stage in alternate["stages"]:
            stage["contact_pose_link"]["rotation_xyzw"] = [0.3, 0.4, 0.5, 0.6]
            stage["source_phase"] = "agent_guess"
            stage["driver_joint"] = "agent_guess"
            stage["forbidden_contact_links"] = []
        right = ph.materialize_execution_defaults(self.task, alternate)

        for execution in (left, right):
            self.assertNotIn("physics_urdf", execution)
            first, second = execution["stages"]
            self.assertEqual(first["contact_pose_link"]["rotation_xyzw"], [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(first["driver_joint"], "joint_2")
            self.assertEqual(second["driver_joint"], "joint_1")
            self.assertEqual(first["target_joint_position"], 1.57)
            self.assertEqual(first["contact_sequence"], second["contact_sequence"])
            self.assertEqual(second["release_before_phase"], "handle_return")
            self.assertEqual(first["forbidden_contact_links"], ["base", "link_0", "link_1"])
            self.assertEqual(
                first["allowed_robot_contact_links"],
                ["panda_leftfinger", "panda_rightfinger"],
            )
            ph._validate_execution_schema(execution)

        self.assertEqual(
            left["stages"][0]["contact_pose_link"]["rotation_xyzw"],
            right["stages"][0]["contact_pose_link"]["rotation_xyzw"],
        )

    def test_contact_link_change_creates_release_and_reacquire_without_plan_release(self) -> None:
        task = {
            "schema_version": 2,
            "task_id": "dishwasher-link-switch-test",
            "inputs": {
                "urdf": "data/causal_data/major_appliances__dishwasher__dishwasher_1/mobility.urdf",
                "plan": (
                    "data/casual_output/major_appliances__dishwasher__dishwasher_1/"
                    "open_bottom_tray/major_appliances__dishwasher__dishwasher_1/plan.json"
                ),
                "robot_urdf": "applications/artimo_robot_contact/assets/panda/panda.urdf",
            },
        }
        execution = {
            "schema_version": 2,
            "control_execution": [
                {
                    "source_phase": "open_door",
                    "source_control_index": 0,
                    "motion_owner": "robot_contact",
                    "stage_id": "door",
                },
                {
                    "source_phase": "pull_bottom_tray",
                    "source_control_index": 0,
                    "motion_owner": "robot_contact",
                    "stage_id": "tray",
                },
                {
                    "source_phase": "settle_hold",
                    "source_control_index": 0,
                    "motion_owner": "hold",
                },
                {
                    "source_phase": "settle_hold",
                    "source_control_index": 1,
                    "motion_owner": "hold",
                },
            ],
            "stages": [
                {
                    "id": "door",
                    "interaction": "explicit_ideal_feasibility",
                    "contact_link": "link_2",
                    "contact_pose_link": {
                        "translation_m": [-0.208, -0.0002, -0.23235]
                    },
                },
                {
                    "id": "tray",
                    "interaction": "explicit_ideal_feasibility",
                    "contact_link": "link_6",
                    "contact_pose_link": {
                        "translation_m": [0.011, 0.2861, 0.0128]
                    },
                },
            ],
        }
        materialized = ph.materialize_execution_defaults(task, execution)
        door, tray = materialized["stages"]
        self.assertNotEqual(door["contact_sequence"], tray["contact_sequence"])
        self.assertEqual(door["release_before_phase"], "pull_bottom_tray")
        self.assertNotIn("release_before_phase", tray)
        plan = ph._read_json(ph._resolve(task["inputs"]["plan"]))
        ph._validate_execution_against_plan(
            plan, materialized, require_release_route=False
        )

    def test_collision_model_uses_source_even_with_legacy_agent_value(self) -> None:
        source = ph._resolve(self.task["inputs"]["urdf"])
        legacy = {"physics_urdf": ".artimo-runs/agent-choice/physics.urdf"}
        actual = ph.resolve_simulation_urdf(self.task, legacy, source)
        self.assertEqual(actual, source)

    def test_release_sweep_stops_before_next_contact_acquisition(self) -> None:
        plan = ph._read_json(
            ph._resolve(
                "data/casual_output/major_appliances__dishwasher__dishwasher_1/"
                "open_bottom_tray/major_appliances__dishwasher__dishwasher_1/plan.json"
            )
        )
        state_before_tray = ph._object_joint_state_before_phase(
            plan, {"joint_0": 0.0, "joint_2": 0.0}, "pull_bottom_tray"
        )

        self.assertEqual(
            ph._object_joint_transitions_from_phase(
                plan,
                state_before_tray,
                "pull_bottom_tray",
                "pull_bottom_tray",
            ),
            [],
        )
        self.assertEqual(
            [
                row["joint"]
                for row in ph._object_joint_transitions_from_phase(
                    plan, state_before_tray, "pull_bottom_tray"
                )
            ],
            ["joint_2"],
        )


if __name__ == "__main__":
    unittest.main()
