from __future__ import annotations

import unittest

import solve_artimo_placement as placement


class PlacementCollisionFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.execution = {
            "stages": [
                {
                    "id": "grasp",
                    "interaction": "explicit_ideal_feasibility",
                    "contact_link": "link_2",
                    "grasp_depth_m": -0.0075,
                    "allowed_robot_contact_links": [
                        "panda_leftfinger",
                        "panda_rightfinger",
                    ],
                    "forbidden_contact_links": ["base", "link_0", "link_1"],
                    "contact_acquisition": {"mode": "open_then_close"},
                }
            ]
        }

    def test_non_target_collision_rejects_depth_without_agent_link_policy(self) -> None:
        best = {
            "stages": [
                {
                    "stage_id": "grasp",
                    "worst_offenders": {
                        "panda_leftfinger|link_1": -0.0018,
                        "panda_hand|link_2": -0.0062,
                    },
                    "dense_full_path_required_clearance_m": 0.0,
                    "dense_full_path_tightest_samples": [],
                    "forbidden_clearance_passed": False,
                    "minimum_forbidden_clearance_m": -0.0018,
                }
            ]
        }

        feedback = placement._collision_rejection_feedback(best, self.execution)

        self.assertFalse(feedback["agent_collision_link_input_accepted"])
        self.assertFalse(feedback["depth_adjustment_valid"])
        self.assertEqual(feedback["status"], "rejected_collision_at_proposed_depth")
        stage = feedback["rejected_stages"][0]
        self.assertEqual(stage["application_forbidden_contact_links"], [
            "base", "link_0", "link_1"
        ])
        self.assertEqual(stage["effective_robot_contact_offset_m"], -0.0225)
        classes = {row["collision_class"] for row in stage["violations"]}
        self.assertEqual(
            classes,
            {"non_target_object_link", "non_allowed_robot_link_on_target"},
        )

    def test_non_collision_failure_does_not_blame_depth(self) -> None:
        best = {
            "stages": [
                {
                    "stage_id": "grasp",
                    "worst_offenders": {},
                    "forbidden_clearance_passed": True,
                    "minimum_forbidden_clearance_m": 0.02,
                    "dense_full_path_tightest_samples": [],
                    "full_path_rejected": "dense_ik_diagnostics",
                }
            ]
        }

        feedback = placement._collision_rejection_feedback(best, self.execution)

        self.assertTrue(feedback["depth_adjustment_valid"])
        self.assertEqual(feedback["status"], "no_collision_attributed_depth_rejection")
        self.assertEqual(feedback["rejected_stages"], [])


if __name__ == "__main__":
    unittest.main()
