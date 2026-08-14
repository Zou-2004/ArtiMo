#!/usr/bin/env python3
"""Regression tests for the real bilateral-contact acquisition gate."""
from __future__ import annotations

import unittest

import run_artimo_physics as physics


def _contact(robot_link: int, object_link: int, normal: tuple[float, float, float]):
    # PyBullet contact tuple fields used by the gate are link A/B at 3/4 and
    # contactNormalOnB at 7.
    return (0, 0, 0, robot_link, object_link, (0, 0, 0), (0, 0, 0), normal, 0.0)


class GraspAcquisitionGateTest(unittest.TestCase):
    def sample(self, contacts, *, speed: float = 0.0):
        return physics._bilateral_grasp_sample(
            contacts,
            {9, 10},
            4,
            [(0.012, speed), (0.012, -speed)],
            0.04,
            0.0064,
        )

    def test_requires_two_real_finger_links_on_same_target(self) -> None:
        row = self.sample([_contact(9, 4, (1.0, 0.0, 0.0))])
        self.assertFalse(row["passed"])
        self.assertFalse(row["both_finger_links_contact_target"])

    def test_rejects_two_fingers_touching_from_same_side(self) -> None:
        row = self.sample(
            [
                _contact(9, 4, (1.0, 0.0, 0.0)),
                _contact(10, 4, (0.9, 0.1, 0.0)),
            ]
        )
        self.assertFalse(row["passed"])
        self.assertFalse(row["opposed_contact_normals"])

    def test_accepts_opposed_settled_bilateral_contact(self) -> None:
        row = self.sample(
            [
                _contact(9, 4, (1.0, 0.0, 0.0)),
                _contact(10, 4, (-1.0, 0.0, 0.0)),
            ]
        )
        self.assertTrue(row["passed"])
        self.assertTrue(row["both_finger_links_contact_target"])
        self.assertTrue(row["opposed_contact_normals"])

    def test_rejects_transient_high_speed_contact(self) -> None:
        row = self.sample(
            [
                _contact(9, 4, (1.0, 0.0, 0.0)),
                _contact(10, 4, (-1.0, 0.0, 0.0)),
            ],
            speed=physics.GRASP_FINGER_MAXIMUM_SPEED_M_S * 2.0,
        )
        self.assertFalse(row["passed"])
        self.assertFalse(row["fingers_settled"])

    def test_rejects_palm_or_other_object_link_contact(self) -> None:
        row = self.sample(
            [
                _contact(9, 4, (1.0, 0.0, 0.0)),
                _contact(10, 4, (-1.0, 0.0, 0.0)),
                _contact(8, 4, (0.0, 0.0, 1.0)),
                _contact(9, 3, (0.0, 1.0, 0.0)),
            ]
        )
        self.assertFalse(row["passed"])
        self.assertFalse(row["non_target_contact_zero"])


if __name__ == "__main__":
    unittest.main()
