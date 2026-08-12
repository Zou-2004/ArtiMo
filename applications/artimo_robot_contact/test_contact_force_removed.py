#!/usr/bin/env python3
"""Regression tests for force-independent contact and causal schemas."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parent


class ContactForceRemovedTest(unittest.TestCase):
    def test_task_acceptance_has_no_contact_force_fields(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "artimo_robot_task.schema.json").read_text(
                encoding="utf-8"
            )
        )
        acceptance = schema["properties"]["acceptance"]
        self.assertNotIn("minimum_contact_force_n", acceptance["required"])
        self.assertNotIn("maximum_peak_force_n", acceptance["required"])
        self.assertNotIn("minimum_contact_force_n", acceptance["properties"])
        self.assertNotIn("maximum_peak_force_n", acceptance["properties"])
        self.assertNotIn("require_second_run", acceptance["required"])
        self.assertNotIn("require_second_run", acceptance["properties"])

    def test_causal_rule_accepts_displacement_and_dwell_without_force(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "artimo_robot_execution.schema.json").read_text(
                encoding="utf-8"
            )
        )
        causal_rule = schema["properties"]["causal_rules"]["items"]
        causal_rule_with_defs = {"$defs": schema["$defs"], **causal_rule}
        candidate = {
            "id": "button-opens-lid",
            "trigger_stage": "press-button",
            "source_effect_phase": "lid-open",
            "minimum_displacement": 0.001,
            "minimum_dwell_s": 0.1,
            "effects": [
                {
                    "source_control_index": 0,
                    "joint": "lid-joint",
                    "target": 1.0,
                    "maximum_force_or_torque": 10.0,
                    "tolerance": 0.01,
                }
            ],
        }
        validator = jsonschema.Draft202012Validator(causal_rule_with_defs)
        validator.validate(candidate)

        candidate["minimum_force_n"] = 0.05
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(candidate)


if __name__ == "__main__":
    unittest.main()
