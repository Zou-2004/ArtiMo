#!/usr/bin/env python3
"""Hard-filter wrist rolls by reviewed visual semantics before planning.

The renderer emits separate immutable images for every candidate and view.  An
agent records a decision and visual priority for every roll.  This tool verifies
that all exact images were reviewed, removes every visual-invalid candidate,
and records the remaining priority order without running IK.  Full-path IK is
deferred to placement, where the real robot base is known; placement jointly
scores every visual-valid contact choice across all manipulation blocks.  The
visual priority is only a deterministic tie-break after whole-task geometric
feasibility.  Nothing here knows an asset or task name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def apply_decisions(
    report_path: Path,
    decisions_path: Path,
    output: Path,
    prior_gate_path: Path | None = None,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    decisions_path = decisions_path.expanduser().resolve()
    report = _read_json(report_path)
    decisions = _read_json(decisions_path)
    if int(report.get("schema_version", 0)) != 4:
        raise ValueError("Orientation report must use schema_version 4")
    if report.get("visual_render_ik_was_run") is not False:
        raise ValueError("Visual orientation report must prove that IK was not run")
    if int(decisions.get("schema_version", 0)) != 4:
        raise ValueError(
            "Visual decisions must use schema_version 4 with angle-only "
            "classification; "
            "rerender legacy orientation reports"
        )
    if decisions.get("report_sha256") != _sha256(report_path):
        raise ValueError("Visual decisions report_sha256 does not match report.json")
    if decisions.get("stage_id") != report.get("stage_id"):
        raise ValueError("Visual decisions stage_id does not match report.json")

    covered_stage_ids = report.get("covered_stage_ids", [report.get("stage_id")])
    covered_stage_indices = report.get(
        "covered_stage_indices", [report.get("stage_index")]
    )
    if (
        not isinstance(covered_stage_ids, list)
        or not covered_stage_ids
        or not all(isinstance(value, str) and value for value in covered_stage_ids)
    ):
        raise ValueError("covered_stage_ids must be a non-empty list of stage ids")
    if (
        not isinstance(covered_stage_indices, list)
        or len(covered_stage_indices) != len(covered_stage_ids)
        or not all(isinstance(value, int) for value in covered_stage_indices)
    ):
        raise ValueError("covered_stage_indices must align with covered_stage_ids")
    if covered_stage_ids[0] != report.get("stage_id"):
        raise ValueError("The rendered stage must be first in covered_stage_ids")

    candidates = report.get("candidates")
    rows = decisions.get("decisions")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Orientation report has no candidates")
    if any("summary" in candidate for candidate in candidates):
        raise ValueError(
            "Visual orientation report may not contain pre-decision numerical summaries"
        )
    if not isinstance(rows, list):
        raise ValueError("Visual decisions decisions[] is required")
    interaction = str(report.get("interaction", "explicit_ideal_feasibility"))
    by_id = {str(item.get("id")): item for item in candidates}
    decision_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError("Every visual decision needs a candidate id")
        candidate_id = row["id"]
        if candidate_id in decision_by_id:
            raise ValueError(f"Duplicate visual decision for {candidate_id}")
        if candidate_id not in by_id:
            raise ValueError(f"Unknown visual decision candidate {candidate_id}")
        status = row.get("visual_status")
        if status not in {"valid", "invalid"}:
            raise ValueError(
                f"{candidate_id}.visual_status must be 'valid' or 'invalid'"
            )
        reason = row.get("reason")
        if not isinstance(reason, str) or len(reason.strip()) < 8:
            raise ValueError(f"{candidate_id}.reason must explain the visual decision")
        priority = row.get("visual_priority")
        angle_status = row.get("angle_status")
        if angle_status not in {"valid", "invalid"}:
            raise ValueError(
                f"{candidate_id}.angle_status must be 'valid' or 'invalid'"
            )
        if status == "valid":
            if angle_status != "valid":
                raise ValueError(
                    f"{candidate_id} cannot be visual-valid when angle_status is invalid"
                )
            if isinstance(priority, bool) or not isinstance(priority, int) or priority < 1:
                raise ValueError(
                    f"{candidate_id}.visual_priority must be a positive integer "
                    "for every visual-valid roll"
                )
        else:
            if priority is not None:
                raise ValueError(
                    f"{candidate_id}.visual_priority must be null for a visual-invalid roll"
                )
            if angle_status == "valid":
                raise ValueError(
                    f"{candidate_id}.visual_status and angle_status must agree"
                )
        expected_images = by_id[candidate_id].get("orientation_images")
        if not isinstance(expected_images, dict) or len(expected_images) != 4:
            raise ValueError(f"{candidate_id} does not declare four separate views")
        reviewed = row.get("reviewed_images")
        if not isinstance(reviewed, list) or set(reviewed) != set(expected_images.values()):
            raise ValueError(
                f"{candidate_id}.reviewed_images must list its exact four separate views"
            )
        for panel, image_value in expected_images.items():
            image_path = Path(image_value).expanduser().resolve()
            if not image_path.is_file():
                raise ValueError(f"Missing reviewed image {panel}: {image_path}")
            expected_hash = by_id[candidate_id]["orientation_image_sha256"].get(panel)
            if expected_hash != _sha256(image_path):
                raise ValueError(f"Reviewed image changed after rendering: {image_path}")
        decision_by_id[candidate_id] = row
    if set(decision_by_id) != set(by_id):
        missing = sorted(set(by_id) - set(decision_by_id))
        raise ValueError(f"Every candidate requires a visual decision; missing {missing}")

    visual_valid = sorted(
        [
            by_id[candidate_id]
            for candidate_id, row in decision_by_id.items()
            if row["visual_status"] == "valid"
        ],
        key=lambda candidate: int(
            decision_by_id[str(candidate["id"])]["visual_priority"]
        ),
    )
    if not visual_valid:
        raise ValueError("No visually valid grasp orientation remains")
    priorities = [
        int(decision_by_id[str(candidate["id"])]["visual_priority"])
        for candidate in visual_valid
    ]
    if priorities != list(range(1, len(visual_valid) + 1)):
        raise ValueError(
            "Visual-valid visual_priority values must be unique and contiguous "
            "starting at 1"
        )

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    chosen = visual_valid[0]
    require_bilateral_contact = interaction == "explicit_ideal_feasibility"
    for candidate in visual_valid:
        candidate_execution = Path(candidate["execution"]).expanduser().resolve()
        if not candidate_execution.is_file():
            raise ValueError(f"Candidate execution does not exist: {candidate_execution}")
        if candidate.get("execution_sha256") != _sha256(candidate_execution):
            raise ValueError(
                f"Candidate {candidate['id']} execution changed after rendering"
            )
    chosen_execution = Path(chosen["execution"]).expanduser().resolve()
    if not chosen_execution.is_file():
        raise ValueError(f"Chosen execution does not exist: {chosen_execution}")
    if chosen.get("execution_sha256") != _sha256(chosen_execution):
        raise ValueError("Chosen candidate execution changed after rendering")

    prior_stages: list[dict[str, Any]] = []
    prior_candidate_groups: list[dict[str, Any]] = []
    if prior_gate_path is not None:
        prior_gate_path = prior_gate_path.expanduser().resolve()
        prior = _read_json(prior_gate_path)
        if int(prior.get("schema_version", 0)) != 1:
            raise ValueError("Prior orientation gate must use schema_version 1")
        if prior.get("execution_sha256") != report.get("execution_template_sha256"):
            raise ValueError(
                "Prior gate execution does not match this orientation report template"
            )
        prior_stages = list(prior.get("stages", []))
        prior_candidate_groups = list(prior.get("placement_candidate_groups", []))
    duplicate_stage_ids = sorted(
        set(covered_stage_ids)
        & {str(item.get("stage_id")) for item in prior_stages}
    )
    if duplicate_stage_ids:
        raise ValueError(f"Stages {duplicate_stage_ids} are already visually gated")

    execution_out = output / "execution.json"
    shutil.copyfile(chosen_execution, execution_out)
    primary_stage_gate = {
        "stage_index": int(covered_stage_indices[0]),
        "stage_id": str(covered_stage_ids[0]),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
        "decisions": str(decisions_path),
        "decisions_sha256": _sha256(decisions_path),
        "selected_candidate_id": str(chosen["id"]),
        "selected_roll_degrees": float(chosen["roll_degrees"]),
        "visual_valid_candidate_ids": sorted(str(item["id"]) for item in visual_valid),
        "visual_invalid_candidate_ids": sorted(
            candidate_id
            for candidate_id, row in decision_by_id.items()
            if row["visual_status"] == "invalid"
        ),
        "visual_priority_candidate_ids": [str(item["id"]) for item in visual_valid],
        "numerically_evaluated_candidate_ids": [],
        "numerical_probe_policy": (
            "deferred_to_joint_whole_task_placement_at_actual_robot_bases"
        ),
        "interaction": interaction,
        "bilateral_contact_required": require_bilateral_contact,
        "agent_decision_scope": "wrist_angle_only",
        "grasp_depth_owner": "application_rule_based_dense_search",
        "nominal_contact_geometry_frozen_by_visual_gate": False,
        "selected_angle_status": str(
            decision_by_id[str(chosen["id"])]["angle_status"]
        ),
        "contact_offset_under_review": report["contact_offset_under_review"],
        "maximum_target_gap_m": float(report["maximum_target_gap_m"]),
        "contact_sequence": report.get("contact_sequence"),
        "covered_stage_ids": list(covered_stage_ids),
        "selected_visual_reason": decision_by_id[str(chosen["id"])]["reason"],
    }
    covered_stage_gates = [primary_stage_gate]
    for stage_index, stage_id in zip(
        covered_stage_indices[1:], covered_stage_ids[1:]
    ):
        covered_stage_gates.append(
            {
                "stage_index": int(stage_index),
                "stage_id": str(stage_id),
                "inherited_from_stage_id": str(covered_stage_ids[0]),
                "contact_sequence": report.get("contact_sequence"),
                "selected_candidate_id": str(chosen["id"]),
                "selected_roll_degrees": float(chosen["roll_degrees"]),
                "numerically_evaluated_candidate_ids": [],
                "numerical_probe_policy": (
                    "continuous_contact_inherits_acquisition_orientation_and_arm_"
                    "reference; joint whole-task placement validates the complete sequence"
                ),
            }
        )
    gate = {
        "schema_version": 1,
        "policy": "visual_invalid_candidates_hard_excluded_before_planning",
        "execution": str(execution_out),
        "execution_sha256": _sha256(execution_out),
        "stages": prior_stages + covered_stage_gates,
        "placement_candidate_groups": prior_candidate_groups
        + [
            {
                "stage_ids": list(covered_stage_ids),
                "candidates": [
                    {
                        "id": str(candidate["id"]),
                        "visual_priority": int(
                            decision_by_id[str(candidate["id"])]["visual_priority"]
                        ),
                        "roll_degrees": float(candidate["roll_degrees"]),
                        "execution": str(Path(candidate["execution"]).resolve()),
                        "execution_sha256": str(candidate["execution_sha256"]),
                    }
                    for candidate in visual_valid
                ],
            }
        ],
    }
    gate_path = output / "orientation_gate.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    return {**gate, "orientation_gate": str(gate_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-gate", type=Path)
    args = parser.parse_args()
    try:
        answer = apply_decisions(
            args.report, args.decisions, args.out, args.prior_gate
        )
        print(json.dumps(answer, indent=2))
        return 0
    except Exception as exc:
        print(f"Orientation decision application failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
