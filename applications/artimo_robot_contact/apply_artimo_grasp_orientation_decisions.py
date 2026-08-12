#!/usr/bin/env python3
"""Hard-filter wrist rolls by reviewed visual semantics before planning.

The renderer emits separate immutable images for every candidate and view.  An
agent records a decision for every roll.  This tool verifies that all exact
images were reviewed, removes every visual-invalid candidate, applies numerical
gates only to the remaining visual-valid set, and emits the sole execution that
may continue to placement/IK/transit.  It never knows an asset or task name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from render_artimo_grasp_orientation_candidates import _candidate_summary


APP_ROOT = Path(__file__).resolve().parent
REPO = APP_ROOT.parents[1]
SCENE_RENDERER = APP_ROOT / "visualize_artimo_scene.py"


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


def _numerically_eligible(
    candidate: dict[str, Any], *, require_bilateral_contact: bool
) -> bool:
    summary = candidate.get("summary")
    if not isinstance(summary, dict):
        return False
    required = (
        "all_samples_ik_success",
        "all_samples_target_contact_geometry_ready",
        "all_samples_forbidden_clearance_passed",
    )
    return all(bool(summary.get(key)) for key in required) and (
        not require_bilateral_contact
        or bool(summary.get("all_samples_bilateral_physical_contact"))
    )


def _rank(candidate: dict[str, Any]) -> tuple[float, float, float, str]:
    summary = candidate["summary"]
    clearance = summary.get("minimum_reported_forbidden_clearance_m")
    numeric_clearance = math.inf if clearance is None else float(clearance)
    return (
        -numeric_clearance,
        float(summary["maximum_ik_position_error_m"]),
        float(summary["maximum_ik_orientation_error_deg"]),
        str(candidate["id"]),
    )


def _probe_candidate(
    candidate: dict[str, Any], report: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Run numerical evidence only after this roll passed visual review."""
    probe_directory = output / "numerical_probes" / str(candidate["id"])
    command = [
        sys.executable,
        str(SCENE_RENDERER),
        "--task-spec",
        str(report["task_spec"]),
        "--execution",
        str(candidate["execution"]),
        "--out",
        str(probe_directory),
        "--stage",
        str(int(report["stage_index"])),
        "--maximum-target-gap-m",
        str(float(report.get("maximum_target_gap_m", 0.006))),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    probe_directory.mkdir(parents=True, exist_ok=True)
    (probe_directory / "probe.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Numerical probe failed for {candidate['id']}; see "
            f"{probe_directory / 'probe.log'}"
        )
    scene_path = probe_directory / "scene.json"
    scene = _read_json(scene_path)
    summary = _candidate_summary(scene)
    summary["probe_scene"] = str(scene_path)
    summary["probe_scene_sha256"] = _sha256(scene_path)
    return summary


def apply_decisions(
    report_path: Path,
    decisions_path: Path,
    output: Path,
    prior_gate_path: Path | None = None,
    probe_candidate: Callable[
        [dict[str, Any], dict[str, Any], Path], dict[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    report_path = report_path.expanduser().resolve()
    decisions_path = decisions_path.expanduser().resolve()
    report = _read_json(report_path)
    decisions = _read_json(decisions_path)
    if int(report.get("schema_version", 0)) != 4:
        raise ValueError("Orientation report must use schema_version 4")
    if report.get("visual_render_ik_was_run") is not False:
        raise ValueError("Visual orientation report must prove that IK was not run")
    if int(decisions.get("schema_version", 0)) != 1:
        raise ValueError("Visual decisions must use schema_version 1")
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

    visual_valid = [
        by_id[candidate_id]
        for candidate_id, row in decision_by_id.items()
        if row["visual_status"] == "valid"
    ]
    if not visual_valid:
        raise ValueError("No visually valid grasp orientation remains")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe = _probe_candidate if probe_candidate is None else probe_candidate
    numerically_probed: list[dict[str, Any]] = []
    for candidate in visual_valid:
        candidate_copy = dict(candidate)
        candidate_copy["summary"] = probe(candidate_copy, report, output)
        numerically_probed.append(candidate_copy)
    interaction = str(report.get("interaction", "explicit_ideal_feasibility"))
    require_bilateral_contact = interaction == "explicit_ideal_feasibility"
    eligible = [
        candidate
        for candidate in numerically_probed
        if _numerically_eligible(
            candidate, require_bilateral_contact=require_bilateral_contact
        )
    ]
    if not eligible:
        contact_requirement = (
            "bilateral physical contact, " if require_bilateral_contact else ""
        )
        raise ValueError(
            "Visually valid orientations all fail IK, "
            f"{contact_requirement}target contact geometry, or clearance"
        )
    chosen = min(eligible, key=_rank)
    chosen_execution = Path(chosen["execution"]).expanduser().resolve()
    if not chosen_execution.is_file():
        raise ValueError(f"Chosen execution does not exist: {chosen_execution}")
    if chosen.get("execution_sha256") != _sha256(chosen_execution):
        raise ValueError("Chosen candidate execution changed after rendering")

    prior_stages: list[dict[str, Any]] = []
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
        "numerically_evaluated_candidate_ids": sorted(
            str(item["id"]) for item in numerically_probed
        ),
        "numerical_probe_policy": "visual_valid_only_after_hard_exclusion",
        "interaction": interaction,
        "bilateral_contact_required": require_bilateral_contact,
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
                    "reference; full placement validates the complete sequence"
                ),
            }
        )
    gate = {
        "schema_version": 1,
        "policy": "visual_invalid_candidates_hard_excluded_before_planning",
        "execution": str(execution_out),
        "execution_sha256": _sha256(execution_out),
        "stages": prior_stages + covered_stage_gates,
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
