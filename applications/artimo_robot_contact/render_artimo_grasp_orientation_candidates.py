#!/usr/bin/env python3
"""Render one immutable batch of wrist-roll candidates for an ArtiMo stage.

The contact point and approach normal remain fixed.  Each candidate rotates the
declared contact frame only about its local +Z surface-normal axis, then reuses
``visualize_artimo_scene.py`` to show the object and a kinematic-free parallel-
jaw proxy.  This first pass never runs IK.  After visual decisions, the decision
tool runs numerical IK/contact/clearance probes only for visual-valid rolls.
"""
from __future__ import annotations

import argparse
import copy
import concurrent.futures
import json
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

APP_ROOT = Path(__file__).resolve().parent
REPO = APP_ROOT.parents[1]
SCENE_RENDERER = REPO / "tools" / "visualize_artimo_scene.py"
DEFAULT_ROLL_DEGREES = (-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0)


def _normalized_quaternion(values: list[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or float(np.linalg.norm(quaternion)) < 1e-12:
        raise ValueError(f"Invalid XYZW quaternion: {values!r}")
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def roll_contact_quaternion_xyzw(
    base_xyzw: list[float], roll_degrees: float
) -> list[float]:
    """Post-multiply a contact pose by a local +Z wrist roll."""
    base = _normalized_quaternion(base_xyzw)
    half_angle = math.radians(float(roll_degrees)) / 2.0
    local_roll = np.asarray(
        [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)],
        dtype=np.float64,
    )
    answer = _quaternion_multiply_xyzw(base, local_roll)
    answer /= np.linalg.norm(answer)
    answer[np.abs(answer) < 1e-15] = 0.0
    return [float(value) for value in answer]


def _roll_id(roll_degrees: float) -> str:
    sign = "p" if roll_degrees >= 0.0 else "m"
    magnitude = abs(float(roll_degrees))
    if abs(magnitude - round(magnitude)) < 1e-9:
        token = f"{int(round(magnitude)):03d}"
    else:
        token = f"{magnitude:07.3f}".replace(".", "p")
    return f"roll_{sign}{token}"


def covered_stage_indices(execution: dict, stage_index: int) -> list[int]:
    """One orientation decision covers an uninterrupted contact sequence."""
    stage = execution["stages"][stage_index]
    sequence = stage.get("contact_sequence")
    if sequence is None:
        return [stage_index]
    indices = [
        index
        for index, item in enumerate(execution["stages"])
        if item.get("contact_sequence") == sequence
    ]
    if not indices or indices[0] != stage_index:
        raise ValueError(
            "Render a continuous contact_sequence only from its first stage"
        )
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("A contact_sequence must occupy consecutive stages")
    invariant_fields = (
        "interaction",
        "contact_link",
        "contact_pose_link",
        "allowed_robot_contact_links",
        "finger_opening_m",
        "grasp_depth_m",
        "robot_tool_contact_offset_eef_m",
        "contact_acquisition",
    )
    first = stage
    for index in indices[1:]:
        changed = [
            field
            for field in invariant_fields
            if execution["stages"][index].get(field) != first.get(field)
        ]
        if changed:
            raise ValueError(
                f"contact_sequence {sequence!r} changes invariant fields {changed}"
            )
    return indices


def _candidate_summary(scene: dict) -> dict:
    samples = list(scene.get("samples", []))
    clearances = [
        float(sample["minimum_forbidden_clearance_m"])
        for sample in samples
        if sample.get("minimum_forbidden_clearance_m") is not None
    ]
    return {
        "all_samples_ik_success": bool(samples)
        and all(bool(sample["ik_success"]) for sample in samples),
        "all_samples_target_contact_geometry_ready": bool(samples)
        and all(bool(sample["target_contact_geometry_ready"]) for sample in samples),
        "all_samples_forbidden_clearance_passed": bool(samples)
        and all(bool(sample["forbidden_clearance_passed"]) for sample in samples),
        "all_samples_bilateral_physical_contact": bool(samples)
        and all(bool(sample.get("bilateral_physical_contact")) for sample in samples),
        "maximum_ik_position_error_m": max(
            (float(sample["ik_position_error_m"]) for sample in samples),
            default=float("inf"),
        ),
        "maximum_ik_orientation_error_deg": max(
            (float(sample["ik_orientation_error_deg"]) for sample in samples),
            default=float("inf"),
        ),
        "minimum_reported_forbidden_clearance_m": (
            min(clearances) if clearances else None
        ),
        "minimum_near_contact_link_count": min(
            (int(sample["near_contact_link_count"]) for sample in samples),
            default=0,
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_candidate(
    candidate: dict,
    task_spec: Path,
    stage_index: int,
    maximum_target_gap_m: float,
) -> dict:
    command = [
        sys.executable,
        str(SCENE_RENDERER),
        "--task-spec",
        str(task_spec),
        "--execution",
        str(candidate["execution_path"]),
        "--out",
        str(candidate["scene_directory"]),
        "--stage",
        str(stage_index),
        "--maximum-target-gap-m",
        str(float(maximum_target_gap_m)),
        "--orientation-only-no-ik",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (candidate["candidate_directory"] / "renderer.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Scene renderer failed for {candidate['id']}; see "
            f"{candidate['candidate_directory'] / 'renderer.log'}"
        )
    return json.loads(
        (candidate["scene_directory"] / "scene.json").read_text(encoding="utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument(
        "--roll-deg",
        type=float,
        nargs="+",
        default=list(DEFAULT_ROLL_DEGREES),
        help="One immutable batch of local +Z wrist rolls in degrees.",
    )
    parser.add_argument("--maximum-target-gap-m", type=float, default=0.006)
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Independent candidate renderers to run concurrently (default: 4).",
    )
    args = parser.parse_args()

    execution = json.loads(args.execution.read_text(encoding="utf-8"))
    if not 0 <= args.stage < len(execution.get("stages", [])):
        raise IndexError(
            f"Stage index {args.stage} outside execution stages "
            f"[0, {len(execution.get('stages', []))})"
        )
    stage = execution["stages"][args.stage]
    covered_indices = covered_stage_indices(execution, int(args.stage))
    base_quaternion = list(stage["contact_pose_link"]["rotation_xyzw"])
    rolls = [float(value) for value in args.roll_deg]
    if len({_roll_id(value) for value in rolls}) != len(rolls):
        raise ValueError("--roll-deg contains duplicate candidate angles")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")

    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = []
    for roll_degrees in rolls:
        candidate_id = _roll_id(roll_degrees)
        candidate_directory = output / "candidates" / candidate_id
        scene_directory = candidate_directory / "scene"
        candidate_directory.mkdir(parents=True, exist_ok=True)
        candidate_execution = copy.deepcopy(execution)
        candidate_quaternion = roll_contact_quaternion_xyzw(
            base_quaternion, roll_degrees
        )
        for covered_index in covered_indices:
            candidate_execution["stages"][covered_index]["contact_pose_link"][
                "rotation_xyzw"
            ] = candidate_quaternion
        execution_path = candidate_directory / "execution.json"
        execution_path.write_text(
            json.dumps(candidate_execution, indent=2) + "\n", encoding="utf-8"
        )
        candidates.append(
            {
                "id": candidate_id,
                "roll_degrees": roll_degrees,
                "contact_rotation_xyzw": candidate_quaternion,
                "execution": str(execution_path),
                "scene_image": str(scene_directory / "scene.png"),
                "candidate_directory": candidate_directory,
                "scene_directory": scene_directory,
                "execution_path": execution_path,
            }
        )

    resolved_task_spec = args.task_spec.expanduser().resolve()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(int(args.jobs), len(candidates))
    ) as executor:
        scene_reports = list(
            executor.map(
                lambda candidate: _render_candidate(
                    candidate,
                    resolved_task_spec,
                    int(args.stage),
                    float(args.maximum_target_gap_m),
                ),
                candidates,
            )
        )
    for candidate, scene_report in zip(candidates, scene_reports):
        preview = scene_report.get("orientation_preview", {})
        if preview.get("ik_was_run") is not False:
            raise RuntimeError(
                f"Visual renderer unexpectedly ran IK for {candidate['id']}"
            )
        relative_images = preview.get("images")
        if not isinstance(relative_images, dict) or len(relative_images) != 4:
            raise RuntimeError(
                f"Scene renderer did not emit four separate orientation views for "
                f"{candidate['id']}"
            )
        candidate["orientation_images"] = {
            panel: str(candidate["scene_directory"] / filename)
            for panel, filename in relative_images.items()
        }
        candidate["orientation_image_sha256"] = {
            panel: _sha256(candidate["scene_directory"] / filename)
            for panel, filename in relative_images.items()
        }
        candidate["execution_sha256"] = _sha256(candidate["execution_path"])

    serializable_candidates = []
    for candidate in candidates:
        serializable = dict(candidate)
        for internal_key in (
            "candidate_directory",
            "scene_directory",
            "execution_path",
        ):
            del serializable[internal_key]
        serializable_candidates.append(serializable)
    report = {
        "schema_version": 4,
        "task_spec": str(args.task_spec.expanduser().resolve()),
        "execution_template": str(args.execution.expanduser().resolve()),
        "stage_index": int(args.stage),
        "stage_id": str(stage["id"]),
        "covered_stage_indices": covered_indices,
        "covered_stage_ids": [
            str(execution["stages"][index]["id"]) for index in covered_indices
        ],
        "contact_sequence": stage.get("contact_sequence"),
        "interaction": str(stage["interaction"]),
        "contact_link": str(stage["contact_link"]),
        "contact_translation_m": list(stage["contact_pose_link"]["translation_m"]),
        "base_contact_rotation_xyzw": base_quaternion,
        "execution_template_sha256": _sha256(args.execution.expanduser().resolve()),
        "roll_axis": "contact_local_+Z_surface_normal",
        "maximum_target_gap_m": float(args.maximum_target_gap_m),
        "visual_render_ik_was_run": False,
        "rendering_policy": "separate_candidate_and_view_files_no_composite",
        "selection": None,
        "selection_policy": (
            "Agent must open all four separate images for every candidate and mark "
            "every roll visual-valid or visual-invalid before any IK. A visual-"
            "invalid roll is a hard exclusion and cannot enter an IK/contact probe, "
            "placement, trajectory, transit, or rollout. Apply decisions only through "
            "applications/artimo_robot_contact/apply_artimo_grasp_orientation_decisions.py."
        ),
        "candidates": serializable_candidates,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    decision_template = {
        "schema_version": 1,
        "report_sha256": _sha256(output / "report.json"),
        "stage_id": str(stage["id"]),
        "decisions": [
            {
                "id": candidate["id"],
                "visual_status": None,
                "reason": "",
                "reviewed_images": list(candidate["orientation_images"].values()),
            }
            for candidate in serializable_candidates
        ],
    }
    (output / "visual_decisions.template.json").write_text(
        json.dumps(decision_template, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
