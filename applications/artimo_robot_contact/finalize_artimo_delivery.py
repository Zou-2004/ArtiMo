#!/usr/bin/env python3
"""Publish a generic three-file delivery from two clean physical rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
FINAL_NAMES = {"video.mp4", "grasp.json", "result.json"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hex(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be 64 lowercase hex characters")


def _native(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    names = {path.name for path in directory.iterdir()} if directory.is_dir() else set()
    if names != FINAL_NAMES:
        raise ValueError(f"Native rollout must contain exactly {sorted(FINAL_NAMES)}: {directory}")
    result = _read(directory / "result.json")
    grasp = _read(directory / "grasp.json")
    return result, grasp


def _visual_qa(path: Path) -> dict[str, Any]:
    value = _read(path)
    required_true = (
        "no_visible_interpenetration",
        "physical_contact_visible",
        "requested_motion_visible",
        "no_rendering_artifacts",
    )
    # Visual review is evidence, not an export gate.  Preserve measured values
    # even when false so a full rollout is always available for human review.
    return {
        "sample_rate_fps": float(value.get("sample_rate_fps", 0.0)),
        **{key: bool(value.get(key, False)) for key in required_true},
    }


def _delivery_passed(
    first: dict[str, Any],
    second: dict[str, Any],
    visual: dict[str, Any],
    reproducibility: dict[str, bool],
) -> bool:
    """Publish every video, but never upgrade failed native evidence."""
    required_visual = (
        "no_visible_interpenetration",
        "physical_contact_visible",
        "requested_motion_visible",
        "no_rendering_artifacts",
    )
    return bool(
        first.get("passed", False)
        and second.get("passed", False)
        and all(bool(visual.get(key, False)) for key in required_visual)
        and all(reproducibility.values())
    )


def finalize(
    task_spec: Path,
    first_dir: Path,
    second_dir: Path,
    visual_qa_path: Path,
    output_dir: Path,
    handoff_lock: str,
    release_lock: str,
) -> dict[str, Any]:
    _require_hex(handoff_lock, "handoff lock")
    _require_hex(release_lock, "release lock")
    task = _read(task_spec)
    first, first_grasp = _native(first_dir)
    second, second_grasp = _native(second_dir)
    visual = _visual_qa(visual_qa_path)
    first_grasp_hash = _sha256(first_dir / "grasp.json")
    second_grasp_hash = _sha256(second_dir / "grasp.json")
    same_execution = first_grasp_hash == second_grasp_hash and first_grasp == second_grasp
    first_schedule = first["physical"]["robot_command_schedule_sha256"]
    second_schedule = second["physical"]["robot_command_schedule_sha256"]
    schedule_match = first_schedule == second_schedule
    contact_match = first["physical"]["contacts"] == second["physical"]["contacts"]
    motion_match = first["physical"]["joint_motion"] == second["physical"]["joint_motion"]
    simulation_input_match = first.get("inputs") == second.get("inputs")
    # Reproducibility comparisons are recorded below but never suppress the
    # first physically executed video.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing non-empty final output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(first_dir / "video.mp4", output_dir / "video.mp4")
    shutil.copy2(first_dir / "grasp.json", output_dir / "grasp.json")

    contacts = first["physical"]["contacts"]
    negative_contacts = first["negative_control"]["contacts"]
    manifest = {
        "schema_version": 2,
        "task_spec_sha256": _sha256(task_spec),
        "handoff_lock_sha256": handoff_lock,
        "release_lock_sha256": release_lock,
        "execution_plan_sha256": _sha256(output_dir / "grasp.json"),
        "physics_engine": "PyBullet",
        "physical_only_video": True,
        "object_trajectory_replay": False,
        "object_joint_resets_after_initialization": int(
            first["physical"]["object_joint_resets_after_initialization"]
        ),
        "fixed_constraint_count": int(first["physical"]["maximum_runtime_constraint_count"]),
        "robot_command_schedule_sha256": first_schedule,
        "seeds": first_grasp["seeds"],
        "physical": {
            "contacts": contacts,
            "joint_motion": first["physical"]["joint_motion"],
        },
        "negative_control": {
            "same_robot_command_schedule": (
                first_schedule == first["negative_control"]["robot_command_schedule_sha256"]
            ),
            "target_contact_observations": sum(
                int(item["target_contact_observations"]) for item in negative_contacts
            ),
            "causal_triggers": int(first["negative_control"]["causal_triggers"]),
            "requested_joint_motion_remained_initial": bool(
                first["negative_control"]["requested_joint_motion_remained_initial"]
            ),
        },
        "visual_qa": visual,
        "second_run": {
            "same_execution_plan": same_execution,
            "robot_command_schedule_match": schedule_match,
            "contact_metrics_match": contact_match,
            "joint_motion_match": motion_match,
        },
    }
    reproducibility_checks = {
        "same_execution_plan": same_execution,
        "robot_command_schedule_match": schedule_match,
        "contact_metrics_match": contact_match,
        "joint_motion_match": motion_match,
        "simulation_input_match": simulation_input_match,
    }
    manifest["second_run"]["simulation_input_match"] = simulation_input_match
    result = {
        "schema_version": 2,
        "passed": _delivery_passed(
            first, second, visual, reproducibility_checks
        ),
        "reproducibility": manifest,
        "native_first": first,
        "native_second": second,
        "publication": {
            "task_id": task["task_id"],
            "video_sha256": _sha256(output_dir / "video.mp4"),
            "grasp_sha256": _sha256(output_dir / "grasp.json"),
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--visual-qa", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--handoff-lock", required=True)
    parser.add_argument("--release-lock", required=True)
    args = parser.parse_args()
    try:
        result = finalize(
            args.task_spec.expanduser().resolve(),
            args.first.expanduser().resolve(),
            args.second.expanduser().resolve(),
            args.visual_qa.expanduser().resolve(),
            args.out.expanduser().resolve(),
            args.handoff_lock,
            args.release_lock,
        )
        print(json.dumps(result["publication"], indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"ArtiMo delivery finalization failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
