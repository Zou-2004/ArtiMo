#!/usr/bin/env python3
"""Prepare or launch the asset-agnostic ArtiMo robot-contact application.

This entry point contains no asset registry, contact pose, task-specific gain,
or simulator route.  Every asset enters through the same URDF, ArtiMo plan,
optional initial-state trajectory, robot, and natural-language goal interface.
Per-task geometry is data in the generated task/execution plans, never Python
branching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
RUNNER = APP_ROOT / "run_agent_task.py"
DEFAULT_ROBOT_URDF = APP_ROOT / "assets" / "panda" / "panda.urdf"
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _repo_file(value: Path, label: str) -> str:
    path = value.expanduser().resolve()
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return str(relative)


def _repo_output(value: Path) -> str:
    path = value.expanduser().resolve()
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"output must be inside the repository: {path}") from exc
    if path == REPO_ROOT:
        raise ValueError("output must be a repository subdirectory")
    return str(relative)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _derived_task_id(urdf: Path) -> str:
    stem = urdf.expanduser().resolve().parent.name.lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", stem).strip("-._")
    return value or "artimo-robot-contact"


def _derived_task_description(plan_path: Path) -> str:
    plan = json.loads(plan_path.expanduser().resolve().read_text(encoding="utf-8"))
    phases = [
        str(phase.get("name"))
        for phase in plan.get("timeline", [])
        if isinstance(phase, dict) and phase.get("name")
    ]
    suffix = ", ".join(phases) if phases else "all declared controls"
    return f"Execute the ArtiMo plan through physical robot contact: {suffix}."


def _task_spec(args: argparse.Namespace) -> dict[str, Any]:
    task_id = args.task_id or _derived_task_id(args.urdf)
    if not TASK_ID_RE.fullmatch(task_id):
        raise ValueError("task-id must match [a-z0-9][a-z0-9._-]*")
    inputs: dict[str, Any] = {
        "urdf": _repo_file(args.urdf, "urdf"),
        "plan": _repo_file(args.plan, "plan"),
        "robot_urdf": _repo_file(args.robot_urdf, "robot-urdf"),
    }
    if args.trajectory is not None:
        inputs["trajectory"] = _repo_file(args.trajectory, "trajectory")
    if args.physics_urdf is not None:
        inputs["physics_urdf"] = _repo_file(args.physics_urdf, "physics-urdf")
    if args.supporting_file:
        inputs["supporting_files"] = [
            _repo_file(path, "supporting-file") for path in args.supporting_file
        ]
    return {
        "schema_version": 2,
        "task_id": task_id,
        "task_description": (
            args.task_description or _derived_task_description(args.plan)
        ),
        "inputs": inputs,
        "output_dir": _repo_output(
            args.out or (REPO_ROOT / "outputs" / "robot_contact" / task_id)
        ),
        "acceptance": {
            # Verified contact gates plan-authoritative joint actuation without
            # creating a runtime grasp constraint. This invariant applies to
            # every asset and is not selected by the task agent.
            "require_zero_fixed_constraints": True,
            "minimum_joint_motion_ratio": args.minimum_joint_motion_ratio,
            "minimum_continuous_contact_s": args.minimum_continuous_contact_s,
            "visual_review_fps": args.visual_review_fps,
            "retain_debug_on_success": args.keep_debug,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task-id")
    parser.add_argument("--task-description")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--physics-urdf", type=Path)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Optional trajectory.jsonl used only for its first-frame joint state",
    )
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--supporting-file", type=Path, action="append")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--minimum-joint-motion-ratio", type=float, default=0.90)
    parser.add_argument("--minimum-continuous-contact-s", type=float, default=0.25)
    parser.add_argument("--visual-review-fps", type=float, default=5.0)
    parser.add_argument("--keep-debug", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the neutral handoff prompt and lock without launching an agent",
    )
    parser.add_argument(
        "--agent-command",
        help=(
            "Agent CLI command that reads the handoff prompt from stdin. "
            "Omit it to prepare a handoff for any interactive agent."
        ),
    )
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()

    try:
        spec = _task_spec(args)
        digest = _canonical_hash(spec)[:16]
        spec_dir = REPO_ROOT / ".artimo-runs" / "generated-specs" / spec["task_id"]
        spec_dir.mkdir(parents=True, exist_ok=True)
        spec_path = spec_dir / f"{digest}.json"
        spec_path.write_text(
            json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        command = [sys.executable, str(RUNNER), "--spec", str(spec_path)]
        if args.prepare_only or not args.agent_command:
            command.append("--prepare-only")
        else:
            command.extend(["--agent-command", args.agent_command])
        if args.allow_existing_output:
            command.append("--allow-existing-output")
        return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
    except Exception as exc:
        print(f"ArtiMo harness failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
