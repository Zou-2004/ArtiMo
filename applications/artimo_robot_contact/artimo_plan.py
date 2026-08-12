"""Canonical, asset-agnostic reading of an ArtiMo ``plan.json``.

The plan is the authoritative object-side action graph.  Every component that
needs to know "which joint must move where, in what order" reads it through this
module so the harness, the finalizer, and the delivery verifier can never
disagree about what the plan asked for.  Nothing here is specific to an asset,
a mechanism, or a task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


#: Plan control modes that name a numeric joint target.
TARGET_MODES = ("joint_position", "spring_return")


def read_plan(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ArtiMo plan root must be a JSON object: {path}")
    if not isinstance(value.get("timeline"), list) or not value["timeline"]:
        raise ValueError(f"ArtiMo plan has no timeline: {path}")
    return value


def control_target(control: dict[str, Any]) -> float | None:
    """Return the numeric joint target a plan control requests, if any.

    ``joint_position`` states its endpoint directly.  ``spring_return`` returns
    to a rest/target position; either spelling is accepted because both appear
    in released ArtiMo plans.  A ``hold_position`` control names no new target
    and yields ``None`` -- holding is a property of the preceding target, not a
    separate one.
    """
    mode = control.get("mode")
    if mode == "joint_position":
        target = control.get("q_target_rad")
    elif mode == "spring_return":
        target = control.get("target_rad", control.get("rest_position"))
    else:
        return None
    return float(target) if isinstance(target, (int, float)) else None


def phase_targets(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every numeric joint request in plan order.

    Consecutive duplicate targets for the same joint are collapsed: a plan phase
    that re-commands a joint to the value it was already asked to reach (a hold
    or settle restated as ``joint_position``) is one physical extremum, not two.
    Collapsing here -- rather than in each caller -- is what keeps the physics
    harness and the verifier in agreement.
    """
    requests: list[dict[str, Any]] = []
    last_target: dict[str, float] = {}
    for phase_index, phase in enumerate(plan.get("timeline", [])):
        if not isinstance(phase, dict):
            continue
        for control_index, control in enumerate(phase.get("controls", [])):
            if not isinstance(control, dict) or not isinstance(control.get("joint"), str):
                continue
            target = control_target(control)
            if target is None:
                continue
            joint = str(control["joint"])
            previous = last_target.get(joint)
            if previous is not None and abs(previous - target) <= 1e-12:
                continue
            last_target[joint] = target
            requests.append(
                {
                    "joint": joint,
                    "target": target,
                    "phase": str(phase.get("name", "")),
                    "phase_index": phase_index,
                    "control_index": control_index,
                    "mode": str(control.get("mode")),
                }
            )
    if not requests:
        raise ValueError("ArtiMo plan contains no numeric joint target request")
    return requests


def timeline_controls(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every timeline control with a stable phase/index identity.

    Unlike :func:`phase_targets`, this table includes duplicate target restates
    and ``hold_position`` controls.  It is the exhaustive checklist used when
    assigning physical motion ownership; a phase can contain controls with
    different executors.
    """
    rows: list[dict[str, Any]] = []
    last_target: dict[str, float] = {}
    for phase_index, phase in enumerate(plan.get("timeline", [])):
        if not isinstance(phase, dict):
            continue
        for control_index, control in enumerate(phase.get("controls", [])):
            if not isinstance(control, dict):
                continue
            joint = str(control.get("joint", ""))
            target = control_target(control)
            introduces_new_target = target is not None and (
                joint not in last_target or abs(last_target[joint] - target) > 1e-12
            )
            rows.append(
                {
                    "source_phase": str(phase.get("name", "")),
                    "phase_index": phase_index,
                    "source_control_index": control_index,
                    "mode": str(control.get("mode", "")),
                    "joint": joint,
                    "target": target,
                    "introduces_new_target": introduces_new_target,
                }
            )
            if target is not None:
                last_target[joint] = target
    if not rows:
        raise ValueError("ArtiMo plan contains no timeline controls")
    return rows


def requested_extrema(plan: dict[str, Any]) -> dict[str, list[float]]:
    """Group :func:`phase_targets` into per-joint ordered extrema."""
    grouped: dict[str, list[float]] = {}
    for request in phase_targets(plan):
        grouped.setdefault(request["joint"], []).append(float(request["target"]))
    return grouped


def phases_by_name(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    phases = {
        str(phase["name"]): phase
        for phase in plan.get("timeline", [])
        if isinstance(phase, dict) and isinstance(phase.get("name"), str)
    }
    if not phases:
        raise ValueError("ArtiMo plan contains no named timeline phases")
    return phases


def phase_joint_target(
    plan: dict[str, Any], phase_name: str, joint: str
) -> float:
    """Return the target a named plan phase requests for one joint.

    Raises if the phase does not exist or does not command that joint, so
    execution data can never quietly invent a target the plan never asked for.
    """
    phase = phases_by_name(plan).get(phase_name)
    if phase is None:
        raise ValueError(f"Plan has no phase named {phase_name!r}")
    for control in phase.get("controls", []):
        if not isinstance(control, dict) or control.get("joint") != joint:
            continue
        target = control_target(control)
        if target is not None:
            return target
    raise ValueError(
        f"Plan phase {phase_name!r} declares no numeric target for joint {joint!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the canonical exhaustive ArtiMo control-ownership checklist."
    )
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {"controls": timeline_controls(read_plan(args.plan.expanduser().resolve()))},
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
