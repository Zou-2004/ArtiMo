#!/usr/bin/env python3
from __future__ import annotations

from typing import Iterable


VERIFIER_VLM = "vlm"
VALID_VERIFIERS = {VERIFIER_VLM}
BASE_MOTION_MODES = {"base_velocity", "base_velocity_decay", "base", "base_decay"}


def normalize_verifier_name(value) -> str | None:
    s = str(value or "").strip().lower()
    if s in VALID_VERIFIERS:
        return s
    return None


def control_mode(ctrl: dict | None) -> str:
    if not isinstance(ctrl, dict):
        return ""
    return str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()


def control_joint_names(ctrl: dict | None) -> list[str]:
    if not isinstance(ctrl, dict):
        return []
    out: list[str] = []
    joint = ctrl.get("joint")
    if isinstance(joint, str) and joint.strip():
        out.append(joint.strip())
    joints = ctrl.get("joints")
    if isinstance(joints, list):
        for value in joints:
            s = str(value or "").strip()
            if s and s not in out:
                out.append(s)
    return out


def iter_plan_controls(plan: dict | None):
    timeline = (plan or {}).get("timeline") or []
    for seg_idx, seg in enumerate(timeline):
        if not isinstance(seg, dict):
            continue
        for ctrl_idx, ctrl in enumerate(seg.get("controls") or []):
            if not isinstance(ctrl, dict):
                continue
            yield seg_idx, ctrl_idx, ctrl


def collect_referenced_timeline_joints(plan: dict | None) -> list[str]:
    out: list[str] = []
    for _seg_idx, _ctrl_idx, ctrl in iter_plan_controls(plan):
        for joint_name in control_joint_names(ctrl):
            if joint_name not in out:
                out.append(joint_name)
    return out


def plan_has_base_motion(plan: dict | None) -> bool:
    for _seg_idx, _ctrl_idx, ctrl in iter_plan_controls(plan):
        if control_mode(ctrl) in BASE_MOTION_MODES:
            return True
    return False


def wheel_joints_from_plan(plan: dict | None) -> list[str]:
    nl_parse = (plan or {}).get("nl_parse") or {}
    out: list[str] = []
    for value in nl_parse.get("wheel_joints") or []:
        s = str(value or "").strip()
        if s and s not in out:
            out.append(s)
    return out


def is_wheel_transport_plan(plan: dict | None) -> bool:
    return bool(wheel_joints_from_plan(plan)) and plan_has_base_motion(plan)


def build_default_joint_verifiers(plan: dict | None) -> dict[str, str]:
    referenced_joints = collect_referenced_timeline_joints(plan)
    return {joint_name: VERIFIER_VLM for joint_name in referenced_joints}


def normalize_joint_verifiers(plan: dict | None) -> dict[str, str]:
    referenced_joints = collect_referenced_timeline_joints(plan)
    defaults = build_default_joint_verifiers(plan)
    nl_parse = (plan or {}).get("nl_parse") or {}
    raw = nl_parse.get("joint_verifiers")
    if not isinstance(raw, dict):
        return defaults
    out: dict[str, str] = {}
    for joint_name in referenced_joints:
        verifier = normalize_verifier_name(raw.get(joint_name))
        out[joint_name] = verifier or defaults.get(joint_name, VERIFIER_VLM)
    return out


def ensure_joint_verifiers(plan: dict) -> dict:
    if not isinstance(plan, dict):
        return plan
    nl_parse = dict(plan.get("nl_parse") or {})
    verifiers = normalize_joint_verifiers(plan)
    nl_parse["joint_verifiers"] = verifiers
    plan["nl_parse"] = nl_parse
    return plan


def joint_verifier(plan: dict | None, joint_name: str, default: str = VERIFIER_VLM) -> str:
    name = str(joint_name or "").strip()
    if not name:
        return default
    verifiers = normalize_joint_verifiers(plan)
    return verifiers.get(name, default)


def control_verifier_owners(plan: dict | None, ctrl: dict | None) -> set[str]:
    owners: set[str] = set()
    for joint_name in control_joint_names(ctrl):
        owners.add(joint_verifier(plan, joint_name))
    return owners


def control_is_mixed_owner(plan: dict | None, ctrl: dict | None) -> bool:
    owners = control_verifier_owners(plan, ctrl)
    return len(owners) > 1


def control_owner(plan: dict | None, ctrl: dict | None) -> str | None:
    owners = control_verifier_owners(plan, ctrl)
    if len(owners) != 1:
        return None
    return next(iter(owners))


def segment_has_base_motion(seg: dict | None) -> bool:
    if not isinstance(seg, dict):
        return False
    for ctrl in seg.get("controls") or []:
        if control_mode(ctrl) in BASE_MOTION_MODES:
            return True
    return False


def iter_segments(plan: dict | None):
    timeline = (plan or {}).get("timeline") or []
    for seg_idx, seg in enumerate(timeline):
        if isinstance(seg, dict):
            yield seg_idx, seg


def segment_joint_owners(plan: dict | None, seg: dict | None) -> set[str]:
    owners: set[str] = set()
    if not isinstance(seg, dict):
        return owners
    for ctrl in seg.get("controls") or []:
        owners.update(control_verifier_owners(plan, ctrl))
    return owners


def list_owned_joints(plan: dict | None, verifier: str) -> list[str]:
    normalized = normalize_verifier_name(verifier)
    if normalized is None:
        return []
    return [joint for joint, owner in normalize_joint_verifiers(plan).items() if owner == normalized]


def any_joint_owned_by(plan: dict | None, joints: Iterable[str], verifier: str) -> bool:
    normalized = normalize_verifier_name(verifier)
    if normalized is None:
        return False
    verifiers = normalize_joint_verifiers(plan)
    for joint_name in joints:
        if verifiers.get(str(joint_name or "").strip()) == normalized:
            return True
    return False
