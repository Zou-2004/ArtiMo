#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import copy
import numpy as np

import apply_plan_patch as app_patch


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_controls(plan):
    for si, seg in enumerate(plan.get("timeline") or []):
        for ci, ctrl in enumerate(seg.get("controls") or []):
            yield si, ci, ctrl


def _patch_change(path, op, value):
    return {"path": path, "op": op, "value": value}


def _iter_segments(plan):
    for si, seg in enumerate(plan.get("timeline") or []):
        yield si, seg


def _control_mode(ctrl):
    return str(ctrl.get("mode") or ctrl.get("type") or "")


def _sanitize_limit_expr(expr: str) -> str:
    s = str(expr or "").strip()
    # Keep only the executable expression prefix if VLM included comments like "(0.12 rad)".
    if "(" in s:
        s = s.split("(", 1)[0].strip()
    return s


def _normalize_phase_type(v) -> str | None:
    s = str(v or "").strip().lower()
    if not s:
        return None
    return s


def _collect_motion_signals(motion_vlm_report: dict | None):
    issue_codes = set()
    fix_types = set()
    issues = []
    affected_segments = []
    affected_phase_types = set()
    affected_states = []
    param_fix_hints = []
    repairability = {}
    if not motion_vlm_report:
        return issue_codes, fix_types, issues, affected_segments, affected_phase_types, affected_states, param_fix_hints, repairability
    for issue in motion_vlm_report.get("issues") or []:
        if isinstance(issue, dict) and issue.get("code"):
            issue_codes.add(str(issue.get("code")))
            issues.append(issue)
    for fx in motion_vlm_report.get("suggested_fixes") or []:
        if isinstance(fx, dict) and fx.get("type"):
            fix_types.add(str(fx.get("type")))
    for x in motion_vlm_report.get("affected_timeline_segments") or []:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in affected_segments and xi >= 0:
            affected_segments.append(xi)
    for st in motion_vlm_report.get("affected_states") or []:
        if isinstance(st, dict):
            affected_states.append(st)
            st_phase = _normalize_phase_type(st.get("phase_type"))
            if st_phase:
                affected_phase_types.add(st_phase)
            try:
                xi = int(st.get("segment_index"))
            except Exception:
                xi = None
            if xi is not None and xi >= 0 and xi not in affected_segments:
                affected_segments.append(xi)
    for h in motion_vlm_report.get("param_fix_hints") or []:
        if isinstance(h, dict):
            param_fix_hints.append(h)
            h_phase = _normalize_phase_type(h.get("phase_type"))
            if h_phase:
                affected_phase_types.add(h_phase)
    rep = motion_vlm_report.get("repairability")
    if isinstance(rep, dict):
        repairability = rep
    return issue_codes, fix_types, issues, affected_segments, affected_phase_types, affected_states, param_fix_hints, repairability


def _path_exists_in_plan(plan: dict, path: str) -> bool:
    try:
        app_patch.apply_patch_to_plan(
            copy.deepcopy(plan),
            {
                "patch_type": "param_only_v1",
                "changes": [{"path": path, "op": "replace", "value": 0}],
            },
        )
        return True
    except Exception:
        return False


def _ctrl_has_leaf(ctrl: dict, leaf_path: str) -> bool:
    cur = ctrl
    for part in str(leaf_path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def _ctrl_has_joint(ctrl: dict, joint_name: str) -> bool:
    if str(ctrl.get("joint") or "") == joint_name:
        return True
    joints = ctrl.get("joints")
    if isinstance(joints, list):
        return any(str(x) == joint_name for x in joints)
    return False


def _ctrl_is_exact_joint_target(ctrl: dict, joint_name: str) -> bool:
    """True only when this control targets exactly one joint, and it is joint_name."""
    if str(ctrl.get("joint") or "") == joint_name:
        return True
    joints = ctrl.get("joints")
    if isinstance(joints, list) and len(joints) == 1 and str(joints[0]) == joint_name:
        return True
    return False


def _ctrl_joint_list(ctrl: dict) -> list[str]:
    if str(ctrl.get("joint") or "").strip():
        return [str(ctrl.get("joint")).strip()]
    joints = ctrl.get("joints")
    if isinstance(joints, list):
        return [str(x).strip() for x in joints if str(x).strip()]
    return []


def _joint_position_flip_would_be_noop(ctrl: dict) -> bool:
    """
    Guardrail for one-sided / near-rest position controls.
    If a flip would likely retarget the control back to its start pose (or nearly so),
    skip it instead of creating a no-op that looks like the object froze.
    """
    try:
        q_start = float(ctrl.get("q_start_rad"))
    except Exception:
        q_start = None

    expr = _sanitize_limit_expr(str(ctrl.get("q_target_expr") or ""))
    if q_start is not None and abs(q_start) <= 1.0e-4:
        # Common one-sided case: start at 0, target upper_limit. Flipping to lower_limit
        # would often collapse back to the start pose on assets whose lower limit is 0.
        if "upper_limit" in expr and "lower_limit" not in expr:
            return True

    if q_start is not None and isinstance(ctrl.get("q_target_rad"), (int, float)):
        try:
            flipped = -float(ctrl.get("q_target_rad"))
        except Exception:
            flipped = None
        if flipped is not None:
            tol = max(1.0e-4, 0.02 * max(1.0, abs(flipped), abs(q_start)))
            if abs(flipped - q_start) <= tol:
                return True
    return False


def _finite_float(v) -> float | None:
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return x


def _joint_limits_from_numeric_report(numeric_report: dict) -> dict[str, dict]:
    limits = numeric_report.get("joint_limits") if isinstance(numeric_report, dict) else None
    if not isinstance(limits, dict):
        return {}
    out = {}
    for jn, lim in limits.items():
        if not isinstance(lim, dict):
            continue
        lo = _finite_float(lim.get("lower"))
        hi = _finite_float(lim.get("upper"))
        if lo is None and hi is None:
            continue
        out[str(jn)] = {"lower": lo, "upper": hi}
    return out


def _clamp_to_joint_limits(value: float, lim: dict | None) -> float:
    out = float(value)
    if isinstance(lim, dict):
        lo = lim.get("lower")
        hi = lim.get("upper")
        if lo is not None:
            out = max(out, float(lo))
        if hi is not None:
            out = min(out, float(hi))
    return out


def _sanitize_q_target_rad_changes(plan: dict, changes: list[dict], joint_limits: dict[str, dict]) -> list[dict]:
    if not joint_limits:
        return changes
    timeline = plan.get("timeline") or []
    q_target_re = re.compile(r"^timeline\[(\d+)\]\.controls\[(\d+)\]\.q_target_rad$")
    seen_q_paths = set()
    out = []
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        path = str(ch.get("path") or "")
        m = q_target_re.match(path)
        if not m:
            out.append(ch)
            continue
        seen_q_paths.add(path)
        si = int(m.group(1))
        ci = int(m.group(2))
        try:
            ctrl = timeline[si].get("controls", [])[ci]
        except Exception:
            out.append(ch)
            continue
        if not isinstance(ctrl, dict) or _control_mode(ctrl) != "joint_position":
            out.append(ch)
            continue
        jn = str(ctrl.get("joint") or "").strip()
        lim = joint_limits.get(jn)
        cur = _finite_float(ctrl.get("q_target_rad"))
        if lim is None or cur is None:
            out.append(ch)
            continue
        op = str(ch.get("op") or "")
        if op == "replace":
            proposed = _finite_float(ch.get("value"))
        elif op == "scale":
            sf = _finite_float(ch.get("value"))
            proposed = None if sf is None else cur * sf
        else:
            proposed = None
        if proposed is None:
            out.append(ch)
            continue
        clamped = _clamp_to_joint_limits(proposed, lim)
        if abs(clamped - cur) <= 1.0e-9:
            # A magnitude patch that only pushes an already-limit target farther
            # out of range is a no-op after clamping. Drop it so the loop cannot
            # degrade a correct initial target.
            continue
        if abs(clamped - proposed) > 1.0e-9:
            out.append(_patch_change(path, "replace", clamped))
        else:
            out.append(ch)

    # If a previous run starts from an already-invalid plan, repair it even when
    # the new diagnosis patch touches only timing or another control.
    existing_paths = {str(ch.get("path") or "") for ch in out if isinstance(ch, dict)}
    for si, seg in enumerate(timeline):
        if not isinstance(seg, dict):
            continue
        for ci, ctrl in enumerate(seg.get("controls") or []):
            if not isinstance(ctrl, dict) or _control_mode(ctrl) != "joint_position":
                continue
            jn = str(ctrl.get("joint") or "").strip()
            lim = joint_limits.get(jn)
            cur = _finite_float(ctrl.get("q_target_rad"))
            if lim is None or cur is None:
                continue
            clamped = _clamp_to_joint_limits(cur, lim)
            path = f"timeline[{si}].controls[{ci}].q_target_rad"
            if abs(clamped - cur) > 1.0e-9 and path not in existing_paths and path not in seen_q_paths:
                out.append(_patch_change(path, "replace", clamped))
                existing_paths.add(path)
    return out


def _rewrite_vlm_patch_path(path: str, plan: dict, affected_states: list[dict]) -> str:
    m = re.match(r"^timeline\[(\d+)\]\.controls\[(\d+)\]\.([A-Za-z0-9_.]+)$", str(path))
    if not m:
        return path
    seg_idx = int(m.group(1))
    leaf = m.group(3)
    timeline = plan.get("timeline") or []
    if seg_idx < 0 or seg_idx >= len(timeline):
        return path
    ctrls = timeline[seg_idx].get("controls") or []
    if not isinstance(ctrls, list) or not ctrls:
        return path
    # Prefer controls pointed by affected_states joint names for this segment.
    target_joints = []
    for st in affected_states or []:
        if not isinstance(st, dict):
            continue
        try:
            si = int(st.get("segment_index"))
        except Exception:
            continue
        if si != seg_idx:
            continue
        jn = st.get("joint")
        if jn:
            target_joints.append(str(jn))

    candidates = []
    for ci, ctrl in enumerate(ctrls):
        if not isinstance(ctrl, dict):
            continue
        if not _ctrl_has_leaf(ctrl, leaf):
            continue
        candidates.append(ci)
    if not candidates:
        return path
    chosen = None
    if target_joints:
        for ci in candidates:
            ctrl = ctrls[ci]
            if any(_ctrl_has_joint(ctrl, jn) for jn in target_joints):
                chosen = ci
                break
    if chosen is None:
        # Fallback to first compatible control in that segment.
        chosen = candidates[0]
    return f"timeline[{seg_idx}].controls[{chosen}].{leaf}"


def _build_rule_patch_single(
    plan: dict,
    numeric_report: dict,
    motion_vlm_report: dict | None = None,
    *,
    use_numeric_heuristics: bool = True,
) -> dict | None:
    failures = set((numeric_report.get("failure_signature") or {}).get("codes") or [])
    issues, fix_types, issues_list, affected_segments, affected_phase_types, affected_states, param_fix_hints, repairability = _collect_motion_signals(motion_vlm_report)
    joint_limits = _joint_limits_from_numeric_report(numeric_report)

    # Start from VLM-provided concrete numeric patch (validated, param-only),
    # then augment/override with rule fixes when required (e.g., direction sign).
    initial_changes = []
    initial_reason_codes = []
    explicit_vlm_paths = set()
    if isinstance(motion_vlm_report, dict):
        vlm_patch = motion_vlm_report.get("proposed_param_patch")
        if isinstance(vlm_patch, dict) and (vlm_patch.get("changes") or []):
            cleaned_changes = []
            for ch in vlm_patch.get("changes") or []:
                if not isinstance(ch, dict):
                    continue
                path = str(ch.get("path") or "")
                path = _rewrite_vlm_patch_path(path, plan, affected_states)
                op = str(ch.get("op") or "")
                if not app_patch.is_allowed_path(path):
                    continue
                if not _path_exists_in_plan(plan, path):
                    continue
                if op not in {"replace", "scale"}:
                    continue
                value = ch.get("value")
                if path.endswith(".q_target_expr") and isinstance(value, str):
                    value = _sanitize_limit_expr(value)
                cleaned_changes.append({"path": path, "op": op, "value": value})
                if op == "replace":
                    explicit_vlm_paths.add(path)
            if cleaned_changes:
                initial_changes = cleaned_changes
                initial_reason_codes = [str(x) for x in (vlm_patch.get("reason_codes") or []) if str(x)]

    changes = list(initial_changes)
    reason_codes = sorted(list(failures | issues | fix_types | set(initial_reason_codes)))
    preferred_segments = set(int(x) for x in affected_segments if isinstance(x, int))
    preferred_phase_types = set(str(x) for x in affected_phase_types if isinstance(x, str) and str(x))
    state_kinds = set()
    preferred_joints = set()
    direction_joints = set()
    for st in affected_states:
        if not isinstance(st, dict):
            continue
        st_state = str(st.get("state") or "")
        if st_state:
            state_kinds.add(st_state)
        st_joint = st.get("joint")
        if st_joint:
            jn = str(st_joint)
            preferred_joints.add(jn)
            if st_state in {"direction", "joint_velocity"}:
                direction_joints.add(jn)
        st_phase = _normalize_phase_type(st.get("phase_type"))
        if st_phase:
            preferred_phase_types.add(st_phase)
    hint_direction_joints = set()
    for h in (param_fix_hints or []):
        if not isinstance(h, dict):
            continue
        h_phase = _normalize_phase_type(h.get("phase_type"))
        if h_phase:
            preferred_phase_types.add(h_phase)
        if str(h.get("type") or "") != "adjust_direction":
            continue
        jn = str(h.get("joint") or "").strip()
        if jn:
            hint_direction_joints.add(jn)
    has_joint_scoped_direction_hints = bool(hint_direction_joints)
    all_joint_velocity_joints = set()
    for _si, _ci, ctrl in _iter_controls(plan):
        if _control_mode(ctrl) != "joint_velocity":
            continue
        if ctrl.get("joint"):
            all_joint_velocity_joints.add(str(ctrl.get("joint")))
        elif isinstance(ctrl.get("joints"), list):
            for x in ctrl.get("joints") or []:
                if str(x).strip():
                    all_joint_velocity_joints.add(str(x).strip())
    allow_unscoped_joint_direction_flip = len(all_joint_velocity_joints) <= 1
    has_base_motion_controls = False
    for _si, _ci, ctrl in _iter_controls(plan):
        if _control_mode(ctrl) in {"base_velocity", "base_velocity_decay"}:
            has_base_motion_controls = True
            break
    def _find_change_idx(path: str):
        for idx, ch in enumerate(changes):
            if ch.get("path") == path:
                return idx
        return None

    def _get_change(path: str):
        idx = _find_change_idx(path)
        return changes[idx] if idx is not None else None

    def has_path(path):
        return _find_change_idx(path) is not None

    def _merge_change(path: str, existing: dict, op: str, value):
        if not isinstance(existing, dict):
            return _patch_change(path, op, value)
        ex_op = str(existing.get("op") or "")
        ex_val = existing.get("value")
        if op == "scale":
            try:
                scale_factor = float(value)
            except Exception:
                return _patch_change(path, op, value)
            if ex_op == "replace":
                try:
                    return _patch_change(path, "replace", float(ex_val) * scale_factor)
                except Exception:
                    return _patch_change(path, op, value)
            if ex_op == "scale":
                try:
                    return _patch_change(path, "scale", float(ex_val) * scale_factor)
                except Exception:
                    return _patch_change(path, op, value)
        return _patch_change(path, op, value)

    def add_change(path, op, value, force: bool = False):
        idx = _find_change_idx(path)
        if idx is None:
            changes.append(_patch_change(path, op, value))
            return
        if force:
            changes[idx] = _merge_change(path, changes[idx], op, value)

    def _effective_numeric_path(path: str, raw_value) -> float | None:
        try:
            cur = float(raw_value)
        except Exception:
            return None
        ch = _get_change(path)
        if not isinstance(ch, dict):
            return cur
        op = str(ch.get("op") or "")
        if op == "replace":
            try:
                return float(ch.get("value"))
            except Exception:
                return cur
        if op == "scale":
            try:
                return cur * float(ch.get("value"))
            except Exception:
                return cur
        return cur

    def _ensure_meta_duration_at_least(t_end: float):
        try:
            target = float(t_end)
        except Exception:
            return
        duration_s = float(((plan.get("meta") or {}).get("duration_s")) or 0.0)
        effective_duration = _effective_numeric_path("meta.duration_s", duration_s)
        cur = float(effective_duration) if effective_duration is not None else duration_s
        if target > cur + 1.0e-6:
            add_change("meta.duration_s", "replace", target, force=True)

    def _add_segment_start_preserve_duration(seg_idx: int, new_t0: float, *, force: bool = True, cascade: bool = True):
        timeline = plan.get("timeline") or []
        if seg_idx < 0 or seg_idx >= len(timeline):
            return
        seg = timeline[seg_idx] or {}
        try:
            raw_t0 = float(seg.get("t0", 0.0))
            raw_t1 = float(seg.get("t1", raw_t0))
            target_t0 = float(new_t0)
        except Exception:
            return
        t0_path = f"timeline[{seg_idx}].t0"
        t1_path = f"timeline[{seg_idx}].t1"
        eff_t0 = _effective_numeric_path(t0_path, raw_t0)
        eff_t1 = _effective_numeric_path(t1_path, raw_t1)
        cur_t0 = float(eff_t0) if eff_t0 is not None else raw_t0
        cur_t1 = float(eff_t1) if eff_t1 is not None else raw_t1
        duration = max(0.05, raw_t1 - raw_t0, cur_t1 - cur_t0)
        if abs(cur_t0 - target_t0) > 1.0e-6:
            add_change(t0_path, "replace", target_t0, force=force)
        target_t1 = max(cur_t1, target_t0 + duration)
        if target_t1 > cur_t1 + 1.0e-6 or cur_t1 < target_t0:
            add_change(t1_path, "replace", target_t1, force=True)
        _ensure_meta_duration_at_least(target_t1)
        if not cascade or seg_idx + 1 >= len(timeline):
            return
        next_seg = timeline[seg_idx + 1] or {}
        try:
            next_raw_t0 = float(next_seg.get("t0", target_t1))
        except Exception:
            return
        next_eff_t0 = _effective_numeric_path(f"timeline[{seg_idx + 1}].t0", next_raw_t0)
        next_cur_t0 = float(next_eff_t0) if next_eff_t0 is not None else next_raw_t0
        if abs(next_raw_t0 - raw_t1) < 1.0e-6 or next_cur_t0 < target_t1 - 1.0e-6:
            _add_segment_start_preserve_duration(seg_idx + 1, target_t1, force=force, cascade=True)

    def _sync_joint_velocity_decay_floor(si: int, ci: int, ctrl: dict, *, scale_factor: float | None = None, zero: bool = False):
        if _control_mode(ctrl) != "joint_velocity":
            return
        decay = ctrl.get("decay")
        if not isinstance(decay, dict) or "min_omega_radps" not in decay:
            return
        floor_path = f"timeline[{si}].controls[{ci}].decay.min_omega_radps"
        try:
            raw_floor = float(decay.get("min_omega_radps"))
        except Exception:
            return
        if zero:
            add_change(floor_path, "replace", 0.0, force=True)
            return
        if scale_factor is not None:
            try:
                sf = abs(float(scale_factor))
            except Exception:
                sf = None
            if sf is not None:
                add_change(floor_path, "scale", sf, force=True)
        speed_caps = []
        for leaf in ("omega_radps", "ramp_to_omega_radps"):
            if leaf not in ctrl:
                continue
            path = f"timeline[{si}].controls[{ci}].{leaf}"
            eff = _effective_numeric_path(path, ctrl.get(leaf))
            if eff is None:
                continue
            try:
                speed_caps.append(abs(float(eff)))
            except Exception:
                continue
        speed_caps = [float(v) for v in speed_caps if np.isfinite(v)]
        if not speed_caps:
            return
        speed_cap = min(speed_caps)
        floor_eff = _effective_numeric_path(floor_path, raw_floor)
        target_floor = max(0.0, 0.95 * float(speed_cap))
        if floor_eff is not None and np.isfinite(floor_eff):
            target_floor = min(float(floor_eff), target_floor)
            if abs(float(floor_eff) - float(target_floor)) <= 1.0e-8:
                return
        add_change(floor_path, "replace", float(target_floor), force=True)

    def _repair_negative_timeline_durations():
        timeline = plan.get("timeline") or []
        for si, seg in enumerate(timeline):
            if not isinstance(seg, dict):
                continue
            try:
                raw_t0 = float(seg.get("t0", 0.0))
                raw_t1 = float(seg.get("t1", raw_t0))
            except Exception:
                continue
            t0_path = f"timeline[{si}].t0"
            t1_path = f"timeline[{si}].t1"
            eff_t0 = _effective_numeric_path(t0_path, raw_t0)
            eff_t1 = _effective_numeric_path(t1_path, raw_t1)
            if eff_t0 is None or eff_t1 is None:
                continue
            if float(eff_t1) >= float(eff_t0) - 1.0e-6:
                _ensure_meta_duration_at_least(float(eff_t1))
                continue
            duration = max(0.05, raw_t1 - raw_t0)
            repaired_t1 = float(eff_t0) + duration
            add_change(t1_path, "replace", repaired_t1, force=True)
            _ensure_meta_duration_at_least(repaired_t1)

    def _group_hint_matches_control(hints, *, ctrl: dict, seg_idx: int, phase_filter: str | None, direction: str, hint_types: set[str]) -> bool:
        ctrl_joints = set(_ctrl_joint_list(ctrl))
        if len(ctrl_joints) <= 1:
            return False
        matched = set()
        for h in hints or []:
            if not isinstance(h, dict):
                continue
            if str(h.get("type") or "") not in hint_types:
                continue
            if str(h.get("direction") or "").lower() != str(direction or "").lower():
                continue
            try:
                if int(h.get("segment_index")) != int(seg_idx):
                    continue
            except Exception:
                continue
            h_phase = _normalize_phase_type(h.get("phase_type"))
            if phase_filter is not None and h_phase is not None and h_phase != phase_filter:
                continue
            jn = str(h.get("joint") or "").strip()
            if jn in ctrl_joints:
                matched.add(jn)
        return matched == ctrl_joints

    def _grouped_wheel_hint_matches_control(hint: dict, *, ctrl: dict, seg_idx: int, phase_filter: str | None) -> bool:
        if not isinstance(hint, dict):
            return False
        if _control_mode(ctrl) != "joint_velocity":
            return False
        ctrl_joints = set(_ctrl_joint_list(ctrl))
        if len(ctrl_joints) <= 1:
            return False
        try:
            if int(hint.get("segment_index")) != int(seg_idx):
                return False
        except Exception:
            return False
        h_phase = _normalize_phase_type(hint.get("phase_type"))
        if phase_filter is not None and h_phase is not None and h_phase != phase_filter:
            return False
        jn = str(hint.get("joint") or "").strip()
        if not jn:
            return False
        return jn in ctrl_joints

    hint_apply_used = []
    hint_apply_ignored = []
    synthesized_hints = []

    def _seg_ok(si: int, required_state: str | None = None):
        if preferred_segments and si not in preferred_segments:
            return False
        if preferred_phase_types:
            timeline = plan.get("timeline") or []
            try:
                seg = timeline[int(si)]
            except Exception:
                return False
            seg_phase = _normalize_phase_type((seg or {}).get("phase_type"))
            if seg_phase is None or seg_phase not in preferred_phase_types:
                return False
        if required_state and state_kinds and required_state not in state_kinds:
            # If VLM specified states and this operation is not among them, avoid broad edits.
            return False
        return True

    def scale_base_velocity(factor: float):
        for si, ci, ctrl in _iter_controls(plan):
            if not _seg_ok(si, "base_velocity"):
                continue
            mode = _control_mode(ctrl)
            if mode not in {"base_velocity", "base_velocity_decay"}:
                continue
            key = "v_mps" if "v_mps" in ctrl else ("v0_mps" if "v0_mps" in ctrl else None)
            if key:
                add_change(f"timeline[{si}].controls[{ci}].{key}", "scale", factor)

    def flip_base_control_axis(seg_filter: int | None = None, phase_filter: str | None = None) -> bool:
        changed = False
        timeline = plan.get("timeline") or []
        for si, ci, ctrl in _iter_controls(plan):
            if seg_filter is not None and si != seg_filter:
                continue
            if phase_filter is not None:
                try:
                    seg = timeline[int(si)]
                except Exception:
                    continue
                seg_phase = _normalize_phase_type((seg or {}).get("phase_type"))
                if seg_phase != phase_filter:
                    continue
            if _control_mode(ctrl) not in {"base_velocity", "base_velocity_decay"}:
                continue
            axis = ctrl.get("axis_world")
            if not (isinstance(axis, list) and len(axis) == 3):
                continue
            try:
                new_axis = [-float(axis[0]), -float(axis[1]), -float(axis[2])]
            except Exception:
                continue
            add_change(f"timeline[{si}].controls[{ci}].axis_world", "replace", new_axis, force=True)
            changed = True
        return changed

    def scale_joint_velocity(factor: float):
        for si, ci, ctrl in _iter_controls(plan):
            if not _seg_ok(si, "joint_velocity"):
                continue
            mode = _control_mode(ctrl)
            if mode != "joint_velocity":
                continue
            if preferred_joints:
                joints = []
                if ctrl.get("joint"):
                    joints = [str(ctrl.get("joint"))]
                elif isinstance(ctrl.get("joints"), list):
                    joints = [str(x) for x in ctrl.get("joints")]
                if not any(j in preferred_joints for j in joints):
                    continue
            if "omega_radps" in ctrl:
                add_change(f"timeline[{si}].controls[{ci}].omega_radps", "scale", factor)
            if "ramp_to_omega_radps" in ctrl:
                add_change(f"timeline[{si}].controls[{ci}].ramp_to_omega_radps", "scale", factor)
            _sync_joint_velocity_decay_floor(si, ci, ctrl, scale_factor=factor)

    def _effective_numeric_change(path: str, ctrl: dict, leaf: str) -> float | None:
        try:
            cur = float(ctrl.get(leaf))
        except Exception:
            return None
        ch = _get_change(path)
        if not isinstance(ch, dict):
            return cur
        op = str(ch.get("op") or "")
        if op == "replace":
            try:
                return float(ch.get("value"))
            except Exception:
                return cur
        if op == "scale":
            try:
                return cur * float(ch.get("value"))
            except Exception:
                return cur
        return cur

    def harmonize_wheel_joint_velocity_medians():
        """
        Wheel-style VLM diagnoses are produced per link, so different wheel joints can
        accumulate inconsistent magnitude tweaks. For wheel-transport segments, unify
        the final wheel velocity magnitudes by using the median absolute value across
        wheel joint-velocity controls in the same segment, while preserving each joint's
        sign after all earlier direction fixes.
        """
        timeline = plan.get("timeline") or []
        for si, seg in _iter_segments(plan):
            if not _seg_ok(si):
                continue
            ctrls = list((seg or {}).get("controls") or [])
            if not any(_control_mode(ctrl) in {"base_velocity", "base_velocity_decay"} for ctrl in ctrls):
                continue
            joint_vel_rows = [(ci, ctrl) for ci, ctrl in enumerate(ctrls) if _control_mode(ctrl) == "joint_velocity"]
            if len(joint_vel_rows) < 2:
                continue
            for leaf in ("omega_radps", "ramp_to_omega_radps"):
                values = []
                rows = []
                for ci, ctrl in joint_vel_rows:
                    if leaf not in ctrl:
                        continue
                    path = f"timeline[{si}].controls[{ci}].{leaf}"
                    eff = _effective_numeric_change(path, ctrl, leaf)
                    if eff is None:
                        continue
                    rows.append((path, ctrl, eff))
                    if abs(float(eff)) > 1.0e-8:
                        values.append(abs(float(eff)))
                if len(rows) < 2 or not values:
                    continue
                median_abs = float(np.median(np.asarray(values, dtype=float)))
                if median_abs <= 1.0e-8:
                    continue
                for path, ctrl, eff in rows:
                    sign_source = float(eff)
                    if abs(sign_source) <= 1.0e-8:
                        try:
                            sign_source = float(ctrl.get(leaf))
                        except Exception:
                            sign_source = 1.0
                    sign = -1.0 if sign_source < 0.0 else 1.0
                    add_change(path, "replace", sign * median_abs, force=True)
                for ci, ctrl in joint_vel_rows:
                    _sync_joint_velocity_decay_floor(si, ci, ctrl)

    def adjust_joint_velocity_direction():
        """
        Fix sign inconsistencies in velocity controls (especially clock hand spin-up):
        - enforce ramp_to_omega_radps to have the same sign as omega_radps
        - if VLM flagged WRONG_DIRECTION on a clock-like asset, allow flipping sign for affected joints
        """
        asset_type = str(numeric_report.get("asset_type") or "")
        force_flip_clock = ("WRONG_DIRECTION" in issues) and (asset_type == "clock_like") and (not has_base_motion_controls)
        # For non-clock assets, if VLM already provided explicit omega replacements, trust those values
        # and avoid unconditional sign flips that can invert user-intended direction.
        force_flip_general = False
        target_joints = set(direction_joints) if direction_joints else set(preferred_joints)
        for si, ci, ctrl in _iter_controls(plan):
            if not _seg_ok(si, "joint_velocity") and not _seg_ok(si, "direction"):
                continue
            if _control_mode(ctrl) != "joint_velocity":
                continue
            joints = []
            if ctrl.get("joint"):
                joints = [str(ctrl.get("joint"))]
            elif isinstance(ctrl.get("joints"), list):
                joints = [str(x) for x in ctrl.get("joints")]
            if target_joints and not any(j in target_joints for j in joints):
                continue
            if "omega_radps" not in ctrl:
                continue
            try:
                omega = float(ctrl.get("omega_radps"))
            except Exception:
                continue
            omega_path = f"timeline[{si}].controls[{ci}].omega_radps"
            existing = _get_change(omega_path)
            if existing and existing.get("op") == "replace":
                try:
                    omega = float(existing.get("value"))
                except Exception:
                    pass
            if abs(omega) < 1e-9:
                continue
            desired_sign = -1.0 if omega < 0 else 1.0
            if force_flip_clock or force_flip_general:
                if omega_path in explicit_vlm_paths:
                    # Respect explicit VLM numeric direction for this control.
                    pass
                else:
                    desired_sign *= -1.0
                    add_change(omega_path, "replace", desired_sign * abs(omega), force=True)
            if "ramp_to_omega_radps" in ctrl:
                try:
                    ramp = float(ctrl.get("ramp_to_omega_radps"))
                except Exception:
                    continue
                # If ramp sign mismatches omega sign (or omega flipped above), fix it while preserving magnitude.
                if abs(ramp) < 1e-9:
                    continue
                ramp_sign = -1.0 if ramp < 0 else 1.0
                omega_sign = -1.0 if (desired_sign * abs(omega)) < 0 else 1.0
                if ramp_sign != omega_sign:
                    ramp_path = f"timeline[{si}].controls[{ci}].ramp_to_omega_radps"
                    if ramp_path not in explicit_vlm_paths:
                        add_change(ramp_path, "replace", desired_sign * abs(ramp), force=True)

    def _joint_name_from_ctrl(ctrl):
        if ctrl.get("joint"):
            return str(ctrl.get("joint"))
        if isinstance(ctrl.get("joints"), list) and len(ctrl.get("joints")) == 1:
            return str(ctrl.get("joints")[0])
        return None

    def _find_prev_nonzero_joint_velocity(seg_idx: int, joint_name: str):
        best = None
        for si, _seg in _iter_segments(plan):
            if si >= seg_idx:
                continue
            for _ci, ctrl in enumerate((_seg.get("controls") or [])):
                if _control_mode(ctrl) != "joint_velocity":
                    continue
                if _joint_name_from_ctrl(ctrl) != joint_name:
                    continue
                try:
                    om = float(ctrl.get("omega_radps"))
                except Exception:
                    continue
                if abs(om) > 1e-6:
                    best = om
        return best

    def enforce_wrong_direction_flip():
        # Strong override: for joints explicitly flagged as direction-problematic,
        # flip omega sign in affected segments.
        if "WRONG_DIRECTION" not in issues:
            return
        if not direction_joints:
            return
        if len(explicit_vlm_paths) > 0:
            return
        for si, ci, ctrl in _iter_controls(plan):
            if preferred_segments and si not in preferred_segments:
                continue
            if _control_mode(ctrl) != "joint_velocity":
                continue
            jn = _joint_name_from_ctrl(ctrl)
            if not jn or jn not in direction_joints:
                continue
            path = f"timeline[{si}].controls[{ci}].omega_radps"
            ch = _get_change(path)
            if ch is not None and ch.get("op") == "replace":
                try:
                    om = float(ch.get("value"))
                except Exception:
                    continue
            else:
                try:
                    om = float(ctrl.get("omega_radps"))
                except Exception:
                    continue
            if abs(om) < 1e-6:
                continue
            add_change(path, "replace", -om, force=True)

    def adjust_timing():
        timeline = plan.get("timeline") or []
        if len(timeline) < 2:
            return
        if preferred_segments:
            target_si = min(preferred_segments)
            if target_si <= 0 or target_si >= len(timeline):
                # Timing boundary edits act on the boundary before the target segment.
                return
            prev_si = target_si - 1
            t0 = float((timeline[prev_si] or {}).get("t0", 0.0))
            t1 = float((timeline[prev_si] or {}).get("t1", t0))
            if t1 <= t0 + 1e-6:
                return
            new_t1 = max(t0 + 0.1, t0 + 0.6 * (t1 - t0))
            add_change(f"timeline[{prev_si}].t1", "replace", new_t1)
            t10 = float((timeline[target_si] or {}).get("t0", new_t1))
            if abs(t10 - t1) < 1e-6 or t10 <= new_t1:
                _add_segment_start_preserve_duration(target_si, new_t1)
            return
        # Generic rule: move first boundary earlier to make ordering more distinct.
        t0 = float((timeline[0] or {}).get("t0", 0.0))
        t1 = float((timeline[0] or {}).get("t1", t0))
        if t1 <= t0 + 1e-6:
            return
        new_t1 = max(t0 + 0.1, t0 + 0.6 * (t1 - t0))
        add_change("timeline[0].t1", "replace", new_t1)
        # If the second segment starts at same boundary, move with it.
        t10 = float((timeline[1] or {}).get("t0", new_t1))
        if abs(t10 - t1) < 1e-6 or t10 <= new_t1:
            _add_segment_start_preserve_duration(1, new_t1)

    def adjust_release_return_timing() -> bool:
        timeline = plan.get("timeline") or []
        if not timeline:
            return False

        def _is_return_segment(si: int, seg: dict) -> bool:
            phase = _normalize_phase_type((seg or {}).get("phase_type"))
            if phase in {"control_release", "settle_return"}:
                return True
            for ctrl in (seg or {}).get("controls") or []:
                if _control_mode(ctrl) == "spring_return":
                    return True
            return False

        candidate_indices: list[int] = []
        for si in preferred_segments:
            if 0 <= si < len(timeline) and _is_return_segment(si, timeline[si]):
                candidate_indices.append(si)
        if not candidate_indices:
            for si, seg in _iter_segments(plan):
                if _is_return_segment(si, seg):
                    candidate_indices.append(si)
        if not candidate_indices:
            return False

        target_si = int(candidate_indices[0])
        seg = timeline[target_si] or {}
        t0 = float(seg.get("t0", 0.0))
        t1 = float(seg.get("t1", t0))
        dur = max(0.10, t1 - t0)
        changed = False

        new_t1 = t0 + dur * 1.35
        t1_path = f"timeline[{target_si}].t1"
        effective_t1 = _effective_numeric_path(t1_path, t1)
        target_t1 = max(float(effective_t1) if effective_t1 is not None else t1, new_t1)
        if target_t1 > t1 + 1.0e-6:
            if effective_t1 is None or target_t1 > float(effective_t1) + 1.0e-6:
                add_change(t1_path, "replace", target_t1, force=True)
            changed = True
            if target_si + 1 < len(timeline):
                next_path = f"timeline[{target_si + 1}].t0"
                next_t0 = float((timeline[target_si + 1] or {}).get("t0", target_t1))
                effective_next_t0 = _effective_numeric_path(next_path, next_t0)
                if abs(next_t0 - t1) < 1.0e-6 or next_t0 <= target_t1:
                    if effective_next_t0 is None or target_t1 > float(effective_next_t0) + 1.0e-6:
                        _add_segment_start_preserve_duration(target_si + 1, target_t1)
            duration_s = float(((plan.get("meta") or {}).get("duration_s")) or 0.0)
            effective_duration = _effective_numeric_path("meta.duration_s", duration_s)
            if target_t1 > duration_s + 1.0e-6:
                if effective_duration is None or target_t1 > float(effective_duration) + 1.0e-6:
                    add_change("meta.duration_s", "replace", target_t1, force=True)

        # Return fixes must not borrow time from the preceding hold/open segment:
        # for open_2s_release-style tasks that duration is itself semantic.
        return changed

    def adjust_joint_target():
        # Generic heuristic:
        # 1) Prefer fixing clearly wrong coupled joint targets that use lower_limit.
        # 2) Avoid touching the likely main target joint if there are multiple joint_position controls.
        for si, seg in _iter_segments(plan):
            if not _seg_ok(si, "joint_target"):
                continue
            seg_name = str((seg or {}).get("name") or "").lower()
            ctrls = list((seg or {}).get("controls") or [])
            jp = [(ci, c) for ci, c in enumerate(ctrls) if _control_mode(c) == "joint_position" and c.get("joint")]
            if preferred_joints:
                jp = [(ci, c) for ci, c in jp if str(c.get("joint")) in preferred_joints]
            if len(jp) < 1:
                continue
            # Candidate: joint_position using lower_limit but not the largest-magnitude target.
            mag_info = []
            for ci, c in jp:
                qrad = c.get("q_target_rad")
                expr = str(c.get("q_target_expr") or "")
                mag = abs(float(qrad)) if isinstance(qrad, (int, float)) else (10.0 if "upper_limit" in expr else 1.0)
                mag_info.append((mag, ci, c))
            mag_info_sorted = sorted(mag_info, reverse=True)
            main_ci = mag_info_sorted[0][1] if mag_info_sorted else None
            fixed_any = False
            for _mag, ci, c in mag_info_sorted:
                expr = str(c.get("q_target_expr") or "")
                if "lower_limit" not in expr:
                    continue
                # Guardrail: a lower_limit target in a "close/return/release" segment is often correct
                # (e.g., drawers, doors). Don't blindly flip it to upper_limit from generic VLM hints.
                if any(tok in seg_name for tok in ("close", "return", "release")):
                    continue
                # If diagnosis is only WRONG_DIRECTION, prefer dedicated direction fixes; don't invert limit side.
                if "WRONG_DIRECTION" in issues:
                    continue
                # Skip likely main target if there is another candidate.
                if ci == main_ci and len(mag_info_sorted) > 1:
                    continue
                new_expr = _sanitize_limit_expr(expr.replace("lower_limit", "upper_limit"))
                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", new_expr)
                # If q_target_rad exists and is near zero/negative, mirror magnitude positive as a generic open-direction guess.
                if isinstance(c.get("q_target_rad"), (int, float)):
                    q = float(c["q_target_rad"])
                    add_change(f"timeline[{si}].controls[{ci}].q_target_rad", "replace", max(abs(q), 0.5))
                fixed_any = True
                break
            if fixed_any:
                return
        # Fallback: amplify joint position targets for visibility
        for si, ci, ctrl in _iter_controls(plan):
            if not _seg_ok(si, "joint_target"):
                continue
            if _control_mode(ctrl) != "joint_position":
                continue
            if preferred_joints and str(ctrl.get("joint")) not in preferred_joints:
                continue
            if "q_target_rad" in ctrl:
                add_change(f"timeline[{si}].controls[{ci}].q_target_rad", "scale", 1.2)
            elif "q_target_expr" in ctrl and isinstance(ctrl["q_target_expr"], str):
                expr = _sanitize_limit_expr(ctrl["q_target_expr"])
                parts = expr.split("*", 1)
                if len(parts) == 2:
                    try:
                        alpha = float(parts[0].strip())
                        alpha = max(min(alpha * 1.2, 1.0), -1.0)
                        add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", f"{alpha}*{parts[1].strip()}")
                    except Exception:
                        pass

    def tune_spring_return(stronger: bool):
        found = False
        for si, ci, ctrl in _iter_controls(plan):
            if not _seg_ok(si, "spring_return"):
                continue
            if _control_mode(ctrl) != "spring_return":
                continue
            found = True
            if "spring_k" in ctrl:
                add_change(f"timeline[{si}].controls[{ci}].spring_k", "scale", 1.25 if stronger else 0.8)
            if "damping_c" in ctrl:
                add_change(f"timeline[{si}].controls[{ci}].damping_c", "scale", 1.1 if stronger else 0.9)
            if "rest_position" in ctrl:
                add_change(f"timeline[{si}].controls[{ci}].rest_position", "replace", 0.0)
        return found

    def _strength_scale(strength: str, direction: str) -> float:
        if str(direction or "").lower() == "zero":
            return 0.0
        base = {"small": 1.1, "medium": 1.2, "large": 1.35, "extra_large": 2.0}.get(str(strength or "").lower(), 1.2)
        if str(direction or "").lower() == "decrease":
            return 1.0 / base
        return base

    def synthesize_param_hints_from_issues():
        if param_fix_hints:
            return []
        out = []
        by_issue = set(issues)
        state_rows = [st for st in (affected_states or []) if isinstance(st, dict)]

        def _append(h):
            if isinstance(h, dict):
                out.append(h)

        def _state_hint(default_type: str, default_direction: str):
            emitted = False
            for st in state_rows:
                seg = st.get("segment_index")
                joint = st.get("joint")
                state = str(st.get("state") or "")
                htype = default_type
                if state == "joint_velocity":
                    htype = "adjust_joint_velocity"
                elif state == "joint_target":
                    htype = "adjust_joint_target"
                elif state == "timing":
                    htype = "adjust_timing"
                elif state == "direction":
                    htype = "adjust_direction"
                elif state == "spring_return" and "NO_RELEASE_RETURN" in by_issue:
                    htype = "adjust_timing"
                row = {"type": htype, "direction": default_direction, "strength": "medium", "why": f"derived_from_{'_'.join(sorted(by_issue))}"}
                if isinstance(seg, int):
                    row["segment_index"] = seg
                if isinstance(joint, str) and joint:
                    row["joint"] = joint
                _append(row)
                emitted = True
            return emitted

        if "WRONG_DIRECTION" in by_issue:
            # Do not synthesize a global joint flip from WRONG_DIRECTION alone.
            # Only emit joint-scoped direction hints when affected_states already localizes the issue.
            _state_hint("adjust_direction", "flip")
        if "NO_RELEASE_RETURN" in by_issue:
            # Do not emit an unscoped generic timing hint for return failures: it can
            # lengthen an unrelated early segment. The dedicated return-timing repair
            # below locates control_release / settle_return / spring_return segments.
            _state_hint("adjust_timing", "increase")

        if "UNEXPECTED_EXTRA_MOTION" in by_issue:
            if not _state_hint("adjust_joint_velocity", "zero"):
                _append({"type": "adjust_joint_velocity", "direction": "zero", "why": "derived_from_UNEXPECTED_EXTRA_MOTION"})
        if "MOTION_TOO_SMALL" in by_issue:
            if not _state_hint("adjust_joint_target", "increase"):
                _append({"type": "adjust_joint_target", "direction": "increase", "strength": "medium", "why": "derived_from_MOTION_TOO_SMALL"})
        if "EXCESSIVE_MOTION" in by_issue:
            if not _state_hint("adjust_joint_target", "decrease"):
                _append({"type": "adjust_joint_target", "direction": "decrease", "strength": "medium", "why": "derived_from_EXCESSIVE_MOTION"})
        return out

    def apply_param_fix_hints(hints):
        for h in hints:
            if not isinstance(h, dict):
                continue
            htype = str(h.get("type") or "")
            direction = str(h.get("direction") or "increase").lower()
            strength = str(h.get("strength") or "medium").lower()
            seg_filter = None
            try:
                seg_filter = int(h.get("segment_index"))
            except Exception:
                seg_filter = None
            phase_filter = _normalize_phase_type(h.get("phase_type"))
            joint_filter = str(h.get("joint") or "").strip() or None
            scale = _strength_scale(strength, direction)
            before_paths = {str(ch.get("path")) for ch in changes if isinstance(ch, dict) and ch.get("path")}
            timeline = plan.get("timeline") or []
            matched_existing_path = False
            skipped_wrap_limited_velocity = False

            if (
                direction == "zero"
                and htype in {"adjust_joint_target", "adjust_joint_velocity"}
                and "WRONG_DIRECTION" in issues
                and not ({"UNEXPECTED_EXTRA_MOTION", "EXCESSIVE_MOTION"} & issues)
            ):
                hint_apply_ignored.append({"hint": h, "reason": "skip_zero_target_for_wrong_direction"})
                continue

            def _match_filters(si: int) -> bool:
                if seg_filter is not None and si != seg_filter:
                    return False
                if phase_filter is not None:
                    try:
                        seg = timeline[int(si)]
                    except Exception:
                        return False
                    seg_phase = _normalize_phase_type((seg or {}).get("phase_type"))
                    if seg_phase != phase_filter:
                        return False
                return True

            def _segment_duration(si: int) -> float | None:
                try:
                    seg = timeline[int(si)]
                    return max(0.0, float(seg.get("t1", 0.0)) - float(seg.get("t0", 0.0)))
                except Exception:
                    return None

            def _segment_has_base_transport(si: int) -> bool:
                try:
                    seg = timeline[int(si)]
                except Exception:
                    return False
                for c in (seg or {}).get("controls") or []:
                    if _control_mode(c) in {"base_velocity", "base_velocity_decay"}:
                        return True
                return False

            def _bounded_revolute_velocity_wrap_sensitive(ctrl: dict, si: int, leaf: str) -> bool:
                # Particulate often exports wheel joints as bounded revolute joints with
                # a one-turn-ish [lower, upper] span.  If a base-transport wheel already
                # rotates at more than one bounded span per segment, VLM "too small"
                # judgments based on net wrapped angle are unreliable; scaling it again
                # can make the final GLB visually drift away from the rendered trajectory.
                if direction != "increase" or _control_mode(ctrl) != "joint_velocity":
                    return False
                if not _segment_has_base_transport(si):
                    return False
                dur = _segment_duration(si)
                if dur is None or dur <= 1.0e-6:
                    return False
                try:
                    omega = abs(float(ctrl.get(leaf)))
                except Exception:
                    return False
                if omega <= 1.0e-8:
                    return False
                joints = _ctrl_joint_list(ctrl)
                if joint_filter:
                    joints = [j for j in joints if j == joint_filter]
                if not joints:
                    return False
                for jn in joints:
                    lim = joint_limits.get(str(jn))
                    if not isinstance(lim, dict):
                        return False
                    lo = _finite_float(lim.get("lower"))
                    hi = _finite_float(lim.get("upper"))
                    if lo is None or hi is None:
                        return False
                    span = float(hi) - float(lo)
                    if not (5.0 <= span <= 7.5):
                        return False
                    if omega * float(dur) < 0.75 * span:
                        return False
                return True

            if htype == "adjust_joint_target":
                for si, ci, ctrl in _iter_controls(plan):
                    if not _match_filters(si):
                        continue
                    if _control_mode(ctrl) != "joint_position":
                        continue
                    jn = str(ctrl.get("joint") or "")
                    if joint_filter and jn != joint_filter:
                        continue
                    if "q_target_rad" in ctrl:
                        cur = ctrl.get("q_target_rad")
                        try:
                            curf = float(cur)
                        except Exception:
                            continue
                        if direction == "zero":
                            add_change(f"timeline[{si}].controls[{ci}].q_target_rad", "replace", 0.0, force=True)
                        elif direction == "flip":
                            if _joint_position_flip_would_be_noop(ctrl):
                                continue
                            add_change(f"timeline[{si}].controls[{ci}].q_target_rad", "replace", -curf, force=True)
                        else:
                            add_change(f"timeline[{si}].controls[{ci}].q_target_rad", "scale", scale, force=True)
                        matched_existing_path = True
                    elif "q_target_expr" in ctrl and isinstance(ctrl.get("q_target_expr"), str):
                        expr = _sanitize_limit_expr(ctrl["q_target_expr"])
                        if direction == "zero":
                            if "*" in expr:
                                parts = expr.split("*", 1)
                                add_change(
                                    f"timeline[{si}].controls[{ci}].q_target_expr",
                                    "replace",
                                    f"0.0*{parts[1].strip()}",
                                    force=True,
                                )
                            elif "upper_limit" in expr:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", "0.0*upper_limit", force=True)
                            elif "lower_limit" in expr:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", "0.0*lower_limit", force=True)
                            else:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", "0.0", force=True)
                            matched_existing_path = True
                        elif direction == "flip":
                            if _joint_position_flip_would_be_noop(ctrl):
                                continue
                            if "upper_limit" in expr:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", expr.replace("upper_limit", "lower_limit"), force=True)
                            elif "lower_limit" in expr:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", expr.replace("lower_limit", "upper_limit"), force=True)
                            matched_existing_path = True
                        else:
                            parts = expr.split("*", 1)
                            if len(parts) == 2:
                                try:
                                    alpha = float(parts[0].strip())
                                    if direction == "increase":
                                        alpha = max(min(alpha * scale, 1.0), -1.0)
                                    else:
                                        alpha = max(min(alpha * scale, 1.0), -1.0)
                                    add_change(
                                        f"timeline[{si}].controls[{ci}].q_target_expr",
                                        "replace",
                                        f"{alpha}*{parts[1].strip()}",
                                        force=True,
                                    )
                                    matched_existing_path = True
                                except Exception:
                                    pass

            elif htype == "adjust_joint_velocity":
                matched_velocity_path = False
                for si, ci, ctrl in _iter_controls(plan):
                    if not _match_filters(si):
                        continue
                    if _control_mode(ctrl) != "joint_velocity":
                        continue
                    if joint_filter and not _ctrl_is_exact_joint_target(ctrl, joint_filter):
                        if not (
                            _grouped_wheel_hint_matches_control(h, ctrl=ctrl, seg_idx=si, phase_filter=phase_filter)
                            or (
                            direction == "flip"
                            and _ctrl_has_joint(ctrl, joint_filter)
                            and _group_hint_matches_control(
                                hints,
                                ctrl=ctrl,
                                seg_idx=si,
                                phase_filter=phase_filter,
                                direction=direction,
                                hint_types={"adjust_joint_velocity", "adjust_direction"},
                            )
                            )
                        ):
                            continue
                    if "omega_radps" in ctrl:
                        if direction == "zero":
                            add_change(f"timeline[{si}].controls[{ci}].omega_radps", "replace", 0.0, force=True)
                            matched_existing_path = True
                            matched_velocity_path = True
                        elif direction == "flip":
                            try:
                                om = float(ctrl.get("omega_radps"))
                                add_change(f"timeline[{si}].controls[{ci}].omega_radps", "replace", -om, force=True)
                                matched_existing_path = True
                                matched_velocity_path = True
                            except Exception:
                                pass
                        else:
                            if _bounded_revolute_velocity_wrap_sensitive(ctrl, si, "omega_radps"):
                                skipped_wrap_limited_velocity = True
                            else:
                                add_change(f"timeline[{si}].controls[{ci}].omega_radps", "scale", scale, force=True)
                                matched_existing_path = True
                                matched_velocity_path = True
                    if "ramp_to_omega_radps" in ctrl:
                        if direction == "zero":
                            add_change(f"timeline[{si}].controls[{ci}].ramp_to_omega_radps", "replace", 0.0, force=True)
                            matched_existing_path = True
                            matched_velocity_path = True
                        elif direction == "flip":
                            try:
                                om = float(ctrl.get("ramp_to_omega_radps"))
                                add_change(f"timeline[{si}].controls[{ci}].ramp_to_omega_radps", "replace", -om, force=True)
                                matched_existing_path = True
                                matched_velocity_path = True
                            except Exception:
                                pass
                        else:
                            if _bounded_revolute_velocity_wrap_sensitive(ctrl, si, "ramp_to_omega_radps"):
                                skipped_wrap_limited_velocity = True
                            else:
                                add_change(f"timeline[{si}].controls[{ci}].ramp_to_omega_radps", "scale", scale, force=True)
                                matched_existing_path = True
                                matched_velocity_path = True
                    if direction == "zero":
                        _sync_joint_velocity_decay_floor(si, ci, ctrl, zero=True)
                    elif direction != "flip":
                        _sync_joint_velocity_decay_floor(si, ci, ctrl, scale_factor=scale)
                    else:
                        _sync_joint_velocity_decay_floor(si, ci, ctrl)
                if direction == "zero" and (not matched_velocity_path):
                    # If a zero hint is scoped to a joint but that joint is driven by
                    # position controls, zeroing only the reported segment can still
                    # leave motion from an earlier segment. Suppress the joint across
                    # the full timeline so stateful joint_position interpolation cannot
                    # open it first and then "fix" it by returning to zero.
                    zero_joint_position_globally = bool(joint_filter)
                    for si, ci, ctrl in _iter_controls(plan):
                        if not zero_joint_position_globally and not _match_filters(si):
                            continue
                        if _control_mode(ctrl) != "joint_position":
                            continue
                        jn = str(ctrl.get("joint") or "")
                        if joint_filter and jn != joint_filter:
                            continue
                        if "q_target_rad" in ctrl:
                            add_change(f"timeline[{si}].controls[{ci}].q_target_rad", "replace", 0.0, force=True)
                            matched_existing_path = True
                        elif "q_target_expr" in ctrl and isinstance(ctrl.get("q_target_expr"), str):
                            expr = _sanitize_limit_expr(ctrl["q_target_expr"])
                            if "*" in expr:
                                parts = expr.split("*", 1)
                                add_change(
                                    f"timeline[{si}].controls[{ci}].q_target_expr",
                                    "replace",
                                    f"0.0*{parts[1].strip()}",
                                    force=True,
                                )
                            elif "upper_limit" in expr:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", "0.0*upper_limit", force=True)
                            elif "lower_limit" in expr:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", "0.0*lower_limit", force=True)
                            else:
                                add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", "0.0", force=True)
                            matched_existing_path = True

            elif htype == "adjust_timing":
                if seg_filter is None:
                    if phase_filter is not None:
                        for si, seg in _iter_segments(plan):
                            if not _match_filters(si):
                                continue
                            t0 = float((seg or {}).get("t0", 0.0))
                            t1 = float((seg or {}).get("t1", t0))
                            dur = max(0.05, t1 - t0)
                            if direction == "flip":
                                continue
                            new_t1 = t0 + (dur * scale)
                            add_change(f"timeline[{si}].t1", "replace", new_t1, force=True)
                            if si + 1 < len(timeline):
                                n0 = float((timeline[si + 1] or {}).get("t0", new_t1))
                                if abs(n0 - t1) < 1e-6:
                                    _add_segment_start_preserve_duration(si + 1, new_t1)
                        continue
                    adjust_timing()
                    continue
                if seg_filter < 0 or seg_filter >= len(timeline):
                    continue
                seg = timeline[seg_filter] or {}
                t0 = float(seg.get("t0", 0.0))
                t1 = float(seg.get("t1", t0))
                dur = max(0.05, t1 - t0)
                if direction in {"flip", "zero"}:
                    continue
                new_t1 = t0 + (dur * scale)
                add_change(f"timeline[{seg_filter}].t1", "replace", new_t1, force=True)
                matched_existing_path = True
                if seg_filter + 1 < len(timeline):
                    n0 = float((timeline[seg_filter + 1] or {}).get("t0", new_t1))
                    if abs(n0 - t1) < 1e-6:
                        _add_segment_start_preserve_duration(seg_filter + 1, new_t1)
                        matched_existing_path = True

            elif htype == "adjust_direction":
                if joint_filter:
                    for si, ci, ctrl in _iter_controls(plan):
                        if seg_filter is not None and si != seg_filter:
                            continue
                        if _control_mode(ctrl) != "joint_velocity":
                            continue
                        if not _ctrl_is_exact_joint_target(ctrl, joint_filter):
                            if not (
                                _grouped_wheel_hint_matches_control(h, ctrl=ctrl, seg_idx=si, phase_filter=phase_filter)
                                or (
                                direction == "flip"
                                and _ctrl_has_joint(ctrl, joint_filter)
                                and _group_hint_matches_control(
                                    hints,
                                    ctrl=ctrl,
                                    seg_idx=si,
                                    phase_filter=phase_filter,
                                    direction=direction,
                                    hint_types={"adjust_joint_velocity", "adjust_direction"},
                                )
                                )
                            ):
                                continue
                        if "omega_radps" in ctrl:
                            if direction == "zero":
                                add_change(f"timeline[{si}].controls[{ci}].omega_radps", "replace", 0.0, force=True)
                                matched_existing_path = True
                            elif direction == "flip":
                                try:
                                    om = float(ctrl.get("omega_radps"))
                                    add_change(f"timeline[{si}].controls[{ci}].omega_radps", "replace", -om, force=True)
                                    matched_existing_path = True
                                except Exception:
                                    pass
                            else:
                                add_change(f"timeline[{si}].controls[{ci}].omega_radps", "scale", scale, force=True)
                                matched_existing_path = True
                        if "ramp_to_omega_radps" in ctrl:
                            if direction == "zero":
                                add_change(f"timeline[{si}].controls[{ci}].ramp_to_omega_radps", "replace", 0.0, force=True)
                                matched_existing_path = True
                            elif direction == "flip":
                                try:
                                    om = float(ctrl.get("ramp_to_omega_radps"))
                                    add_change(f"timeline[{si}].controls[{ci}].ramp_to_omega_radps", "replace", -om, force=True)
                                    matched_existing_path = True
                                except Exception:
                                    pass
                            else:
                                add_change(f"timeline[{si}].controls[{ci}].ramp_to_omega_radps", "scale", scale, force=True)
                                matched_existing_path = True
                        if direction == "zero":
                            _sync_joint_velocity_decay_floor(si, ci, ctrl, zero=True)
                        elif direction != "flip":
                            _sync_joint_velocity_decay_floor(si, ci, ctrl, scale_factor=scale)
                        else:
                            _sync_joint_velocity_decay_floor(si, ci, ctrl)
                elif direction == "zero" and has_base_motion_controls:
                    scale_base_velocity(0.0)
                    matched_existing_path = True
                elif direction == "flip" and has_base_motion_controls:
                    if flip_base_control_axis(seg_filter=seg_filter, phase_filter=phase_filter):
                        matched_existing_path = True
                elif direction_joints or preferred_joints:
                    adjust_joint_velocity_direction()
                    matched_existing_path = True
            after_paths = {str(ch.get("path")) for ch in changes if isinstance(ch, dict) and ch.get("path")}
            new_paths = sorted(list(after_paths - before_paths))
            if new_paths:
                hint_apply_used.append({"hint": h, "generated_paths": new_paths})
            elif skipped_wrap_limited_velocity:
                hint_apply_ignored.append({"hint": h, "reason": "bounded_revolute_wheel_velocity_already_wrap_limited"})
            elif matched_existing_path:
                hint_apply_ignored.append({"hint": h, "reason": "already_covered_by_existing_group_control_path"})
            else:
                hint_apply_ignored.append({"hint": h, "reason": "no_matching_control_or_path"})

    def enforce_joint_flip_hints_not_noop(hints):
        for h in hints or []:
            if not isinstance(h, dict):
                continue
            htype = str(h.get("type") or "")
            direction = str(h.get("direction") or "").lower()
            if htype not in {"adjust_joint_velocity", "adjust_direction"} or direction != "flip":
                continue
            joint_filter = str(h.get("joint") or "").strip()
            if not joint_filter:
                continue
            seg_filter = None
            try:
                seg_filter = int(h.get("segment_index"))
            except Exception:
                seg_filter = None
            phase_filter = _normalize_phase_type(h.get("phase_type"))
            for si, ci, ctrl in _iter_controls(plan):
                if seg_filter is not None and si != seg_filter:
                    continue
                if phase_filter is not None:
                    try:
                        seg = (plan.get("timeline") or [])[int(si)]
                    except Exception:
                        continue
                    if _normalize_phase_type((seg or {}).get("phase_type")) != phase_filter:
                        continue
                if _control_mode(ctrl) != "joint_velocity":
                    continue
                if not _ctrl_is_exact_joint_target(ctrl, joint_filter):
                    continue
                if "omega_radps" in ctrl:
                    path = f"timeline[{si}].controls[{ci}].omega_radps"
                    try:
                        cur = float(ctrl.get("omega_radps"))
                    except Exception:
                        cur = None
                    ch = _get_change(path)
                    if cur is not None:
                        if not isinstance(ch, dict) or ch.get("op") != "replace":
                            add_change(path, "replace", -cur, force=True)
                        else:
                            try:
                                val = float(ch.get("value"))
                            except Exception:
                                val = None
                            if val is not None and abs(val - cur) < 1.0e-9:
                                add_change(path, "replace", -cur, force=True)
                if "ramp_to_omega_radps" in ctrl:
                    path = f"timeline[{si}].controls[{ci}].ramp_to_omega_radps"
                    try:
                        cur = float(ctrl.get("ramp_to_omega_radps"))
                    except Exception:
                        cur = None
                    ch = _get_change(path)
                    if cur is not None:
                        if not isinstance(ch, dict) or ch.get("op") != "replace":
                            add_change(path, "replace", -cur, force=True)
                        else:
                            try:
                                val = float(ch.get("value"))
                            except Exception:
                                val = None
                            if val is not None and abs(val - cur) < 1.0e-9:
                                add_change(path, "replace", -cur, force=True)

    # Apply VLM param hints first (VLM localizes segment/joint; patcher resolves concrete paths).
    fallback_hints = synthesize_param_hints_from_issues()
    if fallback_hints:
        synthesized_hints.extend(fallback_hints)
    all_hints = list(param_fix_hints) + fallback_hints
    apply_param_fix_hints(all_hints)
    enforce_joint_flip_hints_not_noop(all_hints)

    # Direction issues: base/world direction fixes and wheel/joint sign fixes are independent.
    # Apply both when both are present; do not suppress one because the other also exists.
    if has_base_motion_controls and ("BASE_DIRECTION_OPPOSITE" in failures or "flip_base_axis" in fix_types):
        flip_base_control_axis()

    # Route suggested fixes from VLM (generic + reusable).
    if "increase_base_velocity" in fix_types:
        scale_base_velocity(1.25)
    if "decrease_base_velocity" in fix_types:
        scale_base_velocity(0.8)
    if "increase_joint_velocity" in fix_types:
        scale_joint_velocity(1.25)
    if "decrease_joint_velocity" in fix_types:
        scale_joint_velocity(0.8)
    suppress_decay_for_release_return = "NO_RELEASE_RETURN" in issues and not (
        {"EXCESSIVE_MOTION", "UNEXPECTED_EXTRA_MOTION", "MOTION_TOO_SMALL"} & issues
    )
    release_return_timing_changed = False
    if "NO_RELEASE_RETURN" in issues:
        release_return_timing_changed = adjust_release_return_timing()
    if "adjust_timing" in fix_types:
        if not release_return_timing_changed:
            adjust_timing()
    target_issue_present = bool({"MOTION_TOO_SMALL"} & issues)
    only_direction_like = bool(issues) and issues.issubset({"WRONG_DIRECTION"})
    if (("adjust_joint_target" in fix_types and not only_direction_like) or target_issue_present):
        adjust_joint_target()
    if "adjust_joint_velocity_direction" in fix_types or (
        "WRONG_DIRECTION" in issues and str(numeric_report.get("asset_type") or "") == "clock_like"
    ):
        if has_joint_scoped_direction_hints or allow_unscoped_joint_direction_flip:
            adjust_joint_velocity_direction()
    if (not has_joint_scoped_direction_hints) and allow_unscoped_joint_direction_flip:
        enforce_wrong_direction_flip()
    if "NO_RELEASE_RETURN" in issues and not release_return_timing_changed:
        tune_spring_return(stronger=True)

    if use_numeric_heuristics:
        # Magnitude tuning heuristics from numeric checks.
        for chk in numeric_report.get("checks") or []:
            if not isinstance(chk, dict) or chk.get("ok", True):
                continue
            code = str(chk.get("code"))
            if code == "BASE_MOVE_EXPECTED":
                for si, ci, ctrl in _iter_controls(plan):
                    mode = str(ctrl.get("mode") or ctrl.get("type") or "")
                    if mode in {"base_velocity", "base_velocity_decay"}:
                        key = "v_mps" if "v_mps" in ctrl else "v0_mps"
                        if key in ctrl:
                            add_change(f"timeline[{si}].controls[{ci}].{key}", "scale", 1.3)
                            break
            elif code == "WHEEL_ROTATE_EXPECTED":
                jn = chk.get("joint")
                for si, ci, ctrl in _iter_controls(plan):
                    mode = _control_mode(ctrl)
                    if mode != "joint_velocity":
                        continue
                    joints = []
                    if ctrl.get("joint"):
                        joints = [ctrl.get("joint")]
                    elif isinstance(ctrl.get("joints"), list):
                        joints = list(ctrl.get("joints"))
                    if jn is not None and jn not in joints:
                        continue
                    if "omega_radps" in ctrl:
                        add_change(f"timeline[{si}].controls[{ci}].omega_radps", "scale", 1.25)
                        _sync_joint_velocity_decay_floor(si, ci, ctrl, scale_factor=1.25)
                        break
            elif code == "RELEASE_RETURN_PRESENT":
                continue
            elif code == "PEDAL_MOVES_FIRST":
                adjust_timing()

        # LIMIT violations: clamp q_target_rad / q_target_expr coefficient if available
        for chk in numeric_report.get("checks") or []:
            if not isinstance(chk, dict):
                continue
            if chk.get("code") != "LIMIT_NOT_EXCEEDED" or chk.get("ok", True):
                continue
            jn = chk.get("joint")
            for si, ci, ctrl in _iter_controls(plan):
                mode = _control_mode(ctrl)
                if mode != "joint_position" or ctrl.get("joint") != jn:
                    continue
                if "q_target_rad" in ctrl:
                    add_change(f"timeline[{si}].controls[{ci}].q_target_rad", "scale", 0.9)
                elif "q_target_expr" in ctrl and isinstance(ctrl["q_target_expr"], str):
                    expr = _sanitize_limit_expr(ctrl["q_target_expr"])
                    parts = expr.split("*", 1)
                    if len(parts) == 2:
                        try:
                            alpha = float(parts[0].strip())
                            alpha = max(min(alpha * 0.9, 1.0), -1.0)
                            add_change(f"timeline[{si}].controls[{ci}].q_target_expr", "replace", f"{alpha}*{parts[1].strip()}")
                        except Exception:
                            pass
                break

    enforce_joint_flip_hints_not_noop(all_hints)
    harmonize_wheel_joint_velocity_medians()
    _repair_negative_timeline_durations()

    # Deduplicate exact path duplicates; keep LAST change so rule-based overrides
    # can supersede an earlier VLM-proposed value on the same path.
    dedup_map = {}
    order = []
    for ch in changes:
        p = ch.get("path")
        if p is None:
            continue
        if p not in dedup_map:
            order.append(p)
        dedup_map[p] = ch
    dedup = [dedup_map[p] for p in order]
    dedup = _sanitize_q_target_rad_changes(plan, dedup, joint_limits)
    if not dedup:
        return None
    return {
        "patch_type": "param_only_v1",
        "changes": dedup,
        "reason_codes": reason_codes,
        "param_patch_apply_report": {
            "used_param_fix_hints": hint_apply_used,
            "ignored_param_fix_hints": hint_apply_ignored,
            "synthesized_fallback_hints": synthesized_hints,
        },
    }


def build_rule_patch(
    plan: dict,
    numeric_report: dict,
    motion_vlm_report: dict | None = None,
) -> dict | None:
    return _build_rule_patch_single(
        plan,
        numeric_report,
        motion_vlm_report if isinstance(motion_vlm_report, dict) else None,
        use_numeric_heuristics=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Rule-based plan patcher (param-only)")
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--numeric_report", required=True)
    parser.add_argument("--motion_vlm_report", default=None)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()
    plan = _load(Path(args.plan_json))
    numeric = _load(Path(args.numeric_report))
    motion = _load(Path(args.motion_vlm_report)) if args.motion_vlm_report else None
    patch = build_rule_patch(plan, numeric, motion)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if patch is None:
        out_path.write_text(json.dumps({"patch_type": "param_only_v1", "changes": [], "reason_codes": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_path.write_text(json.dumps(patch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
