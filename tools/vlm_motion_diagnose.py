#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import re
from pathlib import Path
from typing import Any

from api_client_utils import generate_content_text

MAX_MOTION_VLM_IMAGES = 9

PER_LINK_ALLOWED_ISSUE_CODES = {
    "WRONG_DIRECTION",
    "UNEXPECTED_EXTRA_MOTION",
    "EXCESSIVE_MOTION",
    "MOTION_TOO_SMALL",
    "NO_RELEASE_RETURN",
}
PER_LINK_ALLOWED_FIX_TYPES = {
    "flip_base_axis",
    "increase_base_velocity",
    "decrease_base_velocity",
    "increase_joint_velocity",
    "decrease_joint_velocity",
    "adjust_timing",
    "adjust_joint_target",
    "adjust_joint_velocity_direction",
}
PER_LINK_ALLOWED_HINT_TYPES = {
    "adjust_joint_target",
    "adjust_joint_velocity",
    "adjust_timing",
    "adjust_direction",
}
PER_LINK_ALLOWED_STATE_TARGETS = {
    "timing",
    "joint_target",
    "joint_velocity",
    "base_velocity",
    "direction",
    "spring_return",
}

ALLOWED_ISSUE_CODES = set(PER_LINK_ALLOWED_ISSUE_CODES)
ALLOWED_FIX_TYPES = set(PER_LINK_ALLOWED_FIX_TYPES)
ALLOWED_STATE_TARGETS = set(PER_LINK_ALLOWED_STATE_TARGETS)
ALLOWED_PATCH_OPS = {"replace", "scale"}
ALLOWED_REPAIR_LEVEL = {"param", "none"}
ALLOWED_HINT_TYPES = set(PER_LINK_ALLOWED_HINT_TYPES)
ALLOWED_HINT_DIRECTIONS = {"increase", "decrease", "flip", "zero"}
ALLOWED_HINT_STRENGTH = {"small", "medium", "large", "extra_large"}
ALLOWED_PHASE_TYPES = {
    "control_activation",
    "control_release",
    "control_to_effect_lag",
    "effect_motion",
    "settle_return",
    "hold",
    "transport",
}
ALLOWED_LINK_MOTION_TYPES = {
    "rotational_cw",
    "rotational_ccw",
    "prismatic",
    "static_or_unclear",
}


def _extract_canonical_link_id(v: Any) -> str | None:
    s = str(v or "").strip()
    m = re.search(r"\blink_[A-Za-z0-9_]+\b", s)
    return m.group(0) if m else None


def _extract_canonical_joint_id(v: Any) -> str | None:
    s = str(v or "").strip()
    m = re.search(r"\bjoint_[A-Za-z0-9_]+\b", s)
    return m.group(0) if m else None


def _sanitize_phase_type(v: Any) -> str | None:
    s = str(v or "").strip().lower()
    if not s:
        return None
    if s in ALLOWED_PHASE_TYPES:
        return s
    return None


def _flip_cw_ccw(direction: Any) -> str | None:
    d = str(direction or "").strip().lower()
    if d == "cw":
        return "ccw"
    if d == "ccw":
        return "cw"
    return None


def _current_view_direction_from_projection(local_motion: dict | None) -> str | None:
    if not isinstance(local_motion, dict):
        return None
    direction = str(local_motion.get("direction") or "").strip().lower()
    if direction not in {"cw", "ccw"}:
        return None
    proj = str(local_motion.get("axis_projection") or local_motion.get("axis_projection_note") or "").strip().lower()
    if "cross" in proj or proj == "cross_in":
        return _flip_cw_ccw(direction)
    if "dot" in proj or proj == "dot_out":
        return direction
    return None


def _attach_current_view_direction(local_motion: dict | None) -> dict | None:
    if not isinstance(local_motion, dict):
        return local_motion
    out = dict(local_motion)
    current = _current_view_direction_from_projection(out)
    if current is not None:
        out["current_view_direction"] = current
        out["conversion_rule"] = "CROSS IN flips axis-relative CW/CCW; DOT OUT keeps it unchanged"
    return out


def _allowed_sets_for_motion_mode(_motion_mode: str | None = None) -> tuple[set[str], set[str], set[str], set[str]]:
    return (
        set(PER_LINK_ALLOWED_ISSUE_CODES),
        set(PER_LINK_ALLOWED_FIX_TYPES),
        set(PER_LINK_ALLOWED_HINT_TYPES),
        set(PER_LINK_ALLOWED_STATE_TARGETS),
    )


def _collect_known_ids(plan_summary: dict) -> tuple[set[str], set[str]]:
    known_links: set[str] = set()
    known_joints: set[str] = set()
    action = plan_summary.get("action") or {}
    tl = action.get("target_link")
    if isinstance(tl, str):
        cid = _extract_canonical_link_id(tl)
        if cid:
            known_links.add(cid)
    tls = action.get("target_links")
    if isinstance(tls, list):
        for x in tls:
            cid = _extract_canonical_link_id(x)
            if cid:
                known_links.add(cid)
    effects = plan_summary.get("effects") or {}
    for jt in effects.get("joint_targets") or []:
        if not isinstance(jt, dict):
            continue
        jn = _extract_canonical_joint_id(jt.get("joint"))
        if jn:
            known_joints.add(jn)
    timeline = plan_summary.get("timeline") or []
    for seg in timeline:
        if not isinstance(seg, dict):
            continue
        for ctrl in seg.get("controls") or []:
            if not isinstance(ctrl, dict):
                continue
            jn = _extract_canonical_joint_id(ctrl.get("joint"))
            if jn:
                known_joints.add(jn)
            if isinstance(ctrl.get("joints"), list):
                for x in ctrl.get("joints"):
                    jx = _extract_canonical_joint_id(x)
                    if jx:
                        known_joints.add(jx)
    legend = plan_summary.get("image_label_legend") or {}
    if isinstance(legend, dict):
        for k in legend.keys():
            cid = _extract_canonical_link_id(k)
            if cid:
                known_links.add(cid)
        for v in legend.values():
            cid = _extract_canonical_link_id(v)
            if cid:
                known_links.add(cid)
    return known_links, known_joints


def _filter_plan_summary_for_tail(plan_summary: dict, row: dict) -> dict:
    if not isinstance(plan_summary, dict):
        return {}
    link_name = str(row.get("link") or "").strip()
    joint_name = str(row.get("joint") or "").strip()
    seg_idx = row.get("segment_index")
    out = {
        "action": dict(plan_summary.get("action") or {}),
        "effects": {},
        "meta": dict(plan_summary.get("meta") or {}),
        "joint_limits": {},
        "timeline": [],
        "image_label_legend": {},
        "motion_diagnosis_mode": "per_link_head_tail_pair",
    }
    effects = dict(plan_summary.get("effects") or {})
    if isinstance(effects.get("joint_targets"), list):
        effects["joint_targets"] = [
            jt for jt in (effects.get("joint_targets") or [])
            if isinstance(jt, dict) and str(_extract_canonical_joint_id(jt.get("joint")) or "") == joint_name
        ]
    out["effects"] = effects
    legend = plan_summary.get("image_label_legend") or {}
    if isinstance(legend, dict) and link_name:
        out["image_label_legend"] = {link_name: legend.get(link_name, legend.get(str(link_name), ""))}
    joint_limits = plan_summary.get("joint_limits") or {}
    if joint_name and isinstance(joint_limits, dict) and isinstance(joint_limits.get(joint_name), dict):
        out["joint_limits"] = {joint_name: dict(joint_limits.get(joint_name) or {})}
    for idx, seg in enumerate(plan_summary.get("timeline") or []):
        if not isinstance(seg, dict):
            continue
        if seg_idx is not None:
            try:
                if int(seg_idx) != idx:
                    continue
            except Exception:
                pass
        seg_out = {k: seg.get(k) for k in ["name", "phase_type", "t0", "t1"] if k in seg}
        controls_out = []
        for ctrl in seg.get("controls") or []:
            if not isinstance(ctrl, dict):
                continue
            mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()
            if mode in {"base_velocity", "base_velocity_decay", "base", "base_decay"}:
                controls_out.append(dict(ctrl))
                continue
            if str(_extract_canonical_joint_id(ctrl.get("joint")) or "") == joint_name:
                controls_out.append(dict(ctrl))
        seg_out["controls"] = controls_out
        out["timeline"].append(seg_out)
    return out


def _filter_trajectory_summary_for_tail(trajectory_summary: dict | None, row: dict) -> dict | None:
    if not isinstance(trajectory_summary, dict):
        return None
    joint_name = str(row.get("joint") or "").strip()
    link_name = str(row.get("link") or "").strip()
    local_motion = None
    local_motion_current_view = None
    seg_idx = row.get("segment_index")
    out = {
        "num_frames": trajectory_summary.get("num_frames"),
        "duration_s": trajectory_summary.get("duration_s"),
        "target_link": link_name,
        "target_joint": joint_name,
        "base_translation_summary": trajectory_summary.get("base_translation_summary"),
    }
    if isinstance(row.get("selected_view"), dict):
        out["selected_view"] = dict(row.get("selected_view") or {})
    joints_summary = trajectory_summary.get("joints_summary") or {}
    cues = row.get("motion_cues") if isinstance(row.get("motion_cues"), dict) else {}
    if isinstance(cues.get("local_motion"), dict):
        local_motion_current_view = _attach_current_view_direction(dict(cues.get("local_motion") or {}))
    if joint_name and joint_name in joints_summary:
        joint_row = dict(joints_summary.get(joint_name) or {})
        if isinstance(joint_row.get("local_motion"), dict):
            local_motion = dict(joint_row.get("local_motion") or {})
        out["joints_summary"] = {joint_name: joint_row}
        out["tracked_joints"] = [joint_name]
    timeline_out = []
    for seg in trajectory_summary.get("timeline_segment_motion") or []:
        if not isinstance(seg, dict):
            continue
        if seg_idx is not None:
            try:
                if int(seg.get("segment_index")) != int(seg_idx):
                    continue
            except Exception:
                pass
        seg_out = {
            "segment_index": seg.get("segment_index"),
            "segment_name": seg.get("segment_name"),
            "phase_type": seg.get("phase_type"),
            "t0": seg.get("t0"),
            "t1": seg.get("t1"),
            "frame_range": seg.get("frame_range"),
            "base_motion": seg.get("base_motion"),
        }
        per_joint = seg.get("per_joint_motion") or {}
        if joint_name and joint_name in per_joint:
            joint_seg_row = dict(per_joint.get(joint_name) or {})
            if local_motion is None and isinstance(joint_seg_row.get("local_motion"), dict):
                local_motion = dict(joint_seg_row.get("local_motion") or {})
            seg_out["tracked_joints"] = [joint_name]
            seg_out["per_joint_motion"] = {joint_name: joint_seg_row}
        timeline_out.append(seg_out)
    if local_motion is None:
        for jt in cues.get("joint_trends") or []:
            if not isinstance(jt, dict):
                continue
            if str(jt.get("joint") or "").strip() != joint_name and str(jt.get("link") or "").strip() != link_name:
                continue
            local_motion = {
                k: jt.get(k)
                for k in [
                    "motion_type",
                    "direction",
                    "axis_world",
                    "signed_axis_world",
                    "axis_label",
                    "local_motion_text",
                    "frame_note",
                ]
                if jt.get(k) is not None
            }
            if local_motion_current_view is None:
                local_motion_current_view = _attach_current_view_direction({
                    k: jt.get(k)
                    for k in [
                        "motion_type",
                        "direction",
                        "axis_world",
                        "signed_axis_world",
                        "axis_label",
                        "local_motion_text",
                        "frame_note",
                        "axis_projection",
                        "axis_projection_note",
                        "axis_projection_tag",
                        "axis_toward_camera",
                    ]
                    if jt.get(k) is not None
                })
            if local_motion:
                break
    if isinstance(local_motion, dict) and local_motion:
        out["local_motion"] = local_motion
    if isinstance(local_motion_current_view, dict) and local_motion_current_view:
        local_motion_current_view = _attach_current_view_direction(local_motion_current_view)
        out["local_motion_current_view"] = local_motion_current_view
    out["timeline_segment_motion"] = timeline_out
    return out


def _filter_scale_context_for_tail(scale_context: dict | None, row: dict) -> dict | None:
    if not isinstance(scale_context, dict):
        return None
    joint_name = str(row.get("joint") or "").strip()
    link_name = str(row.get("link") or "").strip()
    out = {
        "asset": scale_context.get("asset"),
        "unit_assumption": scale_context.get("unit_assumption"),
        "object_bbox_extents_m": scale_context.get("object_bbox_extents_m"),
        "object_diag_m": scale_context.get("object_diag_m"),
        "median_revolute_child_radius_m_est": scale_context.get("median_revolute_child_radius_m_est"),
    }
    if link_name and isinstance(scale_context.get("link_bbox_extents_m"), dict):
        out["link_bbox_extents_m"] = {link_name: (scale_context.get("link_bbox_extents_m") or {}).get(link_name)}
    if joint_name and isinstance(scale_context.get("joint_child_link_bbox_extents_m"), dict):
        out["joint_child_link_bbox_extents_m"] = {joint_name: (scale_context.get("joint_child_link_bbox_extents_m") or {}).get(joint_name)}
    return out


def _is_allowed_patch_path(path: str) -> bool:
    if path in {"meta.fps", "meta.duration_s"}:
        return True
    if any(s in path for s in [".joint", ".joints", ".mode", ".type"]):
        return False
    if not path.startswith("timeline["):
        return False
    allowed_suffixes = {
        ".t0",
        ".t1",
        ".axis_world",
        ".v_mps",
        ".v0_mps",
        ".tau_s",
        ".omega_radps",
        ".ramp_to_omega_radps",
        ".q_target_rad",
        ".q_target_expr",
        ".spring_k",
        ".damping_c",
        ".rest_position",
        ".min_omega_radps",
        ".decay.tau_s",
        ".decay.min_omega_radps",
    }
    return any(path.endswith(s) for s in allowed_suffixes)


def _sanitize_param_patch(obj: Any) -> dict | None:
    if not isinstance(obj, dict):
        return None
    changes = obj.get("changes") or []
    if not isinstance(changes, list):
        return None
    changes_out = []
    for ch in changes:
        if not isinstance(ch, dict):
            continue
        path = str(ch.get("path") or "")
        op = str(ch.get("op") or "")
        if not path or op not in ALLOWED_PATCH_OPS or not _is_allowed_patch_path(path):
            continue
        value = ch.get("value")
        if path.endswith(".q_target_expr"):
            value = str(value or "").strip()
            if "(" in value:
                value = value.split("(", 1)[0].strip()
            if not any(k in value for k in ("upper_limit", "lower_limit")):
                continue
        if op == "scale":
            try:
                value = float(value)
            except Exception:
                continue
        changes_out.append({"path": path, "op": op, "value": value})
    if not changes_out:
        return None
    return {
        "patch_type": "param_only_v1",
        "changes": changes_out,
        "reason_codes": [str(x) for x in (obj.get("reason_codes") or []) if str(x)],
    }


def _sanitize_repairability(obj: Any, has_hints: bool) -> dict:
    rep = obj if isinstance(obj, dict) else {}
    preferred = str(rep.get("preferred_repair_level") or "").strip().lower()
    if preferred not in ALLOWED_REPAIR_LEVEL:
        preferred = "param" if has_hints else "none"
    param_fixable = bool(rep.get("param_fixable", has_hints))
    structural_fix_needed = False
    if preferred == "param":
        param_fixable = True
    if has_hints and not param_fixable:
        param_fixable = True
    if (not has_hints) and preferred == "param":
        preferred = "none"
    return {
        "param_fixable": param_fixable,
        "structural_fix_needed": structural_fix_needed,
        "preferred_repair_level": preferred,
    }


def _sanitize_param_fix_hints(
    obj: Any,
    known_joints: set[str] | None = None,
    allowed_hint_types: set[str] | None = None,
) -> list[dict]:
    hints = obj if isinstance(obj, list) else []
    out = []
    for h in hints:
        if not isinstance(h, dict):
            continue
        htype = str(h.get("type") or "").strip()
        if htype not in ALLOWED_HINT_TYPES:
            continue
        if allowed_hint_types is not None and htype not in allowed_hint_types:
            continue
        row = {"type": htype}
        try:
            seg_idx = int(h.get("segment_index"))
            if seg_idx >= 0:
                row["segment_index"] = seg_idx
        except Exception:
            pass
        phase_type = _sanitize_phase_type(h.get("phase_type"))
        if phase_type is not None:
            row["phase_type"] = phase_type
        jn = _extract_canonical_joint_id(h.get("joint"))
        if jn is not None and (not known_joints or jn in known_joints):
            row["joint"] = jn
        direction = str(h.get("direction") or "").strip().lower()
        if direction in ALLOWED_HINT_DIRECTIONS:
            if direction == "zero" and htype not in {"adjust_joint_target", "adjust_joint_velocity", "adjust_direction"}:
                direction = ""
            if direction:
                row["direction"] = direction
        strength = str(h.get("strength") or "").strip().lower()
        if strength in ALLOWED_HINT_STRENGTH and direction not in {"flip", "zero"}:
            row["strength"] = strength
        why = str(h.get("why") or "").strip()
        if why:
            row["why"] = why[:200]
        out.append(row)
    return out


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _has_effective_mask_meta(meta: Any) -> bool:
    if not isinstance(meta, dict):
        return False
    try:
        if int(meta.get("count", 0)) > 0:
            return True
    except Exception:
        pass
    masks = meta.get("masks")
    if isinstance(masks, list) and len(masks) > 0:
        return True
    mode = str(meta.get("mode") or "").strip().lower()
    return mode not in {"", "none", "empty"}


def _sanitize_link_motion_type(v: Any) -> str:
    s = str(v or "").strip().lower()
    if s in ALLOWED_LINK_MOTION_TYPES:
        return s
    aliases = {
        "cw": "rotational_cw",
        "ccw": "rotational_ccw",
        "clockwise": "rotational_cw",
        "counterclockwise": "rotational_ccw",
        "counter_clockwise": "rotational_ccw",
        "counter-clockwise": "rotational_ccw",
        "translation": "prismatic",
        "translational": "prismatic",
        "prismatic_translation": "prismatic",
        "unclear": "static_or_unclear",
        "static": "static_or_unclear",
    }
    return aliases.get(s, "static_or_unclear")


def _prune_report_fields(report: dict) -> dict:
    if not isinstance(report, dict):
        return report
    out = dict(report)
    for key in ["suggested_fixes", "affected_timeline_segments", "affected_states", "param_fix_hints", "issues"]:
        if key in out and isinstance(out[key], list) and len(out[key]) == 0:
            out.pop(key, None)
    if "detailed_reasoning" in out and not str(out.get("detailed_reasoning") or "").strip():
        out.pop("detailed_reasoning", None)
    return out


def _format_detailed_reasoning(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    markers = [
        "[Trajectory]",
        "[Projection]",
        "[Apparent]",
        "[Conversion]",
        "[Magnitude]",
        "[ExtraMotion]",
        "[Global]",
        "[Return]",
    ]
    for marker in markers:
        s = s.replace(f"\n{marker}", f"\n {marker}")
        s = s.replace(f" {marker}", f"\n {marker}")
        s = s.replace(f"| {marker}", f"\n {marker}")
    s = s.lstrip()
    if s.startswith("["):
        pass
    elif s.startswith(" ["):
        s = s[1:]
    lines = [line.rstrip() for line in s.splitlines()]
    return "\n".join(lines).strip()


def _sanitize_motion_report(
    obj: dict,
    known_links: set[str] | None = None,
    known_joints: set[str] | None = None,
    motion_mode: str | None = None,
) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("motion report must be object")
    allowed_issue_codes, allowed_fix_types, allowed_hint_types, allowed_state_targets = _allowed_sets_for_motion_mode(motion_mode)
    issues = obj.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    issues_out = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        code = str(it.get("code") or "")
        if (not code) or (code not in ALLOWED_ISSUE_CODES) or (code not in allowed_issue_codes):
            continue
        row = {"code": code}
        for k in [
            "expected_axis",
            "observed_axis",
            "expected_link",
            "observed_link",
            "joint",
            "from_link",
            "to_link",
            "expected_joint",
            "observed_joint",
        ]:
            if k in it:
                val = it[k]
                if k in {"expected_link", "observed_link", "from_link", "to_link"}:
                    cid = _extract_canonical_link_id(val)
                    if cid is None:
                        continue
                    if known_links and cid not in known_links:
                        continue
                    val = cid
                if k in {"joint", "expected_joint", "observed_joint"}:
                    jid = _extract_canonical_joint_id(val)
                    if jid is None:
                        continue
                    if known_joints and jid not in known_joints:
                        continue
                    val = jid
                row[k] = val
        issues_out.append(row)
    fixes = obj.get("suggested_fixes") or []
    if not isinstance(fixes, list):
        fixes = []
    fixes_out = []
    for fx in fixes:
        if not isinstance(fx, dict):
            continue
        if not fx.get("type"):
            continue
        if str(fx.get("type")) not in ALLOWED_FIX_TYPES or str(fx.get("type")) not in allowed_fix_types:
            continue
        row = {"type": str(fx.get("type"))}
        for k in ["from_link", "to_link", "joint", "from_joint", "observed_joint", "target_link", "reason"]:
            if k in fx:
                val = fx[k]
                if k in {"from_link", "to_link", "target_link"}:
                    cid = _extract_canonical_link_id(val)
                    if cid is None:
                        continue
                    if known_links and cid not in known_links:
                        continue
                    val = cid
                if k in {"joint", "from_joint", "observed_joint"}:
                    jid = _extract_canonical_joint_id(val)
                    if jid is None:
                        continue
                    if known_joints and jid not in known_joints:
                        continue
                    val = jid
                row[k] = val
        fixes_out.append(row)
    affected_segments = []
    for x in (obj.get("affected_timeline_segments") or []):
        try:
            xi = int(x)
        except Exception:
            continue
        if xi < 0:
            continue
        if xi not in affected_segments:
            affected_segments.append(xi)
    state_targets = []
    for st in (obj.get("affected_states") or []):
        if not isinstance(st, dict):
            continue
        state = str(st.get("state") or "")
        if state not in ALLOWED_STATE_TARGETS or state not in allowed_state_targets:
            continue
        row = {"state": state}
        has_location = False
        try:
            seg_idx = int(st.get("segment_index"))
            if seg_idx >= 0:
                row["segment_index"] = seg_idx
                has_location = True
        except Exception:
            pass
        phase_type = _sanitize_phase_type(st.get("phase_type"))
        if phase_type is not None:
            row["phase_type"] = phase_type
            has_location = True
        if not has_location:
            continue
        if st.get("joint"):
            row["joint"] = str(st.get("joint"))
        if st.get("segment_name"):
            row["segment_name"] = str(st.get("segment_name"))
        if st.get("why"):
            row["why"] = str(st.get("why"))[:200]
        state_targets.append(row)

    conf = obj.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    param_fix_hints = _sanitize_param_fix_hints(
        obj.get("param_fix_hints"),
        known_joints=known_joints,
        allowed_hint_types=allowed_hint_types,
    )
    repairability = _sanitize_repairability(obj.get("repairability"), has_hints=bool(param_fix_hints))
    detailed_reasoning = _format_detailed_reasoning(str(obj.get("detailed_reasoning") or ""))[:2000]
    return _prune_report_fields({
        "semantic_ok": bool(obj.get("semantic_ok", False)),
        "visibility_ok": bool(obj.get("visibility_ok", True)),
        "link_motion_type": _sanitize_link_motion_type(obj.get("link_motion_type")),
        "detailed_reasoning": detailed_reasoning,
        "issues": issues_out,
        "repairability": repairability,
        "param_fix_hints": param_fix_hints,
        "suggested_fixes": fixes_out,
        "affected_timeline_segments": affected_segments,
        "affected_states": state_targets,
        # Optional: VLM may provide direct patch, but downstream patcher can ignore it.
        "proposed_param_patch": _sanitize_param_patch(obj.get("proposed_param_patch")),
        "confidence": conf,
    })


def neutral_motion_report(reason: str = "Motion VLM diagnosis did not return an actionable report.") -> dict:
    return _prune_report_fields({
        "semantic_ok": True,
        "visibility_ok": True,
        "link_motion_type": "static_or_unclear",
        "detailed_reasoning": reason,
        "issues": [],
        "repairability": {
            "param_fixable": False,
            "structural_fix_needed": False,
            "preferred_repair_level": "none",
        },
        "param_fix_hints": [],
        "suggested_fixes": [],
        "affected_timeline_segments": [],
        "affected_states": [],
        "proposed_param_patch": None,
        "confidence": 0.0,
    })


def _merge_motion_reports(reports: list[dict]) -> dict:
    rows = [r for r in reports if isinstance(r, dict)]
    if not rows:
        return neutral_motion_report("No motion VLM sub-report was produced.")

    def _dedupe_dicts(items: list[dict]) -> list[dict]:
        out = []
        seen = set()
        for item in items:
            key = json.dumps(item, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    issues = _dedupe_dicts([it for r in rows for it in (r.get("issues") or []) if isinstance(it, dict)])
    fixes = _dedupe_dicts([it for r in rows for it in (r.get("suggested_fixes") or []) if isinstance(it, dict)])
    hints = _dedupe_dicts([it for r in rows for it in (r.get("param_fix_hints") or []) if isinstance(it, dict)])
    states = _dedupe_dicts([it for r in rows for it in (r.get("affected_states") or []) if isinstance(it, dict)])

    affected_segments = []
    for r in rows:
        for seg in (r.get("affected_timeline_segments") or []):
            try:
                seg_i = int(seg)
            except Exception:
                continue
            if seg_i not in affected_segments:
                affected_segments.append(seg_i)
    affected_segments.sort()

    param_fixable = any(bool((r.get("repairability") or {}).get("param_fixable", False)) for r in rows)
    preferred_levels = {str((r.get("repairability") or {}).get("preferred_repair_level") or "").strip().lower() for r in rows}
    preferred_levels.discard("")
    preferred_levels = {p for p in preferred_levels if p in ALLOWED_REPAIR_LEVEL}
    if len(preferred_levels) == 1:
        preferred = next(iter(preferred_levels))
    elif preferred_levels:
        preferred = "param" if "param" in preferred_levels else "none"
    else:
        preferred = "param" if param_fixable else "none"

    detailed_reasoning = [_format_detailed_reasoning(str(r.get("detailed_reasoning") or "")) for r in rows if str(r.get("detailed_reasoning") or "").strip()]
    confidence_vals = []
    for r in rows:
        try:
            confidence_vals.append(float(r.get("confidence", 0.5)))
        except Exception:
            pass
    confidence = float(sum(confidence_vals) / len(confidence_vals)) if confidence_vals else 0.5
    confidence = max(0.0, min(1.0, confidence))

    merged_detailed_reasoning = "\n\n".join([x for x in detailed_reasoning if x])[:2000]
    return _prune_report_fields({
        "semantic_ok": all(bool(r.get("semantic_ok", False)) for r in rows),
        "visibility_ok": all(bool(r.get("visibility_ok", True)) for r in rows),
        "detailed_reasoning": merged_detailed_reasoning,
        "issues": issues,
        "repairability": {
            "param_fixable": param_fixable,
            "structural_fix_needed": False,
            "preferred_repair_level": preferred,
        },
        "param_fix_hints": hints,
        "suggested_fixes": fixes,
        "affected_timeline_segments": affected_segments,
        "affected_states": states,
        "proposed_param_patch": None,
        "confidence": confidence,
    })


def _build_prompt(
    action_text: str,
    numeric_report: dict,
    plan_summary: dict,
    coverage_report: dict | None = None,
    timeline_sample_catalog: list[dict] | None = None,
    scale_context: dict | None = None,
    trajectory_summary: dict | None = None,
    conditioning_mask_meta: dict | None = None,
    conditioning_text: str | None = None,
    iteration_index: int | None = None,
    max_iterations: int | None = None,
    previous_motion_reports: list[dict] | None = None,
) -> str:
    lines = []
    motion_mode = "per_link_head_tail_pair"
    per_link_pair_mode = True
    lines.append("You are a motion diagnosis assistant for animation loop correction.")
    lines.append("Goal: check whether motion is BOTH (1) action-consistent and (2) physically/plausibly realistic.")
    lines.append("You must output STRICT JSON only. Do not output prose.")
    lines.append("")
    lines.append("How to Read the Motion Images:")
    if per_link_pair_mode:
        lines.append("- Each request diagnoses exactly ONE link/joint in ONE timeline segment.")
        lines.append("- The request may contain a HEAD image and a TAIL image for the same link/joint in the same segment.")
        lines.append("- Each attached image is named as TIMELINE_SAMPLE_n:<filename>. The substring after ':' is the exact PNG filename; use that exact filename to match timeline_sample_catalog.file_name.")
        lines.append("- Per-link filenames use the form timeline_seg_<seg>_<kind>_<joint_token>_<link_token>_f<frame>.png.")
        lines.append("- HEAD image: identity grounding only. It shows the current link bbox/label and the coordinate legend.")
        lines.append("- TAIL image: motion diagnosis image. It shows the same link later in the same segment and contains the local motion cues.")
        lines.append("- The colored bbox and the local motion cue use the same color family. Use that shared color to match the cue to the current link.")
        lines.append("- A separate black bbox may indicate the overall asset/body extent. Treat the black bbox as global context only, not as the current link bbox.")
        lines.append("")
        lines.append("Authoritative information sources:")
        lines.append("- Local motion source #1: trajectory_summary.local_motion for the current link/joint.")
        lines.append("- Current-view disambiguation source #1: trajectory_summary.local_motion_current_view.axis_projection_note / axis_projection_tag when available.")
        lines.append("- Readable signed-axis / DOT OUT / CROSS IN badge text in the TAIL image is a low-priority visual verification cue.")
        lines.append("- It may support trajectory_summary.local_motion and trajectory_summary.local_motion_current_view when readable, but it must not override them.")
        lines.append("- The drawn local arrow shape in the TAIL image is NOT authoritative. It is only a low-priority verification cue.")
        lines.append("- Global motion source #1: the overall/base motion arrow in the same image, plus its signed axis label if present.")
        lines.append("- Optical flow is ONLY a path/continuity/realism cue. It is NOT the primary source for local motion sign.")
        lines.append("")
        lines.append("Top-right axis legend:")
        lines.append("- The top-right legend is the authoritative coordinate reference: +X is red, +Y is green, +Z is blue.")
        lines.append("- If an axis is nearly parallel to the camera and cannot be drawn as a long arrow, it may appear as a small circle marker instead.")
        lines.append("- Dot-in-circle means the POSITIVE axis points out of the screen toward the camera.")
        lines.append("- Cross-in-circle means the POSITIVE axis points into the screen away from the camera.")
        lines.append("- When the relevant axis is shown with a dot/cross marker, you MUST read that marker. It is the decisive current-view axis-direction cue.")
        lines.append("")
        lines.append("Rotation cue (circular arrow) — detailed rule:")
        lines.append("- A circular arrow means rotational motion of the current link/joint.")
        lines.append("- IMPORTANT: do NOT decide rotation direction from the circular arrow shape alone.")
        lines.append("- First read the NAMED signed axis for this rotation, e.g. +X, -Y, +Z.")
        lines.append("- Then read the current-view projection cue for that same axis: DOT OUT / CROSS IN, either from trajectory_summary.local_motion_current_view or from the image badge/legend.")
        lines.append("- For rotational cases, the projection cue must be interpreted for the NAMED SIGNED AXIS itself, not for the unsigned positive axis.")
        lines.append("- Examples:")
        lines.append("-   if the case says '+X DOT OUT', then the named signed axis +X points out of screen")
        lines.append("-   if the case says '-X DOT OUT', then the named signed axis -X points out of screen")
        lines.append("-   if the case says '+Y CROSS IN', then the named signed axis +Y points into screen")
        lines.append("-   if the case says '-Z CROSS IN', then the named signed axis -Z points into screen")
        lines.append("- Do NOT convert the cue back to the positive axis before judging direction.")
        lines.append("- Always apply the keep/flip rule to the named signed axis shown in the local motion cue.")
        lines.append("- Only AFTER the axis and its current-view projection are known, read the circular arrow shape.")
        lines.append("- EXPLICIT CONVERSION RULE BEFORE ANY CORRECT/INCORRECT JUDGMENT: convert trajectory axis-relative direction into the current-view screen direction.")
        lines.append("- If trajectory direction is CW and the named signed axis is DOT OUT, current-view direction is CW.")
        lines.append("- If trajectory direction is CCW and the named signed axis is DOT OUT, current-view direction is CCW.")
        lines.append("- If trajectory direction is CW and the named signed axis is CROSS IN, current-view direction is CCW.")
        lines.append("- If trajectory direction is CCW and the named signed axis is CROSS IN, current-view direction is CW.")
        lines.append("- In short: DOT OUT keeps trajectory CW/CCW; CROSS IN flips trajectory CW/CCW. If trajectory_summary.local_motion_current_view.current_view_direction exists, use it as the authoritative converted current-view direction.")
        lines.append("- [Trajectory] is axis-relative. [Conversion] must be current-view screen direction after DOT/CROSS conversion. Do NOT write the original axis-relative direction again in [Conversion].")
        lines.append("- If the arrow shape conflicts with the converted current-view direction, mark [Apparent] as ambiguous / low-confidence and keep [Conversion] aligned to trajectory_summary.local_motion_current_view.current_view_direction.")
        lines.append("- Therefore, interpret rotation as: rotation about the NAMED SIGNED AXIS, then resolve its rendered clockwise/counterclockwise appearance in the current view.")
        lines.append("- If a compact local badge exists inside or next to the circular arrow, and it contains both the signed axis and the dot/cross projection cue, treat that badge as the primary visual disambiguation cue for this rotational case.")
        lines.append("")
        lines.append("Prismatic cue (straight arrow) — detailed rule:")
        lines.append("- A straight arrow means translational motion of the current link/joint.")
        lines.append("- The true motion is translation ALONG the named signed axis, such as +X, -Y, +Z.")
        lines.append("- The drawn straight arrow is only the screen-space projection of that signed-axis motion in the current camera view.")
        lines.append("- Therefore, do NOT read the straight arrow as a pure image-plane direction by itself.")
        lines.append("- First read the nearby signed axis text, e.g. ALONG -Y, +X, -Z.")
        lines.append("- Then use the top-right axis legend to verify how that signed axis projects into the current camera view.")
        lines.append("- Only then use the drawn straight arrow as a consistency check for the projected direction.")
        lines.append("- Prismatic motion does NOT require a DOT OUT / CROSS IN cue.")
        lines.append("- For prismatic cases, if no dot/cross projection marker exists, treat DOT/CROSS as not applicable and judge direction from the signed axis plus its visible screen-space projection.")
        lines.append("- Do NOT force a CROSS/DOT conversion step for prismatic motion.")
        lines.append("- For prismatic motion, DOT OUT / CROSS IN is optional and may be absent.")
        lines.append("- For prismatic motion, [Conversion] must be none / not applicable.")
        lines.append("")
        lines.append("Mandatory magnitude / speed judgment for articulated cases:")
        lines.append("- Magnitude/speed judgment is REQUIRED for every articulated case, including ALL rotational cases.")
        lines.append("- Direction correctness and magnitude correctness are independent. A case with correct direction may still be EXCESSIVE_MOTION or MOTION_TOO_SMALL.")
        lines.append("- Magnitude judgment is numeric-first. Use trajectory_summary.timeline_segment_motion, trajectory_summary.joints_summary, plan_summary.timeline.controls, plan_summary.joint_limits, and scale_context as the authoritative magnitude references for the current link/joint and current segment.")
        lines.append("- Do NOT judge magnitude mainly from visual apparent size, arrow length, or screen-space displacement. Camera view, asset scale, and part size make visual size unreliable across assets.")
        lines.append("- For rotational cases, read segment start_q/end_q/delta_q/delta_q_magnitude in radians. Also read lower_limit/upper_limit and compute limit_range=abs(upper_limit-lower_limit) when available, then normalized_delta=delta_q_magnitude/limit_range.")
        lines.append("- For prismatic cases, read segment start_q/end_q/delta_q/delta_q_magnitude as translation in URDF units. Normalize by the relevant joint limit range when available; otherwise use scale_context joint_child_link_bbox_extents_m / link_bbox_extents_m / object_diag_m to judge whether the translation is plausible for this asset.")
        lines.append("- Compare numeric actual magnitude from trajectory_summary against numeric intended magnitude from plan_summary.timeline.controls and effects joint_targets. The TAIL image is only a consistency check for visibility/path, not the source of the magnitude value.")
        lines.append("- Do NOT silently accept motion magnitude just because the direction is correct.")
        lines.append("- If numeric intended magnitude is clearly larger than numeric actual trajectory magnitude, use MOTION_TOO_SMALL.")
        lines.append("- If numeric intended magnitude is clearly smaller than numeric actual trajectory magnitude, use EXCESSIVE_MOTION.")
        lines.append("- If numeric magnitude evidence is missing or ambiguous, say so explicitly in [Magnitude] and reduce confidence; do not replace missing numeric evidence with a visual guess.")
        lines.append("")
        lines.append("Rule for unexpected extra motion:")
        lines.append("- UNEXPECTED_EXTRA_MOTION is an independent error type.")
        lines.append("- It must be judged explicitly from whether the current link shows motion that is not required by the intended action or current segment semantics.")
        lines.append("- A case may simultaneously have WRONG_DIRECTION and UNEXPECTED_EXTRA_MOTION, or EXCESSIVE_MOTION and UNEXPECTED_EXTRA_MOTION.")
        lines.append("- Do NOT suppress UNEXPECTED_EXTRA_MOTION just because direction or magnitude has already been analyzed.")
        lines.append("- If no extra motion is present, explicitly treat UNEXPECTED_EXTRA_MOTION as not applicable in the shared reasoning.")
        lines.append("")
        lines.append("Priority rules:")
        lines.append("- Hard priority rule for local motion direction:")
        lines.append("- The final direction verdict MUST be determined from authoritative sources in this order:")
        lines.append("-   (1) trajectory_summary.local_motion")
        lines.append("-   (2) trajectory_summary.local_motion_current_view.axis_projection_note / axis_projection_tag / axis_projection")
        lines.append("-   (3) readable signed-axis / DOT OUT / CROSS IN badge text in the image")
        lines.append("- The drawn circular/straight arrow shape is NOT authoritative. It is only a low-priority verification cue.")
        lines.append("- If the arrow shape appears to disagree with trajectory_summary or the readable badge text, assume the arrow shape may have been visually misread and trust the authoritative sources instead.")
        lines.append("- NEVER let the apparent arrow shape override trajectory_summary.local_motion or the current-view projection cue.")
        lines.append("- Priority for local motion sign is STRICTLY split into two parts:")
        lines.append("-   (A) axis-relative intended rotation/translation sign comes ONLY from trajectory_summary.local_motion.")
        lines.append("-   (B) current-view disambiguation comes ONLY from trajectory_summary.local_motion_current_view.axis_projection_note / axis_projection_tag / axis_projection.")
        lines.append("- Do NOT use trajectory_summary.local_motion_current_view.direction as the final answer for rotation direction.")
        lines.append("- Do NOT interpret trajectory_summary.local_motion.direction as a screen-space clockwise/counterclockwise label.")
        lines.append("- local arrow/tag/badge is only a verification cue after steps (A) and (B) are resolved.")
        lines.append("- Use optical flow only for motion path shape, continuity, and realism. Never use optical-flow orientation alone to decide local sign.")
        lines.append("- Optical flow is last and is only for continuity/realism, never for the primary local sign.")
        lines.append("- If a reliable global/base transport cue exists, use it only as optional auxiliary evidence for realism and consistency.")
        lines.append("- If no reliable global/base transport cue exists, do not force any global-direction judgment.")
        lines.append("")
        lines.append("Forbidden shortcuts for rotational cases:")
        lines.append("- NEVER map trajectory_summary.local_motion.direction directly to the current camera-view clockwise/counterclockwise.")
        lines.append("- NEVER use the circular arrow shape alone as the answer.")
        lines.append("- NEVER use trajectory_summary.local_motion_current_view.direction alone as the answer.")
        lines.append("- The only valid rotational decision order is:")
        lines.append("-   1) read trajectory_summary.local_motion.axis_label and trajectory_summary.local_motion.direction")
        lines.append("-   2) read current-view projection cue (DOT OUT / CROSS IN)")
        lines.append("-   3) read the apparent on-screen circular arrow direction only if it is visually clear and consistent enough; otherwise record [Apparent] as ambiguous / low-confidence")
        lines.append("-   4) derive [Conversion] as the current-view screen direction from trajectory direction plus CROSS/DOT rule; if trajectory_summary.local_motion_current_view.current_view_direction exists, copy that value")
        lines.append("-   5) compare the authoritative converted current-view direction against the image/optical-flow consistency cues and intended action/plan semantics")
        lines.append("- NEVER compare the apparent on-screen CW/CCW directly against trajectory_summary.local_motion.direction.")
        lines.append("- You must first resolve the named signed axis projection and complete the authoritative keep/flip conversion.")
        lines.append("")
        lines.append("Mandatory rotational decision template:")
        lines.append("- Step 1: Intended local motion from trajectory_summary.local_motion:")
        lines.append("-         identify axis_label and axis-relative direction.")
        lines.append("- Step 2: Current-view projection cue:")
        lines.append("-         determine whether the NAMED SIGNED AXIS in this case is DOT OUT or CROSS IN.")
        lines.append("- Step 3: Apparent local circular-arrow direction in the image:")
        lines.append("-         read the naive on-screen CW/CCW only if visually clear and consistent; if it conflicts with authoritative sources, write ambiguous / low-confidence.")
        lines.append("- Step 4: Conversion:")
        lines.append("-         derive the converted current-view screen result from trajectory direction plus CROSS/DOT rule; if CROSS IN => flip cw<->ccw, if DOT OUT => keep cw/ccw. Do not derive [Conversion] from a conflicting arrow shape.")
        lines.append("- Step 5: Optional global/base transport cue:")
        lines.append("-         if a reliable overall/base motion cue exists in the image or trajectory_summary, use it as auxiliary evidence only.")
        lines.append("-         if no reliable overall/base motion cue exists, skip this step.")
        lines.append("- Step 6: Final consistency check:")
        lines.append("-         compare the converted result against trajectory_summary.local_motion and against the intended action/plan semantics.")
        lines.append("-         Global/base motion is optional evidence, not a required prerequisite.")
        lines.append("")
        lines.append("Rule when there is NO overall/base transport motion:")
        lines.append("- If the asset has no reliable overall/base motion cue, do NOT infer one.")
        lines.append("- In that case, judge correctness only from:")
        lines.append("-   (A) action_text and plan_summary semantics")
        lines.append("-   (B) trajectory_summary.local_motion")
        lines.append("-   (C) current-view projection cue (DOT OUT / CROSS IN)")
        lines.append("- The local arrow/tag/badge in the TAIL image may be used only as optional low-priority support.")
        lines.append("- It must not determine the final verdict.")
        lines.append("- Do NOT mark a case incorrect merely because no global/base transport cue is present.")
        lines.append("- IMPORTANT: global/base transport is OPTIONAL evidence, not a required prerequisite for correctness judgment.")
        lines.append("- IMPORTANT: when no global/base transport cue exists, the correctness judgment must still be made from action semantics, plan semantics, trajectory_summary.local_motion, and the DOT OUT / CROSS IN conversion rule.")
        lines.append("")
        lines.append("Rule for release / return errors:")
        lines.append("- NO_RELEASE_RETURN is an independent error type. Do NOT suppress it just because direction or speed is otherwise correct.")
        lines.append("- If the current case includes a link/control that is expected to return after release, you MUST check return completeness separately from direction and magnitude.")
        lines.append("- A case may simultaneously have WRONG_DIRECTION and NO_RELEASE_RETURN, or EXCESSIVE_MOTION and NO_RELEASE_RETURN.")
        lines.append("- If no release/return behavior is expected for the current link/control, explicitly treat NO_RELEASE_RETURN as not applicable rather than forcing a return judgment.")
        lines.append("- Return judgment is trajectory-based, not image-impression-based.")
        lines.append("- For [Return], you MUST read the current joint's URDF lower_limit/upper_limit when available, the intended return/rest position from plan_summary, and the final-frame joint position from trajectory_summary.")
        lines.append("- [Return] MUST explicitly state: lower_limit=..., upper_limit=..., rest/target_return=..., final_q=..., remaining_error=..., and whether that error is small enough to count as fully returned.")
        lines.append("- If the return target is rest_position=0, judge whether final_q is numerically close to 0 relative to the joint limit range; do not judge from the visual image alone.")
        lines.append("- If the return target is lower_limit or upper_limit, judge final_q against that limit numerically using trajectory_summary and the URDF limit range.")
        lines.append("- For revolute/prismatic joints with a known limit range, a return is complete only if the final error is small relative to that range; if the final position remains visibly/numerically far from the target, report NO_RELEASE_RETURN.")
        lines.append("- Do NOT treat decay_estimate as final return completion. decay_estimate only compares this segment's motion magnitude to a previous segment; it does not say where the joint ended.")
        lines.append("- If trajectory_summary does not provide final_q or the relevant joint limit/range, say [Return] is ambiguous / insufficient trajectory data, and do not pretend visual appearance alone proves full return.")
        lines.append("- For NO_RELEASE_RETURN, actively estimate how much more return time is needed from trajectory_summary.timeline_segment_motion for the current return segment.")
        lines.append("- Compute observed_return_delta = abs(start_q - end_q) for the return segment, remaining_error = abs(final_q - rest/target_return), and required_time_multiplier ~= 1.0 + remaining_error / max(observed_return_delta, 1e-6).")
        lines.append("- Choose the adjust_timing strength by matching required_time_multiplier to the fixed strength multipliers: small=1.10x, medium=1.20x, large=1.35x, extra_large=2.00x.")
        lines.append("- Put this trajectory-based return calculation and judgment inside the [Return] section of detailed_reasoning. Do not defer the calculation to issues.")
        lines.append("")
        lines.append("Required reading procedure for the current case:")
        lines.append("  1. Read the exact filename and match timeline_sample_catalog.file_name.")
        lines.append("  2. Confirm segment_index, segment_name, link, joint, and single_link_trace from that catalog row.")
        lines.append("  3. Read trajectory_summary.local_motion for the current link/joint.")
        lines.append("  4. If trajectory_summary.local_motion_current_view exists, read its axis_projection_note / axis_projection_tag.")
        lines.append("  5. In the TAIL image, find the current link bbox and its local motion cue.")
        lines.append("  6. For rotation: read signed axis first, then DOT/CROSS projection, then APPLY THE ROTATIONAL CONVERSION RULE before reading the circular arrow result.")
        lines.append("  7. Rotational conversion rule only: convert trajectory direction to current-view direction; CROSS IN => flip cw<->ccw, DOT OUT => keep cw/ccw. If trajectory_summary.local_motion_current_view.current_view_direction exists, use that value. If the arrow shape conflicts, keep [Conversion] authoritative and write [Apparent] as ambiguous / low-confidence.")
        lines.append("  8. For prismatic motion: read signed axis first, then its visible projected direction, then straight arrow. Do NOT apply any DOT/CROSS conversion step unless a projection marker is explicitly present.")
        lines.append("  9. Compare the local motion against the global/base transport cue in the same image.")
        lines.append("  10. Use optical flow only to judge continuity and realism of the path.")
        lines.append("")
        lines.append("Important interpretation constraints:")
        lines.append("- Local motion must be interpreted relative to the NAMED SIGNED AXIS, not directly from the arrow shape alone.")
        lines.append("- Do NOT infer direction from where the local arrow is placed around the bbox. It may be moved to any nearby blank area for readability.")
        lines.append("- Do NOT transfer a direction verdict from another link or another image. Judge only the current link/joint in the current case.")
        lines.append("- If local arrow/tag and trajectory_summary disagree, trust trajectory_summary.local_motion first, then trajectory_summary.local_motion_current_view, then the local visual badge/arrow, and place optical flow last.")
        lines.append("- If the image is visually ambiguous, say so through visibility_ok / confidence rather than inventing a direction.")
        lines.append("- If the asset also has overall/base transport motion, use the overall asset-motion arrow only as the global transport cue, not as the local joint cue.")
        lines.append("- SPEED RULE: use trajectory_summary and scale_context for magnitude/speed judgment. Do NOT infer speed from arrow length alone.")
    lines.append("")
    lines.append("Your task:")
    if per_link_pair_mode:
        lines.append("- Diagnose ONLY the current per-link case defined by the attached HEAD/TAIL image pair and the matching timeline_sample_catalog rows.")
        lines.append("- Use the HEAD image to verify which link bbox this case refers to in the current case.")
        lines.append("- Use the TAIL image to judge the motion direction, motion magnitude, and realism for that same link in that same timeline segment.")
        lines.append("- Judge local part direction mainly from trajectory_summary.local_motion for the signed axis, then from trajectory_summary.local_motion_current_view.axis_projection_note / axis_projection_tag for the current-view DOT OUT / CROSS IN cue.")
        lines.append("- In the image, explicitly verify that same DOT OUT / CROSS IN marker before reading the circular arrow shape.")
        lines.append("- Use the current local arrow/tag only after trajectory_summary + DOT OUT / CROSS IN are resolved; treat the arrow/tag as a consistency check, not as a higher-priority source.")
        lines.append("- If the local arrow shape is visually ambiguous, partially occluded, or inconsistent with the authoritative sources, do NOT guess from the arrow shape.")
        lines.append("- In that case, rely on trajectory_summary + current-view projection cue, and lower confidence if needed.")
        lines.append("- IMPORTANT: if the apparent arrow shape conflicts with trajectory_summary.local_motion or with a readable signed-axis / DOT OUT / CROSS IN badge, trust trajectory_summary and the readable badge, write [Apparent] as ambiguous / low-confidence, and derive [Conversion] from the authoritative signed-axis rule rather than from the conflicting arrow shape.")
        lines.append("- For prismatic links specifically, remember that the straight arrow is only the current-view projection of the signed axis motion; the signed axis in trajectory_summary.local_motion remains authoritative.")
        lines.append("- Use optical-flow only as trajectory/path reference for how the motion evolves over time. Do NOT use optical-flow as the main basis for direction sign.")
        lines.append("- You MUST explicitly output link_motion_type in JSON for this current link.")
        lines.append("- You MUST explicitly output detailed_reasoning in the following fixed format:")
        lines.append('-   [Trajectory] axis_label=..., axis-relative direction=...\\n ')
        lines.append('-   [Projection] for rotational motion: DOT OUT or CROSS IN = ... ; for prismatic motion: projected signed-axis direction = ... OR "none / not applicable"\\n ')
        lines.append('-   [Apparent] arrow shape read from image = ... OR "ambiguous / low-confidence" when it conflicts with authoritative sources; for prismatic motion: visible straight-arrow direction = ... OR "none / not applicable"\\n ')
        lines.append('-   [Conversion] for rotational motion: authoritative current-view screen direction after applying DOT/CROSS to trajectory direction = ... ; if trajectory_summary.local_motion_current_view.current_view_direction exists, use exactly that value ; if [Apparent] conflicts, do not derive this field from the conflicting arrow shape ; for prismatic motion: "none / not applicable"\\n ')
        lines.append('-   [Magnitude] joint_type=...; segment_start_q=...; segment_end_q=...; delta_q=...; delta_q_magnitude=...; lower_limit=...; upper_limit=...; limit_range=...; normalized_delta=...; intended_target_or_speed=...; scale_context_used=...; numeric_actual_vs_intended=...; image_consistency_check=...; verdict = ...\\n ')
        lines.append('-   [ExtraMotion] any unintended or unrelated motion for the current link in this segment = ... ; verdict = ...\\n ')
        lines.append('-   [Global] overall/base motion direction = ... OR "none / not applicable"\\n ')
        lines.append('-   [Return] expected release/return behavior = ... OR "none / not applicable"; lower_limit=...; upper_limit=...; rest/target_return=...; segment_start_q=...; segment_end_q=...; observed_return_delta=...; final_q=...; remaining_error=...; required_time_multiplier=...; selected_adjust_timing_strength=...; verdict = ...')
        lines.append("- Do not skip any of the above fields.")
        lines.append('- For prismatic motion, [Projection] must describe the projected signed-axis direction or "none / not applicable", and [Conversion] must be "none / not applicable".')
        lines.append('- If no global/base transport cue exists, explicitly write none / not applicable for [Global].')
        lines.append('- If no release/return behavior is expected, explicitly write none / not applicable for [Return].')
        lines.append('- If no unintended or unrelated motion is present, explicitly write that in [ExtraMotion].')
        lines.append("- You MUST complete the shared reasoning template before deciding any final verdict.")
        lines.append("- Only after finishing [Trajectory], [Projection], [Apparent], [Conversion], [Magnitude], [ExtraMotion], [Global], and [Return] may you output:")
        lines.append("-   link_motion_type")
        lines.append("-   issues")
        lines.append("-   param_fix_hints")
        lines.append("- The final verdict must be derived from the completed reasoning, not guessed in advance.")
        lines.append("- [Apparent] is only a visual parse field. It does NOT decide the final verdict by itself, and if it conflicts with authoritative sources it must be written as ambiguous / low-confidence.")
        lines.append("- If [Trajectory], [Projection], and [Conversion] already determine the direction unambiguously, [Apparent] is optional and must not change link_motion_type, issues, or semantic_ok.")
        lines.append('- If [Trajectory] and [Projection] already determine the direction unambiguously, [Apparent] should be recorded as "not needed / optional low-priority support" unless the arrow shape is exceptionally clear.')
        lines.append("- Use trajectory_summary to verify sign, current-view axis projection, timing, and speed magnitude for the current link/joint only, and prefer it whenever there is any conflict with the arrow appearance.")
        lines.append("- Use scale_context only to judge whether the observed motion size/speed is plausible for this link and the whole asset.")
        lines.append("- Do NOT transfer a verdict from another link case. This request is one link, one timeline segment, one verdict.")
        lines.append("- Diagnose only (do not directly rewrite the plan).")
        lines.append("- If MASK_* inputs are attached, they are target masks; use them to determine the intended target link.")
        lines.append("- IMPORTANT: In issues/param_fix_hints, link/joint identifiers MUST use canonical IDs only (e.g., link_5, joint_5).")
        lines.append("- Do NOT output natural-language names or placeholders as IDs.")
        lines.append("- Choose IDs only from the identifiers present in the filtered metadata below.")
    lines.append("")
    if per_link_pair_mode:
        lines.append("Allowed issue codes for per-link head/tail diagnosis (use only if applicable, choose from this list exactly):")
        lines.append("- WRONG_DIRECTION")
        lines.append("- UNEXPECTED_EXTRA_MOTION")
        lines.append("- EXCESSIVE_MOTION")
        lines.append("- MOTION_TOO_SMALL")
        lines.append("- NO_RELEASE_RETURN")
        lines.append("")
        lines.append("Issue code interpretation (authoritative):")
        lines.append("- WRONG_DIRECTION: the current boxed link moves in the wrong sign/direction relative to the intended action for this case.")
        lines.append("- UNEXPECTED_EXTRA_MOTION: the current boxed link shows motion that should be suppressed or absent in this timeline segment.")
        lines.append("- EXCESSIVE_MOTION: the current boxed link moves too much or too fast for the action and object scale.")
        lines.append("- MOTION_TOO_SMALL: the current boxed link moves too little or too slowly for the action and object scale.")
        lines.append("- NO_RELEASE_RETURN: a release/return motion that should come back does not return as expected.")
        lines.append("- For NO_RELEASE_RETURN, judge return completeness mainly from trajectory_summary plus the current joint's URDF limit range when available; do NOT judge only from a vague visual impression.")
        lines.append("- For WRONG_DIRECTION specifically, issues[].expected and issues[].observed MUST be grounded in authoritative sources only.")
        lines.append("- issues[].expected MUST describe the authoritative intended motion from trajectory_summary.local_motion, optionally after applying the current-view projection rule in [Conversion].")
        lines.append("- issues[].observed MUST describe the authoritative interpreted observed result after [Projection] + [Conversion], NOT the raw apparent arrow shape.")
        lines.append("- NEVER populate issues[].expected or issues[].observed directly from [Apparent].")
        lines.append("- If the apparent arrow shape conflicts with authoritative sources, keep issues[].expected / issues[].observed aligned to [Trajectory] / [Projection] / [Conversion], and record the arrow-shape disagreement only inside [Apparent] as ambiguous / low-confidence.")
        lines.append("")
        lines.append("Allowed param_fix_hints.type values for per-link head/tail diagnosis:")
        lines.append("- adjust_joint_target")
        lines.append("- adjust_joint_velocity")
        lines.append("- adjust_timing")
        lines.append("- adjust_direction")
        lines.append("Allowed param_fix_hints.direction values: increase | decrease | flip | zero")
        lines.append("- Use increase/decrease when you want to enlarge or reduce an existing motion magnitude.")
        lines.append("- Use flip when you want to reverse direction/sign only; strength is usually omitted for flip.")
        lines.append("- Use zero when you want to suppress an unexpected extra motion by setting the corresponding motion parameter to zero.")
        lines.append("Allowed param_fix_hints.strength values: small | medium | large | extra_large")
        lines.append("- Strength is mainly for increase/decrease. For flip or zero, strength may be omitted.")
        lines.append("- DEFAULT RULE: if direction is increase or decrease and strength is omitted, downstream patching treats it as strength='medium'.")
        lines.append("- Strength multipliers for increase/decrease: small=1.10x / 0.91x, medium=1.20x / 0.83x, large=1.35x / 0.74x, extra_large=2.00x / 0.50x.")
        lines.append("- For NO_RELEASE_RETURN, choose strength from the [Return] required_time_multiplier calculation. The hint should usually be {type:'adjust_timing', direction:'increase', segment_index:<return segment>, phase_type:'control_release' or 'settle_return', joint:<current joint>, strength:<computed>}.")
        lines.append("Allowed phase_type values (optional locator, choose if available): control_activation | control_release | control_to_effect_lag | effect_motion | settle_return | hold | transport")
        lines.append("")
        lines.append("Output schema:")
        lines.append("- strength demonstration examples for per-link mode:")
        lines.append('  {"type":"adjust_joint_target","joint":"joint_5","segment_index":0,"direction":"increase","strength":"small","why":"the motion is slightly too weak"}')
        lines.append('  {"type":"adjust_joint_velocity","joint":"joint_5","segment_index":0,"direction":"decrease","strength":"large","why":"the motion is far too strong and needs a strong reduction"}')
        lines.append('  {"type":"adjust_joint_velocity","joint":"joint_5","segment_index":0,"direction":"decrease","strength":"extra_large","why":"the motion is drastically too strong and should be cut roughly in half"}')
        lines.append('  {"type":"adjust_direction","joint":"joint_5","segment_index":0,"direction":"flip","why":"the direction sign is reversed; strength is not needed for flip"}')
        lines.append('  {"type":"adjust_joint_velocity","joint":"joint_5","segment_index":0,"direction":"zero","why":"this unexpected extra motion should be removed entirely"}')
        lines.append("{")
        lines.append('  "semantic_ok": true,')
        lines.append('  "visibility_ok": true,')
        lines.append('  "link_motion_type": "rotational_cw",')
        lines.append('  "detailed_reasoning": "[Trajectory] ...\\n [Projection] ...\\n [Apparent] ...\\n [Conversion] ...\\n [Magnitude] ...\\n [ExtraMotion] ...\\n [Global] ...\\n [Return] ...",')
        lines.append('  "repairability": {"param_fixable": true, "structural_fix_needed": false, "preferred_repair_level": "param"},')
        lines.append('  "issues": [')
        lines.append('    {"code":"WRONG_DIRECTION","expected_axis":"+X","observed_axis":"-X"}')
        lines.append('  ],')
        lines.append('  "param_fix_hints": [')
        lines.append('    {"type":"adjust_direction","joint":"joint_5","segment_index":0,"phase_type":"effect_motion","direction":"flip","why":"the current link moves opposite to the intended sign in this camera view"}')
        lines.append('  ],')
        lines.append('  "proposed_param_patch": null,')
        lines.append('  "confidence": 0.0')
        lines.append("}")
        lines.append("")
    lines.append("Important constraints:")
    lines.append("- Use ONLY allowed issue codes and fix types above.")
    lines.append("- MASK priority for target identity: MASK_* regions > canonical IDs in Plan summary > natural-language target name.")
    if per_link_pair_mode:
        lines.append("- In per-link head/tail mode, do NOT output structural or identity/mapping diagnoses.")
        lines.append("- link_motion_type is REQUIRED in every JSON output, even when semantic_ok=true.")
        lines.append("- detailed_reasoning is REQUIRED in every JSON output, and it must explain the conclusion using the HEAD image, the TAIL image, the top-right axis legend, and the filtered metadata for this same case.")
        lines.append("- detailed_reasoning may mention explicit arrows/tags only as optional low-priority support. They are not required evidence for the final direction verdict when authoritative sources already establish the direction.")
        lines.append("- Complete the shared reasoning first; only then derive link_motion_type, issues, and param_fix_hints from that completed reasoning.")
        lines.append("- Use param_fix_hints as the primary actionable output.")
        lines.append("- repairability.preferred_repair_level should normally be 'param', and structural_fix_needed should normally be false in this mode.")
        lines.append("- Do NOT pass a motion as correct if direction appears plausible but the magnitude/speed is physically implausible.")
        lines.append("- Treat realism as first-class: unrealistic magnitude/speed should be flagged with EXCESSIVE_MOTION, MOTION_TOO_SMALL, or UNEXPECTED_EXTRA_MOTION when suppression is appropriate.")
        lines.append("- Judge WRONG_DIRECTION ONLY from [Trajectory], [Projection], and [Conversion].")
        lines.append("- Explicit local/overall arrows may appear only in [Apparent] as low-priority supporting context and must never determine the final direction verdict.")
        lines.append("- Optical flow may support continuity/path realism, but it is not the authoritative cue for direction sign.")
        lines.append("- Localize failures in param_fix_hints using segment_index, phase_type, and joint whenever possible.")
        lines.append("- Prefer segment indices from the current case rows below; do not guess segment ids not shown.")
        lines.append("- issue.code must stay generic; put case-specific details in expected/observed fields or detailed_reasoning.")
        lines.append("- Do NOT invent new issue/fix names. If none applies, return empty issues/param_fix_hints and semantic_ok=true.")
        lines.append("- Keep issues minimal and actionable: 1-3 high-signal issues are preferred.")
        lines.append("- The final issues list must be derived from the completed shared reasoning template.")
        lines.append("- WRONG_DIRECTION must come from [Trajectory], [Projection], and [Conversion]. [Apparent] is optional supporting context only and must not be required when authoritative sources already establish the direction.")
        lines.append("- For WRONG_DIRECTION, if issues[].expected / issues[].observed are present, they must match the authoritative [Trajectory] / [Projection] / [Conversion] chain exactly.")
        lines.append("- For WRONG_DIRECTION, do NOT describe issues[].observed as a naive visual CW/CCW reading when [Conversion] already resolves the motion differently.")
        lines.append("- For rotational cases, link_motion_type must encode the authoritative converted result from [Conversion], not the raw arrow appearance from [Apparent].")
        lines.append("- EXCESSIVE_MOTION and MOTION_TOO_SMALL must come from [Magnitude].")
        lines.append("- issues must not introduce a new magnitude calculation. For EXCESSIVE_MOTION or MOTION_TOO_SMALL, issues may summarize the [Magnitude] verdict only; the numeric evidence belongs in [Magnitude].")
        lines.append("- UNEXPECTED_EXTRA_MOTION must come from [ExtraMotion].")
        lines.append("- NO_RELEASE_RETURN must come from [Return].")
        lines.append("- issues must not introduce a new trajectory calculation. For NO_RELEASE_RETURN, issues may summarize the [Return] verdict only; the numeric evidence belongs in [Return].")
        lines.append("- Do NOT generate patch paths by yourself unless very certain. Prefer param_fix_hints with (joint, segment_index, state).")
        lines.append("- proposed_param_patch is optional; downstream patcher can ignore it.")
        lines.append("- If you provide proposed_param_patch for a base/world-direction correction, target the affected timeline base controls' axis_world fields.")
        lines.append("- Set semantic_ok=true ONLY when both action-consistency and realism checks pass.")
        lines.append("- For NO_RELEASE_RETURN, prefer adjust_timing first: extend or shift the relevant control_release / settle_return time window so the return can finish on screen.")
        lines.append("- For NO_RELEASE_RETURN, first check the numeric return evidence in [Return]: current joint limit range, return target/rest position, final_q, remaining_error, and return completion. The image alone is not enough.")
        lines.append("- If a full return is required for this link/control, require the timeline to allocate enough control_release / settle_return time for final_q to get numerically close to the return target.")
        lines.append("- Do NOT omit NO_RELEASE_RETURN merely because another issue code is already present; return-related failure is independent and should still be reported when applicable.")
        lines.append("- Use direction='zero' in param_fix_hints when UNEXPECTED_EXTRA_MOTION should be removed by zeroing the responsible motion parameter.")
    lines.append("")
    lines.append("Structured metadata references:")
    lines.append("- Timeline sample catalog entries include file_name, segment_index, segment_name, phase_type, link, joint, single_link_trace, referenced_links, and motion_cues; use these for precise localization.")
    lines.append("- single_link_trace=true means that image is intentionally split out for exactly one link/joint; in per-link head/tail mode, the current case should normally include one HEAD row and one TAIL row for that same link.")
    lines.append("- file_name is the authoritative bridge between the attached image name TIMELINE_SAMPLE_n:<filename> and the catalog row; always match by exact filename first.")
    lines.append("- When visual texture is ambiguous, prioritize motion_cues + timeline ordering for semantic judgment.")
    lines.append("- motion_cues.joint_trends schema example (revolute/prismatic):")
    lines.append('  {"joint":"joint_1","link":"link_1","joint_type":"revolute","trend":"increase","delta_q":1.32}')
    lines.append('  {"joint":"joint_5","link":"link_5","joint_type":"prismatic","trend":"increase","delta_q":0.08}')
    lines.append("- motion_cues.base_motion schema example:")
    lines.append('  {"body":"base","motion_type":"translation","axis_world":[-1,0,0],"trend":"negative","delta_proj":-0.12}')
    lines.append("- plan_summary.joint_limits schema example for the current case:")
    lines.append('  {"joint_5":{"lower":0.0,"upper":1.57,"velocity":2.0}}')
    lines.append("- trajectory_summary.local_motion schema example for the current case:")
    lines.append('  {"motion_type":"rotation","direction":"cw","axis_label":"+X","local_motion_text":"cw around +X","frame_note":"axis_relative_not_view_relative"}')
    lines.append("- trajectory_summary.local_motion_current_view schema example for the current case:")
    lines.append('  {"axis_label":"+Y","axis_projection":"cross_in","axis_projection_note":"CROSS IN","axis_projection_tag":"+Y CROSS IN"}')
    lines.append("- trajectory_summary.local_motion and any per-joint local_motion objects are authoritative for local motion sign and axis interpretation.")
    lines.append("")
    legend = plan_summary.get("image_label_legend") or {}
    if isinstance(legend, dict) and legend:
        lines.append("Image label legend (canonical link -> shown image label):")
        for k, v in legend.items():
            lines.append(f"- {str(k)} -> {str(v)}")
        lines.append("- IMPORTANT: image labels may be shortened aliases; always report canonical IDs in JSON outputs.")
    lines.append("")
    lines.append("User action text:")
    lines.append(action_text.strip())
    cond_text = str(conditioning_text or "").strip()
    if cond_text and cond_text != str(action_text or "").strip():
        lines.append("")
        lines.append("Additional conditioning text:")
        lines.append(cond_text)
    if _has_effective_mask_meta(conditioning_mask_meta):
        lines.append("")
        lines.append("Conditioning mask metadata (MASK_* attachments are target masks):")
        lines.append(json.dumps(conditioning_mask_meta, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("Plan summary JSON:")
    lines.append(json.dumps(plan_summary, ensure_ascii=False, indent=2))
    if trajectory_summary is not None:
        lines.append("")
        lines.append("Trajectory summary JSON:")
        lines.append(json.dumps(trajectory_summary, ensure_ascii=False, indent=2))
    if scale_context is not None:
        lines.append("")
        lines.append("Scale context JSON:")
        lines.append(json.dumps(scale_context, ensure_ascii=False, indent=2))
    # Motion diagnosis should focus on trajectory and timeline-linked evidence.
    # Coverage was handled in a previous loop stage and is intentionally omitted here.
    if timeline_sample_catalog:
        lines.append("")
        lines.append("Timeline sample catalog:")
        lines.append(json.dumps(timeline_sample_catalog, ensure_ascii=False, indent=2))
    return "\n".join(lines)


def diagnose_via_api(
    action_text: str,
    numeric_report: dict,
    plan_summary: dict,
    coverage_report: dict | None,
    motion_start_grid: Path | None,
    motion_mid_grid: Path | None,
    motion_end_grid: Path | None,
    timeline_sample_grids: list[Path] | None = None,
    timeline_sample_catalog: list[dict] | None = None,
    scale_context: dict | None = None,
    trajectory_summary: dict | None = None,
    conditioning_mask_images: list[Path] | None = None,
    conditioning_mask_meta: dict | None = None,
    conditioning_text: str | None = None,
    iteration_index: int | None = None,
    max_iterations: int | None = None,
    previous_motion_reports: list[dict] | None = None,
    model: str = "gpt-5.2",
    base_url: str | None = None,
    api_provider: str = "auto",
    api_key: str | None = None,
) -> dict:
    if motion_end_grid is not None and (not motion_end_grid.exists()):
        motion_end_grid = None
    if motion_mid_grid is not None and (not motion_mid_grid.exists()):
        motion_mid_grid = None
    if motion_start_grid is not None and (not motion_start_grid.exists()):
        motion_start_grid = None

    prompt = _build_prompt(
        action_text,
        numeric_report,
        plan_summary,
        None,
        timeline_sample_catalog,
        scale_context,
        trajectory_summary,
        conditioning_mask_meta,
        conditioning_text,
        iteration_index,
        max_iterations,
        previous_motion_reports,
    )
    image_items: list[tuple[str, Path]] = []
    for i, p in enumerate(conditioning_mask_images or [], start=1):
        pp = Path(p)
        if not pp.exists():
            continue
        image_items.append((f"MASK_{i}:{pp.name}", pp))
    if motion_start_grid is not None:
        image_items.append(("MOTION_START_GRID", motion_start_grid))
    if motion_mid_grid is not None:
        image_items.append(("MOTION_MID_GRID", motion_mid_grid))
    if motion_end_grid is not None:
        image_items.append(("MOTION_END_GRID", motion_end_grid))
    if timeline_sample_grids:
        for i, p in enumerate(timeline_sample_grids, start=1):
            if p is None:
                continue
            pp = Path(p)
            if not pp.exists():
                continue
            if any(pp == q for _, q in image_items):
                continue
            image_items.append((f"TIMELINE_SAMPLE_{i}:{pp.name}", pp))
    if not image_items:
        raise FileNotFoundError("no motion diagnostic images available for VLM")
    msg, _cfg = generate_content_text(
        model=model,
        prompt=prompt,
        image_items=image_items,
        provider=api_provider,
        api_key=api_key,
        base_url=base_url,
    )
    known_links, known_joints = _collect_known_ids(plan_summary)
    return _sanitize_motion_report(
        _extract_json(msg),
        known_links=known_links,
        known_joints=known_joints,
        motion_mode="per_link_head_tail_pair",
    )


def diagnose_per_link_head_tail_via_api(
    action_text: str,
    numeric_report: dict,
    plan_summary: dict,
    timeline_sample_grids: list[Path] | None,
    timeline_sample_catalog: list[dict] | None,
    scale_context: dict | None,
    trajectory_summary: dict | None,
    conditioning_mask_images: list[Path] | None,
    conditioning_mask_meta: dict | None,
    conditioning_text: str | None,
    model: str,
    api_provider: str,
    api_key: str | None,
    base_url: str | None,
    per_case_report_dir: Path | None = None,
    per_case_prompt_dir: Path | None = None,
    per_case_trajectory_dir: Path | None = None,
) -> dict:
    catalog = [dict(row) for row in (timeline_sample_catalog or []) if isinstance(row, dict)]
    image_map = {Path(p).name: Path(p) for p in (timeline_sample_grids or []) if p is not None and Path(p).exists()}
    known_links, known_joints = _collect_known_ids(plan_summary)
    if per_case_report_dir is not None:
        Path(per_case_report_dir).mkdir(parents=True, exist_ok=True)
    if per_case_prompt_dir is not None:
        Path(per_case_prompt_dir).mkdir(parents=True, exist_ok=True)
    if per_case_trajectory_dir is not None:
        Path(per_case_trajectory_dir).mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[int, str, str], list[dict]] = {}
    for row in catalog:
        link_name = str(row.get("link") or "").strip()
        if not link_name:
            continue
        try:
            seg_idx = int(row.get("segment_index"))
        except Exception:
            seg_idx = -1
        joint_name = str(row.get("joint") or "").strip()
        grouped.setdefault((seg_idx, link_name, joint_name), []).append(row)

    cases = []
    for key, rows in grouped.items():
        rows_head = [r for r in rows if str(r.get("kind") or "").strip().lower() == "head"]
        rows_tail = [r for r in rows if str(r.get("kind") or "").strip().lower() == "tail"]
        ordered = []
        if rows_head:
            ordered.append(sorted(rows_head, key=lambda r: int(r.get("frame_idx", 0)))[0])
        if rows_tail:
            ordered.append(sorted(rows_tail, key=lambda r: int(r.get("frame_idx", 0)))[-1])
        if not ordered:
            continue
        ref_row = ordered[-1]
        if not str(ref_row.get("file_name") or "").strip():
            continue
        cases.append({"key": key, "rows": ordered, "ref_row": ref_row})
    if not cases:
        raise FileNotFoundError("no per-link head/tail motion cases available for diagnosis")

    def _run_one(case: dict) -> dict:
        rows = list(case.get("rows") or [])
        ref_row = dict(case.get("ref_row") or {})
        file_names = [str(r.get("file_name") or "").strip() for r in rows if str(r.get("file_name") or "").strip()]
        image_items: list[tuple[str, Path]] = []
        for i, p in enumerate(conditioning_mask_images or [], start=1):
            pp = Path(p)
            if pp.exists():
                image_items.append((f"MASK_{i}:{pp.name}", pp))
        case_rows = []
        for idx, row in enumerate(rows, start=1):
            fn = str(row.get("file_name") or "").strip()
            image_path = image_map.get(fn)
            if image_path is None:
                continue
            image_items.append((f"TIMELINE_SAMPLE_{idx}:{image_path.name}", image_path))
            case_rows.append(dict(row))
        if not case_rows:
            raise FileNotFoundError(f"missing images for case: {file_names}")
        filtered_plan_summary = _filter_plan_summary_for_tail(plan_summary, ref_row)
        filtered_trajectory_summary = _filter_trajectory_summary_for_tail(trajectory_summary, ref_row)
        filtered_scale_context = _filter_scale_context_for_tail(scale_context, ref_row)
        prompt = _build_prompt(
            action_text,
            numeric_report,
            filtered_plan_summary,
            None,
            case_rows,
            filtered_scale_context,
            filtered_trajectory_summary,
            conditioning_mask_meta,
            conditioning_text,
            None,
            None,
            None,
        )
        stem = Path(str(ref_row.get("file_name") or "case")).stem or "case"
        if per_case_prompt_dir is not None:
            (Path(per_case_prompt_dir) / f"{stem}.txt").write_text(prompt, encoding="utf-8")
            (Path(per_case_prompt_dir) / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "case_key": list(case.get("key") or []),
                        "file_names": file_names,
                        "catalog_rows": case_rows,
                        "plan_summary_filtered": filtered_plan_summary,
                        "trajectory_summary_filtered": filtered_trajectory_summary,
                        "scale_context_filtered": filtered_scale_context,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        if per_case_trajectory_dir is not None:
            (Path(per_case_trajectory_dir) / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "case_key": list(case.get("key") or []),
                        "file_names": file_names,
                        "catalog_rows": case_rows,
                        "trajectory_summary_filtered": filtered_trajectory_summary,
                        "scale_context_filtered": filtered_scale_context,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        msg, _cfg = generate_content_text(
            model=model,
            prompt=prompt,
            image_items=image_items,
            provider=api_provider,
            api_key=api_key,
            base_url=base_url,
        )
        report = _sanitize_motion_report(
            _extract_json(msg),
            known_links=known_links,
            known_joints=known_joints,
            motion_mode=str(filtered_plan_summary.get("motion_diagnosis_mode") or ""),
        )
        if per_case_report_dir is not None:
            payload = {
                "case_key": list(case.get("key") or []),
                "file_names": file_names,
                "catalog_rows": case_rows,
                "plan_summary_filtered": filtered_plan_summary,
                "trajectory_summary_filtered": filtered_trajectory_summary,
                "scale_context_filtered": filtered_scale_context,
                "report": report,
            }
            (Path(per_case_report_dir) / f"{stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return report

    reports: list[dict] = []
    manifest_rows: list[dict] = []
    max_workers = max(1, min(8, len(cases)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(_run_one, case): case for case in cases}
        for fut in concurrent.futures.as_completed(future_map):
            case = future_map[fut]
            report = fut.result()
            reports.append(report)
            seg_idx, link_name, joint_name = case.get("key") or (-1, "", "")
            manifest_rows.append(
                {
                    "segment_index": seg_idx,
                    "link": link_name,
                    "joint": joint_name,
                    "file_names": [str(r.get("file_name") or "") for r in (case.get("rows") or [])],
                }
            )
    if per_case_report_dir is not None:
        (Path(per_case_report_dir) / "per_case_manifest.json").write_text(
            json.dumps({"cases": manifest_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return _merge_motion_reports(reports)


def diagnose_motion(
    action_text: str,
    numeric_report: dict,
    plan_summary: dict,
    coverage_report: dict | None,
    motion_start_grid: Path | None,
    motion_mid_grid: Path | None,
    motion_end_grid: Path | None,
    timeline_sample_grids: list[Path] | None = None,
    timeline_sample_catalog: list[dict] | None = None,
    scale_context: dict | None = None,
    trajectory_summary: dict | None = None,
    conditioning_mask_images: list[Path] | None = None,
    conditioning_mask_meta: dict | None = None,
    conditioning_text: str | None = None,
    iteration_index: int | None = None,
    max_iterations: int | None = None,
    previous_motion_reports: list[dict] | None = None,
    model: str = "gpt-5.2",
    use_api: bool = True,
    api_provider: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
    per_case_report_dir: Path | None = None,
    per_case_prompt_dir: Path | None = None,
    per_case_trajectory_dir: Path | None = None,
) -> dict:
    motion_mode = "per_link_head_tail_pair"
    if use_api and (timeline_sample_grids or []):
        try:
            return diagnose_per_link_head_tail_via_api(
                action_text,
                numeric_report,
                plan_summary,
                [Path(p) for p in (timeline_sample_grids or [])],
                list(timeline_sample_catalog or []),
                scale_context,
                trajectory_summary,
                [Path(p) for p in (conditioning_mask_images or [])],
                conditioning_mask_meta,
                conditioning_text,
                model=model,
                api_provider=api_provider,
                api_key=api_key,
                base_url=base_url,
                per_case_report_dir=per_case_report_dir,
                per_case_prompt_dir=per_case_prompt_dir,
                per_case_trajectory_dir=per_case_trajectory_dir,
            )
        except Exception as exc:
            print(f"[WARN] Per-link head/tail motion VLM API failed ({exc}); returning neutral motion report.")
    if use_api and (motion_start_grid is not None or motion_mid_grid is not None or motion_end_grid is not None or (timeline_sample_grids or [])):
        try:
            return diagnose_via_api(
                action_text,
                numeric_report,
                plan_summary,
                coverage_report,
                Path(motion_start_grid) if motion_start_grid is not None else None,
                Path(motion_mid_grid) if motion_mid_grid is not None else None,
                Path(motion_end_grid) if motion_end_grid is not None else None,
                [Path(p) for p in (timeline_sample_grids or [])],
                list(timeline_sample_catalog or []),
                scale_context,
                trajectory_summary,
                [Path(p) for p in (conditioning_mask_images or [])],
                conditioning_mask_meta,
                conditioning_text,
                iteration_index,
                max_iterations,
                list(previous_motion_reports or []),
                model=model,
                api_provider=api_provider,
                api_key=api_key,
                base_url=base_url,
            )
        except Exception as exc:
            print(f"[WARN] Motion VLM API failed ({exc}); returning neutral motion report.")
    return neutral_motion_report("Motion VLM API was disabled or failed; no verifier fallback is used.")


def main():
    parser = argparse.ArgumentParser(description="Motion diagnosis JSON (VLM API + heuristic fallback)")
    parser.add_argument("--numeric_report", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--action_text", default="")
    parser.add_argument("--plan_summary", default=None)
    parser.add_argument("--coverage_report", default=None)
    parser.add_argument("--motion_start_grid", default=None)
    parser.add_argument("--motion_mid_grid", default=None)
    parser.add_argument("--motion_end_grid", default=None)
    parser.add_argument("--timeline_sample_images", nargs="*", default=None)
    parser.add_argument("--conditioning_mask_images", nargs="*", default=None)
    parser.add_argument("--conditioning_mask_meta", default=None)
    parser.add_argument("--conditioning_text", default=None)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_provider", default="auto", choices=["auto", "openai", "gemini"])
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--no_api", action="store_true")
    args = parser.parse_args()
    numeric = json.loads(Path(args.numeric_report).read_text(encoding="utf-8"))
    plan_summary = json.loads(Path(args.plan_summary).read_text(encoding="utf-8")) if args.plan_summary else {}
    coverage = json.loads(Path(args.coverage_report).read_text(encoding="utf-8")) if args.coverage_report else None
    out = diagnose_motion(
        args.action_text or "",
        numeric,
        plan_summary,
        coverage,
        Path(args.motion_start_grid) if args.motion_start_grid else None,
        Path(args.motion_mid_grid) if args.motion_mid_grid else None,
        Path(args.motion_end_grid) if args.motion_end_grid else None,
        [Path(p) for p in (args.timeline_sample_images or [])],
        None,
        None,
        [Path(p) for p in (args.conditioning_mask_images or [])],
        json.loads(Path(args.conditioning_mask_meta).read_text(encoding="utf-8")) if args.conditioning_mask_meta else None,
        args.conditioning_text,
        model=args.model,
        use_api=(not args.no_api),
        api_provider=args.api_provider,
        api_key=args.api_key,
        base_url=args.api_base_url,
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
