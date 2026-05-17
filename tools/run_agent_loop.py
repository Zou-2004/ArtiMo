#!/usr/bin/env python3
import argparse
from glob import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import apply_plan_patch as app_patch
import ask_plan as ask_plan_mod
import coverage_verify as cov_verify
import loop_render as lr
import plan_patcher as rule_patcher
import scale_context_utils as scu
import vlm_coverage_views as cov_vlm
import vlm_motion_diagnose as motion_vlm
import wheel_motion_diag_render as wheel_diag
from joint_verifier_utils import is_wheel_transport_plan

MOTION_VLM_MAX_IMAGES_TOTAL = 9
LOOP_MOTION_RENDER_BACKEND = str(os.environ.get("CODEX_LOOP_MOTION_RENDER_BACKEND", "blender")).strip().lower()


def _norm_motion_render_backend() -> str:
    rb = str(LOOP_MOTION_RENDER_BACKEND or "blender").strip().lower()
    if rb != "blender":
        return "blender"
    return "blender"


def _load_saved_reference_backend(asset_out: Path) -> str | None:
    try:
        decision = lr.gop.load_reference_backend_decision(asset_out)
    except Exception:
        decision = None
    if not isinstance(decision, dict):
        return None
    backend = lr.gop.normalize_reference_backend_name(decision.get("reference_backend"), default="auto")
    if backend != "blender":
        return None
    return backend


def _resolve_reference_backend(asset_out: Path, requested_backend: str) -> str:
    saved = _load_saved_reference_backend(asset_out)
    if saved == "blender":
        return saved
    return "blender"
def _build_child_to_joint_map(joints: list[dict]) -> dict[str, str]:
    # Prefer non-fixed joints when multiple joints mention the same child.
    best: dict[str, tuple[int, str]] = {}
    for j in joints or []:
        child = str(j.get("child") or "").strip()
        jn = str(j.get("name") or "").strip()
        if not child or not jn:
            continue
        jt = str(j.get("type") or "").strip().lower()
        pri = 1 if jt == "fixed" else 0
        prev = best.get(child)
        if prev is None or pri < prev[0]:
            best[child] = (pri, jn)
    return {k: v[1] for k, v in best.items()}


def _build_fixed_child_links_map(joints: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for j in joints or []:
        if str(j.get("type") or "").strip().lower() != "fixed":
            continue
        parent = str(j.get("parent") or "").strip()
        child = str(j.get("child") or "").strip()
        if not parent or not child:
            continue
        out.setdefault(parent, []).append(child)
    return out


def _build_parent_joint_by_child_map(joints: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for j in joints or []:
        child = str(j.get("child") or "").strip()
        if not child:
            continue
        jt = str(j.get("type") or "").strip().lower()
        pri = 1 if jt == "fixed" else 0
        prev = out.get(child)
        if prev is None or pri < int(prev.get("_pri", 99)):
            row = dict(j)
            row["_pri"] = pri
            out[child] = row
    return out


def _build_render_link_to_joint_map(joints: list[dict]) -> dict[str, str]:
    parent_joint_by_child = _build_parent_joint_by_child_map(joints)
    fixed_children = _build_fixed_child_links_map(joints)
    all_links = set(parent_joint_by_child.keys()) | set(fixed_children.keys())
    out: dict[str, str] = {}
    for link_name in sorted(all_links):
        cur = str(link_name)
        seen = set()
        chosen = None
        while cur and cur not in seen:
            seen.add(cur)
            j = parent_joint_by_child.get(cur)
            if not isinstance(j, dict):
                break
            jt = str(j.get("type") or "").strip().lower()
            if jt != "fixed":
                chosen = str(j.get("name") or "").strip()
                break
            cur = str(j.get("parent") or "").strip()
        if chosen:
            out[str(link_name)] = chosen
    return out


def _resolve_helper_like_link(ln: str, joints: list[dict]) -> list[str]:
    link_name = str(ln or "").strip()
    if not link_name:
        return []
    if ("_jf_" not in link_name) and (not link_name.endswith("_child_frame")):
        return [link_name]
    fixed_children = _build_fixed_child_links_map(joints)
    out = []
    stack = list(fixed_children.get(link_name, []))
    seen = set()
    while stack:
        cur = stack.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        kids = list(fixed_children.get(cur, []))
        if kids:
            stack.extend(kids)
        else:
            out.append(cur)
    return out if out else [link_name]


def _expand_motion_render_links(link_names: list[str], joints: list[dict]) -> list[str]:
    out = []
    seen = set()
    for ln in link_names or []:
        for resolved in _resolve_helper_like_link(str(ln), joints):
            if resolved and resolved not in seen:
                seen.add(resolved)
                out.append(resolved)
    return out


def _compact_motion_label(raw: str) -> str:
    s = str(raw or "").strip()
    if s.startswith("joint_"):
        tail = s[len("joint_") :].strip()
        return tail if tail else s
    if s.startswith("link_"):
        tail = s[len("link_") :].strip()
        return tail if tail else s
    return s


def _short_motion_alias(raw: str) -> str:
    s = _compact_motion_label(raw)
    nums = re.findall(r"\d+", s)
    if nums:
        return str(nums[-1])
    return s if s else "0"


def _safe_filename_token(raw: str) -> str:
    s = str(raw or "").strip().replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "na"


def _motion_head_cache_key(task: dict, resolution: tuple[int, int]) -> str:
    payload = {
        "sample_kind": str(task.get("sample_kind") or ""),
        "seg_idx": int(task.get("seg_idx", -1)),
        "logical_link": str(task.get("logical_link") or ""),
        "joint_name": str(task.get("joint_name") or ""),
        "render_links": [str(x) for x in (task.get("render_links") or [])],
        "frame_idx": int(((task.get("ts") or {}).get("frame_idx")) or 0),
        "selected_view": dict(task.get("selected_view") or {}),
        "render_label_legend": dict(task.get("render_label_legend") or {}),
        "caption": str(task.get("caption") or ""),
        "resolution": [int(resolution[0]), int(resolution[1])],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _build_motion_label_legend(
    required_links: list[str],
    visual_link_order: list[str],
    joints: list[dict],
) -> dict[str, str]:
    style = str(os.environ.get("CODEX_MOTION_LABEL_STYLE", "joint")).strip().lower()
    global_numeric = {ln: str(i + 1) for i, ln in enumerate(visual_link_order)}
    if style in {"id", "l", "legacy"}:
        return {ln: global_numeric.get(ln, "0") for ln in required_links}
    if style in {"link", "link_name"}:
        out = {}
        for ln in required_links:
            alias = _short_motion_alias(ln)
            out[ln] = alias if alias.isdigit() else global_numeric.get(ln, "0")
        return out
    # Default: show joint IDs for movable links; fallback to canonical link IDs.
    render_link_to_joint = _build_render_link_to_joint_map(joints)
    out = {}
    for ln in required_links:
        alias = _short_motion_alias(render_link_to_joint.get(ln, ln))
        out[ln] = alias if alias.isdigit() else global_numeric.get(ln, "0")
    return out


def find_urdf(asset_root: Path) -> Path | None:
    return next(asset_root.rglob("*.urdf"), None)


def find_glb_scene(asset_root: Path) -> Path | None:
    # Canonical textured mesh name only.
    canonical = asset_root / f"animated_textured_{asset_root.name}.glb"
    return canonical if canonical.exists() else None


def run(cmd, cwd=None):
    print("[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _extract_balanced_json_object(text: str) -> str | None:
    in_string = False
    escape = False
    depth = 0
    start = None
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                cand = text[start : idx + 1].strip()
                try:
                    json.loads(cand)
                    return cand
                except Exception:
                    start = None
            continue
    return None


def _load_json(path: Path):
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except Exception:
        pass
    s = raw.strip()
    for marker in ["```json", "```JSON", "```"]:
        search_from = 0
        while True:
            i = s.find(marker, search_from)
            if i < 0:
                break
            body_start = s.find("\n", i)
            if body_start < 0:
                break
            body_start += 1
            j = s.find("```", body_start)
            if j < 0:
                break
            body = s[body_start:j].strip()
            try:
                return json.loads(body)
            except Exception:
                cand = _extract_balanced_json_object(body)
                if cand is not None:
                    return json.loads(cand)
            search_from = j + 3
    cand = _extract_balanced_json_object(s)
    if cand is not None:
        return json.loads(cand)
    raise json.JSONDecodeError(f"Could not extract JSON from {path}", raw, 0)


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_saved_loop_viewspecs(loop_root: Path) -> tuple[dict | None, Path | None]:
    candidates = [loop_root / "motion_viewspecs_selected.json"]
    coverage_root = loop_root / "coverage"
    if coverage_root.exists():
        for path in sorted(coverage_root.glob("iter*/coverage_vlm_selected_viewspecs.json"), reverse=True):
            candidates.append(path)
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = _load_json(path)
            spec = lr.validate_viewspecs(raw)
            bundle = {"look_at_mode": spec.get("look_at_mode", "object_center"), "views": list(spec.get("views") or [])}
            per_link_views = {}
            raw_per_link = raw.get("per_link_views") if isinstance(raw, dict) else None
            if isinstance(raw_per_link, dict):
                for lk, row in raw_per_link.items():
                    try:
                        single = lr.validate_viewspecs(
                            {
                                "look_at_mode": bundle["look_at_mode"],
                                "views": [
                                    {
                                        "id": "V1",
                                        "azimuth_deg": int(row["azimuth_deg"]),
                                        "elevation_deg": int(row["elevation_deg"]),
                                        "distance_scale": float(row["distance_scale"]),
                                        "fov_deg": int(row["fov_deg"]),
                                    }
                                ],
                            }
                        )
                    except Exception:
                        continue
                    per_link_views[str(lk)] = dict((single.get("views") or [{}])[0])
            if per_link_views:
                bundle["per_link_views"] = per_link_views
            spec = bundle
        except Exception:
            continue
        return spec, path
    return None, None


def _canonical_view_row(view: dict) -> dict:
    return {
        "azimuth_deg": int(view["azimuth_deg"]),
        "elevation_deg": int(view["elevation_deg"]),
        "distance_scale": float(view["distance_scale"]),
        "fov_deg": int(view["fov_deg"]),
    }


def _make_single_view_spec(view: dict, look_at_mode: str = "object_center") -> dict:
    row = _canonical_view_row(view)
    return {
        "look_at_mode": str(look_at_mode or "object_center"),
        "views": [
            {
                "id": "V1",
                "azimuth_deg": int(row["azimuth_deg"]),
                "elevation_deg": int(row["elevation_deg"]),
                "distance_scale": float(row["distance_scale"]),
                "fov_deg": int(row["fov_deg"]),
            }
        ],
    }


def _log_per_link_viewspecs(label: str, per_link_views: dict[str, dict] | None) -> None:
    if not isinstance(per_link_views, dict) or not per_link_views:
        return
    print(f"[INFO] {label}")
    for ln in sorted(per_link_views.keys()):
        row = per_link_views.get(ln) or {}
        try:
            az = int(row.get("azimuth_deg"))
            el = int(row.get("elevation_deg"))
            ds = float(row.get("distance_scale"))
            fov = int(row.get("fov_deg"))
        except Exception:
            continue
        print(f"  - {ln}: AZ={az} EL={el} D={ds:.3f} FOV={fov}")


def _viewspecs_from_per_link_views(look_at_mode: str, per_link_views: dict[str, dict] | None) -> dict | None:
    if not isinstance(per_link_views, dict) or not per_link_views:
        return None
    union = []
    seen = set()
    for row in per_link_views.values():
        if not isinstance(row, dict):
            continue
        try:
            key = (
                int(row["azimuth_deg"]),
                int(row["elevation_deg"]),
                float(row["distance_scale"]),
                int(row["fov_deg"]),
            )
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        union.append(
            {
                "id": f"V{len(union)+1}",
                "azimuth_deg": int(row["azimuth_deg"]),
                "elevation_deg": int(row["elevation_deg"]),
                "distance_scale": float(row["distance_scale"]),
                "fov_deg": int(row["fov_deg"]),
            }
        )
    if not union:
        return None
    return {"look_at_mode": str(look_at_mode or "object_center"), "views": union}


def _build_coverage_motion_hints(causal_obj: dict, plan_obj: dict, joints: list[dict], required_links: list[str]) -> dict:
    joint_by_name = {str(j.get("name") or ""): j for j in (joints or []) if str(j.get("name") or "").strip()}
    joint_to_child = {str(j.get("name") or ""): str(j.get("child") or "") for j in (joints or []) if str(j.get("name") or "").strip()}
    render_link_to_joint = _build_render_link_to_joint_map(joints)
    semantics_rows = ((causal_obj.get("semantics") or {}).get("links") or [])
    semantic_by_link = {}
    for row in semantics_rows:
        if not isinstance(row, dict):
            continue
        ln = str(row.get("name") or "").strip()
        if not ln:
            continue
        semantic_by_link[ln] = {
            "label": str(row.get("label") or ""),
            "description": str(row.get("description") or ""),
        }

    def _append_unique_str(dst: list[str], value) -> None:
        s = str(value or "").strip()
        if s and s not in dst:
            dst.append(s)

    def _action_target_links(action_obj) -> list[str]:
        action = action_obj if isinstance(action_obj, dict) else {}
        out: list[str] = []
        _append_unique_str(out, action.get("target_link"))
        for ln in action.get("target_links") or []:
            _append_unique_str(out, ln)
        return out

    def _effects_joint_names(effects_obj) -> list[str]:
        effects = effects_obj if isinstance(effects_obj, dict) else {}
        out: list[str] = []
        for jt in effects.get("joint_targets") or []:
            if not isinstance(jt, dict):
                continue
            _append_unique_str(out, jt.get("joint"))
        return out

    def _effects_links(effects_obj) -> list[str]:
        effects = effects_obj if isinstance(effects_obj, dict) else {}
        out: list[str] = []
        for ln in effects.get("effect_links") or []:
            _append_unique_str(out, ln)
        for jn in _effects_joint_names(effects):
            _append_unique_str(out, joint_to_child.get(jn))
        return out

    target_links: set[str] = set()
    effect_links: set[str] = set()
    effect_joints_set: set[str] = set()
    temporal_segments: list[dict] = []

    causal_header = causal_obj.get("causal") or {}
    for ln in _action_target_links(causal_header.get("action")):
        target_links.add(ln)
    for ln in _effects_links(causal_header.get("effects")):
        effect_links.add(ln)
    for jn in _effects_joint_names(causal_header.get("effects")):
        effect_joints_set.add(jn)

    for seg_idx, seg in enumerate(causal_obj.get("causal_segments") or []):
        if not isinstance(seg, dict):
            continue
        seg_targets = _action_target_links(seg.get("action"))
        seg_effect_links = _effects_links(seg.get("effects"))
        seg_effect_joints = _effects_joint_names(seg.get("effects"))
        for ln in seg_targets:
            target_links.add(ln)
        for ln in seg_effect_links:
            effect_links.add(ln)
        for jn in seg_effect_joints:
            effect_joints_set.add(jn)
        temporal_segments.append(
            {
                "segment_id": str(seg.get("segment_id") or f"S{seg_idx+1}"),
                "order_index": int(((seg.get("time_hint") or {}).get("order_index")) or seg_idx),
                "action_primitive": str(((seg.get("action") or {}).get("primitive")) or ""),
                "target_links": seg_targets,
                "effect_links": seg_effect_links,
                "effect_joints": seg_effect_joints,
            }
        )

    link_to_plan_segments: dict[str, list[dict]] = {}
    for seg_idx, seg in enumerate((plan_obj or {}).get("timeline") or []):
        seg_name = str(seg.get("name") or f"seg_{seg_idx}")
        phase_type = str(seg.get("phase_type") or "")
        for ctrl in seg.get("controls") or []:
            mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()
            joints_in_ctrl = []
            if ctrl.get("joint"):
                joints_in_ctrl = [str(ctrl.get("joint"))]
            elif isinstance(ctrl.get("joints"), list):
                joints_in_ctrl = [str(x) for x in ctrl.get("joints") if str(x).strip()]
            for jn in joints_in_ctrl:
                ln = joint_to_child.get(jn)
                if not ln:
                    continue
                link_to_plan_segments.setdefault(ln, []).append(
                    {
                        "segment_index": seg_idx,
                        "segment_name": seg_name,
                        "phase_type": phase_type,
                        "mode": mode,
                    }
                )

    def _is_interior_candidate(ln: str) -> bool:
        sem = semantic_by_link.get(ln) or {}
        text = f"{sem.get('label', '')} {sem.get('description', '')}".lower()
        return any(tok in text for tok in {"inside", "internal", "interior", "inner", "cavity", "compartment"})

    hints = []
    for ln in required_links or []:
        ln_s = str(ln).strip()
        if not ln_s:
            continue
        role = "other"
        if ln_s in target_links:
            role = "target"
        related = []
        for j in joints or []:
            if str(j.get("child") or "").strip() == ln_s:
                related.append(j)
        if role == "other" and ln_s in effect_links:
            role = "effect"
        if role == "other":
            for j in related:
                if str(j.get("name") or "").strip() in effect_joints_set:
                    role = "effect"
                    break
        interior_candidate = _is_interior_candidate(ln_s)
        exposure_links = [
            other_s
            for other_s in [str(x).strip() for x in (required_links or [])]
            if other_s and other_s != ln_s and other_s in (target_links | effect_links)
        ]
        if not related:
            hints.append(
                {
                    "link": ln_s,
                    "role": role,
                    "joint": None,
                    "joint_type": "unknown",
                    "expected_motion": "unknown",
                    "semantic_label": (semantic_by_link.get(ln_s) or {}).get("label") or "",
                    "semantic_description": (semantic_by_link.get(ln_s) or {}).get("description") or "",
                    "interior_candidate": interior_candidate,
                    "exposure_links": exposure_links,
                    "plan_segments": link_to_plan_segments.get(ln_s, []),
                }
            )
            continue
        chosen = None
        for j in related:
            if str(j.get("name") or "").strip() in effect_joints_set:
                chosen = j
                break
        if chosen is None:
            for j in related:
                if str(j.get("type") or "").strip().lower() != "fixed":
                    chosen = j
                    break
        if chosen is None:
            chosen = related[0]
        render_joint_name = str(render_link_to_joint.get(ln_s) or "").strip()
        if render_joint_name and str(chosen.get("type") or "").strip().lower() == "fixed":
            chosen = joint_by_name.get(render_joint_name, chosen)
        jt = str(chosen.get("type") or "").strip().lower()
        if jt in {"revolute", "continuous"}:
            em = "rotation"
        elif jt == "prismatic":
            em = "translation"
        elif jt == "fixed":
            em = "static"
        else:
            em = "unknown"
        hints.append(
            {
                "link": ln_s,
                "role": role,
                "joint": str(chosen.get("name") or ""),
                "joint_type": jt if jt else "unknown",
                "expected_motion": em,
                "semantic_label": (semantic_by_link.get(ln_s) or {}).get("label") or "",
                "semantic_description": (semantic_by_link.get(ln_s) or {}).get("description") or "",
                "interior_candidate": interior_candidate,
                "exposure_links": exposure_links,
                "plan_segments": link_to_plan_segments.get(ln_s, []),
            }
        )
    return {
        "required_links_motion_hints": hints,
        "temporal_segments_summary": temporal_segments,
        "target_links": sorted(target_links),
        "effect_links": sorted(effect_links),
        "note": "Use causal semantics and planned temporal motion to prefer the most informative current view, especially when inside links must become visible later.",
    }


def _estimate_plan_base_displacement_m(plan_obj: dict) -> float:
    timeline = (plan_obj or {}).get("timeline") or []
    disp_vec = np.zeros((3,), dtype=float)
    max_norm = 0.0
    for seg in timeline:
        if not isinstance(seg, dict):
            continue
        try:
            dt = max(0.0, float(seg.get("t1", 0.0)) - float(seg.get("t0", 0.0)))
        except Exception:
            dt = 0.0
        if dt <= 0.0:
            continue
        for ctrl in seg.get("controls") or []:
            if not isinstance(ctrl, dict):
                continue
            mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()
            if mode not in {"base_velocity", "base_velocity_decay", "base", "base_decay"}:
                continue
            axis = np.asarray(ctrl.get("axis_world") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
            if axis.size < 3:
                axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
            axis = axis[:3]
            n = float(np.linalg.norm(axis))
            if n <= 1.0e-8:
                axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
            else:
                axis = axis / n
            seg_disp = 0.0
            if mode in {"base_velocity", "base"}:
                try:
                    seg_disp = abs(float(ctrl.get("v_mps") or 0.0)) * dt
                except Exception:
                    seg_disp = 0.0
            else:
                try:
                    v0 = abs(float(ctrl.get("v0_mps") or ctrl.get("v_mps") or 0.0))
                except Exception:
                    v0 = 0.0
                tau = None
                try:
                    tau = float(ctrl.get("tau_s")) if ctrl.get("tau_s") is not None else None
                except Exception:
                    tau = None
                if tau is not None and tau > 1.0e-8:
                    seg_disp = v0 * tau * (1.0 - math.exp(-dt / tau))
                else:
                    seg_disp = v0 * dt
            disp_vec = disp_vec + axis * float(seg_disp)
            max_norm = max(max_norm, float(np.linalg.norm(disp_vec)))
    return float(max_norm)


def _coverage_motion_anchor_radius(rest_radius: float, scale_context: dict | None, plan_obj: dict | None) -> tuple[float, dict]:
    rr = float(max(0.05, rest_radius))
    obj_diag = None
    if isinstance(scale_context, dict):
        try:
            obj_diag = float(scale_context.get("object_diag_m")) if scale_context.get("object_diag_m") is not None else None
        except Exception:
            obj_diag = None
    if obj_diag is None or obj_diag <= 1.0e-8:
        obj_diag = 2.0 * rr
    planned_disp = _estimate_plan_base_displacement_m(plan_obj or {})
    static_reserve = max(0.10 * float(obj_diag), 0.15 * rr)
    motion_reserve = float(planned_disp)
    anchor_radius = rr + static_reserve + motion_reserve
    info = {
        "rest_radius_m": rr,
        "object_diag_m": float(obj_diag),
        "planned_base_displacement_m": float(planned_disp),
        "static_reserve_m": float(static_reserve),
        "recommended_anchor_radius_m": float(anchor_radius),
    }
    return float(anchor_radius), info


def _sanitize_causal_action_targets(causal_obj: dict, joints: list[dict]) -> dict:
    if not isinstance(causal_obj, dict):
        return causal_obj
    known_links = set()
    for j in joints or []:
        if j.get("parent"):
            known_links.add(str(j.get("parent")))
        if j.get("child"):
            known_links.add(str(j.get("child")))
    out = json.loads(json.dumps(causal_obj))
    action = (out.get("causal") or {}).get("action") or {}
    if not isinstance(action, dict):
        return out
    tl = action.get("target_link")
    if isinstance(tl, str):
        t = tl.strip()
        if not t or t not in known_links:
            action.pop("target_link", None)
        else:
            action["target_link"] = t
    tls = action.get("target_links")
    cleaned_tls = []
    if isinstance(tls, list):
        for ln in tls:
            if not isinstance(ln, str):
                continue
            s = ln.strip()
            if s and s in known_links and s not in cleaned_tls:
                cleaned_tls.append(s)
    if cleaned_tls:
        action["target_links"] = cleaned_tls
    else:
        action.pop("target_links", None)
    if "target_link" not in action and cleaned_tls:
        action["target_link"] = cleaned_tls[0]
    if "target_link" in action and cleaned_tls and action["target_link"] not in cleaned_tls:
        action["target_links"] = [action["target_link"]] + cleaned_tls
    return out


def _select_timeline_images_for_vlm(
    timeline_index: list[dict], max_total_images: int = MOTION_VLM_MAX_IMAGES_TOTAL, base_image_count: int = 1
) -> list[Path]:
    budget = max(0, int(max_total_images) - int(base_image_count))
    if budget <= 0:
        return []
    rows = []
    for row in timeline_index or []:
        if not isinstance(row, dict):
            continue
        img = row.get("image")
        if not img:
            continue
        try:
            frame_idx = int(row.get("frame_idx", 0))
        except Exception:
            frame_idx = 0
        try:
            seg_idx = int(row.get("segment_index", -1))
        except Exception:
            seg_idx = -1
        kind = str(row.get("kind") or "mid")
        rows.append(
            {
                "segment_index": seg_idx,
                "kind": kind,
                "frame_idx": frame_idx,
                "image": str(img),
            }
        )
    if not rows:
        return []
    # Prioritize covering every timeline segment with at least one image (prefer "mid"),
    # then use remaining budget for extra samples such as "endish".
    rows = sorted(
        rows,
        key=lambda r: (
            r["segment_index"],
            0 if r["kind"] == "mid" else 1,
            r["frame_idx"],
            r["image"],
        ),
    )
    by_segment = {}
    for r in rows:
        by_segment.setdefault(r["segment_index"], []).append(r)

    seg_keys = sorted(by_segment.keys())
    primary = [by_segment[k][0] for k in seg_keys]
    secondary = []
    for k in seg_keys:
        secondary.extend(by_segment[k][1:])

    selected = []
    seen = set()

    def _append_row(r):
        p = r["image"]
        if p in seen:
            return False
        seen.add(p)
        selected.append(Path(p))
        return True

    if len(primary) <= budget:
        for r in primary:
            _append_row(r)
    else:
        # Too many segments for the budget: sample segment primaries uniformly as best effort.
        n = len(primary)
        for i in range(budget):
            idx = int(round(i * (n - 1) / max(1, budget - 1)))
            _append_row(primary[idx])

    remaining = budget - len(selected)
    if remaining > 0:
        # Fill extras with remaining samples, spread roughly across the timeline.
        secondary = sorted(secondary, key=lambda r: (r["frame_idx"], r["image"]))
        if len(secondary) <= remaining:
            for r in secondary:
                _append_row(r)
        else:
            n = len(secondary)
            for i in range(remaining):
                idx = int(round(i * (n - 1) / max(1, remaining - 1)))
                _append_row(secondary[idx])
            if len(selected) < budget:
                for r in secondary:
                    if len(selected) >= budget:
                        break
                    _append_row(r)
    return selected[:budget]


def _dedupe_and_merge_views(current_spec: dict, proposal: dict, max_total: int = 4) -> dict:
    cur = lr.validate_viewspecs(current_spec)
    views = list(cur.get("views", []))
    existing_keys = {
        (v["azimuth_deg"], v["elevation_deg"], float(v["distance_scale"]), v["fov_deg"]) for v in views
    }
    new_views = []
    for p in proposal.get("proposed_views") or []:
        cand = {
            "id": f"V{len(views) + len(new_views) + 1}",
            "azimuth_deg": int(p["azimuth_deg"]),
            "elevation_deg": int(p["elevation_deg"]),
            "distance_scale": float(p["distance_scale"]),
            "fov_deg": int(p["fov_deg"]),
        }
        key = (cand["azimuth_deg"], cand["elevation_deg"], cand["distance_scale"], cand["fov_deg"])
        if key in existing_keys:
            continue
        existing_keys.add(key)
        new_views.append(cand)
    merged = views + new_views
    # Keep the latest views if oversized to encourage exploration.
    if len(merged) > max_total:
        merged = merged[-max_total:]
    for i, v in enumerate(merged):
        v["id"] = f"V{i+1}"
    return {"look_at_mode": cur.get("look_at_mode", "object_center"), "views": merged}


def _select_views_from_proposal(current_spec: dict, proposal: dict, max_total: int = 1) -> dict | None:
    cur = lr.validate_viewspecs(current_spec)
    views = list(cur.get("views", []))
    if not views:
        return None
    selected_raw = proposal.get("selected_views") or []
    if not isinstance(selected_raw, list) or not selected_raw:
        return None
    selected = []
    seen = set()
    for p in selected_raw[: int(max_total)]:
        if not isinstance(p, dict):
            continue
        try:
            key = (
                int(p.get("azimuth_deg")),
                int(p.get("elevation_deg")),
                float(p.get("distance_scale")),
                int(p.get("fov_deg")),
            )
        except Exception:
            continue
        # Must be chosen from current rendered view set.
        matched = None
        for v in views:
            k = (
                int(v["azimuth_deg"]),
                int(v["elevation_deg"]),
                float(v["distance_scale"]),
                int(v["fov_deg"]),
            )
            if k == key:
                matched = v
                break
        if matched is None or key in seen:
            continue
        seen.add(key)
        selected.append(
            {
                "id": str(matched.get("id") or f"V{len(selected)+1}"),
                "azimuth_deg": int(matched["azimuth_deg"]),
                "elevation_deg": int(matched["elevation_deg"]),
                "distance_scale": float(matched["distance_scale"]),
                "fov_deg": int(matched["fov_deg"]),
            }
        )
    if not selected:
        return None
    return {"look_at_mode": cur.get("look_at_mode", "object_center"), "views": selected[: int(max_total)]}


def _expand_existing_paths(values: list[str] | None) -> list[Path]:
    out: list[Path] = []
    for v in values or []:
        s = str(v).strip()
        if not s:
            continue
        if any(ch in s for ch in ["*", "?", "["]):
            for hit in sorted(glob(s)):
                p = Path(hit).expanduser().resolve()
                if p.exists():
                    out.append(p)
            continue
        p = Path(s).expanduser().resolve()
        if p.exists():
            out.append(p)
    uniq = []
    seen = set()
    for p in out:
        sp = str(p)
        if sp in seen:
            continue
        seen.add(sp)
        uniq.append(p)
    return uniq


def _build_mask_panel(mask_paths: list[Path], out_path: Path) -> tuple[list[Path], dict]:
    def _read_sidecar_label(p: Path) -> str:
        sidecar = p.with_suffix(".txt")
        if not sidecar.exists():
            return ""
        try:
            txt = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""
        if not txt:
            return ""
        parts = [x for x in txt.replace("\t", " ").split(" ") if x]
        if len(parts) >= 2:
            return " ".join(parts[1:]).strip()
        return txt

    if not mask_paths:
        return [], {"count": 0, "mode": "none"}
    if len(mask_paths) == 1:
        return [mask_paths[0]], {
            "count": 1,
            "mode": "single",
            "paths": [str(mask_paths[0])],
            "entries": [
                {
                    "path": str(mask_paths[0]),
                    "name": mask_paths[0].name,
                    "sidecar_label": _read_sidecar_label(mask_paths[0]),
                }
            ],
        }

    imgs = []
    for p in mask_paths:
        try:
            imgs.append((p, Image.open(p).convert("RGB")))
        except Exception:
            continue
    if len(imgs) <= 1:
        valid = [x[0] for x in imgs] if imgs else []
        return valid, {
            "count": len(valid),
            "mode": "single_fallback",
            "paths": [str(p) for p in valid],
            "entries": [
                {"path": str(p), "name": p.name, "sidecar_label": _read_sidecar_label(p)} for p in valid
            ],
        }

    cols = min(2, len(imgs))
    rows = int(math.ceil(len(imgs) / cols))
    cell_w = max(im.width for _, im in imgs)
    cell_h = max(im.height for _, im in imgs)
    pad = 12
    header = 36
    panel_w = cols * cell_w + (cols + 1) * pad
    panel_h = rows * (cell_h + header) + (rows + 1) * pad
    panel = Image.new("RGB", (panel_w, panel_h), (245, 245, 245))
    draw = ImageDraw.Draw(panel)
    for i, (p, im) in enumerate(imgs):
        r = i // cols
        c = i % cols
        x0 = pad + c * (cell_w + pad)
        y0 = pad + r * (cell_h + header + pad)
        label = f"MASK_{i+1}: {p.name}"
        draw.rectangle([x0, y0, x0 + cell_w, y0 + header + cell_h], outline=(80, 80, 80), width=2)
        draw.text((x0 + 6, y0 + 8), label, fill=(10, 10, 10))
        panel.paste(im.resize((cell_w, cell_h)), (x0, y0 + header))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path)
    return [out_path], {
        "count": len(imgs),
        "mode": "stitched_panel",
        "paths": [str(p) for p, _ in imgs],
        "panel_path": str(out_path),
        "entries": [
            {"path": str(p), "name": p.name, "sidecar_label": _read_sidecar_label(p)} for p, _ in imgs
        ],
    }


def _prepare_pipeline_if_needed(args, py: str, asset_out: Path):
    need_prepare = not (args.skip_preprocess and args.skip_vlm and args.skip_llm)
    if not need_prepare:
        return
    cmd = [
        py,
        "tools/run_agent_single.py",
        "--asset_root",
        str(args.asset_root),
        "--action_text",
        args.action_text,
        "--out_root",
        str(args.out_root),
        "--vlm_model",
        args.vlm_model,
        "--llm_model",
        args.llm_model,
        "--api_provider",
        args.api_provider,
        "--skip_plan_exec",
        "--skip_plan_frames",
    ]
    if args.use_glb_scene is not None:
        cmd.extend(["--use_glb_scene", args.use_glb_scene])
    if args.skip_preprocess:
        cmd.append("--skip_preprocess")
    if args.skip_vlm:
        cmd.append("--skip_vlm")
    if args.skip_llm:
        cmd.append("--skip_llm")
    if args.api_key:
        cmd.extend(["--api_key", args.api_key])
    if args.api_base_url:
        cmd.extend(["--api_base_url", args.api_base_url])
    if getattr(args, "vlm_conditioning_images", None):
        cmd.extend(["--vlm_conditioning_images"] + list(args.vlm_conditioning_images))
    if getattr(args, "vlm_conditioning_masks", None):
        cmd.extend(["--vlm_conditioning_masks"] + list(args.vlm_conditioning_masks))
    if getattr(args, "vlm_conditioning_text", None) is not None:
        cmd.extend(["--vlm_conditioning_text", args.vlm_conditioning_text])
    if getattr(args, "vlm_no_auto_images", False):
        cmd.append("--vlm_no_auto_images")
    run(cmd)
    if not (asset_out / "plan.json").exists():
        raise SystemExit(f"plan.json missing after pipeline prep: {asset_out / 'plan.json'}")


def _build_plan_summary(plan: dict, causal: dict, joints: list[dict] | None = None) -> dict:
    joint_limits = _joint_limits_by_name(joints)
    return {
        "action": (causal.get("causal") or {}).get("action") or {},
        "effects": (causal.get("causal") or {}).get("effects") or {},
        "meta": plan.get("meta") or {},
        "joint_limits": joint_limits,
        "timeline": [
            {
                "name": seg.get("name"),
                "phase_type": seg.get("phase_type"),
                "t0": seg.get("t0"),
                "t1": seg.get("t1"),
                "controls": [
                    {
                        k: v
                        for k, v in ctrl.items()
                        if k in {
                            "type",
                            "mode",
                            "joint",
                            "v_mps",
                            "v0_mps",
                            "omega_radps",
                            "q_start_rad",
                            "q_target_expr",
                            "q_target_rad",
                            "tau_s",
                            "spring_k",
                            "damping_c",
                            "rest_position",
                        }
                    }
                    for ctrl in (seg.get("controls") or [])
                ],
            }
            for seg in (plan.get("timeline") or [])
        ],
    }


def _joint_limits_by_name(joints: list[dict] | None = None) -> dict:
    joint_limits = {}
    for joint in joints or []:
        if not isinstance(joint, dict):
            continue
        jn = str(joint.get("name") or "").strip()
        if not jn:
            continue
        lim = joint.get("limit") if isinstance(joint.get("limit"), dict) else None
        if not isinstance(lim, dict):
            continue
        joint_limits[jn] = {
            "lower": lim.get("lower"),
            "upper": lim.get("upper"),
            "effort": lim.get("effort"),
            "velocity": lim.get("velocity"),
        }
    return joint_limits


def _timeline_sample_frames(plan: dict, traj_len: int, fps: int) -> list[dict]:
    samples = []
    if traj_len <= 0:
        return samples
    for si, seg in enumerate(plan.get("timeline") or []):
        try:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", t0))
        except Exception:
            continue
        if t1 <= t0:
            continue
        seg_name = seg.get("name") or f"seg_{si}"
        seg_phase_type = seg.get("phase_type")
        # Per user request: keep each segment's "head" and "tail" frames (instead of mid/end debug frames).
        # Use an inclusive head frame and the last frame still inside the segment for tail.
        f_head = max(0, min(traj_len - 1, int(round(t0 * fps))))
        f_tail = max(0, min(traj_len - 1, int(round(t1 * fps)) - 1))
        if f_tail < f_head:
            f_tail = f_head
        t_head = float(f_head) / float(max(fps, 1))
        t_tail = float(f_tail) / float(max(fps, 1))
        samples.append(
            {
                "segment_index": si,
                "segment_name": seg_name,
                "phase_type": seg_phase_type,
                "kind": "head",
                "segment_t0": t0,
                "segment_t1": t1,
                "t_s": t_head,
                "frame_idx": f_head,
            }
        )
        if f_tail != f_head:
            samples.append(
                {
                    "segment_index": si,
                    "segment_name": seg_name,
                    "phase_type": seg_phase_type,
                    "kind": "tail",
                    "segment_t0": t0,
                    "segment_t1": t1,
                    "t_s": t_tail,
                    "frame_idx": f_tail,
                }
            )
    return samples


def _causal_segment_count(causal_obj: dict) -> int:
    segs = causal_obj.get("causal_segments")
    if isinstance(segs, list) and segs:
        return len(segs)
    if ((causal_obj.get("causal") or {}).get("has_causal")) is False:
        return 0
    return 1


def _segment_has_renderworthy_motion(seg: dict) -> bool:
    for ctrl in (seg.get("controls") or []):
        mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()
        if mode in {
            "joint_position",
            "joint_velocity",
            "base_velocity",
            "base_velocity_decay",
            "spring_return",
            "release_joint",
            "wheel_rolling_decay",
        }:
            return True
    return False


def _dedupe_frame_rows(rows: list[dict], protected_frame_indices: set[int] | None = None) -> list[dict]:
    protected = set(int(x) for x in (protected_frame_indices or set()))
    seen = set()
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            fi = int(row.get("frame_idx", 0))
        except Exception:
            continue
        if fi in protected or fi in seen:
            continue
        seen.add(fi)
        out.append(dict(row))
    return out


def _timeline_head_tail_frames(plan: dict, traj_len: int, fps: int) -> list[dict]:
    if traj_len <= 0:
        return []
    rows = []
    for si, seg in enumerate(plan.get("timeline") or []):
        if not _segment_has_renderworthy_motion(seg):
            continue
        try:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", t0))
        except Exception:
            continue
        if t1 <= t0:
            continue
        f_head = max(0, min(traj_len - 1, int(round(t0 * fps))))
        f_tail = max(f_head, min(traj_len - 1, int(round(t1 * fps)) - 1))
        seg_name = str(seg.get("name") or f"seg_{si}")
        phase_type = str(seg.get("phase_type") or "")
        rows.append(
            {
                "segment_index": si,
                "segment_name": seg_name,
                "phase_type": phase_type,
                "kind": "head",
                "segment_t0": t0,
                "segment_t1": t1,
                "t_s": float(f_head) / float(max(fps, 1)),
                "frame_idx": f_head,
            }
        )
        if f_tail != f_head:
            rows.append(
                {
                    "segment_index": si,
                    "segment_name": seg_name,
                    "phase_type": phase_type,
                    "kind": "tail",
                    "segment_t0": t0,
                    "segment_t1": t1,
                    "t_s": float(f_tail) / float(max(fps, 1)),
                    "frame_idx": f_tail,
                }
            )
    return _dedupe_frame_rows(rows)


def _sample_has_motion(cues: dict | None) -> bool:
    if not isinstance(cues, dict):
        return False
    for jt in (cues.get("joint_trends") or []):
        if not isinstance(jt, dict):
            continue
        if str(jt.get("trend") or "").strip().lower() != "static":
            return True
    bm = cues.get("base_motion")
    if isinstance(bm, dict):
        if str(bm.get("trend") or "").strip().lower() in {"positive", "negative"}:
            return True
    return False


def _signed_base_axis_world_from_control(ctrl: dict) -> list[float] | None:
    if not isinstance(ctrl, dict):
        return None
    mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()
    if mode not in {"base_velocity", "base_velocity_decay", "base", "base_decay"}:
        return None
    axis = ctrl.get("axis_world")
    if not (isinstance(axis, list) and len(axis) == 3):
        return None
    try:
        vals = np.asarray([float(axis[0]), float(axis[1]), float(axis[2])], dtype=float)
    except Exception:
        return None
    mag_key = "v_mps" if mode in {"base_velocity", "base"} else "v0_mps"
    try:
        mag = float(ctrl.get(mag_key, ctrl.get("v_mps", ctrl.get("linear_velocity_mps", 0.0))))
    except Exception:
        mag = 0.0
    if mag < 0.0:
        vals = -vals
    n = float(np.linalg.norm(vals))
    if n <= 1e-9:
        return None
    vals = vals / n
    return [float(vals[0]), float(vals[1]), float(vals[2])]


def _plan_segment_base_axis_world(plan_obj: dict, segment_index: int) -> list[float] | None:
    try:
        seg = (plan_obj.get("timeline") or [])[int(segment_index)]
    except Exception:
        return None
    for ctrl in (seg.get("controls") or []):
        signed_axis = _signed_base_axis_world_from_control(ctrl)
        if signed_axis is not None:
            return signed_axis
    return None


def _plan_overall_base_axis_world(plan_obj: dict) -> list[float] | None:
    for seg in (plan_obj.get("timeline") or []):
        if not isinstance(seg, dict):
            continue
        for ctrl in (seg.get("controls") or []):
            signed_axis = _signed_base_axis_world_from_control(ctrl)
            if signed_axis is not None:
                return signed_axis
    return None


def _segment_frame_window(sample_row: dict, traj_len: int, fps: int) -> tuple[int, int]:
    if int(traj_len) <= 0:
        return 0, 0
    try:
        t0 = float(sample_row.get("segment_t0", 0.0))
    except Exception:
        t0 = 0.0
    try:
        t1 = float(sample_row.get("segment_t1", t0))
    except Exception:
        t1 = t0
    i0 = int(round(t0 * float(max(1, fps))))
    i1 = int(round(t1 * float(max(1, fps)))) - 1
    i0 = max(0, min(int(traj_len) - 1, i0))
    i1 = max(0, min(int(traj_len) - 1, i1))
    if i1 < i0:
        i1 = i0
    return i0, i1


def _motion_case_key(segment_index: int | None, link_name: str | None, joint_name: str | None = None) -> tuple[int, str, str]:
    try:
        seg_idx = int(segment_index)
    except Exception:
        seg_idx = -1
    return (seg_idx, str(link_name or "").strip(), str(joint_name or "").strip())


def _motion_case_key_from_row(row: dict | None) -> tuple[int, str, str] | None:
    if not isinstance(row, dict):
        return None
    link_name = str(row.get("link") or "").strip()
    if not link_name:
        return None
    return _motion_case_key(row.get("segment_index"), link_name, row.get("joint"))


def _motion_case_problematic(report: dict | None) -> bool:
    if not isinstance(report, dict):
        return True
    if not bool(report.get("semantic_ok", False)):
        return True
    if not bool(report.get("visibility_ok", True)):
        return True
    if bool(report.get("issues") or []):
        return True
    if bool(report.get("param_fix_hints") or []):
        return True
    return False


def _collect_motion_case_keys_from_catalog(rows: list[dict] | None) -> set[tuple[int, str, str]]:
    out: set[tuple[int, str, str]] = set()
    for row in rows or []:
        key = _motion_case_key_from_row(row)
        if key is not None:
            out.add(key)
    return out


def _collect_problematic_motion_case_keys(
    vlm_report_dir: Path,
    timeline_catalog: list[dict] | None,
    merged_report: dict | None,
) -> set[tuple[int, str, str]]:
    all_case_keys = _collect_motion_case_keys_from_catalog(timeline_catalog)
    problematic: set[tuple[int, str, str]] = set()
    if isinstance(vlm_report_dir, Path) and vlm_report_dir.exists():
        skip_names = {
            "motion_vlm_report.json",
            "motion_vlm_report_raw.json",
            "per_case_manifest.json",
            "per_tail_manifest.json",
        }
        for path in sorted(vlm_report_dir.glob("*.json")):
            if path.name in skip_names:
                continue
            try:
                payload = _load_json(path)
            except Exception:
                continue
            report = payload.get("report") if isinstance(payload, dict) else None
            if not _motion_case_problematic(report):
                continue
            key = None
            if isinstance(payload, dict) and isinstance(payload.get("case_key"), list) and len(payload.get("case_key") or []) >= 3:
                ck = payload.get("case_key") or []
                key = _motion_case_key(ck[0], ck[1], ck[2])
            else:
                candidate_rows = []
                if isinstance(payload, dict):
                    if isinstance(payload.get("catalog_rows"), list):
                        candidate_rows = payload.get("catalog_rows") or []
                    elif isinstance(payload.get("catalog_row"), dict):
                        candidate_rows = [payload.get("catalog_row")]
                for row in candidate_rows:
                    key = _motion_case_key_from_row(row)
                    if key is not None:
                        break
            if key is not None:
                problematic.add(key)
    if problematic:
        return problematic
    if _motion_case_problematic(merged_report):
        return set(all_case_keys)
    return set()


def _segment_force_rotation_links(
    joints: list[dict],
    traj_npz_data,
    frame_window: tuple[int, int],
    candidate_links: list[str] | None = None,
) -> set[str]:
    return set(
        _segment_rotation_direction_map(
            joints,
            traj_npz_data,
            frame_window,
            candidate_links=candidate_links,
        ).keys()
    )


def _segment_rotation_direction_map(
    joints: list[dict],
    traj_npz_data,
    frame_window: tuple[int, int],
    candidate_links: list[str] | None = None,
) -> dict[str, str]:
    try:
        joint_names = [str(x) for x in traj_npz_data["joint_names"].tolist()]
        joint_angles = np.asarray(traj_npz_data["joint_angles"], dtype=float)
    except Exception:
        return {}
    if joint_angles.ndim != 2 or joint_angles.shape[0] <= 0:
        return {}
    n_frames = int(joint_angles.shape[0])
    i0 = max(0, min(n_frames - 1, int(frame_window[0])))
    i1 = max(0, min(n_frames - 1, int(frame_window[1])))
    if i1 < i0:
        i1 = i0
    child_to_joint = _build_render_link_to_joint_map(joints)
    rotary_links = lr._rotary_child_links_from_joints(joints)
    if isinstance(candidate_links, list) and candidate_links:
        rotary_links = rotary_links.intersection({str(x).strip() for x in candidate_links if str(x).strip()})
    joint_idx = {jn: i for i, jn in enumerate(joint_names)}
    delta_th = float(os.environ.get("CODEX_MOTION_FORCE_ROT_DELTA_RAD", "0.001"))
    out: dict[str, str] = {}
    for ln in rotary_links:
        jn = child_to_joint.get(ln)
        ji = joint_idx.get(jn) if isinstance(jn, str) else None
        if ji is None:
            continue
        delta_signed = float(joint_angles[i1, ji]) - float(joint_angles[i0, ji])
        if abs(delta_signed) >= delta_th:
            # Use timeline-level signed joint delta as a view-invariant rotation direction.
            # Positive delta => ccw, negative delta => cw.
            out[str(ln)] = "ccw" if delta_signed >= 0.0 else "cw"
    return out


def _link_active_in_segment(plan_obj: dict, segment_index: int, link_name: str, joint_name: str | None = None) -> bool:
    try:
        seg = (plan_obj.get("timeline") or [])[int(segment_index)]
    except Exception:
        return False
    if not isinstance(seg, dict):
        return False
    joint_name = str(joint_name or "").strip()
    for ctrl in seg.get("controls") or []:
        if not isinstance(ctrl, dict):
            continue
        mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()
        if mode in {"base_velocity", "base_velocity_decay", "base", "base_decay"}:
            return True
        if joint_name and str(ctrl.get("joint") or "").strip() == joint_name:
            return True
        joints_field = ctrl.get("joints")
        if joint_name and isinstance(joints_field, list) and any(str(x).strip() == joint_name for x in joints_field):
            return True
    return False


def _collect_motion_case_links(plan_obj: dict, joints: list[dict]) -> list[str]:
    child_to_joint = _build_child_to_joint_map(joints)
    out = []
    seen = set()
    for seg in plan_obj.get("timeline") or []:
        if not isinstance(seg, dict):
            continue
        for ctrl in seg.get("controls") or []:
            if not isinstance(ctrl, dict):
                continue
            joint_name = str(ctrl.get("joint") or "").strip()
            if not joint_name:
                continue
            for ln, jn in child_to_joint.items():
                if str(jn) != joint_name:
                    continue
                for resolved in _resolve_helper_like_link(str(ln), joints):
                    if resolved not in seen:
                        seen.add(resolved)
                        out.append(resolved)
    return out


def _link_local_bbox_corners(asset_ctx: dict, link_names: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    link_meshes = asset_ctx.get("link_meshes") or {}
    for ln in link_names:
        meshes = link_meshes.get(ln, []) if isinstance(link_meshes, dict) else []
        valid = [m.copy() for m in meshes if m is not None and getattr(m, "vertices", None) is not None and m.vertices.size > 0]
        if not valid:
            out[str(ln)] = np.zeros((0, 3), dtype=np.float32)
            continue
        merged = lr.trimesh.util.concatenate(valid) if len(valid) > 1 else valid[0]
        try:
            out[str(ln)] = np.asarray(merged.bounding_box.vertices, dtype=np.float32)
        except Exception:
            out[str(ln)] = np.zeros((0, 3), dtype=np.float32)
    return out


def _select_peak_motion_frame(
    asset_ctx: dict,
    traj_npz_path: Path,
    plan_obj: dict,
    joints: list[dict],
    required_links: list[str],
) -> tuple[int | None, dict[str, np.ndarray] | None, dict]:
    traj = np.load(traj_npz_path, allow_pickle=True)
    joint_angles = np.asarray(traj.get("joint_angles", np.zeros((0, 0))), dtype=float)
    if joint_angles.ndim != 2 or joint_angles.shape[0] <= 0:
        return None, None, {"reason": "no_trajectory"}
    n_frames = int(joint_angles.shape[0])
    joint_names = [str(x) for x in traj["joint_names"].tolist()]
    base_translation = np.asarray(traj.get("base_translation", np.zeros((n_frames, 3))), dtype=float)
    time_s = np.asarray(traj.get("time_s", np.zeros((n_frames,))), dtype=float)
    motion_links = _collect_motion_case_links(plan_obj, joints)
    if not motion_links:
        motion_links = [str(x) for x in required_links if str(x).strip()]
    if not motion_links:
        return None, None, {"reason": "no_motion_links"}
    local_corners = _link_local_bbox_corners(asset_ctx, motion_links)
    rest_tf = lr.rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], {})
    rest_pts_world = {
        ln: lr._apply_tf_points(rest_tf.get(ln, np.eye(4)), pts)
        for ln, pts in local_corners.items()
        if np.asarray(pts).size > 0
    }
    best_idx = None
    best_score = -1.0
    best_link_tf = None
    step = max(1, int(round(n_frames / 80.0)))
    sample_ids = list(range(0, n_frames, step))
    if sample_ids[-1] != n_frames - 1:
        sample_ids.append(n_frames - 1)
    for fi in sample_ids:
        joint_pos = {jn: float(joint_angles[fi, j]) for j, jn in enumerate(joint_names)}
        base_tf = np.eye(4)
        if base_translation.ndim == 2 and fi < base_translation.shape[0]:
            base_tf[:3, 3] = np.asarray(base_translation[fi], dtype=float)
        link_tf = lr.rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], joint_pos, base_tf=base_tf)
        score = 0.0
        for ln, rest_pts in rest_pts_world.items():
            cur_pts = lr._apply_tf_points(link_tf.get(ln, np.eye(4)), local_corners[ln])
            if cur_pts.shape == rest_pts.shape and cur_pts.size > 0:
                score = max(score, float(np.linalg.norm(cur_pts - rest_pts, axis=1).max()))
        if base_translation.ndim == 2 and fi < base_translation.shape[0]:
            score = max(score, float(np.linalg.norm(base_translation[fi] - base_translation[0])))
        if score > best_score:
            best_score = score
            best_idx = int(fi)
            best_link_tf = link_tf
    if best_idx is None or best_link_tf is None or best_score <= 1.0e-8:
        return None, None, {"reason": "no_significant_motion", "score_m": float(max(best_score, 0.0))}
    return int(best_idx), best_link_tf, {
        "frame_idx": int(best_idx),
        "time_s": float(time_s[best_idx]) if time_s.ndim == 1 and best_idx < time_s.shape[0] else None,
        "score_m": float(best_score),
        "motion_links": motion_links,
    }


def _adjust_view_distance_scale_for_motion_extent(
    asset_ctx: dict,
    rest_center: np.ndarray,
    rest_radius: float,
    resolution: tuple[int, int],
    per_link_views: dict[str, dict],
    peak_link_tf: dict[str, np.ndarray] | None,
    required_links: list[str],
) -> dict[str, dict]:
    if not isinstance(per_link_views, dict) or not per_link_views:
        return {}
    visual_links = [ln for ln, meshes in (asset_ctx.get("link_meshes") or {}).items() if meshes]
    if not visual_links:
        return {str(k): dict(v) for k, v in per_link_views.items() if isinstance(v, dict)}
    local_corners = _link_local_bbox_corners(asset_ctx, visual_links)
    rest_tf = lr.rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], {})
    rest_points = {
        ln: lr._apply_tf_points(rest_tf.get(ln, np.eye(4)), pts)
        for ln, pts in local_corners.items()
        if np.asarray(pts).size > 0
    }
    peak_points = {
        ln: lr._apply_tf_points((peak_link_tf or {}).get(ln, rest_tf.get(ln, np.eye(4))), local_corners[ln])
        for ln in rest_points.keys()
    }
    margin_x = max(12.0, 0.035 * float(resolution[0]))
    margin_y = max(12.0, 0.04 * float(resolution[1]))

    def _union_box(points_map: dict[str, np.ndarray], cam) -> tuple[float, float, float, float] | None:
        boxes = lr.gop.project_link_boxes(points_map, cam, resolution)
        valid = [b for b in boxes.values() if isinstance(b, (tuple, list)) and len(b) == 4]
        if not valid:
            return None
        return (
            min(float(b[0]) for b in valid),
            min(float(b[1]) for b in valid),
            max(float(b[2]) for b in valid),
            max(float(b[3]) for b in valid),
        )

    def _fits(box) -> bool:
        if not (isinstance(box, (tuple, list)) and len(box) == 4):
            return False
        x0, y0, x1, y1 = [float(v) for v in box]
        return x0 >= margin_x and y0 >= margin_y and x1 <= float(resolution[0]) - margin_x and y1 <= float(resolution[1]) - margin_y

    def _score_scale(view_row: dict, ds: float) -> tuple[bool, float]:
        test_view = dict(view_row)
        test_view["distance_scale"] = float(ds)
        cam = lr.compute_camera_for_viewspec(np.asarray(rest_center, dtype=float), float(rest_radius), test_view)
        rest_box = _union_box(rest_points, cam)
        peak_box = _union_box(peak_points, cam)
        fit_ok = _fits(rest_box) and _fits(peak_box)
        boxes = [b for b in [rest_box, peak_box] if isinstance(b, (tuple, list)) and len(b) == 4]
        if not boxes:
            return False, 1.0e9
        penalties = []
        for box in boxes:
            x0, y0, x1, y1 = [float(v) for v in box]
            penalties.append(max(0.0, margin_x - x0))
            penalties.append(max(0.0, margin_y - y0))
            penalties.append(max(0.0, x1 - (float(resolution[0]) - margin_x)))
            penalties.append(max(0.0, y1 - (float(resolution[1]) - margin_y)))
        return fit_ok, float(sum(penalties))

    adjusted = {}
    for ln, row in per_link_views.items():
        if not isinstance(row, dict):
            continue
        base_view = {
            "azimuth_deg": int(row["azimuth_deg"]),
            "elevation_deg": int(row["elevation_deg"]),
            "distance_scale": float(row["distance_scale"]),
            "fov_deg": int(row["fov_deg"]),
        }
        chosen = dict(base_view)
        ds0 = float(base_view["distance_scale"])
        min_ds = 0.35
        fit0, _penalty0 = _score_scale(base_view, ds0)
        if fit0:
            lo = min_ds
            hi = ds0
            fit_lo, _penalty_lo = _score_scale(base_view, lo)
            if fit_lo:
                best = lo
            else:
                best = hi
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    fit_mid, _pen_mid = _score_scale(base_view, mid)
                    if fit_mid:
                        best = mid
                        hi = mid
                    else:
                        lo = mid
            chosen["distance_scale"] = round(float(best), 3)
            adjusted[str(ln)] = chosen
            continue
        lo = ds0
        hi = max(ds0, 1.05)
        fit_hi, _penalty_hi = _score_scale(base_view, hi)
        max_ds = 6.0
        while (not fit_hi) and hi < max_ds:
            hi = min(max_ds, hi * 1.25)
            fit_hi, _penalty_hi = _score_scale(base_view, hi)
        if fit_hi:
            best = hi
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                fit_mid, _pen_mid = _score_scale(base_view, mid)
                if fit_mid:
                    best = mid
                    hi = mid
                else:
                    lo = mid
            chosen["distance_scale"] = round(float(best), 3)
        else:
            chosen["distance_scale"] = round(float(hi), 3)
        adjusted[str(ln)] = chosen
    for ln in required_links or []:
        if str(ln) not in adjusted and str(ln) in per_link_views:
            adjusted[str(ln)] = dict(per_link_views[str(ln)])
    return adjusted


def _screen_direction_label(v2: np.ndarray) -> str:
    v = np.asarray(v2, dtype=float).reshape(-1)
    if v.size < 2:
        return "screen_static"
    x, y = float(v[0]), float(v[1])
    mag = float(np.linalg.norm([x, y]))
    if mag <= 1e-6:
        return "screen_static"
    nx, ny = x / mag, y / mag
    ax, ay = abs(nx), abs(ny)
    if ax > 0.85:
        return "screen_right" if nx > 0 else "screen_left"
    if ay > 0.85:
        return "screen_down" if ny > 0 else "screen_up"
    if nx > 0 and ny > 0:
        return "screen_down_right"
    if nx > 0 and ny < 0:
        return "screen_up_right"
    if nx < 0 and ny > 0:
        return "screen_down_left"
    return "screen_up_left"


def _signed_axis_label_from_world(axis_world: np.ndarray | list[float] | tuple[float, ...]) -> str:
    vec = np.asarray(axis_world, dtype=float).reshape(-1)
    if vec.size < 3:
        return "+X"
    vec = vec[:3]
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if float(vec[idx]) >= 0.0 else "-"
    return f"{sign}{['X', 'Y', 'Z'][idx]}"


def _trajectory_local_motion_descriptor(
    *,
    joints: list[dict],
    link_tf_map: dict[str, np.ndarray],
    link_name: str,
    joint_name: str,
    joint_type: str,
    delta_q: float,
    trend: str,
) -> dict | None:
    joint = None
    for j in joints or []:
        if str(j.get("name") or "").strip() == str(joint_name or "").strip():
            joint = j
            break
    if not isinstance(joint, dict):
        return None
    axis_local = np.asarray(joint.get("axis") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
    if axis_local.size < 3:
        return None
    axis_local = axis_local[:3]
    n = float(np.linalg.norm(axis_local))
    if n <= 1.0e-8:
        return None
    axis_local = axis_local / n
    tf = np.asarray(link_tf_map.get(str(link_name), np.eye(4)), dtype=float)
    axis_world = np.asarray(tf[:3, :3], dtype=float) @ axis_local
    axis_world = axis_world / max(1.0e-8, float(np.linalg.norm(axis_world)))
    joint_type_norm = str(joint_type or "").strip().lower()
    if joint_type_norm in {"revolute", "continuous"}:
        if abs(float(delta_q)) > 1.0e-12:
            direction = "ccw" if float(delta_q) > 0.0 else "cw"
            sign = 1.0 if float(delta_q) > 0.0 else -1.0
        elif str(trend).strip().lower() == "increase":
            direction = "ccw"
            sign = 1.0
        elif str(trend).strip().lower() == "decrease":
            direction = "cw"
            sign = -1.0
        else:
            direction = "static"
            sign = 1.0
        signed_axis = axis_world * float(sign)
        axis_label = _signed_axis_label_from_world(signed_axis)
        local_motion_text = f"{direction} around {axis_label}" if direction != "static" else f"static around {axis_label}"
        return {
            "motion_type": "rotation",
            "direction": direction,
            "axis_world": [float(axis_world[0]), float(axis_world[1]), float(axis_world[2])],
            "signed_axis_world": [float(signed_axis[0]), float(signed_axis[1]), float(signed_axis[2])],
            "axis_label": axis_label,
            "local_motion_text": local_motion_text,
            "frame_note": "axis_relative_not_view_relative",
        }
    if joint_type_norm == "prismatic":
        if abs(float(delta_q)) > 1.0e-12:
            sign = 1.0 if float(delta_q) > 0.0 else -1.0
        elif str(trend).strip().lower() == "increase":
            sign = 1.0
        elif str(trend).strip().lower() == "decrease":
            sign = -1.0
        else:
            sign = 0.0
        signed_axis = axis_world * (sign if abs(sign) > 0.0 else 1.0)
        axis_label = _signed_axis_label_from_world(signed_axis)
        local_motion_text = f"along {axis_label}" if abs(sign) > 0.0 else f"static along {axis_label}"
        return {
            "motion_type": "prismatic",
            "direction": ("positive_axis" if sign > 0.0 else "negative_axis" if sign < 0.0 else "static"),
            "axis_world": [float(axis_world[0]), float(axis_world[1]), float(axis_world[2])],
            "signed_axis_world": [float(signed_axis[0]), float(signed_axis[1]), float(signed_axis[2])],
            "axis_label": axis_label,
            "local_motion_text": local_motion_text,
            "frame_note": "axis_relative_not_view_relative",
        }
    return None


def _classify_motion_cue_for_view(
    motion: dict,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    force_rotation: bool = False,
    force_rotation_direction: str | None = None,
) -> dict:
    pts = np.stack(
        [
            np.asarray(motion["center_prev_world"], dtype=float),
            np.asarray(motion["center_curr_world"], dtype=float),
            np.asarray(motion["ref_prev_world"], dtype=float),
            np.asarray(motion["ref_curr_world"], dtype=float),
        ],
        axis=0,
    )
    proj = lr.gop.project_points(pts, camera, resolution)
    if proj.shape[0] != 4 or np.any(proj[:, 2] <= 0):
        return {"type": "unknown", "direction": "unknown"}
    c0 = np.asarray([proj[0, 0], proj[0, 1]], dtype=float)
    c1 = np.asarray([proj[1, 0], proj[1, 1]], dtype=float)
    r0 = np.asarray([proj[2, 0], proj[2, 1]], dtype=float)
    r1 = np.asarray([proj[3, 0], proj[3, 1]], dtype=float)
    trans = c1 - c0
    v0 = r0 - c0
    v1 = r1 - c1
    nv0 = float(np.linalg.norm(v0))
    nv1 = float(np.linalg.norm(v1))
    if nv0 <= 1e-6 or nv1 <= 1e-6:
        return {"type": "unknown", "direction": "unknown"}
    cross = float(v0[0] * v1[1] - v0[1] * v1[0])
    dot = float(v0[0] * v1[0] + v0[1] * v1[1])
    angle = float(math.atan2(cross, dot))
    rotate_deg = abs(float(math.degrees(angle)))
    rot_disp = float(np.linalg.norm(v1 - v0))
    trans_mag = float(np.linalg.norm(trans))
    is_rot = (rotate_deg >= float(lr.MOTION_ROTATE_MIN_DEG)) and (rot_disp >= max(2.0, 0.45 * trans_mag))
    if bool(force_rotation) and (rot_disp > 1e-3 or trans_mag > 1e-3):
        is_rot = True
    if is_rot:
        if isinstance(force_rotation_direction, str):
            d = force_rotation_direction.strip().lower()
            if d in {"cw", "ccw"}:
                return {"type": "rotation", "direction": d}
        return {"type": "rotation", "direction": ("cw" if cross > 0 else "ccw")}
    flow = np.asarray(trans, dtype=float)
    if float(np.linalg.norm(flow)) <= 1e-6:
        flow = np.asarray(r1 - r0, dtype=float)
    return {"type": "translation", "direction": _screen_direction_label(flow)}


def _build_sample_motion_cues(
    asset_ctx: dict,
    traj_npz_data,
    frame_idx: int,
    viewspecs: dict,
    camera_anchor_center: np.ndarray,
    camera_anchor_radius: float,
    resolution: tuple[int, int],
    required_links: list[str],
    joints: list[dict],
    label_legend: dict[str, str],
    motion_window: tuple[int, int] | None = None,
    force_rotation_links: list[str] | set[str] | None = None,
    force_rotation_direction_map: dict[str, str] | None = None,
    base_axis_world: list[float] | None = None,
    planned_base_axis_world: list[float] | None = None,
) -> dict:
    _ = (asset_ctx, frame_idx, viewspecs, camera_anchor_center, camera_anchor_radius, resolution, label_legend, force_rotation_links, force_rotation_direction_map)
    joint_names = [str(x) for x in traj_npz_data["joint_names"].tolist()]
    joint_angles = np.asarray(traj_npz_data["joint_angles"], dtype=float)
    base_translation = np.asarray(traj_npz_data["base_translation"], dtype=float)
    n_frames = int(joint_angles.shape[0]) if joint_angles.ndim == 2 else 0
    if n_frames <= 0:
        return {"joint_trends": [], "base_motion": None, "summary_text": "no_trajectory_data"}
    if motion_window is None:
        i0 = max(0, min(n_frames - 1, int(frame_idx) - 1))
        i1 = max(0, min(n_frames - 1, int(frame_idx) + 1))
        if i1 < i0:
            i1 = i0
    else:
        i0 = max(0, min(n_frames - 1, int(motion_window[0])))
        i1 = max(0, min(n_frames - 1, int(motion_window[1])))
        if i1 < i0:
            i1 = i0

    joint_idx = {jn: i for i, jn in enumerate(joint_names)}
    child_to_joint = _build_child_to_joint_map(joints)
    joint_type_by_name = {}
    for j in joints or []:
        jn = str(j.get("name") or "").strip()
        if not jn:
            continue
        joint_type_by_name[jn] = str(j.get("type") or "").strip().lower() or "unknown"
    joint_pos_axis = {jn: float(joint_angles[i0, j]) for j, jn in enumerate(joint_names)}
    base_tf_axis = np.eye(4)
    if base_translation.ndim == 2 and i0 < base_translation.shape[0]:
        base_tf_axis[:3, 3] = np.asarray(base_translation[i0], dtype=float)
    link_tf_axis = lr.rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], joint_pos_axis, base_tf=base_tf_axis)

    delta_q_static_th = float(os.environ.get("CODEX_MOTION_TREND_DELTAQ_STATIC_TH", "1e-4"))
    joint_trends = []
    for ln in required_links:
        jn = child_to_joint.get(ln)
        if not jn:
            continue
        ji = joint_idx.get(jn)
        if ji is None:
            continue
        delta_q = float(joint_angles[i1, ji] - joint_angles[i0, ji])
        if abs(delta_q) <= delta_q_static_th:
            trend = "static"
        else:
            trend = "increase" if delta_q > 0.0 else "decrease"
        joint_trends.append(
            {
                "joint": str(jn),
                "link": str(ln),
                "joint_type": str(joint_type_by_name.get(jn, "unknown")),
                "trend": trend,
                "delta_q": delta_q,
                **(
                    _trajectory_local_motion_descriptor(
                        joints=joints,
                        link_tf_map=link_tf_axis,
                        link_name=str(ln),
                        joint_name=str(jn),
                        joint_type=str(joint_type_by_name.get(jn, "unknown")),
                        delta_q=float(delta_q),
                        trend=str(trend),
                    )
                    or {}
                ),
            }
        )
    joint_trends.sort(key=lambda x: x["joint"])

    active_base_axis_world = None
    if isinstance(base_axis_world, list) and len(base_axis_world) == 3:
        active_base_axis_world = list(base_axis_world)
    elif isinstance(planned_base_axis_world, list) and len(planned_base_axis_world) == 3:
        active_base_axis_world = list(planned_base_axis_world)

    base_motion = None
    if base_translation.ndim == 2 and base_translation.shape[0] > 0:
        base_delta = np.asarray(base_translation[i1] - base_translation[i0], dtype=float)
        if isinstance(active_base_axis_world, list) and len(active_base_axis_world) == 3:
            try:
                axis = np.asarray(
                    [float(active_base_axis_world[0]), float(active_base_axis_world[1]), float(active_base_axis_world[2])],
                    dtype=float,
                )
            except Exception:
                axis = None
            if axis is not None:
                n = float(np.linalg.norm(axis))
                axis = axis / n if n > 1e-9 else None
        else:
            axis = None
        if axis is None:
            k = int(np.argmax(np.abs(base_delta)))
            axis = np.zeros((3,), dtype=float)
            axis[k] = 1.0
        delta_proj = float(np.dot(base_delta, axis))
        delta_proj_static_th = float(os.environ.get("CODEX_MOTION_TREND_BASE_STATIC_TH", "1e-4"))
        if abs(delta_proj) <= delta_proj_static_th:
            base_trend = "static"
        else:
            base_trend = "positive" if delta_proj > 0.0 else "negative"
        base_motion = {
            "body": "base",
            "motion_type": "translation",
            "axis_world": [float(axis[0]), float(axis[1]), float(axis[2])],
            "trend": base_trend,
            "delta_proj": delta_proj,
        }
    planned_base_motion = None
    if isinstance(active_base_axis_world, list) and len(active_base_axis_world) == 3:
        planned_base_motion = {
            "body": "base",
            "motion_type": "translation",
            "axis_world": [
                float(active_base_axis_world[0]),
                float(active_base_axis_world[1]),
                float(active_base_axis_world[2]),
            ],
            "trend": "positive",
        }

    if joint_trends:
        trend_txt = "; ".join(f"{x['joint']}={x['trend']}({x['delta_q']:.4f})" for x in joint_trends)
    else:
        trend_txt = "no_joint_motion_detected"
    if isinstance(base_motion, dict):
        trend_txt += f" | base={base_motion['trend']}({base_motion['delta_proj']:.4f})"
    return {
        "frame_idx": int(frame_idx),
        "window_frames": [int(i0), int(i1)],
        "joint_trends": joint_trends,
        "base_motion": base_motion,
        "planned_base_motion": planned_base_motion,
        "has_planned_base_motion": bool(planned_base_motion),
        "summary_text": trend_txt,
    }


def _filter_motion_cues_for_link(cues: dict | None, link_name: str, joint_name: str | None = None) -> dict:
    if not isinstance(cues, dict):
        return {"joint_trends": [], "base_motion": None, "planned_base_motion": None, "has_planned_base_motion": False, "summary_text": ""}
    joint_name = str(joint_name or "").strip()
    filtered_joint_trends = []
    for jt in (cues.get("joint_trends") or []):
        if not isinstance(jt, dict):
            continue
        jt_joint = str(jt.get("joint") or "").strip()
        jt_link = str(jt.get("link") or "").strip()
        if joint_name:
            if jt_joint != joint_name and jt_link != str(link_name):
                continue
        elif jt_link != str(link_name):
            continue
        filtered_joint_trends.append(dict(jt))
    return {
        "joint_trends": filtered_joint_trends,
        "local_motion": (dict(cues.get("local_motion")) if isinstance(cues.get("local_motion"), dict) else None),
        "base_motion": cues.get("base_motion") if isinstance(cues.get("base_motion"), dict) else None,
        "planned_base_motion": cues.get("planned_base_motion") if isinstance(cues.get("planned_base_motion"), dict) else None,
        "has_planned_base_motion": bool(cues.get("has_planned_base_motion")),
        "summary_text": str(cues.get("summary_text") or ""),
    }


def _build_trajectory_summary(traj_npz_data, plan_obj: dict, timeline_sample_catalog: list[dict]) -> dict:
    def _trend_from_delta(delta: float, static_th: float = 1.0e-4) -> str:
        d = float(delta)
        if abs(d) <= float(static_th):
            return "static"
        return "increase" if d > 0.0 else "decrease"

    def _base_motion_brief(base_arr: np.ndarray, f0: int, f1: int) -> dict:
        if base_arr.ndim != 2 or base_arr.shape[0] == 0:
            return {
                "delta_translation_magnitude": 0.0,
                "trend": "static",
                "dominant_axis": "X",
            }
        delta = np.asarray(base_arr[f1] - base_arr[f0], dtype=float)
        mag = float(np.linalg.norm(delta))
        axis_idx = int(np.argmax(np.abs(delta))) if delta.size >= 3 else 0
        axis_name = ["X", "Y", "Z"][axis_idx]
        signed = float(delta[axis_idx]) if delta.size >= 3 else 0.0
        return {
            "delta_translation_magnitude": mag,
            "trend": _trend_from_delta(signed),
            "dominant_axis": axis_name,
        }

    joint_names = [str(x) for x in traj_npz_data["joint_names"].tolist()]
    joint_angles = np.asarray(traj_npz_data["joint_angles"])
    base_translation = np.asarray(traj_npz_data["base_translation"])
    time_s = np.asarray(traj_npz_data["time_s"])
    joint_idx = {jn: i for i, jn in enumerate(joint_names)}

    # Focus on joints referenced by the current plan timeline to keep prompt compact.
    planned_joints = []
    for seg in (plan_obj.get("timeline") or []):
        for ctrl in (seg.get("controls") or []):
            if isinstance(ctrl.get("joint"), str):
                planned_joints.append(str(ctrl["joint"]))
            elif isinstance(ctrl.get("joints"), list):
                planned_joints.extend([str(j) for j in ctrl.get("joints") if isinstance(j, str)])
    planned_joints = [j for j in dict.fromkeys(planned_joints) if j in joint_idx]
    if not planned_joints:
        planned_joints = joint_names[: min(8, len(joint_names))]

    local_motion_by_joint = {}
    for row in timeline_sample_catalog or []:
        if not isinstance(row, dict):
            continue
        cues = row.get("motion_cues") if isinstance(row.get("motion_cues"), dict) else {}
        for jt in cues.get("joint_trends") or []:
            if not isinstance(jt, dict):
                continue
            joint_name = str(jt.get("joint") or "").strip()
            if not joint_name or joint_name in local_motion_by_joint:
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
            if local_motion:
                local_motion_by_joint[joint_name] = local_motion

    joints_summary = {}
    for jn in planned_joints[:12]:
        arr = np.asarray(joint_angles[:, joint_idx[jn]], dtype=float)
        delta = float(arr[-1] - arr[0])
        row = {
            "start_q": float(arr[0]),
            "final_q": float(arr[-1]),
            "min_q": float(np.min(arr)),
            "max_q": float(np.max(arr)),
            "delta_q": delta,
            "delta_q_magnitude": abs(delta),
            "trend": _trend_from_delta(delta),
        }
        if jn in local_motion_by_joint:
            row["local_motion"] = dict(local_motion_by_joint[jn])
        joints_summary[jn] = row

    base_summary = _base_motion_brief(base_translation, 0, max(0, len(base_translation) - 1))

    return {
        "num_frames": int(joint_angles.shape[0]),
        "duration_s": float(time_s[-1]) if len(time_s) else 0.0,
        "tracked_joints": planned_joints[:12],
        "joints_summary": joints_summary,
        "base_translation_summary": base_summary,
    }


def _build_timeline_trajectory_detail(traj_npz_data, plan_obj: dict, timeline_sample_catalog: list[dict] | None = None) -> dict:
    def _trend_from_delta(delta: float, static_th: float = 1.0e-4) -> str:
        d = float(delta)
        if abs(d) <= float(static_th):
            return "static"
        return "increase" if d > 0.0 else "decrease"

    def _base_motion_brief(base_arr: np.ndarray, f0: int, f1: int) -> dict:
        if base_arr.ndim != 2 or base_arr.shape[0] == 0:
            return {
                "delta_translation_magnitude": 0.0,
                "trend": "static",
                "dominant_axis": "X",
            }
        delta = np.asarray(base_arr[f1] - base_arr[f0], dtype=float)
        mag = float(np.linalg.norm(delta))
        axis_idx = int(np.argmax(np.abs(delta))) if delta.size >= 3 else 0
        axis_name = ["X", "Y", "Z"][axis_idx]
        signed = float(delta[axis_idx]) if delta.size >= 3 else 0.0
        return {
            "delta_translation_magnitude": mag,
            "trend": _trend_from_delta(signed),
            "dominant_axis": axis_name,
        }

    joint_names = [str(x) for x in traj_npz_data["joint_names"].tolist()]
    joint_angles = np.asarray(traj_npz_data["joint_angles"])
    base_translation = np.asarray(traj_npz_data["base_translation"])
    joint_idx = {jn: i for i, jn in enumerate(joint_names)}
    fps = float((plan_obj.get("meta") or {}).get("fps", 30))
    local_motion_by_joint = {}
    for row in timeline_sample_catalog or []:
        if not isinstance(row, dict):
            continue
        cues = row.get("motion_cues") if isinstance(row.get("motion_cues"), dict) else {}
        for jt in cues.get("joint_trends") or []:
            if not isinstance(jt, dict):
                continue
            joint_name = str(jt.get("joint") or "").strip()
            if not joint_name or joint_name in local_motion_by_joint:
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
            if local_motion:
                local_motion_by_joint[joint_name] = local_motion

    timeline_detail = []
    prev_joint_mag_by_name: dict[str, float] = {}

    # Track plan-referenced joints first; fallback to all joints if none referenced.
    planned_joints = []
    for seg in (plan_obj.get("timeline") or []):
        for ctrl in (seg.get("controls") or []):
            if isinstance(ctrl.get("joint"), str):
                planned_joints.append(str(ctrl["joint"]))
            elif isinstance(ctrl.get("joints"), list):
                planned_joints.extend([str(j) for j in ctrl.get("joints") if isinstance(j, str)])
    planned_joints = [j for j in dict.fromkeys(planned_joints) if j in joint_idx]
    if not planned_joints:
        planned_joints = joint_names[: min(12, len(joint_names))]
    tracked = planned_joints[:12]

    for si, seg in enumerate(plan_obj.get("timeline") or []):
        try:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", t0))
        except Exception:
            continue
        if t1 < t0:
            continue
        f0 = int(max(0, min(joint_angles.shape[0] - 1, round(t0 * fps)))) if joint_angles.shape[0] else 0
        f1 = int(max(0, min(joint_angles.shape[0] - 1, round(t1 * fps)))) if joint_angles.shape[0] else 0
        per_joint = {}
        for jn in tracked:
            arr = np.asarray(joint_angles[:, joint_idx[jn]], dtype=float)
            seg_arr = arr[min(f0, f1) : max(f0, f1) + 1]
            delta = float(arr[f1] - arr[f0])
            delta_mag = abs(delta)
            row = {
                "start_q": float(arr[f0]),
                "end_q": float(arr[f1]),
                "min_q": float(np.min(seg_arr)) if seg_arr.size else float(arr[f0]),
                "max_q": float(np.max(seg_arr)) if seg_arr.size else float(arr[f0]),
                "delta_q": delta,
                "delta_q_magnitude": delta_mag,
                "trend": _trend_from_delta(delta),
            }
            prev_mag = prev_joint_mag_by_name.get(jn)
            if prev_mag is not None and prev_mag > 1.0e-6:
                row["decay_estimate"] = float(delta_mag / prev_mag)
            prev_joint_mag_by_name[jn] = delta_mag
            if jn in local_motion_by_joint:
                row["local_motion"] = dict(local_motion_by_joint[jn])
            per_joint[jn] = {
                **row,
            }
        timeline_detail.append(
            {
                "segment_index": si,
                "segment_name": str(seg.get("name") or f"seg_{si}"),
                "phase_type": str(seg.get("phase_type") or ""),
                "t0": t0,
                "t1": t1,
                "frame_range": [int(f0), int(f1)],
                "tracked_joints": tracked,
                "per_joint_motion": per_joint,
                "base_motion": _base_motion_brief(base_translation, f0, f1),
            }
        )
    return {"timeline_segment_motion": timeline_detail}


def execute_plan_once(py: str, asset_root: Path, plan_path: Path, out_dir: Path, resolution, use_glb_scene: str | None, skip_plan_frames=True, debug_motion=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_npz = out_dir / "trajectory.npz"
    traj_jsonl = out_dir / "trajectory.jsonl"
    cmd = [
        py,
        "tools/run_plan.py",
        "--asset_root",
        str(asset_root),
        "--plan_json",
        str(plan_path),
        "--out",
        str(out_dir),
        "--trajectory_npz",
        str(traj_npz),
        "--trajectory_jsonl",
        str(traj_jsonl),
        "--resolution",
        str(resolution[0]),
        str(resolution[1]),
        "--export_animated_glb",
    ]
    if use_glb_scene and use_glb_scene.lower() != "none":
        glb = find_glb_scene(asset_root) if use_glb_scene == "auto" else Path(use_glb_scene)
        if glb is None or not Path(glb).exists():
            raise SystemExit(
                f"Canonical textured mesh not found: "
                f"{asset_root / f'animated_textured_{asset_root.name}.glb'}"
            )
        cmd.extend(["--use_glb_scene", str(glb)])
    if skip_plan_frames:
        cmd.append("--skip_frame_render")
    if debug_motion:
        cmd.append("--debug_motion")
    try:
        run(cmd)
    except subprocess.CalledProcessError:
        # Fallback for assets whose canonical textured GLB has small rest-pose drift
        # relative to URDF (common in third-party converted meshes).
        if use_glb_scene and use_glb_scene.lower() != "none" and "--skip_glb_alignment_check" not in cmd:
            retry_cmd = list(cmd) + ["--skip_glb_alignment_check"]
            print("[WARN] run_plan failed on GLB alignment check; retrying with --skip_glb_alignment_check")
            run(retry_cmd)
        else:
            raise
    return traj_npz, traj_jsonl, (out_dir / "plan_animated.glb")


def _motion_stage_context(traj_npz_path: Path) -> dict:
    traj = np.load(traj_npz_path, allow_pickle=True)
    T = int(np.asarray(traj["joint_angles"]).shape[0]) if "joint_angles" in traj else 0
    stage_frames = {"start": 0, "mid": max(0, min(T - 1, int(round(T * 0.5)))), "end": max(0, T - 1)}
    return {
        "source": "trajectory",
        "stage_frames": stage_frames,
        "checks": [],
        "failure_signature": {"category": "none", "codes": [], "severity": "low"},
    }


def main():
    parser = argparse.ArgumentParser(description="Agent loop: coverage loop + motion loop with param-only plan patches")
    parser.add_argument("--asset_root", required=True)
    parser.add_argument("--action_text", required=True)
    parser.add_argument("--out_root", default="outputs")
    parser.add_argument("--vlm_model", default="gpt-5.4")
    parser.add_argument("--llm_model", default="gpt-5.4")
    parser.add_argument("--api_key", default=None, help="Single API key used by VLM/LLM calls.")
    parser.add_argument("--api_provider", default="auto", choices=["auto", "openai", "gemini"])
    parser.add_argument("--api_base_url", default=None, help="Optional API base URL override")
    parser.add_argument("--vlm_conditioning_images", nargs="*", default=[], help="Optional extra object images for VLM grounding/checks.")
    parser.add_argument("--vlm_conditioning_masks", nargs="*", default=[], help="Optional target mask image(s) for VLM grounding/checks.")
    parser.add_argument("--vlm_conditioning_text", default=None, help="Optional extra text hint for VLM grounding/checks.")
    parser.add_argument("--vlm_no_auto_images", action="store_true", help="For VLM stage, do not attach generated auto images.")
    parser.add_argument("--input_images", nargs="*", default=None, help="Pipeline-level alias of --vlm_conditioning_images.")
    parser.add_argument("--input_masks", nargs="*", default=None, help="Pipeline-level alias of --vlm_conditioning_masks.")
    parser.add_argument("--input_text", default=None, help="Pipeline-level extra text input for grounding; defaults to action_text.")
    parser.add_argument("--resolution", type=int, nargs=2, default=[800, 600])
    parser.add_argument("--use_glb_scene", default="auto")
    parser.add_argument("--debug_motion", action="store_true")
    parser.add_argument("--skip_plan_frames", action="store_true")
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_vlm", action="store_true")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument(
        "--disable_loop_vlm_api",
        action="store_true",
        help="Use heuristic stubs instead of API-based VLM for coverage/motion loop steps",
    )
    parser.add_argument("--enable_coverage_loop", action="store_true")
    parser.add_argument("--enable_motion_loop", action="store_true")
    parser.add_argument(
        "--skip_coverage_loop",
        action="store_true",
        help="Skip coverage loop and reuse previously selected motion/coverage views when available",
    )
    parser.add_argument("--coverage_max_iters", type=int, default=2)
    parser.add_argument("--motion_max_iters", type=int, default=3)
    args = parser.parse_args()

    if args.skip_coverage_loop:
        args.enable_coverage_loop = False
    if args.enable_motion_loop and not args.enable_coverage_loop and not args.skip_coverage_loop:
        # Motion loop depends on viewpoint quality; run coverage loop first by default.
        args.enable_coverage_loop = True
    if not args.enable_coverage_loop and not args.enable_motion_loop:
        if args.skip_coverage_loop:
            args.enable_motion_loop = True
        else:
            args.enable_coverage_loop = True
            args.enable_motion_loop = True
    use_loop_vlm_api = not args.disable_loop_vlm_api
    if args.input_images is not None:
        args.vlm_conditioning_images = list(args.input_images)
    if args.input_masks is not None:
        args.vlm_conditioning_masks = list(args.input_masks)
    if args.input_text is not None:
        args.vlm_conditioning_text = args.input_text

    asset_root = Path(args.asset_root).absolute()
    out_root = Path(args.out_root).absolute()
    if not asset_root.exists():
        raise SystemExit(f"Asset root not found: {asset_root}")
    asset_name = asset_root.name
    asset_out = out_root / asset_name
    asset_out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    t_start = time.time()
    _prepare_pipeline_if_needed(args, py, asset_out)

    manifest_path = asset_out / "prompts" / "vlm_input_manifest.json"
    if manifest_path.exists():
        try:
            manifest = _load_json(manifest_path)
        except Exception:
            manifest = {}
    else:
        manifest = {}
    if (not args.vlm_conditioning_images) and isinstance(manifest.get("conditioning_images"), list):
        args.vlm_conditioning_images = [str(x) for x in manifest.get("conditioning_images") if str(x)]
    if (not args.vlm_conditioning_masks) and isinstance(manifest.get("conditioning_masks"), list):
        args.vlm_conditioning_masks = [str(x) for x in manifest.get("conditioning_masks") if str(x)]
    if (not args.vlm_conditioning_text) and str(manifest.get("conditioning_text") or "").strip():
        args.vlm_conditioning_text = str(manifest.get("conditioning_text")).strip()

    causal_json = asset_out / "causal.json"
    if not causal_json.exists():
        causal_json = asset_out / "output.json"
    plan_json = asset_out / "plan.json"
    if not plan_json.exists() or not causal_json.exists():
        raise SystemExit(f"Missing plan or causal JSON under {asset_out}")

    urdf = find_urdf(asset_root)
    if urdf is None:
        raise SystemExit(f"No URDF found under {asset_root}")
    _, joints = lr.rp.parse_urdf(urdf)
    causal = _sanitize_causal_action_targets(_load_json(causal_json), joints)

    loop_root = asset_out / "loop"
    coverage_root = loop_root / "coverage"
    motion_root = loop_root / "motion"
    iterations_root = loop_root / "iterations"
    coverage_root.mkdir(parents=True, exist_ok=True)
    motion_root.mkdir(parents=True, exist_ok=True)
    iterations_root.mkdir(parents=True, exist_ok=True)

    current_viewspecs = dict(lr.DEFAULT_VIEWSPECS)
    coverage_history = []
    coverage_ok = False
    coverage_report = None
    coverage_report_path_final = None
    current_link_viewspecs: dict[str, dict] = {}

    asset_ctx = lr.load_asset_context(asset_root)
    required_links = cov_verify.build_required_links(causal, joints)
    initial_plan_obj = ask_plan_mod.normalize_plan_json(_load_json(plan_json), asset_root=asset_root, vlm=causal)
    initial_wheel_links = wheel_diag.wheel_links_for_asset(initial_plan_obj, asset_root, joints, required_links=None)
    for ln in initial_wheel_links:
        if str(ln) and str(ln) not in required_links:
            required_links.append(str(ln))
    coverage_render_links = _expand_motion_render_links(
        list(dict.fromkeys(list(required_links) + list(_collect_motion_case_links(initial_plan_obj, joints)))),
        joints,
    )
    visual_link_order = [ln for ln, meshes in asset_ctx.get("link_meshes", {}).items() if meshes]
    motion_label_legend = _build_motion_label_legend(coverage_render_links, visual_link_order, joints)
    rest_link_tf = lr.rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], {})
    rest_world_link_meshes = lr.transform_link_meshes(asset_ctx["link_meshes"], rest_link_tf)
    rest_center, rest_radius = lr.compute_base_center_radius(rest_world_link_meshes)
    scale_context_path = asset_out / "scale_context.json"
    if scale_context_path.exists():
        try:
            scale_context = _load_json(scale_context_path)
        except Exception:
            scale_context = None
    else:
        scale_context = None
    if not isinstance(scale_context, dict):
        glb_for_scale = find_glb_scene(asset_root) if args.use_glb_scene != "none" else None
        scale_context = scu.build_scale_context(asset_name, rest_world_link_meshes, joints=joints, glb_path=glb_for_scale)
        scu.save_scale_context(scale_context_path, scale_context)
    coverage_motion_hints = _build_coverage_motion_hints(causal, initial_plan_obj, joints, required_links)

    conditioning_mask_paths = _expand_existing_paths(args.vlm_conditioning_masks)
    mask_bundle_dir = loop_root / "mask_inputs"
    mask_bundle_images, mask_bundle_meta = _build_mask_panel(
        conditioning_mask_paths,
        mask_bundle_dir / "conditioning_masks_panel.png",
    )
    _write_json(mask_bundle_dir / "mask_bundle_meta.json", mask_bundle_meta)
    has_conditioning_masks = len(mask_bundle_images) > 0
    mask_meta_for_prompt = mask_bundle_meta if has_conditioning_masks else None
    conditioning_text_for_motion = None
    if isinstance(args.vlm_conditioning_text, str):
        txt = args.vlm_conditioning_text.strip()
        if txt and txt != str(args.action_text).strip():
            conditioning_text_for_motion = txt
    requested_motion_render_backend = _norm_motion_render_backend()
    preferred_reference_backend = _resolve_reference_backend(asset_out, requested_motion_render_backend)
    if preferred_reference_backend != requested_motion_render_backend:
        print(
            f"[INFO] Using saved reference render backend decision for {asset_name}: "
            f"{preferred_reference_backend} (requested={requested_motion_render_backend})."
        )
    coverage_ghost_link_tf = None
    coverage_ghost_info = None
    preview_plan_path = iterations_root / "plan_iter00.json"
    preview_causal_path = iterations_root / "causal_iter00.json"
    _write_json(preview_plan_path, initial_plan_obj)
    _write_json(preview_causal_path, causal)
    if args.enable_coverage_loop and not args.skip_coverage_loop:
        try:
            preview_exec_dir = coverage_root / "preview_exec"
            preview_traj_npz, _preview_traj_jsonl, _preview_glb = execute_plan_once(
                py,
                asset_root,
                preview_plan_path,
                preview_exec_dir,
                args.resolution,
                args.use_glb_scene,
                skip_plan_frames=True,
                debug_motion=False,
            )
            best_fi, best_link_tf, ghost_info = _select_peak_motion_frame(
                asset_ctx,
                preview_traj_npz,
                initial_plan_obj,
                joints,
                required_links,
            )
            if best_fi is not None and isinstance(best_link_tf, dict):
                coverage_ghost_link_tf = best_link_tf
                coverage_ghost_info = ghost_info
        except Exception as exc:
            print(f"[WARN] Coverage ghost preview failed ({exc}); continuing without future ghost overlay.")
    if args.skip_coverage_loop:
        saved_viewspecs, saved_viewspecs_path = _load_saved_loop_viewspecs(loop_root)
        if saved_viewspecs is not None:
            current_viewspecs = lr.validate_viewspecs(saved_viewspecs)
            current_link_viewspecs = {
                str(k): dict(v)
                for k, v in ((saved_viewspecs.get("per_link_views") or {}) if isinstance(saved_viewspecs, dict) else {}).items()
                if isinstance(v, dict)
            }
            print(f"[INFO] Skipping coverage loop and reusing saved viewspecs from {saved_viewspecs_path}.")
            _log_per_link_viewspecs("Reused saved per-link motion views:", current_link_viewspecs)
        else:
            print("[INFO] Skipping coverage loop with no saved selected viewspecs; using default motion views.")

    coverage_selected_viewspecs = None
    if args.enable_coverage_loop:
        for c_iter in range(args.coverage_max_iters + 1):
            c_dir = coverage_root / f"iter{c_iter:02d}"
            cov_render = lr.render_coverage_grid(
                asset_ctx,
                current_viewspecs,
                c_dir,
                resolution=tuple(args.resolution),
                label_links=coverage_render_links,
                label_mode="name",
                style="reference",
                preferred_reference_backend=preferred_reference_backend,
                camera_anchor_center=rest_center,
                camera_anchor_radius=float(rest_radius),
            )
            report = cov_verify.verify_coverage_arrays(
                cov_render["link_names"],
                cov_render["viewspecs"]["views"] and [v["id"] for v in cov_render["viewspecs"]["views"]] or [],
                cov_render["visible_px"],
                cov_render["visible_ratio"],
                required_links,
            )
            _write_json(c_dir / "coverage_report.json", report)
            _write_json(c_dir / "coverage_viewspecs.json", cov_render["viewspecs"])
            coverage_report_path_final = c_dir / "coverage_report.json"
            try:
                coverage_prompt_text = cov_vlm._build_prompt(
                    report,
                    current_viewspecs,
                    mask_bundle_meta,
                    causal,
                    coverage_motion_hints,
                )
                (c_dir / "coverage_vlm_prompt.txt").write_text(coverage_prompt_text, encoding="utf-8")
                _write_json(
                    c_dir / "coverage_vlm_prompt_inputs.json",
                    {
                        "coverage_grid": str((c_dir / "coverage_grid.png").absolute()),
                        "coverage_images": list(cov_render.get("view_image_paths") or []),
                        "conditioning_mask_images": [str(Path(p).absolute()) for p in mask_bundle_images],
                        "conditioning_mask_meta": str((mask_bundle_dir / "mask_bundle_meta.json").absolute()),
                        "coverage_report": str((c_dir / "coverage_report.json").absolute()),
                        "coverage_viewspecs": str((c_dir / "coverage_viewspecs.json").absolute()),
                        "causal_json": str((asset_out / "causal.json").absolute()) if (asset_out / "causal.json").exists() else None,
                        "expected_motion_hints": coverage_motion_hints,
                        "use_api": bool(use_loop_vlm_api),
                        "model": args.vlm_model,
                        "coverage_iter": int(c_iter),
                    },
                )
            except Exception:
                pass
            proposal = cov_vlm.propose_views(
                report,
                current_viewspecs,
                coverage_image_paths=[Path(p) for p in (cov_render.get("view_image_paths") or [])],
                conditioning_mask_images=mask_bundle_images,
                conditioning_mask_meta=mask_bundle_meta,
                causal_obj=causal,
                expected_motion_hints=coverage_motion_hints,
                model=args.vlm_model,
                use_api=use_loop_vlm_api,
                api_provider=args.api_provider,
                api_key=args.api_key,
                base_url=args.api_base_url,
            )
            expl = proposal.get("selection_explanation") if isinstance(proposal, dict) else None
            expl_ok = isinstance(expl, dict) and (
                str(expl.get("why_this_view") or "").strip() or str(expl.get("trajectory_reasoning") or "").strip()
            )
            if not expl_ok:
                heur_expl = (
                    cov_vlm.propose_views_heuristic(
                        report,
                        current_viewspecs,
                        expected_motion_hints=coverage_motion_hints,
                    )
                    or {}
                ).get("selection_explanation")
                if isinstance(heur_expl, dict):
                    proposal["selection_explanation"] = heur_expl
            per_link_selected_views = cov_vlm.select_views_by_link(
                report,
                current_viewspecs,
                proposal,
                expected_motion_hints=coverage_motion_hints,
            )
            selected_union = []
            seen_keys = set()
            for row in per_link_selected_views.values():
                try:
                    k = (
                        int(row["azimuth_deg"]),
                        int(row["elevation_deg"]),
                        float(row["distance_scale"]),
                        int(row["fov_deg"]),
                    )
                except Exception:
                    continue
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                selected_union.append(
                    {
                        "id": f"V{len(selected_union)+1}",
                        "azimuth_deg": int(row["azimuth_deg"]),
                        "elevation_deg": int(row["elevation_deg"]),
                        "distance_scale": float(row["distance_scale"]),
                        "fov_deg": int(row["fov_deg"]),
                    }
                )
            if selected_union:
                adjusted_per_link_views = _adjust_view_distance_scale_for_motion_extent(
                    asset_ctx,
                    np.asarray(rest_center, dtype=float),
                    float(rest_radius),
                    tuple(args.resolution),
                    {str(k): dict(v) for k, v in per_link_selected_views.items()},
                    coverage_ghost_link_tf,
                    required_links,
                )
                selected_union = []
                seen_keys = set()
                for row in adjusted_per_link_views.values():
                    try:
                        k = (
                            int(row["azimuth_deg"]),
                            int(row["elevation_deg"]),
                            float(row["distance_scale"]),
                            int(row["fov_deg"]),
                        )
                    except Exception:
                        continue
                    if k in seen_keys:
                        continue
                    seen_keys.add(k)
                    selected_union.append(
                        {
                            "id": f"V{len(selected_union)+1}",
                            "azimuth_deg": int(row["azimuth_deg"]),
                            "elevation_deg": int(row["elevation_deg"]),
                            "distance_scale": float(row["distance_scale"]),
                            "fov_deg": int(row["fov_deg"]),
                        }
                    )
                coverage_selected_viewspecs = {
                    "look_at_mode": str(current_viewspecs.get("look_at_mode", "object_center")),
                    "views": selected_union,
                    "per_link_views": {str(k): _canonical_view_row(v) for k, v in adjusted_per_link_views.items()},
                }
                current_link_viewspecs = {str(k): dict(v) for k, v in adjusted_per_link_views.items()}
                _write_json(c_dir / "coverage_vlm_selected_viewspecs.json", coverage_selected_viewspecs)
                _log_per_link_viewspecs("Coverage-selected per-link motion views:", current_link_viewspecs)
            _write_json(c_dir / "coverage_vlm_proposal.json", proposal)
            coverage_history.append(
                {
                    "iter": c_iter,
                    "coverage_ok": report.get("coverage_ok"),
                    "failures": report.get("failures"),
                    "selected_views_count": len((coverage_selected_viewspecs or {}).get("views") or []),
                    "selected_links_count": len(current_link_viewspecs),
                    "need_more_views": bool(proposal.get("need_more_views", False)),
                }
            )
            coverage_report = report
            if not bool(proposal.get("need_more_views", False)):
                coverage_ok = bool(report.get("coverage_ok", False))
                if coverage_selected_viewspecs is not None:
                    current_viewspecs = lr.validate_viewspecs(coverage_selected_viewspecs)
                break
            if c_iter >= args.coverage_max_iters:
                break
            current_viewspecs = _dedupe_and_merge_views(current_viewspecs, proposal, max_total=4)
    else:
        coverage_report = {"required_links": required_links, "coverage_ok": True, "failures": []}
        coverage_ok = True
        coverage_report_path_final = None
    # Motion loop must use the chosen coverage views (if any), otherwise current coverage spec.
    if coverage_selected_viewspecs is not None:
        current_viewspecs = lr.validate_viewspecs(coverage_selected_viewspecs)
    else:
        current_viewspecs = lr.validate_viewspecs(current_viewspecs)
    if not current_link_viewspecs:
        fallback_view = dict((current_viewspecs.get("views") or [lr.DEFAULT_VIEWSPECS["views"][0]])[0])
        current_link_viewspecs = {str(ln): _canonical_view_row(fallback_view) for ln in required_links}
    _log_per_link_viewspecs("Final per-link motion views used by motion loop:", current_link_viewspecs)
    _write_json(
        loop_root / "motion_viewspecs_selected.json",
        {
            "look_at_mode": str(current_viewspecs.get("look_at_mode", "object_center")),
            "views": list(current_viewspecs.get("views") or []),
            "per_link_views": {str(k): _canonical_view_row(v) for k, v in current_link_viewspecs.items()},
        },
    )

    # Seed iterations with initial plan.
    plan_iter0 = iterations_root / "plan_iter00.json"
    _write_json(plan_iter0, initial_plan_obj)
    causal_iter0 = iterations_root / "causal_iter00.json"
    _write_json(causal_iter0, causal)

    motion_history = []
    active_motion_case_keys: set[tuple[int, str, str]] | None = None
    final_plan_iter = plan_iter0
    final_traj_npz = None
    final_traj_jsonl = None
    final_glb = None
    status = "ok"
    motion_render_backend = preferred_reference_backend

    motion_loops = args.motion_max_iters if args.enable_motion_loop else 0
    if not args.enable_motion_loop:
        exec_dir = motion_root / "iter00" / "exec"
        final_traj_npz, final_traj_jsonl, final_glb = execute_plan_once(
            py,
            asset_root,
            plan_iter0,
            exec_dir,
            args.resolution,
            args.use_glb_scene,
            skip_plan_frames=args.skip_plan_frames,
            debug_motion=args.debug_motion,
        )
        final_plan_iter = plan_iter0
    else:
        def _materialize_final_iteration(plan_obj: dict, causal_obj: dict, iter_idx: int):
            next_plan_path = iterations_root / f"plan_iter{iter_idx:02d}.json"
            next_causal_path = iterations_root / f"causal_iter{iter_idx:02d}.json"
            normalized_plan = ask_plan_mod.normalize_plan_json(plan_obj, asset_root=asset_root, vlm=causal_obj)
            _write_json(next_plan_path, normalized_plan)
            _write_json(next_causal_path, causal_obj)
            next_exec_dir = motion_root / f"iter{iter_idx:02d}" / "exec"
            traj_npz_next, traj_jsonl_next, glb_path_next = execute_plan_once(
                py,
                asset_root,
                next_plan_path,
                next_exec_dir,
                args.resolution,
                args.use_glb_scene,
                skip_plan_frames=args.skip_plan_frames,
                debug_motion=args.debug_motion,
            )
            iter_traj_npz_next = iterations_root / f"trajectory_iter{iter_idx:02d}.npz"
            iter_traj_jsonl_next = iterations_root / f"trajectory_iter{iter_idx:02d}.jsonl"
            shutil.copyfile(traj_npz_next, iter_traj_npz_next)
            shutil.copyfile(traj_jsonl_next, iter_traj_jsonl_next)
            iter_glb_next = iterations_root / f"plan_animated_iter{iter_idx:02d}.glb"
            if glb_path_next.exists():
                shutil.copyfile(glb_path_next, iter_glb_next)
            return next_plan_path, next_causal_path, iter_traj_npz_next, iter_traj_jsonl_next, iter_glb_next

        for m_iter in range(motion_loops + 1):
            cur_plan = iterations_root / f"plan_iter{m_iter:02d}.json"
            cur_causal = iterations_root / f"causal_iter{m_iter:02d}.json"
            if not cur_plan.exists():
                status = "error_missing_plan_iter"
                break
            if not cur_causal.exists():
                cur_causal = causal_json
            current_plan_obj = _load_json(cur_plan)
            m_dir = motion_root / f"iter{m_iter:02d}"
            exec_dir = m_dir / "exec"
            traj_npz, traj_jsonl, glb_path = execute_plan_once(
                py,
                asset_root,
                cur_plan,
                exec_dir,
                args.resolution,
                args.use_glb_scene,
                skip_plan_frames=args.skip_plan_frames,
                debug_motion=args.debug_motion,
            )
            # copy trajectories/glb into iterations namespace
            iter_traj_npz = iterations_root / f"trajectory_iter{m_iter:02d}.npz"
            iter_traj_jsonl = iterations_root / f"trajectory_iter{m_iter:02d}.jsonl"
            shutil.copyfile(traj_npz, iter_traj_npz)
            shutil.copyfile(traj_jsonl, iter_traj_jsonl)
            if glb_path.exists():
                shutil.copyfile(glb_path, iterations_root / f"plan_animated_iter{m_iter:02d}.glb")

            current_causal_obj = _sanitize_causal_action_targets(_load_json(cur_causal), joints)
            wheel_trace_links = wheel_diag.wheel_links_for_asset(current_plan_obj, asset_root, joints, required_links=required_links)
            wheel_transport_mode = bool(is_wheel_transport_plan(current_plan_obj) or wheel_trace_links)
            motion_context = _motion_stage_context(iter_traj_npz)
            motion_context["joint_limits"] = _joint_limits_by_name(joints)
            _write_json(m_dir / "motion_context.json", motion_context)

            # Motion diagnosis images:
            # - always sample non-overlapping head/tail frames for each non-static timeline segment
            # - single moving segment: additionally provide global START/MID/END overview grids
            # - multi moving segments: add global START/END overview grids only when the timeline samples do not already cover them
            stage_frames = motion_context.get("stage_frames") or {"start": 0, "mid": 0, "end": 0}
            # Keep a fixed camera anchor across stage renders, otherwise recentering can hide base translation.
            traj_npz_data = np.load(iter_traj_npz, allow_pickle=True)
            # Motion-loop renders must reuse the same base radius as coverage.
            anchor_radius = float(rest_radius)
            peak_motion_fi, peak_motion_link_tf, peak_motion_info = _select_peak_motion_frame(
                asset_ctx,
                iter_traj_npz,
                current_plan_obj,
                joints,
                required_links,
            )
            if isinstance(peak_motion_link_tf, dict) and current_link_viewspecs:
                adjusted_motion_link_views = _adjust_view_distance_scale_for_motion_extent(
                    asset_ctx,
                    np.asarray(rest_center, dtype=float),
                    float(rest_radius),
                    tuple(args.resolution),
                    {str(k): dict(v) for k, v in current_link_viewspecs.items()},
                    peak_motion_link_tf,
                    required_links,
                )
                if adjusted_motion_link_views:
                    current_link_viewspecs = {str(k): dict(v) for k, v in adjusted_motion_link_views.items()}
                    merged_motion_viewspecs = _viewspecs_from_per_link_views(
                        str(current_viewspecs.get("look_at_mode", "object_center")),
                        current_link_viewspecs,
                    )
                    if isinstance(merged_motion_viewspecs, dict):
                        current_viewspecs = lr.validate_viewspecs(merged_motion_viewspecs)
                    _write_json(
                        m_dir / "motion_viewspecs_peak_adjusted.json",
                        {
                            "peak_motion_frame_idx": (int(peak_motion_fi) if peak_motion_fi is not None else None),
                            "peak_motion_info": peak_motion_info if isinstance(peak_motion_info, dict) else None,
                            "look_at_mode": str(current_viewspecs.get("look_at_mode", "object_center")),
                            "views": list(current_viewspecs.get("views") or []),
                            "per_link_views": {str(k): _canonical_view_row(v) for k, v in current_link_viewspecs.items()},
                        },
                    )
                    _write_json(
                        loop_root / "motion_viewspecs_selected.json",
                        {
                            "look_at_mode": str(current_viewspecs.get("look_at_mode", "object_center")),
                            "views": list(current_viewspecs.get("views") or []),
                            "per_link_views": {str(k): _canonical_view_row(v) for k, v in current_link_viewspecs.items()},
                        },
                    )
            motion_start_path = m_dir / "motion_start_grid.png"
            motion_mid_path = m_dir / "motion_mid_grid.png"
            motion_end_path = m_dir / "motion_end_grid.png"
            for stale in (motion_start_path, motion_mid_path, motion_end_path):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            _write_json(m_dir / "motion_viewspecs.json", current_viewspecs)
            _write_json(m_dir / "motion_label_legend.json", motion_label_legend)
            fps_cur = int(current_plan_obj.get("meta", {}).get("fps", 30))
            n_frames_cur = int(np.asarray(traj_npz_data["joint_angles"]).shape[0]) if "joint_angles" in traj_npz_data else 0
            timeline_candidates = _timeline_head_tail_frames(current_plan_obj, n_frames_cur, fps_cur)
            timeline_candidates = _dedupe_frame_rows(timeline_candidates)
            timeline_frame_indices = {int(row.get("frame_idx", -1)) for row in timeline_candidates if isinstance(row, dict)}
            moving_segment_count = len(
                {
                    int(row.get("segment_index", -1))
                    for row in timeline_candidates
                    if isinstance(row, dict) and int(row.get("segment_index", -1)) >= 0
                }
            )
            stage_specs: list[tuple[str, Path, str]] = []
            if wheel_transport_mode:
                stage_specs = []
            elif moving_segment_count <= 1:
                single_stage_candidates = [
                    ("start", motion_start_path, "START\nTimeline: start\nRange: [0.00s, current action horizon]"),
                    ("mid", motion_mid_path, "MID\nTimeline: midpoint\nRange: [0.00s, current action horizon]"),
                    ("end", motion_end_path, "END\nTimeline: final state\nRange: [0.00s, current action horizon]"),
                ]
                for key, out_path, caption in single_stage_candidates:
                    fi_stage = int(stage_frames.get(key, 0 if key != "end" else max(n_frames_cur - 1, 0)))
                    fi_stage = max(0, min(max(n_frames_cur - 1, 0), fi_stage))
                    if fi_stage in timeline_frame_indices:
                        continue
                    stage_specs.append((key, out_path, caption))
            else:
                if int(stage_frames.get("start", 0)) not in timeline_frame_indices:
                    stage_specs.append(("start", motion_start_path, "START\nTimeline: start\nRange: [0.00s, current action horizon]"))
                end_frame_idx = max(0, min(max(n_frames_cur - 1, 0), int(stage_frames.get("end", max(n_frames_cur - 1, 0)))))
                if end_frame_idx not in timeline_frame_indices:
                    stage_specs.append(("end", motion_end_path, "END\nTimeline: final state\nRange: [0.00s, current action horizon]"))
            rendered_stage = set()
            for key, out_path, caption in stage_specs:
                fi_stage = int(stage_frames.get(key, 0 if key != "end" else max(n_frames_cur - 1, 0)))
                fi_stage = max(0, min(max(n_frames_cur - 1, 0), fi_stage))
                if fi_stage in rendered_stage:
                    continue
                rendered_stage.add(fi_stage)
                lr.render_motion_grid(
                    asset_ctx,
                    iter_traj_npz,
                    fi_stage,
                    current_viewspecs,
                    out_path,
                    resolution=tuple(args.resolution),
                    label_mode=lr.MOTION_LABEL_MODE_DEFAULT,
                    camera_anchor_center=rest_center,
                    camera_anchor_radius=anchor_radius,
                    label_links=coverage_render_links,
                    label_legend=motion_label_legend,
                    grid_caption=caption,
                    render_backend=motion_render_backend,
                    preferred_reference_backend=preferred_reference_backend,
                    motion_window=None,
                    trace_variant_index=m_iter,
                    animated_glb_path=glb_path,
                )
            timeline_samples = []
            plan_overall_base_axis_world = _plan_overall_base_axis_world(current_plan_obj)
            for ts in timeline_candidates:
                fi_ts = int(ts.get("frame_idx", 0))
                ts_keep = dict(ts)
                seg_motion_window = _segment_frame_window(ts_keep, n_frames_cur, fps_cur)
                seg_idx = int(ts.get("segment_index", -1))
                seg_base_axis_world = _plan_segment_base_axis_world(current_plan_obj, seg_idx) if seg_idx >= 0 else None
                ts_keep["_precomputed_motion_cues"] = _build_sample_motion_cues(
                    asset_ctx,
                    traj_npz_data,
                    fi_ts,
                    current_viewspecs,
                    rest_center,
                    anchor_radius,
                    tuple(args.resolution),
                    required_links,
                    joints,
                    motion_label_legend,
                    motion_window=seg_motion_window,
                    base_axis_world=seg_base_axis_world,
                    planned_base_axis_world=plan_overall_base_axis_world,
                )
                timeline_samples.append(ts_keep)
            timeline_dir = m_dir / "timeline_samples"
            trajectory_dir = m_dir / "trajectory"
            vlm_report_dir = m_dir / "vlm_report"
            prompt_dir = m_dir / "prompt"
            timeline_dir.mkdir(parents=True, exist_ok=True)
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            vlm_report_dir.mkdir(parents=True, exist_ok=True)
            prompt_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                m_dir / "active_motion_cases_input.json",
                {
                    "mode": ("all" if active_motion_case_keys is None else "subset"),
                    "case_keys": (
                        [
                            {"segment_index": int(k[0]), "link": str(k[1]), "joint": (str(k[2]) if str(k[2]) else None)}
                            for k in sorted(active_motion_case_keys)
                        ]
                        if active_motion_case_keys is not None
                        else None
                    ),
                },
            )
            for stale in (
                m_dir / "trajectory_summary.json",
                m_dir / "motion_vlm_report.json",
                m_dir / "motion_vlm_report_raw.json",
            ):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            for stale in timeline_dir.glob("*.png"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            for stale in trajectory_dir.glob("*.json*"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            for stale in vlm_report_dir.glob("*.json"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            for stale in prompt_dir.glob("*.json"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            for stale in prompt_dir.glob("*.txt"):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            timeline_index = []
            plan_motion_links = _collect_motion_case_links(current_plan_obj, joints)
            motion_trace_links = []
            for ln in list(plan_motion_links) + list(wheel_trace_links or []):
                s = str(ln).strip()
                if s and s not in motion_trace_links:
                    motion_trace_links.append(s)
            child_to_joint = _build_render_link_to_joint_map(joints)
            render_tasks = []
            for ts in timeline_samples:
                seg_idx = int(ts.get("segment_index", -1))
                seg_base_axis_world = _plan_segment_base_axis_world(current_plan_obj, seg_idx) if seg_idx >= 0 else None
                seg_tag = f"{seg_idx:02d}" if seg_idx >= 0 else "final"
                sample_motion_cues = ts.get("_precomputed_motion_cues")
                if not isinstance(sample_motion_cues, dict):
                    fi_ts = int(ts.get("frame_idx", 0))
                    seg_motion_window = _segment_frame_window(ts, n_frames_cur, fps_cur)
                    sample_motion_cues = _build_sample_motion_cues(
                        asset_ctx,
                        traj_npz_data,
                        fi_ts,
                        current_viewspecs,
                        rest_center,
                        anchor_radius,
                        tuple(args.resolution),
                        motion_trace_links,
                        joints,
                        motion_label_legend,
                        motion_window=seg_motion_window,
                        base_axis_world=seg_base_axis_world,
                        planned_base_axis_world=plan_overall_base_axis_world,
                    )
                sample_kind = str(ts.get("kind") or "").strip().lower()
                for ln in motion_trace_links:
                    joint_name = str(child_to_joint.get(ln) or "").strip()
                    if not _link_active_in_segment(current_plan_obj, int(ts["segment_index"]), str(ln), joint_name):
                        continue
                    render_links = _expand_motion_render_links([str(ln)], joints)
                    if not render_links:
                        render_links = [str(ln)]
                    primary_render_link = str(render_links[0])
                    case_key = _motion_case_key(int(ts["segment_index"]), str(ln), joint_name)
                    if active_motion_case_keys is not None and case_key not in active_motion_case_keys:
                        continue
                    joint_token = _safe_filename_token(joint_name or "no_joint")
                    link_token = _safe_filename_token(str(ln))
                    img_name = f"timeline_seg_{seg_tag}_{sample_kind}_{joint_token}_{link_token}_f{int(ts['frame_idx']):04d}.png"
                    img_path = timeline_dir / img_name
                    single_asset_ctx = asset_ctx
                    link_view_row = dict(
                        current_link_viewspecs.get(primary_render_link)
                        or current_link_viewspecs.get(str(ln))
                        or {}
                    )
                    if not link_view_row:
                        link_view_row = dict((current_viewspecs.get("views") or [lr.DEFAULT_VIEWSPECS["views"][0]])[0])
                    link_viewspecs = _make_single_view_spec(
                        link_view_row,
                        look_at_mode=str(current_viewspecs.get("look_at_mode", "object_center")),
                    )
                    filtered_motion_cues = _filter_motion_cues_for_link(sample_motion_cues, str(ln), joint_name)
                    render_motion_cues = dict(filtered_motion_cues) if isinstance(filtered_motion_cues, dict) else {}
                    render_joint_trends = []
                    for jt_row in (filtered_motion_cues.get("joint_trends") or []) if isinstance(filtered_motion_cues, dict) else []:
                        if not isinstance(jt_row, dict):
                            continue
                        row_copy = dict(jt_row)
                        row_copy["link"] = primary_render_link
                        render_joint_trends.append(row_copy)
                    render_motion_cues["joint_trends"] = render_joint_trends
                    image_local_motion = lr.build_local_motion_descriptor(
                        asset_ctx=asset_ctx,
                        traj_data=traj_npz_data,
                        frame_idx=int(ts["frame_idx"]),
                        viewspec=dict(link_view_row),
                        camera_anchor_center=np.asarray(rest_center, dtype=float),
                        camera_anchor_radius=float(anchor_radius),
                        resolution=tuple(args.resolution),
                        link_name=str(primary_render_link),
                        motion_window=tuple(seg_motion_window),
                        motion_cues=render_motion_cues,
                    )
                    if isinstance(image_local_motion, dict):
                        for jt_row in render_joint_trends:
                            jt_row.update(dict(image_local_motion))
                        render_motion_cues["joint_trends"] = render_joint_trends
                        render_motion_cues["local_motion"] = dict(image_local_motion)
                    render_label_legend = {
                        str(rln): str(
                            motion_label_legend.get(str(ln))
                            or motion_label_legend.get(str(rln))
                            or str(rln)
                        )
                        for rln in render_links
                    }
                    caption = (
                        f"TIMELINE {int(ts['segment_index'])}: {str(ts['segment_name'])}\n"
                        f"RANGE: [{float(ts.get('segment_t0', 0.0)):.2f}s, {float(ts.get('segment_t1', 0.0)):.2f}s]\n"
                        f"LINK: {str(ln)}  JOINT: {joint_name or 'none'}\n"
                        f"SAMPLE: t={float(ts['t_s']):.2f}s  frame={int(ts['frame_idx'])}"
                    )
                    render_tasks.append(
                        {
                            "ts": dict(ts),
                            "seg_idx": int(seg_idx),
                            "seg_tag": str(seg_tag),
                            "sample_kind": str(sample_kind),
                            "logical_link": str(ln),
                            "joint_name": (joint_name if joint_name else None),
                            "render_links": [str(x) for x in render_links],
                            "primary_render_link": primary_render_link,
                            "img_name": str(img_name),
                            "img_path": img_path,
                            "link_view_row": dict(link_view_row),
                            "link_viewspecs": dict(link_viewspecs),
                            "render_motion_cues": dict(render_motion_cues),
                            "render_label_legend": dict(render_label_legend),
                            "caption": str(caption),
                            "selected_view": _canonical_view_row(link_view_row),
                            "motion_window": _segment_frame_window(ts, n_frames_cur, fps_cur),
                        }
                    )
            head_cache_dir = motion_root / "_head_cache"
            head_cache_dir.mkdir(parents=True, exist_ok=True)
            precomputed_ref_images_by_key: dict[tuple, np.ndarray] = {}
            if (
                str(motion_render_backend).strip().lower() == "blender"
                and glb_path is not None
                and Path(glb_path).exists()
                and render_tasks
            ):
                try:
                    motion_radius_scale = float(np.clip(float(lr.MOTION_CAMERA_RADIUS_SCALE), 0.35, 1.5))
                    batch_camera_radius = max(0.05, float(anchor_radius) * motion_radius_scale)
                    batch_views = []
                    batch_keys = []
                    seen_batch_keys = set()
                    for task in render_tasks:
                        view_row = dict(task.get("selected_view") or {})
                        batch_key = (
                            int(task["ts"].get("frame_idx", 0)),
                            int(view_row.get("azimuth_deg", 0)),
                            int(view_row.get("elevation_deg", 20)),
                            round(float(view_row.get("distance_scale", 1.0)), 6),
                            int(view_row.get("fov_deg", 35)),
                        )
                        task["batch_render_key"] = batch_key
                        if batch_key in seen_batch_keys:
                            continue
                        seen_batch_keys.add(batch_key)
                        cam = lr.compute_camera_for_viewspec(
                            np.asarray(rest_center, dtype=float),
                            float(batch_camera_radius),
                            view_row,
                        )
                        eye, target, up = cam
                        batch_views.append(
                            {
                                "id": f"M{len(batch_views) + 1}",
                                "eye": np.asarray(eye, dtype=float).tolist(),
                                "target": np.asarray(target, dtype=float).tolist(),
                                "up": np.asarray(up, dtype=float).tolist(),
                                "frame_idx": int(task["ts"].get("frame_idx", 0)),
                            }
                        )
                        batch_keys.append(batch_key)
                    if batch_views:
                        batch_imgs = lr.br.render_views_from_glb(
                            glb_path,
                            batch_views,
                            tuple(int(x) for x in args.resolution),
                            fov_deg=50.0,
                            keep_animation=True,
                            fps=int(fps_cur),
                        )
                        for key, img in zip(batch_keys, batch_imgs):
                            arr = np.asarray(img, dtype=np.uint8)
                            try:
                                arr = np.asarray(lr.gop.enhance_textured_image(arr), dtype=np.uint8)
                            except Exception:
                                arr = np.asarray(img, dtype=np.uint8)
                            precomputed_ref_images_by_key[key] = np.array(arr, copy=True)
                        print(
                            f"[INFO] Batched Blender motion reference renders: "
                            f"{len(precomputed_ref_images_by_key)} unique frame/view pairs."
                        )
                except Exception as exc:
                    print(f"[WARN] Batched Blender motion reference render failed: {exc}; falling back to per-sample renders.")
                    precomputed_ref_images_by_key = {}
            for task in render_tasks:
                ts = dict(task["ts"])
                sample_kind = str(task["sample_kind"])
                img_path = Path(task["img_path"])
                single_asset_ctx = asset_ctx
                head_cache_path = None
                if sample_kind == "head":
                    head_cache_key = _motion_head_cache_key(task, tuple(args.resolution))
                    head_cache_path = head_cache_dir / f"{head_cache_key}.png"
                    if head_cache_path.exists():
                        shutil.copyfile(head_cache_path, img_path)
                        split_row = {
                            "segment_index": int(ts["segment_index"]),
                            "segment_name": str(ts["segment_name"]),
                            "phase_type": str(ts.get("phase_type") or ""),
                            "kind": str(ts["kind"]),
                            "segment_t0": float(ts.get("segment_t0", 0.0)),
                            "segment_t1": float(ts.get("segment_t1", 0.0)),
                            "t_s": float(ts["t_s"]),
                            "frame_idx": int(ts["frame_idx"]),
                            "image": str(img_path),
                            "file_name": str(task["img_name"]),
                            "link": str(task["primary_render_link"]),
                            "logical_link": str(task["logical_link"]),
                            "joint": (task["joint_name"] if task["joint_name"] else None),
                            "single_link_trace": True,
                            "referenced_links": list(task["render_links"]),
                            "motion_cues": dict(task["render_motion_cues"]),
                            "selected_view": dict(task["selected_view"]),
                        }
                        timeline_index.append(split_row)
                        continue
                precomputed_key = task.get("batch_render_key")
                precomputed_img = (
                    np.array(precomputed_ref_images_by_key.get(precomputed_key), copy=True)
                    if precomputed_key in precomputed_ref_images_by_key
                    else None
                )
                lr.render_motion_grid(
                    single_asset_ctx,
                    iter_traj_npz,
                    int(ts["frame_idx"]),
                    task["link_viewspecs"],
                    img_path,
                    resolution=tuple(args.resolution),
                    label_mode=lr.MOTION_LABEL_MODE_DEFAULT,
                    camera_anchor_center=rest_center,
                    camera_anchor_radius=anchor_radius,
                    label_links=list(task["render_links"]),
                    trace_links=list(task["render_links"]),
                    label_legend=task["render_label_legend"],
                    grid_caption=str(task["caption"]),
                    render_backend=motion_render_backend,
                    preferred_reference_backend=preferred_reference_backend,
                    motion_window=tuple(task["motion_window"]),
                    draw_optical_flow=(sample_kind == "tail"),
                    show_bbox_labels=(sample_kind == "head"),
                    motion_label_scale_override=int(lr.LABEL_SCALE_MOTION),
                    trace_variant_index=m_iter,
                    use_best_trace_candidate=False,
                    use_edge_variant_candidate=False,
                    motion_cues=dict(task["render_motion_cues"]),
                    draw_local_motion_arrows=False,
                    draw_bbox_outlines=(sample_kind == "head"),
                    animated_glb_path=glb_path,
                    precomputed_reference_images=([precomputed_img] if precomputed_img is not None else None),
                )
                split_row = {
                    "segment_index": int(ts["segment_index"]),
                    "segment_name": str(ts["segment_name"]),
                    "phase_type": str(ts.get("phase_type") or ""),
                    "kind": str(ts["kind"]),
                    "segment_t0": float(ts.get("segment_t0", 0.0)),
                    "segment_t1": float(ts.get("segment_t1", 0.0)),
                    "t_s": float(ts["t_s"]),
                    "frame_idx": int(ts["frame_idx"]),
                    "image": str(img_path),
                    "file_name": str(task["img_name"]),
                    "link": str(task["primary_render_link"]),
                    "logical_link": str(task["logical_link"]),
                    "joint": (task["joint_name"] if task["joint_name"] else None),
                    "single_link_trace": True,
                    "referenced_links": list(task["render_links"]),
                    "motion_cues": dict(task["render_motion_cues"]),
                    "selected_view": dict(task["selected_view"]),
                }
                if sample_kind == "tail":
                    try:
                        wheel_diag.postprocess_wheel_tail_image(
                            img_path,
                            asset_root=asset_root,
                            traj_npz_data=traj_npz_data,
                            traj_data={
                                "joint_names": [str(x) for x in traj_npz_data["joint_names"].tolist()],
                                "joint_angles": np.asarray(traj_npz_data["joint_angles"], dtype=float),
                                "base_translation": np.asarray(traj_npz_data["base_translation"], dtype=float),
                                "time_s": np.asarray(traj_npz_data["time_s"], dtype=float) if "time_s" in traj_npz_data else None,
                            },
                            viewspecs=task["link_viewspecs"],
                            rest_center=np.asarray(rest_center, dtype=float),
                            anchor_radius=float(anchor_radius),
                            fps=int(fps_cur),
                            row=split_row,
                            motion_label_legend=motion_label_legend,
                        )
                    except Exception:
                        pass
                elif head_cache_path is not None:
                    try:
                        shutil.copyfile(img_path, head_cache_path)
                    except Exception:
                        pass
                timeline_index.append(split_row)
            _write_json(timeline_dir / "timeline_samples_index.json", {"samples": timeline_index})
            # Per user request: keep all rendered motion-diagnostic images for the VLM input manifest.
            motion_vlm_timeline_images = [Path(row["image"]) for row in timeline_index]
            selected_abs = {str(Path(p).absolute()) for p in motion_vlm_timeline_images}
            selected_timeline_catalog = []
            for row in timeline_index:
                row_abs = str(Path(row["image"]).absolute())
                if row_abs not in selected_abs:
                    continue
                selected_timeline_catalog.append(
                    {
                        "image": row_abs,
                        "file_name": str(row.get("file_name") or Path(row_abs).name),
                        "segment_index": int(row["segment_index"]),
                        "segment_name": str(row["segment_name"]),
                        "phase_type": str(row.get("phase_type") or ""),
                        "kind": str(row["kind"]),
                        "segment_t0": float(row.get("segment_t0", 0.0)),
                        "segment_t1": float(row.get("segment_t1", 0.0)),
                        "t_s": float(row["t_s"]),
                        "frame_idx": int(row["frame_idx"]),
                        "link": str(row.get("link") or ""),
                        "joint": (str(row.get("joint")) if row.get("joint") else None),
                        "single_link_trace": bool(row.get("single_link_trace", False)),
                        "referenced_links": [str(x) for x in (row.get("referenced_links") or []) if str(x)],
                        "motion_cues": row.get("motion_cues") if isinstance(row.get("motion_cues"), dict) else None,
                        "selected_view": (dict(row.get("selected_view")) if isinstance(row.get("selected_view"), dict) else None),
                    }
                )
            _write_json(
                m_dir / "motion_vlm_input_images.json",
                {
                    "motion_render_backend": motion_render_backend,
                    "max_total_images": None,
                    "conditioning_mask_images": [str(Path(p).absolute()) for p in mask_bundle_images],
                    "conditioning_mask_meta": mask_meta_for_prompt,
                    "base_images": [
                        str(p.absolute())
                        for p in (motion_start_path, motion_mid_path, motion_end_path)
                        if p.exists()
                    ],
                    "selection_policy": "timeline samples are rendered per-link per-segment as paired head/tail images under that link's own selected coverage view; head shows the link bbox for identity grounding, tail shows optical flow plus local motion arrow plus overall motion cue when applicable",
                    "timeline_sample_images": [str(Path(p).absolute()) for p in motion_vlm_timeline_images],
                    "timeline_sample_catalog": selected_timeline_catalog,
                    "total_images": len(
                        [
                            p
                            for p in (motion_start_path, motion_mid_path, motion_end_path)
                            if p.exists()
                        ]
                    )
                    + len(motion_vlm_timeline_images),
                },
            )
            plan_summary = _build_plan_summary(current_plan_obj, current_causal_obj, joints)
            plan_summary["image_label_legend"] = motion_label_legend
            plan_summary["motion_diagnosis_mode"] = "per_link_head_tail_pair"
            plan_summary["wheel_transport_plan"] = bool(wheel_transport_mode)
            _write_json(m_dir / "plan_summary.json", plan_summary)
            trajectory_summary = _build_trajectory_summary(traj_npz_data, current_plan_obj, selected_timeline_catalog)
            trajectory_summary.update(_build_timeline_trajectory_detail(traj_npz_data, current_plan_obj, selected_timeline_catalog))
            _write_json(trajectory_dir / "trajectory_summary.json", trajectory_summary)
            try:
                shutil.copyfile(traj_jsonl, trajectory_dir / "trajectory.jsonl")
            except Exception:
                pass
            # Save the exact text prompt and image list used for motion VLM diagnosis (per iteration).
            try:
                motion_prompt_text = motion_vlm._build_prompt(
                    args.action_text,
                    motion_context,
                    plan_summary,
                    None,
                    selected_timeline_catalog,
                    scale_context,
                    trajectory_summary,
                    mask_meta_for_prompt,
                    conditioning_text_for_motion,
                )
                (prompt_dir / "motion_vlm_prompt.txt").write_text(motion_prompt_text, encoding="utf-8")
                _write_json(
                    prompt_dir / "motion_vlm_prompt_inputs.json",
                    {
                        "motion_render_backend": motion_render_backend,
                        "action_text": args.action_text,
                        "motion_context": str((m_dir / "motion_context.json").absolute()),
                        "plan_summary": str((m_dir / "plan_summary.json").absolute()),
                        "trajectory_summary": str((trajectory_dir / "trajectory_summary.json").absolute()),
                        "trajectory_jsonl": str((trajectory_dir / "trajectory.jsonl").absolute()) if (trajectory_dir / "trajectory.jsonl").exists() else None,
                        "coverage_report": None,
                        "motion_start_grid": str(motion_start_path.absolute()) if motion_start_path.exists() else None,
                        "motion_mid_grid": str(motion_mid_path.absolute()) if motion_mid_path.exists() else None,
                        "motion_end_grid": str(motion_end_path.absolute()) if motion_end_path.exists() else None,
                        "conditioning_mask_images": [str(Path(p).absolute()) for p in mask_bundle_images],
                        "conditioning_mask_meta": (str((mask_bundle_dir / "mask_bundle_meta.json").absolute()) if has_conditioning_masks else None),
                        "conditioning_text": conditioning_text_for_motion,
                        "timeline_sample_images": [str(Path(p).absolute()) for p in motion_vlm_timeline_images],
                        "timeline_sample_catalog": selected_timeline_catalog,
                        "scale_context_path": str(scale_context_path.absolute()) if scale_context_path.exists() else None,
                        "use_api": bool(use_loop_vlm_api),
                        "model": args.vlm_model,
                    },
                )
            except Exception:
                pass

            fs = motion_context.get("failure_signature") or {}
            failure_codes = fs.get("codes") or []
            # Always emit a motion diagnosis report so the loop has a consistent
            # stop/continue signal source. Real VLM API can replace this stub later.
            if active_motion_case_keys is not None and len(selected_timeline_catalog) == 0:
                motion_vlm_report_raw = {
                    "semantic_ok": True,
                    "visibility_ok": True,
                    "issues": [],
                    "param_fix_hints": [],
                    "suggested_fixes": [],
                    "affected_timeline_segments": [],
                    "affected_states": [],
                    "proposed_param_patch": None,
                    "case_rationale": "Skipped motion VLM diagnosis because no previously problematic motion cases remained.",
                    "detailed_reasoning": "Skipped motion VLM diagnosis because iter00 already cleared all motion cases and subsequent iterations are restricted to previously problematic motions only.",
                    "confidence": 1.0,
                }
            else:
                motion_vlm_report_raw = motion_vlm.diagnose_motion(
                    args.action_text,
                    motion_context,
                    plan_summary,
                    None,
                    motion_start_grid=motion_start_path,
                    motion_mid_grid=motion_mid_path,
                    motion_end_grid=motion_end_path,
                    timeline_sample_grids=motion_vlm_timeline_images,
                    timeline_sample_catalog=selected_timeline_catalog,
                    scale_context=scale_context,
                    trajectory_summary=trajectory_summary,
                    conditioning_mask_images=mask_bundle_images,
                    conditioning_mask_meta=mask_meta_for_prompt,
                    conditioning_text=conditioning_text_for_motion,
                    model=args.vlm_model,
                    use_api=use_loop_vlm_api,
                    api_provider=args.api_provider,
                    api_key=args.api_key,
                    base_url=args.api_base_url,
                    per_case_report_dir=vlm_report_dir,
                    per_case_prompt_dir=prompt_dir,
                    per_case_trajectory_dir=trajectory_dir,
                )
            motion_vlm_report_raw = motion_vlm._prune_report_fields(motion_vlm_report_raw)
            motion_vlm_report = motion_vlm_report_raw
            next_active_motion_case_keys = _collect_problematic_motion_case_keys(
                vlm_report_dir,
                selected_timeline_catalog,
                motion_vlm_report,
            )
            _write_json(vlm_report_dir / "motion_vlm_report_raw.json", motion_vlm_report_raw)
            _write_json(vlm_report_dir / "motion_vlm_report.json", motion_vlm_report)
            _write_json(
                m_dir / "active_motion_cases_output.json",
                {
                    "problematic_case_keys": [
                        {"segment_index": int(k[0]), "link": str(k[1]), "joint": (str(k[2]) if str(k[2]) else None)}
                        for k in sorted(next_active_motion_case_keys)
                    ]
                },
            )

            semantic_ok = (motion_vlm_report or {}).get("semantic_ok", True)
            visibility_ok = (motion_vlm_report or {}).get("visibility_ok", True)
            motion_history.append(
                {
                    "iter": m_iter,
                    "semantic_ok": semantic_ok,
                    "visibility_ok": visibility_ok,
                    "failure_codes": failure_codes,
                    "active_case_count_in": (None if active_motion_case_keys is None else len(active_motion_case_keys)),
                    "problematic_case_count_out": int(len(next_active_motion_case_keys)),
                }
            )
            final_plan_iter = cur_plan
            final_traj_npz = iter_traj_npz
            final_traj_jsonl = iter_traj_jsonl
            final_glb = iterations_root / f"plan_animated_iter{m_iter:02d}.glb"

            if semantic_ok and visibility_ok and len(next_active_motion_case_keys) == 0:
                status = "ok"
                break
            next_plan_base_obj = current_plan_obj
            next_causal_base_obj = current_causal_obj
            is_last_motion_iter = m_iter >= motion_loops

            patch = rule_patcher.build_rule_patch(
                next_plan_base_obj,
                motion_context,
                motion_vlm_report,
            )
            patch_path = m_dir / "plan_patch.json"
            patch_report_path = m_dir / "param_patch_apply_report.json"
            if patch is None or not (patch.get("changes") or []):
                _write_json(patch_path, {"patch_type": "param_only_v1", "changes": [], "reason_codes": failure_codes})
                _write_json(
                    patch_report_path,
                    {
                        "status": "no_patch",
                        "reason": "No parameter patch generated",
                        "used_param_fix_hints": [],
                        "ignored_param_fix_hints": [],
                        "synthesized_fallback_hints": [],
                    },
                )
                status = "exhausted_no_patch"
                break
            _write_json(patch_path, patch)
            _write_json(
                patch_report_path,
                {
                    "status": "patch_generated",
                    **(patch.get("param_patch_apply_report") or {}),
                    "patch_changes_count": len(patch.get("changes") or []),
                },
            )
            try:
                next_plan = app_patch.apply_patch_to_plan(next_plan_base_obj, patch)
                next_plan = ask_plan_mod.normalize_plan_json(next_plan, asset_root=asset_root, vlm=next_causal_base_obj)
            except Exception as exc:
                _write_json(
                    m_dir / "patch_apply_log.json",
                    {
                        "applied": False,
                        "reason": f"patch_apply_failed: {exc}",
                        "patch_path": str(patch_path),
                    },
                )
                status = "exhausted_patch_apply_failed"
                break
            if is_last_motion_iter:
                (
                    final_plan_iter,
                    _final_causal_path,
                    final_traj_npz,
                    final_traj_jsonl,
                    final_glb,
                ) = _materialize_final_iteration(next_plan, next_causal_base_obj, m_iter + 1)
                patch_log = {
                    "applied": True,
                    "next_plan": str(final_plan_iter),
                    "materialized_final_iter": True,
                }
                _write_json(m_dir / "patch_apply_log.json", patch_log)
                status = "exhausted"
                break
            next_plan_path = iterations_root / f"plan_iter{m_iter+1:02d}.json"
            _write_json(next_plan_path, next_plan)
            patch_log = {"applied": True, "next_plan": str(next_plan_path)}
            _write_json(m_dir / "patch_apply_log.json", patch_log)
            active_motion_case_keys = set(next_active_motion_case_keys)

    # Sync final artifacts back to canonical outputs/<asset>
    if final_plan_iter and Path(final_plan_iter).exists():
        shutil.copyfile(final_plan_iter, asset_out / "plan.json")
    if final_traj_npz and Path(final_traj_npz).exists():
        shutil.copyfile(final_traj_npz, asset_out / "trajectory.npz")
    if final_traj_jsonl and Path(final_traj_jsonl).exists():
        shutil.copyfile(final_traj_jsonl, asset_out / "trajectory.jsonl")
    if final_glb and Path(final_glb).exists():
        shutil.copyfile(final_glb, asset_out / "plan_animated.glb")
    # Sync final causal if a later causal mapping exists.
    final_causal_iter = None
    if args.enable_motion_loop:
        for i in range(motion_loops, -1, -1):
            cand = iterations_root / f"causal_iter{i:02d}.json"
            if cand.exists():
                final_causal_iter = cand
                break
    if final_causal_iter and Path(final_causal_iter).exists():
        shutil.copyfile(final_causal_iter, asset_out / "causal.json")

    # Clear \"this is the final output\" manifest for downstream scripts/users.
    final_manifest = {
        "asset": asset_name,
        "status": status,
        "final_outputs": {
            "causal_json": str((asset_out / "causal.json").absolute()) if (asset_out / "causal.json").exists() else (str(causal_json.absolute()) if causal_json.exists() else None),
            "plan_json": str((asset_out / "plan.json").absolute()) if (asset_out / "plan.json").exists() else None,
            "trajectory_npz": str((asset_out / "trajectory.npz").absolute()) if (asset_out / "trajectory.npz").exists() else None,
            "trajectory_jsonl": str((asset_out / "trajectory.jsonl").absolute()) if (asset_out / "trajectory.jsonl").exists() else None,
            "animated_glb": str((asset_out / "plan_animated.glb").absolute()) if (asset_out / "plan_animated.glb").exists() else None,
        },
        "loop_outputs": {
            "summary": str((iterations_root / "loop_summary.json").absolute()),
            "coverage_root": str(coverage_root.absolute()),
            "motion_root": str(motion_root.absolute()),
            "iterations_root": str(iterations_root.absolute()),
        },
        "selection_rule": "Use outputs/<asset>/{plan.json,trajectory.npz,trajectory.jsonl,plan_animated.glb} as final. loop/* are audit/debug artifacts.",
    }
    _write_json(loop_root / "final_manifest.json", final_manifest)

    summary = {
        "asset": asset_name,
        "status": status,
        "coverage_ok": coverage_ok,
        "coverage_history": coverage_history,
        "motion_history": motion_history,
        "final_plan": str(asset_out / "plan.json"),
        "final_trajectory_npz": str(asset_out / "trajectory.npz"),
        "final_trajectory_jsonl": str(asset_out / "trajectory.jsonl"),
        "final_glb": str(asset_out / "plan_animated.glb"),
        "elapsed_s": round(time.time() - t_start, 3),
        "final_manifest": str((loop_root / "final_manifest.json").absolute()),
    }
    if coverage_report is not None:
        summary["coverage_report_final"] = coverage_report
    _write_json(iterations_root / "loop_summary.json", summary)

    print(f"Loop status: {status}")
    print(f"Final plan: {asset_out / 'plan.json'}")
    print(f"Final trajectory: {asset_out / 'trajectory.npz'}")
    print(f"Final GLB: {asset_out / 'plan_animated.glb'}")


if __name__ == "__main__":
    main()
