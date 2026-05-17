#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np

import run_plan as rp


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fixed_descendant_links(link_name: str, joints: list[dict]) -> list[str]:
    fixed_children: dict[str, list[str]] = {}
    for j in joints or []:
        if str(j.get("type") or "").strip().lower() != "fixed":
            continue
        parent = str(j.get("parent") or "").strip()
        child = str(j.get("child") or "").strip()
        if not parent or not child:
            continue
        fixed_children.setdefault(parent, []).append(child)
    out = []
    stack = list(fixed_children.get(str(link_name), []))
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
    return out


def _resolve_helper_like_link(ln: str, joints: list[dict]) -> list[str]:
    link_name = str(ln or "").strip()
    if not link_name:
        return []
    is_helper_like = ("_jf_" in link_name) or link_name.endswith("_child_frame")
    if not is_helper_like:
        return [link_name]
    descendants = _fixed_descendant_links(link_name, joints)
    return descendants if descendants else [link_name]


def build_required_links(causal: dict, joints: list[dict]) -> list[str]:
    required = []
    seen = set()
    joint_to_child = {j.get("name"): j.get("child") for j in joints if j.get("name") and j.get("child")}

    def _append_link(ln) -> None:
        if not isinstance(ln, str):
            return
        for resolved in _resolve_helper_like_link(str(ln), joints):
            if resolved not in seen:
                seen.add(resolved)
                required.append(resolved)

    def _collect_action(action_obj) -> None:
        action = action_obj if isinstance(action_obj, dict) else {}
        _append_link(action.get("target_link"))
        for ln in action.get("target_links") or []:
            _append_link(ln)

    def _collect_effects(effects_obj) -> None:
        effects = effects_obj if isinstance(effects_obj, dict) else {}
        for ln in effects.get("effect_links") or []:
            _append_link(ln)
        for jt in effects.get("joint_targets") or []:
            if not isinstance(jt, dict):
                continue
            jn = jt.get("joint")
            child = joint_to_child.get(jn)
            _append_link(child)
        for rule in effects.get("coupling_rules") or []:
            if not isinstance(rule, str):
                continue
            for ln in re.findall(r"link_[A-Za-z0-9_]+", rule):
                _append_link(ln)

    causal_obj = causal.get("causal") or {}
    _collect_action(causal_obj.get("action"))
    _collect_effects(causal_obj.get("effects"))
    for seg in causal.get("causal_segments") or []:
        if not isinstance(seg, dict):
            continue
        _collect_action(seg.get("action"))
        _collect_effects(seg.get("effects"))

    if required:
        return required
    # fallback: all movable child links
    for j in joints:
        child = j.get("child")
        if isinstance(child, str) and child not in seen:
            seen.add(child)
            required.append(child)
    return required


def verify_coverage_arrays(link_names, view_ids, visible_px, visible_ratio, required_links, small_links=None):
    small_links = set(small_links or [])
    link_names = [str(x) for x in link_names]
    view_ids = [str(x) for x in view_ids]
    visible_px = np.asarray(visible_px)
    visible_ratio = np.asarray(visible_ratio)
    idx = {ln: i for i, ln in enumerate(link_names)}
    per_link = {}
    failures = []
    for ln in required_links:
        if ln not in idx:
            failures.append({"code": "LINK_NOT_RENDERED", "link": ln})
            per_link[ln] = {
                "best_view": None,
                "best_visible_ratio": 0.0,
                "best_visible_px": 0,
                "visible_in_views": [],
            }
            continue
        j = idx[ln]
        px_col = visible_px[:, j] if visible_px.size else np.array([], dtype=int)
        rt_col = visible_ratio[:, j] if visible_ratio.size else np.array([], dtype=float)
        if px_col.size == 0:
            best_i = 0
            best_px = 0
            best_ratio = 0.0
        else:
            best_i = int(np.argmax(px_col))
            best_px = int(px_col[best_i])
            best_ratio = float(rt_col[best_i])
        visible_views = []
        if px_col.size:
            for vi, px in enumerate(px_col):
                if int(px) > 0:
                    visible_views.append(view_ids[vi])
        thr_ratio = 0.005 if ln in small_links else 0.01
        thr_px = 200 if ln in small_links else 400
        per_link[ln] = {
            "best_view": view_ids[best_i] if view_ids and px_col.size else None,
            "best_visible_ratio": best_ratio,
            "best_visible_px": best_px,
            "visible_in_views": visible_views,
            "visible_ratio_by_view": {
                str(view_ids[vi]): float(rt_col[vi])
                for vi in range(min(len(view_ids), int(rt_col.shape[0]) if hasattr(rt_col, "shape") else len(view_ids)))
            },
            "visible_px_by_view": {
                str(view_ids[vi]): int(px_col[vi])
                for vi in range(min(len(view_ids), int(px_col.shape[0]) if hasattr(px_col, "shape") else len(view_ids)))
            },
        }
        if not (best_ratio >= thr_ratio or best_px >= thr_px):
            failures.append(
                {
                    "code": "OCCLUSION_HIGH",
                    "link": ln,
                    "best_view": per_link[ln]["best_view"],
                    "best_visible_ratio": best_ratio,
                    "best_visible_px": best_px,
                }
            )
    return {
        "required_links": required_links,
        "thresholds": {"visible_ratio_min": 0.01, "visible_px_min": 400},
        "per_link": per_link,
        "coverage_ok": len(failures) == 0,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify coverage from coverage_masks.npz and causal/URDF")
    parser.add_argument("--asset_root", required=True)
    parser.add_argument("--causal_json", required=True)
    parser.add_argument("--coverage_masks", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args()

    asset_root = Path(args.asset_root)
    urdf_path = next(asset_root.rglob("*.urdf"), None)
    if urdf_path is None:
        raise SystemExit(f"No URDF under {asset_root}")
    _, joints = rp.parse_urdf(urdf_path)
    causal = load_json(Path(args.causal_json))
    required_links = build_required_links(causal, joints)

    data = np.load(args.coverage_masks, allow_pickle=True)
    report = verify_coverage_arrays(
        data["link_names"].tolist(),
        data["view_ids"].tolist(),
        data["visible_px"],
        data["visible_ratio"],
        required_links,
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
