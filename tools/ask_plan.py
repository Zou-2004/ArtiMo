#!/usr/bin/env python3
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from api_client_utils import generate_content_text


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_action(action_obj) -> dict:
    action = dict(action_obj) if isinstance(action_obj, dict) else {}
    target_link = action.get("target_link")
    if isinstance(target_link, str):
        target_link = target_link.strip()
    else:
        target_link = ""
    target_links = []
    if isinstance(action.get("target_links"), list):
        for x in action.get("target_links") or []:
            if isinstance(x, str) and x.strip() and x.strip() not in target_links:
                target_links.append(x.strip())
    if target_link and target_link not in target_links:
        target_links = [target_link] + target_links
    if not target_link and target_links:
        target_link = target_links[0]
    if target_link:
        action["target_link"] = target_link
    else:
        action.pop("target_link", None)
    if target_links:
        action["target_links"] = target_links
    else:
        action.pop("target_links", None)
    return action


def _normalize_effects(effects_obj) -> dict:
    effects = dict(effects_obj) if isinstance(effects_obj, dict) else {}
    out = {"joint_targets": [], "modes": [], "coupling_rules": []}
    seen_joint = {}
    for jt in effects.get("joint_targets") or []:
        if not isinstance(jt, dict):
            continue
        jn = str(jt.get("joint") or "").strip()
        if not jn:
            continue
        if jn not in seen_joint:
            item = {"joint": jn}
            if "to" in jt:
                item["to"] = jt.get("to")
            out["joint_targets"].append(item)
            seen_joint[jn] = item
        elif "to" in jt and "to" not in seen_joint[jn]:
            seen_joint[jn]["to"] = jt.get("to")
    for m in effects.get("modes") or []:
        if isinstance(m, dict):
            out["modes"].append(m)
    for r in effects.get("coupling_rules") or []:
        if isinstance(r, str) and r.strip():
            out["coupling_rules"].append(r.strip())
    return out


def _normalize_causal_segments(vlm: dict) -> dict:
    segs_in = vlm.get("causal_segments")
    segs_out = None
    if isinstance(segs_in, list):
        seg_list = []
        for i, seg in enumerate(segs_in):
            if not isinstance(seg, dict):
                continue
            so = dict(seg)
            so["segment_id"] = str(seg.get("segment_id") or f"S{i+1}")
            time_hint = seg.get("time_hint") if isinstance(seg.get("time_hint"), dict) else {}
            th = dict(time_hint)
            try:
                th["order_index"] = int(time_hint.get("order_index", i))
            except Exception:
                th["order_index"] = i
            so["time_hint"] = th
            so["action"] = _normalize_action(seg.get("action"))
            so["effects"] = _normalize_effects(seg.get("effects"))
            seg_list.append(so)
        if seg_list:
            segs_out = seg_list
    vlm["causal_segments"] = segs_out
    return vlm


def _ensure_top_level_causal(vlm: dict) -> dict:
    causal = dict(vlm.get("causal")) if isinstance(vlm.get("causal"), dict) else {}
    has_causal = causal.get("has_causal")
    if isinstance(has_causal, bool):
        has_causal_flag = has_causal
    else:
        has_causal_flag = None
    if has_causal_flag is False:
        causal["has_causal"] = False
        causal["action"] = None
        causal["effects"] = None
    else:
        causal["action"] = _normalize_action(causal.get("action"))
        causal["effects"] = _normalize_effects(causal.get("effects"))

    segs = vlm.get("causal_segments") or []
    if segs:
        # Segments are authoritative when present; rebuild top-level summary from
        # segments to avoid duplicated/conflicting causal.action/effects.
        causal["has_causal"] = True
        if isinstance(segs[0], dict):
            causal["action"] = dict((segs[0].get("action") or {}))
        else:
            causal["action"] = None
        merged = _normalize_effects({})
        for seg in segs:
            if not isinstance(seg, dict):
                continue
            seg_eff = _normalize_effects(seg.get("effects"))
            merged["joint_targets"].extend(seg_eff.get("joint_targets") or [])
            merged["modes"].extend(seg_eff.get("modes") or [])
            merged["coupling_rules"].extend(seg_eff.get("coupling_rules") or [])
        causal["effects"] = _normalize_effects(merged)
    elif not isinstance(causal.get("has_causal"), bool):
        causal["has_causal"] = bool(causal.get("action")) or bool((causal.get("effects") or {}).get("joint_targets"))

    vlm["causal"] = causal
    return vlm


def normalize_vlm_json(vlm: dict) -> dict:
    if not isinstance(vlm, dict):
        return vlm
    vlm = _normalize_causal_segments(vlm)
    vlm = _ensure_top_level_causal(vlm)

    if ((vlm.get("causal") or {}).get("has_causal") is False):
        return vlm

    effects = (vlm.get("causal") or {}).get("effects") or {}
    joint_targets = effects.get("joint_targets") or []
    coupling_rules = effects.get("coupling_rules") or []

    jt_by_name = {}
    for jt in joint_targets:
        jn = jt.get("joint")
        if jn:
            jt_by_name[jn] = jt

    for rule in coupling_rules:
        if not isinstance(rule, str) or "lower_limit" not in rule:
            continue
        joints_in_rule = re.findall(r"joint_[A-Za-z0-9_]+", rule)
        for jn in joints_in_rule:
            jt = jt_by_name.get(jn)
            if jt is None:
                jt = {"joint": jn, "to": "1.0*lower_limit"}
                joint_targets.append(jt)
                jt_by_name[jn] = jt
                continue
            to_val = str(jt.get("to") or "")
            if "upper_limit" in to_val and ("0*upper_limit" in to_val or to_val.strip().startswith("0")):
                jt["to"] = "1.0*lower_limit"

    effects["joint_targets"] = joint_targets
    (vlm.get("causal") or {})["effects"] = effects
    return vlm

def normalize_plan_json(plan: dict, asset_root: Path | None = None, vlm: dict | None = None) -> dict:
    def _mode(ctrl):
        if not isinstance(ctrl, dict):
            return ""
        return str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()

    def _expand_grouped_controls(timeline):
        out_timeline = []
        for seg in (timeline or []):
            if not isinstance(seg, dict):
                continue
            controls = []
            for ctrl in (seg.get("controls") or []):
                if not isinstance(ctrl, dict):
                    continue
                mode = _mode(ctrl)
                grouped = [str(x).strip() for x in (ctrl.get("joints") or []) if str(x).strip()]
                single = str(ctrl.get("joint") or "").strip()
                if single:
                    grouped = [single]
                if mode in {"joint_velocity", "joint_position", "hold_position", "spring_return", "release_joint"} and grouped:
                    for jn in grouped:
                        cnew = dict(ctrl)
                        cnew.pop("joints", None)
                        cnew["joint"] = jn
                        controls.append(cnew)
                else:
                    controls.append(dict(ctrl))
            seg_out = dict(seg)
            seg_out["controls"] = controls
            out_timeline.append(seg_out)
        return out_timeline

    def _joint_from_ctrl(ctrl):
        if not isinstance(ctrl, dict):
            return None
        jn = ctrl.get("joint")
        if isinstance(jn, str) and jn.strip():
            return jn.strip()
        return None

    def _dedupe_control_release_segments(timeline):
        # Remove repeated spring-return releases for the same control joint unless
        # a fresh actuation of that joint appears in between.
        last_actuation_id = {}
        seen_release_after_act = {}
        out_timeline = []
        for seg in (timeline or []):
            if not isinstance(seg, dict):
                continue
            phase = str(seg.get("phase_type") or "").strip().lower()
            controls = list(seg.get("controls") or [])
            # Mark fresh control actuation cycle.
            for ctrl in controls:
                if _mode(ctrl) != "joint_position":
                    continue
                jn = _joint_from_ctrl(ctrl)
                if not jn:
                    continue
                # Count any explicit control actuation as a new cycle for that joint.
                # This is intentionally permissive to avoid deleting valid re-press patterns.
                if phase == "control_actuation" or phase.startswith("control_"):
                    last_actuation_id[jn] = int(last_actuation_id.get(jn, 0)) + 1
                    seen_release_after_act[(jn, last_actuation_id[jn])] = False

            new_controls = []
            for ctrl in controls:
                mode = _mode(ctrl)
                jn = _joint_from_ctrl(ctrl)
                if mode == "spring_return" and jn and phase == "control_release":
                    act_id = int(last_actuation_id.get(jn, 0))
                    key = (jn, act_id)
                    if seen_release_after_act.get(key, False):
                        # Duplicate release for same cycle -> drop.
                        continue
                    seen_release_after_act[key] = True
                new_controls.append(ctrl)

            if not new_controls:
                # Drop empty segment generated by dedupe.
                continue
            seg_out = dict(seg)
            seg_out["controls"] = new_controls
            out_timeline.append(seg_out)
        return out_timeline

    def _load_joint_limits(root: Path | None) -> dict[str, dict[str, float | None]]:
        if root is None:
            return {}
        try:
            urdf = next(Path(root).rglob("*.urdf"))
        except Exception:
            return {}
        out: dict[str, dict[str, float | None]] = {}
        try:
            tree = ET.parse(urdf)
        except Exception:
            return out
        for joint in tree.getroot().findall("joint"):
            name = str(joint.get("name") or "").strip()
            if not name:
                continue
            limit = joint.find("limit")
            if limit is None:
                continue
            row: dict[str, float | None] = {"lower": None, "upper": None}
            for key in ("lower", "upper"):
                raw = limit.get(key)
                if raw is None:
                    continue
                try:
                    row[key] = float(raw)
                except Exception:
                    pass
            out[name] = row
        return out

    def _clamp_joint_position_targets(timeline):
        joint_limits = _load_joint_limits(asset_root)
        if not joint_limits:
            return timeline
        for seg in timeline:
            if not isinstance(seg, dict):
                continue
            for ctrl in seg.get("controls") or []:
                if not isinstance(ctrl, dict) or _mode(ctrl) != "joint_position":
                    continue
                jn = _joint_from_ctrl(ctrl)
                if not jn or jn not in joint_limits or ctrl.get("q_target_rad") is None:
                    continue
                try:
                    q = float(ctrl.get("q_target_rad"))
                except Exception:
                    continue
                lim = joint_limits[jn]
                lo = lim.get("lower")
                hi = lim.get("upper")
                q2 = q
                if lo is not None and q2 < float(lo):
                    q2 = float(lo)
                if hi is not None and q2 > float(hi):
                    q2 = float(hi)
                if abs(q2 - q) > 1.0e-9:
                    ctrl["q_target_rad"] = q2
        return timeline

    meta = plan.get("meta") or {}
    duration_s = float(meta.get("duration_s", 4.0))
    max_t1 = duration_s
    timeline = _expand_grouped_controls(plan.get("timeline") or [])
    timeline = _dedupe_control_release_segments(timeline)
    timeline = _clamp_joint_position_targets(timeline)
    plan["timeline"] = timeline
    for seg in timeline:
        try:
            max_t1 = max(max_t1, float(seg.get("t1", 0.0)))
        except Exception:
            continue
    if max_t1 > duration_s:
        meta["duration_s"] = float(max_t1)
    plan["meta"] = meta
    plan.pop("nl_parse", None)

    return plan


def _extract_json_text(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s2 = "\n".join(lines).strip()
        try:
            json.loads(s2)
            return s2
        except Exception:
            pass
    i = s.find("{")
    j = s.rfind("}")
    if i >= 0 and j > i:
        cand = s[i : j + 1]
        try:
            json.loads(cand)
            return cand
        except Exception:
            return None
    return None


def main():
    parser = argparse.ArgumentParser(description="Call LLM plan compiler using VLM output + plan prompt.")
    parser.add_argument("--asset", required=True, help="Asset name")
    parser.add_argument("--asset_root", default=None, help="Optional asset root path; reserved for asset-aware normalization")
    parser.add_argument(
        "--outputs_root",
        default="/home/zouchunyu/casual_articulation/outputs",
        help="Root outputs directory",
    )
    parser.add_argument("--model", default="gpt-5.2", help="Model name")
    parser.add_argument("--api_key", default=None, help="Optional API key override (otherwise read from env)")
    parser.add_argument("--api_provider", default="auto", choices=["auto", "openai", "gemini"])
    parser.add_argument("--api_base_url", default=None, help="Optional API base URL override")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    prompt_path = outputs_root / args.asset / "prompts" / "llm_plan_prompt.md"
    vlm_json_path = outputs_root / args.asset / "output.json"
    if not prompt_path.exists():
        raise SystemExit(f"Missing plan prompt: {prompt_path}")
    if not vlm_json_path.exists():
        raise SystemExit(f"Missing VLM output JSON: {vlm_json_path}")

    prompt = load_text(prompt_path)
    vlm_text = load_text(vlm_json_path)
    vlm_parsed = None
    try:
        vlm_parsed = json.loads(vlm_text)
        vlm_parsed = normalize_vlm_json(vlm_parsed)
        vlm_json = json.dumps(vlm_parsed, ensure_ascii=False, indent=2)
    except Exception:
        vlm_json = vlm_text

    merged_prompt = prompt.replace("<PASTE VLM JSON HERE>", vlm_json)
    filled_prompt_path = outputs_root / args.asset / "prompts" / "llm_plan_prompt_filled.md"
    filled_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    filled_prompt_path.write_text(merged_prompt, encoding="utf-8")
    print(f"[INFO] Wrote filled LLM prompt to {filled_prompt_path}")

    message, cfg = generate_content_text(
        model=args.model,
        prompt=merged_prompt,
        image_paths=None,
        provider=args.api_provider,
        api_key=args.api_key,
        base_url=args.api_base_url,
    )
    print(f"[INFO] API provider={cfg['provider']} base_url={cfg['base_url']}")
    message = message or "{}"
    print(message)

    out_dir = outputs_root / args.asset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "plan.json"
    cleaned = _extract_json_text(message)
    if cleaned is not None:
        parsed = json.loads(cleaned)
        asset_root = Path(args.asset_root).resolve() if args.asset_root else None
        parsed = normalize_plan_json(parsed, asset_root=asset_root, vlm=vlm_parsed)
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_path.write_text(message, encoding="utf-8")


if __name__ == "__main__":
    main()
