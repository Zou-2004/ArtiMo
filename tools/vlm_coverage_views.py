#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from api_client_utils import generate_content_text
import loop_render as lr


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty response")
    try:
        return json.loads(text)
    except Exception:
        pass
    # fenced JSON fallback
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("no JSON object found")


def _snap_view_dict(p: dict) -> dict | None:
    if not isinstance(p, dict):
        return None
    try:
        az = int(p.get("azimuth_deg"))
        el = int(p.get("elevation_deg"))
        ds = float(p.get("distance_scale"))
        fov = int(p.get("fov_deg"))
    except Exception:
        return None
    az = min(lr.AZIMUTH_SET, key=lambda x: abs(((x - az + 180) % 360) - 180))
    el = min(lr.ELEVATION_SET, key=lambda x: abs(x - el))
    ds = min(lr.DISTANCE_SCALE_SET, key=lambda x: abs(x - ds))
    fov = min(lr.FOV_SET, key=lambda x: abs(x - fov))
    return {"azimuth_deg": az, "elevation_deg": el, "distance_scale": float(ds), "fov_deg": fov}


def _sanitize_selection_explanation(obj) -> dict:
    if not isinstance(obj, dict):
        return {"why_this_view": "", "trajectory_reasoning": "", "link_checks": []}
    checks_raw = obj.get("link_checks")
    if not isinstance(checks_raw, list):
        checks_raw = []
    checks = []
    for it in checks_raw[:32]:
        if not isinstance(it, dict):
            continue
        checks.append(
            {
                "link": str(it.get("link") or ""),
                "expected_motion": str(it.get("expected_motion") or "unknown"),
                "bbox_visible": bool(it.get("bbox_visible", False)),
                "label_readable": bool(it.get("label_readable", False)),
                "trajectory_observability": str(it.get("trajectory_observability") or "unknown"),
                "evidence": str(it.get("evidence") or ""),
                "selection_reason": str(it.get("selection_reason") or ""),
                "rejection_reason": str(it.get("rejection_reason") or ""),
                "distance_reason": str(it.get("distance_reason") or ""),
            }
        )
    return {
        "why_this_view": str(obj.get("why_this_view") or ""),
        "trajectory_reasoning": str(obj.get("trajectory_reasoning") or ""),
        "link_checks": checks,
    }


def _sanitize_proposal(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("proposal must be object")
    required_links = [str(x) for x in (obj.get("_required_links") or []) if str(x)]

    def _sanitize_selected_views_by_link(raw) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not isinstance(raw, dict):
            return out
        for lk, row in raw.items():
            link_name = str(lk or "").strip()
            if not link_name:
                continue
            cand = _snap_view_dict(row)
            if cand is None:
                continue
            out[link_name] = cand
        if required_links:
            out = {ln: out[ln] for ln in required_links if ln in out}
        return out

    selected_views_by_link = _sanitize_selected_views_by_link(obj.get("selected_views_by_link"))
    raw_selected = obj.get("selected_views") or []
    if not isinstance(raw_selected, list):
        raw_selected = []
    selected = []
    for p in raw_selected[:1]:
        cand = _snap_view_dict(p)
        if cand is not None and cand not in selected:
            selected.append(cand)
    if not selected and selected_views_by_link:
        for cand in selected_views_by_link.values():
            if cand not in selected:
                selected.append(cand)
            if len(selected) >= 4:
                break
    raw_proposed = obj.get("proposed_views") or []
    if not isinstance(raw_proposed, list):
        raw_proposed = []
    proposed = []
    for p in raw_proposed[:4]:
        cand = _snap_view_dict(p)
        if cand is not None and cand not in proposed:
            proposed.append(cand)
    need_more_views = bool(obj.get("need_more_views", False)) and len(proposed) > 0
    return {
        "selected_views": selected[:1],
        "selected_views_by_link": selected_views_by_link,
        "need_more_views": need_more_views,
        "proposed_views": proposed[:4],
        "reason": str(obj.get("reason") or ""),
        "selection_explanation": _sanitize_selection_explanation(obj.get("selection_explanation")),
    }


def _expected_motion_hint_map(expected_motion_hints: dict | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not isinstance(expected_motion_hints, dict):
        return out
    rows = expected_motion_hints.get("required_links_motion_hints")
    if not isinstance(rows, list):
        return out
    for it in rows:
        if not isinstance(it, dict):
            continue
        ln = str(it.get("link") or "").strip()
        if not ln:
            continue
        motion = str(it.get("expected_motion") or "unknown").strip().lower()
        role = str(it.get("role") or "").strip().lower()
        w = 1.0
        if motion in {"rotation", "translation", "mixed"}:
            w = 3.0
        elif motion == "static":
            w = 0.5
        if role == "effect":
            w += 0.75
        elif role == "target":
            w += 0.25
        if bool(it.get("interior_candidate")):
            w += 1.25
        if isinstance(it.get("plan_segments"), list):
            w += min(0.5, 0.15 * float(len(it.get("plan_segments"))))
        prev = out.get(ln) or {}
        prev_w = float(prev.get("weight", 0.0))
        row = dict(it)
        row["weight"] = float(w)
        if w > prev_w:
            out[ln] = row
    return out


def _view_metric(per_link_info: dict, view_id: str) -> tuple[int, float]:
    px = int(((per_link_info.get("visible_px_by_view") or {}).get(view_id, 0)) or 0)
    ratio = float(((per_link_info.get("visible_ratio_by_view") or {}).get(view_id, 0.0)) or 0.0)
    return px, ratio


def _view_selection_score(
    view_id: str,
    required_links: list[str],
    per_link: dict,
    hint_map: dict[str, dict],
) -> float:
    present: set[str] = set()
    total = 0.0
    has_target = False
    has_effect = False
    for ln in required_links:
        info = per_link.get(ln) if isinstance(per_link, dict) else None
        if not isinstance(info, dict):
            continue
        px, ratio = _view_metric(info, view_id)
        if px <= 0 and ratio <= 0.0:
            continue
        present.add(ln)
        hint = hint_map.get(ln) or {}
        w = float(hint.get("weight", 1.0))
        role = str(hint.get("role") or "").strip().lower()
        if role == "target":
            has_target = True
        if role == "effect":
            has_effect = True
        total += w * (min(ratio / 0.02, 2.5) + min(float(px) / 1500.0, 2.0))
        if bool(hint.get("interior_candidate")):
            total += 0.35 * w
    total += 0.85 * float(len(present))
    if required_links and len(present) == len(required_links):
        total += 2.5
    if has_target and has_effect:
        total += 1.25
    for ln in list(present):
        hint = hint_map.get(ln) or {}
        if not bool(hint.get("interior_candidate")):
            continue
        exposure_links = [str(x) for x in (hint.get("exposure_links") or []) if str(x)]
        if any(exp in present for exp in exposure_links):
            total += 1.75 * float(hint.get("weight", 1.0))
    return total


def _selected_view_score(
    proposal: dict,
    current_viewspecs: dict,
    coverage_report: dict,
    expected_motion_hints: dict | None = None,
) -> float:
    if not isinstance(proposal, dict):
        return -1.0
    selected = proposal.get("selected_views") or []
    if not isinstance(selected, list) or not selected:
        return -1.0
    snapped = _snap_view_dict(selected[0])
    if not isinstance(snapped, dict):
        return -1.0
    current = lr.validate_viewspecs(current_viewspecs)
    view_id = None
    for v in current.get("views", []):
        if (
            int(v.get("azimuth_deg")) == int(snapped.get("azimuth_deg"))
            and int(v.get("elevation_deg")) == int(snapped.get("elevation_deg"))
            and float(v.get("distance_scale")) == float(snapped.get("distance_scale"))
            and int(v.get("fov_deg")) == int(snapped.get("fov_deg"))
        ):
            view_id = str(v.get("id") or "")
            break
    if not view_id:
        return -1.0
    per_link = coverage_report.get("per_link") or {}
    required_links = [str(x) for x in (coverage_report.get("required_links") or []) if str(x)]
    hint_map = _expected_motion_hint_map(expected_motion_hints)
    return _view_selection_score(view_id, required_links, per_link, hint_map)


def _select_views_by_link_heuristic(
    coverage_report: dict,
    current_viewspecs: dict,
    expected_motion_hints: dict | None = None,
) -> dict[str, dict]:
    cur = lr.validate_viewspecs(current_viewspecs)
    views = list(cur.get("views") or [])
    if not views:
        return {}
    view_by_id = {str(v.get("id") or ""): v for v in views}
    per_link = coverage_report.get("per_link") or {}
    required_links = [str(x) for x in (coverage_report.get("required_links") or []) if str(x)]
    hint_map = _expected_motion_hint_map(expected_motion_hints)
    out: dict[str, dict] = {}
    for ln in required_links:
        info = per_link.get(ln) if isinstance(per_link, dict) else None
        best_vid = ""
        if isinstance(info, dict):
            best_vid = str(info.get("best_view") or "")
        best_view = view_by_id.get(best_vid)
        if best_view is None:
            best_score = None
            best_view = None
            for vid, view in view_by_id.items():
                score = _view_selection_score(vid, [ln], per_link, hint_map)
                if best_score is None or score > best_score:
                    best_score = score
                    best_view = view
        if best_view is None:
            best_view = views[0]
        out[str(ln)] = {
            "azimuth_deg": int(best_view["azimuth_deg"]),
            "elevation_deg": int(best_view["elevation_deg"]),
            "distance_scale": float(best_view["distance_scale"]),
            "fov_deg": int(best_view["fov_deg"]),
        }
    return out


def _select_views_heuristic(
    coverage_report: dict,
    current_viewspecs: dict,
    max_views: int = 1,
    expected_motion_hints: dict | None = None,
) -> list[dict]:
    cur = lr.validate_viewspecs(current_viewspecs)
    views = list(cur.get("views") or [])
    if not views:
        return []
    per_link = coverage_report.get("per_link") or {}
    required_links = [str(x) for x in (coverage_report.get("required_links") or []) if str(x)]
    view_ids = [str(v.get("id") or "") for v in views]
    view_by_id = {str(v.get("id") or ""): v for v in views}
    covered_by_view = {vid: set() for vid in view_ids}
    hint_map = _expected_motion_hint_map(expected_motion_hints)
    best_view_score = {vid: 0.0 for vid in view_ids}
    for vid in view_ids:
        best_view_score[vid] = _view_selection_score(vid, required_links, per_link, hint_map)
    for ln in required_links:
        info = per_link.get(ln) if isinstance(per_link, dict) else None
        if not isinstance(info, dict):
            continue
        for vid in info.get("visible_in_views") or []:
            s = str(vid or "")
            if s in covered_by_view:
                covered_by_view[s].add(ln)
    # If coverage is already sufficient, prioritize a view that best reveals
    # expected moving links (instead of always picking the first rendered view).
    if required_links and int(max_views) == 1 and any(v > 0 for v in best_view_score.values()):
        ranked = sorted(
            view_ids,
            key=lambda vid: (-float(best_view_score.get(vid, 0.0)), view_ids.index(vid)),
        )
        selected_ids = [ranked[0]]
    else:
        selected_ids = []
    uncovered = set(required_links)
    # Greedy set-cover over currently rendered views.
    while uncovered and len(selected_ids) < int(max_views):
        best_id = None
        best_gain = -1.0
        best_weighted_gain = -1.0
        best_idx = 10**9
        for idx, vid in enumerate(view_ids):
            if vid in selected_ids:
                continue
            cov = covered_by_view.get(vid, set()).intersection(uncovered)
            gain = float(len(cov))
            weighted_gain = float(sum(float((hint_map.get(ln) or {}).get("weight", 1.0)) for ln in cov))
            if (
                weighted_gain > best_weighted_gain
                or (weighted_gain == best_weighted_gain and gain > best_gain)
                or (weighted_gain == best_weighted_gain and gain == best_gain and idx < best_idx)
            ):
                best_weighted_gain = weighted_gain
                best_gain = gain
                best_id = vid
                best_idx = idx
        if best_id is None or best_gain <= 0:
            break
        selected_ids.append(best_id)
        uncovered -= covered_by_view.get(best_id, set())
    # Fallback by per-link best_view hints.
    if len(selected_ids) < int(max_views):
        for ln in required_links:
            info = per_link.get(ln) if isinstance(per_link, dict) else None
            if not isinstance(info, dict):
                continue
            bv = str(info.get("best_view") or "")
            if bv and bv in view_by_id and bv not in selected_ids:
                selected_ids.append(bv)
            if len(selected_ids) >= int(max_views):
                break
    if not selected_ids:
        selected_ids = [view_ids[0]]
    out = []
    for vid in selected_ids[: int(max_views)]:
        v = view_by_id.get(vid)
        if not isinstance(v, dict):
            continue
        out.append(
            {
                "azimuth_deg": int(v["azimuth_deg"]),
                "elevation_deg": int(v["elevation_deg"]),
                "distance_scale": float(v["distance_scale"]),
                "fov_deg": int(v["fov_deg"]),
            }
        )
    return out


def propose_views_heuristic(
    coverage_report: dict,
    current_viewspecs: dict,
    expected_motion_hints: dict | None = None,
) -> dict:
    current = lr.validate_viewspecs(current_viewspecs)
    selected_views_by_link = _select_views_by_link_heuristic(
        coverage_report,
        current,
        expected_motion_hints=expected_motion_hints,
    )
    selected_views = _select_views_heuristic(
        coverage_report,
        current,
        max_views=1,
        expected_motion_hints=expected_motion_hints,
    )
    failures = coverage_report.get("failures") or []
    if not failures:
        required_links = [str(x) for x in (coverage_report.get("required_links") or []) if str(x)]
        return {
            "selected_views": selected_views,
            "selected_views_by_link": selected_views_by_link,
            "need_more_views": False,
            "proposed_views": [],
            "reason": "coverage ok; selected best current views",
            "selection_explanation": {
                "why_this_view": "Selected one best current view for each required link.",
                "trajectory_reasoning": "Prefer a per-link view where that link is visible, label-grounded, and favorable for later motion interpretation.",
                "link_checks": [
                    {
                        "link": ln,
                        "expected_motion": "unknown",
                        "bbox_visible": True,
                        "label_readable": True,
                        "trajectory_observability": "high",
                        "evidence": "Heuristic fallback: best available current view.",
                        "selection_reason": "This current view gives the clearest available bbox/label evidence for the link.",
                        "rejection_reason": "Other current views are weaker overall for this link or for later motion readability.",
                        "distance_reason": "This distance_scale is the best available current framing without being obviously too tight or too far.",
                    }
                    for ln in required_links
                ],
            },
        }
    proposed = []
    for f in failures[:4]:
        best_view_id = f.get("best_view")
        base = None
        for v in current.get("views", []):
            if v.get("id") == best_view_id:
                base = v
                break
        if base is None:
            base = current["views"][0]
        candidates = [
            {"azimuth_deg": (int(base["azimuth_deg"]) + 90) % 360, "elevation_deg": 30, "distance_scale": 1.2, "fov_deg": 35},
            {"azimuth_deg": (int(base["azimuth_deg"]) + 180) % 360, "elevation_deg": 45, "distance_scale": 1.0, "fov_deg": 25},
        ]
        for c in candidates:
            c["azimuth_deg"] = min(lr.AZIMUTH_SET, key=lambda x: abs(((x - c["azimuth_deg"] + 180) % 360) - 180))
            c["elevation_deg"] = min(lr.ELEVATION_SET, key=lambda x: abs(x - c["elevation_deg"]))
            c["distance_scale"] = min(lr.DISTANCE_SCALE_SET, key=lambda x: abs(x - c["distance_scale"]))
            c["fov_deg"] = min(lr.FOV_SET, key=lambda x: abs(x - c["fov_deg"]))
            if c not in proposed:
                proposed.append(c)
            if len(proposed) >= 4:
                break
        if len(proposed) >= 4:
            break
        return {
            "selected_views": selected_views,
            "selected_views_by_link": selected_views_by_link,
            "need_more_views": True,
            "proposed_views": proposed[:4],
            "reason": "current selected views are insufficient; propose additional views",
        "selection_explanation": {
            "why_this_view": "Current views do not reliably ground all required links by bbox+label, especially for links that may only become visible after an exposing motion.",
            "trajectory_reasoning": "When an inside link is initially hidden, propose views that are better aligned with the exposing motion so later motion can be observed rather than treating the link as impossible.",
            "link_checks": [],
        },
    }


def _build_prompt(
    coverage_report: dict,
    current_viewspecs: dict,
    conditioning_mask_meta: dict | None = None,
    causal_obj: dict | None = None,
    expected_motion_hints: dict | None = None,
) -> str:
    base = (
        "You are a view proposal assistant for visibility coverage only.\n"
        "Task: (1) select best current views for motion diagnosis, then (2) if needed, propose additional views.\n"
        "Use causal semantics and planned temporal motion ONLY for view ranking; do NOT redesign the action or plan.\n"
        "The attached coverage inputs are REAL textured/reference renders (not overlay points), one image per current view.\n"
        "BBoxes highlight links involved in interaction reasoning (target/effect links).\n"
        "Judge visibility from those boxed links together with the causal/semantic JSON and expected temporal motion hints.\n"
        "If MASK_* attachments are provided, they are target masks; use them to focus visibility of the masked target region.\n"
        "Selection rule:\n"
        "- selected_views_by_link must choose one CURRENT rendered view for each required link whenever possible.\n"
        "- Each link may use a different current view.\n"
        "- selected_views can be the deduplicated union of those per-link current views.\n"
        "- A view is usable only if required link identity is visually grounded by BOTH bbox and readable link label text.\n"
        "- A good view must show the REQUIRED LINK ON THE OUTER / SURFACE-FACING SIDE as clearly as possible; do not choose a backside or reverse-side view when a more surface-visible current view exists.\n"
        "- Prefer the current view with the highest usable surface visibility of the boxed link, not a view where the link is barely visible, back-facing, or mostly hidden behind the object body.\n"
        "- Do NOT infer link identity from bbox position alone.\n"
        "- If label text is missing/ambiguous/illegible for required links in a view, treat that view as NOT usable.\n"
        "- If two views both see the same link, prefer the one where the link's exposed outer surface is larger, clearer, and less occluded.\n"
        "- Explain selection using expected link motion observability (trajectory-friendly view), not only static visibility.\n"
        "- Prefer the best current view for EACH required link independently, not one compromise view for all links.\n"
        "- Use causal semantics to determine which links are target/effect/internal and which moving link may expose an inside link later.\n"
        "- If a required link is inside the object, prefer a view that tracks the exposing mover now and is best positioned to see the inside link after that motion.\n"
        "- IMPORTANT: if a required link is initially hidden inside the object, do NOT treat it as globally rejected or impossible just because every CURRENT view fails to show it.\n"
        "- Instead, use the plan/causal motion to identify the exposing motion and propose future-facing views that will see that link once the exposing part moves.\n"
        "- For such initially hidden links, 'all current views unusable' means 'need_more_views=true', NOT 'drop the link' and NOT 'reject the task'.\n"
        "- Focus primarily on orientation and visibility quality (which side / angle best reveals the link and its likely motion).\n"
        "- Downstream logic may adjust distance_scale after view selection, so you do not need to force distance changes to account for future motion extent.\n"
        "- Still explain whether the current framing feels too near, too far, or acceptable, but prioritize choosing the correct viewing direction.\n"
        "- When selecting a view for a link, explicitly explain why the chosen distance_scale is acceptable and why nearer/farther alternatives are worse if relevant.\n"
        "- Also explain why non-selected candidate views are rejected for that link (for example: poor bbox clarity, poor label readability, too tight, too far, weak trajectory observability, or wrong side for expected motion).\n"
        "Proposal rule:\n"
        "- If current views are insufficient/ambiguous, set need_more_views=true and provide up to 4 proposed_views.\n"
        "- If some required links are completely invisible in all current views because they are inside the object, propose a view that best faces the expected opening/exposing motion so those links can become visible later.\n"
        "- When an inside link is initially invisible, proposed_views should explicitly optimize for its later exposure after the planned motion, not only for its initial hidden state.\n"
        "- Do NOT write a rejection_reason that implies the inside link should be ignored forever; explain that the current views are insufficient now and that a later-exposure angle is needed.\n"
        "- If you can select confidently from current views, set need_more_views=false.\n"
        "- If you cannot confidently select from current views, keep selected_views empty and propose exactly 4 new views.\n"
        "Use ONLY the allowed discrete parameter sets:\n"
        f"- azimuth_deg: {lr.AZIMUTH_SET}\n"
        f"- elevation_deg: {lr.ELEVATION_SET}\n"
        f"- distance_scale: {lr.DISTANCE_SCALE_SET}\n"
        f"- fov_deg: {lr.FOV_SET}\n"
        "Output STRICT JSON only with schema:\n"
        "{\n"
        "  \"selected_views_by_link\": {\n"
        "    \"link_0\": {\"azimuth_deg\":0,\"elevation_deg\":20,\"distance_scale\":1.0,\"fov_deg\":35}\n"
        "  },\n"
        "  \"selected_views\": [\n"
        "    {\"azimuth_deg\":0,\"elevation_deg\":20,\"distance_scale\":1.0,\"fov_deg\":35}\n"
        "  ],\n"
        "  \"need_more_views\": true/false,\n"
        "  \"proposed_views\": [\n"
        "    {\"azimuth_deg\":90,\"elevation_deg\":30,\"distance_scale\":1.2,\"fov_deg\":35}\n"
        "  ],\n"
        "  \"reason\": \"...\"\n"
        "  \"selection_explanation\": {\n"
        "    \"why_this_view\": \"...\",\n"
        "    \"trajectory_reasoning\": \"...\",\n"
        "    \"link_checks\": [\n"
        "      {\n"
        "        \"link\": \"link_X\",\n"
        "        \"expected_motion\": \"rotation|translation|mixed|unknown\",\n"
        "        \"bbox_visible\": true,\n"
        "        \"label_readable\": true,\n"
        "        \"trajectory_observability\": \"high|medium|low|unknown\",\n"
        "        \"evidence\": \"short evidence\",\n"
        "        \"selection_reason\": \"why this view is chosen for this link\",\n"
        "        \"rejection_reason\": \"why the main alternatives are not chosen for this link\",\n"
        "        \"distance_reason\": \"why this distance_scale is neither too near nor too far\"\n"
        "      }\n"
        "    ]\n"
        "  }\n"
        "}\n"
        "Constraints:\n"
        "- selected_views_by_link should contain one current-view choice per required link whenever possible\n"
        "- proposed_views length <= 4\n"
        "- If some links cannot be assigned a confident current view, set need_more_views=true and propose up to 4 new views\n"
        "- A currently hidden internal link that is expected to become visible later MUST remain covered by proposed_views; do not omit it just because it is invisible in the initial state\n"
        "- If coverage is already sufficient, return need_more_views=false and proposed_views=[]\n"
        "The attached images are separate current views. Read them independently; do not compress them mentally into a tiny grid.\n"
        "Current viewspecs JSON:\n"
        + json.dumps(current_viewspecs, ensure_ascii=False, indent=2)
        + "\nCoverage report JSON:\n"
        + json.dumps(coverage_report, ensure_ascii=False, indent=2)
    )
    if isinstance(causal_obj, dict):
        base += "\nCausal/semantic JSON (authoritative for link identity and interaction roles):\n" + json.dumps(causal_obj, ensure_ascii=False, indent=2)
    if isinstance(expected_motion_hints, dict):
        base += "\nExpected motion + temporal visibility hints JSON (authoritative for trajectory-aware view selection):\n" + json.dumps(
            expected_motion_hints, ensure_ascii=False, indent=2
        )
    if conditioning_mask_meta is not None:
        base += "\nConditioning mask metadata JSON:\n" + json.dumps(conditioning_mask_meta, ensure_ascii=False, indent=2)
    return base


def propose_views_via_api(
    coverage_report: dict,
    current_viewspecs: dict,
    coverage_image_paths: list[Path],
    conditioning_mask_images: list[Path] | None = None,
    conditioning_mask_meta: dict | None = None,
    causal_obj: dict | None = None,
    expected_motion_hints: dict | None = None,
    model: str = "gpt-5.4",
    base_url: str | None = None,
    api_provider: str = "auto",
    api_key: str | None = None,
) -> dict:
    valid_coverage_images = [Path(p) for p in (coverage_image_paths or []) if Path(p).exists()]
    if not valid_coverage_images:
        raise FileNotFoundError("no coverage view images found")

    prompt = _build_prompt(
        coverage_report,
        current_viewspecs,
        conditioning_mask_meta,
        causal_obj,
        expected_motion_hints,
    )
    image_items: list[tuple[str, Path]] = []
    for i, p in enumerate(conditioning_mask_images or [], start=1):
        pp = Path(p)
        if pp.exists():
            image_items.append((f"MASK_{i}:{pp.name}", pp))
    for idx, pp in enumerate(valid_coverage_images, start=1):
        image_items.append((f"COVERAGE_VIEW_{idx}:{pp.name}", pp))
    msg, _cfg = generate_content_text(
        model=model,
        prompt=prompt,
        image_items=image_items,
        provider=api_provider,
        api_key=api_key,
        base_url=base_url,
    )
    obj = _extract_json(msg)
    if isinstance(obj, dict):
        obj["_required_links"] = list(coverage_report.get("required_links") or [])
    return _sanitize_proposal(obj)


def propose_views(
    coverage_report: dict,
    current_viewspecs: dict,
    coverage_image_paths: list[Path] | None = None,
    conditioning_mask_images: list[Path] | None = None,
    conditioning_mask_meta: dict | None = None,
    causal_obj: dict | None = None,
    expected_motion_hints: dict | None = None,
    model: str = "gpt-5.4",
    use_api: bool = True,
    api_provider: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict:
    heur = propose_views_heuristic(
        coverage_report,
        current_viewspecs,
        expected_motion_hints=expected_motion_hints,
    )
    if use_api and coverage_image_paths:
        try:
            proposal = propose_views_via_api(
                coverage_report,
                current_viewspecs,
                [Path(p) for p in (coverage_image_paths or [])],
                [Path(p) for p in (conditioning_mask_images or [])],
                conditioning_mask_meta,
                causal_obj,
                expected_motion_hints,
                model=model,
                base_url=base_url,
                api_provider=api_provider,
                api_key=api_key,
            )
            heur_score = _selected_view_score(heur, current_viewspecs, coverage_report, expected_motion_hints)
            api_score = _selected_view_score(proposal, current_viewspecs, coverage_report, expected_motion_hints)
            if heur_score > api_score + 0.5 and (heur.get("selected_views") or []):
                proposal["selected_views"] = heur.get("selected_views") or []
                proposal["selection_explanation"] = heur.get("selection_explanation") or proposal.get("selection_explanation")
                reason = str(proposal.get("reason") or "").strip()
                proposal["reason"] = f"{reason} | heuristic override: higher-information current view".strip(" |")
            return proposal
        except Exception as exc:
            print(f"[WARN] Coverage VLM API failed ({exc}); falling back to heuristic proposal.")
    return heur


def select_views_by_link(
    coverage_report: dict,
    current_viewspecs: dict,
    proposal: dict | None = None,
    expected_motion_hints: dict | None = None,
) -> dict[str, dict]:
    cur = lr.validate_viewspecs(current_viewspecs)
    heur = _select_views_by_link_heuristic(
        coverage_report,
        cur,
        expected_motion_hints=expected_motion_hints,
    )
    if not isinstance(proposal, dict):
        return heur
    raw = proposal.get("selected_views_by_link")
    if not isinstance(raw, dict):
        return heur
    use = {}
    valid_views = {
        (
            int(v["azimuth_deg"]),
            int(v["elevation_deg"]),
            float(v["distance_scale"]),
            int(v["fov_deg"]),
        )
        for v in (cur.get("views") or [])
    }
    for lk, row in raw.items():
        cand = _snap_view_dict(row)
        if cand is None:
            continue
        key = (
            int(cand["azimuth_deg"]),
            int(cand["elevation_deg"]),
            float(cand["distance_scale"]),
            int(cand["fov_deg"]),
        )
        if key not in valid_views:
            continue
        use[str(lk)] = cand
    for lk, row in heur.items():
        use.setdefault(str(lk), row)
    return use


def main():
    parser = argparse.ArgumentParser(description="Propose additional coverage views (VLM API + heuristic fallback)")
    parser.add_argument("--coverage_report", required=True)
    parser.add_argument("--viewspecs_json", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--causal_json", default=None)
    parser.add_argument("--coverage_images", nargs="*", default=None)
    parser.add_argument("--conditioning_mask_images", nargs="*", default=None)
    parser.add_argument("--conditioning_mask_meta", default=None)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--api_provider", default="auto", choices=["auto", "openai", "gemini"])
    parser.add_argument("--api_base_url", default=None)
    parser.add_argument("--no_api", action="store_true")
    args = parser.parse_args()
    report = json.loads(Path(args.coverage_report).read_text(encoding="utf-8"))
    viewspecs = json.loads(Path(args.viewspecs_json).read_text(encoding="utf-8"))
    out = propose_views(
        report,
        viewspecs,
        coverage_image_paths=[Path(p) for p in (args.coverage_images or [])],
        conditioning_mask_images=[Path(p) for p in (args.conditioning_mask_images or [])],
        conditioning_mask_meta=json.loads(Path(args.conditioning_mask_meta).read_text(encoding="utf-8")) if args.conditioning_mask_meta else None,
        causal_obj=(json.loads(Path(args.causal_json).read_text(encoding="utf-8")) if args.causal_json else None),
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
