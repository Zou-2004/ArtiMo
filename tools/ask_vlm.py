#!/usr/bin/env python3
import argparse
import json
import re
from glob import glob
from pathlib import Path

from api_client_utils import generate_content_text


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


def _extract_json_text(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    try:
        json.loads(s)
        return s
    except Exception:
        pass
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
            s2 = s[body_start:j].strip()
            try:
                json.loads(s2)
                return s2
            except Exception:
                cand = _extract_balanced_json_object(s2)
                if cand is not None:
                    return cand
            search_from = j + 3
    cand = _extract_balanced_json_object(s)
    if cand is not None:
        return cand
    return None


def _expand_paths(values: list[str] | None) -> list[Path]:
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
    return out


def _build_conditioning_suffix(
    conditioning_text: str | None,
    conditioning_images: list[Path],
    conditioning_masks: list[Path],
    attachment_labels: list[str],
) -> str:
    lines = []
    has_text = conditioning_text is not None and conditioning_text.strip() != ""
    has_image = len(conditioning_images) > 0
    has_mask = len(conditioning_masks) > 0
    if not (has_text or has_image or has_mask):
        return ""

    lines.append("")
    lines.append("Additional User Conditioning Inputs (for target grounding):")
    lines.append("- These are extra inputs beyond the standard overlay/reference images.")
    lines.append("- If mask images are provided, use them to localize the target region/link first.")
    lines.append("- If both text and masks exist: use mask(s) for target localization, text for action intent/time.")
    lines.append("- If only masks/images exist: still infer target link and plausible causal motion from visuals.")
    lines.append("- If only text exists: ground target from overlay/reference images and URDF summary.")
    lines.append("- IMPORTANT: Any attachment whose label starts with MASK_ is a target mask input, not a normal reference image.")
    lines.append("- For MASK_* attachments, prioritize the highlighted region to identify target_link/target_links.")
    if attachment_labels:
        lines.append("Attachment order and labels (authoritative):")
        for i, lab in enumerate(attachment_labels, start=1):
            lines.append(f"- ATTACHMENT {i}: {lab}")
    if has_text:
        lines.append("Conditioning text:")
        lines.append(conditioning_text.strip())
    if has_image:
        lines.append("Conditioning object image files (same object):")
        for i, p in enumerate(conditioning_images, start=1):
            lines.append(f"- OBJ_IMG_{i}: {p}")
    if has_mask:
        lines.append("Conditioning mask files (target region):")
        for i, p in enumerate(conditioning_masks, start=1):
            lines.append(f"- MASK_{i}: {p}")
        lines.append("- Assume all MASK_* files refer to the same target part across different views unless evidence suggests otherwise.")
    return "\n".join(lines)


def _build_input_availability_suffix(
    base_prompt_has_text: bool,
    has_auto_images: bool,
    conditioning_text: str | None,
    conditioning_images: list[Path],
    conditioning_masks: list[Path],
) -> str:
    has_text = base_prompt_has_text or (conditioning_text is not None and conditioning_text.strip() != "")
    has_image = has_auto_images or len(conditioning_images) > 0 or len(conditioning_masks) > 0
    lines = []
    lines.append("")
    lines.append("Current VLM input availability:")
    lines.append(f"- text_available: {'yes' if has_text else 'no'}")
    lines.append(f"- image_available: {'yes' if has_image else 'no'}")
    lines.append("- If image_available=no, rely on text + URDF only.")
    lines.append("- If text_available=no, rely on image/mask + URDF only.")
    return "\n".join(lines)


def _base_prompt_has_action_text(prompt: str) -> bool:
    marker = "User action prompt:"
    idx = prompt.find(marker)
    if idx < 0:
        return False
    tail = prompt[idx + len(marker) :].splitlines()
    for line in tail:
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("urdf joint summary"):
            return False
        return True
    return False


def _normalize_action_fields(action_obj) -> dict:
    action = dict(action_obj) if isinstance(action_obj, dict) else {}
    target_link = action.get("target_link")
    if isinstance(target_link, str):
        target_link = target_link.strip()
    else:
        target_link = ""
    target_links = []
    tls = action.get("target_links")
    if isinstance(tls, list):
        for x in tls:
            if not isinstance(x, str):
                continue
            s = x.strip()
            if s and s not in target_links:
                target_links.append(s)
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


def _normalize_effects_fields(effects_obj) -> dict:
    effects = dict(effects_obj) if isinstance(effects_obj, dict) else {}
    out = {}

    joint_targets = []
    seen_jt = set()
    for jt in effects.get("joint_targets") or []:
        if not isinstance(jt, dict):
            continue
        joint = str(jt.get("joint") or "").strip()
        if not joint:
            continue
        key = (joint, str(jt.get("to") or "").strip())
        if key in seen_jt:
            continue
        seen_jt.add(key)
        item = {"joint": joint}
        if "to" in jt:
            item["to"] = _canonicalize_joint_target_to_value(jt.get("to"))
        joint_targets.append(item)
    out["joint_targets"] = joint_targets

    modes = []
    for m in effects.get("modes") or []:
        if isinstance(m, dict):
            modes.append(m)
    out["modes"] = modes

    coupling_rules = []
    for r in effects.get("coupling_rules") or []:
        if isinstance(r, str) and r.strip():
            coupling_rules.append(r.strip())
    out["coupling_rules"] = coupling_rules
    return out


def _canonicalize_joint_target_to_value(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if not s.lower().startswith("velocity:"):
        return value
    tail = s.split(":", 1)[1].strip()
    m = re.fullmatch(r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(?:\s*_?\s*radps)?", tail, flags=re.IGNORECASE)
    if m is None:
        return value
    try:
        val = float(m.group(1))
    except Exception:
        return value
    return f"velocity:{val:g}"


def _collect_invalid_velocity_targets(parsed: dict) -> list[str]:
    issues: list[str] = []

    def _scan_effects(effects_obj, prefix: str):
        effects = effects_obj if isinstance(effects_obj, dict) else {}
        for idx, jt in enumerate(effects.get("joint_targets") or []):
            if not isinstance(jt, dict):
                continue
            raw_to = jt.get("to")
            if raw_to is None:
                continue
            s = str(raw_to).strip()
            if not s.lower().startswith("velocity:"):
                continue
            normalized = str(_canonicalize_joint_target_to_value(s) or "").strip()
            tail = normalized.split(":", 1)[1].strip() if ":" in normalized else ""
            if re.fullmatch(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", tail) is not None:
                continue
            joint_name = str(jt.get("joint") or "").strip() or f"joint_targets[{idx}]"
            issues.append(f"{prefix}.{joint_name}={s}")

    causal = parsed.get("causal")
    if isinstance(causal, dict):
        _scan_effects(causal.get("effects"), "causal.effects")
    for seg_idx, seg in enumerate(parsed.get("causal_segments") or []):
        if isinstance(seg, dict):
            _scan_effects(seg.get("effects"), f"causal_segments[{seg_idx}].effects")
    return issues


def _normalize_single_causal(causal_obj) -> dict:
    causal = dict(causal_obj) if isinstance(causal_obj, dict) else {}
    has_causal = causal.get("has_causal")
    if isinstance(has_causal, bool):
        has_causal_flag = has_causal
    else:
        has_causal_flag = None
    if has_causal_flag is False:
        return {"has_causal": False, "action": None, "effects": None}
    action = _normalize_action_fields(causal.get("action"))
    effects = _normalize_effects_fields(causal.get("effects"))
    if has_causal_flag is None:
        has_causal_flag = bool(action) or bool(effects.get("joint_targets"))
    return {
        "has_causal": bool(has_causal_flag),
        "action": action if has_causal_flag else None,
        "effects": effects if has_causal_flag else None,
    }


def _merge_effects(effect_list: list[dict]) -> dict:
    merged = {"joint_targets": [], "modes": [], "coupling_rules": []}
    seen_joint = {}
    for ef in effect_list:
        if not isinstance(ef, dict):
            continue
        for jt in ef.get("joint_targets") or []:
            if not isinstance(jt, dict):
                continue
            joint = str(jt.get("joint") or "").strip()
            if not joint:
                continue
            if joint not in seen_joint:
                item = {"joint": joint}
                if "to" in jt:
                    item["to"] = _canonicalize_joint_target_to_value(jt.get("to"))
                seen_joint[joint] = item
                merged["joint_targets"].append(item)
            elif "to" in jt and "to" not in seen_joint[joint]:
                seen_joint[joint]["to"] = _canonicalize_joint_target_to_value(jt.get("to"))
        for m in ef.get("modes") or []:
            if isinstance(m, dict) and m not in merged["modes"]:
                merged["modes"].append(m)
        for r in ef.get("coupling_rules") or []:
            if isinstance(r, str) and r and r not in merged["coupling_rules"]:
                merged["coupling_rules"].append(r)
    return merged


def _load_renderable_link_hints(outputs_root: Path, asset: str) -> tuple[set[str], str | None]:
    scale_path = outputs_root / asset / "scale_context.json"
    if not scale_path.exists():
        return set(), None
    try:
        scale = json.loads(scale_path.read_text(encoding="utf-8"))
    except Exception:
        return set(), None
    boxes = scale.get("link_bbox_extents_m") if isinstance(scale, dict) else None
    if not isinstance(boxes, dict):
        return set(), None
    renderable = {str(k).strip() for k in boxes.keys() if str(k).strip()}
    best_link = None
    best_score = -1.0
    for link_name, ext in boxes.items():
        if not isinstance(ext, list) or len(ext) != 3:
            continue
        try:
            vals = [abs(float(x)) for x in ext]
        except Exception:
            continue
        score = vals[0] * vals[1] * vals[2]
        if score > best_score:
            best_score = score
            best_link = str(link_name).strip()
    return renderable, best_link


def _is_root_like_link(link_name: str | None) -> bool:
    s = str(link_name or "").strip().lower()
    return s in {"base", "root", "world", "world_root"}


def _action_has_whole_object_motion(action: dict) -> bool:
    if isinstance(action.get("direction_axis_world"), list) and len(action.get("direction_axis_world") or []) == 3:
        return True
    primitive = str(action.get("primitive") or "").strip().lower()
    return primitive in {"push", "pull", "drag", "drive", "transport", "translate", "roll", "slide"}


def _retarget_nonrenderable_root_action(action: dict, renderable_links: set[str], fallback_link: str | None) -> dict:
    if not isinstance(action, dict) or not fallback_link:
        return action
    target = str(action.get("target_link") or "").strip()
    if not target:
        return action
    if target in renderable_links:
        return action
    if not _is_root_like_link(target):
        return action
    if not _action_has_whole_object_motion(action):
        return action
    out = dict(action)
    out["target_link"] = fallback_link
    links = []
    for item in out.get("target_links") or []:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or _is_root_like_link(s):
            continue
        if s in renderable_links and s not in links:
            links.append(s)
    if fallback_link not in links:
        links.insert(0, fallback_link)
    out["target_links"] = links
    return out


def _normalize_vlm_output(parsed: dict, renderable_links: set[str] | None = None, fallback_target_link: str | None = None) -> dict:
    if not isinstance(parsed, dict):
        return parsed
    renderable_links = set(renderable_links or set())

    segments = None
    seg_in = parsed.get("causal_segments")
    if isinstance(seg_in, list):
        seg_list = []
        for i, seg in enumerate(seg_in):
            if not isinstance(seg, dict):
                continue
            seg_out = dict(seg)
            seg_out["segment_id"] = str(seg.get("segment_id") or f"S{i+1}")
            th = seg.get("time_hint") if isinstance(seg.get("time_hint"), dict) else {}
            th_out = dict(th)
            try:
                th_out["order_index"] = int(th.get("order_index", i))
            except Exception:
                th_out["order_index"] = i
            seg_out["time_hint"] = th_out
            seg_out["action"] = _retarget_nonrenderable_root_action(
                _normalize_action_fields(seg.get("action")),
                renderable_links,
                fallback_target_link,
            )
            seg_out["effects"] = _normalize_effects_fields(seg.get("effects"))
            seg_list.append(seg_out)
        if seg_list:
            segments = seg_list

    top_causal = _normalize_single_causal(parsed.get("causal"))
    if isinstance(top_causal.get("action"), dict):
        top_causal["action"] = _retarget_nonrenderable_root_action(
            top_causal["action"],
            renderable_links,
            fallback_target_link,
        )
    if segments is not None:
        # Segments are authoritative when provided. Ignore duplicated/conflicting
        # top-level action/effects from model output and rebuild summary from segments.
        first_action = segments[0].get("action") if isinstance(segments[0].get("action"), dict) else {}
        top_causal["has_causal"] = True
        top_causal["action"] = dict(first_action) if first_action else None
        merged_effects = _merge_effects([s.get("effects") or {} for s in segments])
        top_causal["effects"] = merged_effects
    else:
        if top_causal.get("has_causal") is False:
            top_causal["action"] = None
            top_causal["effects"] = None

    parsed["causal_segments"] = segments
    parsed["causal"] = top_causal
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Send VLM prompt + images to model and save JSON output.")
    parser.add_argument("--asset", required=True, help="Asset name (e.g., bin or car)")
    parser.add_argument(
        "--outputs_root",
        default="/home/zouchunyu/casual_articulation/outputs",
        help="Root outputs directory",
    )
    parser.add_argument("--model", default="gpt-5.2", help="Model name")
    parser.add_argument("--api_key", default=None, help="Optional API key override (otherwise read from env)")
    parser.add_argument("--api_provider", default="auto", choices=["auto", "openai", "gemini"])
    parser.add_argument("--api_base_url", default=None, help="Optional API base URL override")
    parser.add_argument(
        "--vlm_conditioning_images",
        nargs="*",
        default=[],
        help="Optional extra object images for grounding target link (supports globs).",
    )
    parser.add_argument(
        "--vlm_conditioning_masks",
        nargs="*",
        default=[],
        help="Optional extra target mask images for grounding target link (supports globs).",
    )
    parser.add_argument(
        "--vlm_conditioning_text",
        default=None,
        help="Optional extra text hint for VLM grounding (can be empty).",
    )
    parser.add_argument(
        "--vlm_no_auto_images",
        action="store_true",
        help="If set, do not auto-attach outputs/<asset>/images/*.png. Only use conditioning images/masks.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    prompt_path = outputs_root / args.asset / "prompts" / "vlm_prompt.md"
    images_dir = outputs_root / args.asset / "images"
    if not prompt_path.exists():
        raise SystemExit(f"Missing VLM prompt: {prompt_path}")
    if not images_dir.exists() and not args.vlm_no_auto_images:
        raise SystemExit(f"Missing images directory: {images_dir}")

    prompt = prompt_path.read_text(encoding="utf-8")
    auto_images: list[Path] = []
    if not args.vlm_no_auto_images and images_dir.exists():
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            auto_images.extend(sorted(images_dir.rglob(ext)))
    cond_images = _expand_paths(args.vlm_conditioning_images)
    cond_masks = _expand_paths(args.vlm_conditioning_masks)

    attachment_items: list[tuple[str, Path]] = []
    # Put masks first to make target grounding unambiguous.
    for i, p in enumerate(cond_masks, start=1):
        attachment_items.append((f"MASK_{i}:{p.name}", p))
    for i, p in enumerate(cond_images, start=1):
        attachment_items.append((f"COND_IMG_{i}:{p.name}", p))
    for i, p in enumerate(auto_images, start=1):
        attachment_items.append((f"AUTO_IMG_{i}:{p.name}", p))
    # Keep only unique paths while preserving first label/order.
    dedup_items: list[tuple[str, Path]] = []
    seen_paths = set()
    for label, p in attachment_items:
        sp = str(p)
        if sp in seen_paths:
            continue
        seen_paths.add(sp)
        dedup_items.append((label, p))
    attachment_items = dedup_items
    attachment_labels = [label for label, _ in attachment_items]

    prompt_suffix = _build_conditioning_suffix(
        conditioning_text=args.vlm_conditioning_text,
        conditioning_images=cond_images,
        conditioning_masks=cond_masks,
        attachment_labels=attachment_labels,
    )
    availability_suffix = _build_input_availability_suffix(
        base_prompt_has_text=_base_prompt_has_action_text(prompt),
        has_auto_images=len(auto_images) > 0,
        conditioning_text=args.vlm_conditioning_text,
        conditioning_images=cond_images,
        conditioning_masks=cond_masks,
    )
    merged_prompt = prompt + prompt_suffix + availability_suffix

    filled_prompt_path = outputs_root / args.asset / "prompts" / "vlm_prompt_filled.md"
    filled_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    filled_prompt_path.write_text(merged_prompt, encoding="utf-8")
    manifest_path = outputs_root / args.asset / "prompts" / "vlm_input_manifest.json"
    manifest = {
        "conditioning_text": args.vlm_conditioning_text,
        "conditioning_images": [str(p) for p in cond_images],
        "conditioning_masks": [str(p) for p in cond_masks],
        "auto_images": [str(p) for p in auto_images],
        "attachments": [{"label": label, "path": str(path)} for label, path in attachment_items],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote filled VLM prompt to {filled_prompt_path}")
    print(f"[INFO] Wrote VLM input manifest to {manifest_path}")
    print(f"[INFO] VLM images: auto={len(auto_images)} conditioning_obj={len(cond_images)} conditioning_mask={len(cond_masks)} total={len(attachment_items)}")
    for i, (label, path) in enumerate(attachment_items, start=1):
        print(f"[INFO] ATTACHMENT {i}: {label} -> {path}")

    message, cfg = generate_content_text(
        model=args.model,
        prompt=merged_prompt,
        image_items=attachment_items,
        provider=args.api_provider,
        api_key=args.api_key,
        base_url=args.api_base_url,
    )
    print(f"[INFO] API provider={cfg['provider']} base_url={cfg['base_url']}")
    message = message or "{}"
    print(message)

    out_dir = outputs_root / args.asset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "output.json"
    cleaned = _extract_json_text(message)
    if cleaned is not None:
        parsed = json.loads(cleaned)
        renderable_links, fallback_target_link = _load_renderable_link_hints(outputs_root, args.asset)
        parsed = _normalize_vlm_output(
            parsed,
            renderable_links=renderable_links,
            fallback_target_link=fallback_target_link,
        )
        invalid_velocity_targets = _collect_invalid_velocity_targets(parsed)
        if invalid_velocity_targets:
            raw_path = out_dir / "output.invalid_velocity_raw.txt"
            raw_path.write_text(message, encoding="utf-8")
            raise SystemExit(
                "Invalid VLM velocity placeholders detected; expected explicit numeric values from scale context. "
                f"Offending targets: {', '.join(invalid_velocity_targets)}. Raw response saved to {raw_path}"
            )
        out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_path.write_text(message, encoding="utf-8")


if __name__ == "__main__":
    main()
