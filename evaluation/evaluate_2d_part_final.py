#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageSequence

EVALUATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_ROOT.parent
TOOLS_ROOT = REPO_ROOT / "tools"
for _path in (EVALUATION_ROOT, TOOLS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_2d_part as p2d  # noqa: E402
import evaluate_external_4d as ex  # noqa: E402
import gen_overlays_and_prompts as gop  # noqa: E402


VIEW_BY_NAME = {
    "reference_view_01": ("V1", 0.0, 20.0),
    "reference_view_02": ("V2", 90.0, 20.0),
    "reference_view_03": ("V3", 180.0, 20.0),
    "reference_view_04": ("V4", 270.0, 20.0),
    "coverage_view_V1": ("V1", 0.0, 20.0),
    "coverage_view_V2": ("V2", 90.0, 20.0),
    "coverage_view_V3": ("V3", 180.0, 20.0),
    "coverage_view_V4": ("V4", 270.0, 20.0),
}

_LPIPS_MODEL = None
_LPIPS_DEVICE = None
_PUPPET_RENDER_ROWS: list[dict[str, Any]] | None = None
_PROJECT_SEARCH_ROOTS: list[Path] = []


def _add_project_search_root(path: Path | str | None) -> None:
    if path is None:
        return
    try:
        root = Path(path).expanduser().resolve()
    except Exception:
        return
    if not root.exists() or not root.is_dir():
        return
    if root not in _PROJECT_SEARCH_ROOTS:
        _PROJECT_SEARCH_ROOTS.append(root)


def _project_search_roots() -> list[Path]:
    roots = [REPO_ROOT, *_PROJECT_SEARCH_ROOTS]
    out: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            key = root.resolve()
        except Exception:
            key = root
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _configure_project_search_roots(args: argparse.Namespace) -> None:
    global _PUPPET_RENDER_ROWS
    for root in getattr(args, "project_root", None) or []:
        _add_project_search_root(root)
    for path in [
        getattr(args, "puppet_root", None),
        getattr(args, "noncausal_views_json", None),
        getattr(args, "manifest", None),
        getattr(args, "own_3d_dir", None),
        getattr(args, "aam_3d_dir", None),
        getattr(args, "animate3d_3d_dir", None),
        getattr(args, "particulate_3d_dir", None),
    ]:
        if path is None:
            continue
        p = Path(path).expanduser()
        base = p if p.is_dir() else p.parent
        for parent in [base, *base.parents]:
            if (parent / "tools").is_dir() and (
                (parent / "puppet_master_noncausal").exists()
                or (parent / "puppet_agent_style_causal_outputs_saved_drags").exists()
                or (parent / "benchmark_annotations").exists()
            ):
                _add_project_search_root(parent)
                break
    _PUPPET_RENDER_ROWS = None


def _lpips_model(args: argparse.Namespace):
    global _LPIPS_MODEL, _LPIPS_DEVICE
    if not bool(getattr(args, "compute_lpips", False)):
        return None, None
    import torch
    import lpips

    device = str(getattr(args, "lpips_device", "auto") or "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _LPIPS_MODEL is None or _LPIPS_DEVICE != device:
        _LPIPS_MODEL = lpips.LPIPS(net=str(getattr(args, "lpips_net", "alex") or "alex")).to(device).eval()
        _LPIPS_DEVICE = device
    return _LPIPS_MODEL, device


def _resize_binary_mask(mask: np.ndarray, size: int) -> np.ndarray:
    arr = (np.asarray(mask, dtype=np.uint8) * 255)
    return cv2.resize(arr, (int(size), int(size)), interpolation=cv2.INTER_NEAREST)


def _binary_mask_lpips(gt_mask: np.ndarray, pred_mask: np.ndarray, eval_mask: np.ndarray, args: argparse.Namespace) -> float | None:
    model, device = _lpips_model(args)
    if model is None or device is None:
        return None
    bbox = p2d._mask_bbox(eval_mask, pad=2)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    g = _resize_binary_mask(gt_mask[y0 : y1 + 1, x0 : x1 + 1], int(args.lpips_resolution))
    p = _resize_binary_mask(pred_mask[y0 : y1 + 1, x0 : x1 + 1], int(args.lpips_resolution))
    import torch

    def to_tensor(x: np.ndarray):
        t = torch.from_numpy(x.astype(np.float32) / 127.5 - 1.0)
        return t[None, None].repeat(1, 3, 1, 1).to(device)

    with torch.inference_mode():
        return float(model(to_tensor(g), to_tensor(p)).detach().cpu().reshape(-1)[0].item())


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, dtype=np.uint8)
    if int(np.count_nonzero(m)) == 0:
        return np.zeros_like(m, dtype=bool)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    return (m > 0) & (eroded == 0)


def _boundary_f1(gt_mask: np.ndarray, pred_mask: np.ndarray, tolerance_px: int) -> float | None:
    gt_b = _mask_boundary(gt_mask)
    pred_b = _mask_boundary(pred_mask)
    gt_count = int(np.count_nonzero(gt_b))
    pred_count = int(np.count_nonzero(pred_b))
    if gt_count == 0 and pred_count == 0:
        return 1.0
    if gt_count == 0 or pred_count == 0:
        return 0.0
    k = max(1, int(tolerance_px)) * 2 + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    gt_band = cv2.dilate(gt_b.astype(np.uint8), kernel, iterations=1).astype(bool)
    pred_band = cv2.dilate(pred_b.astype(np.uint8), kernel, iterations=1).astype(bool)
    precision = float(np.count_nonzero(pred_b & gt_band)) / max(1, pred_count)
    recall = float(np.count_nonzero(gt_b & pred_band)) / max(1, gt_count)
    if precision + recall <= 1.0e-12:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _contour_chamfer_distance(gt_mask: np.ndarray, pred_mask: np.ndarray, union_mask: np.ndarray, scale_floor_px: float) -> float | None:
    gt_b = _mask_boundary(gt_mask)
    pred_b = _mask_boundary(pred_mask)
    gt_count = int(np.count_nonzero(gt_b))
    pred_count = int(np.count_nonzero(pred_b))
    if gt_count == 0 and pred_count == 0:
        return 0.0
    bbox = p2d._mask_bbox(union_mask, pad=0)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    scale = max(float(math.hypot(max(1, x1 - x0 + 1), max(1, y1 - y0 + 1))), float(scale_floor_px), 1.0)
    if gt_count == 0 or pred_count == 0:
        return 1.0

    def dist_to_boundary(boundary: np.ndarray) -> np.ndarray:
        # distanceTransform returns each non-zero pixel's distance to the nearest zero pixel.
        inv = np.where(boundary, 0, 1).astype(np.uint8)
        return cv2.distanceTransform(inv, cv2.DIST_L2, 3)

    d_gt = dist_to_boundary(gt_b)
    d_pred = dist_to_boundary(pred_b)
    pred_to_gt = float(np.mean(d_gt[pred_b])) if pred_count else scale
    gt_to_pred = float(np.mean(d_pred[gt_b])) if gt_count else scale
    return float(0.5 * (pred_to_gt + gt_to_pred) / scale)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_3d_metric_rows(root: Path | None, *, required: bool) -> list[dict[str, str]]:
    if root is None:
        if required:
            raise FileNotFoundError("Missing 3D evaluation directory")
        return []
    candidates = [root / "aam_metrics.csv", root / "ablation_metrics.csv"]
    for path in candidates:
        if path.exists():
            rows = _read_csv_rows(path)
            per_case_path = root / "diagnose/per_case_metrics.json"
            if per_case_path.exists():
                try:
                    per_case_rows = _read_json(per_case_path)
                except Exception:
                    per_case_rows = []
                if isinstance(per_case_rows, list):
                    by_key = {
                        (
                            str(r.get("case_id") or ""),
                            str(r.get("class") or ""),
                            str(r.get("asset_name") or ""),
                            str(r.get("action_name") or ""),
                            str(r.get("variant") or ""),
                        ): r
                        for r in per_case_rows
                        if isinstance(r, dict)
                    }
                    for row in rows:
                        detail = by_key.get(
                            (
                                str(row.get("case_id") or ""),
                                str(row.get("class") or ""),
                                str(row.get("asset_name") or ""),
                                str(row.get("action_name") or ""),
                                str(row.get("variant") or ""),
                            )
                        )
                        if not isinstance(detail, dict):
                            continue
                        if not row.get("prediction_file"):
                            row["prediction_file"] = str(detail.get("prediction_glb") or "")
                        for key in ("prediction_glb", "prediction_trajectory", "prediction_plan"):
                            if detail.get(key) is not None and not row.get(key):
                                row[key] = str(detail.get(key))
            return rows
    if required:
        raise FileNotFoundError(f"No 3D metrics CSV found under {root}; expected aam_metrics.csv or ablation_metrics.csv")
    return []


def _read_matched_rows(root: Path | None, *, required: bool) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if root is None:
        if required:
            raise FileNotFoundError("Missing 3D evaluation directory")
        return {}
    path = root / "diagnose/matched_frames.json"
    if path.exists():
        return _matched_by_key(path)
    if required:
        raise FileNotFoundError(f"No matched frames found: {path}")
    return {}


def _load_matching_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    obj = _read_json(path)
    return obj if isinstance(obj, dict) else None


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _case_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("class") or ""), str(row.get("asset_name") or ""), str(row.get("action_name") or ""))


def _resolve_project_relative_path(raw: Any, case: dict[str, Any] | None = None) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    bases: list[Path] = []
    if case is not None:
        for key in ("asset_root", "gt_glb", "gt_trajectory", "gt_plan_json", "annotation_path"):
            value = case.get(key)
            if not value:
                continue
            try:
                p = Path(str(value)).resolve()
            except Exception:
                continue
            bases.extend([p.parent, *p.parents])
    bases.extend(_project_search_roots())
    seen: set[Path] = set()
    for base in bases:
        if base in seen:
            continue
        seen.add(base)
        cand = base / path
        if cand.exists():
            return cand.resolve()
    return (REPO_ROOT / path).resolve()


def _matched_by_key(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in _read_json(path):
        out[(str(row.get("asset_name") or ""), str(row.get("action_name") or ""))] = row
    return out


def _manifest_search_bases(path: Path) -> list[Path]:
    p = Path(path).expanduser().resolve()
    bases = [p.parent, *p.parents, REPO_ROOT]
    out: list[Path] = []
    seen: set[Path] = set()
    for base in bases:
        try:
            key = base.resolve()
        except Exception:
            key = base
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _resolve_manifest_path(raw: Any, manifest_path: Path) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve() if path.exists() else path
    for base in _manifest_search_bases(manifest_path):
        cand = base / path
        if cand.exists():
            return cand.resolve()
    return (REPO_ROOT / path).resolve()


def _resolve_case_asset_root(case: dict[str, Any], manifest_path: Path, data_roots: list[Path]) -> Path | None:
    asset = str(case.get("asset_name") or "")
    collection = str(case.get("asset_collection") or ("not_causal_data" if case.get("class") == "non_causal" else "causal_data"))
    roots = [Path(p).expanduser() for p in data_roots]
    roots.extend([REPO_ROOT / "data"])
    roots.extend([base / "data" for base in _manifest_search_bases(manifest_path)])
    seen: set[Path] = set()
    for root in roots:
        try:
            key = root.resolve()
        except Exception:
            key = root
        if key in seen:
            continue
        seen.add(key)
        for cand in (root / collection / asset, root / asset):
            if (cand / "mobility.urdf").exists():
                return cand.resolve()
    raw = _resolve_manifest_path(case.get("asset_root"), manifest_path)
    if raw is not None and raw.exists():
        return raw.resolve()
    return raw


def _resolve_release_case_paths(case: dict[str, Any], manifest_path: Path, data_roots: list[Path]) -> dict[str, Any]:
    out = dict(case)
    asset_root = _resolve_case_asset_root(out, manifest_path, data_roots)
    if asset_root is not None:
        out["asset_root"] = str(asset_root)
    for key in ("annotation_path", "gt_trajectory", "gt_glb", "gt_plan_json"):
        resolved = _resolve_manifest_path(out.get(key), manifest_path)
        if resolved is not None:
            out[key] = str(resolved)
    return out


def _manifest_cases(path: Path, data_roots: list[Path] | None = None) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    data = _read_json(path)
    cases = [_resolve_release_case_paths(case, path, data_roots or []) for case in data.get("cases") or []]
    return {_case_key(case): case for case in cases}


def _matching_filename(cls: str, asset: str, action: str) -> str:
    return f"{cls}__{asset}__{action}.json"


def _view_from_name(path: Path) -> tuple[str, float, float]:
    stem = path.stem
    if stem in VIEW_BY_NAME:
        return VIEW_BY_NAME[stem]
    if path.name in VIEW_BY_NAME:
        return VIEW_BY_NAME[path.name]
    raise ValueError(f"Unsupported selected view name: {path}")


def _load_noncausal_views(path: Path) -> dict[str, tuple[str, float, float]]:
    out = {}
    if not path.exists():
        rel_candidates = [path]
        parts = path.parts
        if "puppet_master_noncausal" in parts:
            idx = parts.index("puppet_master_noncausal")
            rel_candidates.append(Path(*parts[idx:]))
        for root in _project_search_roots():
            for rel in rel_candidates:
                if rel.is_absolute():
                    continue
                cand = root / rel
                if cand.exists():
                    path = cand
                    break
            if path.exists():
                break
    if not path.exists():
        return out
    for row in _read_json(path):
        asset = str(row.get("asset") or "")
        views = row.get("views") or {}
        if not asset or not views:
            continue
        name, spec = next(iter(views.items()))
        view_id, az, el = _view_from_name(Path(name))
        az = float(spec.get("azimuth_deg", az))
        el = float(spec.get("elevation_deg", el))
        out[asset] = (view_id, az, el)
    return out


def _find_puppet_gif(
    case: dict[str, Any],
    puppet_root: Path,
    noncausal_views: dict[str, tuple[str, float, float]],
) -> tuple[Path, tuple[str, float, float]]:
    cls, asset, action = _case_key(case)
    case_id = str(case.get("case_id") or "")
    final_roots = [root / "experiments/final_puppet_results" for root in _project_search_roots()]
    if cls == "hard_case":
        for root in [puppet_root, *final_roots]:
            case_dir = root / "hard" / asset / action
            matches = sorted(case_dir.glob("reference_view_*.gif")) or sorted(case_dir.glob("**/coverage_view_*.gif"))
            if matches:
                return matches[0], _view_from_name(matches[0])
        raise FileNotFoundError(f"No hard Puppet GIF for {asset}/{action}")
    if case_id.startswith("casual_output:"):
        direct_dir = puppet_root / asset / action
        direct_matches = sorted(direct_dir.glob("reference_view_*.gif")) or sorted(direct_dir.glob("**/coverage_view_*.gif"))
        if direct_matches:
            return direct_matches[0], _view_from_name(direct_matches[0])
        for project_root in _project_search_roots():
            legacy_dir = project_root / "experiments/misc/puppetmaster_intermediate/puppet_master_causal" / asset / action / "iter00"
            legacy_matches = sorted(legacy_dir.glob("*.gif"))
            if legacy_matches:
                view_pngs = sorted(legacy_dir.glob("coverage_view_*.png"))
                view = _view_from_name(view_pngs[0]) if view_pngs else ("V4", 270.0, 20.0)
                return legacy_matches[0], view
        for root in [puppet_root, *final_roots]:
            case_dir = root / "causal" / asset / action
            matches = sorted(case_dir.glob("reference_view_*.gif")) or sorted(case_dir.glob("**/coverage_view_*.gif"))
            if matches:
                return matches[0], _view_from_name(matches[0])
        raise FileNotFoundError(f"No causal Puppet GIF for {asset}/{action}")
    for root in [puppet_root, *final_roots]:
        matches = sorted((root / "noncausal" / asset).glob("*.gif"))
        if matches:
            return matches[0], noncausal_views.get(asset, ("V4", 270.0, 20.0))
    for project_root in _project_search_roots():
        matches = sorted((project_root / "puppet_master_noncausal" / asset).glob("*.gif"))
        if matches:
            return matches[0], noncausal_views.get(asset, ("V4", 270.0, 20.0))
    raise FileNotFoundError(f"No noncausal Puppet GIF for {asset}")


def _find_puppet_source_image(case: dict[str, Any], gif_path: Path) -> Path | None:
    cls, asset, action = _case_key(case)
    stem = gif_path.stem
    resolved = gif_path.resolve()
    candidates: list[Path] = []
    roots = _project_search_roots()
    if cls == "hard_case":
        for root in roots:
            candidates.extend(
                [
                    root / "experiments/misc/puppetmaster_intermediate/puppet_hard_agent_style/reference_inputs" / asset / action / f"{stem}.png",
                    root / "experiments/final_puppet_results/reference_inputs/hard_agent_style" / asset / action / f"{stem}.png",
                ]
            )
    elif str(case.get("case_id") or "").startswith("casual_output:"):
        candidates.extend([gif_path.with_name(f"{stem}.png"), resolved.with_name(f"{resolved.stem}.png")])
        for root in roots:
            candidates.extend(
                [
                    root / "experiments/misc/puppetmaster_intermediate/puppet_master_causal" / asset / action / f"{stem}.png",
                    root / "minimal_vlm_output" / asset / action / asset / "reference_views" / f"{stem}.png",
                    root / "workspace_puppet/microwave_riceCooker_safe_singleTrayToasteroven_toasteroven" / asset / action / f"{stem}.png",
                    root / "workspace_puppet/bin_door_kettle_trolley" / asset / action / f"{stem}.png",
                    root / "puppet_agent_style_causal_outputs_saved_drags" / asset / action / f"{stem}.png",
                    root / "puppet_agent_style_causal_outputs_saved_drags" / asset / action / "iter00" / f"{stem}.png",
                    root / "experiments/misc/puppetmaster_intermediate/puppet_agent_style_reference_inputs/causal_selected" / asset / action / "iter00" / f"{stem}.png",
                    root / "experiments/misc/puppetmaster_intermediate/puppet_agent_style_reference_inputs/causal_selected" / asset / action / f"{stem}.png",
                    root / "puppet_agent_style_reference_inputs/causal_selected" / asset / action / "iter00" / f"{stem}.png",
                ]
            )
    else:
        parent = resolved.parent
        candidates.extend(sorted(parent.glob("reference_view_*.png")))
        for root in roots:
            candidates.extend(sorted((root / "puppet_master_noncausal" / asset).glob("reference_view_*.png")))
    candidates.extend([gif_path.with_name("image_process.png"), gif_path.with_name(f"{stem}_preprocessed.png"), resolved.with_name(f"{resolved.stem}_preprocessed.png")])
    for root in roots:
        candidates.extend(
            [
                root / "experiments/misc/puppetmaster_intermediate/puppet_agent_style_causal_infer_inputs/images" / asset / action / "iter00" / "image_process.png",
                root / "puppet_agent_style_causal_outputs_saved_drags" / asset / action / f"{stem}_preprocessed.png",
                root / "puppet_agent_style_causal_outputs_saved_drags" / asset / action / "iter00" / f"{stem}_preprocessed.png",
            ]
        )
    candidates.extend([gif_path.with_suffix(".png"), resolved.with_suffix(".png")])
    for path in candidates:
        try:
            if path.exists():
                return path
        except OSError:
            continue
    return None


def _puppet_render_rows() -> list[dict[str, Any]]:
    global _PUPPET_RENDER_ROWS
    if _PUPPET_RENDER_ROWS is not None:
        return _PUPPET_RENDER_ROWS
    rows: list[dict[str, Any]] = []
    for root in _project_search_roots():
        for path in [
            root / "experiments/misc/puppetmaster_intermediate/puppet_agent_style_reference_inputs/render_manifest.json",
            root / "experiments/final_puppet_results/manifests/hard_render_manifest.json",
            root / "experiments/misc/puppetmaster_intermediate/puppet_hard_agent_style/render_manifest.json",
        ]:
            if path.exists():
                try:
                    data = _read_json(path)
                    rows.extend([dict(r) for r in (data if isinstance(data, list) else data.get("rows", [])) if isinstance(r, dict)])
                except Exception:
                    pass
        noncausal_distances = root / "puppet_master_noncausal/reference_view_distances.json"
        if noncausal_distances.exists():
            try:
                for item in _read_json(noncausal_distances):
                    if not isinstance(item, dict):
                        continue
                    asset = str(item.get("asset") or "")
                    action = str(item.get("action") or "")
                    views = item.get("views") or {}
                    if not asset or not views:
                        continue
                    for name, spec in views.items():
                        if not isinstance(spec, dict):
                            continue
                        rows.append(
                            {
                                "split": "noncausal",
                                "asset": asset,
                                "action": action,
                                "view_name": str(name),
                                "azimuth_deg": float(spec.get("azimuth_deg", 270.0)),
                                "elevation_deg": float(spec.get("elevation_deg", 20.0)),
                                "distance": float(spec.get("distance", item.get("reference_distance", 1.0))),
                                "fov_deg": float(spec.get("fov_deg", 50.0)),
                                "source": "noncausal_reference_view_distances",
                                "outputs": [str(root / "puppet_master_noncausal" / asset / str(name))],
                            }
                        )
            except Exception:
                pass
    _PUPPET_RENDER_ROWS = rows
    return rows


def _find_puppet_render_row(case: dict[str, Any], gif_path: Path | None, view: tuple[str, float, float]) -> dict[str, Any] | None:
    cls, asset, action = _case_key(case)
    view_id = str(view[0])
    view_name = {
        "V1": "reference_view_01.png",
        "V2": "reference_view_02.png",
        "V3": "reference_view_03.png",
        "V4": "reference_view_04.png",
    }.get(view_id)
    coverage_name = f"coverage_view_{view_id}.png"
    gif_stem = gif_path.stem if gif_path is not None else ""
    for row in _puppet_render_rows():
        if str(row.get("asset")) != asset or str(row.get("action")) != action:
            continue
        outputs = [str(x) for x in (row.get("outputs") or [])]
        if gif_stem and any(Path(x).stem == gif_stem for x in outputs):
            return row
        if view_name and (str(row.get("view_name")) == view_name or any(Path(x).name in {view_name, coverage_name} for x in outputs)):
            return row
    return None


def _asset_center_for_puppet_camera(case: dict[str, Any], raw_gt: ex.RawMeshSequence, row: dict[str, Any] | None) -> np.ndarray:
    asset_root = Path(str(case.get("asset_root") or ""))
    bbox_path = asset_root / "bounding_box.json"
    if bbox_path.exists():
        try:
            bbox = _read_json(bbox_path)
            mn = np.asarray(bbox["min"], dtype=float)
            mx = np.asarray(bbox["max"], dtype=float)
            return 0.5 * (mn + mx)
        except Exception:
            pass
    center, _radius = p2d._scene_bounds_from_raw(raw_gt)
    return np.asarray(center, dtype=float)


def _camera_from_distance(center: np.ndarray, azimuth_deg: float, elevation_deg: float, distance: float, fov_deg: float = 50.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    # evaluate_2d_part.project_points has a fixed 50 degree projection. Convert
    # cameras rendered with another fov into an equivalent distance under that
    # projection so the 2D point masks land on the same pixels.
    proj_fov = math.radians(50.0)
    src_fov = math.radians(float(fov_deg or 50.0))
    distance_equiv = float(distance) * math.tan(src_fov / 2.0) / math.tan(proj_fov / 2.0)
    eye = np.asarray(center, dtype=float) + distance_equiv * np.asarray(
        [math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)],
        dtype=float,
    )
    return eye, np.asarray(center, dtype=float), np.asarray([0.0, 0.0, 1.0], dtype=float)


def _camera_from_viewspec(center: np.ndarray, radius: float, view: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = gop.compute_camera(
        np.asarray(center, dtype=float),
        float(radius),
        azim_deg=float(view.get("azimuth_deg", 0.0)),
        elev_deg=float(view.get("elevation_deg", 20.0)),
    )
    eye, target, up = camera
    distance_scale = float(view.get("distance_scale", 1.0) or 1.0)
    # The motion loop's coverage adjustment computes distance_scale under the
    # same fixed-50-degree projection used by the part-mask rasterizer. The
    # viewspec still carries fov_deg for legacy renderers, but applying another
    # fov conversion here would make selected motion views too tight.
    eye = np.asarray(target, dtype=float) + (np.asarray(eye, dtype=float) - np.asarray(target, dtype=float)) * distance_scale
    return eye, np.asarray(target, dtype=float), np.asarray(up, dtype=float)


def _candidate_source_roots(case: dict[str, Any]) -> list[Path]:
    roots: list[Path] = []
    for group_name in ("variant_glbs", "variant_plans", "variants"):
        group = case.get(group_name)
        if not isinstance(group, dict):
            continue
        value = group.get("full_agent")
        if not value:
            continue
        path = Path(str(value))
        roots.append(path.parent)
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _source_viewspec_candidates(case: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for root in _candidate_source_roots(case):
        candidates.extend(
            [
                root / "loop" / "motion_viewspecs_selected.json",
                root / "loop" / "coverage" / "iter00" / "coverage_vlm_selected_viewspecs.json",
            ]
        )
        motion_root = root / "loop" / "motion"
        if motion_root.exists():
            candidates.extend(sorted(motion_root.glob("iter*/motion_viewspecs_peak_adjusted.json"), reverse=True))
            candidates.extend(sorted(motion_root.glob("iter*/motion_viewspecs.json"), reverse=True))
    seen: set[str] = set()
    out: list[Path] = []
    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _select_viewspec_row(spec: dict[str, Any], view: tuple[str, float, float]) -> dict[str, Any] | None:
    rows = [dict(v) for v in (spec.get("views") or []) if isinstance(v, dict)]
    if not rows:
        return None
    view_id = str(view[0])
    for row in rows:
        if str(row.get("id") or "") == view_id:
            return row
    want_az = float(view[1])
    want_el = float(view[2])

    def angular_distance(row: dict[str, Any]) -> float:
        az = float(row.get("azimuth_deg", want_az))
        el = float(row.get("elevation_deg", want_el))
        da = abs(((az - want_az + 180.0) % 360.0) - 180.0)
        return da + abs(el - want_el)

    return min(rows, key=angular_distance)


def _source_motion_camera(case: dict[str, Any], raw_gt: ex.RawMeshSequence, view: tuple[str, float, float]) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]] | None:
    center, radius = p2d._scene_bounds_from_raw(raw_gt)
    for path in _source_viewspec_candidates(case):
        if not path.exists():
            continue
        try:
            spec = _read_json(path)
        except Exception:
            continue
        if not isinstance(spec, dict):
            continue
        row = _select_viewspec_row(spec, view)
        if row is None:
            continue
        camera = _camera_from_viewspec(center, radius, row)
        return camera, {
            "mode": "source_motion_viewspec",
            "viewspec_path": str(path),
            "selected_view": row,
            "requested_view": [view[0], float(view[1]), float(view[2])],
        }
    return None


def _camera_for_selected_view(case: dict[str, Any], raw_gt: ex.RawMeshSequence, view: tuple[str, float, float], gif_path: Path | None = None) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    # The selected-view 2D benchmark is anchored to the actual Puppet input
    # image.  Prefer the Puppet render manifest camera so GT and all mesh
    # baselines are rendered from the same reference/coverage view as Puppet.
    row = _find_puppet_render_row(case, gif_path, view)
    if row is not None and row.get("distance") is not None:
        center = _asset_center_for_puppet_camera(case, raw_gt, row)
        camera = _camera_from_distance(
            center,
            float(row.get("azimuth_deg", view[1])),
            float(row.get("elevation_deg", view[2])),
            float(row["distance"]),
            float(row.get("fov_deg", 50.0)),
        )
        return camera, {"mode": "puppet_render_manifest", **row}
    source_camera = _source_motion_camera(case, raw_gt, view)
    if source_camera is not None:
        return source_camera
    center, radius = p2d._scene_bounds_from_raw(raw_gt)
    return gop.compute_camera(center, radius, azim_deg=float(view[1]), elev_deg=float(view[2])), {"mode": "computed_from_gt_bounds"}


def _camera_for_puppet_view(case: dict[str, Any], raw_gt: ex.RawMeshSequence, view: tuple[str, float, float], gif_path: Path | None = None) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    # PuppetMaster is a 2D baseline: it consumes the already-rendered selected
    # reference image and generates in that image's cropped coordinate system.
    # Do not switch Puppet scoring to the mesh motion-loop camera; reproduce the
    # original Puppet input view and crop instead.
    row = _find_puppet_render_row(case, gif_path, view)
    if row is not None and row.get("distance") is not None:
        center = _asset_center_for_puppet_camera(case, raw_gt, row)
        camera = _camera_from_distance(
            center,
            float(row.get("azimuth_deg", view[1])),
            float(row.get("elevation_deg", view[2])),
            float(row["distance"]),
            float(row.get("fov_deg", 50.0)),
        )
        return camera, {"mode": "puppet_render_manifest", **row}
    center, radius = p2d._scene_bounds_from_raw(raw_gt)
    return gop.compute_camera(center, radius, azim_deg=float(view[1]), elev_deg=float(view[2])), {"mode": "computed_from_gt_bounds_for_puppet"}


def _puppet_input_transform(case: dict[str, Any], gif_path: Path, canvas_shape: tuple[int, int], scale_factor: float) -> dict[str, Any]:
    source = _find_puppet_source_image(case, gif_path)
    if source is None:
        return {"mode": "eval_crop_fallback", "source_image": None}
    image = Image.open(source).convert("RGB")
    if image.width == 256 and image.height == 256 and (source.stem.endswith("_preprocessed") or source.name == "image_process.png"):
        return {
            "mode": "preprocessed_256",
            "source_image": str(source),
            "source_size": [int(image.width), int(image.height)],
            "source_foreground_bbox": list(_foreground_bbox_from_rgb(image) or (0, 0, image.width - 1, image.height - 1)),
        }
    # Most no-segmentation Puppet-Master runs resize the full input image to
    # 256x256 instead of applying the foreground-centered square crop.
    if "without_segmentation" in gif_path.name:
        return {
            "mode": "full_resize",
            "source_image": str(source),
            "source_size": [int(image.width), int(image.height)],
        }
    spec = _crop_spec_from_rgb_foreground(image, float(scale_factor))
    if spec is None:
        return {"mode": "eval_crop_fallback", "source_image": str(source), "source_size": [int(image.width), int(image.height)]}
    canvas_h, canvas_w = int(canvas_shape[0]), int(canvas_shape[1])
    if image.width != canvas_w or image.height != canvas_h:
        sx = float(canvas_w) / max(float(image.width), 1.0)
        sy = float(canvas_h) / max(float(image.height), 1.0)
        s = 0.5 * (sx + sy)
        spec = {
            "x0": int(round(float(spec["x0"]) * sx)),
            "y0": int(round(float(spec["y0"]) * sy)),
            "side": max(1, int(round(float(spec["side"]) * s))),
        }
    return {
        "mode": "square_crop",
        "source_image": str(source),
        "source_size": [int(image.width), int(image.height)],
        "crop_spec": spec,
        "canvas_shape": [int(canvas_shape[0]), int(canvas_shape[1])],
    }


def _stabilize_puppet_transform(transform: dict[str, Any], owner_crop_spec: dict[str, int], canvas_shape: tuple[int, int]) -> dict[str, Any]:
    if str(transform.get("mode") or "") != "square_crop" or not isinstance(transform.get("crop_spec"), dict):
        return transform
    side = float(transform["crop_spec"].get("side", 0))
    max_dim = float(max(int(canvas_shape[0]), int(canvas_shape[1])))
    # Rendered reference images often have a smooth off-white background. A
    # pure RGB foreground threshold can then select the whole canvas, while
    # Puppet-Master's actual segmentation crop was based on SAM. When that
    # happens, use the GT first-frame owner crop: it is the same selected view
    # and reproduces Puppet's object-centered preprocessing without letting
    # background gradients dominate the inverse transform.
    if side > 1.6 * max_dim:
        out = dict(transform)
        out["mode"] = "square_crop"
        out["crop_spec"] = dict(owner_crop_spec)
        out["fallback"] = "gt_first_frame_owner_crop"
        return out
    return transform


def _owner_from_puppet_space_to_eval_crop(
    owner: np.ndarray,
    transform: dict[str, Any],
    eval_crop_spec: dict[str, int],
    canvas_shape: tuple[int, int],
    out_res: tuple[int, int],
) -> np.ndarray:
    mode = str(transform.get("mode") or "")
    if mode == "preprocessed_256":
        inverse_align = transform.get("owner_inverse_align_spec") if isinstance(transform.get("owner_inverse_align_spec"), dict) else None
        return _apply_bbox_align_owner(np.asarray(owner, dtype=np.int32), inverse_align)
    if mode == "square_crop" and isinstance(transform.get("crop_spec"), dict):
        full = _uncrop_resized_owner(owner, transform["crop_spec"], canvas_shape)
        return _crop_resize_owner(full, eval_crop_spec, out_res)
    if mode == "full_resize":
        full = _resize_owner_to_canvas(owner, canvas_shape)
        return _crop_resize_owner(full, eval_crop_spec, out_res)
    return np.asarray(owner, dtype=np.int32)


def _rgb_from_puppet_space_to_eval_crop(
    image: np.ndarray,
    transform: dict[str, Any],
    eval_crop_spec: dict[str, int],
    canvas_shape: tuple[int, int],
    out_res: tuple[int, int],
) -> np.ndarray:
    mode = str(transform.get("mode") or "")
    if mode == "preprocessed_256":
        inverse_align = transform.get("owner_inverse_align_spec") if isinstance(transform.get("owner_inverse_align_spec"), dict) else None
        return _apply_bbox_align_rgb(np.asarray(image, dtype=np.uint8), inverse_align)
    if mode == "square_crop" and isinstance(transform.get("crop_spec"), dict):
        full = _uncrop_resized_rgb(image, transform["crop_spec"], canvas_shape)
        return _crop_resize_rgb(full, eval_crop_spec, out_res)
    if mode == "full_resize":
        h, w = int(canvas_shape[0]), int(canvas_shape[1])
        full = cv2.resize(np.asarray(image, dtype=np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
        return _crop_resize_rgb(full, eval_crop_spec, out_res)
    return np.asarray(image, dtype=np.uint8)


def _crop_spec_from_owner(owner: np.ndarray, scale_factor: float) -> dict[str, int]:
    mask = owner >= 0
    ys, xs = np.where(mask)
    if xs.size == 0:
        h, w = owner.shape
        return {"x0": 0, "y0": 0, "side": max(w, h)}
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = max(1, int(max(x1 - x0 + 1, y1 - y0 + 1) * float(scale_factor)))
    return {"x0": int(cx - side // 2), "y0": int(cy - side // 2), "side": int(side)}


def _foreground_bbox_from_rgb(image: Image.Image, threshold: int = 12) -> tuple[int, int, int, int] | None:
    arr = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate(
        [arr[:4].reshape(-1, 3), arr[-4:].reshape(-1, 3), arr[:, :4].reshape(-1, 3), arr[:, -4:].reshape(-1, 3)],
        axis=0,
    )
    bg = np.median(border, axis=0)
    dist = np.abs(arr - bg).max(axis=2)
    mask = (dist > int(threshold)) | (arr.min(axis=2) < 242)
    # Blender/PNG reference views can contain a one-pixel dark frame at the
    # image boundary. Puppet preprocessing should crop the object, not that
    # frame; otherwise the inferred square crop becomes the whole canvas.
    pad = min(8, max(0, min(mask.shape[:2]) // 16))
    if pad > 0:
        mask[:pad, :] = False
        mask[-pad:, :] = False
        mask[:, :pad] = False
        mask[:, -pad:] = False
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _crop_spec_from_rgb_foreground(image: Image.Image, scale_factor: float) -> dict[str, int] | None:
    bbox = _foreground_bbox_from_rgb(image)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    side = max(1, int(max(x1 - x0 + 1, y1 - y0 + 1) * float(scale_factor)))
    return {"x0": int(cx - side // 2), "y0": int(cy - side // 2), "side": int(side)}


def _crop_resize_owner(owner: np.ndarray, spec: dict[str, int], out_res: tuple[int, int]) -> np.ndarray:
    h, w = owner.shape
    side = int(spec["side"])
    x0 = int(spec["x0"])
    y0 = int(spec["y0"])
    canvas = np.full((side, side), -1, dtype=np.int32)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x0 + side)
    sy1 = min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = owner[sy0:sy1, sx0:sx1]
    # Use OpenCV here instead of PIL's 16-bit nearest resize. PIL is usually
    # fine, but these workers load native FBX/mesh libraries and PIL's 16-bit
    # path has shown intermittent native crashes in long AAM batches.
    resized = cv2.resize(canvas.astype(np.int32), tuple(out_res), interpolation=cv2.INTER_NEAREST)
    return np.asarray(resized, dtype=np.int32)


def _uncrop_resized_owner(owner: np.ndarray, spec: dict[str, int], canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = int(canvas_shape[0]), int(canvas_shape[1])
    side = int(spec["side"])
    x0 = int(spec["x0"])
    y0 = int(spec["y0"])
    full = np.full((h, w), -1, dtype=np.int32)
    square = cv2.resize(np.asarray(owner, dtype=np.int32), (side, side), interpolation=cv2.INTER_NEAREST)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x0 + side)
    sy1 = min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        ox0 = sx0 - x0
        oy0 = sy0 - y0
        full[sy0:sy1, sx0:sx1] = square[oy0 : oy0 + (sy1 - sy0), ox0 : ox0 + (sx1 - sx0)]
    return full


def _resize_owner_to_canvas(owner: np.ndarray, canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = int(canvas_shape[0]), int(canvas_shape[1])
    return cv2.resize(np.asarray(owner, dtype=np.int32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)


def _crop_resize_rgb(image: np.ndarray, spec: dict[str, int], out_res: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    side = int(spec["side"])
    x0 = int(spec["x0"])
    y0 = int(spec["y0"])
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x0 + side)
    sy1 = min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, tuple(out_res), interpolation=cv2.INTER_LINEAR).astype(np.uint8)


def _uncrop_resized_rgb(image: np.ndarray, spec: dict[str, int], canvas_shape: tuple[int, int]) -> np.ndarray:
    h, w = int(canvas_shape[0]), int(canvas_shape[1])
    side = int(spec["side"])
    x0 = int(spec["x0"])
    y0 = int(spec["y0"])
    full = np.full((h, w, 3), 255, dtype=np.uint8)
    square = cv2.resize(np.asarray(image, dtype=np.uint8), (side, side), interpolation=cv2.INTER_LINEAR)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x0 + side)
    sy1 = min(h, y0 + side)
    if sx1 > sx0 and sy1 > sy0:
        ox0 = sx0 - x0
        oy0 = sy0 - y0
        full[sy0:sy1, sx0:sx1] = square[oy0 : oy0 + (sy1 - sy0), ox0 : ox0 + (sx1 - sx0)]
    return full


def _owner_to_color(owner: np.ndarray, link_names: list[str]) -> np.ndarray:
    h, w = owner.shape
    image = np.full((h, w, 3), 255, dtype=np.uint8)
    for idx, link in enumerate(link_names):
        mask = owner == int(idx)
        if np.any(mask):
            image[mask] = np.asarray(p2d._deterministic_color(link), dtype=np.uint8)
    return image


def _owner_bbox(owner: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(owner) >= 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _bbox_align_spec(src_bbox, dst_bbox) -> dict[str, Any] | None:
    if src_bbox is None or dst_bbox is None:
        return None
    sx0, sy0, sx1, sy1 = [float(x) for x in src_bbox]
    dx0, dy0, dx1, dy1 = [float(x) for x in dst_bbox]
    src_w = max(1.0, sx1 - sx0)
    src_h = max(1.0, sy1 - sy0)
    dst_w = max(1.0, dx1 - dx0)
    dst_h = max(1.0, dy1 - dy0)
    scale = min(dst_w / src_w, dst_h / src_h)
    paste_x = int(round(0.5 * (dx0 + dx1) - 0.5 * (sx0 + sx1) * scale))
    paste_y = int(round(0.5 * (dy0 + dy1) - 0.5 * (sy0 + sy1) * scale))
    return {
        "mode": "initial_bbox_scale_translate",
        "scale": float(scale),
        "paste_x": int(paste_x),
        "paste_y": int(paste_y),
        "src_bbox": [int(round(x)) for x in src_bbox],
        "dst_bbox": [int(round(x)) for x in dst_bbox],
    }


def _inverse_bbox_align_spec(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if not spec:
        return None
    src_bbox = spec.get("dst_bbox")
    dst_bbox = spec.get("src_bbox")
    if not (isinstance(src_bbox, list) and isinstance(dst_bbox, list) and len(src_bbox) == 4 and len(dst_bbox) == 4):
        return None
    return _bbox_align_spec(tuple(int(x) for x in src_bbox), tuple(int(x) for x in dst_bbox))


def _apply_bbox_align_owner(owner: np.ndarray, spec: dict[str, Any] | None) -> np.ndarray:
    arr = np.asarray(owner, dtype=np.int32)
    if not spec:
        return arr
    h, w = arr.shape
    scale = float(spec.get("scale", 1.0))
    if scale <= 0:
        return arr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    out = np.full((h, w), -1, dtype=np.int32)
    px = int(spec.get("paste_x", 0))
    py = int(spec.get("paste_y", 0))
    sx0 = max(0, -px)
    sy0 = max(0, -py)
    sx1 = min(new_w, w - px)
    sy1 = min(new_h, h - py)
    if sx1 > sx0 and sy1 > sy0:
        out[max(0, py) : max(0, py) + (sy1 - sy0), max(0, px) : max(0, px) + (sx1 - sx0)] = resized[sy0:sy1, sx0:sx1]
    return out


def _apply_bbox_align_rgb(image: np.ndarray, spec: dict[str, Any] | None) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    if not spec:
        return arr
    h, w = arr.shape[:2]
    scale = float(spec.get("scale", 1.0))
    if scale <= 0:
        return arr
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    px = int(spec.get("paste_x", 0))
    py = int(spec.get("paste_y", 0))
    sx0 = max(0, -px)
    sy0 = max(0, -py)
    sx1 = min(new_w, w - px)
    sy1 = min(new_h, h - py)
    if sx1 > sx0 and sy1 > sy0:
        out[max(0, py) : max(0, py) + (sy1 - sy0), max(0, px) : max(0, px) + (sx1 - sx0)] = resized[sy0:sy1, sx0:sx1]
    return out


def _render_owner_frame(
    raw: ex.RawMeshSequence,
    frame_idx: int,
    link_components: dict[str, list[int]],
    component_faces: list[np.ndarray],
    recipes: dict[str, tuple[np.ndarray, np.ndarray]],
    link_names: list[str],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    render_res: tuple[int, int],
    point_radius: int,
    crop_spec: dict[str, int],
    out_res: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = p2d._points_by_link_for_frame(raw, int(frame_idx), recipes)
    owner, _img = p2d._render_owner_points(points, link_names, camera, render_res, int(point_radius))
    cropped = _crop_resize_owner(owner, crop_spec, out_res)
    return cropped, _owner_to_color(cropped, link_names)


def _owner_to_puppet_space(
    owner: np.ndarray,
    puppet_transform: dict[str, Any],
    fallback_crop_spec: dict[str, int],
    out_res: tuple[int, int],
) -> np.ndarray:
    mode = str(puppet_transform.get("mode") or "")
    if mode == "preprocessed_256":
        base = _crop_resize_owner(owner, fallback_crop_spec, out_res)
        align = puppet_transform.get("owner_align_spec") if isinstance(puppet_transform.get("owner_align_spec"), dict) else None
        return _apply_bbox_align_owner(base, align)
    if mode == "square_crop" and isinstance(puppet_transform.get("crop_spec"), dict):
        return _crop_resize_owner(owner, puppet_transform["crop_spec"], out_res)
    if mode == "full_resize":
        return cv2.resize(np.asarray(owner, dtype=np.int32), out_res, interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return _crop_resize_owner(owner, fallback_crop_spec, out_res)


def _puppet_source_foreground_mask(puppet_transform: dict[str, Any], out_res: tuple[int, int]) -> np.ndarray | None:
    if str(puppet_transform.get("mode") or "") != "preprocessed_256":
        return None
    source = puppet_transform.get("source_image") if isinstance(puppet_transform, dict) else None
    if not source:
        return None
    path = Path(str(source))
    if not path.exists():
        return None
    try:
        image = Image.open(path).convert("RGB").resize(tuple(out_res), Image.Resampling.BILINEAR)
    except Exception:
        return None
    return _foreground_mask(np.asarray(image, dtype=np.uint8))


def _puppet_preprocess_alignment(first_owner_common: np.ndarray, puppet_transform: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    src_bbox = _owner_bbox(first_owner_common)
    dst_raw = puppet_transform.get("source_foreground_bbox") if isinstance(puppet_transform, dict) else None
    if not (isinstance(dst_raw, list) and len(dst_raw) == 4):
        return None, None
    forward = _bbox_align_spec(src_bbox, tuple(int(x) for x in dst_raw))
    inverse = _inverse_bbox_align_spec(forward)
    return forward, inverse


def _scale_translate_alignment_tf(info: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    src_bounds = info.get("src_bounds")
    dst_bounds = info.get("dst_bounds")
    if not src_bounds or not dst_bounds:
        return np.eye(4, dtype=np.float32), {
            "mode": "none",
            "perm": [0, 1, 2],
            "signs": [1.0, 1.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
        }
    src_min, src_max = np.asarray(src_bounds[0], dtype=np.float32), np.asarray(src_bounds[1], dtype=np.float32)
    dst_min, dst_max = np.asarray(dst_bounds[0], dtype=np.float32), np.asarray(dst_bounds[1], dtype=np.float32)
    src_center = 0.5 * (src_min + src_max)
    dst_center = 0.5 * (dst_min + dst_max)
    src_extent = np.maximum(src_max - src_min, 1.0e-6)
    dst_extent = np.maximum(dst_max - dst_min, 1.0e-6)
    scale = float(np.linalg.norm(dst_extent) / max(float(np.linalg.norm(src_extent)), 1.0e-6))
    tf = np.eye(4, dtype=np.float32)
    tf[:3, :3] *= scale
    tf[:3, 3] = dst_center - src_center * scale
    align_info = dict(info)
    align_info.update(
        {
            "mode": "scale_translate_3d",
            "perm": [0, 1, 2],
            "signs": [1.0, 1.0, 1.0],
            "scale": [scale, scale, scale],
        }
    )
    return tf, align_info


def _alignment_tf_from_matching(matching: dict[str, Any], mesh_alignment: str = "eval_3d") -> tuple[np.ndarray, dict[str, Any], str]:
    info = dict(matching.get("alignment") or {})
    mesh_alignment = str(mesh_alignment or "eval_3d")
    mode = str(info.get("mode") or "none")
    info.setdefault("mode", mode)
    info.setdefault("perm", [0, 1, 2])
    info.setdefault("signs", [1.0, 1.0, 1.0])
    info.setdefault("scale", [1.0, 1.0, 1.0])
    motion_scale_mode = str(matching.get("effective_motion_scale_mode") or matching.get("motion_scale_mode") or "scale_motion")
    if mesh_alignment == "none":
        return np.eye(4, dtype=np.float32), {"mode": "none", "perm": [0, 1, 2], "signs": [1.0, 1.0, 1.0], "scale": [1.0, 1.0, 1.0]}, "scale_motion"
    if mesh_alignment == "scale_translate_3d":
        tf, align_info = _scale_translate_alignment_tf(info)
        return tf, align_info, motion_scale_mode
    tf = np.eye(4, dtype=np.float32)
    if mode == "none":
        return tf, info, motion_scale_mode
    src_bounds = info.get("src_bounds")
    dst_bounds = info.get("dst_bounds")
    if not src_bounds or not dst_bounds:
        return tf, {"mode": "none", "perm": [0, 1, 2], "signs": [1.0, 1.0, 1.0], "scale": [1.0, 1.0, 1.0]}, "scale_motion"
    rot, scale = ex._alignment_rotation_and_scale(info)
    linear = np.diag(scale) @ rot
    src_min, src_max = np.asarray(src_bounds[0], dtype=np.float32), np.asarray(src_bounds[1], dtype=np.float32)
    dst_min, dst_max = np.asarray(dst_bounds[0], dtype=np.float32), np.asarray(dst_bounds[1], dtype=np.float32)
    src_center = 0.5 * (src_min + src_max)
    dst_center = 0.5 * (dst_min + dst_max)
    tf[:3, :3] = linear
    tf[:3, 3] = dst_center - src_center @ linear.T
    return tf, info, motion_scale_mode


def _aligned_raw_from_matching(raw: ex.RawMeshSequence, matching: dict[str, Any], mesh_alignment: str = "eval_3d") -> ex.RawMeshSequence:
    align_tf, align_info, motion_scale_mode = _alignment_tf_from_matching(matching, mesh_alignment)
    if str(align_info.get("mode")) == "none":
        return raw
    vertices_by_frame = [
        ex._transform_external_frame(np.asarray(verts, dtype=np.float32), align_tf, align_info, motion_scale_mode)
        for verts in raw.vertices_by_frame
    ]
    return ex.RawMeshSequence(
        path=raw.path,
        vertices_by_frame=vertices_by_frame,
        faces=raw.faces,
        source=f"{raw.source}:aligned",
        times_s=raw.times_s,
        component_vertex_indices=raw.component_vertex_indices,
        component_face_indices=raw.component_face_indices,
        component_names=raw.component_names,
    )


def _has_drag_overlay(frame: np.ndarray) -> bool:
    yellow = (frame[..., 0] > 170) & (frame[..., 1] > 145) & (frame[..., 2] < 130)
    return int(np.count_nonzero(yellow)) > 20


def _load_gif_frames(path: Path, out_res: tuple[int, int]) -> list[np.ndarray]:
    im = Image.open(path)
    frames = []
    for frame in ImageSequence.Iterator(im):
        rgb = frame.convert("RGB").resize(tuple(out_res), Image.Resampling.BILINEAR)
        frames.append(np.asarray(rgb, dtype=np.uint8))
    # Puppet-Master prepends drag-overlay frames to the exported GIF. Those
    # frames are diagnostic UI, not generated animation, and would corrupt
    # optical-flow propagation and endpoint matching.
    first_clean = 0
    while first_clean < len(frames) and _has_drag_overlay(frames[first_clean]):
        first_clean += 1
    return frames[first_clean:] or frames


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [image[:4].reshape(-1, 3), image[-4:].reshape(-1, 3), image[:, :4].reshape(-1, 3), image[:, -4:].reshape(-1, 3)],
        axis=0,
    ).astype(np.int16)
    bg = np.median(border, axis=0)
    dist = np.abs(image.astype(np.int16) - bg).max(axis=2)
    return (dist > 12) | (image.min(axis=2) < 242)


def _label_color_references(owner: np.ndarray, frame: np.ndarray, min_pixels: int = 12) -> dict[int, tuple[np.ndarray, float]]:
    arr = np.asarray(owner, dtype=np.int32)
    lab = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    refs: dict[int, tuple[np.ndarray, float]] = {}
    for label in sorted(int(x) for x in np.unique(arr) if int(x) >= 0):
        mask = arr == label
        if int(np.count_nonzero(mask)) < int(min_pixels):
            continue
        pix = lab[mask]
        med = np.median(pix, axis=0).astype(np.float32)
        d = np.linalg.norm(pix - med[None, :], axis=1)
        # The generated Puppet frames are soft and color-shifted, so keep a
        # loose threshold, but still reject obvious label drift such as wheel
        # labels landing on a red trolley body.
        threshold = float(np.clip(np.percentile(d, 90) + 28.0, 42.0, 90.0))
        refs[label] = (med, threshold)
    return refs


def _filter_owner_by_label_color(owner: np.ndarray, frame: np.ndarray, refs: dict[int, tuple[np.ndarray, float]]) -> np.ndarray:
    if not refs:
        return np.asarray(owner, dtype=np.int32)
    arr = np.asarray(owner, dtype=np.int32).copy()
    lab = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    for label, (med, threshold) in refs.items():
        mask = arr == int(label)
        if not np.any(mask):
            continue
        d = np.linalg.norm(lab[mask] - med[None, :], axis=1)
        bad = np.zeros(mask.shape, dtype=bool)
        bad[mask] = d > float(threshold)
        arr[bad] = -1
    return arr


def _propagate_owner_with_flow(initial_owner: np.ndarray, frames: list[np.ndarray]) -> list[np.ndarray]:
    owners = [initial_owner.astype(np.int32)]
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    for idx in range(1, len(frames)):
        cur_gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
        # Backward flow: destination pixel -> previous-frame coordinate.
        flow = cv2.calcOpticalFlowFarneback(cur_gray, prev_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        h, w = cur_gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        warped = cv2.remap(
            owners[-1].astype(np.float32),
            grid_x + flow[..., 0],
            grid_y + flow[..., 1],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        ).astype(np.int32)
        warped[~_foreground_mask(frames[idx])] = -1
        owners.append(warped)
        prev_gray = cur_gray
    return owners


def _silhouette_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma = a >= 0
    mb = b >= 0
    union = np.count_nonzero(ma | mb)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(ma & mb) / union)


def _overlap_flags_from_states(states: list[dict[str, Any]]) -> list[bool]:
    return [bool(st.get("overlap_allowed_from_previous")) for st in states]


def _apply_overlap_windows_to_indices(scores: np.ndarray, indices: list[int], overlap_prev: list[bool]) -> list[int]:
    if scores.size == 0 or not indices:
        return indices
    m = int(scores.shape[1])
    out = [min(m - 1, max(0, int(x))) for x in indices]
    for i in range(min(len(out), len(overlap_prev))):
        if not overlap_prev[i]:
            continue
        lo = out[i - 2] + 1 if i >= 2 else 0
        hi = out[i + 1] - 1 if i + 1 < len(out) else m - 1
        lo = max(0, min(m - 1, int(lo)))
        hi = max(0, min(m - 1, int(hi)))
        if lo > hi:
            lo, hi = 0, m - 1
        window = np.asarray(scores[i, lo : hi + 1], dtype=np.float32)
        if window.size:
            out[i] = int(lo + int(np.argmax(window)))
    return out


def _ordered_match(gt_owners: list[np.ndarray], pred_owners: list[np.ndarray], overlap_prev: list[bool] | None = None) -> list[int]:
    n = len(gt_owners)
    m = len(pred_owners)
    if n == 0 or m == 0:
        return []
    if n > m:
        return [min(m - 1, round(i * (m - 1) / max(1, n - 1))) for i in range(n)]
    scores = np.asarray([[_silhouette_iou(g, p) for p in pred_owners] for g in gt_owners], dtype=np.float32)
    dp = np.full((n, m), -1.0e9, dtype=np.float32)
    prev = np.full((n, m), -1, dtype=np.int32)
    dp[0] = scores[0]
    for i in range(1, n):
        best_val = -1.0e9
        best_j = -1
        for j in range(m):
            if j > 0 and dp[i - 1, j - 1] > best_val:
                best_val = dp[i - 1, j - 1]
                best_j = j - 1
            if best_j >= 0:
                dp[i, j] = best_val + scores[i, j]
                prev[i, j] = best_j
    j = int(np.argmax(dp[-1]))
    out = [j]
    for i in range(n - 1, 0, -1):
        j = int(prev[i, j])
        out.append(j)
    out = list(reversed(out))
    return _apply_overlap_windows_to_indices(scores, out, overlap_prev or [])


def _strict_increasing_indices(values: list[int], m: int) -> list[int]:
    if not values or m <= 0:
        return []
    n = len(values)
    if n > m:
        return [min(m - 1, max(0, int(v))) for v in values]
    out = [min(m - 1, max(0, int(v))) for v in values]
    for i in range(1, n):
        out[i] = max(out[i], out[i - 1] + 1)
    if out[-1] >= m:
        out[-1] = m - 1
        for i in range(n - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - 1)
    return [min(m - 1, max(0, int(v))) for v in out]


def _temporal_match_from_states(
    states: list[dict[str, Any]],
    num_pred_frames: int,
    pred_owners: list[np.ndarray] | None = None,
    gt_owners: list[np.ndarray] | None = None,
) -> list[int]:
    n = len(states)
    m = int(num_pred_frames)
    if n == 0 or m <= 0:
        return []
    if n == 1:
        return [0]
    times = [s.get("gt_time_s") for s in states]
    if all(t is not None for t in times):
        vals = [float(t) for t in times]
        t0 = vals[0]
        t1 = vals[-1]
        if t1 > t0 + 1.0e-9:
            raw = [int(round((t - t0) / (t1 - t0) * (m - 1))) for t in vals]
            raw[0] = 0
            raw[-1] = m - 1
            out = _strict_increasing_indices(raw, m)
            if pred_owners is not None and gt_owners is not None and len(pred_owners) == m and len(gt_owners) == n:
                scores = np.asarray([[_silhouette_iou(g, p) for p in pred_owners] for g in gt_owners], dtype=np.float32)
                out = _apply_overlap_windows_to_indices(scores, out, _overlap_flags_from_states(states))
            return out
    raw = [int(round(i * (m - 1) / max(1, n - 1))) for i in range(n)]
    out = _strict_increasing_indices(raw, m)
    if pred_owners is not None and gt_owners is not None and len(pred_owners) == m and len(gt_owners) == n:
        scores = np.asarray([[_silhouette_iou(g, p) for p in pred_owners] for g in gt_owners], dtype=np.float32)
        out = _apply_overlap_windows_to_indices(scores, out, _overlap_flags_from_states(states))
    return out


def _score_link(
    gt_owner: np.ndarray,
    pred_owner: np.ndarray,
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    link_idx: int,
    min_visible_px: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    score = p2d._score_link(gt_owner, pred_owner, gt_img, pred_img, link_idx, min_visible_px)
    if score is None:
        return None
    gt_mask = gt_owner == int(link_idx)
    pred_mask = pred_owner == int(link_idx)
    union_mask = gt_mask | pred_mask
    score["P_MaskIoU"] = score.get("P_IoU")
    score["P_BoundaryF1"] = _boundary_f1(gt_mask, pred_mask, int(args.boundary_tolerance_px))
    score["P_ContourCD"] = _contour_chamfer_distance(gt_mask, pred_mask, union_mask, float(args.contour_scale_floor_px))
    if bool(getattr(args, "compute_lpips", False)):
        score["P_LPIPS"] = _binary_mask_lpips(gt_mask, pred_mask, union_mask, args)
    return score


def _case_mesh_metrics(
    case: dict[str, Any],
    metric_row: dict[str, str],
    matched: dict[str, Any],
    matching_dir: Path,
    gt_matching_dir: Path,
    variant: str,
    view: tuple[str, float, float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cls, asset, action = _case_key(case)
    raw_gt_anim = ex._load_mesh_sequence(Path(case["gt_glb"]))
    raw_pred = ex._load_mesh_sequence(Path(metric_row["prediction_file"]))
    matching = _load_matching_or_none(matching_dir / _matching_filename(cls, asset, action))
    gt_matching = _load_matching_or_none(gt_matching_dir / _matching_filename(cls, asset, action))
    if matching is not None:
        raw_pred = _aligned_raw_from_matching(raw_pred, matching, str(args.mesh_alignment))
    pred_component_faces = p2d._component_face_groups(raw_pred, int(args.min_component_faces))
    if matching is not None and gt_matching is not None:
        gt_link_components_anim = p2d._target_link_components_from_matching(gt_matching)
        pred_link_components = p2d._link_components_from_matching(matching)
    else:
        asset_root = _resolve_project_relative_path(case.get("asset_root"), case)
        if asset_root is None or not asset_root.exists():
            raise FileNotFoundError(f"Missing asset_root for fallback component mapping: {case.get('asset_root')}")
        asset_geom = p2d.ev.load_asset_geometry(asset_root, 128, 0.01)
        gt_link_components_anim = p2d._link_components_from_gt(Path(case["gt_glb"]), asset_geom, raw_gt_anim)
        pred_link_components = p2d._link_components_from_gt(Path(metric_row["prediction_file"]), asset_geom, raw_pred)
    states = p2d._selected_states(matched, bool(args.include_terminal))
    if int(args.max_states) > 0:
        states = states[: int(args.max_states)]

    static_cache: dict[str, tuple[ex.RawMeshSequence, list[np.ndarray], dict[str, list[int]], dict[str, tuple[np.ndarray, np.ndarray]]]] = {}

    def static_path_for_state(state: dict[str, Any]) -> Path | None:
        raw_path = state.get("gt_static_glb")
        if not raw_path:
            return None
        return _resolve_project_relative_path(raw_path, case)

    first_static_path = next((static_path_for_state(st) for st in states if static_path_for_state(st) is not None), None)
    if first_static_path is not None:
        first_raw_gt = ex._load_mesh_sequence(first_static_path)
        first_gt_link_components = p2d._link_components_from_static_names(first_raw_gt, set(gt_link_components_anim) | set(pred_link_components))
        raw_for_camera = first_raw_gt
        link_names = sorted(ln for ln in first_gt_link_components if ln in pred_link_components)
    else:
        raw_for_camera = raw_gt_anim
        link_names = sorted(ln for ln in gt_link_components_anim if ln in pred_link_components)
    if not link_names:
        raise ValueError("No common matched links")
    camera, camera_info = _camera_for_selected_view(case, raw_for_camera, view)

    # Use the same sampling seed for GT and prediction. For full_agent the
    # topology is often identical to GT; different random point samples would
    # create artificial mask disagreement even when geometry overlaps.
    sample_seed = f"final2d:selected:{case['case_id']}"
    gt_component_faces_anim = p2d._component_face_groups(raw_gt_anim, int(args.min_component_faces))
    gt_recipes_anim = p2d._sample_link_recipes(
        raw_gt_anim,
        {ln: gt_link_components_anim[ln] for ln in link_names if ln in gt_link_components_anim},
        gt_component_faces_anim,
        int(args.points_per_link),
        f"{sample_seed}:animated",
    )
    pred_recipes = p2d._sample_link_recipes(raw_pred, {ln: pred_link_components[ln] for ln in link_names}, pred_component_faces, int(args.points_per_link), sample_seed)

    def gt_render_inputs(state: dict[str, Any]) -> tuple[ex.RawMeshSequence, int, dict[str, list[int]], list[np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
        static_path = static_path_for_state(state)
        if static_path is None:
            return raw_gt_anim, int(state["gt_frame_index"]), gt_link_components_anim, gt_component_faces_anim, gt_recipes_anim
        key = str(static_path)
        if key not in static_cache:
            raw = ex._load_mesh_sequence(static_path)
            comp_faces = p2d._component_face_groups(raw, int(args.min_component_faces))
            link_components = p2d._link_components_from_static_names(raw, set(gt_link_components_anim) | set(link_names))
            recipes = p2d._sample_link_recipes(
                raw,
                {ln: link_components[ln] for ln in link_names if ln in link_components},
                comp_faces,
                int(args.points_per_link),
                f"{sample_seed}:{static_path.name}",
            )
            static_cache[key] = (raw, comp_faces, link_components, recipes)
        raw, comp_faces, link_components, recipes = static_cache[key]
        return raw, 0, link_components, comp_faces, recipes

    first_state = states[0] if states else {"gt_frame_index": 0}
    first_raw, first_frame_idx, _first_components, _first_faces, first_recipes = gt_render_inputs(first_state)
    first_owner, _ = p2d._render_owner_points(
        p2d._points_by_link_for_frame(first_raw, first_frame_idx, first_recipes),
        link_names,
        camera,
        tuple(args.render_resolution),
        int(args.render_point_radius),
    )
    crop_spec = _crop_spec_from_owner(first_owner, float(args.crop_scale))

    view_id = str(view[0])
    per_link_rows: list[dict[str, Any]] = []
    per_state_rows: list[dict[str, Any]] = []
    for state in states:
        gt_raw, gt_frame_idx, gt_components, gt_faces, gt_recipes = gt_render_inputs(state)
        gt_owner, gt_img = _render_owner_frame(gt_raw, gt_frame_idx, gt_components, gt_faces, gt_recipes, link_names, camera, tuple(args.render_resolution), int(args.render_point_radius), crop_spec, tuple(args.resolution))
        pred_owner, pred_img = _render_owner_frame(raw_pred, int(state["pred_frame_index"]), pred_link_components, pred_component_faces, pred_recipes, link_names, camera, tuple(args.render_resolution), int(args.render_point_radius), crop_spec, tuple(args.resolution))
        scores = []
        for li, link in enumerate(link_names):
            score = _score_link(gt_owner, pred_owner, gt_img, pred_img, li, int(args.min_visible_px), args)
            if score is None:
                continue
            row = {
                "case_id": case["case_id"],
                "class": cls,
                "asset_name": asset,
                "action_name": action,
                "variant": variant,
                "view_id": view_id,
                "phase_id": state["phase_id"],
                "state_index": state["state_index"],
                "gt_frame_index": state["gt_frame_index"],
                "pred_frame_index": state["pred_frame_index"],
                "link": link,
                **score,
            }
            per_link_rows.append(row)
            scores.append(score)
        per_state_rows.append(
            {
                "case_id": case["case_id"],
                "class": cls,
                "asset_name": asset,
                "action_name": action,
                "variant": variant,
                "view_id": view_id,
                "phase_id": state["phase_id"],
                "state_index": state["state_index"],
                "gt_frame_index": state["gt_frame_index"],
                "pred_frame_index": state["pred_frame_index"],
                "num_scored_parts": len(scores),
                "P_MaskIoU": _mean([s.get("P_MaskIoU") for s in scores]),
                "P_BoundaryF1": _mean([s.get("P_BoundaryF1") for s in scores]),
                "P_ContourCD": _mean([s.get("P_ContourCD") for s in scores]),
                "P_IoU": _mean([s.get("P_IoU") for s in scores]),
                "P_PSNR": _mean([s.get("P_PSNR") for s in scores]),
                "P_SSIM": _mean([s.get("P_SSIM") for s in scores]),
                "P_LPIPS": _mean([s.get("P_LPIPS") for s in scores]),
            }
        )
    per_case = {
        "case_id": case["case_id"],
        "class": cls,
        "asset_name": asset,
        "action_name": action,
        "variant": variant,
        "num_states": len(states),
        "num_views": 1,
        "num_links": len(link_names),
        "P_MaskIoU": _mean([r.get("P_MaskIoU") for r in per_state_rows]),
        "P_BoundaryF1": _mean([r.get("P_BoundaryF1") for r in per_state_rows]),
        "P_ContourCD": _mean([r.get("P_ContourCD") for r in per_state_rows]),
        "P_IoU": _mean([r.get("P_IoU") for r in per_state_rows]),
        "P_PSNR": _mean([r.get("P_PSNR") for r in per_state_rows]),
        "P_SSIM": _mean([r.get("P_SSIM") for r in per_state_rows]),
        "P_LPIPS": _mean([r.get("P_LPIPS") for r in per_state_rows]),
    }
    return per_link_rows, per_state_rows, per_case


def _case_puppet_metrics(
    case: dict[str, Any],
    matched: dict[str, Any],
    gif_path: Path,
    view: tuple[str, float, float],
    gt_matching_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    cls, asset, action = _case_key(case)
    raw_gt_anim = ex._load_mesh_sequence(Path(case["gt_glb"]))
    matching = _read_json(gt_matching_dir / _matching_filename(cls, asset, action))
    gt_link_components_anim = p2d._target_link_components_from_matching(matching)
    states = p2d._selected_states(matched, bool(args.include_terminal))
    if int(args.max_states) > 0:
        states = states[: int(args.max_states)]

    static_cache: dict[str, tuple[ex.RawMeshSequence, list[np.ndarray], dict[str, list[int]], dict[str, tuple[np.ndarray, np.ndarray]]]] = {}

    def static_path_for_state(state: dict[str, Any]) -> Path | None:
        raw_path = state.get("gt_static_glb")
        if not raw_path:
            return None
        return _resolve_project_relative_path(raw_path, case)

    first_static_path = next((static_path_for_state(st) for st in states if static_path_for_state(st) is not None), None)
    if first_static_path is not None:
        first_raw_gt = ex._load_mesh_sequence(first_static_path)
        first_gt_link_components = p2d._link_components_from_static_names(first_raw_gt, set(gt_link_components_anim))
        raw_for_camera = first_raw_gt
        link_names = sorted(first_gt_link_components)
    else:
        raw_for_camera = raw_gt_anim
        link_names = sorted(gt_link_components_anim)
    if not link_names:
        raise ValueError("No GT links for Puppet scoring")
    camera, camera_info = _camera_for_selected_view(case, raw_for_camera, view, gif_path)
    gt_component_faces_anim = p2d._component_face_groups(raw_gt_anim, int(args.min_component_faces))
    gt_recipes_anim = p2d._sample_link_recipes(
        raw_gt_anim,
        {ln: gt_link_components_anim[ln] for ln in link_names if ln in gt_link_components_anim},
        gt_component_faces_anim,
        int(args.points_per_link),
        f"final2d:puppet:gt:{case['case_id']}:animated",
    )

    def gt_render_inputs(state: dict[str, Any]) -> tuple[ex.RawMeshSequence, int, dict[str, list[int]], list[np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
        static_path = static_path_for_state(state)
        if static_path is None:
            return raw_gt_anim, int(state["gt_frame_index"]), gt_link_components_anim, gt_component_faces_anim, gt_recipes_anim
        key = str(static_path)
        if key not in static_cache:
            raw = ex._load_mesh_sequence(static_path)
            comp_faces = p2d._component_face_groups(raw, int(args.min_component_faces))
            link_components = p2d._link_components_from_static_names(raw, set(gt_link_components_anim) | set(link_names))
            recipes = p2d._sample_link_recipes(
                raw,
                {ln: link_components[ln] for ln in link_names if ln in link_components},
                comp_faces,
                int(args.points_per_link),
                f"final2d:puppet:gt:{case['case_id']}:{static_path.name}",
            )
            static_cache[key] = (raw, comp_faces, link_components, recipes)
        raw, comp_faces, link_components, recipes = static_cache[key]
        return raw, 0, link_components, comp_faces, recipes

    render_res = tuple(args.render_resolution)
    out_res = tuple(args.resolution)
    first_state = states[0] if states else {"gt_frame_index": 0}
    first_raw, first_frame_idx, _first_components, _first_faces, first_recipes = gt_render_inputs(first_state)
    first_owner, _ = p2d._render_owner_points(
        p2d._points_by_link_for_frame(first_raw, first_frame_idx, first_recipes),
        link_names,
        camera,
        render_res,
        int(args.render_point_radius),
    )
    crop_spec = _crop_spec_from_owner(first_owner, float(args.crop_scale))
    canvas_shape = (int(render_res[1]), int(render_res[0]))
    puppet_transform = _puppet_input_transform(case, gif_path, canvas_shape, float(args.crop_scale))
    if str(puppet_transform.get("mode")) == "preprocessed_256":
        base_initial = _crop_resize_owner(first_owner, crop_spec, out_res)
        align_spec, inverse_align_spec = _puppet_preprocess_alignment(base_initial, puppet_transform)
        if align_spec is not None:
            puppet_transform = dict(puppet_transform)
            puppet_transform["owner_align_spec"] = align_spec
            puppet_transform["owner_inverse_align_spec"] = inverse_align_spec
    initial_owner = _owner_to_puppet_space(first_owner, puppet_transform, crop_spec, out_res)
    source_fg = _puppet_source_foreground_mask(puppet_transform, out_res)
    if source_fg is not None and source_fg.shape == initial_owner.shape:
        initial_owner = np.where(source_fg, initial_owner, -1).astype(np.int32)
    frames = _load_gif_frames(gif_path, tuple(args.resolution))
    if not frames:
        raise ValueError(f"No frames in {gif_path}")
    pred_owners_puppet = _propagate_owner_with_flow(initial_owner, frames)
    pred_owners_puppet = [np.where(_foreground_mask(frame), owner, -1).astype(np.int32) for owner, frame in zip(pred_owners_puppet, frames)]
    if bool(getattr(args, "puppet_label_color_filter", False)):
        label_color_refs = _label_color_references(initial_owner, frames[0])
        pred_owners_puppet = [
            _filter_owner_by_label_color(owner, frame, label_color_refs)
            for owner, frame in zip(pred_owners_puppet, frames)
        ]
    pred_owners_all = [
        _owner_from_puppet_space_to_eval_crop(owner, puppet_transform, crop_spec, canvas_shape, out_res)
        for owner in pred_owners_puppet
    ]
    # Puppet-Master GIF frame 0 is the drag/input frame in our saved results.
    # It may contain control arrows and should not be considered a prediction
    # state when aligning benchmark phases to the generated animation.
    valid_pred_offset = 1 if len(pred_owners_all) > 1 else 0
    pred_owners_for_match = pred_owners_all[valid_pred_offset:]
    gt_owners = []
    gt_imgs = []
    for state in states:
        gt_raw, gt_frame_idx, gt_components, gt_faces, gt_recipes = gt_render_inputs(state)
        owner, img = _render_owner_frame(
            gt_raw,
            gt_frame_idx,
            gt_components,
            gt_faces,
            gt_recipes,
            link_names,
            camera,
            render_res,
            int(args.render_point_radius),
            crop_spec,
            out_res,
        )
        gt_owners.append(owner)
        gt_imgs.append(img)
    overlap_prev = _overlap_flags_from_states(states)
    mask_iou_pred_indices = [
        int(i) + valid_pred_offset for i in _ordered_match(gt_owners, pred_owners_for_match, overlap_prev)
    ]
    if str(getattr(args, "puppet_match_mode", "timestamp")) == "mask_iou":
        pred_indices = mask_iou_pred_indices
    else:
        pred_indices = [
            int(i) + valid_pred_offset
            for i in _temporal_match_from_states(
                states,
                len(pred_owners_for_match),
                pred_owners_for_match,
                gt_owners,
            )
        ]

    view_id = str(view[0])
    per_link_rows: list[dict[str, Any]] = []
    per_state_rows: list[dict[str, Any]] = []
    for state, gt_owner, gt_img, pidx in zip(states, gt_owners, gt_imgs, pred_indices):
        pred_owner = pred_owners_all[int(pidx)]
        pred_img = _owner_to_color(pred_owner, link_names)
        scores = []
        for li, link in enumerate(link_names):
            score = _score_link(gt_owner, pred_owner, gt_img, pred_img, li, int(args.min_visible_px), args)
            if score is None:
                continue
            row = {
                "case_id": case["case_id"],
                "class": cls,
                "asset_name": asset,
                "action_name": action,
                "variant": "puppet_master",
                "view_id": view_id,
                "phase_id": state["phase_id"],
                "state_index": state["state_index"],
                "gt_frame_index": state["gt_frame_index"],
                "pred_frame_index": int(pidx),
                "link": link,
                **score,
            }
            per_link_rows.append(row)
            scores.append(score)
        per_state_rows.append(
            {
                "case_id": case["case_id"],
                "class": cls,
                "asset_name": asset,
                "action_name": action,
                "variant": "puppet_master",
                "view_id": view_id,
                "phase_id": state["phase_id"],
                "state_index": state["state_index"],
                "gt_frame_index": state["gt_frame_index"],
                "pred_frame_index": int(pidx),
                "num_scored_parts": len(scores),
                "P_MaskIoU": _mean([s.get("P_MaskIoU") for s in scores]),
                "P_BoundaryF1": _mean([s.get("P_BoundaryF1") for s in scores]),
                "P_ContourCD": _mean([s.get("P_ContourCD") for s in scores]),
                "P_IoU": _mean([s.get("P_IoU") for s in scores]),
                "P_PSNR": _mean([s.get("P_PSNR") for s in scores]),
                "P_SSIM": _mean([s.get("P_SSIM") for s in scores]),
                "P_LPIPS": _mean([s.get("P_LPIPS") for s in scores]),
            }
        )
    per_case = {
        "case_id": case["case_id"],
        "class": cls,
        "asset_name": asset,
        "action_name": action,
        "variant": "puppet_master",
        "num_states": len(states),
        "num_views": 1,
        "num_links": len(link_names),
        "P_MaskIoU": _mean([r.get("P_MaskIoU") for r in per_state_rows]),
        "P_BoundaryF1": _mean([r.get("P_BoundaryF1") for r in per_state_rows]),
        "P_ContourCD": _mean([r.get("P_ContourCD") for r in per_state_rows]),
        "P_IoU": _mean([r.get("P_IoU") for r in per_state_rows]),
        "P_PSNR": _mean([r.get("P_PSNR") for r in per_state_rows]),
        "P_SSIM": _mean([r.get("P_SSIM") for r in per_state_rows]),
        "P_LPIPS": _mean([r.get("P_LPIPS") for r in per_state_rows]),
    }
    diag = {
        "case_id": case["case_id"],
        "gif": str(gif_path),
        "view": {"id": view[0], "azimuth_deg": view[1], "elevation_deg": view[2]},
        "num_gif_frames": len(frames),
        "pred_frame_indices": pred_indices,
        "mask_iou_pred_frame_indices": mask_iou_pred_indices,
        "overlap_allowed_from_previous": overlap_prev,
        "match_mode": str(getattr(args, "puppet_match_mode", "timestamp")),
        "label_color_filter": bool(getattr(args, "puppet_label_color_filter", False)),
        "pred_owner_bboxes": [list(_owner_bbox(pred_owners_all[int(pidx)]) or []) for pidx in pred_indices],
        "crop_spec": crop_spec,
        "coordinate_space": "common_mesh_eval_crop_with_puppet_pred_unwarped",
        "puppet_input_transform": puppet_transform,
        "camera_info": camera_info,
    }
    return per_link_rows, per_state_rows, per_case, diag


def _aggregate(case_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variant_rows = []
    category_rows = []
    for variant in sorted({str(r["variant"]) for r in case_rows}):
        rows_v = [r for r in case_rows if str(r["variant"]) == variant]
        variant_rows.append(
            {
                "class": "overall",
                "variant": variant,
                "num_cases": len(rows_v),
                "P_MaskIoU": _mean([r.get("P_MaskIoU") for r in rows_v]),
                "P_BoundaryF1": _mean([r.get("P_BoundaryF1") for r in rows_v]),
                "P_ContourCD": _mean([r.get("P_ContourCD") for r in rows_v]),
                "P_IoU": _mean([r.get("P_IoU") for r in rows_v]),
                "P_PSNR": _mean([r.get("P_PSNR") for r in rows_v]),
                "P_SSIM": _mean([r.get("P_SSIM") for r in rows_v]),
                "P_LPIPS": _mean([r.get("P_LPIPS") for r in rows_v]),
            }
        )
        for cls in sorted({str(r["class"]) for r in rows_v}):
            rows_c = [r for r in rows_v if str(r["class"]) == cls]
            category_rows.append(
                {
                    "class": cls,
                    "variant": variant,
                    "num_cases": len(rows_c),
                    "P_MaskIoU": _mean([r.get("P_MaskIoU") for r in rows_c]),
                    "P_BoundaryF1": _mean([r.get("P_BoundaryF1") for r in rows_c]),
                    "P_ContourCD": _mean([r.get("P_ContourCD") for r in rows_c]),
                    "P_IoU": _mean([r.get("P_IoU") for r in rows_c]),
                    "P_PSNR": _mean([r.get("P_PSNR") for r in rows_c]),
                    "P_SSIM": _mean([r.get("P_SSIM") for r in rows_c]),
                    "P_LPIPS": _mean([r.get("P_LPIPS") for r in rows_c]),
                }
            )
    return variant_rows, variant_rows + category_rows


def _worker_main(args: argparse.Namespace) -> None:
    # ufbx can corrupt/crash during Python GC after loading some FBX files.
    # Keep workers short-lived and skip cyclic GC, matching evaluate_2d_part.py.
    gc.disable()
    task = _read_json(Path(args.worker_task_json))
    try:
        if task["kind"] == "mesh":
            lr, sr, cr = _case_mesh_metrics(
                task["case"],
                task["metric_row"],
                task["matched"],
                Path(task["matching_dir"]),
                Path(task["gt_matching_dir"]),
                str(task["variant"]),
                tuple(task["view"]),
                args,
            )
            out = {"per_link": lr, "per_state": sr, "per_case": cr, "diag": None, "error": None}
        elif task["kind"] == "puppet":
            lr, sr, cr, diag = _case_puppet_metrics(
                task["case"],
                task["matched"],
                Path(task["gif_path"]),
                tuple(task["view"]),
                Path(task["gt_matching_dir"]),
                args,
            )
            out = {"per_link": lr, "per_state": sr, "per_case": cr, "diag": diag, "error": None}
        else:
            raise ValueError(f"unknown worker task kind: {task.get('kind')}")
    except Exception as exc:
        out = {"per_link": [], "per_state": [], "per_case": None, "diag": None, "error": str(exc)}
    _write_json(Path(args.worker_out_json), out)
    os._exit(0)


def _run_worker(task: dict[str, Any], args: argparse.Namespace, worker_dir: Path, ordinal: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, str | None]:
    task_path = worker_dir / f"task_{ordinal:05d}.json"
    out_path = worker_dir / f"result_{ordinal:05d}.json"
    _write_json(task_path, task)
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker_task_json",
        str(task_path),
        "--worker_out_json",
        str(out_path),
        "--resolution",
        str(int(args.resolution[0])),
        str(int(args.resolution[1])),
        "--render_resolution",
        str(int(args.render_resolution[0])),
        str(int(args.render_resolution[1])),
        "--crop_scale",
        str(float(args.crop_scale)),
        "--points_per_link",
        str(int(args.points_per_link)),
        "--render_point_radius",
        str(int(args.render_point_radius)),
        "--min_visible_px",
        str(int(args.min_visible_px)),
        "--min_component_faces",
        str(int(args.min_component_faces)),
        "--boundary_tolerance_px",
        str(int(args.boundary_tolerance_px)),
        "--contour_scale_floor_px",
        str(float(args.contour_scale_floor_px)),
        "--mesh_alignment",
        str(args.mesh_alignment),
        "--max_states",
        str(int(args.max_states)),
        "--lpips_net",
        str(args.lpips_net),
        "--lpips_device",
        str(args.lpips_device),
        "--lpips_resolution",
        str(int(args.lpips_resolution)),
        "--puppet_match_mode",
        str(args.puppet_match_mode),
    ]
    for root in getattr(args, "project_root", None) or []:
        cmd.extend(["--project_root", str(root)])
    cmd.extend(["--puppet_root", str(args.puppet_root)])
    cmd.extend(["--noncausal_views_json", str(args.noncausal_views_json)])
    if bool(getattr(args, "puppet_label_color_filter", False)):
        cmd.append("--puppet_label_color_filter")
    cmd.append("--include_terminal" if bool(args.include_terminal) else "--no_terminal")
    if bool(args.compute_lpips):
        cmd.append("--compute_lpips")
    last_error = None
    for attempt in range(1, int(args.worker_retries) + 2):
        if out_path.exists():
            out_path.unlink()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode == 0 and out_path.exists():
            break
        last_error = f"worker_returncode_{proc.returncode}_attempt_{attempt}: {proc.stdout[-2000:]}"
    else:
        return [], [], None, None, last_error or "worker_missing_output"
    if not out_path.exists():
        return [], [], None, None, last_error or "worker_missing_output"
    result = _read_json(out_path)
    if result.get("error"):
        return [], [], None, result.get("diag"), str(result["error"])
    return list(result.get("per_link") or []), list(result.get("per_state") or []), result.get("per_case"), result.get("diag"), None


def main() -> None:
    parser = argparse.ArgumentParser(description="Final selected-view 2D part evaluation for full agent, AAM, and PuppetMaster.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "experiments/final_3d_evaluation/ablation_3d/diagnose/resolved_manifest.json")
    parser.add_argument(
        "--data_root",
        type=Path,
        action="append",
        default=[],
        help="Root containing causal_data/not_causal_data asset folders. Can be passed multiple times.",
    )
    parser.add_argument("--own_3d_dir", type=Path, default=REPO_ROOT / "experiments/final_3d_evaluation/own_method_3d_matching")
    parser.add_argument("--aam_3d_dir", type=Path, default=REPO_ROOT / "experiments/final_3d_evaluation/animate_anymesh_3d")
    parser.add_argument("--animate3d_3d_dir", type=Path, default=None)
    parser.add_argument("--particulate_3d_dir", type=Path, default=REPO_ROOT / "experiments/final_3d_evaluation/particulate_urdf_3d_scale_normalized")
    parser.add_argument("--puppet_root", type=Path, default=REPO_ROOT / "experiments/final_puppet_results")
    parser.add_argument("--noncausal_views_json", type=Path, default=REPO_ROOT / "puppet_master_noncausal/reference_view_distances.json")
    parser.add_argument(
        "--project_root",
        type=Path,
        action="append",
        default=[],
        help="Additional project root to search for legacy render manifests and relative benchmark resources.",
    )
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "experiments/final_2d_evaluation_selected_view")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=["full_agent", "animate_anymesh", "puppet_master"],
        choices=["full_agent", "animate_anymesh", "animate3d", "puppet_master", "particulate_urdf"],
    )
    parser.add_argument("--resolution", type=int, nargs=2, default=(256, 256), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--render_resolution", type=int, nargs=2, default=(800, 600), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--crop_scale", type=float, default=2.2)
    parser.add_argument("--points_per_link", type=int, default=3500)
    parser.add_argument("--render_point_radius", type=int, default=3)
    parser.add_argument("--min_visible_px", type=int, default=24)
    parser.add_argument("--min_component_faces", type=int, default=1)
    parser.add_argument("--boundary_tolerance_px", type=int, default=2)
    parser.add_argument("--contour_scale_floor_px", type=float, default=8.0)
    parser.add_argument(
        "--mesh_alignment",
        choices=["eval_3d", "scale_translate_3d", "none"],
        default="eval_3d",
        help="How mesh-based prediction sequences are aligned before 2D projection. eval_3d preserves old metric alignment; scale_translate_3d preserves visual axes and only normalizes scale/translation.",
    )
    parser.add_argument("--include_terminal", action="store_true", default=False)
    parser.add_argument("--no_terminal", action="store_false", dest="include_terminal")
    parser.add_argument("--compute_lpips", action="store_true", help="Compute color-independent LPIPS on per-link binary masks. Lower is better.")
    parser.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips_device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--lpips_resolution", type=int, default=64)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_states", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker_retries", type=int, default=3)
    parser.add_argument(
        "--puppet_match_mode",
        choices=["timestamp", "mask_iou"],
        default="timestamp",
        help="timestamp aligns Puppet GIF frames to GT phase timestamps; mask_iou uses optical-flow owner masks for temporal matching.",
    )
    parser.add_argument(
        "--puppet_label_color_filter",
        action="store_true",
        help="Apply the legacy color-consistency filter to propagated Puppet owner masks.",
    )
    parser.add_argument("--worker_task_json", type=Path, default=None)
    parser.add_argument("--worker_out_json", type=Path, default=None)
    args = parser.parse_args()
    _configure_project_search_roots(args)

    if args.worker_task_json is not None:
        if args.worker_out_json is None:
            raise ValueError("--worker_out_json is required with --worker_task_json")
        _worker_main(args)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases_by_key = _manifest_cases(args.manifest, list(args.data_root or []))
    active_variants = set(args.variants or [])
    own_required = bool({"full_agent", "puppet_master"} & active_variants)
    aam_required = "animate_anymesh" in active_variants
    animate3d_required = "animate3d" in active_variants
    particulate_required = "particulate_urdf" in active_variants
    own_rows = {_case_key(r): r for r in _read_3d_metric_rows(args.own_3d_dir, required=own_required)}
    aam_rows = {_case_key(r): r for r in _read_3d_metric_rows(args.aam_3d_dir, required=aam_required)}
    animate3d_rows = {_case_key(r): r for r in _read_3d_metric_rows(args.animate3d_3d_dir, required=animate3d_required)}
    particulate_rows = {_case_key(r): r for r in _read_3d_metric_rows(args.particulate_3d_dir, required=particulate_required)}
    own_matched = _read_matched_rows(args.own_3d_dir, required=own_required)
    aam_matched = _read_matched_rows(args.aam_3d_dir, required=aam_required)
    animate3d_matched = _read_matched_rows(args.animate3d_3d_dir, required=animate3d_required)
    particulate_matched = _read_matched_rows(args.particulate_3d_dir, required=particulate_required)
    noncausal_views = _load_noncausal_views(args.noncausal_views_json)

    all_cases = list(cases_by_key.values())
    if int(args.max_cases) > 0:
        all_cases = all_cases[: int(args.max_cases)]

    all_case_rows: list[dict[str, Any]] = []
    all_state_rows: list[dict[str, Any]] = []
    all_link_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    puppet_diag: list[dict[str, Any]] = []
    worker_dir = args.out_dir / "diagnose" / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    ordinal = 0
    task_records: list[dict[str, Any]] = []

    for idx, case in enumerate(all_cases, start=1):
        key = _case_key(case)
        try:
            gif_path, selected_view = _find_puppet_gif(case, args.puppet_root, noncausal_views)
        except Exception as exc:
            selected_view = ("V1", 0.0, 20.0)
            gif_path = None
            if "puppet_master" in args.variants:
                errors.append({"variant": "puppet_master", "case": list(key), "error": str(exc)})

        if "full_agent" in args.variants and key in own_rows:
            matched = own_matched.get((key[1], key[2]))
            if matched is None:
                errors.append({"variant": "full_agent", "case": list(key), "error": "missing_matched_frames"})
            else:
                ordinal += 1
                task = {
                    "kind": "mesh",
                    "variant": "full_agent",
                    "case": case,
                    "metric_row": own_rows[key],
                    "matched": matched,
                    "matching_dir": str(args.own_3d_dir / "diagnose/matching"),
                    "gt_matching_dir": str(args.own_3d_dir / "diagnose/matching"),
                    "view": list(selected_view),
                }
                task_records.append({"ordinal": ordinal, "idx": idx, "total": len(all_cases), "variant": "full_agent", "case_key": list(key), "task": task})

        if "animate_anymesh" in args.variants and key in aam_rows:
            matched = aam_matched.get((key[1], key[2]))
            if matched is None:
                errors.append({"variant": "animate_anymesh", "case": list(key), "error": "missing_matched_frames"})
            else:
                ordinal += 1
                task = {
                    "kind": "mesh",
                    "variant": "animate_anymesh",
                    "case": case,
                    "metric_row": aam_rows[key],
                    "matched": matched,
                    "matching_dir": str(args.aam_3d_dir / "diagnose/matching"),
                    "gt_matching_dir": str(args.own_3d_dir / "diagnose/matching"),
                    "view": list(selected_view),
                }
                task_records.append({"ordinal": ordinal, "idx": idx, "total": len(all_cases), "variant": "animate_anymesh", "case_key": list(key), "task": task})

        if "animate3d" in args.variants and key in animate3d_rows:
            matched = animate3d_matched.get((key[1], key[2]))
            if matched is None:
                errors.append({"variant": "animate3d", "case": list(key), "error": "missing_matched_frames"})
            else:
                ordinal += 1
                task = {
                    "kind": "mesh",
                    "variant": "animate3d",
                    "case": case,
                    "metric_row": animate3d_rows[key],
                    "matched": matched,
                    "matching_dir": str(args.animate3d_3d_dir / "diagnose/matching"),
                    "gt_matching_dir": str(args.own_3d_dir / "diagnose/matching"),
                    "view": list(selected_view),
                }
                task_records.append({"ordinal": ordinal, "idx": idx, "total": len(all_cases), "variant": "animate3d", "case_key": list(key), "task": task})

        if "particulate_urdf" in args.variants and key in particulate_rows:
            matched = particulate_matched.get((key[1], key[2]))
            if matched is None:
                errors.append({"variant": "particulate_urdf", "case": list(key), "error": "missing_matched_frames"})
            else:
                ordinal += 1
                task = {
                    "kind": "mesh",
                    "variant": "particulate_urdf",
                    "case": case,
                    "metric_row": particulate_rows[key],
                    "matched": matched,
                    "matching_dir": str(args.particulate_3d_dir / "diagnose/matching"),
                    "gt_matching_dir": str(args.own_3d_dir / "diagnose/matching"),
                    "view": list(selected_view),
                }
                task_records.append({"ordinal": ordinal, "idx": idx, "total": len(all_cases), "variant": "particulate_urdf", "case_key": list(key), "task": task})

        if "puppet_master" in args.variants and gif_path is not None:
            matched = own_matched.get((key[1], key[2]))
            if matched is None:
                errors.append({"variant": "puppet_master", "case": list(key), "error": "missing_gt_endpoint_frames"})
            else:
                ordinal += 1
                task = {
                    "kind": "puppet",
                    "case": case,
                    "matched": matched,
                    "gif_path": str(gif_path),
                    "view": list(selected_view),
                    "gt_matching_dir": str(args.own_3d_dir / "diagnose/matching"),
                }
                task_records.append({"ordinal": ordinal, "idx": idx, "total": len(all_cases), "variant": "puppet_master", "case_key": list(key), "task": task})

    def run_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None, str | None]:
        lr, sr, cr, diag, err = _run_worker(record["task"], args, worker_dir, int(record["ordinal"]))
        return record, lr, sr, cr, diag, err

    def consume_result(
        record: dict[str, Any],
        lr: list[dict[str, Any]],
        sr: list[dict[str, Any]],
        cr: dict[str, Any] | None,
        diag: dict[str, Any] | None,
        err: str | None,
    ) -> None:
        key_list = list(record["case_key"])
        variant = str(record["variant"])
        idx = int(record["idx"])
        total = int(record["total"])
        if err is None and cr is not None:
            all_link_rows.extend(lr)
            all_state_rows.extend(sr)
            all_case_rows.append(cr)
            if variant == "puppet_master" and diag is not None:
                puppet_diag.append(diag)
            piou = cr.get("P_MaskIoU")
            piou_s = "nan" if piou is None else f"{float(piou):.6f}"
            print(f"[OK] {variant} {idx}/{total} {key_list[1]}/{key_list[2]} P-MaskIoU={piou_s}", flush=True)
        else:
            errors.append({"variant": variant, "case": key_list, "error": err})
            print(f"[ERR] {variant} {idx}/{total} {key_list[1]}/{key_list[2]}: {err}", flush=True)

    worker_count = max(1, int(args.workers))
    print(f"[INFO] Running {len(task_records)} evaluation tasks with workers={worker_count}", flush=True)
    if worker_count <= 1:
        for record in task_records:
            consume_result(*run_record(record))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [pool.submit(run_record, record) for record in task_records]
            for fut in as_completed(futures):
                consume_result(*fut.result())

    def row_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int, str]:
        return (
            str(row.get("class") or ""),
            str(row.get("asset_name") or ""),
            str(row.get("action_name") or ""),
            str(row.get("variant") or ""),
            int(row.get("state_index") or 0),
            int(row.get("pred_frame_index") or 0),
            str(row.get("link") or ""),
        )

    all_case_rows.sort(key=row_key)
    all_state_rows.sort(key=row_key)
    all_link_rows.sort(key=row_key)
    puppet_diag.sort(key=lambda r: (str(r.get("case_id") or ""), str(r.get("view_id") or "")))

    variant_rows, category_rows = _aggregate(all_case_rows)
    metric_fields = ["P_MaskIoU", "P_BoundaryF1", "P_ContourCD"]
    if bool(args.compute_lpips):
        metric_fields.append("P_LPIPS")
    summary = {
        "metrics": metric_fields,
        "notes": [
            "All variants are evaluated in the Puppet-selected view for each case.",
            "GT part masks are rendered from per-phase static endpoint GLBs when matched_frames.json provides gt_static_glbs; older matched files fall back to the compiled GT animation frame.",
            "Mesh-baseline part masks are projected with the same camera, then square-cropped and resized to 256x256.",
            "Puppet masks are initialized in Puppet's saved preprocessing crop, propagated through GIF frames with dense optical flow and per-label color consistency, then inverse-aligned back into the common mesh-evaluation crop before scoring.",
            "Puppet endpoint frames are selected by ordered foreground-mask IoU matching against GT endpoint masks.",
            "P-MaskIoU, P-BoundaryF1, and P-ContourCD are computed on per-link binary masks; no color, texture, or material information is used.",
            "P-ContourCD is normalized by each visible part's union-mask bbox diagonal and is lower-is-better.",
            "P-LPIPS, when enabled, is computed on per-link binary masks and is lower-is-better.",
        ],
        "variants": {str(r["variant"]): r for r in variant_rows},
        "num_case_rows": len(all_case_rows),
        "num_state_rows": len(all_state_rows),
        "num_link_rows": len(all_link_rows),
        "num_errors": len(errors),
        "resolution": list(args.resolution),
        "render_resolution": list(args.render_resolution),
        "crop_scale": float(args.crop_scale),
        "boundary_tolerance_px": int(args.boundary_tolerance_px),
        "contour_scale_floor_px": float(args.contour_scale_floor_px),
        "compute_lpips": bool(args.compute_lpips),
        "lpips_net": str(args.lpips_net),
        "lpips_device": str(args.lpips_device),
        "lpips_resolution": int(args.lpips_resolution),
        "errors": errors,
    }

    _write_csv(args.out_dir / "part2d_case_metrics.csv", all_case_rows, ["case_id", "class", "asset_name", "action_name", "variant", "num_states", "num_views", "num_links", *metric_fields])
    _write_csv(args.out_dir / "part2d_state_metrics.csv", all_state_rows, ["case_id", "class", "asset_name", "action_name", "variant", "view_id", "phase_id", "state_index", "gt_frame_index", "pred_frame_index", "num_scored_parts", *metric_fields])
    _write_csv(args.out_dir / "part2d_link_metrics.csv", all_link_rows, ["case_id", "class", "asset_name", "action_name", "variant", "view_id", "phase_id", "state_index", "gt_frame_index", "pred_frame_index", "link", *metric_fields, "gt_visible_px", "pred_visible_px"])
    _write_csv(args.out_dir / "part2d_variant_summary.csv", variant_rows, ["class", "variant", "num_cases", *metric_fields])
    _write_csv(args.out_dir / "part2d_category_mean.csv", category_rows, ["class", "variant", "num_cases", *metric_fields])
    _write_json(args.out_dir / "diagnose/summary.json", summary)
    _write_json(args.out_dir / "diagnose/errors.json", errors)
    _write_json(args.out_dir / "diagnose/puppet_matching.json", puppet_diag)
    print(json.dumps(summary["variants"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
