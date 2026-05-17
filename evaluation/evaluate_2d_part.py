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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


EVALUATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_ROOT.parent
TOOLS_ROOT = REPO_ROOT / "tools"
for _path in (EVALUATION_ROOT, TOOLS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_ablation_3d as ev  # noqa: E402
import evaluate_external_4d as ex  # noqa: E402
import gen_overlays_and_prompts as gop  # noqa: E402


DEFAULT_VIEWS = (
    ("V1", 0.0, 20.0),
    ("V2", 90.0, 20.0),
    ("V3", 180.0, 20.0),
    ("V4", 270.0, 20.0),
)

DEBUG_PALETTE = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (0, 0, 128),
)


@dataclass
class EvalInput:
    variant: str
    result_dir: Path
    metrics_csv: Path
    matched_frames_json: Path
    matching_dir: Path


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
            out = dict(row)
            for k, v in list(out.items()):
                if isinstance(v, (list, dict)):
                    out[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(out)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _case_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("class") or ""), str(row.get("asset_name") or ""), str(row.get("action_name") or ""))


def _matching_filename(cls: str, asset: str, action: str) -> str:
    return f"{cls}__{asset}__{action}.json"


def _mean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _deterministic_color(name: str) -> tuple[int, int, int]:
    try:
        c = gop._deterministic_color(str(name))
        return tuple(int(x) for x in c[:3])
    except Exception:
        h = ev.stable_seed(str(name))
        return (64 + (h & 127), 64 + ((h >> 8) & 127), 64 + ((h >> 16) & 127))


def _debug_color(index: int, name: str) -> tuple[int, int, int]:
    if 0 <= int(index) < len(DEBUG_PALETTE):
        return DEBUG_PALETTE[int(index)]
    return _deterministic_color(name)


def _mesh_for_faces(vertices: np.ndarray, faces: np.ndarray) -> trimesh.Trimesh | None:
    if vertices.size == 0 or faces.size == 0:
        return None
    return trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float32), faces=np.asarray(faces, dtype=np.int64), process=False)


def _component_face_groups(raw: ex.RawMeshSequence, min_component_faces: int) -> list[np.ndarray]:
    if raw.component_face_indices:
        return [np.asarray(x, dtype=np.int64) for x in raw.component_face_indices]
    comps = ex._component_split(raw.vertices_by_frame[0], raw.faces, int(min_component_faces))
    return [np.asarray(c.face_indices, dtype=np.int64) for c in comps]


def _component_name_to_id(raw: ex.RawMeshSequence) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(raw.component_names or [])}


def _link_components_from_gt(gt_glb: Path, asset: ev.AssetGeometry, raw_gt: ex.RawMeshSequence) -> dict[str, list[int]]:
    _pts, targets = ex._gt_glb_first_frame_targets(gt_glb, asset, 128)
    by_name = _component_name_to_id(raw_gt)
    out: dict[str, list[int]] = {ln: [] for ln in asset.visual_links}
    for target in targets:
        cid = by_name.get(str(target.node_name))
        if cid is not None:
            out.setdefault(str(target.link), []).append(int(cid))
    return {ln: sorted(set(ids)) for ln, ids in out.items() if ids}


def _target_link_components_from_matching(matching: dict[str, Any]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for row in matching.get("component_assignments") or []:
        link = row.get("assigned_link")
        cid = row.get("target_component_id")
        if link is None or cid is None:
            continue
        out.setdefault(str(link), []).append(int(cid))
    return {ln: sorted(set(ids)) for ln, ids in out.items() if ids}


def _link_components_from_matching(matching: dict[str, Any]) -> dict[str, list[int]]:
    raw = matching.get("components_by_link")
    if isinstance(raw, dict) and raw:
        return {
            str(link): sorted({int(x) for x in ids})
            for link, ids in raw.items()
            if isinstance(ids, list) and ids
        }
    out: dict[str, list[int]] = {}
    for row in matching.get("component_assignments") or []:
        link = row.get("assigned_link")
        cid = row.get("component_id")
        if link is None or cid is None:
            continue
        out.setdefault(str(link), []).append(int(cid))
    return {ln: sorted(set(ids)) for ln, ids in out.items() if ids}


def _link_components_from_static_names(raw: ex.RawMeshSequence, candidate_links: list[str] | set[str]) -> dict[str, list[int]]:
    links = sorted({str(x) for x in candidate_links}, key=len, reverse=True)
    out: dict[str, list[int]] = {ln: [] for ln in links}
    for cid, name in enumerate(raw.component_names or []):
        s = str(name)
        for link in links:
            if s == link or s.startswith(f"link_{link}_") or s.startswith(f"{link}_"):
                out.setdefault(link, []).append(int(cid))
                break
    return {ln: sorted(set(ids)) for ln, ids in out.items() if ids}


def _meshes_by_link_for_frame(
    raw: ex.RawMeshSequence,
    frame_idx: int,
    link_components: dict[str, list[int]],
    component_faces: list[np.ndarray],
) -> dict[str, list[trimesh.Trimesh]]:
    fi = min(max(0, int(frame_idx)), len(raw.vertices_by_frame) - 1)
    vertices = raw.vertices_by_frame[fi]
    out: dict[str, list[trimesh.Trimesh]] = {}
    for link, comp_ids in link_components.items():
        face_chunks = []
        for cid in comp_ids:
            if 0 <= int(cid) < len(component_faces):
                faces_idx = component_faces[int(cid)]
                if faces_idx.size:
                    face_chunks.append(raw.faces[faces_idx])
        if not face_chunks:
            continue
        faces = np.concatenate(face_chunks, axis=0)
        mesh = _mesh_for_faces(vertices, faces)
        if mesh is not None:
            out[str(link)] = [mesh]
    return out


def _sample_link_recipes(
    raw: ex.RawMeshSequence,
    link_components: dict[str, list[int]],
    component_faces: list[np.ndarray],
    points_per_link: int,
    seed_prefix: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    vertices0 = np.asarray(raw.vertices_by_frame[0], dtype=np.float32)
    recipes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for link, comp_ids in link_components.items():
        face_chunks = []
        for cid in comp_ids:
            if 0 <= int(cid) < len(component_faces):
                face_idx = np.asarray(component_faces[int(cid)], dtype=np.int64)
                if face_idx.size:
                    face_chunks.append(face_idx)
        if not face_chunks:
            continue
        face_indices = np.concatenate(face_chunks, axis=0)
        faces = raw.faces[face_indices]
        tri = vertices0[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        if not np.any(area > 1.0e-12):
            continue
        prob = area / float(np.sum(area))
        seed = ev.stable_seed(f"{seed_prefix}:{link}:{points_per_link}")
        rng = np.random.default_rng(seed)
        chosen_local = rng.choice(len(face_indices), size=max(1, int(points_per_link)), replace=True, p=prob)
        chosen_faces = face_indices[chosen_local]
        u = rng.random(len(chosen_faces), dtype=np.float32)
        v = rng.random(len(chosen_faces), dtype=np.float32)
        flip = (u + v) > 1.0
        u[flip] = 1.0 - u[flip]
        v[flip] = 1.0 - v[flip]
        bary = np.stack([1.0 - u - v, u, v], axis=1).astype(np.float32)
        recipes[str(link)] = (chosen_faces.astype(np.int64), bary)
    return recipes


def _points_by_link_for_frame(
    raw: ex.RawMeshSequence,
    frame_idx: int,
    recipes: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    fi = min(max(0, int(frame_idx)), len(raw.vertices_by_frame) - 1)
    vertices = np.asarray(raw.vertices_by_frame[fi], dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for link, (face_indices, bary) in recipes.items():
        faces = raw.faces[np.asarray(face_indices, dtype=np.int64)]
        tri = vertices[faces]
        weights = np.asarray(bary, dtype=np.float32)
        out[str(link)] = np.asarray(
            tri[:, 0, :] * weights[:, 0:1]
            + tri[:, 1, :] * weights[:, 1:2]
            + tri[:, 2, :] * weights[:, 2:3],
            dtype=np.float32,
        )
    return out


def _scene_bounds_from_raw(raw: ex.RawMeshSequence) -> tuple[np.ndarray, float]:
    verts = np.asarray(raw.vertices_by_frame[0], dtype=np.float32)
    if verts.size == 0:
        return np.zeros((3,), dtype=float), 1.0
    mn = np.min(verts, axis=0)
    mx = np.max(verts, axis=0)
    center = (mn + mx) * 0.5
    radius = max(0.1, float(np.linalg.norm(mx - mn)) * 0.6)
    return np.asarray(center, dtype=float), float(radius)


def _render_owner(
    meshes_by_link: dict[str, list[trimesh.Trimesh]],
    link_names: list[str],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    owner, _sub = gop.rasterize_link_visibility_maps(
        meshes_by_link,
        link_names,
        camera,
        resolution,
        max_faces=gop.REFERENCE_MAX_FACES,
        return_scene_depth=False,
    )
    height, width = owner.shape
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for idx, link in enumerate(link_names):
        mask = owner == int(idx)
        if np.any(mask):
            image[mask] = np.asarray(_deterministic_color(link), dtype=np.uint8)
    return owner, image


def _render_owner_points(
    points_by_link: dict[str, np.ndarray],
    link_names: list[str],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = int(resolution[0]), int(resolution[1])
    owner = np.full((height, width), -1, dtype=np.int32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    radius = max(0, int(point_radius))
    offsets = [(0, 0)]
    if radius > 0:
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]
    for li, link in enumerate(link_names):
        pts = np.asarray(points_by_link.get(link, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        if pts.size == 0:
            continue
        proj = gop.project_points(pts, camera, resolution)
        valid = proj[:, 2] > 1.0e-6
        if not np.any(valid):
            continue
        xs0 = np.rint(proj[valid, 0]).astype(np.int32)
        ys0 = np.rint(proj[valid, 1]).astype(np.int32)
        zs0 = np.asarray(proj[valid, 2], dtype=np.float32)
        order = np.argsort(zs0)[::-1]
        xs0, ys0, zs0 = xs0[order], ys0[order], zs0[order]
        for dx, dy in offsets:
            xs = xs0 + int(dx)
            ys = ys0 + int(dy)
            inside = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
            if not np.any(inside):
                continue
            xi = xs[inside]
            yi = ys[inside]
            zi = zs0[inside]
            closer = zi <= depth[yi, xi]
            if np.any(closer):
                depth[yi[closer], xi[closer]] = zi[closer]
                owner[yi[closer], xi[closer]] = int(li)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for idx, link in enumerate(link_names):
        mask = owner == int(idx)
        if np.any(mask):
            image[mask] = np.asarray(_deterministic_color(link), dtype=np.uint8)
    return owner, image


def render_case_debug_overlays(
    case: dict[str, Any],
    metric_row: dict[str, str],
    matched: dict[str, Any],
    eval_input: EvalInput,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    cls, asset_name, action_name = _case_key(case)
    raw_gt = ex._load_mesh_sequence(Path(case["gt_glb"]))
    raw_pred = ex._load_mesh_sequence(Path(metric_row["prediction_file"]))
    matching = _read_json(eval_input.matching_dir / _matching_filename(cls, asset_name, action_name))

    gt_component_faces = _component_face_groups(raw_gt, int(args.min_component_faces))
    pred_component_faces = _component_face_groups(raw_pred, int(args.min_component_faces))
    gt_link_components = _target_link_components_from_matching(matching)
    pred_link_components = _link_components_from_matching(matching)
    link_names = sorted(ln for ln in gt_link_components if ln in pred_link_components)
    if not link_names:
        raise ValueError("No common matched links for debug overlay")

    states = _selected_states(matched, bool(args.include_terminal))
    if args.debug_state == "first":
        states = states[:1]
    elif args.debug_state == "terminal":
        states = states[-1:]
    elif str(args.debug_state).isdigit():
        idx = max(0, min(len(states) - 1, int(args.debug_state)))
        states = states[idx : idx + 1]
    if not states:
        raise ValueError("No endpoint states for debug overlay")
    state = states[0]

    gt_recipes = _sample_link_recipes(
        raw_gt,
        {ln: gt_link_components[ln] for ln in link_names},
        gt_component_faces,
        int(args.points_per_link),
        f"part2d:{case['case_id']}",
    )
    pred_recipes = _sample_link_recipes(
        raw_pred,
        {ln: pred_link_components[ln] for ln in link_names},
        pred_component_faces,
        int(args.points_per_link),
        f"part2d:{case['case_id']}",
    )
    center, radius = _scene_bounds_from_raw(raw_gt)

    from PIL import Image, ImageDraw

    case_dir = out_dir / eval_input.variant / f"{cls}__{asset_name}__{action_name}"
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "case_id": case["case_id"],
        "variant": eval_input.variant,
        "prediction_file": str(metric_row["prediction_file"]),
        "gt_glb": str(case["gt_glb"]),
        "phase_id": state["phase_id"],
        "gt_frame_index": state["gt_frame_index"],
        "pred_frame_index": state["pred_frame_index"],
        "links": link_names,
        "views": [],
    }

    def overlay(owner: np.ndarray) -> np.ndarray:
        img = np.full((*owner.shape, 3), 255, dtype=np.uint8)
        alpha = 0.68
        for li, link in enumerate(link_names):
            mask = owner == int(li)
            if not np.any(mask):
                continue
            color = np.asarray(_debug_color(li, link), dtype=np.float32)
            img[mask] = ((1.0 - alpha) * img[mask].astype(np.float32) + alpha * color).astype(np.uint8)
        return img

    for view_id, az, el in DEFAULT_VIEWS[: int(args.num_views)]:
        camera = gop.compute_camera(center, radius, azim_deg=float(az), elev_deg=float(el))
        gt_points = _points_by_link_for_frame(raw_gt, int(state["gt_frame_index"]), gt_recipes)
        pred_points = _points_by_link_for_frame(raw_pred, int(state["pred_frame_index"]), pred_recipes)
        gt_owner, _gt_img = _render_owner_points(gt_points, link_names, camera, tuple(args.resolution), int(args.point_radius))
        pred_owner, _pred_img = _render_owner_points(pred_points, link_names, camera, tuple(args.resolution), int(args.point_radius))
        gt_overlay = overlay(gt_owner)
        pred_overlay = overlay(pred_owner)

        h, w = gt_overlay.shape[:2]
        canvas = np.full((h + 44, w * 2, 3), 245, dtype=np.uint8)
        canvas[44:, :w] = gt_overlay
        canvas[44:, w:] = pred_overlay
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        draw.text((8, 10), f"GT {view_id} frame {state['gt_frame_index']} {state['phase_id']}", fill=(0, 0, 0))
        draw.text((w + 8, 10), f"{eval_input.variant} frame {state['pred_frame_index']}", fill=(0, 0, 0))
        for li, link in enumerate(link_names[:18]):
            x = 8 + (li % 6) * 120
            y = h + 22 + (li // 6) * 18
            if y >= h + 44:
                break
            color = _debug_color(li, link)
            draw.rectangle([x, y, x + 10, y + 10], fill=color)
            draw.text((x + 14, y - 2), link, fill=(0, 0, 0))
        out_path = case_dir / f"{view_id}_{state['phase_id']}.png"
        pil.save(out_path)
        manifest["views"].append({"view_id": view_id, "path": str(out_path), "azimuth_deg": az, "elevation_deg": el})

    _write_json(case_dir / "manifest.json", manifest)
    return manifest


def _mask_bbox(mask: np.ndarray, pad: int = 2) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - int(pad))
    y0 = max(0, int(ys.min()) - int(pad))
    x1 = min(w - 1, int(xs.max()) + int(pad))
    y1 = min(h - 1, int(ys.max()) + int(pad))
    return x0, y0, x1, y1


def _masked_psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float | None:
    if int(np.count_nonzero(mask)) == 0:
        return None
    diff = gt.astype(np.float32) - pred.astype(np.float32)
    vals = diff[mask]
    if vals.size == 0:
        return None
    mse = float(np.mean(vals * vals))
    if mse <= 1.0e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def _masked_ssim(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float | None:
    bbox = _mask_bbox(mask, pad=2)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    g = gt[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    p = pred[y0 : y1 + 1, x0 : x1 + 1].astype(np.float32)
    m = mask[y0 : y1 + 1, x0 : x1 + 1]
    if int(np.count_nonzero(m)) == 0:
        return None
    bg = np.full_like(g, 255.0)
    g = np.where(m[:, :, None], g, bg)
    p = np.where(m[:, :, None], p, bg)
    wg = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    g1 = np.tensordot(g, wg, axes=([2], [0]))
    p1 = np.tensordot(p, wg, axes=([2], [0]))
    mu_g = float(np.mean(g1))
    mu_p = float(np.mean(p1))
    var_g = float(np.var(g1))
    var_p = float(np.var(p1))
    cov = float(np.mean((g1 - mu_g) * (p1 - mu_p)))
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    denom = (mu_g * mu_g + mu_p * mu_p + c1) * (var_g + var_p + c2)
    if abs(denom) <= 1.0e-12:
        return 1.0
    return float(((2.0 * mu_g * mu_p + c1) * (2.0 * cov + c2)) / denom)


def _score_link(
    gt_owner: np.ndarray,
    pred_owner: np.ndarray,
    gt_img: np.ndarray,
    pred_img: np.ndarray,
    link_idx: int,
    min_visible_px: int,
) -> dict[str, Any] | None:
    gt_mask = gt_owner == int(link_idx)
    pred_mask = pred_owner == int(link_idx)
    gt_px = int(np.count_nonzero(gt_mask))
    pred_px = int(np.count_nonzero(pred_mask))
    if max(gt_px, pred_px) < int(min_visible_px):
        return None
    inter = int(np.count_nonzero(gt_mask & pred_mask))
    union_mask = gt_mask | pred_mask
    union = int(np.count_nonzero(union_mask))
    iou = float(inter / union) if union else None
    return {
        "P_IoU": iou,
        "P_PSNR": _masked_psnr(gt_img, pred_img, union_mask),
        "P_SSIM": _masked_ssim(gt_img, pred_img, union_mask),
        "gt_visible_px": gt_px,
        "pred_visible_px": pred_px,
    }


def _load_eval_inputs(args: argparse.Namespace) -> list[EvalInput]:
    return [
        EvalInput(
            variant="full_agent",
            result_dir=Path(args.own_3d_dir),
            metrics_csv=Path(args.own_3d_dir) / "aam_metrics.csv",
            matched_frames_json=Path(args.own_3d_dir) / "diagnose" / "matched_frames.json",
            matching_dir=Path(args.own_3d_dir) / "diagnose" / "matching",
        ),
        EvalInput(
            variant="animate_anymesh",
            result_dir=Path(args.aam_3d_dir),
            metrics_csv=Path(args.aam_3d_dir) / "aam_metrics.csv",
            matched_frames_json=Path(args.aam_3d_dir) / "diagnose" / "matched_frames.json",
            matching_dir=Path(args.aam_3d_dir) / "diagnose" / "matching",
        ),
    ]


def _eval_input_to_json(eval_input: EvalInput) -> dict[str, str]:
    return {
        "variant": eval_input.variant,
        "result_dir": str(eval_input.result_dir),
        "metrics_csv": str(eval_input.metrics_csv),
        "matched_frames_json": str(eval_input.matched_frames_json),
        "matching_dir": str(eval_input.matching_dir),
    }


def _eval_input_from_json(data: dict[str, Any]) -> EvalInput:
    return EvalInput(
        variant=str(data["variant"]),
        result_dir=Path(data["result_dir"]),
        metrics_csv=Path(data["metrics_csv"]),
        matched_frames_json=Path(data["matched_frames_json"]),
        matching_dir=Path(data["matching_dir"]),
    )


def _manifest_cases(manifest_path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    data = _read_json(manifest_path)
    out = {}
    for case in data.get("cases") or []:
        out[_case_key(case)] = case
    return out


def _matched_by_key(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for row in _read_json(path):
        out[(str(row.get("asset_name") or ""), str(row.get("action_name") or ""))] = row
    return out


def _selected_states(matched: dict[str, Any], include_terminal: bool) -> list[dict[str, Any]]:
    states = []
    gt_frames = matched.get("gt_frame_indices") or []
    gt_source_frames = matched.get("gt_source_frame_indices") or []
    gt_static_glbs = matched.get("gt_static_glbs") or []
    pred_frames = matched.get("pred_frame_indices") or []
    phases = matched.get("phases") or []
    gt_times = matched.get("gt_timestamps") or []
    pred_times = matched.get("pred_timestamps") or []
    overlap_prev = matched.get("overlap_allowed_from_previous") or []
    for i, (gfi, pfi) in enumerate(zip(gt_frames, pred_frames)):
        states.append(
            {
                "state_index": i,
                "phase_id": str(phases[i] if i < len(phases) else f"phase_{i}"),
                "gt_frame_index": int(gt_source_frames[i] if i < len(gt_source_frames) else gfi),
                "gt_static_glb": str(gt_static_glbs[i]) if i < len(gt_static_glbs) and gt_static_glbs[i] else None,
                "pred_frame_index": int(pfi),
                "gt_time_s": float(gt_times[i]) if i < len(gt_times) else None,
                "pred_time_s": float(pred_times[i]) if i < len(pred_times) else None,
                "overlap_allowed_from_previous": bool(overlap_prev[i]) if i < len(overlap_prev) else False,
            }
        )
    if include_terminal and matched.get("terminal_gt_frame_index") is not None and matched.get("terminal_pred_frame_index") is not None:
        states.append(
            {
                "state_index": len(states),
                "phase_id": "__terminal_final_state__",
                "gt_frame_index": int(matched["terminal_gt_frame_index"]),
                "pred_frame_index": int(matched["terminal_pred_frame_index"]),
                "gt_time_s": float(matched["terminal_gt_timestamp"]) if matched.get("terminal_gt_timestamp") is not None else None,
                "pred_time_s": float(matched["terminal_pred_timestamp"]) if matched.get("terminal_pred_timestamp") is not None else None,
            }
        )
    # Deduplicate identical endpoint pairs while preserving phase labels.
    dedup = []
    seen = set()
    for st in states:
        key = (int(st["gt_frame_index"]), int(st["pred_frame_index"]), str(st["phase_id"]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(st)
    return dedup


def _case_2d_metrics(
    case: dict[str, Any],
    metric_row: dict[str, str],
    matched: dict[str, Any],
    eval_input: EvalInput,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cls, asset_name, action_name = _case_key(case)
    raw_gt_anim = ex._load_mesh_sequence(Path(case["gt_glb"]))
    pred_file = Path(metric_row["prediction_file"])
    raw_pred = ex._load_mesh_sequence(pred_file)

    matching_path = eval_input.matching_dir / _matching_filename(cls, asset_name, action_name)
    matching = _read_json(matching_path)

    pred_component_faces = _component_face_groups(raw_pred, int(args.min_component_faces))
    gt_link_components_anim = _target_link_components_from_matching(matching)
    pred_link_components = _link_components_from_matching(matching)

    states = _selected_states(matched, bool(args.include_terminal))
    if args.max_states and int(args.max_states) > 0:
        states = states[: int(args.max_states)]
    if not states:
        raise ValueError("No matched endpoint states")

    static_cache: dict[str, tuple[ex.RawMeshSequence, list[np.ndarray], dict[str, list[int]], dict[str, tuple[np.ndarray, np.ndarray]]]] = {}

    def static_path_for_state(state: dict[str, Any]) -> Path | None:
        raw_path = state.get("gt_static_glb")
        if not raw_path:
            return None
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = REPO_ROOT / path
        return path

    first_static_path = next((static_path_for_state(st) for st in states if static_path_for_state(st) is not None), None)
    if first_static_path is not None:
        first_raw = ex._load_mesh_sequence(first_static_path)
        first_gt_components = _link_components_from_static_names(first_raw, set(gt_link_components_anim) | set(pred_link_components))
        raw_for_camera = first_raw
        link_names = sorted(ln for ln in first_gt_components if ln in pred_link_components)
    else:
        raw_for_camera = raw_gt_anim
        link_names = sorted(ln for ln in gt_link_components_anim if ln in pred_link_components)
    if not link_names:
        raise ValueError("No common matched links for 2D scoring")

    center, radius = _scene_bounds_from_raw(raw_for_camera)
    gt_component_faces_anim = _component_face_groups(raw_gt_anim, int(args.min_component_faces))
    gt_recipes_anim = _sample_link_recipes(
        raw_gt_anim,
        {ln: gt_link_components_anim[ln] for ln in link_names if ln in gt_link_components_anim},
        gt_component_faces_anim,
        int(args.points_per_link),
        f"part2d:{case['case_id']}:animated",
    )
    pred_recipes = _sample_link_recipes(
        raw_pred,
        {ln: pred_link_components[ln] for ln in link_names},
        pred_component_faces,
        int(args.points_per_link),
        f"part2d:{case['case_id']}",
    )

    def gt_render_inputs(state: dict[str, Any]) -> tuple[ex.RawMeshSequence, int, dict[str, list[int]], list[np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
        static_path = static_path_for_state(state)
        if static_path is None:
            return raw_gt_anim, int(state["gt_frame_index"]), gt_link_components_anim, gt_component_faces_anim, gt_recipes_anim
        key = str(static_path)
        if key not in static_cache:
            raw = ex._load_mesh_sequence(static_path)
            comp_faces = _component_face_groups(raw, int(args.min_component_faces))
            link_components = _link_components_from_static_names(raw, set(gt_link_components_anim) | set(link_names))
            recipes = _sample_link_recipes(
                raw,
                {ln: link_components[ln] for ln in link_names if ln in link_components},
                comp_faces,
                int(args.points_per_link),
                f"part2d:{case['case_id']}:{static_path.name}",
            )
            static_cache[key] = (raw, comp_faces, link_components, recipes)
        raw, comp_faces, link_components, recipes = static_cache[key]
        return raw, 0, link_components, comp_faces, recipes

    per_link_rows: list[dict[str, Any]] = []
    per_state_rows: list[dict[str, Any]] = []
    for view_id, az, el in DEFAULT_VIEWS[: int(args.num_views)]:
        camera = gop.compute_camera(center, radius, azim_deg=float(az), elev_deg=float(el))
        for state in states:
            gt_raw, gt_frame_idx, gt_components, gt_faces, gt_recipes = gt_render_inputs(state)
            if str(args.render_mode) == "mesh":
                gt_meshes = _meshes_by_link_for_frame(gt_raw, gt_frame_idx, gt_components, gt_faces)
                pred_meshes = _meshes_by_link_for_frame(raw_pred, int(state["pred_frame_index"]), pred_link_components, pred_component_faces)
                gt_owner, gt_img = _render_owner(gt_meshes, link_names, camera, tuple(args.resolution))
                pred_owner, pred_img = _render_owner(pred_meshes, link_names, camera, tuple(args.resolution))
            else:
                gt_points = _points_by_link_for_frame(gt_raw, gt_frame_idx, gt_recipes)
                pred_points = _points_by_link_for_frame(raw_pred, int(state["pred_frame_index"]), pred_recipes)
                gt_owner, gt_img = _render_owner_points(gt_points, link_names, camera, tuple(args.resolution), int(args.point_radius))
                pred_owner, pred_img = _render_owner_points(pred_points, link_names, camera, tuple(args.resolution), int(args.point_radius))

            link_scores = []
            for li, link in enumerate(link_names):
                score = _score_link(gt_owner, pred_owner, gt_img, pred_img, li, int(args.min_visible_px))
                if score is None:
                    continue
                row = {
                    "case_id": case["case_id"],
                    "class": cls,
                    "asset_name": asset_name,
                    "action_name": action_name,
                    "variant": eval_input.variant,
                    "view_id": view_id,
                    "phase_id": state["phase_id"],
                    "state_index": state["state_index"],
                    "gt_frame_index": state["gt_frame_index"],
                    "pred_frame_index": state["pred_frame_index"],
                    "link": link,
                    **score,
                }
                per_link_rows.append(row)
                link_scores.append(score)
            per_state_rows.append(
                {
                    "case_id": case["case_id"],
                    "class": cls,
                    "asset_name": asset_name,
                    "action_name": action_name,
                    "variant": eval_input.variant,
                    "view_id": view_id,
                    "phase_id": state["phase_id"],
                    "state_index": state["state_index"],
                    "gt_frame_index": state["gt_frame_index"],
                    "pred_frame_index": state["pred_frame_index"],
                    "num_scored_parts": len(link_scores),
                    "P_IoU": _mean([x.get("P_IoU") for x in link_scores]),
                    "P_PSNR": _mean([x.get("P_PSNR") for x in link_scores]),
                    "P_SSIM": _mean([x.get("P_SSIM") for x in link_scores]),
                }
            )
    case_summary = {
        "case_id": case["case_id"],
        "class": cls,
        "asset_name": asset_name,
        "action_name": action_name,
        "variant": eval_input.variant,
        "num_states": len(states),
        "num_views": min(int(args.num_views), len(DEFAULT_VIEWS)),
        "num_links": len(link_names),
        "P_IoU": _mean([r.get("P_IoU") for r in per_state_rows]),
        "P_PSNR": _mean([r.get("P_PSNR") for r in per_state_rows]),
        "P_SSIM": _mean([r.get("P_SSIM") for r in per_state_rows]),
    }
    return per_link_rows, per_state_rows, case_summary


def _aggregate_case_rows(case_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variant_rows = []
    category_rows = []
    variants = sorted({str(r["variant"]) for r in case_rows})
    for variant in variants:
        rows_v = [r for r in case_rows if str(r["variant"]) == variant]
        variant_rows.append(
            {
                "class": "overall",
                "variant": variant,
                "num_cases": len(rows_v),
                "P_IoU": _mean([r.get("P_IoU") for r in rows_v]),
                "P_PSNR": _mean([r.get("P_PSNR") for r in rows_v]),
                "P_SSIM": _mean([r.get("P_SSIM") for r in rows_v]),
            }
        )
        for cls in sorted({str(r["class"]) for r in rows_v}):
            rows_c = [r for r in rows_v if str(r["class"]) == cls]
            category_rows.append(
                {
                    "class": cls,
                    "variant": variant,
                    "num_cases": len(rows_c),
                    "P_IoU": _mean([r.get("P_IoU") for r in rows_c]),
                    "P_PSNR": _mean([r.get("P_PSNR") for r in rows_c]),
                    "P_SSIM": _mean([r.get("P_SSIM") for r in rows_c]),
                }
            )
    return variant_rows, variant_rows + category_rows


def _run_worker_task(args: argparse.Namespace) -> None:
    gc.disable()
    task = _read_json(Path(args.worker_task_json))
    eval_input = _eval_input_from_json(task["eval_input"])
    case = task["case"]
    metric_row = task["metric_row"]
    matched = task["matched"]
    if args.worker_mode == "debug_overlay":
        try:
            manifest = render_case_debug_overlays(case, metric_row, matched, eval_input, args, Path(args.debug_out_dir))
            _write_json(Path(args.worker_out_json), {"manifest": manifest, "error": None})
        except Exception as exc:
            _write_json(
                Path(args.worker_out_json),
                {"manifest": None, "error": {"variant": eval_input.variant, "case": list(_case_key(case)), "error": str(exc)}},
            )
        os._exit(0)

    result: dict[str, Any] = {"per_link": [], "per_state": [], "per_case": None, "error": None}
    try:
        per_link, per_state, per_case = _case_2d_metrics(case, metric_row, matched, eval_input, args)
        result = {"per_link": per_link, "per_state": per_state, "per_case": per_case, "error": None}
    except Exception as exc:
        result["error"] = {
            "variant": eval_input.variant,
            "case": list(_case_key(case)),
            "error": str(exc),
        }
    _write_json(Path(args.worker_out_json), result)
    # ufbx can segfault during interpreter teardown after FBX loads. Workers are
    # intentionally short-lived, so skip native destructors once the JSON is safe.
    os._exit(0)


def _run_case_isolated(
    eval_input: EvalInput,
    case: dict[str, Any],
    metric_row: dict[str, str],
    matched: dict[str, Any],
    args: argparse.Namespace,
    worker_dir: Path,
    ordinal: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
    task_path = worker_dir / f"case_{ordinal:04d}.json"
    out_path = worker_dir / f"result_{ordinal:04d}.json"
    if not out_path.exists():
        _write_json(
            task_path,
            {
                "eval_input": _eval_input_to_json(eval_input),
                "case": case,
                "metric_row": metric_row,
                "matched": matched,
            },
        )
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
            "--num_views",
            str(int(args.num_views)),
            "--min_visible_px",
            str(int(args.min_visible_px)),
            "--min_component_faces",
            str(int(args.min_component_faces)),
            "--render_mode",
            str(args.render_mode),
            "--points_per_link",
            str(int(args.points_per_link)),
            "--point_radius",
            str(int(args.point_radius)),
            "--max_states",
            str(int(args.max_states)),
        ]
        cmd.append("--include_terminal" if bool(args.include_terminal) else "--no_terminal")
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            return [], [], None, {
                "variant": eval_input.variant,
                "case": list(_case_key(case)),
                "error": f"worker_failed_returncode_{proc.returncode}",
                "output": proc.stdout[-4000:],
            }
    if not out_path.exists():
        return [], [], None, {
            "variant": eval_input.variant,
            "case": list(_case_key(case)),
            "error": "worker_missing_output",
        }
    result = _read_json(out_path)
    if result.get("error"):
        return [], [], None, result["error"]
    return list(result.get("per_link") or []), list(result.get("per_state") or []), result.get("per_case"), None


def main() -> None:
    parser = argparse.ArgumentParser(description="Part-level 2D endpoint evaluation for mesh-sequence methods.")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "experiments" / "final_3d_evaluation" / "ablation_3d" / "diagnose" / "resolved_manifest.json")
    parser.add_argument("--own_3d_dir", type=Path, default=REPO_ROOT / "experiments" / "final_3d_evaluation" / "own_method_3d_matching")
    parser.add_argument("--aam_3d_dir", type=Path, default=REPO_ROOT / "experiments" / "final_3d_evaluation" / "animate_anymesh_3d")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "experiments" / "final_2d_evaluation")
    parser.add_argument("--resolution", type=int, nargs=2, default=(256, 256), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--render_mode", choices=["points", "mesh"], default="points")
    parser.add_argument("--points_per_link", type=int, default=3500)
    parser.add_argument("--point_radius", type=int, default=2)
    parser.add_argument("--num_views", type=int, default=4)
    parser.add_argument("--min_visible_px", type=int, default=24)
    parser.add_argument("--min_component_faces", type=int, default=1)
    parser.add_argument("--include_terminal", action="store_true", default=True)
    parser.add_argument("--no_terminal", action="store_false", dest="include_terminal")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--max_states", type=int, default=0)
    parser.add_argument("--variants", nargs="*", default=["full_agent", "animate_anymesh"], choices=["full_agent", "animate_anymesh"])
    parser.add_argument("--worker_task_json", type=Path, default=None)
    parser.add_argument("--worker_out_json", type=Path, default=None)
    parser.add_argument("--worker_mode", choices=["metrics", "debug_overlay"], default="metrics")
    parser.add_argument("--debug_out_dir", type=Path, default=REPO_ROOT / "experiments" / "final_2d_evaluation" / "debug_overlays")
    parser.add_argument("--debug_state", default="terminal", help="Endpoint to visualize: first, terminal, or numeric state index.")
    args = parser.parse_args()

    if args.worker_task_json is not None:
        if args.worker_out_json is None:
            raise ValueError("--worker_out_json is required with --worker_task_json")
        _run_worker_task(args)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases_by_key = _manifest_cases(Path(args.manifest))
    eval_inputs = [ei for ei in _load_eval_inputs(args) if ei.variant in set(args.variants)]

    all_case_rows: list[dict[str, Any]] = []
    all_state_rows: list[dict[str, Any]] = []
    all_link_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    worker_dir = args.out_dir / "diagnose" / "workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    ordinal = 0
    for eval_input in eval_inputs:
        metric_rows = _read_csv_rows(eval_input.metrics_csv)
        matched_rows = _matched_by_key(eval_input.matched_frames_json)
        if int(args.max_cases) > 0:
            metric_rows = metric_rows[: int(args.max_cases)]
        for idx, metric_row in enumerate(metric_rows, start=1):
            ordinal += 1
            key = _case_key(metric_row)
            case = cases_by_key.get(key)
            matched = matched_rows.get((key[1], key[2]))
            if case is None or matched is None:
                errors.append({"variant": eval_input.variant, "case": list(key), "error": "missing_manifest_or_matched_frames"})
                continue
            try:
                if eval_input.variant == "animate_anymesh":
                    per_link, per_state, per_case, error = _run_case_isolated(
                        eval_input, case, metric_row, matched, args, worker_dir, ordinal
                    )
                    if error is not None:
                        errors.append(error)
                        print(f"[ERR] {eval_input.variant} {idx}/{len(metric_rows)} {key[1]}/{key[2]}: {error.get('error')}", flush=True)
                        continue
                    assert per_case is not None
                else:
                    per_link, per_state, per_case = _case_2d_metrics(case, metric_row, matched, eval_input, args)
                all_link_rows.extend(per_link)
                all_state_rows.extend(per_state)
                all_case_rows.append(per_case)
                print(
                    f"[OK] {eval_input.variant} {idx}/{len(metric_rows)} {key[1]}/{key[2]} "
                    f"P-IoU={per_case['P_IoU']:.6f} P-SSIM={per_case['P_SSIM']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                errors.append({"variant": eval_input.variant, "case": list(key), "error": str(exc)})
                print(f"[ERR] {eval_input.variant} {idx}/{len(metric_rows)} {key[1]}/{key[2]}: {exc}", flush=True)

    variant_summary, category_rows = _aggregate_case_rows(all_case_rows)
    summary = {
        "metrics": ["P_IoU", "P_PSNR", "P_SSIM"],
        "notes": [
            "Masks are propagated by matched 3D component faces and re-projected at endpoint frames.",
            "RGB metrics use deterministic per-link color renders; LPIPS is not computed because lpips is not installed in casual_agent.",
            "Endpoint matching reuses the corresponding final 3D matched_frames.json files.",
        ],
        "variants": {str(r["variant"]): r for r in variant_summary},
        "num_case_rows": len(all_case_rows),
        "num_state_rows": len(all_state_rows),
        "num_link_rows": len(all_link_rows),
        "num_errors": len(errors),
        "errors": errors,
        "resolution": list(args.resolution),
        "render_mode": str(args.render_mode),
        "points_per_link": int(args.points_per_link),
        "point_radius": int(args.point_radius),
        "views": [{"id": v[0], "azimuth_deg": v[1], "elevation_deg": v[2]} for v in DEFAULT_VIEWS[: int(args.num_views)]],
    }

    _write_csv(args.out_dir / "part2d_case_metrics.csv", all_case_rows, ["case_id", "class", "asset_name", "action_name", "variant", "num_states", "num_views", "num_links", "P_IoU", "P_PSNR", "P_SSIM"])
    _write_csv(args.out_dir / "part2d_state_metrics.csv", all_state_rows, ["case_id", "class", "asset_name", "action_name", "variant", "view_id", "phase_id", "state_index", "gt_frame_index", "pred_frame_index", "num_scored_parts", "P_IoU", "P_PSNR", "P_SSIM"])
    _write_csv(args.out_dir / "part2d_link_metrics.csv", all_link_rows, ["case_id", "class", "asset_name", "action_name", "variant", "view_id", "phase_id", "state_index", "gt_frame_index", "pred_frame_index", "link", "P_IoU", "P_PSNR", "P_SSIM", "gt_visible_px", "pred_visible_px"])
    _write_csv(args.out_dir / "part2d_category_mean.csv", category_rows, ["class", "variant", "num_cases", "P_IoU", "P_PSNR", "P_SSIM"])
    _write_csv(args.out_dir / "part2d_variant_summary.csv", variant_summary, ["class", "variant", "num_cases", "P_IoU", "P_PSNR", "P_SSIM"])
    _write_json(args.out_dir / "diagnose" / "summary.json", summary)
    _write_json(args.out_dir / "diagnose" / "errors.json", errors)
    print(json.dumps(summary["variants"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
