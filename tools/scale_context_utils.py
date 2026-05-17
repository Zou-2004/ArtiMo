#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import trimesh


def _extents_from_meshes(link_meshes: dict[str, list]) -> tuple[list[float], float, dict[str, list[float]]]:
    mins = []
    maxs = []
    per_link = {}
    for ln, meshes in (link_meshes or {}).items():
        link_mins = []
        link_maxs = []
        for m in meshes or []:
            try:
                b = np.asarray(m.bounds, dtype=float)
            except Exception:
                continue
            if b.shape != (2, 3):
                continue
            link_mins.append(b[0])
            link_maxs.append(b[1])
            mins.append(b[0])
            maxs.append(b[1])
        if link_mins and link_maxs:
            lo = np.min(np.stack(link_mins, axis=0), axis=0)
            hi = np.max(np.stack(link_maxs, axis=0), axis=0)
            per_link[str(ln)] = [float(x) for x in (hi - lo)]
    if mins and maxs:
        lo = np.min(np.stack(mins, axis=0), axis=0)
        hi = np.max(np.stack(maxs, axis=0), axis=0)
        ext = hi - lo
        diag = float(np.linalg.norm(ext))
        return [float(x) for x in ext], diag, per_link
    return [0.0, 0.0, 0.0], 0.0, per_link


def _scene_extents(scene) -> tuple[list[float] | None, float | None]:
    if scene is None:
        return None, None
    try:
        if isinstance(scene, trimesh.Trimesh):
            b = np.asarray(scene.bounds, dtype=float)
        else:
            if not getattr(scene, "geometry", None):
                return None, None
            b = np.asarray(scene.bounds, dtype=float)
        if b.shape != (2, 3):
            return None, None
        ext = b[1] - b[0]
        return [float(x) for x in ext], float(np.linalg.norm(ext))
    except Exception:
        return None, None


def _load_glb_scene(glb_path: Path | None):
    if glb_path is None:
        return None
    p = Path(glb_path)
    if not p.exists():
        return None
    try:
        scene = trimesh.load(p, force="scene", process=False)
        if isinstance(scene, trimesh.Scene) and scene.geometry:
            return scene
        if isinstance(scene, trimesh.Trimesh):
            sc = trimesh.Scene()
            sc.add_geometry(scene)
            return sc
    except Exception:
        return None
    return None


def build_scale_context(
    asset_name: str,
    world_link_meshes: dict[str, list],
    joints: list[dict] | None = None,
    glb_path: Path | None = None,
) -> dict:
    obj_ext, obj_diag, link_ext = _extents_from_meshes(world_link_meshes)
    joint_child_ext = {}
    for j in joints or []:
        jn = str(j.get("name") or "")
        child = str(j.get("child") or "")
        if not jn or not child:
            continue
        if child in link_ext:
            joint_child_ext[jn] = list(link_ext[child])

    glb_scene = _load_glb_scene(glb_path)
    glb_ext, glb_diag = _scene_extents(glb_scene)
    ratio = None
    if glb_diag is not None and obj_diag > 1e-9:
        ratio = float(glb_diag / obj_diag)

    # Heuristic "small rotating part radius" estimate for downstream prompts/checks.
    candidate_radii = []
    for j in joints or []:
        jtype = str(j.get("type") or "").lower()
        if jtype not in {"revolute", "continuous"}:
            continue
        ext = joint_child_ext.get(str(j.get("name") or ""))
        if not ext or len(ext) != 3:
            continue
        ex, ey, ez = [float(x) for x in ext]
        candidate_radii.append(0.5 * max(min(ex, ey), min(ex, ez), min(ey, ez)))
    candidate_radii = [r for r in candidate_radii if r > 1e-6]
    median_revolute_child_radius = float(np.median(candidate_radii)) if candidate_radii else None

    return {
        "asset": str(asset_name),
        "unit_assumption": "meters_like_urdf_units",
        "object_bbox_extents_m": obj_ext,
        "object_diag_m": float(obj_diag),
        "link_bbox_extents_m": link_ext,
        "joint_child_link_bbox_extents_m": joint_child_ext,
        "median_revolute_child_radius_m_est": median_revolute_child_radius,
        "reference_glb_path": str(Path(glb_path).absolute()) if glb_path else None,
        "reference_glb_bbox_extents_m": glb_ext,
        "reference_glb_diag_m": glb_diag,
        "glb_to_urdf_scale_ratio": ratio,
    }


def save_scale_context(path: Path, scale_context: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scale_context, ensure_ascii=False, indent=2), encoding="utf-8")
