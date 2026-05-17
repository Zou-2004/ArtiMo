#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import trimesh

import blender_render as br
import gen_overlays_and_prompts as gop
import run_plan as rp
import torch_accel as tacc

AZIMUTH_SET = list(range(0, 360, 30))
ELEVATION_SET = [10, 20, 30, 45, 60]
DISTANCE_SCALE_SET = [0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
FOV_SET = [25, 35, 45]
DEFAULT_VIEWSPECS = {
    "look_at_mode": "object_center",
    "views": [
        {"id": "V1", "azimuth_deg": 0, "elevation_deg": 20, "distance_scale": 1.0, "fov_deg": 35},
        {"id": "V2", "azimuth_deg": 90, "elevation_deg": 20, "distance_scale": 1.0, "fov_deg": 35},
        {"id": "V3", "azimuth_deg": 180, "elevation_deg": 20, "distance_scale": 1.0, "fov_deg": 35},
        {"id": "V4", "azimuth_deg": 270, "elevation_deg": 20, "distance_scale": 1.0, "fov_deg": 35},
    ],
}
# Label scale used in motion/timeline grids.
LABEL_SCALE_MOTION = 4
# Label scale used in coverage grids.
LABEL_SCALE_COVERAGE = max(1, int(os.environ.get("CODEX_LABEL_SCALE_COVERAGE", str(LABEL_SCALE_MOTION))))
# Header text scale used for grid tile headers.
HEADER_SCALE = 2
# Caption scale for the whole motion grid title block.
MOTION_GRID_CAPTION_SCALE = int(os.environ.get("CODEX_MOTION_GRID_CAPTION_SCALE", "1"))
# Default label mode used in motion renders ("id" or "name").
MOTION_LABEL_MODE_DEFAULT = "id"
# Multiplier applied to camera radius in motion renders.
MOTION_CAMERA_RADIUS_SCALE = float(os.environ.get("CODEX_MOTION_CAMERA_RADIUS_SCALE", "1.0"))
# Minimum pixel length for the dedicated START->END flow arrow.
MOTION_ARROW_MIN_PX = int(os.environ.get("CODEX_MOTION_ARROW_MIN_PX", "44"))
# Maximum pixel length for the dedicated START->END flow arrow.
MOTION_ARROW_MAX_PX = int(os.environ.get("CODEX_MOTION_ARROW_MAX_PX", "120"))
# Stroke thickness of the dedicated flow arrow.
MOTION_ARROW_THICKNESS = int(os.environ.get("CODEX_MOTION_ARROW_THICKNESS", "4"))
# Visual scaling factor used when converting motion magnitude to arrow length.
MOTION_ARROW_VIS_SCALE = float(os.environ.get("CODEX_MOTION_ARROW_VIS_SCALE", "5.0"))
# Text scale for START/END flow labels.
MOTION_ARROW_TEXT_SCALE = int(os.environ.get("CODEX_MOTION_ARROW_TEXT_SCALE", "3"))
# Margin reserved around the flow arrow box and captions.
MOTION_ARROW_BOX_MARGIN = int(os.environ.get("CODEX_MOTION_ARROW_BOX_MARGIN", "10"))
# Minimum apparent rotation (degrees) before a rotation cue is considered meaningful.
MOTION_ROTATE_MIN_DEG = float(os.environ.get("CODEX_MOTION_ROTATE_MIN_DEG", "9.0"))
# Background-difference threshold used to build occupancy masks for label/indicator placement.
MOTION_OCCUPANCY_BG_DELTA = int(os.environ.get("CODEX_MOTION_OCCUPANCY_BG_DELTA", "10"))
# Thickness used for motion-trace polylines/arcs when a line-style trace is rendered.
MOTION_TRACE_THICKNESS = int(os.environ.get("CODEX_MOTION_TRACE_THICKNESS", "2"))
# Alpha used for faint future-pose ghost overlays in coverage renders.
COVERAGE_GHOST_ALPHA = float(os.environ.get("CODEX_COVERAGE_GHOST_ALPHA", "0.34"))
# Draw bbox overlays in coverage-loop reference renders.
COVERAGE_DRAW_BBOX = str(os.environ.get("CODEX_COVERAGE_DRAW_BBOX", "1")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Box computation mode for coverage-loop reference renders.
COVERAGE_BOX_MODE = str(os.environ.get("CODEX_COVERAGE_BOX_MODE", "points")).strip().lower()
# Box computation mode for motion-loop/timeline reference renders.
# Keep raster as default to preserve the previous visual behavior; faster point-mode
# remains available as an explicit opt-in for experiments.
MOTION_BOX_MODE = str(os.environ.get("CODEX_MOTION_BOX_MODE", "raster")).strip().lower()
# Radius of regular sampled trace dots along the optical-flow trajectory.
MOTION_TRACE_POINT_RADIUS = int(os.environ.get("CODEX_MOTION_TRACE_POINT_RADIUS", "5"))
# Radius of the larger solid START/END dots on the trajectory.
MOTION_TRACE_START_POINT_RADIUS = int(os.environ.get("CODEX_MOTION_TRACE_START_POINT_RADIUS", "10"))
# If a projected trace path is shorter than this many pixels, add an explicit
# local trend arrow near that link because optical-flow alone can be ambiguous.
MOTION_SMALL_TRACE_PATH_THRESHOLD_PX = float(os.environ.get("CODEX_MOTION_SMALL_TRACE_PATH_THRESHOLD_PX", "28.0"))
# Number of independently tracked points per moving link.
MOTION_TRACE_SAMPLES_PER_LINK = int(os.environ.get("CODEX_MOTION_TRACE_SAMPLES_PER_LINK", "1"))
# Temporal stride, in frames, between sampled trace points.
MOTION_TRACE_SAMPLE_EVERY_FRAMES = int(os.environ.get("CODEX_MOTION_TRACE_SAMPLE_EVERY_FRAMES", "2"))
# Number of candidate perimeter points considered when selecting one tracked point on a rotating part.
MOTION_TRACE_CANDIDATE_COUNT_SINGLE = int(os.environ.get("CODEX_MOTION_TRACE_CANDIDATE_COUNT_SINGLE", "12"))
# Force motion diagnostics to use a single canonical front view instead of the selected view set.
MOTION_FRONT_VIEW_ONLY = str(os.environ.get("CODEX_MOTION_FRONT_VIEW_ONLY", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Maximum number of mesh points sampled per link when building motion bbox/visibility geometry.
MOTION_BBOX_MAX_POINTS = int(os.environ.get("CODEX_MOTION_BBOX_MAX_POINTS", "4000"))
# Timeline single-link bbox scale relative to the projected link box.
TIMELINE_LINK_BBOX_SCALE = float(os.environ.get("CODEX_TIMELINE_LINK_BBOX_SCALE", "1.1"))
# Timeline whole-asset bbox scale relative to the projected asset box.
TIMELINE_ASSET_BBOX_SCALE = float(os.environ.get("CODEX_TIMELINE_ASSET_BBOX_SCALE", "1.1"))

_TRAJECTORY_CACHE: dict[tuple[str, int, int], dict] = {}


def _put_limited_cache(cache_root: dict, key, value, max_entries: int = 8) -> None:
    cache_root[key] = value
    if len(cache_root) <= int(max_entries):
        return
    try:
        oldest = next(iter(cache_root))
        if oldest != key:
            cache_root.pop(oldest, None)
    except Exception:
        pass


def deterministic_color(link_name: str):
    return gop._deterministic_color(link_name)


def validate_viewspecs(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("viewspecs must be object")
    views = spec.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("viewspecs.views must be non-empty list")
    out = {"look_at_mode": spec.get("look_at_mode", "object_center"), "views": []}
    for i, v in enumerate(views):
        az = int(v.get("azimuth_deg"))
        el = int(v.get("elevation_deg"))
        ds = float(v.get("distance_scale"))
        fov = int(v.get("fov_deg"))
        if az not in AZIMUTH_SET:
            raise ValueError(f"azimuth_deg not allowed: {az}")
        if el not in ELEVATION_SET:
            raise ValueError(f"elevation_deg not allowed: {el}")
        if not np.isfinite(ds) or ds <= 0.0:
            raise ValueError(f"distance_scale invalid: {ds}")
        if fov not in FOV_SET:
            raise ValueError(f"fov_deg not allowed: {fov}")
        out["views"].append(
            {
                "id": str(v.get("id") or f"V{i+1}"),
                "azimuth_deg": az,
                "elevation_deg": el,
                "distance_scale": ds,
                "fov_deg": fov,
            }
        )
    return out


def load_asset_context(asset_root: Path) -> dict:
    urdf_path = next(asset_root.rglob("*.urdf"), None)
    if urdf_path is None:
        raise FileNotFoundError(f"No URDF found under {asset_root}")
    links, joints = rp.parse_urdf(urdf_path)
    link_meshes = rp.load_link_meshes(links, urdf_path.parent, textured=False)
    try:
        link_meshes_textured = rp.load_link_meshes(links, urdf_path.parent, textured=True)
    except Exception:
        link_meshes_textured = None
    rest_link_tf = rp.compute_link_transforms(links, joints, {})
    reference_glb_scene = None
    glb_link_rest_meshes = None
    glb_link_to_nodes = None
    ref_glb = None
    try:
        ref_glb = gop.find_reference_glb_path(asset_root)
        if ref_glb is not None:
            reference_glb_scene = trimesh.load(ref_glb, force="scene", process=False)
            if isinstance(reference_glb_scene, trimesh.Trimesh):
                _s = trimesh.Scene()
                _s.add_geometry(reference_glb_scene)
                reference_glb_scene = _s
            if not isinstance(reference_glb_scene, trimesh.Scene) or not reference_glb_scene.geometry:
                reference_glb_scene = None
    except Exception:
        reference_glb_scene = None
    if reference_glb_scene is not None and gop.scene_has_effective_textures(reference_glb_scene):
        try:
            mapped = _map_textured_glb_nodes_to_links(reference_glb_scene, links, link_meshes, rest_link_tf)
            if isinstance(mapped, dict):
                glb_link_rest_meshes = mapped.get("meshes")
                glb_link_to_nodes = mapped.get("node_names")
        except Exception:
            glb_link_rest_meshes = None
            glb_link_to_nodes = None
    return {
        "asset_root": asset_root,
        "urdf_path": urdf_path,
        "links": links,
        "joints": joints,
        "link_meshes": link_meshes,
        "link_meshes_textured": link_meshes_textured,
        "rest_link_tf": rest_link_tf,
        "reference_glb_path": ref_glb,
        "reference_glb_scene": reference_glb_scene,
        "glb_link_rest_meshes": glb_link_rest_meshes,
        "glb_link_to_nodes": glb_link_to_nodes,
        "joint_by_name": {j.get("name"): j for j in joints},
    }


def _shape_feature(mesh: trimesh.Trimesh):
    ext = np.asarray(mesh.bounding_box.extents, dtype=np.float32)
    m = float(np.max(ext))
    if m <= 1e-9:
        return None
    return np.sort(ext / m)


def _scene_node_world_meshes(scene: trimesh.Scene):
    out = []
    for node_name in scene.graph.nodes_geometry:
        tf, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name].copy()
        geom.apply_transform(tf)
        out.append((str(node_name), geom))
    return out


def _map_textured_glb_nodes_to_links(
    reference_glb_scene: trimesh.Scene,
    links: dict,
    link_meshes: dict[str, list],
    rest_link_tf: dict[str, np.ndarray],
):
    # Prefer deterministic Particulate mapping when node names are part_node_i.
    # This avoids ambiguous shape-matching for repeated parts (e.g., many wheels).
    try:
        by_order = rp.match_links_to_nodes_particulate_by_order(links, link_meshes, reference_glb_scene)
    except Exception:
        by_order = None
    if isinstance(by_order, dict) and by_order:
        mapping = {}
        node_name_mapping = {}
        for ln, node_names in by_order.items():
            meshes = []
            for nn in (node_names or []):
                try:
                    tf, geom_name = reference_glb_scene.graph[str(nn)]
                    geom = reference_glb_scene.geometry[geom_name].copy()
                    geom.apply_transform(tf)
                    meshes.append(geom)
                except Exception:
                    continue
            if meshes:
                mapping[ln] = meshes
                node_name_mapping[ln] = [str(x) for x in node_names]
        if mapping:
            return {"meshes": mapping, "node_names": node_name_mapping}

    # Fallback for non-part_node GLBs: shape-based matching.
    world_link_meshes = transform_link_meshes(link_meshes, rest_link_tf)
    visual_links = []
    for ln, meshes in world_link_meshes.items():
        if not meshes:
            continue
        merged = trimesh.util.concatenate([m.copy() for m in meshes if m.vertices.size > 0])
        if merged.vertices.size == 0:
            continue
        feat = _shape_feature(merged)
        if feat is None:
            continue
        visual_links.append((ln, merged, feat))
    nodes = []
    for nn, mesh in _scene_node_world_meshes(reference_glb_scene):
        feat = _shape_feature(mesh)
        if feat is None:
            continue
        nodes.append((nn, mesh, feat))
    if len(visual_links) != len(nodes) or not visual_links:
        return None
    candidates = []
    for li, (_ln, _lm, lf) in enumerate(visual_links):
        for ni, (_nn, _nm, nf) in enumerate(nodes):
            candidates.append((float(np.linalg.norm(lf - nf)), li, ni))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    used_l, used_n = set(), set()
    mapping = {}
    node_name_mapping = {}
    for cost, li, ni in candidates:
        if li in used_l or ni in used_n:
            continue
        used_l.add(li)
        used_n.add(ni)
        ln = visual_links[li][0]
        mapping[ln] = [nodes[ni][1].copy()]
        node_name_mapping[ln] = [str(nodes[ni][0])]
    if len(mapping) != len(visual_links):
        return None
    return {"meshes": mapping, "node_names": node_name_mapping}


def _build_motion_textured_scene_from_glb_mapping(asset_ctx: dict, link_tf: dict[str, np.ndarray], cache_key=None):
    if cache_key is not None:
        cache_root = asset_ctx.get("_motion_glb_frame_cache")
        if isinstance(cache_root, dict) and cache_key in cache_root:
            cached = cache_root.get(cache_key)
            if isinstance(cached, tuple) and len(cached) == 3:
                return cached
    glb_link_rest_meshes = asset_ctx.get("glb_link_rest_meshes")
    rest_link_tf = asset_ctx.get("rest_link_tf")
    if not isinstance(glb_link_rest_meshes, dict) or not isinstance(rest_link_tf, dict):
        return None, None, None
    scene = trimesh.Scene()
    points_by_link = {}
    meshes_by_link = {}
    for ln, meshes in glb_link_rest_meshes.items():
        if ln not in link_tf or ln not in rest_link_tf:
            continue
        try:
            delta = np.asarray(link_tf[ln], dtype=float) @ np.linalg.inv(np.asarray(rest_link_tf[ln], dtype=float))
        except Exception:
            delta = np.eye(4)
        moved_meshes = []
        for m in meshes:
            mc = m.copy()
            mc.apply_transform(delta)
            moved_meshes.append(mc)
            scene.add_geometry(mc)
        if moved_meshes:
            meshes_by_link[ln] = [m.copy() for m in moved_meshes]
        if moved_meshes:
            try:
                pts = [np.asarray(m.vertices, dtype=np.float32) for m in moved_meshes if getattr(m, "vertices", None) is not None and m.vertices.size > 0]
                if pts:
                    all_pts = np.concatenate(pts, axis=0) if len(pts) > 1 else pts[0]
                    points_by_link[ln] = gop._downsample_points_deterministic(
                        all_pts, gop.BBOX_POINTS_PER_LINK, f"motion-refbbox:{ln}"
                    )
                else:
                    points_by_link[ln] = np.zeros((0, 3), dtype=np.float32)
            except Exception:
                points_by_link[ln] = np.zeros((0, 3), dtype=np.float32)
    if not scene.geometry:
        return None, None, None
    out = (scene, points_by_link, meshes_by_link)
    if cache_key is not None:
        cache_root = asset_ctx.get("_motion_glb_frame_cache")
        if not isinstance(cache_root, dict):
            cache_root = {}
            asset_ctx["_motion_glb_frame_cache"] = cache_root
        _put_limited_cache(cache_root, cache_key, out, max_entries=10)
    return out


def _build_scene_from_world_link_meshes(link_meshes_world: dict[str, list]):
    scene = trimesh.Scene()
    for ms in (link_meshes_world or {}).values():
        for m in (ms or []):
            try:
                scene.add_geometry(m.copy())
            except Exception:
                continue
    return scene if scene.geometry else None


def _compute_glb_node_transforms_for_frame(asset_ctx: dict, link_tf: dict[str, np.ndarray]):
    rest_link_tf = asset_ctx.get("rest_link_tf")
    ref_scene = asset_ctx.get("reference_glb_scene")
    glb_link_to_nodes = asset_ctx.get("glb_link_to_nodes")
    if not isinstance(rest_link_tf, dict) or ref_scene is None or not isinstance(glb_link_to_nodes, dict):
        return None
    node_rest = {}
    try:
        for node_name in ref_scene.graph.nodes_geometry:
            node_rest[str(node_name)] = np.asarray(ref_scene.graph[node_name][0], dtype=float)
    except Exception:
        return None
    out = {}
    for ln, nodes in glb_link_to_nodes.items():
        if ln not in link_tf or ln not in rest_link_tf:
            continue
        try:
            delta = np.asarray(link_tf[ln], dtype=float) @ np.linalg.inv(np.asarray(rest_link_tf[ln], dtype=float))
        except Exception:
            delta = np.eye(4)
        for nn in (nodes or []):
            nn = str(nn)
            if nn not in node_rest:
                continue
            out[nn] = (delta @ node_rest[nn]).tolist()
    return out if out else None


def _render_views_with_blender(
    asset_ctx: dict,
    link_tf: dict[str, np.ndarray],
    cams: list[tuple],
    views: list[dict],
    resolution,
    animated_glb_path: str | Path | None = None,
    animated_frame_idx: int | None = None,
    animated_fps: int | None = None,
):
    payload_views = []
    for idx, (cam, view) in enumerate(zip(cams, views)):
        eye, target, up = cam
        payload_views.append(
            {
                "id": str(view.get("id", f"V{idx+1}")),
                "eye": np.asarray(eye, dtype=float).tolist(),
                "target": np.asarray(target, dtype=float).tolist(),
                "up": np.asarray(up, dtype=float).tolist(),
            }
        )
    textured_src = asset_ctx.get("link_meshes_textured")
    scene_urdf = None
    if isinstance(textured_src, dict):
        try:
            world_link_meshes_textured = transform_link_meshes(textured_src, link_tf)
            scene_urdf = _build_scene_from_world_link_meshes(world_link_meshes_textured)
        except Exception:
            scene_urdf = None
    # First choice for motion diagnostics: render directly from this iteration's
    # exported animated GLB frame so the image is guaranteed to match plan_animated.
    if animated_glb_path is not None and Path(animated_glb_path).exists():
        try:
            imgs = br.render_views_from_glb(
                animated_glb_path,
                payload_views,
                tuple(int(x) for x in resolution),
                fov_deg=50.0,
                frame_idx=(int(animated_frame_idx) if animated_frame_idx is not None else None),
                fps=(int(animated_fps) if animated_fps is not None else None),
                keep_animation=True,
            )
            out = []
            for img in imgs:
                try:
                    img = gop.enhance_textured_image(img)
                except Exception:
                    pass
                out.append(img)
            try:
                scene_for_frame, _pts_for_frame, _meshes_for_frame = _build_motion_textured_scene_from_glb_mapping(asset_ctx, link_tf)
            except Exception:
                scene_for_frame = None
            out, effective_backend = _prefer_software_when_blender_washed_out(
                out,
                scene_for_frame,
                cams,
                resolution,
                "Blender animated-GLB motion render looks washed out",
            )
            if effective_backend == "blender" and any(gop.is_reference_image_too_dark(img) for img in out):
                print("[WARN] Blender animated-GLB motion render looks dark; keeping it to preserve exact trajectory alignment.")
            return out, effective_backend
        except Exception as exc:
            print(f"[WARN] Blender animated-GLB motion render failed: {exc}; trying fallback paths.")

    # Fallback 1: canonical GLB + per-node transforms.
    glb_path = asset_ctx.get("reference_glb_path")
    node_transforms = _compute_glb_node_transforms_for_frame(asset_ctx, link_tf)
    if glb_path is not None and Path(glb_path).exists() and node_transforms:
        try:
            imgs = br.render_views_from_glb(
                glb_path,
                payload_views,
                tuple(int(x) for x in resolution),
                fov_deg=50.0,
                node_transforms=node_transforms,
            )
            out = []
            for img in imgs:
                try:
                    img = gop.enhance_textured_image(img)
                except Exception:
                    pass
                out.append(img)
            scene_for_frame = None
            try:
                scene_for_frame, _pts_for_frame, _meshes_for_frame = _build_motion_textured_scene_from_glb_mapping(asset_ctx, link_tf)
            except Exception:
                scene_for_frame = None
            out, effective_backend = _prefer_software_when_blender_washed_out(
                out,
                scene_for_frame,
                cams,
                resolution,
                "Blender GLB motion render looks washed out",
            )
            if not any(gop.is_reference_image_too_dark(img) for img in out):
                return out, effective_backend
            print("[WARN] Blender GLB motion render outputs are too dark; trying scene-based fallback.")
        except Exception as exc:
            print(f"[WARN] Blender GLB motion render failed: {exc}; trying scene-based fallback.")

    # Final textured fallback: render from URDF-assembled scene geometry.
    if scene_urdf is not None and len(scene_urdf.geometry) > 0 and gop.scene_has_effective_textures(scene_urdf):
        try:
            imgs = br.render_views_from_scene(
                scene_urdf,
                payload_views,
                tuple(int(x) for x in resolution),
                fov_deg=50.0,
            )
            out = []
            for img in imgs:
                try:
                    img = gop.enhance_textured_image(img)
                except Exception:
                    pass
                out.append(img)
            out, effective_backend = _prefer_software_when_blender_washed_out(
                out,
                scene_urdf,
                cams,
                resolution,
                "Blender textured-URDF motion render looks washed out",
            )
            if effective_backend == "blender" and not any(gop.is_reference_image_too_dark(img) for img in out):
                return out, effective_backend
            print("[WARN] Blender textured-URDF motion render looks dark; trying scene-based fallback.")
        except Exception as exc:
            print(f"[WARN] Blender textured-URDF motion render failed: {exc}; trying scene-based fallback.")

    # Fallback: render from per-frame assembled scene geometry.
    try:
        scene_for_frame, _pts_for_frame, _meshes_for_frame = _build_motion_textured_scene_from_glb_mapping(asset_ctx, link_tf)
    except Exception:
        scene_for_frame = None
    if scene_for_frame is not None and len(scene_for_frame.geometry) > 0:
        try:
            imgs = br.render_views_from_scene(
                scene_for_frame,
                payload_views,
                tuple(int(x) for x in resolution),
                fov_deg=50.0,
            )
            out = []
            for img in imgs:
                try:
                    img = gop.enhance_textured_image(img)
                except Exception:
                    pass
                out.append(img)
            out, effective_backend = _prefer_software_when_blender_washed_out(
                out,
                scene_for_frame,
                cams,
                resolution,
                "Blender scene-based motion render looks washed out",
            )
            if effective_backend == "blender" and any(gop.is_reference_image_too_dark(img) for img in out):
                raise RuntimeError("Blender scene-based outputs are too dark")
            return out, effective_backend
        except Exception as exc:
            print(f"[WARN] Blender scene-based motion render failed: {exc}")
    return None, None


def _render_views_with_software_scene(scene, cams, resolution):
    if scene is None or len(getattr(scene, "geometry", {})) == 0:
        return None
    out = []
    for cam in cams:
        img = gop.render_reference_textured(scene, cam, resolution)
        out.append(np.asarray(img, dtype=np.uint8))
    return out


def _prefer_software_when_blender_washed_out(blender_imgs, software_scene, cams, resolution, warn_label: str):
    if not blender_imgs:
        return blender_imgs, "blender"
    if not any(gop.is_reference_image_too_dark(img) or gop.reference_image_may_be_washed_out(img) for img in blender_imgs):
        return blender_imgs, "blender"
    dark_views = [idx + 1 for idx, img in enumerate(blender_imgs) if gop.is_reference_image_too_dark(img)]
    washed_out_views = [idx + 1 for idx, img in enumerate(blender_imgs) if gop.reference_image_may_be_washed_out(img)]
    detail_parts = []
    if dark_views:
        detail_parts.append(f"dark_views={dark_views}")
    if washed_out_views:
        detail_parts.append(f"washed_out_views={washed_out_views}")
    detail_suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
    print(f"[WARN] {warn_label}; Blender-only mode keeps Blender output{detail_suffix}.")
    return blender_imgs, "blender"


def compute_base_center_radius(link_meshes: dict[str, list]) -> tuple[np.ndarray, float]:
    lm = {k: [m.copy() for m in v] for k, v in link_meshes.items()}
    return gop.compute_scene_bounds(lm)


def compute_camera_for_viewspec(center, radius, view: dict):
    eye, target, up = gop.compute_camera(center, radius, azim_deg=view["azimuth_deg"], elev_deg=view["elevation_deg"])
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    eye = target + (eye - target) * float(view.get("distance_scale", 1.0))
    return eye, target, up


def transform_link_meshes(link_meshes: dict[str, list], link_transforms: dict[str, np.ndarray]) -> dict[str, list]:
    out = {}
    for ln, meshes in link_meshes.items():
        tf = link_transforms.get(ln, np.eye(4))
        out_meshes = []
        for m in meshes:
            mc = m.copy()
            mc.apply_transform(tf)
            out_meshes.append(mc)
        out[ln] = out_meshes
    return out


def _project_point_masks(points_by_link, colors_by_link, camera, resolution):
    width, height = resolution
    torch_raster = tacc.rasterize_points_torch(
        points_by_link,
        colors_by_link=colors_by_link,
        camera=camera,
        resolution=resolution,
        point_size=gop.POINT_SIZE,
    )
    if torch_raster is not None:
        color_buffer = np.asarray(torch_raster["image"], dtype=np.uint8)
        owner = np.asarray(torch_raster["owner"], dtype=np.int32)
        link_names = list(torch_raster.get("link_names") or list(points_by_link.keys()))
        label_positions = {}
        visible_px = np.zeros((len(link_names),), dtype=np.int32)
        for i, link_name in enumerate(link_names):
            ys, xs = np.where(owner == int(i))
            visible_px[i] = int(xs.size)
            if xs.size > 0:
                label_positions[link_name] = (float(np.median(xs)), float(np.median(ys)))
        visible_ratio = visible_px.astype(np.float32) / float(max(1, width * height))
        return color_buffer, label_positions, owner, link_names, visible_px, visible_ratio

    color_buffer = np.full((height, width, 3), 255, dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    owner = np.full((height, width), -1, dtype=np.int32)
    label_positions = {}
    link_names = list(points_by_link.keys())
    name_to_idx = {n: i for i, n in enumerate(link_names)}

    for link_name in link_names:
        points = points_by_link.get(link_name)
        if points is None or points.shape[0] == 0:
            continue
        proj = gop.project_points(points, camera, resolution)
        xs = proj[:, 0].round().astype(np.int32)
        ys = proj[:, 1].round().astype(np.int32)
        zs = proj[:, 2]
        mask = (zs > 0) & (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        xs = xs[mask]
        ys = ys[mask]
        zs = zs[mask]
        if xs.size == 0:
            continue

        rgb = (np.clip(colors_by_link[link_name][:3], 0, 1) * 255).astype(np.uint8)
        lid = name_to_idx[link_name]
        for x, y, z in zip(xs, ys, zs):
            if z < z_buffer[y, x]:
                z_buffer[y, x] = z
                owner[y, x] = lid
                color_buffer[y, x] = rgb
                if gop.POINT_SIZE > 1:
                    r = gop.POINT_SIZE // 2
                    x0 = max(0, x - r)
                    y0 = max(0, y - r)
                    x1 = min(width, x0 + gop.POINT_SIZE)
                    y1 = min(height, y0 + gop.POINT_SIZE)
                    color_buffer[y0:y1, x0:x1] = rgb
                    for yy in range(y0, y1):
                        for xx in range(x0, x1):
                            if z < z_buffer[yy, xx]:
                                z_buffer[yy, xx] = z
                                owner[yy, xx] = lid
        label_positions[link_name] = (float(np.median(xs)), float(np.median(ys)))

    visible_px = np.zeros((len(link_names),), dtype=np.int32)
    for i in range(len(link_names)):
        visible_px[i] = int(np.sum(owner == i))
    visible_ratio = visible_px.astype(np.float32) / float(width * height)
    return color_buffer, label_positions, owner, link_names, visible_px, visible_ratio


def _annotate_view_header(img: np.ndarray, view: dict, scale: int = HEADER_SCALE):
    txt = (
        f"{view.get('id','V?')} AZ{int(view['azimuth_deg'])} EL{int(view['elevation_deg'])} "
        f"D{float(view['distance_scale']):.1f} F{int(view['fov_deg'])}"
    )
    gop.draw_text(img, 110, 14, txt, scale=scale, color=(255, 255, 255), bg=(40, 40, 40))


def _draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color=(60, 60, 60), thickness: int = 1):
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    x1 = max(0, min(w - 1, int(x1)))
    y0 = max(0, min(h - 1, int(y0)))
    y1 = max(0, min(h - 1, int(y1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    col = np.array(color, dtype=np.uint8)
    for t in range(max(1, thickness)):
        yt0 = max(0, y0 - t)
        yt1 = min(h - 1, y1 + t)
        xt0 = max(0, x0 - t)
        xt1 = min(w - 1, x1 + t)
        img[yt0, xt0 : xt1 + 1] = col
        img[yt1, xt0 : xt1 + 1] = col
        img[yt0 : yt1 + 1, xt0] = col
        img[yt0 : yt1 + 1, xt1] = col


def _make_grid(images: list[np.ndarray], rows: int, cols: int, bg=(230, 230, 230), pad: int = 4, border: int = 1) -> np.ndarray:
    if not images:
        raise ValueError("No images for grid")
    h, w = images[0].shape[:2]
    canvas_h = rows * h + (rows + 1) * pad
    canvas_w = cols * w + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    for idx, img in enumerate(images[: rows * cols]):
        r = idx // cols
        c = idx % cols
        y0 = pad + r * (h + pad)
        x0 = pad + c * (w + pad)
        y1 = y0 + h
        x1 = x0 + w
        canvas[y0:y1, x0:x1] = img[:, :, :3]
        _draw_rect(canvas, x0, y0, x1 - 1, y1 - 1, color=(70, 70, 70), thickness=border)
    return canvas


def _grid_shape_for_nviews(n_views: int) -> tuple[int, int]:
    n = max(1, int(n_views))
    if n <= 1:
        return 1, 1
    if n <= 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    cols = int(math.ceil(math.sqrt(float(n))))
    rows = int(math.ceil(float(n) / float(cols)))
    return rows, cols


def _annotate_grid_caption(grid: np.ndarray, caption: str | None, scale: int = 2) -> np.ndarray:
    if not caption:
        return grid
    text = str(caption).strip()
    if not text:
        return grid
    scale = max(1, int(scale))
    pil = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil, "RGBA")
    pad = 8
    x0, y0 = 8, 8
    # Draw text on a temporary canvas, then upscale with nearest-neighbor so it stays legible
    # without depending on external font files.
    tmp = Image.new("RGBA", (max(64, pil.width // scale), max(32, pil.height // scale)), (0, 0, 0, 0))
    dtmp = ImageDraw.Draw(tmp, "RGBA")
    tx = max(1, (x0 + pad) // scale)
    ty = max(1, (y0 + pad) // scale)
    bbox = dtmp.multiline_textbbox((tx, ty), text, spacing=2)
    if bbox is None:
        return np.asarray(pil)
    dtmp.multiline_text((tx, ty), text, fill=(20, 20, 20, 255), spacing=2)
    tmp_arr = np.asarray(tmp)
    ys, xs = np.where(tmp_arr[:, :, 3] > 0)
    if ys.size == 0 or xs.size == 0:
        return np.asarray(pil)
    crop = tmp.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    crop = crop.resize((crop.width * scale, crop.height * scale), resample=Image.NEAREST)
    x1 = min(pil.width - 8, x0 + pad + crop.width + pad)
    y1 = min(pil.height - 8, y0 + pad + crop.height + pad)
    draw.rounded_rectangle((x0, y0, x1, y1), radius=6, fill=(255, 255, 255, 235), outline=(70, 70, 70, 255), width=2)
    crop_rgba = crop
    if crop_rgba.mode != "RGBA":
        crop_rgba = crop_rgba.convert("RGBA")
    pil.paste(crop_rgba, (x0 + pad, y0 + pad), crop_rgba)
    return np.asarray(pil)


def _motion_axes_overlay_box(width: int, height: int) -> tuple[int, int, int, int]:
    box_w = min(184, max(138, int(0.23 * float(width))))
    box_h = min(144, max(114, int(0.19 * float(height))))
    x1 = int(width) - 8
    y0 = 10
    x0 = max(8, x1 - box_w)
    y1 = min(int(height) - 8, y0 + box_h)
    return (x0, y0, x1, y1)


def _draw_motion_corner_axes_box(image: np.ndarray, camera, resolution) -> None:
    if image is None or camera is None:
        return
    h, w = image.shape[:2]
    box = _motion_axes_overlay_box(w, h)
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    draw.rounded_rectangle(box, radius=12, fill=(255, 255, 255), outline=(210, 210, 210), width=2)

    def _draw_arrow_segment(p0, p1, color, width_main=5, width_bg=7):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        dx = x1 - x0
        dy = y1 - y0
        n = math.hypot(dx, dy)
        if n <= 1.0e-6:
            return
        ux, uy = dx / n, dy / n
        px, py = -uy, ux
        head_len = max(10.0, 2.8 * float(width_main))
        head_w = max(10.0, 2.3 * float(width_main))
        shaft_end = (x1 - ux * head_len, y1 - uy * head_len)
        left = (x1 - ux * head_len + px * 0.5 * head_w, y1 - uy * head_len + py * 0.5 * head_w)
        right = (x1 - ux * head_len - px * 0.5 * head_w, y1 - uy * head_len - py * 0.5 * head_w)
        for col, width in (((0, 0, 0), width_bg), (tuple(int(v) for v in color), width_main)):
            draw.line((x0, y0, shaft_end[0], shaft_end[1]), fill=col, width=int(width))
            draw.polygon([(x1, y1), left, right], fill=col)

    eye, target, up = camera
    pose = gop.camera_pose_from_lookat(np.asarray(eye, dtype=float), np.asarray(target, dtype=float), np.asarray(up, dtype=float))
    right_vec = np.asarray(pose[:3, 0], dtype=float)
    up_vec = np.asarray(pose[:3, 1], dtype=float)
    forward_vec = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward_vec = forward_vec / max(1.0e-8, float(np.linalg.norm(forward_vec)))

    basis = [
        ("X", np.asarray([1.0, 0.0, 0.0], dtype=float), (220, 40, 40)),
        ("Y", np.asarray([0.0, 1.0, 0.0], dtype=float), (40, 180, 40)),
        ("Z", np.asarray([0.0, 0.0, 1.0], dtype=float), (40, 80, 220)),
    ]
    axis_rows = []
    hidden_rows = []
    for label, axis, color in basis:
        vx = float(np.dot(axis, right_vec))
        vy = -float(np.dot(axis, up_vec))
        vec2 = np.asarray([vx, vy], dtype=float)
        norm2 = float(np.linalg.norm(vec2))
        # Only collapse to dot/cross when the screen projection is truly tiny.
        if norm2 <= 0.08:
            hidden_rows.append((label, float(np.dot(axis, forward_vec)), color))
            continue
        vec2 = vec2 / norm2
        alignment = abs(float(np.dot(axis, forward_vec)))
        facing = float(np.dot(axis, forward_vec))
        axis_rows.append((norm2, alignment, label, vec2, color, facing))
    if not axis_rows:
        image[:] = np.asarray(pil, dtype=np.uint8)
        axis_rows = []
    axis_rows.sort(key=lambda row: (-row[0], row[2]))

    box_w = float(box[2] - box[0])
    box_h = float(box[3] - box[1])
    inset_origin = np.asarray([box[0] + 0.37 * box_w, box[1] + 0.56 * box_h], dtype=float)
    scale = float(min(0.30 * box_w, 0.34 * box_h))
    ox, oy = float(inset_origin[0]), float(inset_origin[1])
    draw.ellipse((ox - 5, oy - 5, ox + 5, oy + 5), fill=(80, 80, 80))
    image[:] = np.asarray(pil, dtype=np.uint8)
    # Draw longer projections first, then shorter ones on top so partially aligned
    # axes remain visible as nested arrows instead of being collapsed into dot/cross.
    for proj_norm, _align, label, vec2, color, _facing in axis_rows:
        arrow_scale = scale * (0.35 + 0.65 * float(np.clip(proj_norm, 0.0, 1.0)))
        tip = inset_origin + vec2 * arrow_scale
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        _draw_arrow_segment(inset_origin, tip, color, width_main=5, width_bg=7)
        image[:] = np.asarray(pil, dtype=np.uint8)
        label_dx = 10 if tip[0] >= inset_origin[0] else -14
        label_dy = -10 if tip[1] <= inset_origin[1] else 6
        gop.draw_text(image, float(tip[0] + label_dx), float(tip[1] + label_dy), label, scale=3, color=color, bg=None)
    if hidden_rows:
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        base_x = float(box[0] + 0.18 * box_w)
        base_y = float(box[1] + 0.78 * box_h)
        for idx, (label, facing, color) in enumerate(hidden_rows[:2]):
            cy = base_y + 22.0 * float(idx)
            gop.draw_text(image, base_x - 16.0, cy, label, scale=3, color=color, bg=None)
            pil = Image.fromarray(image)
            draw = ImageDraw.Draw(pil)
            cx = base_x + 16.0
            r = 13.5
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255), outline=tuple(int(v) for v in color), width=3)
            if facing < 0.0:
                draw.ellipse((cx - 2.6, cy - 2.6, cx + 2.6, cy + 2.6), fill=(0, 0, 0))
                draw.ellipse((cx - 1.8, cy - 1.8, cx + 1.8, cy + 1.8), fill=tuple(int(v) for v in color))
            else:
                arm = 2.2
                draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill=(0, 0, 0), width=3)
                draw.line((cx - arm, cy + arm, cx + arm, cy - arm), fill=(0, 0, 0), width=3)
                draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill=tuple(int(v) for v in color), width=2)
                draw.line((cx - arm, cy + arm, cx + arm, cy - arm), fill=tuple(int(v) for v in color), width=2)
            image[:] = np.asarray(pil, dtype=np.uint8)


def _load_trajectory_data(trajectory_npz: Path) -> dict:
    path = Path(trajectory_npz)
    try:
        st = path.stat()
        key = (str(path.resolve()), int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        key = (str(path), -1, -1)
    cached = _TRAJECTORY_CACHE.get(key)
    if isinstance(cached, dict):
        return cached
    data = np.load(path, allow_pickle=True)
    out = {
        "joint_names": [str(x) for x in data["joint_names"].tolist()],
        "joint_angles": np.asarray(data["joint_angles"], dtype=float),
        "base_translation": np.asarray(data["base_translation"], dtype=float),
        "time_s": np.asarray(data["time_s"], dtype=float) if "time_s" in data else None,
        "_cache_key": key,
    }
    _TRAJECTORY_CACHE[key] = out
    return out


def _frame_state_from_traj(traj_data: dict, frame_idx: int):
    joint_names = traj_data["joint_names"]
    joint_angles = traj_data["joint_angles"]
    base_translation = traj_data["base_translation"]
    fi = int(np.clip(frame_idx, 0, joint_angles.shape[0] - 1))
    joint_pos = {jn: float(joint_angles[fi, j]) for j, jn in enumerate(joint_names)}
    base_tf = np.eye(4)
    base_tf[:3, 3] = np.asarray(base_translation[fi], dtype=float)
    return fi, joint_pos, base_tf


def _apply_tf_point(tf: np.ndarray, p3: np.ndarray) -> np.ndarray:
    ph = np.ones((4,), dtype=float)
    ph[:3] = np.asarray(p3, dtype=float)
    out = np.asarray(tf, dtype=float) @ ph
    return np.asarray(out[:3], dtype=float)


def _apply_tf_points(tf: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts3, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    tf_arr = np.asarray(tf, dtype=float)
    rot = tf_arr[:3, :3]
    trans = tf_arr[:3, 3]
    return pts @ rot.T + trans.reshape(1, 3)


def _apply_tf_direction(tf: np.ndarray, v3: np.ndarray) -> np.ndarray:
    vec = np.asarray(v3, dtype=float).reshape(-1)
    if vec.size < 3:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    rot = np.asarray(tf, dtype=float)[:3, :3]
    out = rot @ vec[:3]
    n = float(np.linalg.norm(out))
    if n <= 1e-8:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return out / n


def _get_motion_bbox_local_points(asset_ctx: dict, link_names: list[str]) -> dict[str, np.ndarray]:
    cache = asset_ctx.get("_motion_bbox_local_points")
    if isinstance(cache, dict):
        return {str(k): np.asarray(v, dtype=np.float32) for k, v in cache.items()}
    out: dict[str, np.ndarray] = {}
    link_meshes = asset_ctx.get("link_meshes") or {}
    for ln in link_names:
        meshes = link_meshes.get(ln, []) if isinstance(link_meshes, dict) else []
        valid = [m.copy() for m in meshes if m is not None and getattr(m, "vertices", None) is not None and m.vertices.size > 0]
        if not valid:
            out[str(ln)] = np.zeros((0, 3), dtype=np.float32)
            continue
        merged = trimesh.util.concatenate(valid) if len(valid) > 1 else valid[0]
        pts = gop._mesh_points_for_bbox(merged, f"motionlocal:{ln}", max_points=MOTION_BBOX_MAX_POINTS)
        out[str(ln)] = np.asarray(pts, dtype=np.float32)
    asset_ctx["_motion_bbox_local_points"] = out
    return out


def _get_motion_sample_local_points(asset_ctx: dict, link_names: list[str]) -> dict[str, np.ndarray]:
    cache = asset_ctx.get("_motion_sample_local_points")
    if isinstance(cache, dict):
        return {str(k): np.asarray(v, dtype=np.float32) for k, v in cache.items()}
    out: dict[str, np.ndarray] = {}
    link_meshes = asset_ctx.get("link_meshes") or {}
    for ln in link_names:
        meshes = link_meshes.get(ln, []) if isinstance(link_meshes, dict) else []
        valid = [m.copy() for m in meshes if m is not None and getattr(m, "vertices", None) is not None and m.vertices.size > 0]
        if not valid:
            out[str(ln)] = np.zeros((0, 3), dtype=np.float32)
            continue
        merged = trimesh.util.concatenate(valid) if len(valid) > 1 else valid[0]
        if merged.vertices.size == 0:
            out[str(ln)] = np.zeros((0, 3), dtype=np.float32)
            continue
        seed = int(hashlib.md5(str(ln).encode("utf-8")).hexdigest()[:8], 16) & 0xFFFFFFFF
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            try:
                pts = merged.sample(gop.POINTS_PER_LINK)
            except Exception:
                pts, _ = trimesh.sample.sample_surface(merged, gop.POINTS_PER_LINK)
        finally:
            np.random.set_state(state)
        out[str(ln)] = np.asarray(pts, dtype=np.float32)
    asset_ctx["_motion_sample_local_points"] = out
    return out


def _transform_local_points_by_link(
    local_points_by_link: dict[str, np.ndarray],
    link_tf: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for ln, pts in (local_points_by_link or {}).items():
        tf = np.asarray(link_tf.get(str(ln), np.eye(4)), dtype=float)
        out[str(ln)] = np.asarray(_apply_tf_points(tf, np.asarray(pts, dtype=np.float32)), dtype=np.float32)
    return out


def _compute_motion_union_boxes(
    asset_ctx: dict,
    traj_data: dict,
    link_names: list[str],
    camera,
    resolution,
    motion_window: tuple[int, int] | None,
) -> dict[str, tuple[int, int, int, int]]:
    if motion_window is None:
        return {}
    joint_angles = np.asarray(traj_data.get("joint_angles", np.zeros((0, 0))), dtype=float)
    n_frames = int(joint_angles.shape[0]) if joint_angles.ndim >= 1 else 0
    if n_frames <= 0:
        return {}
    try:
        i0 = max(0, min(n_frames - 1, int(motion_window[0])))
        i1 = max(0, min(n_frames - 1, int(motion_window[1])))
    except Exception:
        return {}
    if i1 < i0:
        i0, i1 = i1, i0
    if i1 == i0:
        frame_ids = [i0]
    else:
        span = i1 - i0 + 1
        if span <= 6:
            frame_ids = list(range(i0, i1 + 1))
        else:
            frame_ids = sorted({int(round(x)) for x in np.linspace(i0, i1, num=6)})
    local_points_by_link = _get_motion_bbox_local_points(asset_ctx, link_names)
    union_boxes: dict[str, tuple[int, int, int, int]] = {}
    for fi in frame_ids:
        _, joint_pos_f, base_tf_f = _frame_state_from_traj(traj_data, fi)
        link_tf_f = rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], joint_pos_f, base_tf=base_tf_f)
        for ln in link_names:
            pts_local = np.asarray(local_points_by_link.get(ln, np.zeros((0, 3), dtype=np.float32)), dtype=float)
            if pts_local.size == 0 or ln not in link_tf_f:
                continue
            pts_world = _apply_tf_points(link_tf_f[ln], pts_local)
            proj = gop.project_points(pts_world, camera, resolution)
            if proj.shape[0] == 0:
                continue
            mask = proj[:, 2] > 0
            if not np.any(mask):
                continue
            xs = proj[mask, 0]
            ys = proj[mask, 1]
            if xs.size == 0 or ys.size == 0:
                continue
            x0 = int(np.floor(xs.min()))
            y0 = int(np.floor(ys.min()))
            x1 = int(np.ceil(xs.max()))
            y1 = int(np.ceil(ys.max()))
            box = gop.expand_bbox((x0, y0, x1, y1), resolution)
            prev = union_boxes.get(ln)
            if prev is None:
                union_boxes[ln] = box
            else:
                union_boxes[ln] = (
                    min(int(prev[0]), int(box[0])),
                    min(int(prev[1]), int(box[1])),
                    max(int(prev[2]), int(box[2])),
                    max(int(prev[3]), int(box[3])),
                )
    return union_boxes


def _build_link_local_keypoints(link_meshes: dict[str, list]) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for ln, meshes in (link_meshes or {}).items():
        verts = []
        for m in meshes or []:
            v = getattr(m, "vertices", None)
            if v is None:
                continue
            arr = np.asarray(v, dtype=float)
            if arr.size == 0:
                continue
            verts.append(arr)
        if not verts:
            continue
        pts = np.concatenate(verts, axis=0) if len(verts) > 1 else verts[0]
        center = pts.mean(axis=0)
        bmin = pts.min(axis=0)
        bmax = pts.max(axis=0)
        ext = np.maximum(bmax - bmin, 0.0)
        idx = int(np.argmax(ext))
        ref = center.copy()
        if float(ext[idx]) > 1e-9:
            ref[idx] += 0.5 * float(ext[idx])
        out[ln] = {"center_local": center, "ref_local": ref}
    return out


def _build_link_trace_points(
    asset_ctx: dict,
    link_names: list[str],
    samples_per_link: int = MOTION_TRACE_SAMPLES_PER_LINK,
    trace_variant_index: int = 0,
) -> dict[str, dict[str, np.ndarray | str]]:
    cache_root = asset_ctx.get("_motion_trace_points_cache")
    cache_key = (int(samples_per_link), int(trace_variant_index))
    if isinstance(cache_root, dict):
        cache = cache_root.get(cache_key)
        if isinstance(cache, dict) and all(str(ln) in cache for ln in link_names):
            out_cached = {}
            for key, value in cache.items():
                if isinstance(value, dict):
                    out_cached[str(key)] = {
                        "points": np.asarray(value.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32),
                        "space": str(value.get("space") or "local"),
                        "parent": str(value.get("parent") or ""),
                        "axis_local": np.asarray(value.get("axis_local", np.zeros((0,), dtype=np.float32)), dtype=np.float32),
                    }
            return out_cached

    out: dict[str, dict[str, np.ndarray | str]] = {}
    candidates_local = _get_motion_bbox_local_points(asset_ctx, link_names)
    glb_link_rest_meshes = asset_ctx.get("glb_link_rest_meshes") if isinstance(asset_ctx.get("glb_link_rest_meshes"), dict) else None
    child_axis = {}
    continuous_children = set()
    child_parent = {}
    for joint in asset_ctx.get("joints") or []:
        jt = str(joint.get("type") or "").strip().lower()
        child = str(joint.get("child") or "").strip()
        parent = str(joint.get("parent") or "").strip()
        axis = np.asarray(joint.get("axis") or [0.0, 0.0, 1.0], dtype=float).reshape(-1)
        if child and parent:
            child_parent[child] = parent
        if jt in {"continuous", "revolute"} and child and axis.size == 3:
            norm = float(np.linalg.norm(axis))
            if norm > 1e-8:
                child_axis[child] = axis / norm
        if jt == "continuous" and child:
            continuous_children.add(child)

    def _select_rim_points(points: np.ndarray, axis: np.ndarray, count: int, variant_index: int) -> np.ndarray:
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        center = points.mean(axis=0)
        rel = points - center.reshape(1, 3)
        axial = rel @ axis.reshape(3, 1)
        axial = axial.reshape(-1)
        radial = rel - axial.reshape(-1, 1) * axis.reshape(1, 3)
        radial_norm = np.linalg.norm(radial, axis=1)
        if float(radial_norm.max()) <= 1e-8:
            return np.zeros((0, 3), dtype=np.float32)
        radius = float(np.quantile(radial_norm, 0.95))
        radius = max(radius, 1e-4)
        axial_abs = np.abs(axial - float(np.median(axial)))
        axial_keep = np.quantile(axial_abs, 0.35) if axial_abs.size > 4 else float(axial_abs.max())
        keep_mask = axial_abs <= max(axial_keep, 1e-4)
        candidate_idx = np.where(keep_mask)[0]
        if candidate_idx.size < max(2, count):
            candidate_idx = np.argsort(-radial_norm)[: max(8, count * 4)]
        candidate_pts = points[candidate_idx]
        candidate_axial = axial[candidate_idx]
        candidate_radial = radial[candidate_idx]
        candidate_radial_norm = radial_norm[candidate_idx]
        candidate_center = center + axis * float(np.median(candidate_axial))
        ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(ref, axis))) > 0.85:
            ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
        basis_u = np.cross(axis, ref)
        basis_u = basis_u / max(1e-8, float(np.linalg.norm(basis_u)))
        basis_v = np.cross(axis, basis_u)
        basis_v = basis_v / max(1e-8, float(np.linalg.norm(basis_v)))
        if count <= 1:
            angles = np.linspace(0.0, 2.0 * math.pi, num=max(8, int(MOTION_TRACE_CANDIDATE_COUNT_SINGLE)), endpoint=False).tolist()
        elif count == 2:
            angles = [0.0, math.pi]
        else:
            angles = np.linspace(0.0, 2.0 * math.pi, num=int(count), endpoint=False).tolist()
        if angles:
            phase_offset = (2.0 * math.pi * float(int(variant_index) % len(angles))) / float(len(angles))
            angles = [float(ang) + phase_offset for ang in angles]
        cand_dirs = candidate_radial / np.maximum(candidate_radial_norm.reshape(-1, 1), 1e-8)
        used: set[int] = set()
        out_pts = []
        for ang in angles[:count]:
            radial_dir = math.cos(float(ang)) * basis_u + math.sin(float(ang)) * basis_v
            target = candidate_center + radius * radial_dir
            score = cand_dirs @ radial_dir.reshape(3, 1)
            score = score.reshape(-1)
            score -= 0.12 * np.abs(candidate_radial_norm - radius) / max(radius, 1e-6)
            score -= 0.04 * np.abs(candidate_axial - float(np.median(candidate_axial))) / max(axial_keep, 1e-4)
            order = np.argsort(-score)
            chosen = None
            for idx_local in order.tolist():
                if idx_local not in used:
                    chosen = idx_local
                    break
            if chosen is None:
                chosen = int(order[0]) if order.size > 0 else None
            if chosen is None:
                out_pts.append(target)
            else:
                used.add(int(chosen))
                out_pts.append(candidate_pts[int(chosen)])
        return np.asarray(out_pts, dtype=np.float32)

    for ln in link_names:
        pts = None
        space = "local"
        parent = child_parent.get(str(ln), "")
        axis = child_axis.get(str(ln))
        axis_for_pts = np.asarray(axis, dtype=float) if axis is not None else None
        pts_local = np.asarray(candidates_local.get(ln, np.zeros((0, 3), dtype=np.float32)), dtype=float)
        if pts_local.ndim == 2 and pts_local.shape[0] > 0:
            pts = pts_local
            space = "local"
        elif isinstance(glb_link_rest_meshes, dict) and ln in glb_link_rest_meshes:
            meshes = glb_link_rest_meshes.get(ln) or []
            verts = []
            for mesh in meshes:
                v = getattr(mesh, "vertices", None)
                if v is None:
                    continue
                arr = np.asarray(v, dtype=float)
                if arr.size == 0:
                    continue
                verts.append(arr)
            if verts:
                pts = np.concatenate(verts, axis=0) if len(verts) > 1 else verts[0]
                space = "rest_world"
                if axis is not None and ln in asset_ctx.get("rest_link_tf", {}):
                    axis_for_pts = _apply_tf_direction(asset_ctx["rest_link_tf"][ln], axis)
        if pts is None:
            pts = pts_local
        if pts.ndim != 2 or pts.shape[0] == 0:
            out[str(ln)] = {"points": np.zeros((0, 3), dtype=np.float32), "space": space, "parent": parent}
            continue
        if axis_for_pts is not None:
            if int(samples_per_link) <= 1:
                desired = max(8, int(MOTION_TRACE_CANDIDATE_COUNT_SINGLE))
            elif str(ln) in continuous_children:
                desired = max(3, min(int(samples_per_link), 6))
            else:
                desired = max(2, min(int(samples_per_link), 6))
            rim_pts = _select_rim_points(
                np.asarray(pts, dtype=float),
                np.asarray(axis_for_pts, dtype=float),
                desired,
                int(trace_variant_index),
            )
            if rim_pts.shape[0] > 0:
                out[str(ln)] = {
                    "points": rim_pts,
                    "space": space,
                    "parent": parent,
                    "axis_local": np.asarray(axis, dtype=np.float32) if axis is not None else np.zeros((0,), dtype=np.float32),
                }
                continue
        out[str(ln)] = {
            "points": np.asarray(pts[: max(1, min(len(pts), int(samples_per_link)))], dtype=np.float32),
            "space": space,
            "parent": parent,
            "axis_local": np.asarray(axis, dtype=np.float32) if axis is not None else np.zeros((0,), dtype=np.float32),
        }

    if not isinstance(cache_root, dict):
        cache_root = {}
        asset_ctx["_motion_trace_points_cache"] = cache_root
    cache_root[cache_key] = out
    return out


def _rasterize_current_link_silhouette_features(
    asset_ctx: dict,
    link_tf_cur: dict[str, np.ndarray],
    link_names: list[str],
    camera,
    resolution,
) -> dict[str, dict[str, np.ndarray]]:
    cache_token = asset_ctx.get("_current_motion_cache_token")
    current_view = asset_ctx.get("_current_motion_primary_view")
    cache_key = None
    if cache_token is not None and isinstance(current_view, dict):
        cache_key = (
            cache_token,
            int(current_view.get("azimuth_deg", 0)),
            int(current_view.get("elevation_deg", 0)),
            round(float(current_view.get("distance_scale", 1.0)), 6),
            int(current_view.get("fov_deg", 35)),
            int(resolution[0]),
            int(resolution[1]),
        )
        cache_root = asset_ctx.get("_motion_silhouette_features_cache")
        if isinstance(cache_root, dict) and cache_key in cache_root:
            cached = cache_root.get(cache_key)
            if isinstance(cached, dict):
                return {
                    str(ln): {
                        "center_xy": np.asarray(v.get("center_xy", np.zeros((2,), dtype=np.float32)), dtype=np.float32),
                        "max_radius_by_bin": np.asarray(v.get("max_radius_by_bin", np.zeros((0,), dtype=np.float32)), dtype=np.float32),
                    }
                    for ln, v in cached.items()
                    if str(ln) in {str(x) for x in link_names}
                }
    mesh_src = asset_ctx.get("link_meshes_textured")
    if not isinstance(mesh_src, dict):
        mesh_src = asset_ctx.get("link_meshes")
    if not isinstance(mesh_src, dict):
        return {}
    try:
        world_link_meshes = transform_link_meshes(mesh_src, link_tf_cur)
    except Exception:
        return {}
    all_link_names = [str(ln) for ln, meshes in world_link_meshes.items() if meshes]
    if not all_link_names:
        return {}
    try:
        owner_link, _owner_sub = gop.rasterize_link_visibility_maps(
            world_link_meshes,
            all_link_names,
            camera,
            resolution,
            max_faces=gop.REFERENCE_MAX_FACES,
            return_scene_depth=False,
        )
    except Exception:
        return {}
    if owner_link.size == 0:
        return {}
    link_to_idx = {str(ln): i for i, ln in enumerate(all_link_names)}
    n_bins = 128
    features: dict[str, dict[str, np.ndarray]] = {}
    for ln in link_names:
        li = link_to_idx.get(str(ln))
        if li is None:
            continue
        mask = owner_link == int(li)
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        center_xy = np.asarray([float(np.median(xs)), float(np.median(ys))], dtype=np.float32)
        dx = xs.astype(np.float32) - float(center_xy[0])
        dy = ys.astype(np.float32) - float(center_xy[1])
        radius = np.hypot(dx, dy).astype(np.float32)
        if radius.size == 0 or float(radius.max()) <= 1.0e-6:
            continue
        angles = np.arctan2(dy, dx)
        bins = np.floor(((angles + math.pi) / (2.0 * math.pi)) * float(n_bins)).astype(np.int32) % int(n_bins)
        max_radius_by_bin = np.zeros((n_bins,), dtype=np.float32)
        for bi, rr in zip(bins.tolist(), radius.tolist()):
            if float(rr) > float(max_radius_by_bin[int(bi)]):
                max_radius_by_bin[int(bi)] = float(rr)
        nonzero = np.where(max_radius_by_bin > 0.0)[0]
        if nonzero.size > 0 and nonzero.size < n_bins:
            for bi in range(n_bins):
                if max_radius_by_bin[bi] > 0.0:
                    continue
                dist = np.abs(nonzero - int(bi))
                dist = np.minimum(dist, n_bins - dist)
                nearest = int(nonzero[int(np.argmin(dist))])
                max_radius_by_bin[bi] = max_radius_by_bin[nearest]
        features[str(ln)] = {
            "center_xy": center_xy.astype(np.float32),
            "max_radius_by_bin": max_radius_by_bin.astype(np.float32),
        }
    if cache_key is not None:
        cache_root = asset_ctx.get("_motion_silhouette_features_cache")
        if not isinstance(cache_root, dict):
            cache_root = {}
            asset_ctx["_motion_silhouette_features_cache"] = cache_root
        _put_limited_cache(cache_root, cache_key, features, max_entries=12)
    return {str(ln): features[str(ln)] for ln in link_names if str(ln) in features}


def _silhouette_outer_score(point_xy: np.ndarray, silhouette_feature: dict | None) -> float:
    if not isinstance(silhouette_feature, dict):
        return 0.0
    center_xy = np.asarray(silhouette_feature.get("center_xy", np.zeros((2,), dtype=np.float32)), dtype=float).reshape(-1)
    max_radius_by_bin = np.asarray(
        silhouette_feature.get("max_radius_by_bin", np.zeros((0,), dtype=np.float32)),
        dtype=float,
    ).reshape(-1)
    pt = np.asarray(point_xy, dtype=float).reshape(-1)
    if center_xy.size < 2 or pt.size < 2 or max_radius_by_bin.size == 0:
        return 0.0
    delta = pt[:2] - center_xy[:2]
    radius = float(np.linalg.norm(delta))
    if radius <= 1.0e-6:
        return 0.0
    angle = float(math.atan2(float(delta[1]), float(delta[0])))
    n_bins = int(max_radius_by_bin.size)
    idx = int(math.floor(((angle + math.pi) / (2.0 * math.pi)) * float(n_bins))) % max(1, n_bins)
    nb = np.asarray([(idx - 1) % n_bins, idx, (idx + 1) % n_bins], dtype=np.int32)
    max_radius = float(np.max(max_radius_by_bin[nb])) if nb.size > 0 else 0.0
    if max_radius <= 1.0e-6:
        return 0.0
    return float(np.clip(radius / max_radius, 0.0, 1.25))


def _compute_link_motion_vectors(
    asset_ctx: dict,
    traj_data: dict,
    frame_idx: int,
    link_tf_cur: dict[str, np.ndarray],
    link_names: list[str],
    motion_window: tuple[int, int] | None = None,
    trace_variant_index: int = 0,
    use_best_trace_candidate: bool = False,
    use_edge_variant_candidate: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    joint_angles = np.asarray(traj_data.get("joint_angles", np.zeros((0, 0))), dtype=float)
    n_frames = int(joint_angles.shape[0]) if joint_angles.ndim >= 1 else 0
    if n_frames <= 0:
        return {}
    fi = int(np.clip(frame_idx, 0, n_frames - 1))
    i0 = max(0, fi - 1)
    i1 = min(n_frames - 1, fi + 1)
    if motion_window is not None:
        try:
            mw0 = int(motion_window[0])
            mw1 = int(motion_window[1])
            i0 = max(0, min(n_frames - 1, mw0))
            i1 = max(0, min(n_frames - 1, mw1))
        except Exception:
            pass
    if i1 <= i0:
        if i1 < n_frames - 1:
            i1 = i0 + 1
        elif i0 > 0:
            i0 = i1 - 1

    if i1 < i0:
        i0, i1 = i1, i0
    step = max(1, int(MOTION_TRACE_SAMPLE_EVERY_FRAMES))
    frame_ids = list(range(i0, i1 + 1, step))
    if not frame_ids:
        frame_ids = [i0]
    if frame_ids[-1] != i1:
        frame_ids.append(i1)

    frame_tf_cache_key = (traj_data.get("_cache_key"), tuple(int(x) for x in frame_ids))
    frame_tf_cache_root = asset_ctx.get("_motion_frame_tf_cache")
    frame_link_tf_by_fi = frame_tf_cache_root.get(frame_tf_cache_key) if isinstance(frame_tf_cache_root, dict) else None
    if not isinstance(frame_link_tf_by_fi, dict):
        frame_link_tf_by_fi = {}
        for fi_sample in frame_ids:
            _, joint_pos_f, base_tf_f = _frame_state_from_traj(traj_data, fi_sample)
            frame_link_tf_by_fi[int(fi_sample)] = rp.compute_link_transforms(
                asset_ctx["links"],
                asset_ctx["joints"],
                joint_pos_f,
                base_tf=base_tf_f,
            )
        if not isinstance(frame_tf_cache_root, dict):
            frame_tf_cache_root = {}
            asset_ctx["_motion_frame_tf_cache"] = frame_tf_cache_root
        _put_limited_cache(frame_tf_cache_root, frame_tf_cache_key, frame_link_tf_by_fi, max_entries=10)

    _, joint_pos_0, base_tf_0 = _frame_state_from_traj(traj_data, i0)
    _, joint_pos_1, base_tf_1 = _frame_state_from_traj(traj_data, i1)
    link_tf_0 = rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], joint_pos_0, base_tf=base_tf_0)
    link_tf_1 = rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], joint_pos_1, base_tf=base_tf_1)

    keypoints = _build_link_local_keypoints(asset_ctx.get("link_meshes") or {})
    trace_points = _build_link_trace_points(asset_ctx, link_names, trace_variant_index=trace_variant_index)
    rest_link_tf = asset_ctx.get("rest_link_tf") if isinstance(asset_ctx.get("rest_link_tf"), dict) else {}
    camera_anchor_center = asset_ctx.get("_current_motion_camera_center")
    camera_anchor_radius = asset_ctx.get("_current_motion_camera_radius")
    current_view = asset_ctx.get("_current_motion_primary_view")
    current_resolution = asset_ctx.get("_current_motion_primary_resolution")
    cam_for_pick = asset_ctx.get("_current_motion_primary_camera")
    if not (isinstance(cam_for_pick, tuple) and len(cam_for_pick) == 3):
        if (
            isinstance(camera_anchor_center, np.ndarray)
            and isinstance(camera_anchor_radius, (int, float))
            and isinstance(current_view, dict)
            and isinstance(current_resolution, tuple)
            and len(current_resolution) == 2
        ):
            cam_for_pick = compute_camera_for_viewspec(
                np.asarray(camera_anchor_center, dtype=float),
                float(camera_anchor_radius),
                current_view,
            )
        else:
            cam_for_pick = None
    silhouette_features = {}
    if isinstance(current_resolution, tuple) and len(current_resolution) == 2 and cam_for_pick is not None:
        try:
            silhouette_features = _rasterize_current_link_silhouette_features(
                asset_ctx,
                link_tf_cur,
                link_names,
                cam_for_pick,
                current_resolution,
            )
        except Exception:
            silhouette_features = {}
    out = {}
    for ln in link_names:
        kp = keypoints.get(ln)
        tf1 = link_tf_1.get(ln)
        tf0 = link_tf_0.get(ln)
        if kp is None or tf1 is None or tf0 is None:
            continue
        c0 = _apply_tf_point(tf0, kp["center_local"])
        c1 = _apply_tf_point(tf1, kp["center_local"])
        r0 = _apply_tf_point(tf0, kp["ref_local"])
        r1 = _apply_tf_point(tf1, kp["ref_local"])
        vec = np.asarray(r1 - r0, dtype=float)
        if float(np.linalg.norm(vec)) < 1e-8:
            vec = np.asarray(c1 - c0, dtype=float)
        center_track = []
        for fi_sample in frame_ids:
            _, joint_pos_f, base_tf_f = _frame_state_from_traj(traj_data, fi_sample)
            link_tf_f = rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], joint_pos_f, base_tf=base_tf_f)
            tf_f = link_tf_f.get(ln)
            if tf_f is None:
                continue
            center_track.append(_apply_tf_point(tf_f, kp["center_local"]))
        tracks = []
        tp = trace_points.get(ln) or {}
        pts_local_raw = np.asarray(tp.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=float)
        link_rest_tf = np.asarray(rest_link_tf.get(ln, np.eye(4)), dtype=float)
        try:
            link_rest_tf_inv = np.linalg.inv(link_rest_tf)
        except Exception:
            link_rest_tf_inv = np.eye(4)
        if str(tp.get("space") or "local") == "rest_world":
            pts_local = []
            for point_world in pts_local_raw:
                try:
                    pts_local.append(_apply_tf_point(link_rest_tf_inv, point_world))
                except Exception:
                    pts_local.append(np.asarray(point_world, dtype=float))
            pts_local = np.asarray(pts_local, dtype=float)
        else:
            pts_local = np.asarray(pts_local_raw, dtype=float)
        if pts_local.ndim == 2 and pts_local.shape[0] > 1:
            center_track = np.asarray(center_track, dtype=float)
            if center_track.ndim == 2 and center_track.shape[0] >= 2:
                try:
                    if (
                        cam_for_pick is not None
                        and isinstance(current_resolution, tuple)
                        and len(current_resolution) == 2
                        and int(MOTION_TRACE_SAMPLES_PER_LINK) <= 1
                    ):
                        silhouette_feature = silhouette_features.get(ln)
                        candidate_scores = []
                        for point_local in pts_local:
                            track_xy = []
                            for fi_sample in frame_ids:
                                link_tf_f = frame_link_tf_by_fi.get(int(fi_sample), {})
                                tf_f = link_tf_f.get(ln)
                                if tf_f is None:
                                    continue
                                pt_world_f = _apply_tf_point(tf_f, point_local)
                                proj_f = gop.project_points(
                                    np.asarray([pt_world_f], dtype=float),
                                    cam_for_pick,
                                    current_resolution,
                                )
                                if proj_f.shape[0] != 1 or proj_f[0, 2] <= 0:
                                    continue
                                track_xy.append(np.asarray(proj_f[0, :2], dtype=float))
                            if len(track_xy) < 2:
                                continue
                            track_xy_arr = np.asarray(track_xy, dtype=float)
                            path_len = float(np.linalg.norm(np.diff(track_xy_arr, axis=0), axis=1).sum())
                            chord = np.asarray(track_xy_arr[-1] - track_xy_arr[0], dtype=float)
                            chord_norm = float(np.linalg.norm(chord))
                            if chord_norm <= 1.0e-6:
                                arc_score = 0.0
                            else:
                                rel = track_xy_arr - track_xy_arr[0].reshape(1, 2)
                                arc_score = float(np.max(np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0])) / chord_norm)
                            current_proj_xy = None
                            current_depth = float("inf")
                            silhouette_score = 0.0
                            tf_cur = link_tf_cur.get(ln)
                            if tf_cur is not None and cam_for_pick is not None:
                                pt_world_cur = _apply_tf_point(tf_cur, point_local)
                                proj_cur = gop.project_points(
                                    np.asarray([pt_world_cur], dtype=float),
                                    cam_for_pick,
                                    current_resolution,
                                )
                                if proj_cur.shape[0] == 1 and proj_cur[0, 2] > 0:
                                    current_proj_xy = np.asarray(proj_cur[0, :2], dtype=float)
                                    current_depth = float(proj_cur[0, 2])
                                    silhouette_score = _silhouette_outer_score(current_proj_xy, silhouette_feature)
                            candidate_scores.append(
                                {
                                    "path_len": path_len,
                                    "curve_score": path_len / max(1.0e-6, float(np.linalg.norm(track_xy_arr[-1] - track_xy_arr[0]))),
                                    "arc_score": arc_score,
                                    "silhouette_score": silhouette_score,
                                    "current_depth": current_depth,
                                    "current_proj_xy": current_proj_xy,
                                    "point_local": np.asarray(point_local, dtype=float),
                                }
                            )
                        if candidate_scores:
                            if bool(use_edge_variant_candidate):
                                candidate_scores.sort(
                                    key=lambda x: (float(x["arc_score"]), float(x["curve_score"]), float(x["path_len"])),
                                    reverse=True,
                                )
                                best_arc = max(1.0e-6, float(candidate_scores[0]["arc_score"]))
                                best_curve = max(1.0e-6, float(candidate_scores[0]["curve_score"]))
                                best_path = max(1.0e-6, float(candidate_scores[0]["path_len"]))
                                candidate_pool = [
                                    item
                                    for item in candidate_scores
                                    if (
                                        float(item["arc_score"]) >= 0.55 * best_arc
                                        or float(item["curve_score"]) >= 0.85 * best_curve
                                        or float(item["path_len"]) >= 0.90 * best_path
                                    )
                                ]
                                keep_n = max(2, min(len(candidate_scores), 6))
                                if len(candidate_pool) < keep_n:
                                    candidate_pool = list(candidate_scores[:keep_n])
                                candidate_pool.sort(
                                    key=lambda x: (
                                        float(x.get("silhouette_score", 0.0)),
                                        -float(x.get("current_depth", float("inf"))),
                                        float(x["arc_score"]),
                                        float(x["curve_score"]),
                                        float(x["path_len"]),
                                    ),
                                    reverse=True,
                                )
                                top_k = max(1, min(len(candidate_pool), 4))
                                pts_local = np.asarray([candidate_pool[int(trace_variant_index) % top_k]["point_local"]], dtype=float)
                            elif bool(use_best_trace_candidate):
                                candidate_scores.sort(key=lambda x: float(x["path_len"]), reverse=True)
                                pts_local = np.asarray([candidate_scores[0]["point_local"]], dtype=float)
                            else:
                                candidate_scores.sort(key=lambda x: float(x["path_len"]), reverse=True)
                                best_path_len = max(1.0e-6, float(candidate_scores[0]["path_len"]))
                                strong_candidates = [
                                    item
                                    for item in candidate_scores
                                    if float(item["path_len"]) >= 0.55 * best_path_len
                                ]
                                keep_n = max(2, min(len(candidate_scores), 4))
                                candidate_pool = list(strong_candidates[:keep_n])
                                if len(candidate_pool) < keep_n:
                                    candidate_pool = list(candidate_scores[:keep_n])
                                if not candidate_pool:
                                    candidate_pool = list(candidate_scores[:1])
                                # Rotate across strong visible edge candidates across motion-loop iterations
                                # so different iterations probe different boundary features instead of the same rim point.
                                link_offset = sum(ord(ch) for ch in str(ln)) % max(1, len(candidate_pool))
                                pick_idx = (int(trace_variant_index) + int(link_offset)) % max(1, len(candidate_pool))
                                pts_local = np.asarray([candidate_pool[pick_idx]["point_local"]], dtype=float)
                except Exception:
                    pass
        if pts_local.ndim == 2 and pts_local.shape[0] > 0:
            for point_local in pts_local:
                pts_world = []
                for fi_sample in frame_ids:
                    link_tf_f = frame_link_tf_by_fi.get(int(fi_sample), {})
                    tf_f = link_tf_f.get(ln)
                    if tf_f is None:
                        continue
                    pts_world.append(_apply_tf_point(tf_f, point_local))
                if len(pts_world) >= 2:
                    tracks.append(np.asarray(pts_world, dtype=float))
        if len(tracks) < 1:
            ref_track = []
            for fi_sample in frame_ids:
                link_tf_f = frame_link_tf_by_fi.get(int(fi_sample), {})
                tf_f = link_tf_f.get(ln)
                if tf_f is None:
                    continue
                ref_track.append(_apply_tf_point(tf_f, kp["ref_local"]))
            if len(ref_track) >= 2:
                tracks.append(np.asarray(ref_track, dtype=float))
        if len(tracks) < 1 and len(center_track) >= 2:
            tracks.append(np.asarray(center_track, dtype=float))
        out[ln] = {
            "center_prev_world": c0,
            "center_curr_world": c1,
            "ref_prev_world": r0,
            "ref_curr_world": r1,
            "vector_world": vec,
            "center_track_world": np.asarray(center_track, dtype=float) if len(center_track) >= 2 else np.zeros((0, 3), dtype=float),
            "tracks_world": tracks,
        }
    return out


def _rotary_child_links_from_joints(joints: list[dict]) -> set[str]:
    out: set[str] = set()
    for j in joints or []:
        jt = str(j.get("type") or "").strip().lower()
        if jt not in {"revolute", "continuous"}:
            continue
        child = str(j.get("child") or "").strip()
        if child:
            out.add(child)
    return out


def _digits_only_label(link_name: str, raw_label: str | None, fallback_index: int) -> str:
    import re

    cand = str(raw_label or "").strip()
    nums = re.findall(r"\d+", cand)
    if nums:
        return str(nums[-1])
    nums_ln = re.findall(r"\d+", str(link_name or ""))
    if nums_ln:
        return str(nums_ln[-1])
    return str(int(fallback_index) + 1)


def _sanitize_motion_label_texts(
    visual_links: list[str],
    label_texts_raw: dict[str, str],
) -> dict[str, str]:
    out = {}
    for i, ln in enumerate(visual_links):
        out[ln] = _digits_only_label(ln, label_texts_raw.get(ln), i)
    return out


def _draw_arrow_with_bg(image: np.ndarray, p0: np.ndarray, p1: np.ndarray, rgb: np.ndarray):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    n = float(np.linalg.norm(d))
    if n <= 1e-6:
        return
    u = d / n
    t_main = int(max(1, MOTION_ARROW_THICKNESS))
    t_bg = int(max(t_main + 2, 4))
    gop._draw_line(image, p0[0], p0[1], p1[0], p1[1], np.array([0, 0, 0], dtype=np.uint8), thickness=t_bg)
    gop._draw_line(image, p0[0], p0[1], p1[0], p1[1], rgb, thickness=t_main)
    perp = np.asarray([-u[1], u[0]], dtype=float)
    head_len = max(8.0, n * 0.30)
    head_w = max(6.0, n * 0.20)
    left = p1 - u * head_len + perp * (0.5 * head_w)
    right = p1 - u * head_len - perp * (0.5 * head_w)
    gop._draw_line(image, p1[0], p1[1], left[0], left[1], np.array([0, 0, 0], dtype=np.uint8), thickness=t_bg)
    gop._draw_line(image, p1[0], p1[1], right[0], right[1], np.array([0, 0, 0], dtype=np.uint8), thickness=t_bg)
    gop._draw_line(image, p1[0], p1[1], left[0], left[1], rgb, thickness=t_main)
    gop._draw_line(image, p1[0], p1[1], right[0], right[1], rgb, thickness=t_main)


def _scale_projected_box(
    box: tuple[int, int, int, int] | list[int] | None,
    resolution: tuple[int, int],
    *,
    scale: float = 1.0,
    min_size_px: int = 4,
) -> tuple[int, int, int, int] | None:
    if not (isinstance(box, (tuple, list)) and len(box) == 4):
        return None
    width, height = int(resolution[0]), int(resolution[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    scale_f = max(1.0, float(scale))
    half_w = max(0.5 * float(min_size_px), 0.5 * max(1.0, x1 - x0) * scale_f)
    half_h = max(0.5 * float(min_size_px), 0.5 * max(1.0, y1 - y0) * scale_f)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    sx0 = int(round(cx - half_w))
    sy0 = int(round(cy - half_h))
    sx1 = int(round(cx + half_w))
    sy1 = int(round(cy + half_h))
    sx0 = max(0, min(width - 1, sx0))
    sy0 = max(0, min(height - 1, sy0))
    sx1 = max(0, min(width - 1, sx1))
    sy1 = max(0, min(height - 1, sy1))
    if sx1 <= sx0:
        sx1 = min(width - 1, sx0 + max(2, int(min_size_px)))
    if sy1 <= sy0:
        sy1 = min(height - 1, sy0 + max(2, int(min_size_px)))
    return (int(sx0), int(sy0), int(sx1), int(sy1))


def _box_from_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    resolution: tuple[int, int],
    *,
    pad: int = 8,
) -> tuple[int, int, int, int]:
    width, height = int(resolution[0]), int(resolution[1])
    a = np.asarray(p0, dtype=float).reshape(-1)
    b = np.asarray(p1, dtype=float).reshape(-1)
    x0 = max(0, min(width - 1, int(math.floor(min(a[0], b[0]) - pad))))
    y0 = max(0, min(height - 1, int(math.floor(min(a[1], b[1]) - pad))))
    x1 = max(0, min(width - 1, int(math.ceil(max(a[0], b[0]) + pad))))
    y1 = max(0, min(height - 1, int(math.ceil(max(a[1], b[1]) + pad))))
    return (x0, y0, x1, y1)


def _clip_segment_to_canvas(
    p0: np.ndarray,
    p1: np.ndarray,
    resolution: tuple[int, int],
    *,
    margin: float = 28.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    width, height = int(resolution[0]), int(resolution[1])
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    bounds = [
        (-dx, x0 - margin),
        (dx, float(width - margin) - x0),
        (-dy, y0 - margin),
        (dy, float(height - margin) - y0),
    ]
    for p, q in bounds:
        if abs(p) <= 1.0e-9:
            if q < 0.0:
                return None
            continue
        t = q / p
        if p < 0.0:
            if t > t1:
                return None
            if t > t0:
                t0 = t
        else:
            if t < t0:
                return None
            if t < t1:
                t1 = t
    if t1 <= t0:
        return None
    a = np.asarray([x0 + dx * t0, y0 + dy * t0], dtype=float)
    b = np.asarray([x0 + dx * t1, y0 + dy * t1], dtype=float)
    if float(np.linalg.norm(b - a)) <= 10.0:
        return None
    return (a, b)


def _overall_motion_dir_from_cues(
    motion_cues: dict | None,
    *,
    origin_world: np.ndarray,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
) -> np.ndarray | None:
    if not isinstance(motion_cues, dict):
        return None
    planned = motion_cues.get("planned_base_motion") if isinstance(motion_cues.get("planned_base_motion"), dict) else None
    base_motion = motion_cues.get("base_motion") if isinstance(motion_cues.get("base_motion"), dict) else None
    axis = None
    sign = 1.0
    if isinstance(base_motion, dict):
        axis = base_motion.get("axis_world") or (planned.get("axis_world") if isinstance(planned, dict) else None)
        trend = str(base_motion.get("trend") or "").strip().lower()
        if trend == "negative":
            sign = -1.0
        elif trend != "positive" and isinstance(planned, dict):
            axis = planned.get("axis_world")
            sign = 1.0
    elif isinstance(planned, dict):
        axis = planned.get("axis_world")
    if not (isinstance(axis, list) and len(axis) == 3):
        return None
    return _world_axis_to_screen_dir(origin_world, np.asarray(axis, dtype=float), camera, resolution, sign=sign)


def _overall_axis_tag_from_cues(motion_cues: dict | None) -> str | None:
    if not isinstance(motion_cues, dict):
        return None
    planned = motion_cues.get("planned_base_motion") if isinstance(motion_cues.get("planned_base_motion"), dict) else None
    base_motion = motion_cues.get("base_motion") if isinstance(motion_cues.get("base_motion"), dict) else None
    axis = None
    sign = 1.0
    if isinstance(base_motion, dict):
        axis = base_motion.get("axis_world") or (planned.get("axis_world") if isinstance(planned, dict) else None)
        trend = str(base_motion.get("trend") or "").strip().lower()
        if trend == "negative":
            sign = -1.0
    elif isinstance(planned, dict):
        axis = planned.get("axis_world")
    if not (isinstance(axis, list) and len(axis) == 3):
        return None
    signed_axis = np.asarray(axis, dtype=float).reshape(-1)[:3] * float(sign)
    return _signed_axis_label_from_world(signed_axis)


def _draw_overall_motion_context(
    image: np.ndarray,
    *,
    asset_box: tuple[int, int, int, int] | None,
    overall_dir: np.ndarray | None,
    overall_axis_tag: str | None,
    resolution: tuple[int, int],
    overall_rgba: np.ndarray,
    overall_rgb: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    if not (isinstance(asset_box, (tuple, list)) and len(asset_box) == 4):
        return []
    asset_box_vis = _scale_projected_box(asset_box, resolution, scale=TIMELINE_ASSET_BBOX_SCALE)
    if asset_box_vis is None:
        return []
    gop.draw_bbox_outline(image, asset_box_vis, overall_rgba, thickness=3)
    protected = [asset_box_vis]
    d = np.asarray(overall_dir if overall_dir is not None else np.zeros((2,), dtype=float), dtype=float).reshape(-1)
    if d.size < 2:
        return protected
    d = d[:2]
    dn = float(np.linalg.norm(d))
    if dn <= 1.0e-8:
        return protected
    d = d / dn
    width, height = int(resolution[0]), int(resolution[1])
    header_box = (0, 0, min(width - 1, int(0.38 * width)), min(height - 1, int(0.14 * height)))
    axes_box = _motion_axes_overlay_box(width, height)
    arrow_len = float(np.clip(0.42 * max(asset_box_vis[2] - asset_box_vis[0], asset_box_vis[3] - asset_box_vis[1]) + 72.0, 86.0, 150.0))
    ax0, ay0, ax1, _ay1 = [float(v) for v in asset_box_vis]
    cx = 0.5 * (ax0 + ax1)
    arrow_center_y = max(float(header_box[3]) + 18.0, ay0 - 28.0)
    center0 = np.asarray([cx, arrow_center_y], dtype=float)
    candidates = [
        center0,
        np.asarray([cx + 0.18 * (ax1 - ax0), arrow_center_y], dtype=float),
        np.asarray([cx - 0.18 * (ax1 - ax0), arrow_center_y], dtype=float),
    ]
    best = None
    best_score = None
    half_vec = d * (0.5 * arrow_len)
    for center2 in candidates:
        clipped = _clip_segment_to_canvas(center2 - half_vec, center2 + half_vec, resolution, margin=34.0)
        if clipped is None:
            continue
        start, end = clipped
        arrow_box = _box_from_segment(start, end, resolution, pad=max(8, int(MOTION_ARROW_THICKNESS + 6)))
        overlap = _rect_intersection_area(arrow_box, header_box) + _rect_intersection_area(arrow_box, axes_box)
        overlap += _rect_intersection_area(arrow_box, asset_box_vis)
        height_penalty = max(0.0, float(end[1] - (ay0 - 4.0))) + max(0.0, float(start[1] - (ay0 - 4.0)))
        length_penalty = max(0.0, 42.0 - float(np.linalg.norm(end - start)))
        lateral_penalty = abs(float(center2[0]) - cx)
        score = (overlap, height_penalty, length_penalty, lateral_penalty)
        if best_score is None or score < best_score:
            best = (start, end, arrow_box)
            best_score = score
            if overlap <= 1.0e-6 and height_penalty <= 1.0e-6:
                break
    if best is None:
        return protected
    _draw_arrow_with_bg(image, best[0], best[1], np.asarray(overall_rgb, dtype=np.uint8))
    protected.append(tuple(int(v) for v in best[2]))
    tag = str(overall_axis_tag or "").strip()
    if tag:
        mx = 0.5 * float(best[0][0] + best[1][0])
        above_y = min(float(best[0][1]), float(best[1][1])) - 18.0
        below_y = max(float(best[0][1]), float(best[1][1])) + 18.0
        tx, ty, box = gop._label_box(mx, above_y, tag, 4, width, height)
        if box[1] <= 6:
            tx, ty, box = gop._label_box(mx, below_y, tag, 4, width, height)
        gop.draw_text(image, tx, ty, tag, scale=4, color=(255, 255, 255), bg=(20, 20, 20))
        protected.append(tuple(int(v) for v in box))
    return protected


def _rank_box_sides(
    box: tuple[int, int, int, int],
    resolution: tuple[int, int],
    prefer_dir: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    w, h = int(resolution[0]), int(resolution[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    spaces = {
        "left": max(0.0, x0),
        "right": max(0.0, (w - 1) - x1),
        "top": max(0.0, y0),
        "bottom": max(0.0, (h - 1) - y1),
    }
    normals = {
        "left": np.array([-1.0, 0.0], dtype=float),
        "right": np.array([1.0, 0.0], dtype=float),
        "top": np.array([0.0, -1.0], dtype=float),
        "bottom": np.array([0.0, 1.0], dtype=float),
    }
    pd = None
    if prefer_dir is not None:
        p = np.asarray(prefer_dir, dtype=float).reshape(-1)
        if p.size >= 2:
            pd = p[:2]
            n = float(np.linalg.norm(pd))
            if n > 1e-6:
                pd = pd / n
            else:
                pd = None
    denom = max(1.0, float(max(w, h)))
    scored = []
    for side in ["right", "left", "top", "bottom"]:
        score = 0.75 * (float(spaces[side]) / denom)
        if pd is not None:
            score += 0.25 * float(np.dot(normals[side], pd))
        scored.append((score, side))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(side, normals[side]) for _score, side in scored]


def _side_anchor_point(
    box: tuple[int, int, int, int],
    side: str,
    normal: np.ndarray,
    margin: float,
) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if side == "left":
        p = np.asarray([x0, cy], dtype=float)
    elif side == "right":
        p = np.asarray([x1, cy], dtype=float)
    elif side == "top":
        p = np.asarray([cx, y0], dtype=float)
    else:
        p = np.asarray([cx, y1], dtype=float)
    return p + np.asarray(normal, dtype=float) * float(margin)


def _estimate_background_rgb(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    m = max(8, int(min(h, w) * 0.06))
    patches = [
        image[:m, :m],
        image[:m, max(0, w - m) : w],
        image[max(0, h - m) : h, :m],
        image[max(0, h - m) : h, max(0, w - m) : w],
    ]
    vals = []
    for p in patches:
        if p.size == 0:
            continue
        vals.append(p.reshape(-1, 3))
    if not vals:
        return np.array([240, 240, 240], dtype=float)
    arr = np.concatenate(vals, axis=0).astype(float)
    return np.median(arr, axis=0)


def _blend_future_pose_ghost(base_img: np.ndarray, ghost_img: np.ndarray, alpha: float = COVERAGE_GHOST_ALPHA) -> np.ndarray:
    base = np.asarray(base_img, dtype=np.uint8).copy()
    ghost = np.asarray(ghost_img, dtype=np.uint8)
    if base.shape != ghost.shape or base.ndim != 3 or base.shape[2] != 3:
        return base
    bg = _estimate_background_rgb(ghost)
    diff = np.max(np.abs(ghost.astype(np.int16) - bg.reshape(1, 1, 3).astype(np.int16)), axis=2)
    mask = diff >= int(max(8, MOTION_OCCUPANCY_BG_DELTA))
    if not np.any(mask):
        return base
    alpha_f = float(np.clip(alpha, 0.0, 1.0))
    ghost_light = np.clip(np.round(ghost.astype(np.float32) * 0.35 + 255.0 * 0.65), 0, 255).astype(np.uint8)
    out = base.astype(np.float32)
    out[mask] = (1.0 - alpha_f) * out[mask] + alpha_f * ghost_light[mask]
    return np.asarray(np.clip(np.round(out), 0, 255), dtype=np.uint8)


def _mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int], pad: int = 0):
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0 = max(0, x0 - int(pad))
    y0 = max(0, y0 - int(pad))
    x1 = min(w - 1, x1 + int(pad))
    y1 = min(h - 1, y1 + int(pad))
    if x1 < x0 or y1 < y0:
        return
    mask[y0 : y1 + 1, x0 : x1 + 1] = True


def _build_indicator_occupancy_mask(
    image: np.ndarray,
    resolution: tuple[int, int],
    view_boxes: dict[str, tuple[int, int, int, int]] | None,
    label_pos: dict[str, tuple[float, float]] | None,
    label_texts: dict[str, str] | None,
    label_scale: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    bg = _estimate_background_rgb(image)
    diff = np.max(np.abs(image.astype(np.int16) - bg.reshape(1, 1, 3).astype(np.int16)), axis=2)
    occ = diff >= int(max(1, MOTION_OCCUPANCY_BG_DELTA))
    # Reserve top-left timeline header region.
    _mark_rect(occ, (0, 0, min(w - 1, int(0.38 * w)), min(h - 1, int(0.14 * h))), pad=0)
    # Keep bboxes clear.
    for box in (view_boxes or {}).values():
        if isinstance(box, (tuple, list)) and len(box) == 4:
            _mark_rect(occ, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), pad=3)
    # Keep label areas clear even before label rendering.
    for ln, pos in (label_pos or {}).items():
        txt = str((label_texts or {}).get(ln, "") or "").strip()
        if not txt:
            continue
        try:
            _x, _y, box = gop._label_box(pos[0], pos[1], txt, max(1, int(label_scale)), int(resolution[0]), int(resolution[1]))
            _mark_rect(occ, (int(box[0]), int(box[1]), int(box[2]), int(box[3])), pad=2)
        except Exception:
            continue
    return occ


def _line_overlap_ratio(mask: np.ndarray | None, p0: np.ndarray, p1: np.ndarray, radius_px: int) -> float:
    if mask is None:
        return 0.0
    h, w = mask.shape[:2]
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    steps = int(max(1.0, np.linalg.norm(p1 - p0)))
    r = max(1, int(radius_px))
    occ = 0
    total = 0
    for i in range(steps + 1):
        t = float(i) / float(steps)
        x = int(round(p0[0] * (1.0 - t) + p1[0] * t))
        y = int(round(p0[1] * (1.0 - t) + p1[1] * t))
        for yy in range(y - r, y + r + 1):
            for xx in range(x - r, x + r + 1):
                total += 1
                if xx < 0 or yy < 0 or xx >= w or yy >= h or bool(mask[yy, xx]):
                    occ += 1
    return float(occ) / float(max(1, total))


def _arc_points(
    center: np.ndarray,
    radius: float,
    text_normal: np.ndarray,
    clockwise: bool,
    nseg: int = 44,
) -> np.ndarray:
    c = np.asarray(center, dtype=float).reshape(2)
    r = float(max(10.0, radius))
    nvec = np.asarray(text_normal, dtype=float).reshape(-1)
    if nvec.size < 2:
        nvec = np.asarray([1.0, 0.0], dtype=float)
    else:
        nvec = nvec[:2]
    nnv = float(np.linalg.norm(nvec))
    if nnv > 1e-6:
        nvec = nvec / nnv
    else:
        nvec = np.asarray([1.0, 0.0], dtype=float)
    phi = float(math.atan2(nvec[1], nvec[0]))
    span = float(np.deg2rad(165.0))
    if clockwise:
        t0 = phi - 0.5 * span
        t1 = phi + 0.5 * span
    else:
        t0 = phi + 0.5 * span
        t1 = phi - 0.5 * span
    ts = np.linspace(t0, t1, num=max(8, int(nseg)), dtype=float)
    return np.stack([c[0] + r * np.cos(ts), c[1] + r * np.sin(ts)], axis=1)


def _arc_overlap_ratio(mask: np.ndarray | None, arc_pts: np.ndarray, radius_px: int) -> float:
    if mask is None:
        return 0.0
    if arc_pts.shape[0] < 2:
        return 1.0
    scores = []
    for i in range(arc_pts.shape[0] - 1):
        scores.append(_line_overlap_ratio(mask, arc_pts[i], arc_pts[i + 1], radius_px))
    return float(np.mean(scores)) if scores else 1.0


def _mark_polyline(mask: np.ndarray, pts: np.ndarray, radius_px: int):
    if pts is None or pts.shape[0] < 2:
        return
    h, w = mask.shape[:2]
    r = max(1, int(radius_px))
    for i in range(pts.shape[0] - 1):
        p0 = np.asarray(pts[i], dtype=float)
        p1 = np.asarray(pts[i + 1], dtype=float)
        steps = int(max(1.0, np.linalg.norm(p1 - p0)))
        for s in range(steps + 1):
            t = float(s) / float(steps)
            x = int(round(p0[0] * (1.0 - t) + p1[0] * t))
            y = int(round(p0[1] * (1.0 - t) + p1[1] * t))
            x0 = max(0, x - r)
            x1 = min(w - 1, x + r)
            y0 = max(0, y - r)
            y1 = min(h - 1, y + r)
            mask[y0 : y1 + 1, x0 : x1 + 1] = True


def _rect_overlap_ratio(mask: np.ndarray | None, rect: tuple[int, int, int, int]) -> float:
    if mask is None:
        return 0.0
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in rect]
    x0 = max(0, min(w - 1, x0))
    x1 = max(0, min(w - 1, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(0, min(h - 1, y1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    occ = int(np.count_nonzero(mask[y0 : y1 + 1, x0 : x1 + 1]))
    return float(occ) / float(area)


def _project_motion_tracks_2d(
    motion: dict,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
) -> list[np.ndarray]:
    tracks_world = motion.get("tracks_world")
    if not isinstance(tracks_world, list) or not tracks_world:
        return []
    h, w = int(resolution[1]), int(resolution[0])
    out: list[np.ndarray] = []
    for track_world in tracks_world:
        pts3 = np.asarray(track_world, dtype=float)
        if pts3.ndim != 2 or pts3.shape[0] < 2:
            continue
        proj = gop.project_points(pts3, camera, resolution)
        if proj.shape[0] != pts3.shape[0]:
            continue
        mask = proj[:, 2] > 0
        if int(np.count_nonzero(mask)) < 2:
            continue
        pts2 = np.asarray(proj[mask, :2], dtype=float)
        pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
        pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)
        out.append(pts2)
    return out


def _track_path_len_px(track: np.ndarray) -> float:
    pts = np.asarray(track, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def _track_overall_dir_2d(track: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(track, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return None
    vec = np.asarray(pts[-1] - pts[0], dtype=float)
    n = float(np.linalg.norm(vec))
    if n <= 1.0e-8:
        return None
    return vec / n


def _rotation_direction_from_optical_flow_residual(
    motion: dict,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
) -> str | None:
    if not isinstance(motion, dict):
        return None
    center_track_world = np.asarray(motion.get("center_track_world") or [], dtype=float)
    tracks_world = motion.get("tracks_world")
    if center_track_world.ndim != 2 or center_track_world.shape[0] < 2 or center_track_world.shape[1] < 3:
        return None
    if not isinstance(tracks_world, list) or not tracks_world:
        return None
    center_proj = gop.project_points(center_track_world[:, :3], camera, resolution)
    if center_proj.shape[0] != center_track_world.shape[0]:
        return None
    signed_total = 0.0
    evidence_total = 0.0
    for track_world in tracks_world:
        pts3 = np.asarray(track_world, dtype=float)
        if pts3.ndim != 2 or pts3.shape[0] < 2 or pts3.shape[1] < 3:
            continue
        n_steps = min(int(pts3.shape[0]), int(center_track_world.shape[0]))
        if n_steps < 2:
            continue
        proj = gop.project_points(pts3[:n_steps, :3], camera, resolution)
        if proj.shape[0] != n_steps:
            continue
        for i in range(n_steps - 1):
            if (
                center_proj[i, 2] <= 0.0
                or center_proj[i + 1, 2] <= 0.0
                or proj[i, 2] <= 0.0
                or proj[i + 1, 2] <= 0.0
            ):
                continue
            radial = np.asarray(proj[i, :2] - center_proj[i, :2], dtype=float)
            residual = np.asarray((proj[i + 1, :2] - proj[i, :2]) - (center_proj[i + 1, :2] - center_proj[i, :2]), dtype=float)
            r_norm = float(np.linalg.norm(radial))
            f_norm = float(np.linalg.norm(residual))
            if r_norm <= 1.0e-6 or f_norm <= 1.0e-6:
                continue
            radial_math = np.asarray([radial[0], -radial[1]], dtype=float)
            residual_math = np.asarray([residual[0], -residual[1]], dtype=float)
            cross = float(radial_math[0] * residual_math[1] - radial_math[1] * residual_math[0])
            if abs(cross) <= 1.0e-9:
                continue
            weight = min(r_norm, 3.0 * f_norm)
            signed_total += cross * weight
            evidence_total += abs(cross) * weight
    if evidence_total <= 1.0e-6 or abs(signed_total) <= 1.0e-9:
        return None
    return "cw" if signed_total > 0.0 else "ccw"


def _rotation_direction_from_motion_projection(
    motion: dict,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    axis_sign: float = 1.0,
    signed_axis_world: np.ndarray | None = None,
) -> str | None:
    if not isinstance(motion, dict):
        return None
    optical_flow_dir = _rotation_direction_from_optical_flow_residual(motion, camera, resolution)
    if optical_flow_dir is not None:
        return optical_flow_dir
    try:
        pts = np.stack(
            [
                np.asarray(motion["center_prev_world"], dtype=float),
                np.asarray(motion["center_curr_world"], dtype=float),
                np.asarray(motion["ref_prev_world"], dtype=float),
                np.asarray(motion["ref_curr_world"], dtype=float),
            ],
            axis=0,
        )
    except Exception:
        return None
    proj = gop.project_points(pts, camera, resolution)
    if proj.shape[0] != 4 or np.any(proj[:, 2] <= 0):
        return None
    c0 = np.asarray(proj[0, :2], dtype=float)
    c1 = np.asarray(proj[1, :2], dtype=float)
    r0 = np.asarray(proj[2, :2], dtype=float)
    r1 = np.asarray(proj[3, :2], dtype=float)
    sign_factor = float(axis_sign)
    if signed_axis_world is not None:
        axis_proj = _axis_projection_for_camera(
            np.asarray(motion["center_curr_world"], dtype=float),
            np.asarray(signed_axis_world, dtype=float),
            camera,
        )
        facing = float((axis_proj or {}).get("axis_facing_score") or 0.0)
        if abs(facing) > 1.0e-8:
            sign_factor = facing
    cross = float((r0 - c0)[0] * (r1 - c1)[1] - (r0 - c0)[1] * (r1 - c1)[0]) * sign_factor
    if abs(cross) <= 1.0e-6:
        return None
    return "cw" if cross > 0.0 else "ccw"


def _rotation_direction_from_track_about_pivot(
    motion: dict,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    pivot_world: np.ndarray,
) -> str | None:
    if not isinstance(motion, dict):
        return None
    pivot = np.asarray(pivot_world, dtype=float).reshape(-1)
    if pivot.size < 3:
        return None
    proj_pivot = gop.project_points(np.asarray([pivot[:3]], dtype=float), camera, resolution)
    if proj_pivot.shape[0] != 1 or proj_pivot[0, 2] <= 0:
        return None
    pivot2 = np.asarray(proj_pivot[0, :2], dtype=float)
    tracks2d = _project_motion_tracks_2d(motion, camera, resolution)
    signed_total = 0.0
    evidence_total = 0.0
    for track in tracks2d:
        pts = np.asarray(track, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        prev = np.asarray(pts[0] - pivot2, dtype=float)
        if float(np.linalg.norm(prev)) <= 1.0:
            continue
        signed_track = 0.0
        evidence_track = 0.0
        for p in pts[1:]:
            cur = np.asarray(p - pivot2, dtype=float)
            r_prev = float(np.linalg.norm(prev))
            r_cur = float(np.linalg.norm(cur))
            if r_prev <= 1.0 or r_cur <= 1.0:
                prev = cur
                continue
            cross = float(prev[0] * cur[1] - prev[1] * cur[0])
            dot = float(np.dot(prev, cur))
            dtheta = float(math.atan2(cross, dot))
            weight = min(r_prev, r_cur)
            signed_track += dtheta * weight
            evidence_track += abs(dtheta) * weight
            prev = cur
        if evidence_track <= 1.0e-6:
            continue
        signed_total += signed_track
        evidence_total += evidence_track
    if evidence_total <= 1.0e-4 or abs(signed_total) <= 1.0e-6:
        return None
    return "ccw" if signed_total > 0.0 else "cw"


def _world_axis_to_screen_dir(
    origin_world: np.ndarray,
    axis_world: np.ndarray,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    sign: float = 1.0,
) -> np.ndarray | None:
    origin = np.asarray(origin_world, dtype=float).reshape(-1)
    axis = np.asarray(axis_world, dtype=float).reshape(-1)
    if origin.size < 3 or axis.size < 3:
        return None
    axis = axis[:3]
    n = float(np.linalg.norm(axis))
    if n <= 1.0e-8:
        return None
    axis = axis / n
    pts3 = np.stack([origin[:3], origin[:3] + float(sign) * axis], axis=0)
    proj = gop.project_points(pts3, camera, resolution)
    if proj.shape[0] != 2 or np.any(proj[:, 2] <= 0):
        return None
    vec = np.asarray(proj[1, :2] - proj[0, :2], dtype=float)
    nv = float(np.linalg.norm(vec))
    if nv <= 1.0e-8:
        return None
    return vec / nv


def _axis_projection_for_camera(
    origin_world: np.ndarray,
    axis_world: np.ndarray,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict | None:
    origin = np.asarray(origin_world, dtype=float).reshape(-1)
    axis = np.asarray(axis_world, dtype=float).reshape(-1)
    eye = np.asarray(camera[0], dtype=float).reshape(-1)
    if origin.size < 3 or axis.size < 3 or eye.size < 3:
        return None
    axis = axis[:3]
    na = float(np.linalg.norm(axis))
    if na <= 1.0e-8:
        return None
    axis = axis / na
    cam_vec = eye[:3] - origin[:3]
    nc = float(np.linalg.norm(cam_vec))
    if nc <= 1.0e-8:
        return None
    cam_vec = cam_vec / nc
    facing = float(np.dot(axis, cam_vec))
    toward_camera = bool(facing >= 0.0)
    return {
        "axis_toward_camera": toward_camera,
        "axis_facing_score": facing,
        "axis_projection": ("dot_out" if toward_camera else "cross_in"),
        "axis_projection_note": ("DOT OUT" if toward_camera else "CROSS IN"),
    }


def _find_joint_for_child_link(joints: list[dict], link_name: str) -> dict | None:
    best = None
    best_pri = 99
    target = str(link_name or "").strip()
    for joint in joints or []:
        if str(joint.get("child") or "").strip() != target:
            continue
        jt = str(joint.get("type") or "").strip().lower()
        pri = 1 if jt == "fixed" else 0
        if pri < best_pri:
            best = joint
            best_pri = pri
    return best


def _matching_joint_trend(motion_cues: dict | None, link_name: str) -> dict | None:
    if not isinstance(motion_cues, dict):
        return None
    target = str(link_name or "").strip()
    for jt in motion_cues.get("joint_trends") or []:
        if not isinstance(jt, dict):
            continue
        if str(jt.get("link") or "").strip() == target:
            return jt
    return None


def _signed_axis_label_from_world(axis_world: np.ndarray | list[float] | tuple[float, ...]) -> str:
    vec = np.asarray(axis_world, dtype=float).reshape(-1)
    if vec.size < 3:
        return "+X"
    vec = vec[:3]
    idx = int(np.argmax(np.abs(vec)))
    sign = "+" if float(vec[idx]) >= 0.0 else "-"
    return f"{sign}{['X', 'Y', 'Z'][idx]}"


def _local_motion_descriptor_for_link(
    *,
    asset_ctx: dict,
    motion: dict,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    link_tf: dict[str, np.ndarray],
    link_name: str,
    motion_cues: dict | None,
) -> dict | None:
    jt = _matching_joint_trend(motion_cues, link_name)
    if not isinstance(jt, dict):
        return None
    joint_type = str(jt.get("joint_type") or "").strip().lower()
    joint = _find_joint_for_child_link(asset_ctx.get("joints", []), link_name)
    if not isinstance(joint, dict):
        return None
    delta_q = float(jt.get("delta_q") or 0.0)
    trend = str(jt.get("trend") or "").strip().lower()
    signed_joint_dir = 0.0
    if abs(delta_q) > 1.0e-12:
        signed_joint_dir = 1.0 if delta_q > 0.0 else -1.0
    elif trend == "increase":
        signed_joint_dir = 1.0
    elif trend == "decrease":
        signed_joint_dir = -1.0
    if signed_joint_dir == 0.0:
        return None
    tf = np.asarray(link_tf.get(link_name, np.eye(4)), dtype=float)
    center_world = np.asarray(tf[:3, 3], dtype=float)
    axis_local = np.asarray(joint.get("axis") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
    if axis_local.size < 3:
        return None
    axis_local = axis_local[:3]
    axis_local = axis_local / max(1.0e-8, float(np.linalg.norm(axis_local)))
    world_axis = np.asarray(tf[:3, :3], dtype=float) @ axis_local
    world_axis = world_axis / max(1.0e-8, float(np.linalg.norm(world_axis)))
    signed_axis = np.asarray(world_axis * signed_joint_dir, dtype=float)
    axis_projection = _axis_projection_for_camera(center_world, signed_axis, camera) or {}

    if joint_type in {"revolute", "continuous"}:
        if abs(delta_q) > 1.0e-12:
            rot_dir = "ccw" if delta_q > 0.0 else "cw"
        elif trend == "increase":
            rot_dir = "ccw"
        elif trend == "decrease":
            rot_dir = "cw"
        else:
            rot_dir = "static"
        axis_label = _signed_axis_label_from_world(signed_axis)
        return {
            "motion_type": "rotation",
            "direction": str(rot_dir),
            "axis_world": [float(world_axis[0]), float(world_axis[1]), float(world_axis[2])],
            "signed_axis_world": [float(signed_axis[0]), float(signed_axis[1]), float(signed_axis[2])],
            "axis_label": axis_label,
            "local_motion_text": f"{str(rot_dir)} around {axis_label}",
            "frame_note": "axis_relative_not_view_relative",
            "axis_projection": axis_projection.get("axis_projection"),
            "axis_projection_note": axis_projection.get("axis_projection_note"),
            "axis_projection_tag": (
                f"{axis_label} {axis_projection.get('axis_projection_note')}"
                if axis_projection.get("axis_projection_note")
                else None
            ),
            "axis_toward_camera": axis_projection.get("axis_toward_camera"),
        }

    if joint_type == "prismatic":
        axis_label = _signed_axis_label_from_world(signed_axis)
        return {
            "motion_type": "prismatic",
            "direction": ("positive_axis" if signed_joint_dir > 0.0 else "negative_axis"),
            "axis_world": [float(world_axis[0]), float(world_axis[1]), float(world_axis[2])],
            "signed_axis_world": [float(signed_axis[0]), float(signed_axis[1]), float(signed_axis[2])],
            "axis_label": axis_label,
            "local_motion_text": f"along {axis_label}",
            "frame_note": "axis_relative_not_view_relative",
            # Prismatic motion should be read from the signed-axis screen projection,
            # not from DOT/CROSS rotational disambiguation cues.
            "axis_projection": None,
            "axis_projection_note": None,
            "axis_projection_tag": None,
            "axis_toward_camera": None,
        }
    return None


def build_local_motion_descriptor(
    *,
    asset_ctx: dict,
    traj_data,
    frame_idx: int,
    viewspec: dict,
    camera_anchor_center: np.ndarray,
    camera_anchor_radius: float,
    resolution: tuple[int, int],
    link_name: str,
    motion_window: tuple[int, int] | None,
    motion_cues: dict | None,
) -> dict | None:
    cam = compute_camera_for_viewspec(np.asarray(camera_anchor_center, dtype=float), float(camera_anchor_radius), dict(viewspec))
    fi, _joint_pos, _base_tf = _frame_state_from_traj(traj_data, frame_idx)
    link_tf = rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], _joint_pos, base_tf=_base_tf)
    motion_vectors = _compute_link_motion_vectors(
        asset_ctx,
        traj_data,
        fi,
        link_tf,
        [str(link_name)],
        motion_window=motion_window,
        trace_variant_index=0,
        use_best_trace_candidate=False,
        use_edge_variant_candidate=False,
    )
    motion = motion_vectors.get(str(link_name)) if isinstance(motion_vectors, dict) else None
    if not isinstance(motion, dict):
        return None
    return _local_motion_descriptor_for_link(
        asset_ctx=asset_ctx,
        motion=motion,
        camera=cam,
        resolution=resolution,
        link_tf=link_tf,
        link_name=str(link_name),
        motion_cues=motion_cues,
    )


def _choose_local_arrow_rect(
    *,
    resolution: tuple[int, int],
    link_box: tuple[int, int, int, int],
    asset_box: tuple[int, int, int, int] | None,
    label_box: tuple[int, int, int, int] | None,
    overlay_box_size: tuple[float, float],
    occupancy_mask: np.ndarray | None,
    extra_obstacles: list[tuple[int, int, int, int]] | None = None,
) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = [float(v) for v in link_box]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w, h = int(resolution[0]), int(resolution[1])
    box_w, box_h = float(overlay_box_size[0]), float(overlay_box_size[1])
    xyz_box = _motion_axes_overlay_box(int(w), int(h))
    obstacles = [
        (8.0, 8.0, 152.0, 72.0),
        tuple(float(v) for v in xyz_box),
        (x0, y0, x1, y1),
    ]
    if isinstance(asset_box, (tuple, list)) and len(asset_box) == 4:
        ax0, ay0, ax1, ay1 = [float(v) for v in asset_box]
        obstacles.append((ax0 - 2.0, ay0 - 2.0, ax1 + 2.0, ay1 + 2.0))
    else:
        ax0 = ay0 = ax1 = ay1 = None
    if isinstance(label_box, (tuple, list)) and len(label_box) == 4:
        obstacles.append(tuple(float(v) for v in label_box))
    for ob in extra_obstacles or []:
        if isinstance(ob, (tuple, list)) and len(ob) == 4:
            obstacles.append(tuple(float(v) for v in ob))

    def _clamp_candidate(cand_x: float, cand_y: float) -> tuple[float, float]:
        max_x = max(4.0, float(w) - 4.0 - box_w)
        max_y = max(4.0, float(h) - 4.0 - box_h)
        return (
            min(max(4.0, float(cand_x)), max_x),
            min(max(4.0, float(cand_y)), max_y),
        )

    def _score_rect(rect: tuple[float, float, float, float]) -> float:
        rx0, ry0, rx1, ry1 = rect
        if rx0 < 4 or ry0 < 4 or rx1 > w - 4 or ry1 > h - 4:
            return 1.0e9
        padded_rect = (rx0 - 10.0, ry0 - 10.0, rx1 + 10.0, ry1 + 10.0)
        overlap = 0.0
        for ob in obstacles:
            overlap += _rect_intersection_area(padded_rect, ob)
        occ_ratio = _rect_overlap_ratio(
            occupancy_mask,
            (int(round(rx0)), int(round(ry0)), int(round(rx1)), int(round(ry1))),
        )
        dist = abs((0.5 * (rx0 + rx1)) - cx) + abs((0.5 * (ry0 + ry1)) - cy)
        return overlap * 1000.0 + occ_ratio * 800.0 + dist

    outer_gap = 14.0
    top_y = ((ay0 if ax0 is not None else y0) - box_h - outer_gap)
    if label_box is not None:
        top_y = min(top_y, float(label_box[1]) - box_h - 8.0)
    left_anchor = ((ax0 if ax0 is not None else x0) - box_w - outer_gap)
    right_anchor = ((ax1 if ax1 is not None else x1) + outer_gap)
    below_y = ((ay1 if ay1 is not None else y1) + outer_gap)
    candidates = [
        (cx - 0.5 * box_w, top_y),
        (left_anchor, cy - 0.5 * box_h),
        (right_anchor, cy - 0.5 * box_h),
        (cx - 0.5 * box_w, below_y),
    ]
    best_rect = None
    best_score = None
    for cand_x, cand_y in candidates:
        cand_x, cand_y = _clamp_candidate(cand_x, cand_y)
        rect = (cand_x, cand_y, cand_x + box_w, cand_y + box_h)
        score = _score_rect(rect)
        if best_score is None or score < best_score:
            best_rect = rect
            best_score = score
    return best_rect


def _draw_small_motion_direction_arrow_for_link(
    image: np.ndarray,
    *,
    asset_ctx: dict,
    traj_data: dict,
    frame_idx: int,
    motion_window: tuple[int, int] | None,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    link_tf: dict[str, np.ndarray],
    link_name: str,
    motion: dict,
    link_box: tuple[int, int, int, int],
    asset_box: tuple[int, int, int, int] | None,
    rgb: np.ndarray,
    motion_cues: dict | None,
    label_box: tuple[int, int, int, int] | None,
    occupancy_mask: np.ndarray | None = None,
    extra_obstacles: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    descriptor = _local_motion_descriptor_for_link(
        asset_ctx=asset_ctx,
        motion=motion,
        camera=camera,
        resolution=resolution,
        link_tf=link_tf,
        link_name=link_name,
        motion_cues=motion_cues,
    )
    if not isinstance(descriptor, dict):
        return []
    jt = _matching_joint_trend(motion_cues, link_name)
    if not isinstance(jt, dict):
        return []
    trend = str(jt.get("trend") or "").strip().lower()
    tracks2d = _project_motion_tracks_2d(motion, camera, resolution)
    if not tracks2d:
        return []
    max_path_len = max((_track_path_len_px(track) for track in tracks2d), default=0.0)
    if max_path_len <= 1.0e-8 or max_path_len > float(MOTION_SMALL_TRACE_PATH_THRESHOLD_PX):
        return []

    joint_type = str(jt.get("joint_type") or "").strip().lower()
    joint = _find_joint_for_child_link(asset_ctx.get("joints", []), link_name)
    if not isinstance(joint, dict):
        return []
    delta_q = float(jt.get("delta_q") or 0.0)
    signed_joint_dir = 0.0
    if abs(delta_q) > 1.0e-12:
        signed_joint_dir = 1.0 if delta_q > 0.0 else -1.0
    elif trend == "increase":
        signed_joint_dir = 1.0
    elif trend == "decrease":
        signed_joint_dir = -1.0
    protected: list[tuple[int, int, int, int]] = []
    tf = np.asarray(link_tf.get(link_name, np.eye(4)), dtype=float)
    center_world = np.asarray(tf[:3, 3], dtype=float)

    if joint_type in {"revolute", "continuous"}:
        if signed_joint_dir == 0.0:
            return []
        axis_local = np.asarray(joint.get("axis") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
        if axis_local.size < 3:
            return []
        axis_local = axis_local[:3]
        n = float(np.linalg.norm(axis_local))
        if n <= 1.0e-8:
            return []
        axis_local = axis_local / n
        rot_dir = str(descriptor.get("direction") or "cw").strip().lower()
        target_rect = _choose_local_arrow_rect(
            resolution=resolution,
            link_box=link_box,
            asset_box=asset_box,
            label_box=label_box,
            overlay_box_size=(96.0, 72.0),
            occupancy_mask=occupancy_mask,
            extra_obstacles=extra_obstacles,
        )
        if target_rect is None:
            return []
        import wheel_motion_diag_render as wheel_diag

        wheel_diag._draw_rotational_arrow_reference_style(
            image,
            target_rect=target_rect,
            color=np.asarray(rgb, dtype=np.uint8),
            rot_dir=rot_dir,
            stroke_width=5,
        )
        protected.append(tuple(int(round(v)) for v in target_rect))
        tag_info = wheel_diag._draw_motion_direction_tag(
            image,
            anchor_rect=target_rect,
            tag_text=("CW" if rot_dir == "cw" else "CCW"),
            base_rgba=np.asarray([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0], dtype=float),
            resolution=resolution,
            obstacles=list(extra_obstacles or []) + protected + ([label_box] if label_box is not None else []),
        )
        if isinstance(tag_info, tuple) and len(tag_info) == 3 and isinstance(tag_info[2], tuple):
            protected.append(tuple(int(v) for v in tag_info[2]))
        return protected

    if joint_type == "prismatic":
        if signed_joint_dir == 0.0:
            return []
        axis_local = np.asarray(joint.get("axis") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
        if axis_local.size < 3:
            return []
        axis_local = axis_local[:3]
        n = float(np.linalg.norm(axis_local))
        if n <= 1.0e-8:
            return []
        axis_local = axis_local / n
        world_axis = np.asarray(tf[:3, :3], dtype=float) @ axis_local
        signed_axis = np.asarray(descriptor.get("signed_axis_world") or (world_axis * signed_joint_dir), dtype=float)
        direction = _world_axis_to_screen_dir(
            center_world,
            world_axis,
            camera,
            resolution,
            sign=signed_joint_dir,
        )
        if direction is None and tracks2d:
            best_track = max(tracks2d, key=_track_path_len_px)
            direction = _track_overall_dir_2d(best_track)
        if direction is None:
            return []
        target_rect = _choose_local_arrow_rect(
            resolution=resolution,
            link_box=link_box,
            asset_box=asset_box,
            label_box=label_box,
            overlay_box_size=(118.0, 58.0),
            occupancy_mask=occupancy_mask,
            extra_obstacles=extra_obstacles,
        )
        if target_rect is None:
            return []
        rx0, ry0, rx1, ry1 = [float(v) for v in target_rect]
        center2 = np.asarray([0.5 * (rx0 + rx1), 0.5 * (ry0 + ry1)], dtype=float)
        half_len = 0.28 * min(float(rx1 - rx0), float(ry1 - ry0)) + 18.0
        start = center2 - direction * half_len
        end = center2 + direction * half_len
        _draw_arrow_with_bg(image, start, end, np.asarray(rgb, dtype=np.uint8))
        protected.append(tuple(int(round(v)) for v in target_rect))
        import wheel_motion_diag_render as wheel_diag

        tag_info = wheel_diag._draw_motion_direction_tag(
            image,
            anchor_rect=(rx0, ry0, rx1, ry1),
            tag_text=wheel_diag._signed_axis_label(np.asarray(signed_axis, dtype=float)),
            base_rgba=np.asarray([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 1.0], dtype=float),
            resolution=resolution,
            obstacles=list(extra_obstacles or []) + protected + ([label_box] if label_box is not None else []),
        )
        if isinstance(tag_info, tuple) and len(tag_info) == 3 and isinstance(tag_info[2], tuple):
            protected.append(tuple(int(v) for v in tag_info[2]))
    return protected


def _draw_arc_arrow(
    image: np.ndarray,
    center: np.ndarray,
    radius: float,
    clockwise: bool,
    rgb: np.ndarray,
    text_normal: np.ndarray,
    occupancy_mask: np.ndarray | None = None,
):
    c = np.asarray(center, dtype=float).reshape(2)
    r = float(max(10.0, radius))
    pts = _arc_points(c, r, text_normal, clockwise=clockwise, nseg=44)
    t_main = int(max(1, MOTION_ARROW_THICKNESS))
    t_bg = int(max(t_main + 2, 4))
    for i in range(pts.shape[0] - 1):
        a = pts[i]
        b = pts[i + 1]
        gop._draw_line(image, a[0], a[1], b[0], b[1], np.array([0, 0, 0], dtype=np.uint8), thickness=t_bg)
        gop._draw_line(image, a[0], a[1], b[0], b[1], rgb, thickness=t_main)
    end = pts[-1]
    pre = pts[-3]
    tangent = end - pre
    tn = float(np.linalg.norm(tangent))
    if tn > 1e-6:
        u = tangent / tn
        perp = np.asarray([-u[1], u[0]], dtype=float)
        head_len = max(9.0, r * 0.38)
        head_w = max(7.0, r * 0.26)
        left = end - u * head_len + perp * (0.5 * head_w)
        right = end - u * head_len - perp * (0.5 * head_w)
        gop._draw_line(image, end[0], end[1], left[0], left[1], np.array([0, 0, 0], dtype=np.uint8), thickness=t_bg)
        gop._draw_line(image, end[0], end[1], right[0], right[1], np.array([0, 0, 0], dtype=np.uint8), thickness=t_bg)
        gop._draw_line(image, end[0], end[1], left[0], left[1], rgb, thickness=t_main)
        gop._draw_line(image, end[0], end[1], right[0], right[1], rgb, thickness=t_main)
    txt = "cw" if clockwise else "ccw"
    tn = np.asarray(text_normal, dtype=float).reshape(-1)
    if tn.size < 2:
        tn = np.asarray([1.0, 0.0], dtype=float)
    else:
        tn = tn[:2]
    nn = float(np.linalg.norm(tn))
    if nn > 1e-6:
        tn = tn / nn
    else:
        tn = np.asarray([0.0, -1.0], dtype=float)
    tp = np.asarray([-tn[1], tn[0]], dtype=float)
    th, tw = image.shape[:2]
    text_scale = max(2, int(MOTION_ARROW_TEXT_SCALE))
    occupied = None
    if occupancy_mask is not None:
        occupied = np.asarray(occupancy_mask, dtype=bool).copy()
        _mark_polyline(occupied, pts, radius_px=max(2, int(MOTION_ARROW_THICKNESS + 2)))
    best_pos = c + tn * float(r + 28.0)
    best_score = 1e9
    for dist in [r + 28.0, r + 40.0, r + 54.0, r + 70.0]:
        for lat in [0.0, 14.0, -14.0, 24.0, -24.0]:
            cand = c + tn * float(dist) + tp * float(lat)
            cx, cy, box = gop._label_box(cand[0], cand[1], txt, text_scale, tw, th)
            score = _rect_overlap_ratio(occupied, box)
            # Prefer slightly farther than arc when overlap ties.
            score += 0.001 * max(0.0, 36.0 - float(dist - r))
            if score < best_score:
                best_score = score
                best_pos = np.asarray([cx, cy], dtype=float)
            if score <= 0.01:
                break
        if best_score <= 0.01:
            break
    gop.draw_text(
        image,
        float(best_pos[0]),
        float(best_pos[1]),
        txt,
        scale=text_scale,
        color=(255, 255, 255),
        bg=(20, 20, 20),
    )


def _draw_motion_indicator_for_link(
    image: np.ndarray,
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    motion: dict,
    box: tuple[int, int, int, int],
    rgb: np.ndarray,
    occupancy_mask: np.ndarray | None = None,
    force_rotation: bool = False,
    force_rotation_direction: str | None = None,
    prefer_center_track: bool = False,
    anchor_box: tuple[int, int, int, int] | None = None,
    draw_trace_body: bool = True,
    draw_endpoints: bool = True,
):
    if not isinstance(motion, dict):
        return

    def _draw_disc(img: np.ndarray, xy: np.ndarray, radius: int, color: np.ndarray):
        cx = int(round(float(xy[0])))
        cy = int(round(float(xy[1])))
        rr = max(1, int(radius))
        h, w = img.shape[:2]
        x0 = max(0, cx - rr)
        x1 = min(w - 1, cx + rr)
        y0 = max(0, cy - rr)
        y1 = min(h - 1, cy + rr)
        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy) <= rr * rr:
                    img[yy, xx] = color

    def _draw_hollow_disc(img: np.ndarray, xy: np.ndarray, radius: int, color: np.ndarray, inner_bg: np.ndarray):
        rr = max(2, int(radius))
        ring = max(2, int(max(3, rr * 0.52)))
        _draw_disc(img, xy, rr, color)
        _draw_disc(img, xy, max(1, rr - ring), inner_bg)

    def _trace_color(rgb_base: np.ndarray, t: float) -> np.ndarray:
        # Start: darker/more saturated.
        # End: lighter, with a slightly faster fade than a linear ramp.
        tt = float(np.clip(t, 0.0, 1.0))
        eased = math.sqrt(tt)
        dark_scale = 0.82 + 0.18 * eased
        light_mix = 0.06 + 0.54 * eased
        col = np.asarray(rgb_base, dtype=float) * dark_scale
        col = (1.0 - light_mix) * col + light_mix * 255.0
        return np.asarray(np.clip(np.round(col), 0, 255), dtype=np.uint8)

    def _draw_trace_arrow(
        img: np.ndarray,
        point: np.ndarray,
        direction: np.ndarray,
        color: np.ndarray,
        radius: int,
        thickness: int,
        offset_sign: float = 1.0,
    ):
        p = np.asarray(point, dtype=float).reshape(-1)
        d = np.asarray(direction, dtype=float).reshape(-1)
        if p.size < 2 or d.size < 2:
            return
        d = d[:2]
        dn = float(np.linalg.norm(d))
        if dn <= 1.0e-6:
            return
        u = d / dn
        perp = np.asarray([-u[1], u[0]], dtype=float)
        arrow_len = float(max(radius * 6.0, min(max(radius * 7.4, 20.0), 0.82 * dn)))
        head_len = float(max(radius * 2.2, min(max(radius * 2.8, 9.0), 0.36 * arrow_len)))
        head_w = float(max(radius * 5.2, min(max(radius * 6.2, 16.0), 0.48 * arrow_len)))
        arrow_t = int(max(2, min(int(max(3, thickness + 1)), int(max(3, round(radius * 1.05))))))
        arrow_t_bg = int(max(arrow_t + 1, 4))
        offset = perp * float(offset_sign) * max(0.75, radius * 0.25)
        center = p[:2] + offset
        tail = center - u * (0.38 * arrow_len)
        tip = center + u * (0.62 * arrow_len)
        shaft_end = tip - u * head_len
        left = tip - u * head_len + perp * (0.5 * head_w)
        right = tip - u * head_len - perp * (0.5 * head_w)
        inner_color = np.asarray(np.clip(np.round(np.asarray(color, dtype=float) * 0.62), 0, 255), dtype=np.uint8)
        outline_color = np.asarray([0, 0, 0], dtype=np.uint8)
        for col, thick in ((outline_color, arrow_t_bg), (inner_color, arrow_t)):
            gop._draw_line(img, tail[0], tail[1], shaft_end[0], shaft_end[1], col, thickness=thick)
            gop._draw_line(img, tip[0], tip[1], left[0], left[1], col, thickness=thick)
            gop._draw_line(img, tip[0], tip[1], right[0], right[1], col, thickness=thick)

    def _draw_endpoint_label(
        img: np.ndarray,
        point: np.ndarray,
        text: str,
        flow_dir: np.ndarray,
        occupancy: np.ndarray | None,
        outward_sign: float,
        perp_sign: float,
    ):
        p = np.asarray(point, dtype=float)
        d = np.asarray(flow_dir, dtype=float).reshape(-1)
        if d.size < 2:
            d = np.asarray([1.0, 0.0], dtype=float)
        else:
            d = d[:2]
        dn = float(np.linalg.norm(d))
        if dn <= 1e-6:
            d = np.asarray([1.0, 0.0], dtype=float)
        else:
            d = d / dn
        perp = np.asarray([-d[1], d[0]], dtype=float)
        h, w = img.shape[:2]
        best = None
        best_score = 1e9
        for dist in [18.0, 28.0, 40.0]:
            for lat in [12.0 * perp_sign, 22.0 * perp_sign, 0.0, -12.0 * perp_sign, -22.0 * perp_sign]:
                cand = p + d * float(outward_sign * dist) + perp * float(lat)
                cx, cy, box = gop._label_box(cand[0], cand[1], text, max(2, int(MOTION_ARROW_TEXT_SCALE)), w, h)
                score = _rect_overlap_ratio(occupancy, box)
                if score < best_score:
                    best_score = score
                    best = (cx, cy)
                if score <= 0.01:
                    break
            if best_score <= 0.01:
                break
        if best is None:
            best = (float(p[0]), float(p[1]))
        gop.draw_text(
            img,
            float(best[0]),
            float(best[1]),
            text,
            scale=max(2, int(MOTION_ARROW_TEXT_SCALE)),
            color=(255, 255, 255),
            bg=(20, 20, 20),
        )

    tracks_world = motion.get("tracks_world")
    if isinstance(tracks_world, list) and tracks_world:
        h, w = image.shape[:2]
        kept_tracks = []
        for track_world in tracks_world:
            pts3 = np.asarray(track_world, dtype=float)
            if pts3.ndim != 2 or pts3.shape[0] < 2:
                continue
            proj = gop.project_points(pts3, camera, resolution)
            if proj.shape[0] != pts3.shape[0]:
                continue
            mask = proj[:, 2] > 0
            if int(np.count_nonzero(mask)) < 2:
                continue
            pts2 = np.asarray(proj[mask, :2], dtype=float)
            pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
            pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)
            kept_tracks.append(pts2)
        if kept_tracks:
            dot_r, end_r = _adaptive_trace_radii_from_box(tuple(int(v) for v in box))
            line_t = int(max(2, min(float(MOTION_TRACE_THICKNESS), round(dot_r * 0.9))))
            bg_rgb = _estimate_background_rgb(image)
            endpoint_markers = []
            def _track_len(track: np.ndarray) -> float:
                if track.ndim != 2 or track.shape[0] < 2:
                    return 0.0
                return float(np.linalg.norm(np.diff(track, axis=0), axis=1).sum())

            kept_tracks.sort(key=_track_len, reverse=True)
            for pts2 in kept_tracks:
                n_pts = int(pts2.shape[0])
                if n_pts < 2:
                    continue
                arrow_indices: list[int] = []
                global_dir = np.asarray(pts2[-1] - pts2[0], dtype=float)
                global_dir_norm = float(np.linalg.norm(global_dir))
                if global_dir_norm > 1.0e-6:
                    global_dir = global_dir / global_dir_norm

                def _smoothed_dir(track_pts: np.ndarray, idx: int) -> np.ndarray:
                    wdir = max(2, min(4, int(len(track_pts) // 8) + 1))
                    prev_idx = max(0, int(idx) - int(wdir))
                    next_idx = min(len(track_pts) - 1, int(idx) + int(wdir))
                    if next_idx == prev_idx:
                        return np.asarray([0.0, 0.0], dtype=float)
                    return np.asarray(track_pts[next_idx] - track_pts[prev_idx], dtype=float)

                def _pick_arrow_idx(lo: int, hi: int, fallback_idx: int) -> int:
                    lo_i = max(0, min(n_pts - 2, int(lo)))
                    hi_i = max(lo_i, min(n_pts - 2, int(hi)))
                    best_idx = max(0, min(n_pts - 2, int(fallback_idx)))
                    best_score = -1.0e18
                    for idx_i in range(lo_i, hi_i + 1):
                        local_dir = _smoothed_dir(pts2, idx_i)
                        local_norm = float(np.linalg.norm(local_dir))
                        if local_norm <= 1.0e-6:
                            continue
                        align = float(np.dot(local_dir / local_norm, global_dir)) if global_dir_norm > 1.0e-6 else 0.0
                        score = align + 0.02 * local_norm
                        if score > best_score:
                            best_score = score
                            best_idx = idx_i
                    return int(best_idx)

                if draw_trace_body and n_pts >= 3:
                    arrow_indices.append(0)
                    if n_pts >= 14:
                        arrow_indices.extend(
                            [
                                _pick_arrow_idx(max(1, n_pts // 5), max(1, n_pts // 2 - 2), max(1, n_pts // 3)),
                                _pick_arrow_idx(max(1, n_pts // 2 + 1), max(1, n_pts - 4), max(1, (2 * n_pts) // 3)),
                            ]
                        )
                    elif n_pts >= 8:
                        arrow_indices.append(_pick_arrow_idx(max(1, n_pts // 4), max(1, n_pts - 4), max(1, n_pts // 2)))
                    late_offset = 2 if n_pts >= 7 else 1
                    arrow_indices.append(_pick_arrow_idx(max(1, n_pts - 5), n_pts - 2, max(1, n_pts - 1 - late_offset)))
                    deduped_arrow_indices = []
                    seen_arrow_idx: set[int] = set()
                    for idx in arrow_indices:
                        idx_i = int(idx)
                        if idx_i < 0 or idx_i >= n_pts - 1:
                            continue
                        if idx_i in seen_arrow_idx:
                            continue
                        seen_arrow_idx.add(idx_i)
                        deduped_arrow_indices.append(idx_i)
                    arrow_indices = deduped_arrow_indices
                if draw_trace_body:
                    for i in range(n_pts - 1):
                        t = float(i + 1) / float(max(1, n_pts - 1))
                        seg_rgb = _trace_color(rgb, t)
                        gop._draw_line(image, pts2[i][0], pts2[i][1], pts2[i + 1][0], pts2[i + 1][1], seg_rgb, thickness=line_t)
                    for arrow_order, idx in enumerate(arrow_indices):
                        t = float(idx) / float(max(1, n_pts - 1))
                        arrow_rgb = _trace_color(rgb, t)
                        tangent = _smoothed_dir(pts2, idx)
                        if float(np.linalg.norm(tangent)) <= 1.0e-6:
                            if idx <= 0:
                                tangent = np.asarray(pts2[1] - pts2[0], dtype=float)
                            elif idx >= n_pts - 2:
                                tangent = np.asarray(pts2[n_pts - 1] - pts2[idx], dtype=float)
                            else:
                                tangent = np.asarray(pts2[idx + 1] - pts2[idx - 1], dtype=float)
                        offset_sign = 1.0 if (arrow_order % 2 == 0) else -1.0
                        _draw_trace_arrow(image, pts2[idx], tangent, arrow_rgb, dot_r, line_t, offset_sign=offset_sign)
                for i, p in enumerate(pts2):
                    if i == 0:
                        endpoint_markers.append(("start", np.asarray(p, dtype=float), np.asarray(rgb, dtype=np.uint8)))
                    elif i == n_pts - 1:
                        endpoint_markers.append(("end", np.asarray(p, dtype=float), np.asarray(rgb, dtype=np.uint8)))
                    elif draw_trace_body:
                        t = float(i) / float(max(1, n_pts - 1))
                        pt_rgb = _trace_color(rgb, t)
                        if i not in arrow_indices:
                            _draw_disc(image, p, dot_r, pt_rgb)
            if draw_endpoints:
                for marker_kind, marker_pt, marker_rgb in endpoint_markers:
                    if marker_kind == "start":
                        _draw_disc(image, marker_pt, end_r, marker_rgb)
                    else:
                        _draw_hollow_disc(image, marker_pt, end_r, marker_rgb, bg_rgb)
            return
    return


def _adaptive_trace_radii_from_box(box: tuple[int, int, int, int]) -> tuple[int, int]:
    # The rendered trace size is box-adaptive but capped by the trace hyperparameters
    # above. Increase MOTION_TRACE_POINT_RADIUS / MOTION_TRACE_START_POINT_RADIUS if
    # you want larger sampled dots and larger solid START/END points globally.
    bw = max(1.0, float(box[2] - box[0]))
    bh = max(1.0, float(box[3] - box[1]))
    base_dim = min(bw, bh)
    mid_r = int(max(3, min(float(MOTION_TRACE_POINT_RADIUS), round(base_dim * 0.10))))
    end_r = int(max(mid_r + 3, min(float(MOTION_TRACE_START_POINT_RADIUS), round(base_dim * 0.18))))
    return mid_r, end_r


def _rect_intersection_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(iw * ih)


def _boxes_overlap_local(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 2) -> bool:
    ax0, ay0, ax1, ay1 = [int(v) for v in a]
    bx0, by0, bx1, by1 = [int(v) for v in b]
    return not (ax1 + pad <= bx0 or bx1 + pad <= ax0 or ay1 + pad <= by0 or by1 + pad <= ay0)


def _box_gap_local(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    dx = max(0.0, bx0 - ax1, ax0 - bx1)
    dy = max(0.0, by0 - ay1, ay0 - by1)
    return float(math.hypot(dx, dy))


def _collect_motion_endpoint_protected_boxes(
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolution: tuple[int, int],
    motion_vectors: dict[str, dict],
    view_boxes: dict[str, tuple[int, int, int, int]],
    label_links: list[str],
) -> list[tuple[int, int, int, int]]:
    protected = []
    width, height = resolution
    for ln in label_links or []:
        motion = motion_vectors.get(ln)
        box = view_boxes.get(ln)
        if not isinstance(motion, dict) or not isinstance(box, (tuple, list)) or len(box) != 4:
            continue
        tracks_world = motion.get("tracks_world")
        if not isinstance(tracks_world, list) or not tracks_world:
            continue
        pts3 = np.asarray(tracks_world[0], dtype=float)
        if pts3.ndim != 2 or pts3.shape[0] < 2:
            continue
        proj = gop.project_points(pts3, camera, resolution)
        if proj.shape[0] != pts3.shape[0]:
            continue
        mask = proj[:, 2] > 0
        if int(np.count_nonzero(mask)) < 2:
            continue
        pts2 = np.asarray(proj[mask, :2], dtype=float)
        _, end_r = _adaptive_trace_radii_from_box(tuple(int(v) for v in box))
        pad = int(max(6, end_r + 4))
        for p in (pts2[0], pts2[-1]):
            cx = int(round(float(p[0])))
            cy = int(round(float(p[1])))
            protected.append(
                (
                    max(0, cx - pad),
                    max(0, cy - pad),
                    min(width - 1, cx + pad),
                    min(height - 1, cy + pad),
                )
            )
    return protected


def _adjust_motion_labels_away_from_boxes(
    label_pos: dict[str, tuple[float, float]],
    label_texts: dict[str, str],
    resolution: tuple[int, int],
    scale: int,
    target_boxes: dict[str, tuple[int, int, int, int]] | None = None,
    occupancy_mask: np.ndarray | None = None,
    protected_boxes: list[tuple[int, int, int, int]] | None = None,
) -> dict[str, tuple[float, float]]:
    width, height = resolution
    items = list((label_pos or {}).items())
    items.sort(key=lambda kv: (kv[1][1], kv[1][0], kv[0]))
    step = 12 * max(1, int(scale))
    offsets = [(0, 0)]
    for r in range(1, 5):
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                if dx == 0 and dy == 0:
                    continue
                offsets.append((dx * step, dy * step))
    placed = {}
    occupied = [tuple(int(v) for v in box) for box in (protected_boxes or [])]
    for link_name, (x0, y0) in items:
        text = str(label_texts.get(link_name, "") or "")
        target_box = None
        if isinstance(target_boxes, dict):
            box_raw = target_boxes.get(link_name)
            if isinstance(box_raw, (tuple, list)) and len(box_raw) == 4:
                target_box = tuple(int(v) for v in box_raw)
        best = None
        best_score = None
        for dx, dy in offsets:
            cx, cy, box = gop._label_box(x0 + dx, y0 + dy, text, scale, width, height)
            protected_overlap = sum(1 for ob in occupied if _boxes_overlap_local(box, ob, pad=2))
            target_overlap = 1 if target_box is not None and _boxes_overlap_local(box, target_box, pad=2) else 0
            occ_ratio = _rect_overlap_ratio(occupancy_mask, box)
            box_gap = _box_gap_local(box, target_box) if target_box is not None else 0.0
            center_penalty = abs(float(cx) - float(x0)) + abs(float(cy) - float(y0))
            score = (protected_overlap + target_overlap, occ_ratio, box_gap, center_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best = (float(cx), float(cy), box)
            if protected_overlap == 0 and target_overlap == 0 and occ_ratio <= 0.01:
                break
        if best is None:
            cx, cy, box = gop._label_box(x0, y0, text, scale, width, height)
            best = (float(cx), float(cy), box)
        placed[link_name] = (best[0], best[1])
        occupied.append(tuple(int(v) for v in best[2]))
    return placed


def _box_area(box: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = [int(v) for v in box]
    return float(max(0, x1 - x0) * max(0, y1 - y0))


def _box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in box]
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _should_use_rasterized_box(
    structured_box: tuple[int, int, int, int],
    rasterized_box: tuple[int, int, int, int],
) -> bool:
    s_area = _box_area(structured_box)
    r_area = _box_area(rasterized_box)
    if s_area <= 1.0:
        return r_area > 1.0
    if r_area <= 1.0:
        return False
    ratio = r_area / max(1.0, s_area)
    # Guard against tiny slivers/noise from id-pass visibility.
    if ratio < 0.35 or ratio > 3.5:
        return False
    sx, sy = _box_center(structured_box)
    rx, ry = _box_center(rasterized_box)
    sw = max(1.0, float(structured_box[2] - structured_box[0]))
    sh = max(1.0, float(structured_box[3] - structured_box[1]))
    norm = max(1.0, math.hypot(sw, sh))
    center_shift = math.hypot(rx - sx, ry - sy) / norm
    return center_shift <= 0.65


def _merge_reference_boxes(
    *,
    raster_boxes: dict,
    raster_stats: dict | None = None,
    structured_boxes: dict,
    overlay_boxes: dict,
    visual_links: list[str],
    static_big_link: str | None,
    resolution,
) -> dict:
    """
    Prefer boxes rasterized from the same geometry used for reference rendering.
    Fall back to structured/projected boxes only when a link is missing.
    """
    merged = {}
    aggregate_raster_box = gop.compute_union_box(
        raster_boxes or {},
        visual_links,
        resolution,
    )
    preserve_static_big = static_big_link is not None and aggregate_raster_box is not None
    if preserve_static_big:
        merged[static_big_link] = aggregate_raster_box
    for ln in visual_links or []:
        if preserve_static_big and str(ln) == str(static_big_link):
            continue
        raster_box = (raster_boxes or {}).get(ln)
        structured_box = (structured_boxes or {}).get(ln)
        chosen_box = raster_box
        if raster_box is not None and structured_box is not None:
            raster_box_area = int(((raster_stats or {}).get(str(ln)) or {}).get("box_area_px") or 0)
            sx0, sy0, sx1, sy1 = [int(v) for v in structured_box]
            structured_box_area = max(1, max(0, sx1 - sx0) * max(0, sy1 - sy0))
            if (
                raster_box_area <= int(getattr(gop, "REFERENCE_RASTER_TINY_BOX_AREA_MAX", 400))
                or raster_box_area < float(getattr(gop, "REFERENCE_RASTER_STRUCTURED_FALLBACK_AREA_RATIO", 0.2)) * float(structured_box_area)
            ):
                chosen_box = structured_box
        elif chosen_box is None:
            chosen_box = structured_box
        if chosen_box is not None:
            merged[str(ln)] = chosen_box
    for ln, box in (overlay_boxes or {}).items():
        merged.setdefault(ln, box)
    return merged


def _motion_view_cache_key(cache_token, view: dict, resolution, mode: str):
    return (
        cache_token,
        str(mode),
        int(view.get("azimuth_deg", 0)),
        int(view.get("elevation_deg", 0)),
        round(float(view.get("distance_scale", 1.0)), 6),
        int(view.get("fov_deg", 35)),
        int(resolution[0]),
        int(resolution[1]),
    )


def render_coverage_grid(
    asset_ctx: dict,
    viewspecs: dict,
    out_dir: Path,
    resolution=(800, 600),
    label_links: list[str] | None = None,
    label_mode: str = "name",
    style: str = "overlay",
    preferred_reference_backend: str | None = None,
    camera_anchor_center=None,
    camera_anchor_radius=None,
    ghost_link_transforms: dict[str, np.ndarray] | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    draw_coverage_boxes = bool(COVERAGE_DRAW_BBOX)
    coverage_box_mode = COVERAGE_BOX_MODE if COVERAGE_BOX_MODE in {"points", "raster"} else "points"
    viewspecs = validate_viewspecs(viewspecs)
    link_transforms = rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], {})
    world_link_meshes = transform_link_meshes(asset_ctx["link_meshes"], link_transforms)
    center_cur, radius_cur = compute_base_center_radius(world_link_meshes)
    if camera_anchor_center is None:
        center = np.asarray(center_cur, dtype=float)
    else:
        center = np.asarray(camera_anchor_center, dtype=float)
    if camera_anchor_radius is None:
        radius = float(radius_cur)
    else:
        radius = float(camera_anchor_radius)
    visual_links = [ln for ln, meshes in world_link_meshes.items() if meshes]
    if label_links is None:
        label_links = visual_links
    label_links = [ln for ln in label_links if ln in world_link_meshes and world_link_meshes[ln]]
    colors = gop.build_distinct_link_color_map(visual_links)
    label_texts_raw = {ln: (ln if label_mode == "name" else f"L{i+1}") for i, ln in enumerate(visual_links)}
    label_texts = _sanitize_motion_label_texts(visual_links, label_texts_raw)
    points_by_link = gop.sample_link_points(world_link_meshes, visual_links)
    bbox_points_by_link = (
        gop.collect_link_bbox_points(world_link_meshes, visual_links)
        if draw_coverage_boxes and coverage_box_mode == "raster"
        else {str(k): np.asarray(v, dtype=np.float32) for k, v in points_by_link.items()}
    )
    world_link_meshes_textured = None
    textured_src = asset_ctx.get("link_meshes_textured")
    if isinstance(textured_src, dict):
        try:
            world_link_meshes_textured = transform_link_meshes(textured_src, link_transforms)
        except Exception:
            world_link_meshes_textured = None
    movable_links = gop._movable_links_from_joints(asset_ctx.get("joints", []))
    movable_visual_links = [ln for ln in visual_links if ln in movable_links] if draw_coverage_boxes else []
    line_like_visual_links = (
        gop._line_like_visual_links(
            world_link_meshes_textured if world_link_meshes_textured else world_link_meshes,
            visual_links,
        )
        if draw_coverage_boxes
        else set()
    )
    static_big_link = (
        gop._select_primary_static_link(
            visual_links,
            movable_links,
            world_link_meshes_textured if world_link_meshes_textured else world_link_meshes,
            asset_ctx.get("joints", []),
        )
        if draw_coverage_boxes
        else None
    )
    ref_scene = None
    ref_bbox_points_by_link = None
    reference_meshes_by_link = None
    ghost_ref_scene = None
    ghost_ref_imgs = None
    if style == "reference":
        glb_scene, glb_label_points, glb_bbox_meshes = _build_motion_textured_scene_from_glb_mapping(asset_ctx, link_transforms)
        if glb_scene is not None:
            ref_scene = glb_scene
        if draw_coverage_boxes and coverage_box_mode == "raster" and isinstance(glb_label_points, dict) and glb_label_points:
            ref_bbox_points_by_link = {
                ln: np.asarray(glb_label_points.get(ln, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
                for ln in visual_links
            }
        if world_link_meshes_textured:
            ref_scene = ref_scene or _build_scene_from_world_link_meshes(world_link_meshes_textured)
        if ref_scene is None:
            ref_scene = _build_scene_from_world_link_meshes(world_link_meshes)
        if ref_scene is not None:
            try:
                reference_meshes_by_link = gop.build_reference_meshes_by_link(ref_scene, visual_links)
            except Exception:
                reference_meshes_by_link = None
        if draw_coverage_boxes and ref_bbox_points_by_link is None:
            if isinstance(reference_meshes_by_link, dict) and reference_meshes_by_link:
                ref_bbox_points_by_link = gop.collect_link_bbox_points(reference_meshes_by_link, visual_links)
            elif world_link_meshes_textured:
                if coverage_box_mode == "raster":
                    ref_bbox_points_by_link = gop.collect_link_bbox_points(world_link_meshes_textured, visual_links)
                else:
                    ref_bbox_points_by_link = {str(k): np.asarray(v, dtype=np.float32) for k, v in points_by_link.items()}
            else:
                ref_bbox_points_by_link = dict(bbox_points_by_link)
        if isinstance(ghost_link_transforms, dict) and ghost_link_transforms:
            ghost_world_link_meshes_textured = None
            if isinstance(textured_src, dict):
                try:
                    ghost_world_link_meshes_textured = transform_link_meshes(textured_src, ghost_link_transforms)
                except Exception:
                    ghost_world_link_meshes_textured = None
            ghost_glb_scene, _ghost_label_points, _ghost_bbox_meshes = _build_motion_textured_scene_from_glb_mapping(asset_ctx, ghost_link_transforms)
            if ghost_glb_scene is not None:
                ghost_ref_scene = ghost_glb_scene
            elif ghost_world_link_meshes_textured:
                ghost_ref_scene = _build_scene_from_world_link_meshes(ghost_world_link_meshes_textured)
            else:
                ghost_world_link_meshes = transform_link_meshes(asset_ctx["link_meshes"], ghost_link_transforms)
                ghost_ref_scene = _build_scene_from_world_link_meshes(ghost_world_link_meshes)

    view_images = []
    view_image_paths = []
    all_visible_px = []
    all_visible_ratio = []
    view_ids = []
    link_names_order = visual_links
    blender_ref_imgs = None
    software_ref_imgs = None
    effective_render_backend = "overlay"
    if style == "reference":
        cams = [compute_camera_for_viewspec(center, radius, view) for view in viewspecs["views"]]
        preferred_backend = gop.normalize_reference_backend_name(preferred_reference_backend, default="auto")
        glb_path = asset_ctx.get("reference_glb_path")
        if blender_ref_imgs is None and preferred_backend != "software" and glb_path is not None and Path(glb_path).exists():
            payload_views = []
            for idx, (cam, view) in enumerate(zip(cams, viewspecs["views"])):
                eye, target, up = cam
                payload_views.append(
                    {
                        "id": str(view.get("id", f"V{idx+1}")),
                        "eye": np.asarray(eye, dtype=float).tolist(),
                        "target": np.asarray(target, dtype=float).tolist(),
                        "up": np.asarray(up, dtype=float).tolist(),
                    }
                )
            try:
                blender_ref_imgs = br.render_views_from_glb(
                    glb_path,
                    payload_views,
                    tuple(int(x) for x in resolution),
                    fov_deg=50.0,
                    frame_idx=0,
                    keep_animation=True,
                )
                blender_ref_imgs = [gop.enhance_textured_image(img) for img in blender_ref_imgs]
                blender_ref_imgs, effective_render_backend = _prefer_software_when_blender_washed_out(
                    blender_ref_imgs,
                    ref_scene,
                    cams,
                    resolution,
                    "Blender coverage render looks washed out",
                )
                if effective_render_backend == "blender" and any(gop.is_reference_image_too_dark(img) for img in blender_ref_imgs):
                    print("[WARN] Blender coverage render includes dark views; keeping Blender batch to preserve textures.")
            except Exception:
                blender_ref_imgs = None
        if blender_ref_imgs is None and preferred_backend != "software" and ref_scene is not None and len(ref_scene.geometry) > 0 and gop.scene_has_effective_textures(ref_scene):
            payload_views = []
            for idx, (cam, view) in enumerate(zip(cams, viewspecs["views"])):
                eye, target, up = cam
                payload_views.append(
                    {
                        "id": str(view.get("id", f"V{idx+1}")),
                        "eye": np.asarray(eye, dtype=float).tolist(),
                        "target": np.asarray(target, dtype=float).tolist(),
                        "up": np.asarray(up, dtype=float).tolist(),
                    }
                )
            try:
                blender_ref_imgs = br.render_views_from_scene(
                    ref_scene,
                    payload_views,
                    tuple(int(x) for x in resolution),
                    fov_deg=50.0,
                )
                blender_ref_imgs = [gop.enhance_textured_image(img) for img in blender_ref_imgs]
                blender_ref_imgs, effective_render_backend = _prefer_software_when_blender_washed_out(
                    blender_ref_imgs,
                    ref_scene,
                    cams,
                    resolution,
                    "Blender coverage render looks washed out",
                )
                if effective_render_backend == "blender" and any(gop.is_reference_image_too_dark(img) for img in blender_ref_imgs):
                    print("[WARN] Blender coverage render includes dark views; keeping Blender batch to preserve textures.")
            except Exception:
                blender_ref_imgs = None
        try:
            if preferred_backend != "software" and blender_ref_imgs is None:
                blender_ref_imgs, effective_render_backend = _render_views_with_blender(
                    asset_ctx,
                    link_transforms,
                    cams,
                    viewspecs["views"],
                    resolution,
                )
        except Exception:
            blender_ref_imgs = None
        if blender_ref_imgs is None and ref_scene is not None and len(ref_scene.geometry) > 0:
            software_ref_imgs = _render_views_with_software_scene(ref_scene, cams, resolution)
        if ghost_ref_scene is not None and len(getattr(ghost_ref_scene, "geometry", {})) > 0:
            ghost_ref_imgs = _render_views_with_software_scene(ghost_ref_scene, cams, resolution)
        if blender_ref_imgs is not None:
            if effective_render_backend == "overlay":
                effective_render_backend = "blender"
        elif software_ref_imgs is not None:
            effective_render_backend = "software"
    for view in viewspecs["views"]:
        cam = compute_camera_for_viewspec(center, radius, view)
        overlay_img, label_pos, _owner, link_names_order, visible_px, visible_ratio = _project_point_masks(
            points_by_link, colors, cam, resolution
        )
        if style == "reference":
            if blender_ref_imgs is not None and len(view_images) < len(blender_ref_imgs):
                img = np.array(blender_ref_imgs[len(view_images)], copy=True)
            elif software_ref_imgs is not None and len(view_images) < len(software_ref_imgs):
                img = np.array(software_ref_imgs[len(view_images)], copy=True)
                effective_render_backend = "software"
            elif ref_scene is not None and len(ref_scene.geometry) > 0:
                img = gop.render_reference_textured(ref_scene, cam, resolution)
                effective_render_backend = "software"
            else:
                img = np.array(overlay_img, copy=True)
            if ghost_ref_imgs is not None and len(view_images) < len(ghost_ref_imgs):
                try:
                    ghost_img = np.asarray(ghost_ref_imgs[len(view_images)], dtype=np.uint8)
                    img = _blend_future_pose_ghost(img, ghost_img, alpha=COVERAGE_GHOST_ALPHA)
                except Exception:
                    pass
            if draw_coverage_boxes:
                reference_points_by_link = ref_bbox_points_by_link if ref_bbox_points_by_link is not None else bbox_points_by_link
                projected_full_boxes = gop.project_link_boxes(reference_points_by_link, cam, resolution)
                structured_boxes = gop.build_structured_overlay_boxes(
                    reference_points_by_link,
                    cam,
                    resolution,
                    visual_links=visual_links,
                    movable_visual_links=movable_visual_links,
                    static_big_link=static_big_link,
                )
                overlay_boxes, overlay_box_stats = gop.project_visible_link_boxes(
                    reference_points_by_link,
                    cam,
                    resolution,
                    return_stats=True,
                )
                if not overlay_boxes:
                    overlay_boxes = gop.project_link_boxes(reference_points_by_link, cam, resolution)
                raster_boxes = {}
                raster_pixel_stats = {}
                reference_visibility_ratio_stats = {
                    str(k): {"visible_ratio": float((v or {}).get("visible_ratio") or 0.0)}
                    for k, v in (overlay_box_stats or {}).items()
                }
                raster_meshes = None
                if isinstance(reference_meshes_by_link, dict) and reference_meshes_by_link:
                    raster_meshes = reference_meshes_by_link
                elif isinstance(glb_bbox_meshes, dict) and glb_bbox_meshes:
                    raster_meshes = glb_bbox_meshes
                elif world_link_meshes_textured:
                    raster_meshes = world_link_meshes_textured
                else:
                    raster_meshes = world_link_meshes
                visibility_links = sorted(
                    {
                        str(x)
                        for x in (list(label_links or []) + ([static_big_link] if static_big_link else []))
                        if str(x)
                    }
                )
                if raster_meshes is not None and visibility_links:
                    scene_depth_vis = None
                    try:
                        _vis_boxes, vis_pixel_stats, scene_depth_vis = gop.project_visible_link_boxes_rasterized(
                            raster_meshes,
                            visibility_links,
                            cam,
                            resolution,
                            return_stats=True,
                            return_scene_depth=True,
                        )
                        reference_visibility_ratio_stats = gop.reference_visible_ratios_by_link_rasterized(
                            raster_meshes,
                            visibility_links,
                            cam,
                            resolution,
                            visible_pixel_stats=vis_pixel_stats,
                            points_by_link={
                                str(ln): np.asarray(reference_points_by_link.get(str(ln), np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
                                for ln in visibility_links
                            },
                            scene_depth=scene_depth_vis,
                        )
                    except Exception:
                        pass
                if coverage_box_mode == "raster" and ref_bbox_points_by_link is not None:
                    scene_depth = None
                    try:
                        raster_boxes, raster_pixel_stats, scene_depth = gop.project_visible_link_boxes_rasterized(
                            raster_meshes,
                            visual_links,
                            cam,
                            resolution,
                            return_stats=True,
                            return_scene_depth=True,
                        )
                        reference_visibility_ratio_stats = gop.reference_visible_ratios_by_link_rasterized(
                            raster_meshes,
                            visual_links,
                            cam,
                            resolution,
                            visible_pixel_stats=raster_pixel_stats,
                            points_by_link=reference_points_by_link,
                            scene_depth=scene_depth,
                        )
                    except Exception:
                        raster_boxes = {}
                        raster_pixel_stats = {}
                        reference_visibility_ratio_stats = {}
                boxes = _merge_reference_boxes(
                    raster_boxes=raster_boxes,
                    raster_stats=raster_pixel_stats,
                    structured_boxes=structured_boxes,
                    overlay_boxes=overlay_boxes,
                    visual_links=visual_links,
                    static_big_link=static_big_link,
                    resolution=resolution,
                )
                # Coverage loop should always box the plan/render links, even when they are
                # interior or weakly visible in the current reference view.
                for ln in label_links or []:
                    if ln not in boxes and ln in projected_full_boxes:
                        boxes[ln] = projected_full_boxes[ln]
                boxes = {k: v for k, v in boxes.items() if k in label_links}
                for ln, box in boxes.items():
                    if ln in colors:
                        gop.draw_bbox_outline(img, box, colors[ln], thickness=2)
                label_pos = gop.adjust_caption_positions_from_boxes(
                    boxes,
                    {ln: label_texts[ln] for ln in boxes.keys()},
                    resolution,
                    scale=LABEL_SCALE_COVERAGE,
                )
            else:
                label_pos = {k: v for k, v in label_pos.items() if k in label_links}
                label_pos = gop.adjust_label_positions(
                    label_pos,
                    {ln: label_texts[ln] for ln in label_pos.keys()},
                    resolution,
                    scale=LABEL_SCALE_COVERAGE,
                )
        else:
            img = overlay_img
            label_pos = {k: v for k, v in label_pos.items() if k in label_links}
            label_pos = gop.adjust_label_positions(
                label_pos,
                {ln: label_texts[ln] for ln in label_pos.keys()},
                resolution,
                scale=LABEL_SCALE_COVERAGE,
            )
        _draw_motion_corner_axes_box(img, cam, resolution)
        for ln, pos in label_pos.items():
            gop.draw_label(img, pos[0], pos[1], label_texts[ln], colors[ln], scale=LABEL_SCALE_COVERAGE)
        _annotate_view_header(img, view, scale=HEADER_SCALE)
        view_id = str(view.get("id") or f"V{len(view_images)+1}")
        view_path = out_dir / f"coverage_view_{view_id}.png"
        Image.fromarray(img).save(view_path)
        view_image_paths.append(view_path)
        view_images.append(img)
        all_visible_px.append(visible_px)
        all_visible_ratio.append(visible_ratio)
        view_ids.append(view["id"])

    rows, cols = _grid_shape_for_nviews(len(view_images))
    grid = _make_grid(view_images, rows=rows, cols=cols)
    grid_path = out_dir / "coverage_grid.png"
    Image.fromarray(grid).save(grid_path)

    visible_px_arr = np.stack(all_visible_px, axis=0) if all_visible_px else np.zeros((0, 0), dtype=np.int32)
    visible_ratio_arr = np.stack(all_visible_ratio, axis=0) if all_visible_ratio else np.zeros((0, 0), dtype=np.float32)
    masks_path = out_dir / "coverage_masks.npz"
    np.savez(
        masks_path,
        link_names=np.array(link_names_order if all_visible_px else visual_links, dtype=object),
        view_ids=np.array(view_ids, dtype=object),
        visible_px=visible_px_arr,
        visible_ratio=visible_ratio_arr,
        resolution=np.array(resolution, dtype=np.int32),
    )
    views_path = out_dir / "coverage_viewspecs.json"
    views_path.write_text(json.dumps(viewspecs, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "grid_path": grid_path,
        "view_image_paths": [str(Path(p).absolute()) for p in view_image_paths],
        "masks_path": masks_path,
        "viewspecs_path": views_path,
        "viewspecs": viewspecs,
        "link_names": (link_names_order if all_visible_px else visual_links),
        "visible_px": visible_px_arr,
        "visible_ratio": visible_ratio_arr,
        "center": np.asarray(center, dtype=float),
        "radius": float(radius),
        "effective_render_backend": effective_render_backend,
    }


def render_motion_grid(
    asset_ctx: dict,
    trajectory_npz: Path,
    frame_idx: int,
    viewspecs: dict,
    out_path: Path,
    resolution=(800, 600),
    label_mode: str = "name",
    style: str = "reference",
    camera_anchor_center=None,
    camera_anchor_radius=None,
    label_links: list[str] | None = None,
    trace_links: list[str] | None = None,
    label_legend: dict[str, str] | None = None,
    grid_caption: str | None = None,
    render_backend: str = "blender",
    preferred_reference_backend: str | None = None,
    motion_window: tuple[int, int] | None = None,
    draw_optical_flow: bool = False,
    show_bbox_labels: bool = True,
    motion_label_scale_override: int | None = None,
    trace_variant_index: int = 0,
    use_best_trace_candidate: bool = False,
    use_edge_variant_candidate: bool = False,
    force_rotation_links: list[str] | set[str] | None = None,
    force_rotation_direction_map: dict[str, str] | None = None,
    motion_cues: dict | None = None,
    draw_local_motion_arrows: bool = True,
    draw_bbox_outlines: bool = True,
    animated_glb_path: str | Path | None = None,
    precomputed_reference_images: list[np.ndarray] | None = None,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    motion_box_mode = MOTION_BOX_MODE if MOTION_BOX_MODE in {"points", "raster"} else "points"
    traj_data = _load_trajectory_data(trajectory_npz)
    fi, joint_pos, base_tf = _frame_state_from_traj(traj_data, frame_idx)
    time_s = np.asarray(traj_data.get("time_s")) if traj_data.get("time_s") is not None else np.zeros((0,), dtype=float)
    animated_fps = None
    if time_s.size >= 2:
        dt = np.diff(time_s)
        dt = dt[np.abs(dt) > 1.0e-8]
        if dt.size > 0:
            animated_fps = int(round(1.0 / float(np.median(dt))))
    asset_ctx["_current_motion_cache_token"] = (traj_data.get("_cache_key"), int(fi))
    link_tf = rp.compute_link_transforms(asset_ctx["links"], asset_ctx["joints"], joint_pos, base_tf=base_tf)
    world_link_meshes = transform_link_meshes(asset_ctx["link_meshes"], link_tf)
    textured_src = asset_ctx.get("link_meshes_textured")
    world_link_meshes_textured = None
    if isinstance(textured_src, dict):
        try:
            world_link_meshes_textured = transform_link_meshes(textured_src, link_tf)
        except Exception:
            world_link_meshes_textured = None
    center_cur, radius_cur = compute_base_center_radius(world_link_meshes)
    if camera_anchor_center is None:
        center = center_cur
    else:
        center = np.asarray(camera_anchor_center, dtype=float)
    if camera_anchor_radius is None:
        radius = float(radius_cur)
    else:
        radius = float(camera_anchor_radius)
    # Motion renders should match the selected coverage-loop view distance by default.
    # Keep the env var for explicit overrides, but default it to 1.0.
    motion_radius_scale = float(np.clip(float(MOTION_CAMERA_RADIUS_SCALE), 0.35, 1.5))
    camera_radius = max(0.05, float(radius) * motion_radius_scale)
    viewspecs = validate_viewspecs(viewspecs)
    if MOTION_FRONT_VIEW_ONLY:
        viewspecs = {
            "look_at_mode": "object_center",
            "views": [
                {"id": "V1", "azimuth_deg": 0, "elevation_deg": 20, "distance_scale": 1.0, "fov_deg": 35},
            ],
        }
    cams = [compute_camera_for_viewspec(center, camera_radius, view) for view in viewspecs["views"]]
    asset_ctx["_current_motion_primary_camera"] = cams[0] if cams else None
    asset_ctx["_current_motion_camera_center"] = np.asarray(center, dtype=float)
    asset_ctx["_current_motion_camera_radius"] = float(camera_radius)
    asset_ctx["_current_motion_primary_view"] = dict((viewspecs.get("views") or [{}])[0])
    asset_ctx["_current_motion_primary_resolution"] = tuple(int(x) for x in resolution)
    visual_links = [ln for ln, meshes in world_link_meshes.items() if meshes]
    rotary_links_default = _rotary_child_links_from_joints(asset_ctx.get("joints", []))
    if force_rotation_links is None:
        force_rotation_links_set = set(rotary_links_default)
    else:
        force_rotation_links_set = {str(x).strip() for x in force_rotation_links if str(x).strip()}
    colors = gop.build_distinct_link_color_map(visual_links)
    if label_legend is not None:
        label_texts_raw = dict(label_legend)
    else:
        label_texts_raw = {ln: (ln if label_mode == "name" else f"L{i+1}") for i, ln in enumerate(visual_links)}
    label_texts = _sanitize_motion_label_texts(visual_links, label_texts_raw)
    if label_links is None:
        label_links = visual_links
    label_links = [ln for ln in label_links if ln in label_texts]
    single_link_mode = len(label_links) == 1
    draw_label_links = list(label_links) if bool(show_bbox_labels) else []
    motion_vectors = {}
    local_points_by_link = _get_motion_sample_local_points(asset_ctx, visual_links)
    points_by_link = _transform_local_points_by_link(local_points_by_link, link_tf)
    bbox_points_by_link = {str(k): np.asarray(v, dtype=np.float32) for k, v in points_by_link.items()}
    movable_links = gop._movable_links_from_joints(asset_ctx.get("joints", []))
    movable_visual_links = [ln for ln in visual_links if ln in movable_links]
    static_big_link = gop._select_primary_static_link(
        visual_links,
        movable_links,
        world_link_meshes_textured if world_link_meshes_textured else world_link_meshes,
        asset_ctx.get("joints", []),
    )
    ref_scene = None
    ref_bbox_points_by_link = None
    ref_bbox_meshes_by_link = None
    glb_scene, glb_label_points, glb_bbox_meshes = _build_motion_textured_scene_from_glb_mapping(
        asset_ctx,
        link_tf,
        cache_key=asset_ctx.get("_current_motion_cache_token"),
    )
    if glb_scene is not None:
        ref_scene = glb_scene
    # For motion diagnostics, prefer boxes derived from the current frame's
    # kinematic link meshes. They stay aligned with the executed trajectory
    # more reliably than pre-mapped GLB nodes for thin rotating parts.
    if world_link_meshes_textured:
        ref_bbox_points_by_link = {str(k): np.asarray(v, dtype=np.float32) for k, v in points_by_link.items()}
        ref_bbox_meshes_by_link = world_link_meshes_textured
    else:
        ref_bbox_points_by_link = dict(bbox_points_by_link)
        ref_bbox_meshes_by_link = world_link_meshes
    if ref_scene is None and world_link_meshes_textured:
        try:
            scene_urdf = _build_scene_from_world_link_meshes(world_link_meshes_textured)
        except Exception:
            scene_urdf = None
        if scene_urdf is not None:
            ref_scene = scene_urdf
    # If a link is missing from URDF-derived bbox geometry, fall back to the GLB mapping for that link only.
    if motion_box_mode == "raster" and isinstance(glb_bbox_meshes, dict) and glb_bbox_meshes:
        for ln in visual_links:
            if ln not in ref_bbox_meshes_by_link or not ref_bbox_meshes_by_link.get(ln):
                ref_bbox_meshes_by_link[ln] = glb_bbox_meshes.get(ln, [])
    if motion_box_mode == "raster" and isinstance(glb_label_points, dict) and glb_label_points:
        for ln in visual_links:
            pts = ref_bbox_points_by_link.get(ln)
            if pts is None or np.asarray(pts).size == 0:
                ref_bbox_points_by_link[ln] = np.asarray(glb_label_points.get(ln, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)

    blender_imgs = None
    software_ref_imgs = None
    effective_render_backend = "overlay"
    preferred_backend = gop.normalize_reference_backend_name(preferred_reference_backend, default="auto")
    if style == "reference" and isinstance(precomputed_reference_images, list) and precomputed_reference_images:
        blender_imgs = [np.asarray(img, dtype=np.uint8).copy() for img in precomputed_reference_images]
        effective_render_backend = "blender_batch"
    elif style == "reference" and preferred_backend != "software" and str(render_backend).lower() == "blender":
        try:
            blender_imgs, effective_render_backend = _render_views_with_blender(
                asset_ctx,
                link_tf,
                cams,
                viewspecs["views"],
                resolution,
                animated_glb_path=animated_glb_path,
                animated_frame_idx=int(fi),
                animated_fps=animated_fps,
            )
        except Exception:
            blender_imgs = None
    if style == "reference" and blender_imgs is None and ref_scene is not None and len(ref_scene.geometry) > 0:
        software_ref_imgs = _render_views_with_software_scene(ref_scene, cams, resolution)
    if style == "reference":
        if blender_imgs is not None:
            if effective_render_backend == "overlay":
                effective_render_backend = "blender"
        elif software_ref_imgs is not None:
            effective_render_backend = "software"
    imgs = []
    draw_motion_indicators = bool(draw_optical_flow) or str(os.environ.get("CODEX_MOTION_DRAW_INDICATORS", "0")).strip().lower() in {"1", "true", "yes", "on"}
    if draw_motion_indicators:
        motion_trace_links = list(trace_links or label_links or visual_links)
        motion_vectors = _compute_link_motion_vectors(
            asset_ctx,
            traj_data,
            fi,
            link_tf,
            motion_trace_links,
            motion_window=motion_window,
            trace_variant_index=trace_variant_index,
            use_best_trace_candidate=use_best_trace_candidate,
            use_edge_variant_candidate=use_edge_variant_candidate,
        )
    for vi, view in enumerate(viewspecs["views"]):
        cam = cams[vi]
        focus_link_name = str(label_links[0]) if single_link_mode and label_links else None
        motion_label_scale = (
            max(2, int(motion_label_scale_override))
            if motion_label_scale_override is not None
            else max(2, int(LABEL_SCALE_MOTION) - 1)
        )
        view_boxes = {}
        reference_points_by_link = ref_bbox_points_by_link if ref_bbox_points_by_link is not None else bbox_points_by_link
        box_links = [str(ln) for ln in label_links if str(ln) in reference_points_by_link]
        if not box_links:
            box_links = [str(ln) for ln in visual_links if str(ln) in reference_points_by_link]
        box_movable_visual_links = [ln for ln in movable_visual_links if ln in box_links]
        box_static_big_link = static_big_link if static_big_link in box_links else None
        view_cache_key = _motion_view_cache_key(asset_ctx.get("_current_motion_cache_token"), view, resolution, motion_box_mode)
        view_cache_root = asset_ctx.get("_motion_view_box_cache")
        cached_view_boxes = view_cache_root.get(view_cache_key) if isinstance(view_cache_root, dict) else None
        if not isinstance(cached_view_boxes, dict):
            projected_full_boxes_all = gop.project_link_boxes(reference_points_by_link, cam, resolution)
            overlay_boxes_all = gop.project_visible_link_boxes(reference_points_by_link, cam, resolution)
            if not overlay_boxes_all:
                overlay_boxes_all = gop.project_link_boxes(reference_points_by_link, cam, resolution)
            structured_boxes_all = gop.build_structured_overlay_boxes(
                reference_points_by_link,
                cam,
                resolution,
                visual_links=visual_links,
                movable_visual_links=movable_visual_links,
                static_big_link=static_big_link,
            )
            if not structured_boxes_all:
                structured_boxes_all = dict(overlay_boxes_all)
            raster_boxes_all = {}
            if motion_box_mode == "raster":
                try:
                    per_link_box_mode = {
                        str(ln): "largest_submesh"
                        for ln in force_rotation_links_set
                        if str(ln) in visual_links
                    }
                    raster_boxes_all = gop.project_visible_link_boxes_rasterized(
                        ref_bbox_meshes_by_link,
                        visual_links,
                        cam,
                        resolution,
                        aggregation_mode_by_link=per_link_box_mode,
                    )
                except Exception:
                    raster_boxes_all = {}
            cached_view_boxes = {
                "projected_full_boxes": projected_full_boxes_all,
                "overlay_boxes": overlay_boxes_all,
                "structured_boxes": structured_boxes_all,
                "raster_boxes": raster_boxes_all,
            }
            if not isinstance(view_cache_root, dict):
                view_cache_root = {}
                asset_ctx["_motion_view_box_cache"] = view_cache_root
            _put_limited_cache(view_cache_root, view_cache_key, cached_view_boxes, max_entries=24)
        projected_full_boxes = dict(cached_view_boxes.get("projected_full_boxes") or {})
        overlay_boxes_for_ref = {str(k): v for k, v in (cached_view_boxes.get("overlay_boxes") or {}).items() if str(k) in box_links}
        if not overlay_boxes_for_ref:
            overlay_boxes_for_ref = {str(k): v for k, v in projected_full_boxes.items() if str(k) in box_links}
        structured_boxes = {str(k): v for k, v in (cached_view_boxes.get("structured_boxes") or {}).items() if str(k) in box_links}
        if not structured_boxes:
            structured_boxes = dict(overlay_boxes_for_ref)
        motion_union_boxes = {}
        if draw_motion_indicators:
            motion_union_boxes = _compute_motion_union_boxes(
                asset_ctx,
                traj_data,
                visual_links,
                cam,
                resolution,
                motion_window=motion_window,
            )
        if style == "reference":
            if blender_imgs is not None and vi < len(blender_imgs):
                img = blender_imgs[vi]
            elif software_ref_imgs is not None and vi < len(software_ref_imgs):
                img = software_ref_imgs[vi]
                effective_render_backend = "software"
            else:
                scene = ref_scene
                if scene is None and world_link_meshes_textured:
                    scene = _build_scene_from_world_link_meshes(world_link_meshes_textured)
                if scene is not None and len(scene.geometry) > 0:
                    img = gop.render_reference_textured(scene, cam, resolution)
                    effective_render_backend = "software"
                else:
                    meshes = []
                    for ms in world_link_meshes.values():
                        meshes.extend([m.copy() for m in ms])
                    img = gop.render_reference_solid(meshes, cam, resolution)
            raster_boxes = {str(k): v for k, v in (cached_view_boxes.get("raster_boxes") or {}).items() if str(k) in box_links}
            ref_boxes = _merge_reference_boxes(
                raster_boxes=raster_boxes,
                structured_boxes=structured_boxes,
                overlay_boxes=overlay_boxes_for_ref,
                visual_links=box_links,
                static_big_link=box_static_big_link,
                resolution=resolution,
            )
            for ln in label_links or []:
                if ln not in ref_boxes and ln in projected_full_boxes:
                    ref_boxes[ln] = projected_full_boxes[ln]
            # Keep moving links on their rasterized box so they do not inherit
            # long support bars or motion-window unions.
            for ln in box_movable_visual_links:
                if ln in force_rotation_links_set:
                    continue
                if ln in projected_full_boxes:
                    ref_boxes[ln] = projected_full_boxes[ln]
                if ln in motion_union_boxes:
                    ref_boxes[ln] = motion_union_boxes[ln]
            ref_boxes = {k: v for k, v in ref_boxes.items() if k in label_links}
            if focus_link_name:
                base_box = (
                    projected_full_boxes.get(focus_link_name)
                    or overlay_boxes_for_ref.get(focus_link_name)
                    or ref_boxes.get(focus_link_name)
                )
                scaled_link_box = _scale_projected_box(base_box, resolution, scale=TIMELINE_LINK_BBOX_SCALE)
                if scaled_link_box is not None:
                    ref_boxes = {focus_link_name: scaled_link_box}
            view_boxes = dict(ref_boxes)
            if draw_bbox_outlines:
                for ln, box in ref_boxes.items():
                    if ln in colors:
                        gop.draw_bbox_outline(img, box, colors[ln], thickness=3)
            label_pos = gop.adjust_caption_positions_from_boxes(
                {ln: box for ln, box in ref_boxes.items() if ln in draw_label_links},
                {ln: label_texts[ln] for ln in draw_label_links if ln in ref_boxes},
                resolution,
                scale=motion_label_scale,
            )
        else:
            img, _lbl = gop.render_overlay_points(points_by_link, colors, cam, resolution)
            label_pos = gop.project_label_positions(points_by_link, cam, resolution)
            label_pos = {k: v for k, v in label_pos.items() if k in draw_label_links}
            view_boxes = {k: v for k, v in structured_boxes.items() if k in label_links}
            if not view_boxes:
                view_boxes = {k: v for k, v in overlay_boxes_for_ref.items() if k in label_links}
            for ln in label_links or []:
                if ln not in view_boxes and ln in projected_full_boxes:
                    view_boxes[ln] = projected_full_boxes[ln]
            if focus_link_name:
                base_box = (
                    projected_full_boxes.get(focus_link_name)
                    or overlay_boxes_for_ref.get(focus_link_name)
                    or view_boxes.get(focus_link_name)
                )
                scaled_link_box = _scale_projected_box(base_box, resolution, scale=TIMELINE_LINK_BBOX_SCALE)
                if scaled_link_box is not None:
                    view_boxes = {focus_link_name: scaled_link_box}
            label_pos = gop.adjust_label_positions(
                label_pos,
                {ln: label_texts[ln] for ln in label_pos.keys()},
                resolution,
                scale=motion_label_scale,
            )
        _draw_motion_corner_axes_box(img, cam, resolution)
        overall_box = gop.compute_union_box(view_boxes, list(view_boxes.keys()), resolution)
        asset_box_all = gop.compute_union_box(
            projected_full_boxes if projected_full_boxes else overlay_boxes_for_ref,
            visual_links,
            resolution,
        )
        overall_motion_protected_boxes: list[tuple[int, int, int, int]] = []
        if isinstance(motion_cues, dict) and bool(motion_cues.get("has_planned_base_motion")):
            overall_motion_dir = _overall_motion_dir_from_cues(
                motion_cues,
                origin_world=np.asarray(center, dtype=float),
                camera=cam,
                resolution=resolution,
            )
            overall_axis_tag = _overall_axis_tag_from_cues(motion_cues)
            overall_motion_protected_boxes = _draw_overall_motion_context(
                img,
                asset_box=(tuple(int(v) for v in asset_box_all) if isinstance(asset_box_all, (tuple, list)) and len(asset_box_all) == 4 else None),
                overall_dir=overall_motion_dir,
                overall_axis_tag=overall_axis_tag,
                resolution=resolution,
                overall_rgba=np.asarray([0.16, 0.16, 0.16, 1.0], dtype=float),
                overall_rgb=np.asarray([46, 46, 46], dtype=np.uint8),
            )
        small_motion_arrow_boxes: list[tuple[int, int, int, int]] = []
        if draw_motion_indicators:
            occ = _build_indicator_occupancy_mask(
                img,
                resolution,
                view_boxes,
                label_pos,
                label_texts,
                motion_label_scale,
            )
            for box in overall_motion_protected_boxes:
                _mark_rect(occ, box, pad=3)
            for ln in label_links:
                mv = motion_vectors.get(ln)
                if not isinstance(mv, dict):
                    continue
                if ln not in colors:
                    continue
                box = view_boxes.get(ln)
                if not isinstance(box, (tuple, list)) or len(box) != 4:
                    continue
                rgb = (np.clip(colors[ln][:3], 0, 1) * 255).astype(np.uint8)
                try:
                    _draw_motion_indicator_for_link(
                        img,
                        cam,
                        resolution,
                        mv,
                        box,
                        rgb,
                        occupancy_mask=occ,
                        force_rotation=(ln in force_rotation_links_set),
                        force_rotation_direction=(
                            str(force_rotation_direction_map.get(ln)).strip().lower()
                            if isinstance(force_rotation_direction_map, dict) and force_rotation_direction_map.get(ln) is not None
                            else None
                        ),
                        prefer_center_track=False,
                        anchor_box=overall_box,
                        draw_trace_body=True,
                        draw_endpoints=False,
                    )
                except Exception:
                    continue
            for ln in label_links:
                mv = motion_vectors.get(ln)
                if not isinstance(mv, dict):
                    continue
                if ln not in colors:
                    continue
                box = view_boxes.get(ln)
                if not isinstance(box, (tuple, list)) or len(box) != 4:
                    continue
                rgb = (np.clip(colors[ln][:3], 0, 1) * 255).astype(np.uint8)
                try:
                    _draw_motion_indicator_for_link(
                        img,
                        cam,
                        resolution,
                        mv,
                        box,
                        rgb,
                        occupancy_mask=occ,
                        force_rotation=(ln in force_rotation_links_set),
                        force_rotation_direction=(
                            str(force_rotation_direction_map.get(ln)).strip().lower()
                            if isinstance(force_rotation_direction_map, dict) and force_rotation_direction_map.get(ln) is not None
                            else None
                        ),
                        prefer_center_track=False,
                        anchor_box=overall_box,
                        draw_trace_body=False,
                        draw_endpoints=True,
                    )
                except Exception:
                    continue
            if draw_local_motion_arrows and isinstance(motion_cues, dict):
                label_box_map = {}
                for ln, pos in (label_pos or {}).items():
                    try:
                        _x, _y, lbox = gop._label_box(
                            pos[0],
                            pos[1],
                            str(label_texts.get(ln, "") or ""),
                            motion_label_scale,
                            int(resolution[0]),
                            int(resolution[1]),
                        )
                        label_box_map[str(ln)] = tuple(int(v) for v in lbox)
                    except Exception:
                        continue
                for ln in label_links:
                    mv = motion_vectors.get(ln)
                    if not isinstance(mv, dict):
                        continue
                    if ln not in colors:
                        continue
                    box = view_boxes.get(ln)
                    if not isinstance(box, (tuple, list)) or len(box) != 4:
                        continue
                    rgb = (np.clip(colors[ln][:3], 0, 1) * 255).astype(np.uint8)
                    try:
                        new_boxes = _draw_small_motion_direction_arrow_for_link(
                            img,
                            asset_ctx=asset_ctx,
                            traj_data=traj_data,
                            frame_idx=fi,
                            motion_window=motion_window,
                            camera=cam,
                            resolution=resolution,
                            link_tf=link_tf,
                            link_name=str(ln),
                            motion=mv,
                            link_box=tuple(int(v) for v in box),
                            asset_box=(tuple(int(v) for v in overall_box) if isinstance(overall_box, (tuple, list)) and len(overall_box) == 4 else None),
                            rgb=rgb,
                            motion_cues=motion_cues,
                            label_box=label_box_map.get(str(ln)),
                            occupancy_mask=occ,
                            extra_obstacles=list(small_motion_arrow_boxes),
                        )
                        if isinstance(new_boxes, list) and new_boxes:
                            small_motion_arrow_boxes.extend(
                                [tuple(int(v) for v in nb) for nb in new_boxes if isinstance(nb, (tuple, list)) and len(nb) == 4]
                            )
                    except Exception:
                        continue
        if draw_label_links and label_pos:
            protected_boxes = _collect_motion_endpoint_protected_boxes(
                cam,
                resolution,
                motion_vectors,
                view_boxes,
                draw_label_links,
            )
            if overall_motion_protected_boxes:
                protected_boxes.extend(list(overall_motion_protected_boxes))
            if small_motion_arrow_boxes:
                protected_boxes.extend(list(small_motion_arrow_boxes))
            if protected_boxes:
                label_pos = _adjust_motion_labels_away_from_boxes(
                    label_pos,
                    label_texts,
                    resolution,
                    motion_label_scale,
                    target_boxes=view_boxes,
                    occupancy_mask=occ if draw_motion_indicators else None,
                    protected_boxes=protected_boxes,
                )
        for ln, pos in label_pos.items():
            gop.draw_label(img, pos[0], pos[1], label_texts[ln], colors[ln], scale=motion_label_scale)
        imgs.append(img)
    rows, cols = _grid_shape_for_nviews(len(imgs))
    grid = _make_grid(imgs, rows=rows, cols=cols)
    grid = _annotate_grid_caption(grid, grid_caption, scale=MOTION_GRID_CAPTION_SCALE)
    Image.fromarray(grid).save(out_path)
    return {
        "frame_idx": fi,
        "out_path": out_path,
        "center_used": np.asarray(center).tolist(),
        "radius_used": float(radius),
        "camera_radius_used": float(camera_radius),
        "camera_radius_scale_used": float(motion_radius_scale),
        "label_legend": label_texts,
        "effective_render_backend": effective_render_backend,
    }


def main():
    parser = argparse.ArgumentParser(description="Render loop coverage/motion diagnostic grids")
    parser.add_argument("--asset_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mode", choices=["coverage", "motion"], required=True)
    parser.add_argument("--viewspecs_json", default=None)
    parser.add_argument("--trajectory_npz", default=None)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--resolution", type=int, nargs=2, default=[800, 600])
    args = parser.parse_args()

    asset_root = Path(args.asset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    ctx = load_asset_context(asset_root)
    if args.viewspecs_json:
        spec = json.loads(Path(args.viewspecs_json).read_text(encoding="utf-8"))
    else:
        spec = DEFAULT_VIEWSPECS
    if args.mode == "coverage":
        render_coverage_grid(ctx, spec, out_dir, resolution=tuple(args.resolution))
    else:
        if not args.trajectory_npz:
            raise SystemExit("--trajectory_npz required for motion mode")
        render_motion_grid(
            ctx,
            Path(args.trajectory_npz),
            args.frame_idx,
            spec,
            Path(out_dir) / "motion_grid.png",
            resolution=tuple(args.resolution),
        )


if __name__ == "__main__":
    main()
