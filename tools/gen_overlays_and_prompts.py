#!/usr/bin/env python3
import argparse
import hashlib
import io
import json
import math
import os
import re
import time
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
import zlib

import blender_render as br
import numpy as np
import scale_context_utils as scu
import torch_accel as tacc

try:
    import trimesh
except Exception as exc:
    print(f"Failed to import trimesh: {exc}")
    sys.exit(1)

try:
    import pybullet as p  # type: ignore
    _HAS_PYBULLET = True
except Exception:
    _HAS_PYBULLET = False

try:
    import pyrender  # type: ignore
    _HAS_PYRENDER = True
except Exception:
    _HAS_PYRENDER = False

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes  # type: ignore
    from pytorch3d.structures import Meshes  # type: ignore
    _HAS_PYTORCH3D = True
except Exception:
    _HAS_PYTORCH3D = False

_TORCH_RASTER_BACKEND_LOGGED = False
_PYTORCH3D_VIS_BACKEND_LOGGED = False

def setup_pyrender_headless():
    if os.environ.get("PYOPENGL_PLATFORM") is None:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    try:
        import pyglet  # type: ignore
        pyglet.options["headless"] = True
    except Exception:
        pass

SOFTWARE_MAX_FACES = 20000
REFERENCE_MAX_FACES = 200000
REFERENCE_SUPERSAMPLE = 3
POINTS_PER_LINK = 6000
BBOX_POINTS_PER_LINK = 0  # 0 = use all vertices for bbox projection
BBOX_SURFACE_POINTS_PER_LINK = int(os.environ.get("BBOX_SURFACE_POINTS_PER_LINK", "80000"))
VISIBLE_BBOX_TRIM_PERCENT = float(os.environ.get("VISIBLE_BBOX_TRIM_PERCENT", "0.0"))
BBOX_LINK_AGGREGATION = os.environ.get("BBOX_LINK_AGGREGATION", "union").strip().lower()
BBOX_EXPAND_RATIO = float(os.environ.get("BBOX_EXPAND_RATIO", "0.03"))
BBOX_EXPAND_MIN_PX = int(os.environ.get("BBOX_EXPAND_MIN_PX", "3"))
BBOX_EXPAND_MAX_PX = int(os.environ.get("BBOX_EXPAND_MAX_PX", "10"))
BBOX_OVERLAY_DILATE_ITERS = int(os.environ.get("BBOX_OVERLAY_DILATE_ITERS", "1"))
BBOX_OVERLAY_TOP_BIAS = float(os.environ.get("BBOX_OVERLAY_TOP_BIAS", "0.30"))
BBOX_OVERLAY_DIST_BIAS = float(os.environ.get("BBOX_OVERLAY_DIST_BIAS", "0.15"))
BBOX_OVERLAY_TOP_PERCENTILE = float(os.environ.get("BBOX_OVERLAY_TOP_PERCENTILE", "55.0"))
REFERENCE_VISIBLE_RATIO_MIN = float(os.environ.get("REFERENCE_VISIBLE_RATIO_MIN", "0.18"))
REFERENCE_SURFACE_DEPTH_REL_EPS = float(os.environ.get("REFERENCE_SURFACE_DEPTH_REL_EPS", "0.005"))
REFERENCE_RASTER_STRUCTURED_FALLBACK_AREA_RATIO = float(os.environ.get("REFERENCE_RASTER_STRUCTURED_FALLBACK_AREA_RATIO", "0.2"))
REFERENCE_RASTER_TINY_BOX_AREA_MAX = int(os.environ.get("REFERENCE_RASTER_TINY_BOX_AREA_MAX", "400"))
SMALL_PART_PROJECTED_BOX_AREA_MAX = int(os.environ.get("SMALL_PART_PROJECTED_BOX_AREA_MAX", "2500"))
SMALL_PART_VISUAL_STD_MIN = float(os.environ.get("SMALL_PART_VISUAL_STD_MIN", "18.0"))
SMALL_PART_VISUAL_CONTRAST_MIN = float(os.environ.get("SMALL_PART_VISUAL_CONTRAST_MIN", "12.0"))
MOVABLE_RASTER_VISIBLE_RATIO_MIN = float(os.environ.get("MOVABLE_RASTER_VISIBLE_RATIO_MIN", "0.03"))
PREPROCESS_REFERENCE_BOX_MODE = os.environ.get("CODEX_PREPROCESS_REFERENCE_BOX_MODE", "points").strip().lower()
PREPROCESS_REFERENCE_VIS_SCALE = float(os.environ.get("CODEX_PREPROCESS_REFERENCE_VIS_SCALE", "0.5"))
POINT_SIZE = 2
VIEW_ANGLES = [(0.0, 20.0), (90.0, 20.0), (180.0, 20.0), (270.0, 20.0)]
RENDER_BACKEND_DECISION_FILENAME = "render_backend_decision.json"
IMAGE_CACHE_META_FILENAME = "image_cache_meta.json"
IMAGE_CACHE_VERSION = 12
PYRENDER_DISABLED = True


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _torch_raster_device():
    return tacc.get_raster_device()


def _write_png(path, array):
    if not isinstance(array, np.ndarray):
        raise ValueError("_write_png expects numpy array")
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected HxWx3 uint8 array")

    height, width, _ = array.shape
    raw = b"".join(b"\x00" + array[i].tobytes() for i in range(height))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag, data):
        return struct.pack(">I", len(data)
        ) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    with open(path, "wb") as f:
        f.write(signature)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def _save_image(image, path):
    if isinstance(image, (bytes, bytearray)):
        with open(path, "wb") as f:
            f.write(image)
    elif isinstance(image, np.ndarray):
        _write_png(path, image)
    else:
        raise ValueError("Unsupported image type for saving")


def normalize_reference_backend_name(value, default: str = "auto") -> str:
    s = str(value or "").strip().lower()
    if s == "software":
        return "software"
    if s == "pyrender":
        return "blender" if PYRENDER_DISABLED else "pyrender"
    if s == "blender":
        return "blender"
    return str(default or "auto").strip().lower()


def reference_backend_decision_path(asset_out: str | Path) -> Path:
    return Path(asset_out) / RENDER_BACKEND_DECISION_FILENAME


def load_reference_backend_decision(asset_out: str | Path) -> dict | None:
    path = reference_backend_decision_path(asset_out)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def save_reference_backend_decision(asset_out: str | Path, decision: dict) -> None:
    path = reference_backend_decision_path(asset_out)
    path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


def image_cache_meta_path(asset_out: str | Path) -> Path:
    return Path(asset_out) / IMAGE_CACHE_META_FILENAME


def load_image_cache_meta(asset_out: str | Path) -> dict | None:
    path = image_cache_meta_path(asset_out)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def save_image_cache_meta(asset_out: str | Path, meta: dict) -> None:
    image_cache_meta_path(asset_out).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _reference_glb_cache_stamp(glb_path: str | Path | None) -> dict[str, int | str | None]:
    if not glb_path:
        return {
            "reference_glb_path": None,
            "reference_glb_mtime_ns": None,
            "reference_glb_size": None,
        }
    p = Path(glb_path)
    try:
        st = p.stat()
        return {
            "reference_glb_path": str(p.resolve()),
            "reference_glb_mtime_ns": int(st.st_mtime_ns),
            "reference_glb_size": int(st.st_size),
        }
    except Exception:
        return {
            "reference_glb_path": str(p.resolve()) if p.exists() else str(p),
            "reference_glb_mtime_ns": None,
            "reference_glb_size": None,
        }


def can_reuse_rendered_images(asset_out: str | Path, resolution, view_count: int, *, reference_glb_path: str | Path | None = None) -> bool:
    asset_out = Path(asset_out)
    if not reference_backend_decision_path(asset_out).exists():
        return False
    for idx in range(1, int(view_count) + 1):
        if not (asset_out / "images" / f"overlay_view_{idx:02d}.png").exists():
            return False
        if not (asset_out / "images" / f"reference_view_{idx:02d}.png").exists():
            return False
    expected_resolution = [int(resolution[0]), int(resolution[1])]
    meta = load_image_cache_meta(asset_out)
    if isinstance(meta, dict):
        meta_resolution = meta.get("resolution")
        expected_glb_stamp = _reference_glb_cache_stamp(reference_glb_path)
        if (
            [int(x) for x in (meta_resolution or [])] == expected_resolution
            and int(meta.get("view_count") or 0) == int(view_count)
            and int(meta.get("version") or 0) == int(IMAGE_CACHE_VERSION)
            and str(meta.get("reference_glb_path") or "") == str(expected_glb_stamp.get("reference_glb_path") or "")
            and meta.get("reference_glb_mtime_ns") == expected_glb_stamp.get("reference_glb_mtime_ns")
            and meta.get("reference_glb_size") == expected_glb_stamp.get("reference_glb_size")
        ):
            return True
    return False


def _parse_floats(text, default=None):
    if text is None:
        return default
    parts = [p for p in text.replace(",", " ").split() if p]
    if not parts:
        return default
    return [float(v) for v in parts]


def _rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    return trimesh.transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")


def _quat_to_matrix(quat):
    # Accept either xyzw or wxyz; detect by assuming last is w if abs(last) > 0.5
    if len(quat) != 4:
        raise ValueError("Quaternion must have 4 elements")
    q = np.array(quat, dtype=float)
    if abs(q[3]) >= 0.5:
        x, y, z, w = q
    else:
        w, x, y, z = q
    return trimesh.transformations.quaternion_matrix([w, x, y, z])


def _origin_to_matrix(origin_xyz, origin_rpy, origin_quat):
    transform = np.eye(4)
    if origin_rpy is not None:
        transform = _rpy_to_matrix(origin_rpy)
    elif origin_quat is not None:
        transform = _quat_to_matrix(origin_quat)
    if origin_xyz is not None:
        transform[:3, 3] = origin_xyz
    return transform


def _resolve_mesh_path(mesh_filename, urdf_dir):
    if mesh_filename is None:
        return None
    mesh_filename = mesh_filename.strip()
    if mesh_filename.startswith("package://"):
        mesh_filename = mesh_filename[len("package://") :]
    if mesh_filename.startswith("file://"):
        mesh_filename = mesh_filename[len("file://") :]
    mesh_path = Path(mesh_filename)
    if mesh_path.is_absolute():
        return mesh_path
    candidate = (urdf_dir / mesh_path).resolve()
    if candidate.exists():
        return candidate
    # Fallback: search by basename under URDF directory
    basename = mesh_path.name
    for found in urdf_dir.rglob(basename):
        return found.resolve()
    return candidate


def _material_rgba(material_tag, material_colors):
    if material_tag is None:
        return None
    color = material_tag.find("color")
    if color is not None:
        return _parse_floats(color.get("rgba"), default=None)
    name = material_tag.get("name")
    if name:
        return material_colors.get(name)
    return None


def _primitive_mesh_from_visual(visual: dict):
    gtype = str(visual.get("geometry_type") or "mesh").lower()
    if gtype == "box":
        size = visual.get("size") or [1.0, 1.0, 1.0]
        return trimesh.creation.box(extents=np.asarray(size, dtype=float))
    if gtype == "cylinder":
        radius = float(visual.get("radius") or 0.5)
        length = float(visual.get("length") or 1.0)
        return trimesh.creation.cylinder(radius=radius, height=length, sections=48)
    if gtype == "sphere":
        radius = float(visual.get("radius") or 0.5)
        return trimesh.creation.icosphere(radius=radius, subdivisions=3)
    return None


def _apply_visual_rgba(mesh, rgba):
    if rgba is None or not isinstance(mesh, trimesh.Trimesh):
        return
    vals = list(rgba)[:4]
    if len(vals) == 3:
        vals.append(1.0)
    if len(vals) != 4:
        return
    color = np.clip(np.asarray(vals, dtype=float), 0.0, 1.0)
    mesh.visual.vertex_colors = np.tile((color * 255.0).astype(np.uint8), (len(mesh.vertices), 1))


def _deterministic_color(name):
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    h = int(digest[:8], 16) / float(0xFFFFFFFF)
    s = 0.68
    v = 0.92
    r, g, b = colorsys_hsv_to_rgb(h, s, v)
    return np.array([r, g, b, 1.0], dtype=float)


def build_distinct_link_color_map(link_names: list[str]) -> dict[str, np.ndarray]:
    """
    Build a deterministic, per-asset color map with unique RGB colors across links.
    This avoids collisions from hashing each link independently.
    """
    ordered = sorted({str(x).strip() for x in (link_names or []) if str(x).strip()})
    if not ordered:
        return {}
    seed = hashlib.md5("|".join(ordered).encode("utf-8")).hexdigest()
    hue_offset = int(seed[:8], 16) / float(0xFFFFFFFF)
    sat_cycle = [0.78, 0.68, 0.58]
    val_cycle = [0.92, 0.86]
    used_rgb8 = set()
    out = {}
    phi = 0.61803398875
    n = len(ordered)
    for i, name in enumerate(ordered):
        h = (hue_offset + float(i) / float(max(1, n))) % 1.0
        s = float(sat_cycle[i % len(sat_cycle)])
        v = float(val_cycle[(i // len(sat_cycle)) % len(val_cycle)])
        r, g, b = colorsys_hsv_to_rgb(h, s, v)
        rgb8 = (int(round(r * 255.0)), int(round(g * 255.0)), int(round(b * 255.0)))
        if rgb8 in used_rgb8:
            # Rare fallback: shift hue until 8-bit color becomes unique.
            for k in range(1, 32):
                hh = (h + phi * float(k)) % 1.0
                rr, gg, bb = colorsys_hsv_to_rgb(hh, s, v)
                rgb8_new = (int(round(rr * 255.0)), int(round(gg * 255.0)), int(round(bb * 255.0)))
                if rgb8_new not in used_rgb8:
                    r, g, b = rr, gg, bb
                    rgb8 = rgb8_new
                    break
        used_rgb8.add(rgb8)
        out[name] = np.array([r, g, b, 1.0], dtype=float)
    return out


def colorsys_hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def _sanitize_filename(name):
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _load_obj_simple(path):
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line:
                continue
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                idx = []
                for p in parts:
                    if "/" in p:
                        p = p.split("/")[0]
                    if not p:
                        continue
                    vi = int(p)
                    if vi < 0:
                        vi = len(vertices) + vi + 1
                    idx.append(vi - 1)
                if len(idx) >= 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    if not vertices or not faces:
        raise ValueError("OBJ has no vertices or faces")
    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)


def _load_mesh(path):
    ext = path.suffix.lower()
    try:
        return trimesh.load(path, force="mesh", skip_materials=True)
    except Exception as exc:
        if "PIL" in str(exc) and ext == ".obj":
            return _load_obj_simple(path)
        raise


def find_urdf(asset_dir):
    urdfs = sorted(Path(asset_dir).rglob("*.urdf"))
    if not urdfs:
        return None
    return urdfs[0]


def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    material_colors = {}
    for material in root.findall("material"):
        name = material.get("name")
        color = material.find("color")
        if name and color is not None:
            rgba = _parse_floats(color.get("rgba"), default=None)
            if rgba is not None:
                material_colors[name] = rgba

    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name:
            continue
        visuals = []
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            origin = visual.find("origin")
            origin_xyz = _parse_floats(origin.get("xyz")) if origin is not None else None
            origin_rpy = _parse_floats(origin.get("rpy")) if origin is not None else None
            origin_quat = _parse_floats(origin.get("quat")) if origin is not None else None
            material_rgba = _material_rgba(visual.find("material"), material_colors)
            mesh_tag = geom.find("mesh")
            if mesh_tag is not None:
                visuals.append(
                    {
                        "geometry_type": "mesh",
                        "filename": mesh_tag.get("filename") or mesh_tag.get("file"),
                        "scale": _parse_floats(mesh_tag.get("scale"), default=[1.0, 1.0, 1.0]),
                        "origin_xyz": origin_xyz,
                        "origin_rpy": origin_rpy,
                        "origin_quat": origin_quat,
                        "material_rgba": material_rgba,
                    }
                )
                continue
            box_tag = geom.find("box")
            if box_tag is not None:
                visuals.append(
                    {
                        "geometry_type": "box",
                        "filename": None,
                        "scale": [1.0, 1.0, 1.0],
                        "size": _parse_floats(box_tag.get("size"), default=[1.0, 1.0, 1.0]),
                        "origin_xyz": origin_xyz,
                        "origin_rpy": origin_rpy,
                        "origin_quat": origin_quat,
                        "material_rgba": material_rgba,
                    }
                )
                continue
            cylinder_tag = geom.find("cylinder")
            if cylinder_tag is not None:
                visuals.append(
                    {
                        "geometry_type": "cylinder",
                        "filename": None,
                        "scale": [1.0, 1.0, 1.0],
                        "radius": float(cylinder_tag.get("radius") or 0.5),
                        "length": float(cylinder_tag.get("length") or 1.0),
                        "origin_xyz": origin_xyz,
                        "origin_rpy": origin_rpy,
                        "origin_quat": origin_quat,
                        "material_rgba": material_rgba,
                    }
                )
                continue
            sphere_tag = geom.find("sphere")
            if sphere_tag is not None:
                visuals.append(
                    {
                        "geometry_type": "sphere",
                        "filename": None,
                        "scale": [1.0, 1.0, 1.0],
                        "radius": float(sphere_tag.get("radius") or 0.5),
                        "origin_xyz": origin_xyz,
                        "origin_rpy": origin_rpy,
                        "origin_quat": origin_quat,
                        "material_rgba": material_rgba,
                    }
                )
        links[name] = visuals

    joints = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        jtype = joint.get("type")
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        origin = joint.find("origin")
        joints.append(
            {
                "name": name,
                "type": jtype,
                "parent": parent.get("link") if parent is not None else None,
                "child": child.get("link") if child is not None else None,
                "axis": _parse_floats(axis.get("xyz")) if axis is not None else None,
                "limit": {
                    "lower": float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
                    "upper": float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
                    "effort": float(limit.get("effort")) if limit is not None and limit.get("effort") else None,
                    "velocity": float(limit.get("velocity")) if limit is not None and limit.get("velocity") else None,
                }
                if limit is not None
                else None,
                "origin": {
                    "xyz": _parse_floats(origin.get("xyz")) if origin is not None else None,
                    "rpy": _parse_floats(origin.get("rpy")) if origin is not None else None,
                }
                if origin is not None
                else None,
            }
        )

    return links, joints


def compute_link_transforms(links, joints):
    joint_tree = {}
    for joint in joints:
        parent = joint.get("parent")
        child = joint.get("child")
        if not parent or not child:
            continue
        joint_tree.setdefault(parent, []).append(joint)

    child_links = set(joint.get("child") for joint in joints if joint.get("child"))
    root_links = [ln for ln in links.keys() if ln not in child_links]
    if not root_links:
        root_links = list(links.keys())

    link_transforms = {root: np.eye(4) for root in root_links}

    def compute_children(parent_link):
        parent_tf = link_transforms[parent_link]
        for joint in joint_tree.get(parent_link, []):
            origin = joint.get("origin") or {}
            origin_tf = _origin_to_matrix(origin.get("xyz"), origin.get("rpy"), None)
            # Zero configuration motion
            motion_tf = np.eye(4)
            joint_tf = origin_tf @ motion_tf
            child_link = joint.get("child")
            if child_link:
                link_transforms[child_link] = parent_tf @ joint_tf
                compute_children(child_link)

    for root in root_links:
        compute_children(root)

    for ln in links.keys():
        link_transforms.setdefault(ln, np.eye(4))

    return link_transforms


def load_link_meshes(links, urdf_dir, link_transforms):
    link_meshes = {}
    for link_name, visuals in links.items():
        meshes = []
        for visual in visuals:
            if str(visual.get("geometry_type") or "mesh").lower() == "mesh":
                mesh_path = _resolve_mesh_path(visual.get("filename"), urdf_dir)
                if mesh_path is None:
                    continue
                if not mesh_path.exists():
                    print(f"[WARN] Mesh not found: {mesh_path}")
                    continue
                try:
                    mesh = _load_mesh(mesh_path)
                except Exception as exc:
                    print(f"[WARN] Failed to load mesh {mesh_path}: {exc}")
                    continue
            else:
                mesh = _primitive_mesh_from_visual(visual)
                if mesh is None:
                    continue
            if isinstance(mesh, trimesh.Scene):
                meshes_list = []
                for geom in mesh.geometry.values():
                    meshes_list.append(geom.copy())
                if meshes_list:
                    mesh = trimesh.util.concatenate(meshes_list)
                else:
                    continue
            else:
                mesh = mesh.copy()
            _apply_visual_rgba(mesh, visual.get("material_rgba"))

            scale = visual["scale"] or [1.0, 1.0, 1.0]
            scale_mat = np.eye(4)
            scale_mat[0, 0] = scale[0]
            scale_mat[1, 1] = scale[1]
            scale_mat[2, 2] = scale[2]

            visual_mat = _origin_to_matrix(
                visual.get("origin_xyz"), visual.get("origin_rpy"), visual.get("origin_quat")
            )
            link_tf = link_transforms.get(link_name, np.eye(4))

            mesh.apply_transform(scale_mat)
            mesh.apply_transform(visual_mat)
            mesh.apply_transform(link_tf)
            meshes.append(mesh)

        if meshes:
            link_meshes[link_name] = meshes
        else:
            link_meshes[link_name] = []
    return link_meshes


def make_identity_transforms(links):
    return {ln: np.eye(4) for ln in links.keys()}


def compute_scene_bounds(link_meshes):
    points = []
    for meshes in link_meshes.values():
        for mesh in meshes:
            if mesh.vertices.size == 0:
                continue
            points.append(mesh.bounds)
    if not points:
        return np.array([0.0, 0.0, 0.0]), 1.0
    bounds = np.vstack(points)
    min_corner = bounds[:, :3].min(axis=0)
    max_corner = bounds[:, :3].max(axis=0)
    center = (min_corner + max_corner) / 2.0
    radius = np.linalg.norm(max_corner - min_corner) * 0.6
    radius = max(radius, 0.1)
    return center, radius


def compute_camera(centroid, radius, azim_deg=45.0, elev_deg=25.0):
    azim = math.radians(azim_deg)
    elev = math.radians(elev_deg)
    dist = radius * 2.5
    eye = np.array(
        [
            centroid[0] + dist * math.cos(elev) * math.cos(azim),
            centroid[1] + dist * math.cos(elev) * math.sin(azim),
            centroid[2] + dist * math.sin(elev),
        ],
        dtype=float,
    )
    target = np.array(centroid, dtype=float)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    return eye, target, up


def camera_pose_from_lookat(eye, target, up):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def project_points(points, camera, resolution):
    width, height = resolution
    eye, target, up = camera
    pose = camera_pose_from_lookat(eye, target, up)
    view = np.linalg.inv(pose)

    fov = math.radians(50.0)
    aspect = width / height
    f = 1.0 / math.tan(fov / 2.0)

    homog = np.column_stack([points, np.ones(len(points))])
    cam = (view @ homog.T).T
    z = -cam[:, 2]
    x_ndc = (cam[:, 0] * f / aspect) / z
    y_ndc = (cam[:, 1] * f) / z
    x_screen = (x_ndc + 1.0) * 0.5 * (width - 1)
    y_screen = (1.0 - (y_ndc + 1.0) * 0.5) * (height - 1)
    return np.column_stack([x_screen, y_screen, z])


def _project_points_ndc(points, camera, resolution):
    width, height = resolution
    eye, target, up = camera
    pose = camera_pose_from_lookat(eye, target, up)
    view = np.linalg.inv(pose)
    pts = np.asarray(points, dtype=np.float32)
    homog = np.column_stack([pts, np.ones((pts.shape[0],), dtype=np.float32)])
    cam = (view @ homog.T).T
    z = -cam[:, 2]
    aspect = float(width) / float(max(1, height))
    f = 1.0 / math.tan(math.radians(50.0) * 0.5)
    z_safe = np.where(np.abs(z) > 1.0e-8, z, np.full_like(z, 1.0e-8))
    x_ndc = (cam[:, 0] * f / aspect) / z_safe
    y_ndc = (cam[:, 1] * f) / z_safe
    return np.column_stack([x_ndc, y_ndc, z]).astype(np.float32, copy=False)


def rasterize_link_visibility_maps(
    link_meshes,
    link_names,
    camera,
    resolution,
    max_faces=REFERENCE_MAX_FACES,
    return_scene_depth: bool = False,
):
    width, height = resolution
    link_names = [str(x) for x in (link_names or [])]
    if width <= 0 or height <= 0 or not link_names:
        owner_link = np.full((max(0, int(height)), max(0, int(width))), -1, dtype=np.int32)
        owner_sub = np.full_like(owner_link, -1, dtype=np.int32)
        if return_scene_depth:
            return owner_link, owner_sub, np.full_like(owner_link, np.inf, dtype=np.float32)
        return owner_link, owner_sub

    device = _torch_raster_device()
    if _HAS_PYTORCH3D and device is not None and _env_true("CODEX_PYTORCH3D_VIS_RASTER", True):
        try:
            verts_blocks = []
            faces_blocks = []
            face_link_idx = []
            face_sub_idx = []
            vert_offset = 0
            for li, link_name in enumerate(link_names):
                mesh_list = link_meshes.get(link_name, []) if isinstance(link_meshes, dict) else []
                for sub_idx, mesh in enumerate(mesh_list):
                    if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
                        continue
                    if getattr(mesh, "faces", None) is None or mesh.faces.size == 0:
                        continue
                    verts = np.asarray(mesh.vertices, dtype=np.float32)
                    faces = np.asarray(mesh.faces, dtype=np.int64)
                    ndc = _project_points_ndc(verts, camera, resolution)
                    if faces.shape[0] > max_faces:
                        step = max(1, faces.shape[0] // max_faces)
                        faces = faces[::step][:max_faces]
                    z = ndc[:, 2]
                    valid_face = np.all(z[faces] > 1.0e-6, axis=1)
                    faces = faces[valid_face]
                    if faces.shape[0] == 0:
                        continue
                    verts_blocks.append(ndc)
                    faces_blocks.append(faces + int(vert_offset))
                    face_link_idx.append(np.full((faces.shape[0],), int(li), dtype=np.int32))
                    face_sub_idx.append(np.full((faces.shape[0],), int(sub_idx), dtype=np.int32))
                    vert_offset += int(verts.shape[0])
            if verts_blocks and faces_blocks:
                verts_cat = np.concatenate(verts_blocks, axis=0)
                faces_cat = np.concatenate(faces_blocks, axis=0)
                face_link_idx_cat = np.concatenate(face_link_idx, axis=0)
                face_sub_idx_cat = np.concatenate(face_sub_idx, axis=0)
                if faces_cat.shape[0] > max_faces:
                    step = max(1, faces_cat.shape[0] // max_faces)
                    keep = np.arange(0, faces_cat.shape[0], step, dtype=np.int64)[:max_faces]
                    faces_cat = faces_cat[keep]
                    face_link_idx_cat = face_link_idx_cat[keep]
                    face_sub_idx_cat = face_sub_idx_cat[keep]
                verts_t = torch.as_tensor(verts_cat, dtype=torch.float32, device=device)
                faces_t = torch.as_tensor(faces_cat, dtype=torch.int64, device=device)
                meshes = Meshes(verts=[verts_t], faces=[faces_t])
                global _PYTORCH3D_VIS_BACKEND_LOGGED
                if not _PYTORCH3D_VIS_BACKEND_LOGGED and _env_true("CODEX_TORCH_RASTER_LOG", False):
                    print(f"[INFO] PyTorch3D visibility raster active on {device}.")
                    _PYTORCH3D_VIS_BACKEND_LOGGED = True
                pix_to_face, zbuf, _bary, _dists = rasterize_meshes(
                    meshes,
                    image_size=(int(height), int(width)),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    bin_size=0,
                    perspective_correct=False,
                    cull_backfaces=False,
                )
                pix = np.asarray(pix_to_face[0, :, :, 0].detach().cpu(), dtype=np.int32)
                z = np.asarray(zbuf[0, :, :, 0].detach().cpu(), dtype=np.float32)
                owner_link = np.full((int(height), int(width)), -1, dtype=np.int32)
                owner_sub = np.full((int(height), int(width)), -1, dtype=np.int32)
                valid = pix >= 0
                if np.any(valid):
                    owner_link[valid] = face_link_idx_cat[pix[valid]]
                    owner_sub[valid] = face_sub_idx_cat[pix[valid]]
                if return_scene_depth:
                    scene_depth = np.full((int(height), int(width)), np.inf, dtype=np.float32)
                    if np.any(valid):
                        scene_depth[valid] = z[valid]
                    return owner_link, owner_sub, scene_depth
                return owner_link, owner_sub
        except Exception as exc:
            if _env_true("CODEX_TORCH_RASTER_WARN", True):
                print(f"[WARN] PyTorch3D visibility raster failed ({exc}); falling back to software rasterizer.")

    meshes = []
    colors = []
    code_to_meta = {}
    next_code = 1
    for link_name in link_names:
        mesh_list = link_meshes.get(link_name, []) if isinstance(link_meshes, dict) else []
        for sub_idx, mesh in enumerate(mesh_list):
            if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
                continue
            if getattr(mesh, "faces", None) is None or mesh.faces.size == 0:
                continue
            m = mesh.copy()
            code = int(next_code)
            next_code += 1
            r = (code & 0xFF) / 255.0
            g = ((code >> 8) & 0xFF) / 255.0
            b = ((code >> 16) & 0xFF) / 255.0
            meshes.append(m)
            colors.append(np.array([r, g, b, 1.0], dtype=float))
            code_to_meta[code] = (str(link_name), int(sub_idx))
    if not meshes:
        owner_link = np.full((int(height), int(width)), -1, dtype=np.int32)
        owner_sub = np.full((int(height), int(width)), -1, dtype=np.int32)
        if return_scene_depth:
            return owner_link, owner_sub, np.full((int(height), int(width)), np.inf, dtype=np.float32)
        return owner_link, owner_sub

    if return_scene_depth:
        id_img, scene_depth = render_software(meshes, colors, camera, resolution, max_faces=max_faces, return_depth=True)
    else:
        id_img = render_software(meshes, colors, camera, resolution, max_faces=max_faces)
        scene_depth = None
    ids = (
        id_img[:, :, 0].astype(np.int32)
        + (id_img[:, :, 1].astype(np.int32) << 8)
        + (id_img[:, :, 2].astype(np.int32) << 16)
    )
    owner_link = np.full((int(height), int(width)), -1, dtype=np.int32)
    owner_sub = np.full((int(height), int(width)), -1, dtype=np.int32)
    link_to_idx = {str(ln): i for i, ln in enumerate(link_names)}
    for code, (link_name, sub_idx) in code_to_meta.items():
        mask = ids == int(code)
        if not np.any(mask):
            continue
        owner_link[mask] = int(link_to_idx.get(str(link_name), -1))
        owner_sub[mask] = int(sub_idx)
    if return_scene_depth:
        return owner_link, owner_sub, np.asarray(scene_depth, dtype=np.float32)
    return owner_link, owner_sub


_FONT_5X7 = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "I": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
}


def draw_text(image, x, y, text, scale=2, color=(0, 0, 0), bg=(220, 220, 220)):
    height, width, _ = image.shape
    text = text.upper()
    char_w = 5 * scale
    char_h = 7 * scale
    spacing = scale
    text_w = len(text) * char_w + max(0, len(text) - 1) * spacing
    text_h = char_h

    x0 = max(0, int(x - text_w // 2) - 2)
    y0 = max(0, int(y - text_h // 2) - 2)
    x1 = min(width, x0 + text_w + 4)
    y1 = min(height, y0 + text_h + 4)

    if bg is not None:
        image[y0:y1, x0:x1] = bg

    cursor_x = x0 + 2
    cursor_y = y0 + 2
    for ch in text:
        pattern = _FONT_5X7.get(ch)
        if pattern is None:
            cursor_x += char_w + spacing
            continue
        for row_idx, row in enumerate(pattern):
            for col_idx, val in enumerate(row):
                if val == "1":
                    ys = cursor_y + row_idx * scale
                    xs = cursor_x + col_idx * scale
                    if ys + scale <= height and xs + scale <= width:
                        image[ys : ys + scale, xs : xs + scale] = color
        cursor_x += char_w + spacing


def draw_label(image, x, y, text, link_color, scale=2):
    # link_color expected in [0,1] RGBA
    rgb = np.clip(link_color[:3], 0, 1) * 255
    bg = tuple(int(v) for v in rgb)
    # choose text color for contrast
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    fg = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    draw_text(image, x, y, text, scale=scale, color=fg, bg=bg)


def _draw_line(image, x0, y0, x1, y1, color, thickness=2):
    height, width = image.shape[:2]
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    dx = x1 - x0
    dy = y1 - y0
    steps = max(abs(dx), abs(dy), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(round(x0 + dx * t))
        y = int(round(y0 + dy * t))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        x0b = max(0, x - thickness // 2)
        x1b = min(width, x0b + thickness)
        y0b = max(0, y - thickness // 2)
        y1b = min(height, y0b + thickness)
        image[y0b:y1b, x0b:x1b] = color


def _corner_axes_overlay_box(width: int, height: int) -> tuple[int, int, int, int]:
    box_w = min(184, max(138, int(0.23 * float(width))))
    box_h = min(144, max(114, int(0.19 * float(height))))
    x1 = int(width) - 8
    y0 = 10
    x0 = max(8, x1 - box_w)
    y1 = min(int(height) - 8, y0 + box_h)
    return (x0, y0, x1, y1)


def _draw_corner_axes_overlay_latest(image, camera, resolution) -> None:
    if image is None or camera is None:
        return
    from PIL import Image, ImageDraw

    h, w = image.shape[:2]
    box = _corner_axes_overlay_box(w, h)
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
    pose = camera_pose_from_lookat(np.asarray(eye, dtype=float), np.asarray(target, dtype=float), np.asarray(up, dtype=float))
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
    for label, axis, color in basis:
        vx = float(np.dot(axis, right_vec))
        vy = -float(np.dot(axis, up_vec))
        vec2 = np.asarray([vx, vy], dtype=float)
        norm2 = float(np.linalg.norm(vec2))
        if norm2 <= 0.14:
            continue
        vec2 = vec2 / norm2
        alignment = abs(float(np.dot(axis, forward_vec)))
        axis_rows.append((norm2, alignment, label, vec2, color))
    if not axis_rows:
        image[:] = np.asarray(pil, dtype=np.uint8)
        return
    axis_rows.sort(key=lambda row: (-row[0], row[2]))
    filtered_rows = []
    min_separation_cos = math.cos(math.radians(14.0))
    for row in axis_rows:
        vec2 = row[3]
        if any(float(np.dot(vec2, kept[3])) >= min_separation_cos for kept in filtered_rows):
            continue
        filtered_rows.append(row)
    filtered_rows.sort(key=lambda row: (row[1], row[2]))

    box_w = float(box[2] - box[0])
    box_h = float(box[3] - box[1])
    inset_origin = np.asarray([box[0] + 0.37 * box_w, box[1] + 0.56 * box_h], dtype=float)
    scale = float(min(0.30 * box_w, 0.34 * box_h))
    ox, oy = float(inset_origin[0]), float(inset_origin[1])
    draw.ellipse((ox - 5, oy - 5, ox + 5, oy + 5), fill=(80, 80, 80))
    image[:] = np.asarray(pil, dtype=np.uint8)
    for _proj_norm, _align, label, vec2, color in filtered_rows:
        tip = inset_origin + vec2 * scale
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil)
        _draw_arrow_segment(inset_origin, tip, color, width_main=5, width_bg=7)
        image[:] = np.asarray(pil, dtype=np.uint8)
        label_dx = 10 if tip[0] >= inset_origin[0] else -14
        label_dy = -10 if tip[1] <= inset_origin[1] else 6
        draw_text(image, float(tip[0] + label_dx), float(tip[1] + label_dy), label, scale=3, color=color, bg=None)


def draw_axes_overlay(image, camera, resolution, origin, axis_len, scale=2, corner=False):
    if axis_len <= 0:
        return
    if corner:
        _draw_corner_axes_overlay_latest(image, camera, resolution)
        return
    origin = np.asarray(origin, dtype=float)
    points = np.stack(
        [
            origin,
            origin + np.array([axis_len, 0.0, 0.0], dtype=float),
            origin + np.array([0.0, axis_len, 0.0], dtype=float),
            origin + np.array([0.0, 0.0, axis_len], dtype=float),
        ],
        axis=0,
    )
    proj = project_points(points, camera, resolution)
    # z > 0 indicates in front of camera
    if proj.shape[0] != 4:
        return
    o, px, py, pz = proj
    # axis colors
    col_x = np.array([220, 40, 40], dtype=np.uint8)
    col_y = np.array([40, 180, 40], dtype=np.uint8)
    col_z = np.array([40, 80, 220], dtype=np.uint8)
    if o[2] <= 0:
        return

    if px[2] > 0:
        _draw_line(image, o[0], o[1], px[0], px[1], col_x, thickness=2)
        draw_text(image, px[0], px[1], "+X", scale=scale, color=(220, 40, 40), bg=None)
    if py[2] > 0:
        _draw_line(image, o[0], o[1], py[0], py[1], col_y, thickness=2)
        draw_text(image, py[0], py[1], "+Y", scale=scale, color=(40, 180, 40), bg=None)

    if pz[2] > 0:
        _draw_line(image, o[0], o[1], pz[0], pz[1], col_z, thickness=2)
        draw_text(image, pz[0], pz[1], "+Z", scale=scale, color=(40, 80, 220), bg=None)


def _label_box(x, y, text, scale, width, height):
    char_w = 5 * scale
    char_h = 7 * scale
    spacing = scale
    text_w = len(text) * char_w + max(0, len(text) - 1) * spacing
    text_h = char_h

    # center-based box, with padding matching draw_text
    x0 = int(x - text_w // 2) - 2
    y0 = int(y - text_h // 2) - 2
    x1 = x0 + text_w + 4
    y1 = y0 + text_h + 4

    # clamp inside image by shifting center
    dx = 0
    dy = 0
    if x0 < 0:
        dx = -x0
    elif x1 > width:
        dx = width - x1
    if y0 < 0:
        dy = -y0
    elif y1 > height:
        dy = height - y1

    if dx != 0 or dy != 0:
        x += dx
        y += dy
        x0 = int(x - text_w // 2) - 2
        y0 = int(y - text_h // 2) - 2
        x1 = x0 + text_w + 4
        y1 = y0 + text_h + 4

    return x, y, (x0, y0, x1, y1)


def _boxes_overlap(a, b, pad=2):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 + pad <= bx0 or bx1 + pad <= ax0 or ay1 + pad <= by0 or by1 + pad <= ay0)


def adjust_label_positions(label_positions, label_texts, resolution, scale=2):
    width, height = resolution
    items = list(label_positions.items())
    # stable ordering for deterministic placement
    items.sort(key=lambda kv: (kv[1][1], kv[1][0], kv[0]))

    # candidate offsets (spiral-ish grid)
    step = 10 * scale
    offsets = [(0, 0)]
    for r in range(1, 4):
        for dx in (-r, 0, r):
            for dy in (-r, 0, r):
                if dx == 0 and dy == 0:
                    continue
                offsets.append((dx * step, dy * step))

    placed = {}
    occupied = []
    for link_name, (x0, y0) in items:
        text = label_texts.get(link_name, "")
        best = None
        best_overlap = None
        for dx, dy in offsets:
            x = x0 + dx
            y = y0 + dy
            x, y, box = _label_box(x, y, text, scale, width, height)
            overlap = 0
            for ob in occupied:
                if _boxes_overlap(box, ob):
                    # approximate overlap score
                    overlap += 1
            if overlap == 0:
                best = (x, y, box)
                break
            if best_overlap is None or overlap < best_overlap:
                best_overlap = overlap
                best = (x, y, box)
        if best is None:
            best = (x0, y0, _label_box(x0, y0, text, scale, width, height)[2])
        placed[link_name] = (best[0], best[1])
        occupied.append(best[2])

    return placed


def project_link_boxes(points_by_link, camera, resolution):
    width, height = resolution
    boxes = {}
    for link_name, points in points_by_link.items():
        if points.shape[0] == 0:
            continue
        proj = project_points(points, camera, resolution)
        xs = proj[:, 0]
        ys = proj[:, 1]
        zs = proj[:, 2]
        mask = (zs > 0) & np.isfinite(xs) & np.isfinite(ys)
        if not np.any(mask):
            continue
        xs = xs[mask]
        ys = ys[mask]
        x0 = int(np.floor(xs.min()))
        y0 = int(np.floor(ys.min()))
        x1 = int(np.ceil(xs.max()))
        y1 = int(np.ceil(ys.max()))
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes[link_name] = expand_bbox((x0, y0, x1, y1), resolution)
    return boxes


def project_visible_link_boxes(points_by_link, camera, resolution, return_stats: bool = False):
    """
    Project per-link points and compute 2D bboxes from *visible* pixels only.
    Visibility is approximated with a point z-buffer across all links.
    """
    width, height = resolution
    link_names = list(points_by_link.keys())
    if not link_names:
        return ({}, {}) if return_stats else {}

    torch_raster = tacc.rasterize_points_torch(
        points_by_link,
        colors_by_link=None,
        camera=camera,
        resolution=resolution,
        point_size=1,
    )
    if torch_raster is not None:
        owner_buffer = np.asarray(torch_raster["owner"], dtype=np.int32)
        projected_counts = {
            str(k): int(v)
            for k, v in (torch_raster.get("projected_counts") or {}).items()
        }
        boxes = {}
        stats = {}
        for li, link_name in enumerate(link_names):
            ys, xs = np.where(owner_buffer == int(li))
            if xs.size == 0:
                continue
            trim = max(0.0, min(49.0, float(VISIBLE_BBOX_TRIM_PERCENT)))
            if trim > 0.0 and xs.size >= 50:
                x0 = int(np.floor(np.percentile(xs, trim)))
                y0 = int(np.floor(np.percentile(ys, trim)))
                x1 = int(np.ceil(np.percentile(xs, 100.0 - trim)))
                y1 = int(np.ceil(np.percentile(ys, 100.0 - trim)))
            else:
                x0 = int(xs.min())
                y0 = int(ys.min())
                x1 = int(xs.max())
                y1 = int(ys.max())
            x0 = max(0, min(width - 1, x0))
            y0 = max(0, min(height - 1, y0))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            boxes[link_name] = expand_bbox((x0, y0, x1, y1), resolution)
            visible_count = int(xs.size)
            projected_count = int(projected_counts.get(str(link_name), 0))
            stats[str(link_name)] = {
                "visible_point_count": visible_count,
                "projected_point_count": projected_count,
                "visible_ratio": float(visible_count) / float(max(1, projected_count)),
            }
        return (boxes, stats) if return_stats else boxes

    all_x = []
    all_y = []
    all_z = []
    all_l = []
    projected_counts = {}
    for li, link_name in enumerate(link_names):
        points = points_by_link.get(link_name)
        if points is None or points.shape[0] == 0:
            continue
        proj = project_points(points, camera, resolution)
        xs = np.round(proj[:, 0]).astype(np.int32)
        ys = np.round(proj[:, 1]).astype(np.int32)
        zs = proj[:, 2]
        mask = (zs > 0) & np.isfinite(zs) & (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        projected_counts[str(link_name)] = int(np.count_nonzero(mask))
        if not np.any(mask):
            continue
        all_x.append(xs[mask])
        all_y.append(ys[mask])
        all_z.append(np.asarray(zs[mask], dtype=np.float32))
        all_l.append(np.full(int(np.sum(mask)), li, dtype=np.int32))

    if not all_x:
        return ({}, {}) if return_stats else {}

    xs = np.concatenate(all_x, axis=0)
    ys = np.concatenate(all_y, axis=0)
    zs = np.concatenate(all_z, axis=0)
    ls = np.concatenate(all_l, axis=0)
    flat = ys.astype(np.int64) * int(width) + xs.astype(np.int64)

    # Per-pixel nearest point by depth.
    order = np.argsort(zs, kind="mergesort")
    flat_sorted = flat[order]
    keep = np.ones(flat_sorted.shape[0], dtype=bool)
    if flat_sorted.shape[0] > 1:
        keep[1:] = flat_sorted[1:] != flat_sorted[:-1]
    sel = order[keep]

    vis_flat = flat[sel]
    vis_link = ls[sel]
    boxes = {}
    stats = {}
    for li, link_name in enumerate(link_names):
        m = vis_link == li
        if not np.any(m):
            continue
        f = vis_flat[m]
        x = (f % int(width)).astype(np.int32)
        y = (f // int(width)).astype(np.int32)
        trim = max(0.0, min(49.0, float(VISIBLE_BBOX_TRIM_PERCENT)))
        if trim > 0.0 and x.size >= 50:
            x0 = int(np.floor(np.percentile(x, trim)))
            y0 = int(np.floor(np.percentile(y, trim)))
            x1 = int(np.ceil(np.percentile(x, 100.0 - trim)))
            y1 = int(np.ceil(np.percentile(y, 100.0 - trim)))
        else:
            x0 = int(x.min())
            y0 = int(y.min())
            x1 = int(x.max())
            y1 = int(y.max())
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes[link_name] = expand_bbox((x0, y0, x1, y1), resolution)
        visible_count = int(np.count_nonzero(m))
        projected_count = int(projected_counts.get(str(link_name), 0))
        stats[str(link_name)] = {
            "visible_point_count": visible_count,
            "projected_point_count": projected_count,
            "visible_ratio": float(visible_count) / float(max(1, projected_count)),
        }
    return (boxes, stats) if return_stats else boxes


def _stack_bbox_points(points_by_link, link_names):
    pts = []
    for ln in link_names:
        p = points_by_link.get(ln)
        if p is None:
            continue
        p = np.asarray(p, dtype=np.float32)
        if p.size == 0:
            continue
        pts.append(p)
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def _movable_links_from_joints(joints):
    out = set()
    for j in joints or []:
        jt = str(j.get("type") or "").strip().lower()
        if jt in {"", "fixed"}:
            continue
        c = j.get("child")
        if c:
            out.add(str(c))
    return out


def _child_to_joint_map(joints):
    # Prefer non-fixed joints when multiple joints mention the same child link.
    best = {}
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


def _compact_image_label(raw):
    s = str(raw or "").strip()
    if s.startswith("joint_"):
        tail = s[len("joint_") :].strip()
        return tail if tail else s
    if s.startswith("link_"):
        tail = s[len("link_") :].strip()
        return tail if tail else s
    return s


def _digits_only_compact_label(link_name, fallback_index):
    nums = re.findall(r"\d+", str(link_name or ""))
    if nums:
        v = nums[-1].lstrip("0")
        return v if v else "0"
    return str(int(fallback_index) + 1)


def build_compact_image_labels(visual_links, joints):
    labels = {}
    used = set()
    next_auto = 1
    for i, ln in enumerate(visual_links):
        lab = _digits_only_compact_label(ln, i)
        if lab in used:
            while str(next_auto) in used:
                next_auto += 1
            lab = str(next_auto)
            next_auto += 1
        labels[ln] = lab
        used.add(lab)
    return labels


def _select_primary_static_link(visual_links, movable_links, link_meshes, joints):
    candidates = [ln for ln in visual_links if ln not in movable_links]
    if not candidates:
        return visual_links[0] if visual_links else None

    # Prefer root-like static links first.
    child_links = {str(j.get("child")) for j in (joints or []) if j.get("child")}
    roots = [ln for ln in candidates if ln not in child_links]
    if roots:
        candidates = roots

    def _volume(ln):
        meshes = link_meshes.get(ln, []) if isinstance(link_meshes, dict) else []
        if not meshes:
            return 0.0
        try:
            merged = trimesh.util.concatenate([m.copy() for m in meshes if m is not None and m.vertices.size > 0])
            ext = np.asarray(merged.bounding_box.extents, dtype=float)
            return float(max(ext[0], 1e-9) * max(ext[1], 1e-9) * max(ext[2], 1e-9))
        except Exception:
            return 0.0

    return max(candidates, key=_volume)


def _line_like_visual_links(link_meshes, visual_links) -> set[str]:
    out = set()
    for ln in visual_links or []:
        meshes = link_meshes.get(ln, []) if isinstance(link_meshes, dict) else []
        valid = [m.copy() for m in meshes if m is not None and getattr(m, "vertices", None) is not None and m.vertices.size > 0]
        if not valid:
            continue
        try:
            merged = trimesh.util.concatenate(valid) if len(valid) > 1 else valid[0]
            ext = np.sort(np.asarray(merged.bounding_box.extents, dtype=float))[::-1]
        except Exception:
            continue
        if ext.size < 3:
            continue
        if float(ext[0]) <= 1e-9:
            continue
        # Long thin parts like clock hands: dominant principal axis, small secondary axes.
        if float(ext[1]) / float(ext[0]) <= 0.25:
            out.add(str(ln))
    return out


def build_structured_overlay_boxes(
    bbox_points_by_link,
    camera,
    resolution,
    *,
    visual_links,
    movable_visual_links,
    static_big_link,
):
    """
    Build exactly one global big box + one small box per movable link.
    """
    boxes = {}
    movable_pts = {
        ln: np.asarray(bbox_points_by_link.get(ln, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        for ln in movable_visual_links
    }

    # Small boxes: movable links only (old visible-point projection path).
    if movable_pts:
        small, small_stats = project_visible_link_boxes(movable_pts, camera, resolution, return_stats=True)
        for ln in movable_visual_links:
            fb = project_link_boxes({ln: movable_pts.get(ln, np.zeros((0, 3), dtype=np.float32))}, camera, resolution)
            chosen = small.get(ln)
            if chosen is not None:
                stats = small_stats.get(str(ln)) or {}
                visible_ratio = float(stats.get("visible_ratio") or 0.0)
                x0, y0, x1, y1 = [int(v) for v in chosen]
                box_area = max(1, max(0, x1 - x0) * max(0, y1 - y0))
                # Small movable parts are easy to lose in the visible-only path.
                # If the visible box is too tiny or weak, fall back to the full projected box.
                if (
                    visible_ratio >= max(0.05, float(REFERENCE_VISIBLE_RATIO_MIN) * 0.5)
                    and box_area > int(SMALL_PART_PROJECTED_BOX_AREA_MAX)
                ) or ln not in fb:
                    boxes[ln] = chosen
                    continue
            if ln in fb:
                boxes[ln] = fb[ln]

    # Big box: all visible links together.
    all_pts = _stack_bbox_points(bbox_points_by_link, visual_links)
    if all_pts.shape[0] > 0:
        key = "__all__"
        big = project_visible_link_boxes({key: all_pts}, camera, resolution)
        if key not in big:
            big = project_link_boxes({key: all_pts}, camera, resolution)
        if key in big and static_big_link is not None:
            boxes[static_big_link] = big[key]

    # Final fallback: if everything failed, return old behavior on all links.
    if not boxes:
        all_link_boxes = project_visible_link_boxes(
            {ln: np.asarray(bbox_points_by_link.get(ln, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32) for ln in visual_links},
            camera,
            resolution,
        )
        if not all_link_boxes:
            all_link_boxes = project_link_boxes(
                {ln: np.asarray(bbox_points_by_link.get(ln, np.zeros((0, 3), dtype=np.float32)), dtype=np.float32) for ln in visual_links},
                camera,
                resolution,
            )
        boxes = all_link_boxes
    return boxes


def compute_union_box(
    boxes_by_link,
    link_names=None,
    resolution=None,
):
    if not isinstance(boxes_by_link, dict) or not boxes_by_link:
        return None
    if link_names is None:
        keys = list(boxes_by_link.keys())
    else:
        keys = [str(ln) for ln in link_names if str(ln) in boxes_by_link]
    coords = []
    for key in keys:
        box = boxes_by_link.get(key)
        if box is None:
            continue
        try:
            x0, y0, x1, y1 = [int(v) for v in box]
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue
        coords.append((x0, y0, x1, y1))
    if not coords:
        return None
    x0 = min(box[0] for box in coords)
    y0 = min(box[1] for box in coords)
    x1 = max(box[2] for box in coords)
    y1 = max(box[3] for box in coords)
    if resolution is not None:
        width, height = [int(v) for v in resolution]
        x0 = max(0, min(width - 1, x0))
        y0 = max(0, min(height - 1, y0))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (int(x0), int(y0), int(x1), int(y1))


def merge_reference_boxes(
    *,
    raster_boxes,
    raster_stats,
    structured_boxes,
    overlay_boxes,
    visual_links,
    static_big_link,
    resolution,
    surface_only: bool = False,
):
    """
    Prefer rasterized boxes from the same rendered reference geometry.
    Fall back to projected point boxes only when rasterized boxes are missing.
    """
    merged = {}
    aggregate_raster_box = compute_union_box(
        raster_boxes or {},
        visual_links,
        resolution,
    )
    aggregate_structured_box = compute_union_box(
        structured_boxes or {},
        visual_links,
        resolution,
    )
    preserve_static_big = static_big_link is not None and (aggregate_raster_box is not None or aggregate_structured_box is not None)
    if preserve_static_big:
        merged[static_big_link] = aggregate_raster_box if aggregate_raster_box is not None else aggregate_structured_box
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
                raster_box_area <= int(REFERENCE_RASTER_TINY_BOX_AREA_MAX)
                or raster_box_area < float(REFERENCE_RASTER_STRUCTURED_FALLBACK_AREA_RATIO) * float(structured_box_area)
            ):
                chosen_box = structured_box
        elif chosen_box is None:
            chosen_box = structured_box
        if chosen_box is not None:
            merged[str(ln)] = chosen_box
    if surface_only:
        return merged
    if static_big_link is not None and aggregate_raster_box is not None:
        merged[static_big_link] = aggregate_raster_box
    for ln, box in (overlay_boxes or {}).items():
        merged.setdefault(ln, box)
    return merged


def expand_bbox(box, resolution, *, pad_ratio=BBOX_EXPAND_RATIO, min_pad_px=BBOX_EXPAND_MIN_PX, max_pad_px=BBOX_EXPAND_MAX_PX):
    try:
        x0, y0, x1, y1 = [int(v) for v in box]
    except Exception:
        return box
    width, height = [int(v) for v in resolution]
    if x1 <= x0 or y1 <= y0 or width <= 0 or height <= 0:
        return (x0, y0, x1, y1)
    box_w = max(1, x1 - x0)
    box_h = max(1, y1 - y0)
    ratio = max(0.0, float(pad_ratio))
    min_pad = max(0, int(min_pad_px))
    max_pad = max(min_pad, int(max_pad_px))
    pad_x = min(max_pad, max(min_pad, int(round(box_w * ratio))))
    pad_y = min(max_pad, max(min_pad, int(round(box_h * ratio))))
    ex0 = max(0, x0 - pad_x)
    ey0 = max(0, y0 - pad_y)
    ex1 = min(width - 1, x1 + pad_x)
    ey1 = min(height - 1, y1 + pad_y)
    if ex1 <= ex0 or ey1 <= ey0:
        return (x0, y0, x1, y1)
    return (int(ex0), int(ey0), int(ex1), int(ey1))


def project_visible_link_boxes_rasterized(
    link_meshes,
    link_names,
    camera,
    resolution,
    max_faces=REFERENCE_MAX_FACES,
    aggregation_mode_by_link=None,
    return_stats: bool = False,
    return_scene_depth: bool = False,
):
    """
    Compute visible bboxes via rasterized id pass at submesh granularity.
    Then aggregate submesh boxes per link (default: largest visible submesh).
    """
    width, height = resolution

    def _aggregate_from_owner_maps(owner_link, owner_sub, scene_depth_arr):
        per_link_candidates = {str(ln): [] for ln in link_names}
        link_to_idx = {str(ln): i for i, ln in enumerate(link_names)}
        for link_name, li in link_to_idx.items():
            sub_map = owner_sub if owner_sub is not None else np.full_like(owner_link, -1, dtype=np.int32)
            link_mask = owner_link == int(li)
            if not np.any(link_mask):
                continue
            sub_ids = np.unique(sub_map[link_mask])
            sub_ids = [int(x) for x in sub_ids.tolist() if int(x) >= 0]
            if not sub_ids:
                sub_ids = [0]
            for sub_idx in sub_ids:
                mask = link_mask if owner_sub is None else (link_mask & (sub_map == int(sub_idx)))
                ys, xs = np.where(mask)
                if xs.size == 0:
                    continue
                x0 = int(xs.min())
                y0 = int(ys.min())
                x1 = int(xs.max())
                y1 = int(ys.max())
                if x1 <= x0 or y1 <= y0:
                    continue
                per_link_candidates.setdefault(str(link_name), []).append(
                    {
                        "sub_idx": int(sub_idx),
                        "pixels": int(xs.size),
                        "box": (x0, y0, x1, y1),
                    }
                )
        boxes = {}
        stats = {}
        for link_name in link_names:
            cands = per_link_candidates.get(str(link_name)) or []
            if not cands:
                continue
            visible_pixels = int(sum(int(c.get("pixels") or 0) for c in cands))
            mode = str(
                ((aggregation_mode_by_link or {}).get(link_name))
                or (BBOX_LINK_AGGREGATION if BBOX_LINK_AGGREGATION in {"largest_submesh", "union"} else "union")
            ).strip().lower()
            if mode not in {"largest_submesh", "union"}:
                mode = "union"
            if mode == "union":
                x0 = min(c["box"][0] for c in cands)
                y0 = min(c["box"][1] for c in cands)
                x1 = max(c["box"][2] for c in cands)
                y1 = max(c["box"][3] for c in cands)
                raw_box = (int(max(0, x0)), int(max(0, y0)), int(min(width - 1, x1)), int(min(height - 1, y1)))
                boxes[str(link_name)] = expand_bbox(raw_box, resolution)
                box_area = max(1, int(max(0, raw_box[2] - raw_box[0])) * int(max(0, raw_box[3] - raw_box[1])))
                stats[str(link_name)] = {
                    "visible_pixels": visible_pixels,
                    "raw_box": raw_box,
                    "box_area_px": box_area,
                    "visible_fill_ratio": float(visible_pixels) / float(max(1, box_area)),
                }
                continue
            best = max(cands, key=lambda c: (int(c["pixels"]), -int(c["sub_idx"])))
            raw_box = tuple(int(v) for v in best["box"])
            boxes[str(link_name)] = expand_bbox(raw_box, resolution)
            box_area = max(1, int(max(0, raw_box[2] - raw_box[0])) * int(max(0, raw_box[3] - raw_box[1])))
            stats[str(link_name)] = {
                "visible_pixels": int(best.get("pixels") or 0),
                "raw_box": raw_box,
                "box_area_px": box_area,
                "visible_fill_ratio": float(int(best.get("pixels") or 0)) / float(max(1, box_area)),
            }
        if return_stats and return_scene_depth:
            return boxes, stats, scene_depth_arr
        if return_stats:
            return boxes, stats
        if return_scene_depth:
            return boxes, scene_depth_arr
        return boxes
    if return_scene_depth:
        owner_link, owner_sub, scene_depth = rasterize_link_visibility_maps(
            link_meshes,
            link_names,
            camera,
            resolution,
            max_faces=max_faces,
            return_scene_depth=True,
        )
    else:
        owner_link, owner_sub = rasterize_link_visibility_maps(
            link_meshes,
            link_names,
            camera,
            resolution,
            max_faces=max_faces,
            return_scene_depth=False,
        )
        scene_depth = None
    return _aggregate_from_owner_maps(owner_link, owner_sub, scene_depth)


def _render_link_projected_mask(mesh_list, camera, resolution, max_faces=REFERENCE_MAX_FACES, return_depth: bool = False):
    meshes = [m.copy() for m in (mesh_list or []) if getattr(m, "vertices", None) is not None and m.vertices.size > 0 and getattr(m, "faces", None) is not None and m.faces.size > 0]
    if not meshes:
        empty_mask = np.zeros((int(resolution[1]), int(resolution[0])), dtype=bool)
        if return_depth:
            empty_depth = np.full((int(resolution[1]), int(resolution[0])), np.inf, dtype=np.float32)
            return empty_mask, empty_depth
        return empty_mask
    colors = [np.array([0.0, 0.0, 0.0, 1.0], dtype=float) for _ in meshes]
    rendered = render_software(meshes, colors, camera, resolution, max_faces=max_faces, return_depth=return_depth)
    if return_depth:
        img, depth = rendered
    else:
        img = rendered
    arr = np.asarray(img, dtype=np.uint8)
    mask = np.any(arr < 250, axis=2)
    if return_depth:
        return mask, np.asarray(depth, dtype=np.float32)
    return mask


def reference_visible_ratios_by_link_rasterized(
    link_meshes,
    link_names,
    camera,
    resolution,
    visible_pixel_stats: dict[str, dict] | None = None,
    points_by_link: dict[str, np.ndarray] | None = None,
    max_faces=REFERENCE_MAX_FACES,
    scene_depth: np.ndarray | None = None,
):
    # Fast path: if we already have a scene depth pass and per-link sampled points,
    # estimate visibility from projected surface points directly instead of
    # re-rasterizing every link one-by-one.
    if scene_depth is not None and isinstance(points_by_link, dict):
        scene_depth = np.asarray(scene_depth, dtype=np.float32)
        h, w = scene_depth.shape
        stats = {}
        for link_name in link_names or []:
            pts = np.asarray(points_by_link.get(str(link_name), np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            vis = 0
            if isinstance(visible_pixel_stats, dict):
                vis = int((visible_pixel_stats.get(str(link_name)) or {}).get("visible_pixels") or 0)
            if pts.ndim != 2 or pts.shape[0] == 0:
                stats[str(link_name)] = {
                    "visible_pixels": int(vis),
                    "total_projected_pixels": 0,
                    "tie_surface_visible_pixels": 0,
                    "surface_visible_ratio": 0.0,
                    "visible_ratio": 0.0,
                }
                continue
            proj = project_points(pts, camera, resolution)
            xs = np.round(proj[:, 0]).astype(np.int32)
            ys = np.round(proj[:, 1]).astype(np.int32)
            zs = np.asarray(proj[:, 2], dtype=np.float32)
            keep = (zs > 0) & np.isfinite(zs) & (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
            if not np.any(keep):
                stats[str(link_name)] = {
                    "visible_pixels": int(vis),
                    "total_projected_pixels": 0,
                    "tie_surface_visible_pixels": 0,
                    "surface_visible_ratio": 0.0,
                    "visible_ratio": 0.0,
                }
                continue
            xs = xs[keep]
            ys = ys[keep]
            zs = zs[keep]
            scene_vals = np.asarray(scene_depth[ys, xs], dtype=np.float32)
            eps = np.maximum(1e-3, float(REFERENCE_SURFACE_DEPTH_REL_EPS) * np.maximum(1.0, scene_vals))
            near = np.isfinite(scene_vals) & (zs <= scene_vals + eps)
            total = int(zs.shape[0])
            surface_visible_pixels = int(np.count_nonzero(near))
            surface_visible_ratio = float(surface_visible_pixels) / float(max(1, total))
            visible_ratio = float(surface_visible_ratio)
            stats[str(link_name)] = {
                "visible_pixels": int(max(vis, surface_visible_pixels)),
                "total_projected_pixels": int(total),
                "tie_surface_visible_pixels": 0,
                "surface_visible_ratio": float(surface_visible_ratio),
                "visible_ratio": float(visible_ratio),
            }
        return stats

    if scene_depth is None:
        all_meshes = []
        all_colors = []
        for link_name in link_names or []:
            for mesh in (link_meshes.get(link_name) or []) if isinstance(link_meshes, dict) else []:
                if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
                    continue
                if getattr(mesh, "faces", None) is None or mesh.faces.size == 0:
                    continue
                all_meshes.append(mesh.copy())
                all_colors.append(np.array([0.0, 0.0, 0.0, 1.0], dtype=float))
        if all_meshes:
            _img_all, scene_depth = render_software(
                all_meshes,
                all_colors,
                camera,
                resolution,
                max_faces=max_faces,
                return_depth=True,
            )
            scene_depth = np.asarray(scene_depth, dtype=np.float32)
    elif scene_depth is not None:
        scene_depth = np.asarray(scene_depth, dtype=np.float32)

    stats = {}
    for link_name in link_names or []:
        vis = 0
        if isinstance(visible_pixel_stats, dict):
            vis = int((visible_pixel_stats.get(str(link_name)) or {}).get("visible_pixels") or 0)
        mask, link_depth = _render_link_projected_mask(
            (link_meshes.get(link_name) or []) if isinstance(link_meshes, dict) else [],
            camera,
            resolution,
            max_faces=max_faces,
            return_depth=True,
        )
        total = int(np.count_nonzero(mask))
        tie_surface_visible_pixels = 0
        surface_visible_ratio = 0.0
        if scene_depth is not None and total > 0 and vis <= 0:
            valid = mask & np.isfinite(link_depth) & np.isfinite(scene_depth)
            if np.any(valid):
                scene_vals = np.asarray(scene_depth[valid], dtype=np.float32)
                link_vals = np.asarray(link_depth[valid], dtype=np.float32)
                eps = np.maximum(1e-3, 2e-3 * np.maximum(1.0, scene_vals))
                near_surface = np.abs(link_vals - scene_vals) <= eps
                tie_surface_visible_pixels = int(np.count_nonzero(near_surface))
                if tie_surface_visible_pixels > 0:
                    vis = max(int(vis), int(tie_surface_visible_pixels))
        if scene_depth is not None and isinstance(points_by_link, dict):
            pts = np.asarray(points_by_link.get(str(link_name), np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            if pts.ndim == 2 and pts.shape[0] > 0:
                proj = project_points(pts, camera, resolution)
                xs = np.round(proj[:, 0]).astype(np.int32)
                ys = np.round(proj[:, 1]).astype(np.int32)
                zs = np.asarray(proj[:, 2], dtype=np.float32)
                h, w = scene_depth.shape
                keep = (zs > 0) & np.isfinite(zs) & (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
                if np.any(keep):
                    xs = xs[keep]
                    ys = ys[keep]
                    zs = zs[keep]
                    scene_vals = np.asarray(scene_depth[ys, xs], dtype=np.float32)
                    eps = np.maximum(1e-3, float(REFERENCE_SURFACE_DEPTH_REL_EPS) * np.maximum(1.0, scene_vals))
                    near = np.isfinite(scene_vals) & (zs <= scene_vals + eps)
                    surface_visible_ratio = float(np.count_nonzero(near)) / float(max(1, zs.shape[0]))
        raster_visible_ratio = (float(vis) / float(max(1, total)) if total > 0 else 0.0)
        visible_ratio = max(raster_visible_ratio, float(surface_visible_ratio))
        if total > 0 and surface_visible_ratio > raster_visible_ratio:
            vis = max(int(vis), int(round(float(surface_visible_ratio) * float(total))))
        stats[str(link_name)] = {
            "visible_pixels": int(vis),
            "total_projected_pixels": int(total),
            "tie_surface_visible_pixels": int(tie_surface_visible_pixels),
            "surface_visible_ratio": float(surface_visible_ratio),
            "visible_ratio": float(visible_ratio),
        }
    return stats


def draw_bbox_outline(image, box, link_color, thickness=2):
    rgb = (np.clip(np.asarray(link_color[:3]), 0, 1) * 255).astype(np.uint8)
    x0, y0, x1, y1 = [int(v) for v in box]
    _draw_line(image, x0, y0, x1, y0, rgb, thickness=thickness)
    _draw_line(image, x1, y0, x1, y1, rgb, thickness=thickness)
    _draw_line(image, x1, y1, x0, y1, rgb, thickness=thickness)
    _draw_line(image, x0, y1, x0, y0, rgb, thickness=thickness)


def _small_box_has_visible_signal(image: np.ndarray, box) -> bool:
    try:
        x0, y0, x1, y1 = [int(v) for v in box]
    except Exception:
        return False
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    h, w = arr.shape[:2]
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    if x1 <= x0 or y1 <= y0:
        return False
    patch = arr[y0:y1, x0:x1, :3].astype(np.float32)
    if patch.size == 0:
        return False
    gray = patch.mean(axis=2)
    patch_std = float(np.std(gray))
    pad = max(6, int(round(0.35 * float(max(x1 - x0, y1 - y0)))))
    xa = max(0, x0 - pad)
    ya = max(0, y0 - pad)
    xb = min(w, x1 + pad)
    yb = min(h, y1 + pad)
    outer = arr[ya:yb, xa:xb, :3].astype(np.float32).mean(axis=2)
    if outer.size == 0:
        return patch_std >= float(SMALL_PART_VISUAL_STD_MIN)
    mask = np.ones_like(outer, dtype=bool)
    mask[(y0 - ya) : (y1 - ya), (x0 - xa) : (x1 - xa)] = False
    ring = outer[mask]
    ring_mean = float(np.mean(ring)) if ring.size > 0 else float(np.mean(gray))
    patch_mean = float(np.mean(gray))
    contrast = abs(patch_mean - ring_mean)
    return patch_std >= float(SMALL_PART_VISUAL_STD_MIN) or contrast >= float(SMALL_PART_VISUAL_CONTRAST_MIN)


def filter_reference_boxes_by_visibility(
    ref_boxes: dict,
    reference_visibility_ratio_stats: dict,
    movable_visual_links,
    line_like_visual_links,
    ref_img: np.ndarray,
):
    movable_set = {str(x) for x in (movable_visual_links or [])}
    line_like_set = {str(x) for x in (line_like_visual_links or [])}
    filtered_ref_boxes = {}
    for link_name, box in (ref_boxes or {}).items():
        stats = (reference_visibility_ratio_stats or {}).get(str(link_name)) or {}
        raster_visible_ratio_all = float(stats.get("visible_ratio") or 0.0)
        is_movable = str(link_name) in movable_set
        keep_box = raster_visible_ratio_all >= float(MOVABLE_RASTER_VISIBLE_RATIO_MIN)
        if is_movable:
            try:
                x0, y0, x1, y1 = [int(v) for v in box]
                box_area = max(1, max(0, x1 - x0) * max(0, y1 - y0))
            except Exception:
                box_area = 0
            if (
                not keep_box
                and str(link_name) in line_like_set
                and box_area <= int(max(SMALL_PART_PROJECTED_BOX_AREA_MAX, 15000))
            ):
                keep_box = _small_box_has_visible_signal(ref_img, box)
        if keep_box:
            filtered_ref_boxes[str(link_name)] = box
    return filtered_ref_boxes


def _scaled_visibility_resolution(resolution) -> tuple[int, int]:
    try:
        w = int(resolution[0])
        h = int(resolution[1])
    except Exception:
        return (800, 600)
    scale = float(PREPROCESS_REFERENCE_VIS_SCALE)
    if not np.isfinite(scale):
        scale = 1.0
    scale = max(0.25, min(1.0, scale))
    if scale >= 0.999:
        return (w, h)
    return (max(160, int(round(w * scale))), max(120, int(round(h * scale))))


def adjust_caption_positions_from_boxes(boxes_by_link, label_texts, resolution, scale=2):
    width, height = resolution
    items = sorted(boxes_by_link.items(), key=lambda kv: (kv[1][1], kv[1][0], kv[0]))
    placed = {}
    occupied = []
    pad = 6 * scale
    for link_name, (x0, y0, x1, y1) in items:
        text = label_texts.get(link_name, "")
        # Candidate label centers around box corners (outside first, then inside).
        candidates = [
            (x0, y0 - pad),
            (x1, y0 - pad),
            (x0, y1 + pad),
            (x1, y1 + pad),
            (x0 - pad, y0),
            (x1 + pad, y0),
            (x0 - pad, y1),
            (x1 + pad, y1),
            ((x0 + x1) / 2.0, y0 - pad),
            ((x0 + x1) / 2.0, y1 + pad),
            ((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        ]
        best = None
        best_score = None
        for cx, cy in candidates:
            cx, cy, cap_box = _label_box(cx, cy, text, scale, width, height)
            overlap = 0
            for ob in occupied:
                if _boxes_overlap(cap_box, ob, pad=2):
                    overlap += 1
            # Prefer captions farther from the part center only if overlap ties.
            center_penalty = abs(cx - (x0 + x1) * 0.5) + abs(cy - (y0 + y1) * 0.5)
            score = (overlap, center_penalty)
            if best_score is None or score < best_score:
                best_score = score
                best = (cx, cy, cap_box)
            if overlap == 0:
                break
        if best is None:
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            cx, cy, cap_box = _label_box(cx, cy, text, scale, width, height)
            best = (cx, cy, cap_box)
        placed[link_name] = (best[0], best[1])
        occupied.append(best[2])
    return placed


def sample_link_points(link_meshes, link_names):
    points_by_link = {}
    for link_name in link_names:
        meshes = link_meshes.get(link_name, [])
        if not meshes:
            points_by_link[link_name] = np.zeros((0, 3), dtype=np.float32)
            continue
        merged = trimesh.util.concatenate([m for m in meshes if m.vertices.size > 0])
        if merged.vertices.size == 0:
            points_by_link[link_name] = np.zeros((0, 3), dtype=np.float32)
            continue
        seed = int(hashlib.md5(link_name.encode("utf-8")).hexdigest()[:8], 16) & 0xFFFFFFFF
        np.random.seed(seed)
        try:
            pts = merged.sample(POINTS_PER_LINK)
        except Exception:
            pts, _ = trimesh.sample.sample_surface(merged, POINTS_PER_LINK)
        points_by_link[link_name] = np.asarray(pts, dtype=np.float32)
    return points_by_link


def _downsample_points_deterministic(points: np.ndarray, max_points: int, seed_name: str):
    pts = np.asarray(points, dtype=np.float32)
    if int(max_points) <= 0:
        return pts
    if pts.shape[0] <= int(max_points):
        return pts
    seed = int(hashlib.md5(str(seed_name).encode("utf-8")).hexdigest()[:8], 16) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
    return pts[idx]


def _surface_sample_mesh_deterministic(mesh, n: int, seed_name: str):
    if mesh is None or mesh.vertices.size == 0 or int(n) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if getattr(mesh, "faces", None) is None or mesh.faces.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    seed = int(hashlib.md5(str(seed_name).encode("utf-8")).hexdigest()[:8], 16) & 0xFFFFFFFF
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        try:
            pts = mesh.sample(int(n))
        except Exception:
            pts, _ = trimesh.sample.sample_surface(mesh, int(n))
    finally:
        np.random.set_state(state)
    return np.asarray(pts, dtype=np.float32)


def _mesh_points_for_bbox(mesh, seed_name: str, max_points=BBOX_POINTS_PER_LINK):
    if mesh is None or mesh.vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    samples = _surface_sample_mesh_deterministic(mesh, int(BBOX_SURFACE_POINTS_PER_LINK), f"surf:{seed_name}")
    if samples.shape[0] > 0:
        pts = np.concatenate([verts, samples], axis=0)
    else:
        pts = verts
    if int(max_points) > 0:
        pts = _downsample_points_deterministic(pts, int(max_points), f"down:{seed_name}")
    return pts


def collect_link_bbox_points(link_meshes, link_names, max_points=BBOX_POINTS_PER_LINK):
    """
    Build high-density per-link points for precise 2D bbox projection.
    Use vertices + deterministic surface samples to avoid sparse-corner bias.
    """
    points_by_link = {}
    for link_name in link_names:
        meshes = link_meshes.get(link_name, []) if isinstance(link_meshes, dict) else []
        valid = [m.copy() for m in meshes if m is not None and getattr(m, "vertices", None) is not None and m.vertices.size > 0]
        if not valid:
            points_by_link[link_name] = np.zeros((0, 3), dtype=np.float32)
            continue
        merged = trimesh.util.concatenate(valid) if len(valid) > 1 else valid[0]
        pts = _mesh_points_for_bbox(merged, f"bbox:{link_name}", max_points=max_points)
        points_by_link[link_name] = pts
    return points_by_link


def _sample_mesh_points_for_label(mesh, seed_name, n=POINTS_PER_LINK):
    if mesh is None or mesh.vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    seed = int(hashlib.md5(str(seed_name).encode("utf-8")).hexdigest()[:8], 16) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        try:
            pts = mesh.sample(n)
        except Exception:
            pts, _ = trimesh.sample.sample_surface(mesh, n)
    finally:
        np.random.set_state(state)
    # Add tiny deterministic shuffle via local rng to avoid identical wheel tie layouts.
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] > 1:
        order = rng.permutation(pts.shape[0])
        pts = pts[order]
    return pts


def _scene_node_world_meshes(scene):
    out = []
    if scene is None:
        return out
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        mesh = geom.copy()
        mesh.apply_transform(transform)
        out.append((str(node_name), mesh))
    return out


def _shape_feature_from_mesh(mesh):
    if mesh is None or mesh.vertices.size == 0:
        return None
    ext = np.asarray(mesh.bounding_box.extents, dtype=np.float32)
    m = float(np.max(ext))
    if m <= 1e-9:
        return None
    s = np.sort(ext / m)
    return s.astype(np.float32)


def _extract_part_node_index(node_name):
    if not isinstance(node_name, str) or not node_name.startswith("part_node_"):
        return None
    m = re.match(r"^part_node_(\d+)(?:$|[^0-9].*)", node_name)
    if not m:
        return None
    return int(m.group(1))


def build_reference_meshes_by_link(scene, link_names):
    """
    Map a reference scene back to URDF visual links using grouped part_node_<i> ordering.
    Supports scenes where one logical part is split across many sub-nodes such as
    part_node_3__sub_01, part_node_3__sub_02, ...
    """
    nodes = _scene_node_world_meshes(scene)
    if not nodes:
        return None
    part_groups: dict[int, list[trimesh.Trimesh]] = {}
    for nn, mesh in nodes:
        idx = _extract_part_node_index(nn)
        if idx is None:
            return None
        if mesh is None or getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
            continue
        part_groups.setdefault(int(idx), []).append(mesh.copy())
    if not part_groups:
        return None
    got = sorted(part_groups.keys())
    expected = list(range(len(got)))
    if got != expected:
        return None
    if len(part_groups) != len(link_names):
        return None

    out = {}
    for i, ln in enumerate(link_names):
        out[str(ln)] = [m.copy() for m in (part_groups.get(i) or [])]
    return out


def _build_reference_points_by_part_order(scene, link_names):
    """
    Deterministic mapping for Particulate-style GLBs:
    part_node_0..N-1 follow URDF visual link order.
    """
    mapped = build_reference_meshes_by_link(scene, link_names)
    if not isinstance(mapped, dict):
        return None

    out = {}
    for ln in link_names:
        meshes = [m.copy() for m in (mapped.get(str(ln)) or []) if getattr(m, "vertices", None) is not None and m.vertices.size > 0]
        if not meshes:
            out[str(ln)] = np.zeros((0, 3), dtype=np.float32)
            continue
        merged = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        out[str(ln)] = _mesh_points_for_bbox(merged, f"reforder:{ln}", max_points=BBOX_POINTS_PER_LINK)
    return out


def build_reference_points_by_link(scene, link_meshes, link_names):
    """
    Build label anchor points from the same reference scene geometry to avoid
    label drift when reference GLB uses a different global orientation than URDF.
    Returns None on failure so caller can fall back to URDF-based points.
    """
    # First try deterministic part_node ordering (most stable for repeated shapes).
    by_order = _build_reference_points_by_part_order(scene, link_names)
    if by_order is not None:
        return by_order

    nodes = _scene_node_world_meshes(scene)
    if not nodes:
        return None
    link_mesh_objs = []
    for ln in link_names:
        meshes = link_meshes.get(ln, [])
        if not meshes:
            continue
        try:
            merged = trimesh.util.concatenate([m.copy() for m in meshes if m.vertices.size > 0])
        except Exception:
            continue
        if merged.vertices.size == 0:
            continue
        link_mesh_objs.append((ln, merged))
    if len(link_mesh_objs) != len(nodes):
        return None

    link_feats = []
    for ln, m in link_mesh_objs:
        f = _shape_feature_from_mesh(m)
        if f is None:
            return None
        link_feats.append((ln, m, f))
    node_feats = []
    for nn, m in nodes:
        f = _shape_feature_from_mesh(m)
        if f is None:
            return None
        node_feats.append((nn, m, f))

    # Greedy one-to-one by normalized bbox shape. This is sufficient for clock-like assets
    # where the global orientation may differ but each part has distinct aspect ratios.
    candidates = []
    for li, (ln, _lm, lf) in enumerate(link_feats):
        for ni, (nn, _nm, nf) in enumerate(node_feats):
            cost = float(np.linalg.norm(lf - nf))
            candidates.append((cost, li, ni))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    used_l = set()
    used_n = set()
    mapping = {}
    for cost, li, ni in candidates:
        if li in used_l or ni in used_n:
            continue
        used_l.add(li)
        used_n.add(ni)
        mapping[link_feats[li][0]] = node_feats[ni][1]
    if len(mapping) != len(link_feats):
        return None

    out = {}
    for ln in link_names:
        mesh = mapping.get(ln)
        if mesh is None:
            out[ln] = np.zeros((0, 3), dtype=np.float32)
            continue
        out[ln] = _mesh_points_for_bbox(mesh, f"refbbox:{ln}", max_points=BBOX_POINTS_PER_LINK)
    return out


def render_overlay_points(points_by_link, colors_by_link, camera, resolution, return_visible_pixels=False):
    width, height = resolution
    link_names = list(points_by_link.keys())
    link_to_idx = {ln: i for i, ln in enumerate(link_names)}

    torch_raster = tacc.rasterize_points_torch(
        points_by_link,
        colors_by_link=colors_by_link,
        camera=camera,
        resolution=resolution,
        point_size=POINT_SIZE,
    )
    if torch_raster is not None:
        color_buffer = np.asarray(torch_raster["image"], dtype=np.uint8)
        owner_buffer = np.asarray(torch_raster["owner"], dtype=np.int32)
        label_positions = {}
        visible_pixels_by_link = {}
        for ln, li in link_to_idx.items():
            ys, xs = np.where(owner_buffer == int(li))
            if xs.size == 0:
                visible_pixels_by_link[ln] = np.zeros((0, 2), dtype=np.int32)
                continue
            y_cut = np.percentile(ys, 45.0)
            top_mask = ys <= y_cut
            if np.any(top_mask):
                xa = xs[top_mask]
                ya = ys[top_mask]
            else:
                xa = xs
                ya = ys
            label_positions[ln] = (float(np.median(xa)), float(np.median(ya)))
            visible_pixels_by_link[ln] = np.column_stack([xs.astype(np.int32), ys.astype(np.int32)])
        if return_visible_pixels:
            return color_buffer, label_positions, visible_pixels_by_link
        return color_buffer, label_positions

    color_buffer = np.full((height, width, 3), 255, dtype=np.uint8)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    owner_buffer = np.full((height, width), -1, dtype=np.int32)

    for link_name, points in points_by_link.items():
        if points.shape[0] == 0:
            continue
        proj = project_points(points, camera, resolution)
        xs = proj[:, 0].round().astype(np.int32)
        ys = proj[:, 1].round().astype(np.int32)
        zs = proj[:, 2]
        mask = (zs > 0) & (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        xs = xs[mask]
        ys = ys[mask]
        zs = zs[mask]
        if xs.size == 0:
            continue

        color = np.clip(colors_by_link[link_name][:3], 0, 1) * 255
        color = color.astype(np.uint8)
        owner_id = int(link_to_idx.get(link_name, -1))

        for x, y, z in zip(xs, ys, zs):
            if z < z_buffer[y, x]:
                z_buffer[y, x] = z
                owner_buffer[y, x] = owner_id
                color_buffer[y, x] = color
                if POINT_SIZE > 1:
                    x0 = max(0, x - POINT_SIZE // 2)
                    x1 = min(width, x0 + POINT_SIZE)
                    y0 = max(0, y - POINT_SIZE // 2)
                    y1 = min(height, y0 + POINT_SIZE)
                    color_buffer[y0:y1, x0:x1] = color

    label_positions = {}
    visible_pixels_by_link = {}
    for ln, li in link_to_idx.items():
        ys, xs = np.where(owner_buffer == int(li))
        if xs.size == 0:
            visible_pixels_by_link[ln] = np.zeros((0, 2), dtype=np.int32)
            continue
        y_cut = np.percentile(ys, 45.0)
        top_mask = ys <= y_cut
        if np.any(top_mask):
            xa = xs[top_mask]
            ya = ys[top_mask]
        else:
            xa = xs
            ya = ys
        label_positions[ln] = (float(np.median(xa)), float(np.median(ya)))
        visible_pixels_by_link[ln] = np.column_stack([xs.astype(np.int32), ys.astype(np.int32)])

    if return_visible_pixels:
        return color_buffer, label_positions, visible_pixels_by_link
    return color_buffer, label_positions


def _dilate_mask(mask: np.ndarray, iterations: int = 1):
    out = np.asarray(mask, dtype=bool)
    iters = max(0, int(iterations))
    for _ in range(iters):
        acc = out.copy()
        acc[:-1, :] |= out[1:, :]
        acc[1:, :] |= out[:-1, :]
        acc[:, :-1] |= out[:, 1:]
        acc[:, 1:] |= out[:, :-1]
        acc[:-1, :-1] |= out[1:, 1:]
        acc[1:, 1:] |= out[:-1, :-1]
        acc[:-1, 1:] |= out[1:, :-1]
        acc[1:, :-1] |= out[:-1, 1:]
        out = acc
    return out


def _component_boxes(mask: np.ndarray):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        stack = [(y0, x0)]
        visited[y0, x0] = True
        min_x = max_x = int(x0)
        min_y = max_y = int(y0)
        area = 0
        sum_x = 0.0
        sum_y = 0.0
        while stack:
            y, x = stack.pop()
            area += 1
            sum_x += float(x)
            sum_y += float(y)
            if x < min_x:
                min_x = x
            if x > max_x:
                max_x = x
            if y < min_y:
                min_y = y
            if y > max_y:
                max_y = y
            for ny in range(max(0, y - 1), min(h - 1, y + 1) + 1):
                for nx in range(max(0, x - 1), min(w - 1, x + 1) + 1):
                    if visited[ny, nx] or not mask[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(
            {
                "area": int(area),
                "bbox": (int(min_x), int(min_y), int(max_x), int(max_y)),
                "center": (float(sum_x / max(area, 1)), float(sum_y / max(area, 1))),
            }
        )
    return comps


def infer_bboxes_from_overlay(overlay_img, link_color_map, label_positions, resolution, visible_pixels_by_link=None):
    """
    Infer per-link bboxes directly from overlay visible pixels.
    Prefer the connected component nearest to the link label anchor.
    """
    width, height = resolution
    boxes = {}
    img = np.asarray(overlay_img, dtype=np.uint8)
    diag = float(np.hypot(width, height))
    for link_name, color in link_color_map.items():
        if visible_pixels_by_link is not None and link_name in visible_pixels_by_link:
            pts = np.asarray(visible_pixels_by_link.get(link_name), dtype=np.int32)
            mask = np.zeros((height, width), dtype=bool)
            if pts.size > 0:
                xs = np.clip(pts[:, 0], 0, width - 1)
                ys = np.clip(pts[:, 1], 0, height - 1)
                top_p = max(1.0, min(100.0, float(BBOX_OVERLAY_TOP_PERCENTILE)))
                y_cut = float(np.percentile(ys, top_p))
                top_mask = ys <= y_cut
                if np.any(top_mask) and int(np.sum(top_mask)) >= 24:
                    xs = xs[top_mask]
                    ys = ys[top_mask]
                mask[ys, xs] = True
        else:
            rgb = (np.clip(np.asarray(color[:3]), 0.0, 1.0) * 255.0).astype(np.uint8)
            mask = np.all(np.abs(img.astype(np.int16) - rgb.reshape(1, 1, 3).astype(np.int16)) <= 2, axis=2)
        if not np.any(mask):
            continue
        comps = _component_boxes(_dilate_mask(mask, iterations=BBOX_OVERLAY_DILATE_ITERS))
        if not comps:
            continue
        # Filter tiny noise components from sparse point rasterization.
        comps = [c for c in comps if c["area"] >= 12] or comps
        max_area = max(int(c["area"]) for c in comps)
        major = [c for c in comps if int(c["area"]) >= max(12, int(0.25 * max_area))]
        if not major:
            major = comps
        anchor = label_positions.get(link_name)
        if anchor is not None:
            ax, ay = float(anchor[0]), float(anchor[1])
        else:
            ax, ay = float(width) * 0.5, float(height) * 0.25
        best = None
        for c in major:
            area_norm = float(c["area"]) / max(float(max_area), 1.0)
            y_norm = float(c["center"][1]) / max(float(height), 1.0)
            dist = float(np.hypot(c["center"][0] - ax, c["center"][1] - ay))
            dist_norm = dist / max(diag, 1.0)
            score = area_norm - float(BBOX_OVERLAY_TOP_BIAS) * y_norm - float(BBOX_OVERLAY_DIST_BIAS) * dist_norm
            if best is None or score > best[0]:
                best = (score, c)
        if best is None:
            continue
        x0, y0, x1, y1 = best[1]["bbox"]
        x0 = max(0, min(width - 1, int(x0)))
        y0 = max(0, min(height - 1, int(y0)))
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes[link_name] = (x0, y0, x1, y1)
    return boxes


def project_label_positions(points_by_link, camera, resolution):
    width, height = resolution
    label_positions = {}
    for link_name, points in points_by_link.items():
        if points.shape[0] == 0:
            continue
        proj = project_points(points, camera, resolution)
        xs = proj[:, 0].round().astype(np.int32)
        ys = proj[:, 1].round().astype(np.int32)
        zs = proj[:, 2]
        mask = (zs > 0) & (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        xs = xs[mask]
        ys = ys[mask]
        if xs.size == 0:
            continue
        label_positions[link_name] = (float(np.median(xs)), float(np.median(ys)))
    return label_positions


def _load_mesh_textured(path):
    try:
        return trimesh.load(path, force=None, process=False, skip_materials=False)
    except Exception as exc:
        if "PIL" in str(exc) and path.suffix.lower() == ".obj":
            return _load_obj_simple(path)
        raise


def _build_reference_scene_from_urdf(links, urdf_dir, link_transforms):
    scene = trimesh.Scene()
    for link_name, visuals in links.items():
        link_tf = link_transforms.get(link_name, np.eye(4))
        for visual in visuals:
            if str(visual.get("geometry_type") or "mesh").lower() == "mesh":
                mesh_path = _resolve_mesh_path(visual.get("filename"), urdf_dir)
                if mesh_path is None or not mesh_path.exists():
                    continue
                try:
                    mesh_obj = _load_mesh_textured(mesh_path)
                except Exception:
                    continue
            else:
                mesh_obj = _primitive_mesh_from_visual(visual)
                if mesh_obj is None:
                    continue

            scale = visual["scale"] or [1.0, 1.0, 1.0]
            scale_mat = np.eye(4)
            scale_mat[0, 0] = scale[0]
            scale_mat[1, 1] = scale[1]
            scale_mat[2, 2] = scale[2]
            visual_mat = _origin_to_matrix(
                visual.get("origin_xyz"), visual.get("origin_rpy"), visual.get("origin_quat")
            )
            if isinstance(mesh_obj, trimesh.Scene):
                for node_name in mesh_obj.graph.nodes_geometry:
                    node_tf, geom_name = mesh_obj.graph[node_name]
                    geom = mesh_obj.geometry[geom_name].copy()
                    geom.apply_transform(scale_mat)
                    geom.apply_transform(node_tf)
                    geom.apply_transform(visual_mat)
                    geom.apply_transform(link_tf)
                    scene.add_geometry(geom)
            elif isinstance(mesh_obj, trimesh.Trimesh):
                geom = mesh_obj.copy()
                _apply_visual_rgba(geom, visual.get("material_rgba"))
                geom.apply_transform(scale_mat)
                geom.apply_transform(visual_mat)
                geom.apply_transform(link_tf)
                scene.add_geometry(geom)
    if scene.geometry:
        return scene
    return None


def load_reference_scene(asset_dir, links=None, urdf_dir=None, link_transforms=None):
    candidates = []
    canonical = asset_dir / f"animated_textured_{asset_dir.name}.glb"
    if canonical.exists():
        candidates.append(canonical)
    candidates += sorted([p for p in asset_dir.glob("animated_textured*.glb") if p != canonical])
    candidates += sorted(asset_dir.glob("textured*.glb"))
    candidates += sorted(asset_dir.glob("mesh_parts_with_axes*.glb"))

    urdf_scene = None
    if links and urdf_dir and link_transforms:
        urdf_scene = _build_reference_scene_from_urdf(links, urdf_dir, link_transforms)

    # Prefer canonical textured GLB first so reference images and motion grids
    # use the same geometry/material source.
    first_glb_scene = None
    for path in candidates:
        try:
            scene = trimesh.load(path, force="scene", process=False)
        except Exception:
            continue
        if isinstance(scene, trimesh.Trimesh):
            temp_scene = trimesh.Scene()
            temp_scene.add_geometry(scene)
            scene = temp_scene
        if not isinstance(scene, trimesh.Scene) or not scene.geometry:
            continue
        if first_glb_scene is None:
            first_glb_scene = scene
        if scene_has_effective_textures(scene):
            return scene

    if urdf_scene is not None and scene_has_effective_textures(urdf_scene):
        return urdf_scene
    if urdf_scene is not None:
        return urdf_scene
    return first_glb_scene


def find_reference_glb_path(asset_dir):
    candidates = []
    canonical = asset_dir / f"animated_textured_{asset_dir.name}.glb"
    if canonical.exists():
        candidates.append(canonical)
    candidates += sorted([p for p in asset_dir.glob("animated_textured*.glb") if p != canonical])
    candidates += sorted(asset_dir.glob("textured*.glb"))
    candidates += sorted(asset_dir.glob("mesh_parts_with_axes*.glb"))
    return candidates[0] if candidates else None


def scene_to_meshes(scene):
    meshes = []
    if scene is None:
        return meshes
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        mesh = geom.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def scene_has_texture_images(scene):
    if scene is None:
        return False
    for geom in scene.geometry.values():
        visual = getattr(geom, "visual", None)
        if isinstance(visual, trimesh.visual.texture.TextureVisuals):
            mat = getattr(visual, "material", None)
            if _get_material_texture_image(mat) is not None:
                return True
    return False


def _color_array_has_effective_appearance(arr, *, mean_thresh: float = 245.0, std_thresh: float = 6.0) -> bool:
    try:
        arr = np.asarray(arr, dtype=np.float32)
    except Exception:
        return False
    if arr.size == 0:
        return False
    if arr.ndim >= 2 and arr.shape[-1] >= 3:
        rgb = arr[..., :3]
    elif arr.ndim == 1 and arr.shape[0] >= 3:
        rgb = arr[:3]
    else:
        return False
    if float(np.max(rgb)) <= 1.0:
        rgb = rgb * 255.0
    mean = float(np.mean(rgb))
    std = float(np.std(rgb))
    return std > float(std_thresh) or mean < float(mean_thresh)


def _visual_has_effective_nonimage_appearance(visual) -> bool:
    if visual is None:
        return False
    try:
        vcols = getattr(visual, "vertex_colors", None)
        if vcols is not None and _color_array_has_effective_appearance(vcols):
            return True
    except Exception:
        pass
    try:
        fcols = getattr(visual, "face_colors", None)
        if fcols is not None and _color_array_has_effective_appearance(fcols):
            return True
    except Exception:
        pass
    mat = getattr(visual, "material", None)
    if mat is None:
        return False
    for attr in ("baseColorFactor", "main_color", "diffuse", "ambient", "specular"):
        try:
            val = getattr(mat, attr, None)
        except Exception:
            val = None
        if val is not None and _color_array_has_effective_appearance(val, mean_thresh=245.0, std_thresh=3.0):
            return True
    return False


def scene_has_effective_textures(scene):
    """
    Return True when the scene carries meaningful visual appearance, either via
    texture images or via non-image material/vertex/face colors.
    White/flat placeholder appearance is treated as ineffective.
    """
    if scene is None:
        return False
    found_texture = False
    found_nonimage = False
    for geom in scene.geometry.values():
        visual = getattr(geom, "visual", None)
        if _visual_has_effective_nonimage_appearance(visual):
            found_nonimage = True
        if not isinstance(visual, trimesh.visual.texture.TextureVisuals):
            continue
        mat = getattr(visual, "material", None)
        img = _get_material_texture_image(mat)
        if img is None:
            continue
        found_texture = True
        try:
            arr = np.array(img.convert("RGB"), dtype=np.float32)
        except Exception:
            continue
        if arr.size == 0:
            continue
        mean = float(arr.mean())
        std = float(arr.std())
        # Non-flat or non-white textures are treated as effective textures.
        if std > 8.0 or mean < 235.0:
            return True
    return bool(found_nonimage)


def _get_material_texture_image(mat):
    if mat is None:
        return None
    img = getattr(mat, "image", None)
    if img is not None:
        return img
    # GLB/PBR materials often store the diffuse texture here.
    img = getattr(mat, "baseColorTexture", None)
    if img is not None:
        return img
    return None


def enhance_textured_image(image):
    try:
        from PIL import Image, ImageEnhance

        arr = np.asarray(image, dtype=np.uint8)
        obj_mask = np.any(arr < 245, axis=2)
        if np.count_nonzero(obj_mask) == 0:
            return arr

        vals = arr[obj_mask].astype(np.float32)
        luma = 0.2126 * vals[:, 0] + 0.7152 * vals[:, 1] + 0.0722 * vals[:, 2]
        p50 = float(np.percentile(luma, 50))
        p90 = float(np.percentile(luma, 90))

        # Adaptive exposure: darker assets get more lift, but cap to avoid washout.
        gain = min(3.8, max(1.0, 120.0 / max(p50, 1.0)))
        if p90 > 185:
            gain = min(gain, 1.35)
        if p90 > 215:
            gain = min(gain, 1.15)

        img = Image.fromarray(arr)
        img = ImageEnhance.Brightness(img).enhance(gain)
        img = ImageEnhance.Color(img).enhance(1.30 if gain > 1.35 else 1.15)
        img = ImageEnhance.Contrast(img).enhance(1.12)
        out = np.array(img, dtype=np.uint8)

        # Extra lift for very dark textures (e.g., black/near-black albedo maps).
        if p50 < 40:
            gamma = 0.58
            lut = np.array([pow(i / 255.0, gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            out = lut[out]
            out = np.clip(out.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

        return out
    except Exception:
        return image


def is_reference_image_too_dark(image) -> bool:
    """
    Detect failed/degenerate dark renders from Blender (common on some GPU/material combos).
    """
    try:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return False
        obj_mask = np.any(arr < 245, axis=2)
        obj_ratio = float(np.count_nonzero(obj_mask)) / float(arr.shape[0] * arr.shape[1])
        min_obj_ratio = float(os.environ.get("CODEX_REF_DARK_MIN_OBJ_RATIO", "0.01"))
        if obj_ratio < min_obj_ratio:
            return False
        vals = arr[obj_mask].astype(np.float32)
        luma = 0.2126 * vals[:, 0] + 0.7152 * vals[:, 1] + 0.0722 * vals[:, 2]
        p50 = float(np.percentile(luma, 50))
        p90 = float(np.percentile(luma, 90))
        mean = float(luma.mean())
        std = float(luma.std())
        return (p50 < 40.0 and p90 < 160.0) or (mean < 28.0 and std < 35.0)
    except Exception:
        return False


def reference_image_chroma_stats(image) -> dict[str, float]:
    try:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Expected HxWx3 image")
        obj_mask = np.any(arr < 245, axis=2)
        obj_ratio = float(np.count_nonzero(obj_mask)) / float(arr.shape[0] * arr.shape[1])
        if obj_ratio <= 0.0:
            return {"obj_ratio": 0.0, "sat_mean": 0.0, "chroma_mean": 0.0}
        vals = arr[obj_mask].astype(np.float32) / 255.0
        v_max = vals.max(axis=1)
        v_min = vals.min(axis=1)
        chroma = v_max - v_min
        sat = np.where(v_max > 1.0e-6, chroma / np.maximum(v_max, 1.0e-6), 0.0)
        colorful_sat_min = float(os.environ.get("CODEX_REF_WASHED_COLORFUL_SAT_MIN", "0.10"))
        colorful_chroma_min = float(os.environ.get("CODEX_REF_WASHED_COLORFUL_CHROMA_MIN", "0.12"))
        colorful_ratio = float(np.mean((sat >= colorful_sat_min) | (chroma >= colorful_chroma_min)))
        return {
            "obj_ratio": obj_ratio,
            "sat_mean": float(np.mean(sat)),
            "chroma_mean": float(np.mean(chroma)),
            "colorful_ratio": colorful_ratio,
        }
    except Exception:
        return {"obj_ratio": 0.0, "sat_mean": 0.0, "chroma_mean": 0.0, "colorful_ratio": 0.0}


def reference_image_may_be_washed_out(image) -> bool:
    stats = reference_image_chroma_stats(image)
    min_obj_ratio = float(os.environ.get("CODEX_REF_WASHED_MIN_OBJ_RATIO", "0.01"))
    sat_mean_max = float(os.environ.get("CODEX_REF_WASHED_SAT_MEAN_MAX", "0.02"))
    chroma_mean_max = float(os.environ.get("CODEX_REF_WASHED_CHROMA_MEAN_MAX", "0.03"))
    return (
        stats.get("obj_ratio", 0.0) >= min_obj_ratio
        and stats.get("sat_mean", 0.0) <= sat_mean_max
        and stats.get("chroma_mean", 0.0) <= chroma_mean_max
    )


def is_reference_image_washed_out_vs_fallback(primary_image, fallback_image) -> bool:
    try:
        primary = reference_image_chroma_stats(primary_image)
        fallback = reference_image_chroma_stats(fallback_image)
        min_obj_ratio = float(os.environ.get("CODEX_REF_WASHED_MIN_OBJ_RATIO", "0.01"))
        min_fallback_sat = float(os.environ.get("CODEX_REF_WASHED_FALLBACK_SAT_MIN", "0.06"))
        min_fallback_chroma = float(os.environ.get("CODEX_REF_WASHED_FALLBACK_CHROMA_MIN", "0.08"))
        min_fallback_colorful_ratio = float(os.environ.get("CODEX_REF_WASHED_FALLBACK_COLORFUL_RATIO_MIN", "0.18"))
        min_ratio = float(os.environ.get("CODEX_REF_WASHED_RATIO_MIN", "2.5"))
        if primary["obj_ratio"] < min_obj_ratio or fallback["obj_ratio"] < min_obj_ratio:
            return False
        sat_ratio = fallback["sat_mean"] / max(primary["sat_mean"], 1.0e-4)
        chroma_ratio = fallback["chroma_mean"] / max(primary["chroma_mean"], 1.0e-4)
        return (
            reference_image_may_be_washed_out(primary_image)
            and fallback["colorful_ratio"] >= min_fallback_colorful_ratio
            and (
                (fallback["sat_mean"] >= min_fallback_sat and sat_ratio >= min_ratio)
                or (fallback["chroma_mean"] >= min_fallback_chroma and chroma_ratio >= min_ratio)
            )
        )
    except Exception:
        return False


def decide_reference_backend_for_batch(blender_images, fallback_images) -> dict[str, object]:
    blender_count = len(blender_images or [])
    fallback_count = len(fallback_images or [])
    decision: dict[str, object] = {
        "reference_backend": "blender",
        "reason": "blender_default",
        "blender_image_count": int(blender_count),
        "fallback_image_count": int(fallback_count),
        "dark_view_indices": [],
        "washed_out_view_indices": [],
    }
    if blender_count <= 0:
        if fallback_count > 0:
            decision["reference_backend"] = "software"
            decision["reason"] = "blender_unavailable"
        else:
            decision["reason"] = "no_reference_renderer_available"
        return decision
    if fallback_count != blender_count:
        return decision
    dark_views = [
        idx + 1
        for idx, img in enumerate(blender_images)
        if is_reference_image_too_dark(img)
    ]
    if dark_views:
        decision["reason"] = "blender_too_dark_blender_only"
        decision["dark_view_indices"] = dark_views
        return decision
    washed_out_views = [
        idx + 1
        for idx, (b_img, s_img) in enumerate(zip(blender_images, fallback_images))
        if is_reference_image_washed_out_vs_fallback(b_img, s_img)
    ]
    if washed_out_views:
        decision["reason"] = "blender_washed_out_blender_only"
        decision["washed_out_view_indices"] = washed_out_views
    return decision


def compute_scene_bounds_from_scene(scene):
    meshes = scene_to_meshes(scene)
    if not meshes:
        return np.array([0.0, 0.0, 0.0]), 1.0
    link_meshes = {"scene": meshes}
    return compute_scene_bounds(link_meshes)


def render_reference_textured(scene, camera, resolution):
    if scene is None:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    has_textures = scene_has_effective_textures(scene)
    if not has_textures:
        meshes = scene_to_meshes(scene)
        return render_reference_wireframe(meshes, camera, resolution)
    if _HAS_PYRENDER and not PYRENDER_DISABLED:
        setup_pyrender_headless()
        pyr_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.3, 0.3, 0.3])
        for node_name in scene.graph.nodes_geometry:
            transform, geom_name = scene.graph[node_name]
            geom = scene.geometry[geom_name]
            mesh = geom.copy()
            mesh.apply_transform(transform)
            try:
                material = None
                # If no texture or colors are near-white, force a darker neutral material for visibility.
                try:
                    has_texture = False
                    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
                        mat = mesh.visual.material
                        has_texture = _get_material_texture_image(mat) is not None
                    if not has_texture:
                        colors = None
                        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                            colors = mesh.visual.vertex_colors
                        elif hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
                            colors = mesh.visual.face_colors
                        if colors is None or colors.size == 0:
                            avg = 1.0
                        else:
                            avg = float(colors[:, :3].mean() / 255.0)
                        if avg > 0.85:
                            material = pyrender.MetallicRoughnessMaterial(
                                baseColorFactor=[0.45, 0.45, 0.45, 1.0], metallicFactor=0.0, roughnessFactor=1.0
                            )
                except Exception:
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.45, 0.45, 0.45, 1.0], metallicFactor=0.0, roughnessFactor=1.0
                    )
                pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False, material=material)
            except Exception:
                continue
            pyr_scene.add(pyr_mesh)
        eye, target, up = camera
        camera_pose = camera_pose_from_lookat(eye, target, up)
        camera_node = pyrender.PerspectiveCamera(yfov=np.deg2rad(50.0))
        pyr_scene.add(camera_node, pose=camera_pose)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
        pyr_scene.add(light, pose=camera_pose)
        try:
            dist = float(np.linalg.norm(np.asarray(eye) - np.asarray(target)))
            eye2 = np.asarray(target) + np.array([0.0, dist * 0.5, dist * 0.5], dtype=float)
            light_pose2 = camera_pose_from_lookat(eye2, target, up)
            light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
            pyr_scene.add(light2, pose=light_pose2)
        except Exception:
            pass
        def _render_pyrender_once():
            scale = max(1, int(REFERENCE_SUPERSAMPLE))
            w = int(resolution[0] * scale)
            h = int(resolution[1] * scale)
            renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
            flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT
            color, _ = renderer.render(pyr_scene, flags=flags)
            renderer.delete()
            color = color[:, :, :3]
            if scale > 1:
                try:
                    from PIL import Image

                    img = Image.fromarray(color)
                    img = img.resize((resolution[0], resolution[1]), resample=Image.LANCZOS)
                    out = np.array(img)
                    if has_textures:
                        out = enhance_textured_image(out)
                    return out
                except Exception:
                    out = color[::scale, ::scale, :]
                    if has_textures:
                        out = enhance_textured_image(out)
                    return out
            if has_textures:
                color = enhance_textured_image(color)
            return color
        try:
            return _render_pyrender_once()
        except Exception as exc:
            msg = str(exc)
            # Some servers export an invalid EGL_DEVICE_ID. Retry once without it.
            if "Invalid device ID" in msg and os.environ.get("EGL_DEVICE_ID") is not None:
                bad_id = os.environ.pop("EGL_DEVICE_ID", None)
                try:
                    print(f"[WARN] Pyrender offscreen failed ({exc}); retrying without EGL_DEVICE_ID={bad_id}.")
                    return _render_pyrender_once()
                except Exception as exc2:
                    print(f"[WARN] Pyrender offscreen retry failed ({exc2}); falling back to solid reference.")
            else:
                print(f"[WARN] Pyrender offscreen failed ({exc}); falling back to solid reference.")
    try:
        png = scene.save_image(resolution=resolution, visible=True)
        if png is not None:
            if has_textures:
                png = enhance_textured_image(png)
            return png
    except Exception as exc:
        fallback_kind = "solid" if has_textures else "wireframe"
        print(f"[WARN] Textured render unavailable ({exc}); falling back to {fallback_kind} reference.")
    meshes = scene_to_meshes(scene)
    if has_textures:
        return render_reference_solid(meshes, camera, resolution, max_faces=REFERENCE_MAX_FACES)
    # Wireframe tends to preserve thin parts (clock hands) better than solid shading.
    return render_reference_wireframe(meshes, camera, resolution)


def render_reference_points(meshes, camera, resolution):
    if not meshes:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    merged = trimesh.util.concatenate([m for m in meshes if m.vertices.size > 0])
    if merged.vertices.size == 0:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    np.random.seed(1234)
    try:
        points = merged.sample(POINTS_PER_LINK * 2)
    except Exception:
        points, _ = trimesh.sample.sample_surface(merged, POINTS_PER_LINK * 2)
    points = np.asarray(points, dtype=np.float32)
    grey = np.array([120, 120, 120], dtype=np.uint8)
    color_buffer = np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    z_buffer = np.full((resolution[1], resolution[0]), np.inf, dtype=np.float32)
    proj = project_points(points, camera, resolution)
    xs = proj[:, 0].round().astype(np.int32)
    ys = proj[:, 1].round().astype(np.int32)
    zs = proj[:, 2]
    mask = (zs > 0) & (xs >= 0) & (ys >= 0) & (xs < resolution[0]) & (ys < resolution[1])
    xs = xs[mask]
    ys = ys[mask]
    zs = zs[mask]
    for x, y, z in zip(xs, ys, zs):
        if z < z_buffer[y, x]:
            z_buffer[y, x] = z
            color_buffer[y, x] = grey
    return color_buffer


def render_reference_wireframe(meshes, camera, resolution, max_edges=200000):
    if not meshes:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    geoms = [m for m in meshes if m.vertices.size > 0 and m.faces.size > 0]
    if not geoms:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    merged = trimesh.util.concatenate(geoms)
    if merged.vertices.size == 0 or merged.faces.size == 0:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    edges = merged.edges_unique
    if edges.shape[0] > max_edges:
        step = max(1, edges.shape[0] // max_edges)
        edges = edges[::step][:max_edges]
    verts = np.asarray(merged.vertices, dtype=np.float32)
    proj = project_points(verts, camera, resolution)
    img = np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    col = np.array([30, 30, 30], dtype=np.uint8)
    for a, b in edges:
        pa = proj[a]
        pb = proj[b]
        if pa[2] <= 0 and pb[2] <= 0:
            continue
        _draw_line(img, pa[0], pa[1], pb[0], pb[1], col, thickness=1)
    return img


def render_reference_solid(meshes, camera, resolution, max_faces=REFERENCE_MAX_FACES):
    if not meshes:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    geoms = [m for m in meshes if m.vertices.size > 0 and m.faces.size > 0]
    if not geoms:
        return np.full((resolution[1], resolution[0], 3), 255, dtype=np.uint8)
    merged = trimesh.util.concatenate(geoms)
    if merged.faces.shape[0] > max_faces:
        step = max(1, merged.faces.shape[0] // max_faces)
        merged = trimesh.Trimesh(vertices=merged.vertices, faces=merged.faces[::step][:max_faces], process=False)
    color = np.array([0.6, 0.6, 0.6, 1.0], dtype=float)
    return render_software([merged], [color], camera, resolution, max_faces=max_faces)

def _render_software_torch(meshes, colors, camera, resolution, max_faces=SOFTWARE_MAX_FACES, return_depth: bool = False):
    device = _torch_raster_device()
    if device is None:
        raise RuntimeError("torch raster device unavailable")
    width, height = int(resolution[0]), int(resolution[1])
    color_buffer = torch.full((height, width, 3), 255.0, dtype=torch.float32, device=device)
    z_buffer = torch.full((height, width), float("inf"), dtype=torch.float32, device=device)

    eye, target, up = camera
    pose = camera_pose_from_lookat(np.asarray(eye, dtype=float), np.asarray(target, dtype=float), np.asarray(up, dtype=float))
    view = np.linalg.inv(pose)
    view_t = torch.as_tensor(view, dtype=torch.float32, device=device)

    fov = math.radians(50.0)
    aspect = float(width) / float(max(1, height))
    f = 1.0 / math.tan(fov / 2.0)

    def project(points: np.ndarray) -> torch.Tensor:
        pts = torch.as_tensor(np.asarray(points, dtype=np.float32), dtype=torch.float32, device=device)
        ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
        homog = torch.cat([pts, ones], dim=1)
        cam_pts = homog @ view_t.T
        z = -cam_pts[:, 2]
        z_safe = torch.where(torch.abs(z) > 1.0e-8, z, torch.full_like(z, 1.0e-8))
        x_ndc = (cam_pts[:, 0] * f / aspect) / z_safe
        y_ndc = (cam_pts[:, 1] * f) / z_safe
        x_screen = (x_ndc + 1.0) * 0.5 * float(width - 1)
        y_screen = (1.0 - (y_ndc + 1.0) * 0.5) * float(height - 1)
        return torch.stack([x_screen, y_screen, z], dim=1)

    def edge(a: torch.Tensor, b: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        return (cx - a[0]) * (b[1] - a[1]) - (cy - a[1]) * (b[0] - a[0])

    for mesh, color in zip(meshes, colors):
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        proj = project(np.asarray(mesh.vertices, dtype=np.float32))
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.shape[0] > max_faces:
            step = max(1, faces.shape[0] // max_faces)
            faces = faces[::step][:max_faces]
        rgba = np.clip(np.asarray(color, dtype=np.float32), 0.0, 1.0)
        rgb = torch.as_tensor(rgba[:3] * 255.0, dtype=torch.float32, device=device)
        alpha = float(rgba[3])

        for face in faces:
            tri = proj[torch.as_tensor(face, dtype=torch.long, device=device)]
            if bool(torch.all(tri[:, 2] <= 1.0e-6)):
                continue
            min_x = max(int(torch.floor(torch.min(tri[:, 0])).item()), 0)
            max_x = min(int(torch.ceil(torch.max(tri[:, 0])).item()), width - 1)
            min_y = max(int(torch.floor(torch.min(tri[:, 1])).item()), 0)
            max_y = min(int(torch.ceil(torch.max(tri[:, 1])).item()), height - 1)
            if max_x < min_x or max_y < min_y:
                continue
            p0, p1, p2 = tri[0], tri[1], tri[2]
            area = edge(p0, p1, p2[0], p2[1])
            if abs(float(area.item())) <= 1.0e-12:
                continue
            xs = torch.arange(min_x, max_x + 1, dtype=torch.float32, device=device) + 0.5
            ys = torch.arange(min_y, max_y + 1, dtype=torch.float32, device=device) + 0.5
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            w0 = edge(p1, p2, xx, yy)
            w1 = edge(p2, p0, xx, yy)
            w2 = edge(p0, p1, xx, yy)
            inside = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0))
            if not bool(torch.any(inside)):
                continue
            inv_area = 1.0 / area
            w0 = w0 * inv_area
            w1 = w1 * inv_area
            w2 = w2 * inv_area
            depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
            z_region = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
            update = inside & (depth < z_region)
            if not bool(torch.any(update)):
                continue
            z_buffer[min_y : max_y + 1, min_x : max_x + 1] = torch.where(update, depth, z_region)
            if alpha < 1.0:
                c_region = color_buffer[min_y : max_y + 1, min_x : max_x + 1]
                blended = rgb.view(1, 1, 3) * alpha + c_region * (1.0 - alpha)
                color_buffer[min_y : max_y + 1, min_x : max_x + 1] = torch.where(update.unsqueeze(-1), blended, c_region)
            else:
                c_region = color_buffer[min_y : max_y + 1, min_x : max_x + 1]
                color_buffer[min_y : max_y + 1, min_x : max_x + 1] = torch.where(update.unsqueeze(-1), rgb.view(1, 1, 3), c_region)

    image = torch.clamp(color_buffer, 0.0, 255.0).to(torch.uint8).cpu().numpy()
    if return_depth:
        return image, z_buffer.to(torch.float32).cpu().numpy()
    return image


def _render_software_numpy(meshes, colors, camera, resolution, max_faces=SOFTWARE_MAX_FACES, return_depth: bool = False):
    width, height = resolution
    color_buffer = np.full((height, width, 3), 255, dtype=np.float32)
    z_buffer = np.full((height, width), np.inf, dtype=np.float32)

    eye, target, up = camera
    pose = camera_pose_from_lookat(eye, target, up)
    view = np.linalg.inv(pose)

    fov = math.radians(50.0)
    aspect = width / height
    f = 1.0 / math.tan(fov / 2.0)

    def project(points):
        homog = np.column_stack([points, np.ones(len(points))])
        cam = (view @ homog.T).T
        z = -cam[:, 2]
        x_ndc = (cam[:, 0] * f / aspect) / z
        y_ndc = (cam[:, 1] * f) / z
        x_screen = (x_ndc + 1.0) * 0.5 * (width - 1)
        y_screen = (1.0 - (y_ndc + 1.0) * 0.5) * (height - 1)
        return np.column_stack([x_screen, y_screen, z])

    def edge(ax, ay, bx, by, cx, cy):
        return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax)

    for mesh, color in zip(meshes, colors):
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        proj = project(verts)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.shape[0] > max_faces:
            step = max(1, faces.shape[0] // max_faces)
            faces = faces[::step][:max_faces]
        rgba = np.clip(color, 0.0, 1.0)
        rgb = rgba[:3] * 255.0
        alpha = rgba[3]

        for face in faces:
            p0, p1, p2 = proj[face]
            if p0[2] <= 1e-6 and p1[2] <= 1e-6 and p2[2] <= 1e-6:
                continue
            xs = [p0[0], p1[0], p2[0]]
            ys = [p0[1], p1[1], p2[1]]
            min_x = max(int(math.floor(min(xs))), 0)
            max_x = min(int(math.ceil(max(xs))), width - 1)
            min_y = max(int(math.floor(min(ys))), 0)
            max_y = min(int(math.ceil(max(ys))), height - 1)
            area = edge(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
            if area == 0:
                continue
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    w0 = edge(p1[0], p1[1], p2[0], p2[1], x + 0.5, y + 0.5)
                    w1 = edge(p2[0], p2[1], p0[0], p0[1], x + 0.5, y + 0.5)
                    w2 = edge(p0[0], p0[1], p1[0], p1[1], x + 0.5, y + 0.5)
                    if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                        w0 /= area
                        w1 /= area
                        w2 /= area
                        depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
                        if depth < z_buffer[y, x]:
                            z_buffer[y, x] = depth
                            if alpha < 1.0:
                                color_buffer[y, x] = rgb * alpha + color_buffer[y, x] * (1.0 - alpha)
                            else:
                                color_buffer[y, x] = rgb

    image = np.clip(color_buffer, 0, 255).astype(np.uint8)
    if return_depth:
        return image, z_buffer.astype(np.float32, copy=False)
    return image


def render_software(meshes, colors, camera, resolution, max_faces=SOFTWARE_MAX_FACES, return_depth: bool = False):
    global _TORCH_RASTER_BACKEND_LOGGED
    device = _torch_raster_device()
    if device is not None:
        if not _TORCH_RASTER_BACKEND_LOGGED and _env_true("CODEX_TORCH_RASTER_LOG", False):
            print(f"[INFO] Torch raster backend active on {device}.")
            _TORCH_RASTER_BACKEND_LOGGED = True
        try:
            return _render_software_torch(
                meshes,
                colors,
                camera,
                resolution,
                max_faces=max_faces,
                return_depth=return_depth,
            )
        except Exception as exc:
            if _env_true("CODEX_TORCH_RASTER_WARN", True):
                print(f"[WARN] Torch raster backend failed ({exc}); falling back to numpy software rasterizer.")
    elif not _TORCH_RASTER_BACKEND_LOGGED and _env_true("CODEX_TORCH_RASTER_LOG", False):
        print("[INFO] Torch raster backend unavailable; using numpy software rasterizer.")
        _TORCH_RASTER_BACKEND_LOGGED = True
    return _render_software_numpy(
        meshes,
        colors,
        camera,
        resolution,
        max_faces=max_faces,
        return_depth=return_depth,
    )


class Renderer:
    def __init__(self):
        if _HAS_PYBULLET:
            self.backend = "pybullet"
        else:
            self.backend = "trimesh"

    def render(self, meshes, colors, camera, resolution, wireframe=False):
        if self.backend == "pybullet":
            try:
                return render_pybullet(meshes, colors, camera, resolution)
            except Exception as exc:
                print(f"[WARN] PyBullet render failed: {exc}. Falling back to trimesh.")
                return render_trimesh(meshes, colors, camera, resolution)
        return render_trimesh(meshes, colors, camera, resolution)


def render_pybullet(meshes, colors, camera, resolution):
    width, height = resolution
    eye, target, up = camera
    cid = p.connect(p.DIRECT)
    p.resetSimulation()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    for mesh, color in zip(meshes, colors):
        if mesh.vertices.size == 0 or mesh.faces.size == 0:
            continue
        vertices = mesh.vertices.tolist()
        indices = mesh.faces.tolist()
        rgba = [float(color[0]), float(color[1]), float(color[2]), float(color[3])]
        shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            vertices=vertices,
            indices=indices,
            rgbaColor=rgba,
        )
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape_id)

    view = p.computeViewMatrix(cameraEyePosition=eye.tolist(), cameraTargetPosition=target.tolist(), cameraUpVector=up.tolist())
    proj = p.computeProjectionMatrixFOV(fov=50, aspect=width / height, nearVal=0.01, farVal=1000.0)
    _, _, px, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER)
    p.disconnect(cid)

    img = np.array(px, dtype=np.uint8).reshape(height, width, 4)
    rgb = img[:, :, :3]
    # Replace pure black with white background
    mask = np.all(rgb < 5, axis=2)
    rgb[mask] = 255
    return rgb


def render_pyrender(meshes, colors, camera, resolution, wireframe=False):
    width, height = resolution
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[1.0, 1.0, 1.0])
    for mesh, color in zip(meshes, colors):
        if mesh.vertices.size == 0:
            continue
        rgba = [float(color[0]), float(color[1]), float(color[2]), float(color[3])]
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=rgba, metallicFactor=0.0, roughnessFactor=1.0)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        scene.add(mesh)

    eye, target, up = camera
    camera_pose = camera_pose_from_lookat(eye, target, up)
    camera_node = pyrender.PerspectiveCamera(yfov=np.deg2rad(50.0))
    scene.add(camera_node, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    flags = pyrender.RenderFlags.RGBA
    if wireframe:
        flags |= pyrender.RenderFlags.WIREFRAME

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, _ = renderer.render(scene, flags=flags)
    renderer.delete()
    return color[:, :, :3]


def render_trimesh(meshes, colors, camera, resolution):
    width, height = resolution
    scene = trimesh.Scene()
    scene.background = [255, 255, 255, 255]
    for mesh, color in zip(meshes, colors):
        if mesh.vertices.size == 0:
            continue
        mesh = mesh.copy()
        rgba = (np.clip(color, 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, face_colors=rgba)
        scene.add_geometry(mesh)

    eye, target, up = camera
    camera_pose = camera_pose_from_lookat(eye, target, up)
    scene.camera = trimesh.scene.cameras.Camera(resolution=(width, height), fov=(50, 50))
    scene.camera_transform = camera_pose

    try:
        png = scene.save_image(resolution=(width, height), visible=True)
        return png
    except Exception as exc:
        print(f"[WARN] Trimesh render failed ({exc}); using software rasterizer.")
        return render_software(meshes, colors, camera, resolution, max_faces=SOFTWARE_MAX_FACES)


def build_mesh_list(link_meshes):
    meshes = []
    link_names = []
    for link_name, mesh_list in link_meshes.items():
        for mesh in mesh_list:
            meshes.append(mesh)
            link_names.append(link_name)
    return meshes, link_names


def gather_links_root(links, joints):
    link_names = list(links.keys())
    children = set(j["child"] for j in joints if j.get("child"))
    root_links = [ln for ln in link_names if ln not in children]
    connected = set()
    for j in joints:
        if j.get("parent"):
            connected.add(j["parent"])
        if j.get("child"):
            connected.add(j["child"])
    info = {}
    for ln in link_names:
        info[ln] = {
            "is_root": ln in root_links,
            "connected": ln in connected,
        }
    return info


def write_vlm_prompt(
    asset_name,
    out_dir,
    links_info,
    image_entries,
    joints,
    user_prompt=None,
    link_ids=None,
    link_order=None,
    label_texts=None,
    scale_context=None,
):
    lines = []
    # lines.append(f"Asset: {asset_name}")
    # lines.append("")
    lines.append("Prompt to VLM (Causal Grounding & Motion Semantics)")
    lines.append("You are a visual causal-grounding and motion-semantics assistant.")
    lines.append("Input: (1) user action text, (2) overlay/reference images with labels and world-axis legend, (3) link inventory and image-label mapping, (4) Scale context JSON when available, (5) URDF joint summary.")
    lines.append("Output: ONLY valid JSON following the schema below. No extra text.")
    lines.append("")
    lines.append("Task: infer per-link semantics, ground the user's action to the correct manipulated link(s), and predict the causal action/effect structure and relevant joint targets.")
    lines.append("- Cover every visible link in semantics.links.")
    lines.append("- Distinguish direct control/manipulation from downstream effect motion.")
    lines.append("- Use causal_segments only when the action truly contains multiple temporal stages.")
    lines.append("- Use the world-axis legend and Scale context when judging direction and magnitude.")
    lines.append("")
    if user_prompt:
        lines.append("User action text:")
        lines.append(user_prompt.strip())
        lines.append("")
    lines.append("Grounding data provided below:")
    lines.append("Links:")
    if link_order is None:
        link_order = list(links_info.keys())
    for link_name in link_order:
        info = links_info.get(link_name, {"is_root": False, "connected": False})
        lines.append(
            f"- {link_name} (root: {'yes' if info['is_root'] else 'no'}, joint_connected: {'yes' if info['connected'] else 'no'})"
        )
    if link_ids:
        lines.append("")
        lines.append("Image Labels (shown in overlay/reference images):")
        for link_name in link_order:
            if link_name in link_ids:
                lines.append(f"- {link_ids[link_name]} = {link_name}")
    lines.append("")
    lines.append("Images:")
    for entry in image_entries:
        lines.append(f"- {entry['path']}: {entry['desc']}")
    lines.append("")
    if scale_context:
        lines.append("Scale context (same unit convention will be used in later planning/checking):")
        lines.append(json.dumps(scale_context, ensure_ascii=False, indent=2))
        obj_ext = scale_context.get("object_bbox_extents_m") if isinstance(scale_context, dict) else None
        if isinstance(obj_ext, list) and len(obj_ext) == 3:
            try:
                ex, ey, ez = [float(x) for x in obj_ext]
                horiz_axis = "+X" if abs(ex) >= abs(ey) else "+Y"
                lines.append(
                    f"- Orientation hint from bbox extents: extents≈[X:{ex:.4f}, Y:{ey:.4f}, Z:{ez:.4f}], dominant horizontal axis≈{horiz_axis}."
                )
            except Exception:
                pass
        lines.append("")
    lines.append("URDF joint summary:")
    if joints:
        for joint in joints:
            lines.append(format_joint_summary(joint))
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Instructions:")
    lines.append("- Use compact image labels (e.g., 15_12, 15) to identify parts.")
    lines.append("- Canonical link mapping is listed above in 'Image Labels'.")
    lines.append("- Both overlay and reference images include a world coordinate axis legend in the image corner: +X=red, +Y=green, +Z=blue (with explicit XYZ marks).")
    lines.append("- World frame is fixed across all views/stages. Use world axes (not image left/right) for direction judgment.")
    lines.append("- Coordinate convention for this pipeline: +Z is up, -Z is down. Treat motion direction in the horizontal plane as ±X/±Y unless action implies vertical motion.")
    lines.append("- Reference images are rendered from the assembled mesh with textures when available.")
    lines.append("- Reference images reuse the same bbox layout as the matching overlay view for the same camera.")
    lines.append("- Therefore, bbox positions stay aligned between overlay and reference for each view.")
    lines.append("- Overlay images may still show internal links for identity grounding.")
    lines.append("- Reference images omit links that look like enclosed internal parts, to reduce confusion with outer-surface parts.")
    lines.append("- Use overlay images as the authoritative source for link identity and bbox placement.")
    lines.append("- Treat the object as a real physical object; infer plausible causal behavior even if not explicit in URDF.")
    lines.append("- Use the Scale context to judge motion magnitude relative to object/link size (e.g., object_diag_m, link_bbox_extents_m).")
    lines.append("- Step 1: For EVERY link, provide a semantic label and a short visual description (1–2 sentences).")
    lines.append("- Step 2: Ground the user action to the first directly manipulated control link(s) in the real-world causal chain, not merely the final visibly moving part(s).")
    lines.append("- IMPORTANT: Distinguish between the DIRECTLY ACTED-ON control link and the FINAL EFFECT link.")
    lines.append("- target_link means the first link that the user directly manipulates in the causal chain (e.g., pedal, button, latch, switch, knob, handle).")
    lines.append("- Do NOT automatically choose the visually opening/moving part as target_link when a dedicated control is plausibly required first.")
    lines.append('- For outcome-oriented prompts (e.g., "open the rice cooker", "open the microwave"), prefer the dedicated release/control link as target_link when visible/plausible; represent lid/door opening in effects.')
    lines.append("- Only choose lid/door/drawer as target_link when the object is normally opened by directly grasping/pulling/lifting that part and no separate control is required.")
    lines.append("- Target grounding priority:")
    lines.append("  1) If prompt explicitly names a control part (pedal/button/latch/knob/handle), use that as target_link.")
    lines.append("  2) Else, if outcome normally requires a dedicated control and it is visible/plausible, use that control as target_link.")
    lines.append("  3) Else, use the directly manipulated moving part (lid/door/drawer) as target_link.")
    lines.append("- Step 3: First decide whether there is a clear causal effect from the action.")
    lines.append("- If there is NO clear causal effect, set causal.has_causal=false and causal.effects=null.")
    # lines.append("- Continuous-motion PRIORITY RULE: prompts like 'run', 'keep running', 'spin', 'rotate continuously', 'faster/slower' are causal by default.")
    lines.append("- For continuous-motion or speed-change prompts, set causal.has_causal=true unless the prompt is explicitly non-actional.")
    lines.append("- If no explicit external control part is visible for an ongoing mechanism, use the mechanism/root/body link as target_link with target_role='control'.")
    lines.append('- Add action.control_return_behavior when useful: "self_return" | "stays" | "unknown".')
    lines.append("- Outcome-preservation rule for return behavior: choose action.control_return_behavior so the final state requested by the user remains true at the end of the clip.")
    lines.append("- If releasing a control would mechanically undo the requested final effect (for example a sustaining pedal/lever whose release closes a lid/door), set control_return_behavior=stays unless the user explicitly asks to release/let go/return/close.")
    lines.append("- If the user explicitly includes release/let go/return/close in the action, then model the release as part of the action and include the corresponding downstream reverse effect when applicable.")
    lines.append("- If a transient control returns but its return does NOT undo the requested final effect (for example a latch/handle returns after an already-open door remains open), control_return_behavior=self_return is still valid.")
    lines.append("- IMPORTANT: transient rotary controls and persistent rotary controls must be distinguished.")
    lines.append("- Rotary does NOT imply persistent; a rotary control may still be a transient self-return control.")
    lines.append("- If a rotary control is used only to trigger unlatching / releasing / enabling another link, and is not itself the final held state, prefer action.control_return_behavior=self_return.")
    lines.append("- If a rotary control represents a persistent state setting or selectable retained state, prefer action.control_return_behavior=stays.")
    lines.append("- For continuous-motion prompts, effects.joint_targets MUST use NUMERIC velocity-style targets for all relevant moving joints, formatted as velocity:<signed_float_radps> (example: velocity:-6.5).")
    lines.append("- Do NOT use abstract placeholders such as velocity:omega_radps, velocity:fast, or velocity:unknown.")
    # Keep prompt rules asset-agnostic; avoid class-specific requirements such as clocks/wheels here.
    lines.append("- Do NOT output causal.has_causal=false for ongoing dynamic-state prompts that imply time evolution.")
    lines.append("- Step 4: If there IS causal effect, predict which joints/parts move as primary and coupled effects.")
    lines.append("- causal_segments is OPTIONAL and used only when the action must be split into multiple temporal phases.")
    lines.append("- If the action is single-phase, do NOT force segmentation; set causal_segments=null.")
    lines.append("- If the action is multi-phase (e.g., push then pull, open then close), split into ordered causal_segments.")
    lines.append("- IMPORTANT: causal_segments should capture causally distinct temporal phases, not only different manipulated links.")
    lines.append("- IMPORTANT: causal order does NOT imply that the compiled timeline must be strictly non-overlapping in time.")
    lines.append("- If a control remains engaged while the downstream effect begins, ordered causal segments may later compile into overlapping control_actuation and effect_motion windows.")
    lines.append("- A latency constraint only limits the earliest effect onset; it does NOT require the effect to wait until the control has completely ended.")
    # Keep segmentation rules generic; do not hard-code asset-class priors such as wheeled transport.
    lines.append("- For whole-object transport or sustained-motion actions, default to a single phase unless the prompt explicitly describes multiple temporal stages.")
    # lines.append("- Do NOT invent extra release/coast/deceleration phases from physics intuition alone when the prompt does not describe them.")
    lines.append("- If the action implies whole-object interaction with the environment (push/pull/drag/roll/slide/translate), direction_axis_world is REQUIRED for the corresponding causal action unit.")
    lines.append("- direction_axis_world must be expressed in WORLD frame (not camera frame).")
    lines.append("- direction_axis_world must be one of: [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1].")
    lines.append("- direction_axis_world indicates the intended motion direction in the horizontal plane (±X/±Y) or vertical direction (±Z).")
    lines.append("- Judge direction_axis_world from the combination of predicted semantics and the user action intent: use what the object is, how it is normally manipulated, and what the action asks it to do.")
    lines.append("- Infer world direction from the visible object orientation, the world-axis legend, and the provided action prompt; do not rely on hard-coded asset-category priors.")
    lines.append("- If causal_segments is null, put direction_axis_world in causal.action.")
    lines.append("- If causal_segments is used, put direction_axis_world in EACH relevant causal_segments[i].action (per-segment direction may differ).")
    lines.append("- IMPORTANT: When causal_segments is used, do NOT duplicate full action/effects in top-level causal.")
    lines.append("- If action requires first operating a control link and then separately manipulating the released part, use causal_segments for ordered phases.")
    lines.append("- If operating the control link alone directly causes the effect part to move (e.g., spring-loaded lid pops open), segmentation is optional and causal.action may still target the control link.")
    lines.append("- IMPORTANT: All coupled motions mentioned in coupling_rules MUST also appear explicitly in effects.joint_targets.")
    lines.append("- For continuous-rotation joints, do NOT use limit targets (upper/lower). Use a NUMERIC velocity-style target instead: {\"joint\":\"...\",\"to\":\"velocity:-6.5\"}.")
    lines.append("- Use the Scale context to choose a plausible numeric magnitude for each velocity target. The magnitude must be explicit in VLM output; do not leave it symbolic.")
    lines.append("- Confirm causal motion based on the real-world object, not just the URDF.")
    lines.append("- You may use either upper_limit or lower_limit in joint_targets, whichever matches the action.")
    lines.append("- Optional external conditioning inputs may be attached at VLM call time: conditioning object image(s), conditioning mask image(s), and/or conditioning text.")
    lines.append("- If conditioning masks are provided, first localize the masked region to the corresponding link(s), then infer action grounding and causal effects.")
    lines.append("- If multiple masks/regions are targets, output action.target_links with ALL matched links; also set action.target_link as the primary link (first/most salient).")
    lines.append("- If both mask(s) and text are provided, use masks for target localization and text for intent/time constraints.")
    lines.append("- If only one modality is provided (image/mask only or text only), still produce best-effort target grounding and causal prediction.")
    lines.append("")
    lines.append("Please answer with STRICT JSON only. No extra text, no Markdown.")
    lines.append("Detailed output requirements before schema:")
    lines.append("- semantics.links: must cover EVERY link listed above.")
    lines.append("- causal (REQUIRED) is a status header and must always exist.")
    lines.append("- causal.has_causal must be boolean.")
    lines.append("- If causal.has_causal=false: set causal.action=null and causal.effects=null, and set causal_segments=null.")
    lines.append("- For continuous-motion prompts (run/spin/rotate/faster/slower), causal.has_causal=false is INVALID unless the prompt explicitly denies any motion.")
    lines.append("- If causal.has_causal=true and causal_segments is null: fill causal.action and causal.effects (single-phase case).")
    lines.append("- If causal.has_causal=true and causal_segments is non-empty: set causal.action=null and causal.effects=null.")
    lines.append("- If causal.has_causal=true: action.target_role is REQUIRED and must be one of: control | direct_object.")
    lines.append("- If causal.has_causal=true: effects.effect_links is REQUIRED and lists the primary consequence links (e.g., lid/door/drawer).")
    lines.append("- causal_segments is OPTIONAL: only provide a non-empty list when temporal segmentation is necessary.")
    lines.append("- If causal_segments is provided, each segment must include: segment_id, time_hint.order_index, action, effects.")
    lines.append("- time_hint.order_index is REQUIRED.")
    lines.append("- time_hint.overlap_ok is OPTIONAL and should be used only when that causal segment may overlap in time with the next segment while still preserving causal order.")
    lines.append("- In causal_segments[*], action.target_role and effects.effect_links follow the same semantics as single-phase causal.action / causal.effects.")
    lines.append("- If causal_segments is null, top-level causal carries the whole action.")
    lines.append("- If causal_segments is non-empty, causal_segments is the ONLY authoritative source of detailed action/effects.")
    lines.append("Output schema:")
    lines.append("{")
    lines.append('  "semantics": {')
    lines.append('    "links": [')
    lines.append('      {"name": "link_X", "label": "semantic_label", "affordance": ["affordance"], "conf": 0.0, "description": "visual description"}')
    lines.append("    ]")
    lines.append("  },")
    lines.append('  "causal": {')
    lines.append('    "has_causal": true,')
    lines.append('    "action": {"primitive": "action_verb", "target_link": "link_X", "target_links": ["link_X"], "target_role": "control", "magnitude": 0.0, "direction_axis_world": [1,0,0], "control_return_behavior": "unknown"},')
    lines.append('    "effects": {"effect_links": ["link_Y"], "joint_targets": [{"joint": "joint_name", "to": "alpha*upper_limit"}], "modes": [{"name": "mode_name", "set": true}], "coupling_rules": ["if mode then joint change"]}')
    lines.append("  },")
    lines.append('  "causal_segments": null')
    lines.append("}")
    lines.append("When segmentation is needed, use this causal_segments format:")
    lines.append('  "causal": {"has_causal": true, "action": null, "effects": null},')
    lines.append('  "causal_segments": [')
    lines.append('    {"segment_id":"S1","time_hint":{"order_index":0,"overlap_ok":true},"action":{"primitive":"press","target_link":"link_button","target_role":"control","control_return_behavior":"self_return",...},"effects":{"effect_links":["link_lid"],...}},')
    lines.append('    {"segment_id":"S2","time_hint":{"order_index":1},"action":{"primitive":"lift","target_link":"link_lid","target_role":"direct_object",...},"effects":{"effect_links":["link_lid"],...}}')
    lines.append("  ]")
    lines.append("Continuous-motion single-phase example (no segmentation needed):")
    lines.append('  "causal": {"has_causal": true, "action": {"primitive":"set_speed","target_link":"link_body","target_role":"control"}, "effects": {"effect_links":["link_hand_a","link_hand_b"], "joint_targets":[{"joint":"joint_a","to":"velocity:0.2"},{"joint":"joint_b","to":"velocity:12.0"}], "modes":[{"name":"running","set":true}], "coupling_rules":["linked continuous rotation"]}},')
    lines.append('  "causal_segments": null')
    lines.append("Multi-stage transport example (segmentation needed only when the prompt explicitly says release/coast or another later stage):")
    lines.append('  "causal": {"has_causal": true, "action": null, "effects": null},')
    lines.append('  "causal_segments": [')
    lines.append('    {"segment_id":"S1","time_hint":{"order_index":0},"action":{"primitive":"push","target_link":"link_body","target_role":"direct_object","direction_axis_world":[1,0,0]},"effects":{"effect_links":["link_body"],"joint_targets":[{"joint":"joint_aux_a","to":"velocity:8.0"},{"joint":"joint_aux_b","to":"velocity:8.0"}],"modes":[{"name":"transport","set":true}],"coupling_rules":["direct push drives body transport and coupled auxiliary rotation"]}},')
    lines.append('    {"segment_id":"S2","time_hint":{"order_index":1},"action":{"primitive":"release","target_link":"link_body","target_role":"direct_object","direction_axis_world":[1,0,0]},"effects":{"effect_links":["link_body"],"joint_targets":[{"joint":"joint_aux_a","to":"velocity:4.0"},{"joint":"joint_aux_b","to":"velocity:4.0"}],"modes":[{"name":"coasting","set":true}],"coupling_rules":["after release the body continues briefly, then decelerates"]}}')
    lines.append('  ]')
    lines.append("")
    lines.append("Notes:")
    lines.append("- Do not use canned examples; infer labels and motions from the images and joint summary.")
    lines.append("- Do not duplicate detailed action/effects in both causal and causal_segments.")
    lines.append("- Single-phase: use top-level causal.action/effects and set causal_segments=null.")
    lines.append("- Multi-phase: use causal_segments for details and keep top-level causal.action/effects null.")
    # Keep notes generic; avoid asset-specific transport assumptions here.
    lines.append("- direction_axis_world is required for whole-object motion actions (push/pull/drag/roll/slide/translate); otherwise omit it.")
    lines.append("- target_link is the primary target. target_links is optional but REQUIRED when multiple target links are involved (e.g., multiple masks).")
    lines.append("- target_link should indicate what the user directly manipulates first; effects.effect_links should indicate what moves as consequence.")
    lines.append("- If you include velocity-style joint targets, their sign should be consistent with direction_axis_world when they are mechanically coupled to that transport direction.")
    lines.append("- Every velocity-style joint target must contain an explicit numeric magnitude. Symbolic placeholders are invalid.")
    lines.append("- If unsure, use label \"unknown\" with low confidence, but still provide a visual description.")

    out_path = Path(out_dir) / "vlm_prompt.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_llm_plan_prompt(asset_name, out_dir, joints, user_prompt=None, vlm_json_placeholder=True, scale_context=None):
    lines = []
    lines.append("Prompt to LLM (Parameter & Plan Compiler)")
    lines.append("You are a simulation plan compiler.")
    lines.append("Input: (1) user action text, (2) Scale context JSON (object/link sizes, same unit convention for all stages), (3) URDF joint summary (with numeric limits), (4) VLM causal JSON (semantics + causal_segments + causal summary).")
    lines.append("Output: ONLY valid JSON following the schema below. No extra text.")
    lines.append("")
    lines.append("Task: Convert the action text into a time-parameterized executable plan for animation/simulation.")
    lines.append("The plan MUST include:")
    lines.append("- total duration (seconds), fps")
    lines.append("- time segments (t0, t1)")
    lines.append("- each segment must include phase_type")
    lines.append("- for each segment: which joints move (from the VLM JSON), and how (position/velocity/decay; use spring_return only when a real return-to-rest behavior is justified)")
    lines.append("- If VLM causal_segments is present, preserve its temporal order when compiling timeline segments.")
    lines.append("- If per-segment direction_axis_world differs across causal segments, preserve those direction changes in timeline controls.")
    lines.append("- If VLM causal.has_causal=false (and effects is null), generate a no-op/hold timeline instead of inventing motion.")
    # Keep planner instructions asset-agnostic; avoid wheel/trolley-specific planning rules here.
    lines.append("- For whole-object transport or sustained-motion actions, default to a single constant-speed drive segment unless the prompt explicitly mentions a later release/coast/deceleration/acceleration/stop stage.")
    lines.append("- numeric values: durations, velocities, target angles")
    lines.append("- a timing_checks block summarizing control/effect timing consistency")
    lines.append("- choose distance/speed magnitudes using the provided Scale context ONLY when the VLM causal JSON does not already provide an explicit numeric value")
    lines.append("- if a target is expressed as alpha*upper_limit, you MUST resolve it to a numeric rad value using the URDF limits and include both the expression and the numeric value.")
    lines.append("- Preserve causal order, but adjacent phases may overlap in time when semantics justify concurrent control and effect.")
    lines.append("")
    lines.append("VLM Numeric Authority Rules:")
    lines.append("- If VLM effects.joint_targets or VLM causal_segments[*].effects.joint_targets contains an explicit numeric target, treat that numeric value as authoritative.")
    lines.append("- For explicit VLM velocity targets such as velocity:<number>, compile the timeline around that numeric value instead of inventing a new speed from the user text.")
    lines.append("- For explicit VLM limit targets such as upper_limit, lower_limit, alpha*upper_limit, or alpha*lower_limit, preserve the exact per-joint target expression from VLM.")
    lines.append("- Never normalize, mirror, simplify, or 'make symmetric' explicit per-joint limit targets from VLM.")
    lines.append("- If VLM assigns different joints to different sides of their limits (for example one joint -> upper_limit and another joint -> lower_limit), you MUST preserve that exact assignment in the plan.")
    lines.append("- Do NOT replace lower_limit with upper_limit just because another nearby or visually similar joint uses upper_limit, and do NOT replace upper_limit with lower_limit just because the mechanism looks symmetric.")
    lines.append("- For paired handles, mirrored links, left/right parts, or visually symmetric mechanisms, treat each joint's VLM target as independent unless VLM explicitly says they are the same.")
    lines.append("- The LLM is responsible for timeline structure, phase timing, decay shape, and control arrangement; it is NOT allowed to replace an explicit VLM numeric magnitude with a newly inferred one.")
    # lines.append("- User text such as faster/slower, 50x/100x, or natural-language intensity may help choose duration and staging, but must NOT override an explicit numeric VLM joint target.")
    # lines.append("- When VLM gives explicit per-joint values, preserve their relative magnitudes and signs unless the VLM itself is internally inconsistent.")
    # lines.append("- If VLM provides an explicit numeric value for one joint but not another, keep the explicit one fixed and infer only the missing values.")
    lines.append("- If you must refine a VLM numeric value because it is clearly incomplete, keep the change minimal and explain it through timeline shaping rather than replacing the magnitude outright.")
    lines.append("")
    lines.append("Temporal Causality Hard Rules:")
    lines.append("- If action.target_role is control and the primary effect link differs from target_link, timeline MUST be split into ordered phases.")
    lines.append("- Ordered phases preserve causal order, but they do NOT have to be strictly non-overlapping in time.")
    lines.append("- Ordered phases are logical causal phases; they may overlap in time and do not imply mutually exclusive time windows.")
    lines.append("- Minimum required phase structure for control->effect coupling:")
    lines.append("  control_actuation -> causal_latency_or_release -> effect_motion")
    lines.append("- Do NOT put control-joint actuation and effect-joint main motion in the same phase unless direct rigid coupling is explicitly justified.")
    lines.append("- Enforce ordering: effect_start_time >= control_onset_time + delay_min.")
    lines.append("- effect_start_time may be earlier than control_end_time if the control remains engaged while the effect starts.")
    lines.append("- Default delay_min guidance:")
    lines.append("  0.05s for button/latch/switch release coupling; 0.10s for spring-loaded lid/door; 0.00s only for direct grasped-part motion.")
    lines.append("- If uncertain, insert a short latency phase (0.05~0.15s) instead of collapsing phases.")
    lines.append("- Anti-collapse rule: plans that merge control_actuation and effect_motion into one phase are invalid when target_link != primary effect_link.")
    lines.append("- A single timeline window may contain multiple concurrent controls/actions if the semantics justify overlap.")
    lines.append("- controls do not need to be one-per-phase; concurrent control and effect may be represented either by overlapping segments or by multiple controls active in the same time window when semantically justified.")
    # lines.append("- Control-release dedup rule: for the same control joint, emit at most ONE control_release/spring_return phase unless there is a NEW control_actuation for that joint later.")
    # lines.append("- Do NOT create an extra late control_release in settle/final phases if that control joint already returned earlier.")
    # lines.append("- If a control is already released and held at rest, use hold_position or no control; do NOT add another spring_return for the same joint.")
    # lines.append("")
    lines.append("- Control-release rule: emit a control_release / spring_return phase ONLY when the control itself physically returns toward rest after release.")
    lines.append("- Outcome-preservation rule: do NOT emit control_release / spring_return if that release would contradict the user's requested final outcome unless the user explicitly asks for the release/return/close phase.")
    lines.append("- For sustaining controls, keep the control held with hold_position through the effect/final hold when needed to preserve the requested final state.")
    lines.append("- Example: for 'fully open the trash bin' on a pedal bin, keep the pedal pressed/held while the lid remains open; do not add pedal_return unless the action says to release the pedal, step off, or let the lid close.")
    lines.append("- Example: for 'open the door' with a latch/handle, a handle self-return can be valid if the door remains open after unlatching; the return does not contradict the final open state.")
    lines.append("- Use release return when the articulated part is force-loaded during actuation and, once the external force is removed, the mechanism itself tends to rebound/restore toward a rest pose because of spring force, elastic restoring torque, gravity-biased self-closing, or another built-in restoring effect.")
    lines.append("- Do NOT use release return when the articulated part can stably remain at multiple poses/states after actuation and does not inherently rebound toward one rest pose once released.")
    lines.append("- A release return is justified by mechanism behavior, not by the fact that the action has ended. If removing the hand/force would leave the part where it is, omit control_release / spring_return.")
    lines.append("- For rotary controls, do not decide return behavior from joint type alone; decide it from control semantics.")
    lines.append("- Persistent rotary controls that represent retained settings/states usually stay where actuated unless VLM/URDF explicitly indicates self-return.")
    lines.append("- Transient rotary unlatch/release controls may self-return and should not be suppressed just because they are rotary.")
    lines.append("- If VLM action.control_return_behavior == self_return, prefer generating a short control_release / spring_return phase back toward neutral/rest.")
    # lines.append("- For controls that stay where actuated (for example safe dials, many knobs, many rigid handles), omit release return entirely unless an explicit visible return is required.")
    lines.append("")
    lines.append("Action-dependent controls (general rules):")
    lines.append("- If the action implies whole-object motion (e.g., push/pull/drag), include base motion controls: base_velocity or base_velocity_decay.")
    lines.append("- Direction source: if causal_segments exists, use each segment action.direction_axis_world; otherwise use causal.action.direction_axis_world.")
    lines.append("- Do not reinterpret direction in camera/image frame.")
    lines.append("- If VLM action.target_links exists, treat all listed links as intended targets; target_link is only the primary target.")
    lines.append("- If the action implies articulated motion (e.g., press/open/rotate), use joint_position/joint_velocity controls.")
    lines.append("- Include only control fields that are justified by the VLM semantics and the executable schema; do not introduce asset-class-specific extras.")
    lines.append("- IMPORTANT: Follow the VLM causal JSON strictly. Do NOT invent new motions or swap joints.")
    lines.append("- IMPORTANT: Treat VLM as the authoritative source for target/effect identity, temporal order, direction_axis_world, motion direction, and any explicit numeric joint targets.")
    lines.append("- IMPORTANT: Downstream plan compilation may refine timeline structure and separate grouped controls into per-joint controls, but must preserve explicit VLM numeric magnitudes unless the VLM omits them.")
    lines.append("- IMPORTANT: If VLM gives explicit joint_targets.to values, the plan must preserve those exact per-joint targets in joint_position controls; do not flip lower_limit/upper_limit for any joint unless a later VLM segment for that same joint explicitly changes it.")
    lines.append("- Your job is to compile the VLM findings into an executable plan, and you may separate coupled articulated controls into per-joint controls when that improves executability or allows one joint to differ from the others.")
    lines.append("- Do NOT blindly preserve a shared sign or magnitude across all joints if only some joints should be reversed or adjusted.")
    lines.append("- If VLM implies a grouped joint direction implicitly, you may emit different per-joint signs in timeline controls when individual joints need different directions.")
    lines.append("- If visual evidence is ambiguous, prefer explicit per-joint controls over one shared grouped control.")
    lines.append("- However, ambiguity is NEVER a reason to override an explicit VLM target side such as lower_limit or upper_limit.")
    lines.append("- If you override a VLM joint-direction/sign choice, keep the rest of the VLM structure stable and explain the distinction through more explicit per-joint timeline controls.")
    lines.append("- If causal_segments exists, use per-segment effects.joint_targets for segment-local controls.")
    lines.append("- Use only joints mentioned in VLM effects.joint_targets or VLM causal_segments[*].effects.joint_targets; do not add extra joints unless explicitly specified there.")
    lines.append("- Preserve joint identity and ordering from VLM, but you may assign different control values/signs to different joints.")
    lines.append("- If a joint represents continuous rotation or repeated cyclic motion, prefer joint_velocity over joint_position.")
    lines.append("- Use Scale context to keep base distance, base speed, and joint angular speed consistent with object size only when VLM has not already provided an explicit numeric target.")
    lines.append("")
    lines.append("Constraints:")
    lines.append("- fps in {24,30}")
    lines.append("- duration_s in [1.0, 8.0]")
    lines.append("- omega_radps in [0.0, 20.0]")
    lines.append("- spring_k in [0.0, 20.0], damping_c in [0.0, 5.0]")
    lines.append("")
    lines.append("Optional fields (use only when relevant):")
    lines.append("- nl_parse: lightweight metadata only (for example action/target text if useful for readability).")
    lines.append("- Do NOT use nl_parse to encode executable motion direction, joint sign, grouped-control structure, or any other executable configuration.")
    lines.append("- Encode actual motion direction only inside timeline controls, especially base_velocity/base_velocity_decay.axis_world and joint_velocity controls.")
    lines.append("- physics.ground_friction_mu and physics.rolling_resistance (only when surface/terrain is mentioned)")
    lines.append("")
    lines.append("Control format rules:")
    lines.append("- Each control uses either a \"type\" field OR a \"mode\" field.")
    lines.append("- If using \"mode\", it MUST be one of:")
    lines.append("  joint_position | joint_velocity | base_velocity | base_velocity_decay | spring_return | hold_position | mode_set")
    lines.append("- Use joint_* modes for articulated parts whenever explicit joint motion is needed.")
    lines.append("- Use base_* modes only when the whole object should move.")
    lines.append("- Do NOT invent new field names; use only the allowed fields below.")
    lines.append("- STRICT OUTPUT RULE: every articulated control MUST target exactly one joint using the field \"joint\".")
    lines.append("- NEVER emit grouped controls with \"joints\": [...] even if multiple joints share the same magnitude.")
    lines.append("- If multiple joints should all rotate, emit separate joint_velocity controls, one per joint.")
    lines.append("")
    lines.append("Allowed fields by control type (strict):")
    lines.append("- base_velocity: {type/mode, axis_world, v_mps}")
    lines.append("- base_velocity_decay: {type/mode, axis_world, v0_mps, tau_s}")
    lines.append("- joint_velocity: {type/mode, joint, omega_radps, ramp_to_omega_radps, decay}")
    lines.append("  decay is OPTIONAL. Omit decay for steady driven rotation, including normal pushed/pulled trolley wheel rolling.")
    lines.append("  Use decay only when the action/segment explicitly describes release, coast, deceleration, spin-down, slowdown, or stop behavior.")
    lines.append("  If decay is used, it must be an object: {\"type\":\"exponential\", \"tau_s\":..., \"min_omega_radps\":...}.")
    lines.append("  IMPORTANT: min_omega_radps is an unsigned magnitude floor (>= 0), not a signed velocity.")
    lines.append("  The rotation/translation sign always comes from omega_radps or ramp_to_omega_radps; keep min_omega_radps positive even when omega_radps is negative.")
    lines.append("  Example: omega_radps=-10.0 with decay.min_omega_radps=9.5 means the joint stays in the negative direction, but its speed magnitude should not decay below 9.5 rad/s.")
    lines.append("- joint_position: {type/mode, joint, q_target_expr OR q_target_rad, q_start_rad, curve}")
    lines.append("- hold_position: {type/mode, joint}")
    lines.append("- spring_return: {type/mode, joint, spring_k, damping_c, rest_position}")
    lines.append("- mode_set: {type/mode, name, set}")
    lines.append("")
    lines.append("Action-to-control consistency:")
    lines.append("- If action is push/pull/drag: include base_velocity or base_velocity_decay controls.")
    lines.append("- If VLM semantics indicates additional articulated joints should move during transport, include explicit controls for those joints.")
    lines.append("- If VLM already provides joint_targets.to for a joint, joint_position controls for that joint must use the same target expression/limit side rather than recomputing a new side from action text or symmetry.")
    lines.append("- If action is press/open/rotate: include joint_position or joint_velocity for the target joints.")
    lines.append("- If a control joint appears in both control_actuation and control_release, keep exactly one control_release for that actuation cycle.")
    lines.append("- Only use spring_return when the control is expected to visibly return toward a rest pose after release (e.g., spring-loaded button/lever/handle).")
    lines.append("- More generally: add spring_return only when releasing the actuation force should cause the joint itself to move back because the mechanism has a real restoring effect; omit it when the mechanism can simply stay at the reached state.")
    lines.append("- Before adding spring_return, check whether that return would undo any effect joint that must remain in the requested final state. If yes, omit spring_return and hold the sustaining control instead, unless the user action explicitly includes release/return/close.")
    lines.append("- If the control is not expected to self-return, do NOT create a control_release segment just to 'finish' the action.")
    lines.append("- If the mechanism can stop and remain stable in multiple states, model that with no release return unless the VLM/URDF clearly indicates a real rebound toward rest.")
    lines.append("- For spring_return / control_release phases, if full return must be visible, prefer giving the return phase enough time (adjust t0/t1) before increasing spring_k or damping_c.")
    lines.append("- If a link/control must fully return, size the return phase against the URDF joint limit span and intended return distance; do NOT assume a short default settle window is enough.")
    lines.append("")
    lines.append("Examples (generic, not asset-specific):")
    lines.append("1) Whole-object transport with repeated rotating joints:")
    lines.append("{")
    lines.append("  \"timeline\": [")
    lines.append("    {\"name\": \"push_phase\", \"phase_type\": \"effect_motion\", \"t0\": 0.0, \"t1\": 1.0, \"controls\": [")
    lines.append("      {\"mode\":\"base_velocity\", \"axis_world\":[-1,0,0], \"v_mps\":0.6},")
    lines.append("      {\"mode\":\"joint_velocity\", \"joint\":\"joint_aux_A\", \"omega_radps\":6.0},")
    lines.append("      {\"mode\":\"joint_velocity\", \"joint\":\"joint_aux_B\", \"omega_radps\":6.0}")
    lines.append("    ]}")
    lines.append("  ]")
    lines.append("}")
    lines.append("2) Control->effect with latency (button then door/lid):")
    lines.append("{")
    lines.append("  \"timeline\": [")
    lines.append("    {\"name\":\"control_press\", \"phase_type\":\"control_actuation\", \"t0\":0.00, \"t1\":0.12, \"controls\":[{\"mode\":\"joint_position\",\"joint\":\"joint_control\",\"q_start_rad\":0.0,\"q_target_expr\":\"0.8*upper_limit\",\"curve\":\"ease_in_out\"}]},")
    lines.append("    {\"name\":\"latency\", \"phase_type\":\"causal_latency\", \"t0\":0.12, \"t1\":0.20, \"controls\":[{\"mode\":\"hold_position\",\"joint\":\"joint_control\"}]},")
    lines.append("    {\"name\":\"effect_open\", \"phase_type\":\"effect_motion\", \"t0\":0.20, \"t1\":1.10, \"controls\":[{\"mode\":\"joint_position\",\"joint\":\"joint_effect\",\"q_start_rad\":0.0,\"q_target_expr\":\"upper_limit\",\"curve\":\"ease_out\"}]},")
    lines.append("    {\"name\":\"control_return\", \"phase_type\":\"control_release\", \"t0\":1.10, \"t1\":1.40, \"controls\":[{\"mode\":\"spring_return\",\"joint\":\"joint_control\",\"spring_k\":4.0,\"damping_c\":0.6,\"rest_position\":0.0}]}")
    lines.append("  ],")
    lines.append("  \"timing_checks\": {\"control_onset_time\":0.00, \"control_peak_time\":0.12, \"effect_start_time\":0.20, \"enforced_delay_s\":0.08}")
    lines.append("}")
    lines.append("2a) Ordered but overlapping control->effect coupling:")
    lines.append("{")
    lines.append("  \"timeline\": [")
    lines.append("    {\"name\":\"control_actuation\", \"phase_type\":\"control_actuation\", \"t0\":0.00, \"t1\":0.50, \"controls\":[{\"mode\":\"joint_velocity\",\"joint\":\"joint_control\",\"omega_radps\":4.0}]},")
    lines.append("    {\"name\":\"effect_begin\", \"phase_type\":\"effect_motion\", \"t0\":0.22, \"t1\":1.20, \"controls\":[{\"mode\":\"joint_position\",\"joint\":\"joint_effect\",\"q_start_rad\":0.0,\"q_target_expr\":\"upper_limit\",\"curve\":\"ease_out\"}]}")
    lines.append("  ],")
    lines.append("  \"timing_checks\": {\"control_onset_time\":0.00, \"effect_start_time\":0.22, \"enforced_delay_s\":0.22}")
    lines.append("}")
    lines.append("- The example above is valid because control_actuation continues while effect_motion begins; causal order is preserved even though the windows overlap.")
    lines.append("2b) Sustaining control without release return (pedal/lever keeps final effect open):")
    lines.append("{")
    lines.append("  \"timeline\": [")
    lines.append("    {\"name\":\"pedal_press\", \"phase_type\":\"control_actuation\", \"t0\":0.00, \"t1\":0.20, \"controls\":[{\"mode\":\"joint_position\",\"joint\":\"joint_pedal\",\"q_start_rad\":0.0,\"q_target_expr\":\"upper_limit\",\"curve\":\"ease_in_out\"}]},")
    lines.append("    {\"name\":\"causal_latency\", \"phase_type\":\"causal_latency\", \"t0\":0.20, \"t1\":0.30, \"controls\":[{\"mode\":\"hold_position\",\"joint\":\"joint_pedal\"}]},")
    lines.append("    {\"name\":\"lid_open\", \"phase_type\":\"effect_motion\", \"t0\":0.30, \"t1\":1.20, \"controls\":[{\"mode\":\"hold_position\",\"joint\":\"joint_pedal\"},{\"mode\":\"joint_position\",\"joint\":\"joint_lid\",\"q_start_rad\":0.0,\"q_target_expr\":\"upper_limit\",\"curve\":\"ease_out\"}]},")
    lines.append("    {\"name\":\"hold_open\", \"phase_type\":\"hold\", \"t0\":1.20, \"t1\":1.80, \"controls\":[{\"mode\":\"hold_position\",\"joint\":\"joint_pedal\"},{\"mode\":\"hold_position\",\"joint\":\"joint_lid\"}]}")
    lines.append("  ],")
    lines.append("  \"timing_checks\": {\"control_onset_time\":0.00, \"control_peak_time\":0.20, \"effect_start_time\":0.30, \"enforced_delay_s\":0.10}")
    lines.append("}")
    lines.append("- The example above is valid because releasing the pedal would undo the requested final open state; the action did not ask for release/close.")
    lines.append("2c) Control without automatic release return (dial/knob/rigid handle):")
    lines.append("{")
    lines.append("  \"timeline\": [")
    lines.append("    {\"name\":\"dial_turn\", \"phase_type\":\"control_actuation\", \"t0\":0.00, \"t1\":0.80, \"controls\":[{\"mode\":\"joint_velocity\",\"joint\":\"joint_dial\",\"omega_radps\":3.14}]},")
    lines.append("    {\"name\":\"handle_turn\", \"phase_type\":\"control_actuation\", \"t0\":0.90, \"t1\":1.40, \"controls\":[{\"mode\":\"joint_velocity\",\"joint\":\"joint_handle\",\"omega_radps\":3.14}]},")
    lines.append("    {\"name\":\"effect_open\", \"phase_type\":\"effect_motion\", \"t0\":1.52, \"t1\":2.92, \"controls\":[{\"mode\":\"joint_position\",\"joint\":\"joint_door\",\"q_start_rad\":0.0,\"q_target_expr\":\"upper_limit\",\"curve\":\"ease_out\"}]}")
    lines.append("  ],")
    lines.append("  \"timing_checks\": {\"control_onset_time\":0.90, \"control_peak_time\":1.40, \"effect_start_time\":1.52, \"enforced_delay_s\":0.12}")
    lines.append("}")
    lines.append("- The example above is valid because the dial/handle are not assumed to spring back automatically.")
    lines.append("3) Invalid duplicate release pattern (do NOT do this):")
    lines.append("{")
    lines.append("  \"timeline\": [")
    lines.append("    {\"name\":\"control_return_1\", \"phase_type\":\"control_release\", \"controls\":[{\"mode\":\"spring_return\",\"joint\":\"joint_control\"}]},")
    lines.append("    {\"name\":\"control_return_2\", \"phase_type\":\"control_release\", \"controls\":[{\"mode\":\"spring_return\",\"joint\":\"joint_control\"}]}")
    lines.append("  ]")
    lines.append("}")
    lines.append("- The second release above is invalid unless another control_actuation of joint_control occurs between them.")
    lines.append("")
    lines.append("Schema (must match exactly):")
    lines.append("{")
    lines.append('\"meta\": {\"fps\": 30, \"duration_s\": 0.0},')
    lines.append('\"physics\": {...},')
    lines.append('\"timeline\": [')
    lines.append('  {\"name\": \"...\", \"phase_type\": \"control_actuation|control_release|causal_latency|effect_motion|settle\", \"t0\": 0.0, \"t1\": 0.0, \"controls\": [ ... ]}')
    lines.append('],')
    lines.append('\"timing_checks\": {\"control_onset_time\": 0.0, \"control_peak_time\": 0.0, \"effect_start_time\": 0.0, \"enforced_delay_s\": 0.0}')
    lines.append("}")
    lines.append("")
    lines.append("User action text:")
    if user_prompt:
        lines.append(user_prompt.strip())
    else:
        lines.append("<PASTE USER ACTION HERE>")
    lines.append("")
    if scale_context:
        lines.append("Scale context JSON:")
        lines.append(json.dumps(scale_context, ensure_ascii=False, indent=2))
        lines.append("")
    lines.append("URDF joint summary:")
    if joints:
        for joint in joints:
            lines.append(format_joint_summary(joint))
    else:
        lines.append("- None")
    lines.append("")
    lines.append("VLM causal JSON:")
    lines.append("<PASTE VLM JSON HERE>")

    out_path = Path(out_dir) / "llm_plan_prompt.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def format_joint_summary(joint):
    axis = joint.get("axis")
    axis_str = "[" + ", ".join(f"{v:.4f}" for v in axis) + "]" if axis else "None"
    limit = joint.get("limit") or {}
    limit_str = "None"
    if limit:
        limit_str = f"lower={limit.get('lower')}, upper={limit.get('upper')}, effort={limit.get('effort')}, velocity={limit.get('velocity')}"
    origin = joint.get("origin") or {}
    origin_str = "None"
    if origin:
        origin_str = f"xyz={origin.get('xyz')}, rpy={origin.get('rpy')}"
    return f"- {joint.get('name')} | type={joint.get('type')} | parent={joint.get('parent')} | child={joint.get('child')} | axis={axis_str} | limit={limit_str} | origin={origin_str}"


def write_llm_prompt(asset_name, out_dir, joints):
    lines = []
    # lines.append(f"Asset: {asset_name}")
    # lines.append("")
    lines.append("Example text actions:")
    lines.append('- Example A: "press the visible control to trigger the mechanism"')
    lines.append('- Example B: "move the object forward by directly manipulating its body or handle"')
    lines.append("")
    lines.append("URDF joint summary:")
    if joints:
        for joint in joints:
            lines.append(format_joint_summary(joint))
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Please output STRICT JSON only. No extra text.")
    lines.append("Output schema: CausalSpec JSON with fields:")
    lines.append('{"action_grounding": {"target_link": "...", "primitive": "...", "magnitude": 0.0},')
    lines.append(' "modes": ["..."],')
    lines.append(' "joint_targets": [{"joint": "...", "target": "alpha*upper_limit"}],')
    lines.append(' "coupling_rules": ["if action/mode then joint change"]}')

    out_path = Path(out_dir) / "llm_prompt.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate overlay images and prompts from URDF assets")
    parser.add_argument("--assets_root", required=True, help="Root folder containing asset subfolders")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--resolution", type=int, nargs=2, default=[800, 600], help="Width height")
    parser.add_argument("--asset_name", default=None, help="Only process a single asset subfolder name")
    parser.add_argument("--action_text", default=None, help="Override user action text instead of asset user_prompt.txt")
    args = parser.parse_args()

    assets_root = Path(args.assets_root).absolute()
    out_root = Path(args.out).absolute()

    if not assets_root.exists():
        print(f"Assets root not found: {assets_root}")
        sys.exit(1)

    print("Reference renderer: blender")

    asset_dirs = [p for p in assets_root.iterdir() if p.is_dir()]
    if args.asset_name:
        asset_dirs = [p for p in asset_dirs if p.name == args.asset_name]
    if not asset_dirs:
        print("No asset directories found.")
        sys.exit(1)

    for asset_dir in sorted(asset_dirs):
        asset_name = asset_dir.name
        urdf_path = find_urdf(asset_dir)
        if urdf_path is None:
            print(f"[WARN] No URDF found under {asset_dir}")
            continue
        print(f"Asset: {asset_name}")
        print(f"URDF: {urdf_path}")

        links, joints = parse_urdf(urdf_path)
        print(f"Links: {len(links)} | Joints: {len(joints)}")

        link_transforms = compute_link_transforms(links, joints)
        link_meshes = load_link_meshes(links, urdf_path.parent, link_transforms)
        all_meshes, link_names = build_mesh_list(link_meshes)
        if not all_meshes:
            print(f"[WARN] No meshes loaded for {asset_name}")
            continue

        out_images = out_root / asset_name / "images"
        out_prompts = out_root / asset_name / "prompts"
        out_images.mkdir(parents=True, exist_ok=True)
        out_prompts.mkdir(parents=True, exist_ok=True)

        link_list = list(link_meshes.keys())
        # Only links with visual meshes can be labeled/seen in overlays.
        visual_links = [ln for ln in link_list if link_meshes.get(ln)]
        link_color_map = build_distinct_link_color_map(visual_links)
        # Use compact canonical labels on images (e.g., 15_12, 15).
        link_ids = build_compact_image_labels(visual_links, joints)
        label_texts = dict(link_ids)
        movable_links = _movable_links_from_joints(joints)
        movable_visual_links = [ln for ln in visual_links if ln in movable_links]
        line_like_visual_links = _line_like_visual_links(link_meshes, visual_links)
        static_big_link = _select_primary_static_link(visual_links, movable_links, link_meshes, joints)
        ref_glb_path = find_reference_glb_path(asset_dir)
        reuse_rendered_images = can_reuse_rendered_images(
            out_root / asset_name,
            args.resolution,
            len(VIEW_ANGLES),
            reference_glb_path=ref_glb_path,
        )
        scale_context = scu.build_scale_context(asset_name, link_meshes, joints=joints, glb_path=ref_glb_path)
        scu.save_scale_context(out_root / asset_name / "scale_context.json", scale_context)
        if reuse_rendered_images:
            print(f"[INFO] Reusing cached images for {asset_name} at {tuple(int(x) for x in args.resolution)}.")
        else:
            assembled_center, assembled_radius = compute_scene_bounds(link_meshes)
            points_by_link = sample_link_points(link_meshes, visual_links)

            reference_scene = load_reference_scene(asset_dir, links, urdf_path.parent, link_transforms)
            if reference_scene is None:
                reference_scene = trimesh.Scene()
                for mesh in all_meshes:
                    reference_scene.add_geometry(mesh.copy())
            reference_meshes_by_link = build_reference_meshes_by_link(reference_scene, visual_links)
            ref_bbox_points_by_link = build_reference_points_by_link(reference_scene, link_meshes, visual_links)
            if ref_bbox_points_by_link is None:
                if isinstance(reference_meshes_by_link, dict):
                    ref_bbox_points_by_link = collect_link_bbox_points(reference_meshes_by_link, visual_links)
                else:
                    bbox_points_by_link = collect_link_bbox_points(link_meshes, visual_links)
                    ref_bbox_points_by_link = dict(bbox_points_by_link)
            urdf_reference_scene = _build_reference_scene_from_urdf(links, urdf_path.parent, link_transforms)
            reference_cameras = [
                compute_camera(assembled_center, assembled_radius, azim_deg=azim, elev_deg=elev)
                for azim, elev in VIEW_ANGLES
            ]
            software_ref_imgs = None
            blender_ref_imgs = None
            blender_source_scene = None
            use_reference_glb = ref_glb_path is not None and Path(ref_glb_path).exists()
            if not use_reference_glb and urdf_reference_scene is not None and scene_has_effective_textures(urdf_reference_scene):
                blender_source_scene = urdf_reference_scene
            if blender_source_scene is not None or use_reference_glb:
                blender_views = []
                for view_idx, (azim, elev) in enumerate(VIEW_ANGLES, start=1):
                    camera_ref = compute_camera(assembled_center, assembled_radius, azim_deg=azim, elev_deg=elev)
                    eye, target, up = camera_ref
                    blender_views.append(
                        {
                            "id": f"V{view_idx}",
                            "eye": np.asarray(eye, dtype=float).tolist(),
                            "target": np.asarray(target, dtype=float).tolist(),
                            "up": np.asarray(up, dtype=float).tolist(),
                        }
                    )
                try:
                    t_blender_ref = time.perf_counter()
                    source_label = str(Path(ref_glb_path).name) if use_reference_glb else "scene_export"
                    print(f"[INFO] Starting Blender reference render from {source_label} with {len(blender_views)} views.")
                    if use_reference_glb:
                        blender_ref_imgs = br.render_views_from_glb(
                            ref_glb_path,
                            blender_views,
                            tuple(int(x) for x in args.resolution),
                            fov_deg=50.0,
                            frame_idx=0,
                            keep_animation=True,
                        )
                    else:
                        blender_ref_imgs = br.render_views_from_scene(
                            blender_source_scene,
                            blender_views,
                            tuple(int(x) for x in args.resolution),
                            fov_deg=50.0,
                        )
                    blender_ref_imgs = [enhance_textured_image(img) for img in blender_ref_imgs]
                    dark_count = sum(1 for img in blender_ref_imgs if is_reference_image_too_dark(img))
                    if dark_count > 0:
                        print(
                            f"[WARN] Blender reference outputs include {dark_count}/{len(blender_ref_imgs)} dark views; "
                            "keeping Blender batch to preserve textured references."
                        )
                    print(f"[INFO] Reference renders via Blender Cycles in {time.perf_counter() - t_blender_ref:.2f}s.")
                except Exception as exc:
                    blender_ref_imgs = None
                    print(f"[WARN] Blender reference render failed ({exc}); using non-pyrender software fallback.")

            if blender_ref_imgs is None and reference_scene is not None and len(reference_scene.geometry) > 0:
                software_ref_imgs = [
                    render_reference_textured(reference_scene, cam, args.resolution)
                    for cam in reference_cameras
                ]

            batch_backend_decision = decide_reference_backend_for_batch(blender_ref_imgs, software_ref_imgs)
            batch_backend = normalize_reference_backend_name(batch_backend_decision.get("reference_backend"), default="blender")
            if batch_backend == "software" and software_ref_imgs is not None:
                reason = str(batch_backend_decision.get("reason") or "blender_fallback")
                dark_views = batch_backend_decision.get("dark_view_indices") or []
                washed_out_views = batch_backend_decision.get("washed_out_view_indices") or []
                detail_parts = []
                if dark_views:
                    detail_parts.append(f"dark_views={dark_views}")
                    if washed_out_views:
                        detail_parts.append(f"washed_out_views={washed_out_views}")
                detail_suffix = f" ({', '.join(detail_parts)})" if detail_parts else ""
                print(f"[WARN] Using non-pyrender software reference batch for {asset_name}: {reason}{detail_suffix}.")
            save_reference_backend_decision(
                out_root / asset_name,
                {
                    "reference_backend": batch_backend,
                    "reason": str(batch_backend_decision.get("reason") or ""),
                    "dark_view_indices": batch_backend_decision.get("dark_view_indices") or [],
                    "washed_out_view_indices": batch_backend_decision.get("washed_out_view_indices") or [],
                    "blender_image_count": int(batch_backend_decision.get("blender_image_count") or 0),
                    "fallback_image_count": int(batch_backend_decision.get("fallback_image_count") or 0),
                    "source": "gen_overlays_and_prompts",
                },
            )
            selected_ref_imgs = blender_ref_imgs if blender_ref_imgs is not None else software_ref_imgs
            preprocess_box_mode = PREPROCESS_REFERENCE_BOX_MODE if PREPROCESS_REFERENCE_BOX_MODE in {"points", "raster"} else "points"
            vis_resolution = _scaled_visibility_resolution(args.resolution)
            print(f"[INFO] Preprocess reference box mode: {preprocess_box_mode}")
            if tuple(int(x) for x in vis_resolution) != tuple(int(x) for x in args.resolution):
                print(f"[INFO] Preprocess visibility raster resolution: {vis_resolution[0]}x{vis_resolution[1]}")
            globally_exposed_links = {str(static_big_link)} if static_big_link is not None else set()
            overlay_scan_stats = {
                str(ln): {
                    "visible_views": 0,
                    "total_pixels": 0,
                    "min_fill": 1.0,
                    "all_inside": True,
                }
                for ln in visual_links
            }
            try:
                for camera_scan in reference_cameras:
                    _scan_img, _scan_labels, scan_visible_pixels = render_overlay_points(
                        points_by_link,
                        link_color_map,
                        camera_scan,
                        args.resolution,
                        return_visible_pixels=True,
                    )
                    static_pts = np.asarray(
                        (scan_visible_pixels or {}).get(str(static_big_link), np.zeros((0, 2), dtype=np.int32)),
                        dtype=np.int32,
                    )
                    static_box = None
                    inner_margin = 0
                    if static_pts.shape[0] > 0:
                        sx0, sy0 = static_pts.min(axis=0).tolist()
                        sx1, sy1 = static_pts.max(axis=0).tolist()
                        static_box = (int(sx0), int(sy0), int(sx1), int(sy1))
                        inner_margin = max(6, int(round(0.05 * float(min(max(1, sx1 - sx0 + 1), max(1, sy1 - sy0 + 1))))))
                    for ln in visual_links:
                        pts = np.asarray((scan_visible_pixels or {}).get(str(ln), np.zeros((0, 2), dtype=np.int32)), dtype=np.int32)
                        if pts.shape[0] > 0:
                            scan_stat = overlay_scan_stats.setdefault(
                                str(ln),
                                {"visible_views": 0, "total_pixels": 0, "min_fill": 1.0, "all_inside": True},
                            )
                            scan_stat["visible_views"] = int(scan_stat.get("visible_views") or 0) + 1
                            scan_stat["total_pixels"] = int(scan_stat.get("total_pixels") or 0) + int(pts.shape[0])
                            x0, y0 = pts.min(axis=0).tolist()
                            x1, y1 = pts.max(axis=0).tolist()
                            box_area = max(1, (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1))
                            fill = float(pts.shape[0]) / float(box_area)
                            scan_stat["min_fill"] = min(float(scan_stat.get("min_fill") or 1.0), float(fill))
                            inside = False
                            if static_box is not None and str(ln) != str(static_big_link):
                                sx0, sy0, sx1, sy1 = static_box
                                inside = (
                                    int(x0) >= int(sx0) + int(inner_margin)
                                    and int(y0) >= int(sy0) + int(inner_margin)
                                    and int(x1) <= int(sx1) - int(inner_margin)
                                    and int(y1) <= int(sy1) - int(inner_margin)
                                )
                            scan_stat["all_inside"] = bool(scan_stat.get("all_inside", True)) and bool(inside)
                            globally_exposed_links.add(str(ln))
            except Exception:
                globally_exposed_links = {str(ln) for ln in visual_links}
            if not globally_exposed_links:
                globally_exposed_links = {str(ln) for ln in visual_links}
            interior_enclosed_links = []
            line_like_set = {str(ln) for ln in (line_like_visual_links or [])}
            for ln in visual_links:
                if str(ln) == str(static_big_link):
                    continue
                stat = overlay_scan_stats.get(str(ln)) or {}
                visible_views = int(stat.get("visible_views") or 0)
                avg_pixels = float(stat.get("total_pixels") or 0.0) / float(max(1, visible_views))
                min_fill = float(stat.get("min_fill") or 0.0)
                all_inside = bool(stat.get("all_inside", False))
                if (
                    visible_views >= 2
                    and all_inside
                    and min_fill >= 0.15
                    and avg_pixels >= 1000.0
                    and str(ln) not in line_like_set
                ):
                    interior_enclosed_links.append(str(ln))
            if interior_enclosed_links:
                globally_exposed_links = {str(ln) for ln in globally_exposed_links if str(ln) not in set(interior_enclosed_links)}
                print(f"[INFO] Reference omits enclosed internal links: {interior_enclosed_links}")
            hidden_internal_links = [str(ln) for ln in visual_links if str(ln) not in globally_exposed_links]
            if hidden_internal_links:
                print(f"[INFO] Omit fully hidden internal links across all views: {hidden_internal_links}")
            reference_retained_links = [str(ln) for ln in visual_links if str(ln) in globally_exposed_links]
            reference_link_color_map = {str(ln): link_color_map[str(ln)] for ln in reference_retained_links if str(ln) in link_color_map}
            reference_link_ids = {str(ln): link_ids[str(ln)] for ln in reference_retained_links if str(ln) in link_ids}
            reference_label_texts = dict(reference_link_ids)

            for idx, (azim, elev) in enumerate(VIEW_ANGLES, start=1):
                camera_overlay = compute_camera(assembled_center, assembled_radius, azim_deg=azim, elev_deg=elev)
                overlay_img, label_positions = render_overlay_points(
                    points_by_link, link_color_map, camera_overlay, args.resolution
                )
                overlay_boxes = build_structured_overlay_boxes(
                    points_by_link,
                    camera_overlay,
                    args.resolution,
                    visual_links=visual_links,
                    movable_visual_links=movable_visual_links,
                    static_big_link=static_big_link,
                )
                for link_name, box in (overlay_boxes or {}).items():
                    if link_name in link_color_map:
                        draw_bbox_outline(overlay_img, box, link_color_map[link_name], thickness=2)
                axis_len = max(assembled_radius * 0.6, 0.05)
                draw_axes_overlay(overlay_img, camera_overlay, args.resolution, assembled_center, axis_len, scale=2, corner=True)
                if overlay_boxes:
                    label_positions = adjust_caption_positions_from_boxes(overlay_boxes, label_texts, args.resolution, scale=2)
                else:
                    label_positions = adjust_label_positions(label_positions, label_texts, args.resolution, scale=2)
                for link_name, pos in label_positions.items():
                    draw_label(overlay_img, pos[0], pos[1], label_texts[link_name], link_color_map[link_name], scale=2)
                overlay_path = out_images / f"overlay_view_{idx:02d}.png"
                _save_image(overlay_img, overlay_path)
                print(f"Wrote {overlay_path}")

                camera_ref = camera_overlay
                if selected_ref_imgs is not None and idx - 1 < len(selected_ref_imgs):
                    ref_img = np.array(selected_ref_imgs[idx - 1], copy=True)
                else:
                    ref_img = (
                        np.array(software_ref_imgs[idx - 1], copy=True)
                        if software_ref_imgs is not None and idx - 1 < len(software_ref_imgs)
                        else np.array(overlay_img, copy=True)
                    )
                if preprocess_box_mode == "points":
                    visible_ref_boxes, visible_ref_stats = project_visible_link_boxes(
                        points_by_link,
                        camera_ref,
                        args.resolution,
                        return_stats=True,
                    )
                    projected_ref_boxes = project_link_boxes(
                        points_by_link,
                        camera_ref,
                        args.resolution,
                    )
                else:
                    visible_ref_boxes, visible_ref_stats = project_visible_link_boxes(
                        ref_bbox_points_by_link,
                        camera_ref,
                        args.resolution,
                        return_stats=True,
                    )
                    projected_ref_boxes = project_link_boxes(
                        ref_bbox_points_by_link,
                        camera_ref,
                        args.resolution,
                    )
                raster_boxes = {}
                raster_stats = {}
                reference_raster_meshes = reference_meshes_by_link if isinstance(reference_meshes_by_link, dict) and reference_meshes_by_link else link_meshes
                reference_visibility_ratio_stats = {
                    str(k): {
                        "visible_pixels": int((v or {}).get("visible_point_count") or 0),
                        "total_projected_pixels": int((v or {}).get("projected_point_count") or 0),
                        "tie_surface_visible_pixels": 0,
                        "surface_visible_ratio": float((v or {}).get("visible_ratio") or 0.0),
                        "visible_ratio": float((v or {}).get("visible_ratio") or 0.0),
                    }
                    for k, v in (visible_ref_stats or {}).items()
                }
                try:
                    all_raster_boxes, all_raster_pixel_stats, all_scene_depth = project_visible_link_boxes_rasterized(
                        reference_raster_meshes,
                        visual_links,
                        camera_ref,
                        vis_resolution,
                        return_stats=True,
                        return_scene_depth=True,
                    )
                    reference_visibility_ratio_stats = reference_visible_ratios_by_link_rasterized(
                        reference_raster_meshes,
                        visual_links,
                        camera_ref,
                        vis_resolution,
                        visible_pixel_stats=all_raster_pixel_stats,
                        points_by_link=ref_bbox_points_by_link,
                        scene_depth=all_scene_depth,
                    )
                except Exception:
                    all_raster_boxes = {}
                    all_raster_pixel_stats = {}
                if preprocess_box_mode == "raster":
                    scene_depth = None
                    try:
                        raster_boxes, raster_stats, scene_depth = project_visible_link_boxes_rasterized(
                            reference_raster_meshes,
                            visual_links,
                            camera_ref,
                            vis_resolution,
                            return_stats=True,
                            return_scene_depth=True,
                        )
                    except Exception:
                        raster_boxes = {}
                        raster_stats = {}
                        scene_depth = None
                    try:
                        reference_visibility_ratio_stats = reference_visible_ratios_by_link_rasterized(
                            reference_raster_meshes,
                            visual_links,
                            camera_ref,
                            vis_resolution,
                            visible_pixel_stats=raster_stats,
                            points_by_link=ref_bbox_points_by_link,
                            scene_depth=scene_depth,
                        )
                    except Exception:
                        reference_visibility_ratio_stats = {
                            str(k): {
                                "visible_pixels": int((v or {}).get("visible_point_count") or 0),
                                "total_projected_pixels": int((v or {}).get("projected_point_count") or 0),
                                "tie_surface_visible_pixels": 0,
                                "surface_visible_ratio": float((v or {}).get("visible_ratio") or 0.0),
                                "visible_ratio": float((v or {}).get("visible_ratio") or 0.0),
                            }
                            for k, v in (visible_ref_stats or {}).items()
                        }
                # Reference-view bbox geometry mirrors the overlay for the same camera.
                ref_boxes = {
                    str(ln): box
                    for ln, box in dict(overlay_boxes or {}).items()
                    if str(ln) in globally_exposed_links
                }
                if not ref_boxes and globally_exposed_links:
                    ref_boxes = {
                        str(ln): box
                        for ln, box in dict(projected_ref_boxes or visible_ref_boxes or {}).items()
                        if str(ln) in globally_exposed_links
                    }
                for link_name, box in ref_boxes.items():
                    draw_bbox_outline(ref_img, box, reference_link_color_map[link_name], thickness=2)
                ref_label_positions = adjust_caption_positions_from_boxes(ref_boxes, reference_label_texts, args.resolution, scale=2)
                ref_axis_len = max(assembled_radius * 0.6, 0.05)
                draw_axes_overlay(ref_img, camera_ref, args.resolution, assembled_center, ref_axis_len, scale=2, corner=True)
                for link_name, pos in ref_label_positions.items():
                    draw_label(ref_img, pos[0], pos[1], reference_label_texts[link_name], reference_link_color_map[link_name], scale=2)
                ref_path = out_images / f"reference_view_{idx:02d}.png"
                _save_image(ref_img, ref_path)
                print(f"Wrote {ref_path}")

            save_image_cache_meta(
                out_root / asset_name,
                {
                    "version": int(IMAGE_CACHE_VERSION),
                    "resolution": [int(args.resolution[0]), int(args.resolution[1])],
                    "view_count": int(len(VIEW_ANGLES)),
                    **_reference_glb_cache_stamp(ref_glb_path),
                },
            )
            print("Per-link highlight images disabled (overlay-only mode).")

        links_info = gather_links_root(links, joints)

        image_entries = []
        for idx in range(1, len(VIEW_ANGLES) + 1):
            image_entries.append(
                {
                    "path": f"images/overlay_view_{idx:02d}.png",
                    "desc": "Overlay with colored parts and compact image labels; includes the full grounded link inventory for this view.",
                }
            )
        for idx in range(1, len(VIEW_ANGLES) + 1):
            image_entries.append(
                {
                    "path": f"images/reference_view_{idx:02d}.png",
                    "desc": "Reference assembled mesh view (textures when available) with axis; enclosed internal links may be omitted to avoid confusion with outer-surface parts.",
                }
            )

        user_prompt_path = asset_dir / "user_prompt.txt"
        user_prompt = None
        if args.action_text:
            user_prompt = args.action_text
        elif user_prompt_path.exists():
            user_prompt = user_prompt_path.read_text(encoding="utf-8")

        write_vlm_prompt(
            asset_name,
            out_prompts,
            links_info,
            image_entries,
            joints,
            user_prompt=user_prompt,
            link_ids=link_ids,
            link_order=link_list,
            label_texts=label_texts,
            scale_context=scale_context,
        )
        write_llm_plan_prompt(asset_name, out_prompts, joints, user_prompt=user_prompt, scale_context=scale_context)

        print(f"Wrote prompts to {out_prompts}")
        print("")


if __name__ == "__main__":
    main()
