#!/usr/bin/env python3
import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import shutil
import zlib
import copy
import re

import numpy as np

import blender_render as br
import torch_accel as tacc

try:
    import trimesh
except Exception as exc:
    raise SystemExit(f"Failed to import trimesh: {exc}")

try:
    import pyrender  # type: ignore
    _HAS_PYRENDER = True
except Exception:
    _HAS_PYRENDER = False

try:
    import imageio.v2 as imageio  # type: ignore
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False

try:
    import imageio_ffmpeg  # type: ignore
    _HAS_IMAGEIO_FFMPEG = True
except Exception:
    _HAS_IMAGEIO_FFMPEG = False

try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def setup_pyrender_headless():
    if os.environ.get("PYOPENGL_PLATFORM") is None:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
    try:
        import pyglet  # type: ignore
        pyglet.options["headless"] = True
    except Exception:
        pass


def _write_png(path, array):
    if array.dtype != np.uint8:
        array = array.astype(np.uint8)
    height, width, _ = array.shape
    raw = b"".join(b"\x00" + array[i].tobytes() for i in range(height))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    with open(path, "wb") as f:
        f.write(signature)
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def _parse_floats(text, default=None):
    if text is None:
        return default
    parts = [p for p in text.replace(",", " ").split() if p]
    if not parts:
        return default
    out = []
    for v in parts:
        token = str(v).strip().lower()
        if token in {"none", "null", "nan"}:
            out.append(0.0)
        else:
            out.append(float(v))
    return out


def _rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    return trimesh.transformations.euler_matrix(roll, pitch, yaw, axes="sxyz")


def _origin_to_matrix(origin_xyz, origin_rpy):
    transform = np.eye(4)
    if origin_rpy is not None:
        transform = _rpy_to_matrix(origin_rpy)
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
    basename = mesh_path.name
    for found in urdf_dir.rglob(basename):
        return found.resolve()
    return candidate


def _load_obj_simple(path: Path):
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
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


def _scene_to_mesh_simple(scene: trimesh.Scene) -> trimesh.Trimesh:
    meshes = []
    for geom in scene.geometry.values():
        if geom.vertices.size == 0 or geom.faces.size == 0:
            continue
        meshes.append(
            trimesh.Trimesh(
                vertices=geom.vertices.copy(),
                faces=geom.faces.copy(),
                process=False,
            )
        )
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes)


def _scene_geoms_with_transforms(scene: trimesh.Scene):
    geoms = []
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        mesh = geom.copy()
        mesh.apply_transform(transform)
        geoms.append(mesh)
    return geoms


def _load_mesh(path: Path, textured: bool):
    force_mode = None if textured else "mesh"
    try:
        return trimesh.load(path, force=force_mode, skip_materials=not textured)
    except Exception as exc:
        if "PIL" in str(exc) and path.suffix.lower() == ".obj":
            return _load_obj_simple(path)
        raise


def parse_urdf(urdf_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()

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
            mesh_tag = geom.find("mesh")
            if mesh_tag is None:
                continue
            filename = mesh_tag.get("filename") or mesh_tag.get("file")
            scale = _parse_floats(mesh_tag.get("scale"), default=[1.0, 1.0, 1.0])
            origin = visual.find("origin")
            origin_xyz = _parse_floats(origin.get("xyz")) if origin is not None else None
            origin_rpy = _parse_floats(origin.get("rpy")) if origin is not None else None
            visuals.append(
                {
                    "filename": filename,
                    "scale": scale,
                    "origin_xyz": origin_xyz,
                    "origin_rpy": origin_rpy,
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


def load_link_meshes(links, urdf_dir, textured=False):
    link_meshes = {}
    for link_name, visuals in links.items():
        meshes = []
        for visual in visuals:
            mesh_path = _resolve_mesh_path(visual["filename"], urdf_dir)
            if mesh_path is None or not mesh_path.exists():
                continue
            try:
                mesh = _load_mesh(mesh_path, textured=textured)
            except Exception as exc:
                print(f"[WARN] Failed to load mesh {mesh_path}: {exc}")
                continue
            scale = visual["scale"] or [1.0, 1.0, 1.0]
            scale_mat = np.eye(4)
            scale_mat[0, 0] = scale[0]
            scale_mat[1, 1] = scale[1]
            scale_mat[2, 2] = scale[2]
            origin_mat = _origin_to_matrix(visual.get("origin_xyz"), visual.get("origin_rpy"))
            if isinstance(mesh, trimesh.Scene):
                if textured:
                    for g in _scene_geoms_with_transforms(mesh):
                        gg = g.copy()
                        gg.apply_transform(scale_mat)
                        gg.apply_transform(origin_mat)
                        meshes.append(gg)
                    continue
                mesh = _scene_to_mesh_simple(mesh)
            else:
                mesh = mesh.copy()
            mesh.apply_transform(scale_mat)
            mesh.apply_transform(origin_mat)
            meshes.append(mesh)
        link_meshes[link_name] = meshes
    return link_meshes


def compute_link_transforms(links, joints, joint_positions, base_tf=None):
    joint_tree = {}
    for joint in joints:
        parent = joint.get("parent")
        child = joint.get("child")
        if not parent or not child:
            continue
        joint_tree.setdefault(parent, []).append(joint)

    child_links = set(j.get("child") for j in joints if j.get("child"))
    root_links = [ln for ln in links.keys() if ln not in child_links]
    if not root_links:
        root_links = list(links.keys())

    link_transforms = {root: np.eye(4) for root in root_links}

    def compute_children(parent_link):
        parent_tf = link_transforms[parent_link]
        for joint in joint_tree.get(parent_link, []):
            origin = joint.get("origin") or {}
            origin_tf = _origin_to_matrix(origin.get("xyz"), origin.get("rpy"))
            motion_tf = np.eye(4)
            q = joint_positions.get(joint.get("name"), 0.0)
            axis = np.array(joint.get("axis") or [0.0, 0.0, 1.0], dtype=float)
            if np.linalg.norm(axis) > 0:
                axis = axis / np.linalg.norm(axis)
            if joint.get("type") in ("revolute", "continuous"):
                motion_tf = trimesh.transformations.rotation_matrix(q, axis)
            elif joint.get("type") == "prismatic":
                motion_tf[:3, 3] = axis * q
            joint_tf = origin_tf @ motion_tf
            child_link = joint.get("child")
            if child_link:
                link_transforms[child_link] = parent_tf @ joint_tf
                compute_children(child_link)

    for root in root_links:
        link_transforms[root] = np.eye(4)
        compute_children(root)

    if base_tf is not None:
        for ln in link_transforms.keys():
            link_transforms[ln] = base_tf @ link_transforms[ln]

    return link_transforms


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


def compute_link_bbox(link_meshes, link_tf_map):
    info = {}
    for link, meshes in link_meshes.items():
        if not meshes:
            continue
        merged = trimesh.util.concatenate([m.copy() for m in meshes if m.vertices.size > 0])
        if merged.vertices.size == 0:
            continue
        tf = link_tf_map.get(link, np.eye(4))
        merged.apply_transform(tf)
        ext = merged.extents
        center = merged.bounds.mean(axis=0)
        info[link] = (ext, center)
    return info


def compute_scene_node_bbox(scene):
    info = {}
    for node_name in scene.graph.nodes_geometry:
        tf, geom_name = scene.graph[node_name]
        geom = scene.geometry[geom_name]
        mesh = geom.copy()
        mesh.apply_transform(tf)
        ext = mesh.extents
        center = mesh.bounds.mean(axis=0)
        info[node_name] = (ext, center)
    return info


def estimate_similarity_from_mapping(link_meshes, link_tf_map, scene, link_to_nodes):
    src_centers = []
    dst_centers = []
    for link, node_names in (link_to_nodes or {}).items():
        meshes = link_meshes.get(link) or []
        if not meshes:
            continue
        try:
            merged = trimesh.util.concatenate([m.copy() for m in meshes if m.vertices.size > 0])
        except Exception:
            continue
        if merged.vertices.size == 0:
            continue
        merged.apply_transform(link_tf_map.get(link, np.eye(4)))
        src_centers.append(merged.bounds.mean(axis=0))
        nn = node_names[0] if node_names else None
        if nn is None:
            src_centers.pop()
            continue
        try:
            tf, geom_name = scene.graph[nn]
            geom = scene.geometry[geom_name].copy()
            geom.apply_transform(tf)
        except Exception:
            src_centers.pop()
            continue
        dst_centers.append(geom.bounds.mean(axis=0))

    if len(src_centers) < 2:
        return np.eye(4)

    src = np.asarray(src_centers, dtype=float)
    dst = np.asarray(dst_centers, dtype=float)
    T, _scale, _R, _t = tacc.umeyama_similarity(src, dst)
    return T


def _extract_gltf_image_bytes(gltf, image, *, gltf_path: str | None = None) -> bytes | None:
    if image is None:
        return None
    uri = getattr(image, "uri", None)
    if uri:
        uri = str(uri)
        if uri.startswith("data:"):
            try:
                payload = uri.split(",", 1)[1]
                return base64.b64decode(payload)
            except Exception:
                return None
        if gltf_path:
            try:
                return Path(gltf_path).resolve().with_name(uri).read_bytes()
            except Exception:
                return None
    buffer_view_idx = getattr(image, "bufferView", None)
    if buffer_view_idx is None:
        return None
    try:
        bv = gltf.bufferViews[int(buffer_view_idx)]
    except Exception:
        return None
    blob = gltf.binary_blob() or b""
    if not blob:
        return None
    start = int(getattr(bv, "byteOffset", 0) or 0)
    length = int(getattr(bv, "byteLength", 0) or 0)
    end = start + length
    if start < 0 or length <= 0 or end > len(blob):
        return None
    return bytes(blob[start:end])


def _image_mean_luma_from_bytes(data: bytes | None) -> float | None:
    if not _HAS_PIL or not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    except Exception:
        return None
    if arr.size <= 0:
        return None
    luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(np.mean(luma) / 255.0)


def _add_emissive_boost_for_basecolor_textures(gltf, *, gltf_path: str | None = None) -> int:
    try:
        from pygltflib import TextureInfo
    except Exception:
        return 0
    if not getattr(gltf, "materials", None) or not getattr(gltf, "textures", None) or not getattr(gltf, "images", None):
        return 0
    try:
        dark_luma_max = float(os.environ.get("CODEX_GLTF_DARK_EMISSIVE_LUMA_MAX", "0.35"))
    except Exception:
        dark_luma_max = 0.35
    try:
        dark_emissive_strength = float(os.environ.get("CODEX_GLTF_DARK_EMISSIVE_STRENGTH", "0.0"))
    except Exception:
        dark_emissive_strength = 0.0
    try:
        bright_luma_min = float(os.environ.get("CODEX_GLTF_BRIGHT_EMISSIVE_LUMA_MIN", "0.70"))
    except Exception:
        bright_luma_min = 0.70
    try:
        bright_emissive_strength = float(os.environ.get("CODEX_GLTF_BRIGHT_EMISSIVE_STRENGTH", "0.0"))
    except Exception:
        bright_emissive_strength = 0.0
    if dark_emissive_strength <= 0.0 and bright_emissive_strength <= 0.0:
        return 0

    tex_luma_cache: dict[int, float | None] = {}
    boosted = 0
    for mat in gltf.materials:
        if mat is None:
            continue
        pbr = getattr(mat, "pbrMetallicRoughness", None)
        base_tex = getattr(pbr, "baseColorTexture", None) if pbr is not None else None
        tex_idx = getattr(base_tex, "index", None)
        if tex_idx is None:
            continue
        tex_idx = int(tex_idx)
        if tex_idx not in tex_luma_cache:
            luma = None
            try:
                tex = gltf.textures[tex_idx]
                src_idx = getattr(tex, "source", None)
                if src_idx is not None:
                    img = gltf.images[int(src_idx)]
                    luma = _image_mean_luma_from_bytes(_extract_gltf_image_bytes(gltf, img, gltf_path=gltf_path))
            except Exception:
                luma = None
            tex_luma_cache[tex_idx] = luma
        luma = tex_luma_cache.get(tex_idx)
        if luma is None:
            continue
        emissive_strength = 0.0
        if dark_emissive_strength > 0.0 and luma <= dark_luma_max:
            emissive_strength = max(emissive_strength, dark_emissive_strength)
        if bright_emissive_strength > 0.0 and luma >= bright_luma_min:
            emissive_strength = max(emissive_strength, bright_emissive_strength)
        if emissive_strength <= 0.0:
            continue
        if getattr(mat, "emissiveTexture", None) is None:
            mat.emissiveTexture = TextureInfo(index=tex_idx, texCoord=int(getattr(base_tex, "texCoord", 0) or 0))
        prev = list(getattr(mat, "emissiveFactor", None) or [0.0, 0.0, 0.0])
        target = max(float(x) for x in prev + [emissive_strength])
        mat.emissiveFactor = [target, target, target]
        boosted += 1
    return boosted


def _normalize_gltf_rgb(rgb) -> list[float]:
    vals = [float(x) for x in list(rgb or [])[:3]]
    if not vals:
        return [0.0, 0.0, 0.0]
    if max(abs(v) for v in vals) > 1.0:
        vals = [v / 255.0 for v in vals]
    return [float(np.clip(v, 0.0, 1.0)) for v in vals]


def _add_emissive_boost_for_bright_solid_materials(gltf) -> int:
    if not getattr(gltf, "materials", None):
        return 0
    try:
        solid_luma_min = float(os.environ.get("CODEX_GLTF_SOLID_BRIGHT_EMISSIVE_LUMA_MIN", "0.55"))
    except Exception:
        solid_luma_min = 0.55
    try:
        solid_scale = float(os.environ.get("CODEX_GLTF_SOLID_BRIGHT_EMISSIVE_SCALE", "0.0"))
    except Exception:
        solid_scale = 0.0
    if solid_luma_min <= 0.0 or solid_scale <= 0.0:
        return 0

    boosted = 0
    for mat in gltf.materials:
        if mat is None:
            continue
        pbr = getattr(mat, "pbrMetallicRoughness", None)
        if pbr is None or getattr(pbr, "baseColorTexture", None) is not None:
            continue
        base_rgb = _normalize_gltf_rgb(getattr(pbr, "baseColorFactor", None) or [1.0, 1.0, 1.0, 1.0])
        luma = 0.2126 * base_rgb[0] + 0.7152 * base_rgb[1] + 0.0722 * base_rgb[2]
        if luma < solid_luma_min:
            continue
        prev = _normalize_gltf_rgb(getattr(mat, "emissiveFactor", None) or [0.0, 0.0, 0.0])
        boosted_rgb = [float(np.clip(c * solid_scale, 0.0, 1.0)) for c in base_rgb]
        mat.emissiveFactor = [max(prev[i], boosted_rgb[i]) for i in range(3)]
        boosted += 1
    return boosted


def match_links_to_nodes(link_bbox, node_bbox):
    # greedy matching by center distance (with light extents penalty)
    remaining_nodes = set(node_bbox.keys())
    mapping = {}
    for link, (ext, center) in link_bbox.items():
        best = None
        best_score = None
        for node in remaining_nodes:
            n_ext, n_center = node_bbox[node]
            ext_score = np.linalg.norm((ext - n_ext) / (ext + 1e-6))
            center_score = np.linalg.norm(center - n_center)
            score = center_score + 0.05 * ext_score
            if best_score is None or score < best_score:
                best_score = score
                best = node
        if best is not None:
            mapping[link] = [best]
            remaining_nodes.remove(best)
    return mapping


def _extract_part_node_index(node_name):
    if not isinstance(node_name, str):
        return None
    m = re.match(r"^part_node_(\d+)(?:$|[^0-9].*)", node_name)
    if not m:
        return None
    return int(m.group(1))


def match_links_to_nodes_particulate_by_order(links, link_meshes, scene):
    """
    For Particulate-exported animated GLBs, nodes are named part_node_<i> and are
    emitted in the same mesh_parts order used to export the URDF links. Since the
    URDF exports links in the same loop over unique_part_ids, we can map mesh links
    in URDF declaration order to part_node_0..N-1 directly (more stable than bbox).
    """
    if not isinstance(scene, trimesh.Scene):
        return None
    node_names = list(scene.graph.nodes_geometry)
    part_groups = {}
    for n in node_names:
        idx = _extract_part_node_index(n)
        if idx is None:
            continue
        part_groups.setdefault(idx, []).append(n)
    if not part_groups:
        return None
    got = sorted(part_groups.keys())
    expected = list(range(len(got)))
    if got != expected:
        return None

    mesh_links_in_order = [ln for ln in links.keys() if link_meshes.get(ln)]
    if len(mesh_links_in_order) != len(part_groups):
        return None

    mapping = {}
    for i, link_name in enumerate(mesh_links_in_order):
        mapping[link_name] = sorted(part_groups[i])
    return mapping


def compute_mapping_alignment_metrics(link_meshes, link_tf_map, scene, link_to_nodes):
    max_center_delta = 0.0
    max_scale_err = 0.0
    matched = 0
    for link, node_names in (link_to_nodes or {}).items():
        meshes = link_meshes.get(link) or []
        if not meshes or not node_names:
            continue
        try:
            link_mesh = trimesh.util.concatenate([m.copy() for m in meshes if m.vertices.size > 0])
        except Exception:
            continue
        if link_mesh.vertices.size == 0:
            continue
        link_mesh.apply_transform(link_tf_map.get(link, np.eye(4)))
        node_world_meshes = []
        for node_name in node_names:
            try:
                tf, geom_name = scene.graph[node_name]
                node_mesh = scene.geometry[geom_name].copy()
                node_mesh.apply_transform(tf)
                node_world_meshes.append(node_mesh)
            except Exception:
                continue
        if not node_world_meshes:
            continue
        try:
            node_mesh = trimesh.util.concatenate(node_world_meshes)
        except Exception:
            node_mesh = node_world_meshes[0]
        lc = link_mesh.bounds.mean(axis=0)
        nc = node_mesh.bounds.mean(axis=0)
        max_center_delta = max(max_center_delta, float(np.linalg.norm(nc - lc)))
        ratio = node_mesh.extents / (link_mesh.extents + 1e-9)
        max_scale_err = max(max_scale_err, float(np.max(np.abs(ratio - 1.0))))
        matched += 1
    return {"matched_links": matched, "max_center_delta": max_center_delta, "max_scale_err": max_scale_err}


def compute_camera(centroid, radius, azim_deg=45.0, elev_deg=25.0):
    azim = math.radians(azim_deg)
    elev = math.radians(elev_deg)
    dist = radius * 2.8
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


def render_frame_pyrender(link_meshes, link_transforms, colors, camera, resolution):
    setup_pyrender_headless()
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[1.0, 1.0, 1.0])
    for link_name, meshes in link_meshes.items():
        for mesh in meshes:
            mesh = mesh.copy()
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=colors[link_name], metallicFactor=0.0, roughnessFactor=1.0
            )
            try:
                pyr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
            except Exception:
                continue
            pose = link_transforms.get(link_name, np.eye(4))
            scene.add(pyr_mesh, pose=pose)

    eye, target, up = camera
    camera_pose = camera_pose_from_lookat(eye, target, up)
    camera_node = pyrender.PerspectiveCamera(yfov=np.deg2rad(45.0))
    scene.add(camera_node, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()
    return color[:, :, :3]


def export_mesh_sequence(out_dir, frames, link_meshes, links, joints):
    mesh_dir = Path(out_dir) / "animated_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for idx, (joint_pos, base_tf) in enumerate(frames):
        link_transforms = compute_link_transforms(links, joints, joint_pos, base_tf=base_tf)
        scene = trimesh.Scene()
        for link_name, meshes in link_meshes.items():
            for mesh in meshes:
                mesh = mesh.copy()
                transform = link_transforms.get(link_name, np.eye(4))
                mesh.apply_transform(transform)
                scene.add_geometry(mesh)
        out_path = mesh_dir / f"frame_{idx:04d}.glb"
        scene.export(out_path)


def get_ffmpeg_path():
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    if _HAS_IMAGEIO_FFMPEG:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def matrix_to_trs(mat):
    translation, rotation, _scale = matrix_to_trs_scale(mat)
    return translation, rotation


def matrix_to_trs_scale(mat):
    translation = mat[:3, 3]
    linear = np.array(mat[:3, :3], dtype=float)
    sx = float(np.linalg.norm(linear[:, 0]))
    sy = float(np.linalg.norm(linear[:, 1]))
    sz = float(np.linalg.norm(linear[:, 2]))
    scale = np.array([sx, sy, sz], dtype=float)
    scale_safe = scale.copy()
    scale_safe[scale_safe < 1e-12] = 1.0
    rot3 = linear / scale_safe[np.newaxis, :]
    if np.linalg.det(rot3) < 0:
        # Keep right-handed rotation; fold reflection into Z scale.
        rot3[:, 2] *= -1.0
        scale[2] *= -1.0
    rot4 = np.eye(4, dtype=float)
    rot4[:3, :3] = rot3
    # trimesh returns quaternion as [w, x, y, z]; glTF expects [x, y, z, w]
    q_wxyz = trimesh.transformations.quaternion_from_matrix(rot4)
    rotation = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=float)
    return translation, rotation, scale


def ensure_quaternion_track_continuity(rotations):
    arr = np.asarray(rotations, dtype=np.float32).copy()
    if arr.ndim != 2 or arr.shape[0] <= 1 or arr.shape[1] != 4:
        return arr
    for i in range(1, arr.shape[0]):
        prev = arr[i - 1]
        cur = arr[i]
        if float(np.dot(prev, cur)) < 0.0:
            arr[i] = -cur
    return arr


def export_animated_glb(
    out_path,
    link_meshes,
    frames,
    links,
    joints,
    fps,
    glb_scene=None,
    link_to_nodes_override=None,
    glb_scene_path=None,
):
    from pygltflib import (
        GLTF2,
        Animation,
        AnimationChannel,
        AnimationSampler,
        Accessor,
        BufferView,
        AnimationChannelTarget,
    )
    import numpy as np

    tmp_path = None
    if glb_scene_path is not None:
        gltf = GLTF2().load(str(glb_scene_path))
        scene = glb_scene
    elif glb_scene is None:
        scene = trimesh.Scene()
        for link_name, meshes in link_meshes.items():
            for mi, mesh in enumerate(meshes):
                node_name = f"link_{link_name}_{mi}"
                scene.add_geometry(mesh, node_name=node_name)
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp_path = tmp.name
        scene.export(tmp_path)
        gltf = GLTF2().load(tmp_path)
    else:
        scene = glb_scene
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as tmp:
            tmp_path = tmp.name
        scene.export(tmp_path)
        gltf = GLTF2().load(tmp_path)
    force_dielectric = os.environ.get("CODEX_GLTF_FORCE_DIELECTRIC", "1") not in {"0", "false", "False"}
    if gltf.materials:
        for mat in gltf.materials:
            if mat is None:
                continue
            mat.doubleSided = True
            pbr = getattr(mat, "pbrMetallicRoughness", None)
            if pbr is not None and force_dielectric:
                # OBJ/MTL -> GLB conversion commonly defaults to metallic=1,
                # which makes textured plastics appear unnaturally gray/dark.
                pbr.metallicFactor = 0.0
                pbr.roughnessFactor = 1.0
                bcf = getattr(pbr, "baseColorFactor", None)
                if bcf is not None and len(bcf) >= 4 and float(bcf[3]) < 0.999:
                    mat.alphaMode = "BLEND"
    _add_emissive_boost_for_basecolor_textures(gltf, gltf_path=tmp_path)
    _add_emissive_boost_for_bright_solid_materials(gltf)

    link_to_nodes = {}
    node_rest_map = {}
    if link_to_nodes_override:
        # convert node names -> indices if needed
        if gltf.nodes:
            node_name_to_idx = {node.name: i for i, node in enumerate(gltf.nodes) if node.name}
        else:
            node_name_to_idx = {}
        if glb_scene is not None:
            node_tf_by_name = {n: glb_scene.graph[n][0] for n in glb_scene.graph.nodes_geometry}
            for name, idx in node_name_to_idx.items():
                if name in node_tf_by_name:
                    node_rest_map[idx] = node_tf_by_name[name]
        for link, nodes in link_to_nodes_override.items():
            idxs = []
            for n in nodes:
                if isinstance(n, str):
                    if n in node_name_to_idx:
                        idxs.append(node_name_to_idx[n])
                else:
                    idxs.append(int(n))
            if idxs:
                link_to_nodes[link] = idxs
    else:
        if gltf.nodes:
            for i, node in enumerate(gltf.nodes):
                if not node.name:
                    continue
                if node.name.startswith("link_"):
                    # name format: link_<linkname>_<idx>
                    parts = node.name.split("_")
                    if len(parts) >= 3:
                        link = "_".join(parts[1:-1])
                    else:
                        link = parts[1]
                    link_to_nodes.setdefault(link, []).append(i)
                node_rest_map[i] = np.eye(4)

    gltf.animations = []
    gltf.animations.append(Animation(channels=[], samplers=[]))
    anim = gltf.animations[-1]

    binary_data = bytearray(gltf.binary_blob() or b"")
    if tmp_path is not None:
        os.unlink(tmp_path)

    def add_to_binary(data_bytes):
        nonlocal binary_data
        while len(binary_data) % 4 != 0:
            binary_data.append(0)
        start = len(binary_data)
        binary_data.extend(data_bytes)
        gltf.buffers[0].byteLength = len(binary_data)
        return start, len(data_bytes)

    # time accessor
    times = np.linspace(0, (len(frames) - 1) / fps, len(frames)).astype(np.float32)
    time_start, time_length = add_to_binary(times.tobytes())
    time_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(buffer=0, byteOffset=time_start, byteLength=time_length))
    time_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        Accessor(
            bufferView=time_bv_idx,
            componentType=5126,
            count=len(times),
            type="SCALAR",
            min=[float(times.min())],
            max=[float(times.max())],
        )
    )

    # Base (reference) transforms used to compute animation deltas.
    # For external canonical GLB scenes, reference must be URDF rest pose
    # (zero joints + identity base). Using frame0 here can introduce pivot drift
    # when frame0 already has non-zero base translation.
    use_external_scene = glb_scene is not None or glb_scene_path is not None
    if use_external_scene:
        zero_joint_pos = {j.get("name"): 0.0 for j in joints if j.get("name")}
        link_tf0_map = compute_link_transforms(links, joints, zero_joint_pos, base_tf=np.eye(4))
    else:
        base_joint_pos, base_tf = frames[0]
        link_tf0_map = compute_link_transforms(links, joints, base_joint_pos, base_tf=base_tf)

    # For reconstructed scenes, set node base transforms to URDF rest pose.
    # For external GLB scenes, convert animated nodes to explicit TRS so animation is applied.
    if glb_scene is None:
        for link_name, node_indices in link_to_nodes.items():
            link_tf0 = link_tf0_map.get(link_name, np.eye(4))
            t0, q0 = matrix_to_trs(link_tf0)
            for node_idx in node_indices:
                node = gltf.nodes[node_idx]
                node.matrix = None
                node.translation = [float(v) for v in t0]
                node.rotation = [float(v) for v in q0]
                node.scale = [1.0, 1.0, 1.0]
    else:
        for node_indices in link_to_nodes.values():
            for node_idx in node_indices:
                node = gltf.nodes[node_idx]
                rest_tf = np.array(node_rest_map.get(node_idx, np.eye(4)), dtype=float)
                t0, q0, s0 = matrix_to_trs_scale(rest_tf)
                node.matrix = None
                node.translation = [float(v) for v in t0]
                node.rotation = [float(v) for v in q0]
                node.scale = [float(v) for v in s0]

    # build per-link animation (delta from base)
    link_names = list(link_meshes.keys()) if link_meshes else list(link_to_nodes.keys())
    for link_name in link_names:
        node_indices = link_to_nodes.get(link_name, [])
        if not node_indices:
            continue

        link_tf0 = link_tf0_map.get(link_name, np.eye(4))
        link_tf0_inv = np.linalg.inv(link_tf0)
        # For external GLB scenes, keep per-node rest transforms.
        # Otherwise we animate in reconstructed URDF-local space.
        frame_delta_tfs = []
        frame_node_tfs = []
        for joint_pos, base_tf in frames:
            link_tf = compute_link_transforms(links, joints, joint_pos, base_tf=base_tf).get(link_name, np.eye(4))
            if glb_scene is not None:
                frame_delta_tfs.append(link_tf @ link_tf0_inv)
            else:
                frame_node_tfs.append(link_tf0_inv @ link_tf)

        if glb_scene is not None:
            # External canonical GLBs may map one URDF link to multiple nodes.
            # Each node needs its own animated TRS track from its own rest transform.
            for node_idx in node_indices:
                rest_tf = np.asarray(node_rest_map.get(node_idx, np.eye(4)), dtype=float)
                translations = []
                rotations = []
                for delta_tf in frame_delta_tfs:
                    node_tf = delta_tf @ rest_tf
                    t, q = matrix_to_trs(node_tf)
                    translations.append(t.astype(np.float32))
                    rotations.append(q.astype(np.float32))
                translations = np.stack(translations)
                rotations = ensure_quaternion_track_continuity(np.stack(rotations))

                t_start, t_length = add_to_binary(translations.tobytes())
                t_bv_idx = len(gltf.bufferViews)
                gltf.bufferViews.append(BufferView(buffer=0, byteOffset=t_start, byteLength=t_length))
                t_acc_idx = len(gltf.accessors)
                gltf.accessors.append(
                    Accessor(
                        bufferView=t_bv_idx,
                        componentType=5126,
                        count=len(translations),
                        type="VEC3",
                    )
                )

                r_start, r_length = add_to_binary(rotations.tobytes())
                r_bv_idx = len(gltf.bufferViews)
                gltf.bufferViews.append(BufferView(buffer=0, byteOffset=r_start, byteLength=r_length))
                r_acc_idx = len(gltf.accessors)
                gltf.accessors.append(
                    Accessor(
                        bufferView=r_bv_idx,
                        componentType=5126,
                        count=len(rotations),
                        type="VEC4",
                    )
                )

                t_sampler_idx = len(anim.samplers)
                anim.samplers.append(AnimationSampler(input=time_acc_idx, output=t_acc_idx, interpolation="LINEAR"))
                anim.channels.append(
                    AnimationChannel(sampler=t_sampler_idx, target=AnimationChannelTarget(node=node_idx, path="translation"))
                )

                r_sampler_idx = len(anim.samplers)
                anim.samplers.append(AnimationSampler(input=time_acc_idx, output=r_acc_idx, interpolation="LINEAR"))
                anim.channels.append(
                    AnimationChannel(sampler=r_sampler_idx, target=AnimationChannelTarget(node=node_idx, path="rotation"))
                )
        else:
            translations = []
            rotations = []
            for node_tf in frame_node_tfs:
                t, q = matrix_to_trs(node_tf)
                translations.append(t.astype(np.float32))
                rotations.append(q.astype(np.float32))
            translations = np.stack(translations)
            rotations = ensure_quaternion_track_continuity(np.stack(rotations))

            t_start, t_length = add_to_binary(translations.tobytes())
            t_bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(BufferView(buffer=0, byteOffset=t_start, byteLength=t_length))
            t_acc_idx = len(gltf.accessors)
            gltf.accessors.append(
                Accessor(
                    bufferView=t_bv_idx,
                    componentType=5126,
                    count=len(translations),
                    type="VEC3",
                )
            )

            r_start, r_length = add_to_binary(rotations.tobytes())
            r_bv_idx = len(gltf.bufferViews)
            gltf.bufferViews.append(BufferView(buffer=0, byteOffset=r_start, byteLength=r_length))
            r_acc_idx = len(gltf.accessors)
            gltf.accessors.append(
                Accessor(
                    bufferView=r_bv_idx,
                    componentType=5126,
                    count=len(rotations),
                    type="VEC4",
                )
            )

            t_sampler_idx = len(anim.samplers)
            anim.samplers.append(AnimationSampler(input=time_acc_idx, output=t_acc_idx, interpolation="LINEAR"))
            for node_idx in node_indices:
                anim.channels.append(
                    AnimationChannel(sampler=t_sampler_idx, target=AnimationChannelTarget(node=node_idx, path="translation"))
                )

            r_sampler_idx = len(anim.samplers)
            anim.samplers.append(AnimationSampler(input=time_acc_idx, output=r_acc_idx, interpolation="LINEAR"))
            for node_idx in node_indices:
                anim.channels.append(
                    AnimationChannel(sampler=r_sampler_idx, target=AnimationChannelTarget(node=node_idx, path="rotation"))
                )

    gltf.set_binary_blob(binary_data)
    gltf.save(out_path)


def parse_target_value(val, joint_limits):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        expr = val.replace(" ", "")
        if "lower_limit" in val and "upper_limit" not in val:
            lower = joint_limits.get("lower")
            if lower is None:
                return 0.0
            try:
                factor = float(expr.split("*")[0])
            except Exception:
                factor = 1.0
            return factor * lower
        if "upper_limit" in val:
            upper = joint_limits.get("upper")
            lower = joint_limits.get("lower")
            if upper is None:
                return 0.0
            if "lower_limit" in val and lower is not None:
                if "(" in expr and "upper_limit-lower_limit" in expr:
                    try:
                        k_str = expr.split("+", 1)[1].split("*", 1)[0]
                        k = float(k_str)
                        return lower + k * (upper - lower)
                    except Exception:
                        pass
            try:
                factor = float(expr.split("*")[0])
            except Exception:
                factor = 1.0
            return factor * upper
        try:
            return float(val)
        except Exception:
            return 0.0
    return 0.0


def _plan_effective_duration_s(plan: dict) -> float:
    meta = plan.get("meta") or {}
    duration_s = float(meta.get("duration_s", 4.0))
    max_t1 = duration_s
    for seg in (plan.get("timeline") or []):
        try:
            max_t1 = max(max_t1, float(seg.get("t1", 0.0)))
        except Exception:
            continue
    return max(duration_s, max_t1)


def _resolve_canonical_glb_scene(asset_root: Path, use_glb_scene_arg: str | None) -> Path | None:
    canonical = asset_root / f"animated_textured_{asset_root.name}.glb"
    if not use_glb_scene_arg:
        return canonical if canonical.exists() else None
    if str(use_glb_scene_arg).lower() == "auto":
        if not canonical.exists():
            raise SystemExit(
                f'Canonical textured mesh not found for --use_glb_scene auto: {canonical}. '
                "Generate it first with tools/build_textured_animated_glb.py."
            )
        return canonical
    p = Path(use_glb_scene_arg)
    if p.name != canonical.name:
        raise SystemExit(
            f"Only canonical textured mesh is accepted: {canonical}. "
            f"Got: {p.resolve() if p.exists() else p}"
        )
    if not p.exists():
        raise SystemExit(f"Textured mesh not found: {p}")
    return p


def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap_to_joint_limit_span(q, limits):
    lo = limits.get("lower")
    hi = limits.get("upper")
    if lo is None or hi is None:
        return q
    span = float(hi) - float(lo)
    if span <= 1.0e-9:
        return q
    return float(lo) + ((float(q) - float(lo)) % span)


def main():
    parser = argparse.ArgumentParser(description="Execute LLM plan JSON and render with pyrender")
    parser.add_argument("--asset_root", required=True)
    parser.add_argument("--plan_json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, default=[800, 600])
    parser.add_argument("--export_mesh_sequence", action="store_true", help="Export per-frame animated mesh GLB sequence")
    parser.add_argument("--export_animated_glb", action="store_true", help="Export a single animated GLB for Blender")
    parser.add_argument(
        "--use_glb_scene",
        default=None,
        help='Canonical textured GLB path, or "auto" -> <asset_root>/animated_textured_<asset>.glb',
    )
    parser.add_argument("--debug_motion", action="store_true", help="Print per-second wheel/base motion stats")
    parser.add_argument("--trajectory_npz", default=None, help="Output trajectory npz path (default: <out>/trajectory.npz)")
    parser.add_argument("--trajectory_jsonl", default=None, help="Output trajectory jsonl path (default: <out>/trajectory.jsonl)")
    parser.add_argument(
        "--skip_frame_render",
        action="store_true",
        help="Do not render PNG frames or MP4 (still exports trajectory and animated GLB).",
    )
    parser.add_argument("--skip_glb_alignment_check", action="store_true", help="Disable preflight URDF<->GLB alignment check")
    parser.add_argument("--glb_center_tol", type=float, default=1e-3, help="Max allowed center offset (meters) for URDF<->GLB mapping")
    parser.add_argument("--glb_scale_tol", type=float, default=1e-2, help="Max allowed extents ratio error for URDF<->GLB mapping")
    args = parser.parse_args()

    asset_root = Path(args.asset_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    urdf_path = next(asset_root.rglob("*.urdf"), None)
    if urdf_path is None:
        raise SystemExit(f"No URDF found under {asset_root}")

    links, joints = parse_urdf(urdf_path)
    link_meshes = load_link_meshes(links, urdf_path.parent, textured=False)

    with open(args.plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)

    fps = int(plan.get("meta", {}).get("fps", 30))
    duration_s = _plan_effective_duration_s(plan)
    T = max(1, int(round(duration_s * fps)))
    dt = 1.0 / fps

    # state
    joint_limits = {j["name"]: j.get("limit") or {} for j in joints}
    joint_pos = {j["name"]: 0.0 for j in joints}
    joint_vel = {j["name"]: 0.0 for j in joints}
    base_pos = np.zeros(3, dtype=float)
    modes = {}

    # spring dynamics
    spring_cfg = {d["joint"]: d for d in plan.get("physics", {}).get("joint_dynamics", []) if d.get("joint")}

    timeline = plan.get("timeline", [])
    joint_by_name = {j.get("name"): j for j in joints}
    velocity_controlled_joints = set()
    for seg in timeline or []:
        for ctrl in seg.get("controls", []) or []:
            mode = str(ctrl.get("type") or ctrl.get("mode") or "").strip().lower()
            if mode not in {"velocity", "joint_velocity"}:
                continue
            if ctrl.get("joint"):
                velocity_controlled_joints.add(str(ctrl.get("joint")))
            for jn in ctrl.get("joints") or []:
                if jn:
                    velocity_controlled_joints.add(str(jn))

    def active_segments(t):
        return [seg for seg in timeline if seg.get("t0", 0.0) <= t < seg.get("t1", 0.0)]

    frames = []
    warned_unknown_joints = set()
    trajectory_records = []
    def resolve_joint_target(ctrl, joint_limits):
        if ctrl.get("q_target_rad") is not None:
            return float(ctrl.get("q_target_rad"))
        if ctrl.get("target_rad") is not None:
            return float(ctrl.get("target_rad"))
        expr = ctrl.get("q_target_expr") or ctrl.get("target_expr")
        return parse_target_value(expr, joint_limits)

    for i in range(T):
        t = i * dt
        segments = active_segments(t)

        # reset per-frame control accumulators
        base_vel = np.zeros(3, dtype=float)
        base_vel_set = False
        # Joint velocities are command-driven in this executor. If no active control
        # mentions a joint this frame, it should not keep moving from a stale prior segment.
        next_joint_vel = {jn: 0.0 for jn in joint_pos.keys()}
        # apply controls
        for seg in segments:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", t0))
            seg_tau = max(t1 - t0, dt)
            local_t = (t - t0) / seg_tau
            for ctrl in seg.get("controls", []):
                ctype = ctrl.get("type")
                if not ctype and "mode" in ctrl:
                    mode = str(ctrl.get("mode"))
                    if mode in ("position", "joint_position"):
                        ctype = "joint_position"
                    elif mode in ("velocity", "joint_velocity"):
                        ctype = "joint_velocity"
                    elif mode in ("base_velocity", "base"):
                        ctype = "base_velocity"
                    elif mode in ("base_velocity_decay", "base_decay"):
                        ctype = "base_velocity_decay"
                    elif mode == "spring_return":
                        ctype = "spring_return"
                    elif mode in ("hold", "hold_position"):
                        ctype = "hold_position"
                if ctype == "base_velocity":
                    axis = ctrl.get("axis_world") or [1.0, 0.0, 0.0]
                    v = float(
                        ctrl.get(
                            "v_mps",
                            ctrl.get("linear_velocity_mps", ctrl.get("linear_mps", 0.0)),
                        )
                    )
                    base_vel += np.array(axis, dtype=float) * v
                    base_vel_set = True
                elif ctype == "base_velocity_decay":
                    axis = ctrl.get("axis_world") or [1.0, 0.0, 0.0]
                    v0 = float(
                        ctrl.get(
                            "v0_mps",
                            ctrl.get(
                                "linear_velocity_mps",
                                ctrl.get("linear_mps_initial", ctrl.get("linear_mps", 0.0)),
                            ),
                        )
                    )
                    tau = float(ctrl.get("tau_s", ctrl.get("decay_tau_s", 1.0)))
                    v = v0 * math.exp(-(t - t0) / max(tau, 1e-6))
                    base_vel += np.array(axis, dtype=float) * v
                    base_vel_set = True
                elif ctype == "joint_velocity":
                    omega = float(ctrl.get("omega_radps", 0.0))
                    if ctrl.get("ramp_to_omega_radps") is not None:
                        omega1 = float(ctrl.get("ramp_to_omega_radps", omega))
                        omega = omega + (omega1 - omega) * min(max(local_t, 0.0), 1.0)
                    decay = ctrl.get("decay")
                    if isinstance(decay, dict) and decay.get("type") == "exponential":
                        tau = float(decay.get("tau_s", 1.0))
                        min_omega = float(decay.get("min_omega_radps", 0.0))
                        omega_decay = omega * math.exp(-(t - t0) / max(tau, 1e-6))
                        if abs(omega_decay) < min_omega and min_omega > 0.0:
                            omega = math.copysign(min_omega, omega if abs(omega) > 1e-8 else omega_decay)
                        else:
                            omega = omega_decay
                    joint_list = []
                    if ctrl.get("joint"):
                        joint_list = [ctrl.get("joint")]
                    elif isinstance(ctrl.get("joints"), list):
                        joint_list = list(ctrl.get("joints"))
                    for jn in joint_list:
                        if not jn:
                            continue
                        if jn not in joint_pos:
                            if jn not in warned_unknown_joints:
                                print(f"[WARN] Unknown joint in plan control (joint_velocity): {jn} (ignored)")
                                warned_unknown_joints.add(jn)
                            continue
                        next_joint_vel[jn] = omega
                elif ctype == "joint_position":
                    jn = ctrl.get("joint")
                    if jn:
                        if jn not in joint_pos:
                            if jn not in warned_unknown_joints:
                                print(f"[WARN] Unknown joint in plan control (joint_position): {jn} (ignored)")
                                warned_unknown_joints.add(jn)
                            continue
                        q_target = resolve_joint_target(ctrl, joint_limits.get(jn, {}))
                        if args.debug_motion and i == 0:
                            expr_dbg = ctrl.get("q_target_expr") or ctrl.get("target_expr")
                            print(f"[dbg target] joint={jn} q_target={q_target} expr={expr_dbg}")
                        curve = ctrl.get("curve", "linear")
                        alpha = ease_in_out(local_t) if curve == "ease_in_out" else local_t
                        joint_pos[jn] = joint_pos[jn] * (1 - alpha) + float(q_target) * alpha
                        next_joint_vel[jn] = 0.0
                elif ctype == "hold_position":
                    joint_list = []
                    if ctrl.get("joint"):
                        joint_list = [ctrl.get("joint")]
                    elif isinstance(ctrl.get("joints"), list):
                        joint_list = list(ctrl.get("joints"))
                    for jn in joint_list:
                        if jn:
                            if jn not in joint_pos:
                                if jn not in warned_unknown_joints:
                                    print(f"[WARN] Unknown joint in plan control (hold_position): {jn} (ignored)")
                                    warned_unknown_joints.add(jn)
                                continue
                            next_joint_vel[jn] = 0.0
                elif ctype == "release_joint":
                    jn = ctrl.get("joint")
                    if jn:
                        if jn not in joint_pos:
                            if jn not in warned_unknown_joints:
                                print(f"[WARN] Unknown joint in plan control (release_joint): {jn} (ignored)")
                                warned_unknown_joints.add(jn)
                            continue
                        next_joint_vel[jn] = 0.0
                elif ctype == "spring_return":
                    jn = ctrl.get("joint")
                    if jn:
                        if jn not in joint_pos:
                            if jn not in warned_unknown_joints:
                                print(f"[WARN] Unknown joint in plan control (spring_return): {jn} (ignored)")
                                warned_unknown_joints.add(jn)
                            continue
                        cfg = spring_cfg.get(jn, {})
                        k = float(ctrl.get("spring_k", cfg.get("spring_k", 0.0)))
                        c = float(ctrl.get("damping_c", cfg.get("damping_c", 0.0)))
                        rest = float(
                            ctrl.get(
                                "target_rad",
                                ctrl.get("rest_position", cfg.get("rest_position", 0.0)),
                            )
                        )
                        q = joint_pos.get(jn, 0.0)
                        qd = joint_vel.get(jn, 0.0)
                        qdd = -k * (q - rest) - c * qd
                        next_joint_vel[jn] = qd + qdd * dt
                elif ctype == "mode_set":
                    mode_name = ctrl.get("mode") or ctrl.get("name")
                    if mode_name:
                        modes[mode_name] = bool(ctrl.get("value", ctrl.get("set", True)))
                elif ctype == "wheel_rolling_decay":
                    tau = float(ctrl.get("tau_s", 1.0))
                    for jn in ctrl.get("wheel_joints", []):
                        if jn not in joint_pos:
                            if jn not in warned_unknown_joints:
                                print(f"[WARN] Unknown joint in plan control (wheel_rolling_decay): {jn} (ignored)")
                                warned_unknown_joints.add(jn)
                            continue
                        next_joint_vel[jn] = joint_vel.get(jn, 0.0) * math.exp(-(t - t0) / max(tau, 1e-6))

        # integrate
        joint_vel = next_joint_vel
        base_pos += base_vel * dt
        for jn in joint_pos.keys():
            joint_pos[jn] = joint_pos.get(jn, 0.0) + joint_vel.get(jn, 0.0) * dt

        base_tf = np.eye(4)
        base_tf[:3, 3] = base_pos
        # Clamp only bounded joints for rendering/export. Preserve continuous joint
        # winding so wheel-like motion does not fold back when exported to GLB.
        joint_pos_render = joint_pos.copy()
        for jn in joint_pos_render.keys():
            jinfo = joint_by_name.get(jn, {})
            limits = joint_limits.get(jn, {})
            if jinfo.get("type") == "continuous":
                continue
            if jn in velocity_controlled_joints and jinfo.get("type") == "revolute":
                joint_pos_render[jn] = _wrap_to_joint_limit_span(joint_pos_render[jn], limits)
                continue
            lo = limits.get("lower")
            hi = limits.get("upper")
            if lo is not None and joint_pos_render[jn] < lo:
                joint_pos_render[jn] = lo
            if hi is not None and joint_pos_render[jn] > hi:
                joint_pos_render[jn] = hi

        frames.append((joint_pos_render.copy(), base_tf.copy()))
        trajectory_records.append(
            {
                "frame_idx": i,
                "time_s": float(t),
                "joint_pos": copy.deepcopy(joint_pos_render),
                "base_translation": base_pos.copy(),
                "base_rotation_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            }
        )

        if args.debug_motion and (i % max(1, int(fps)) == 0 or i == T - 1):
            moving_joint_state = {}
            for seg in segments:
                for ctrl in seg.get("controls", []):
                    if str(ctrl.get("type") or ctrl.get("mode") or "").strip().lower() != "joint_velocity":
                        continue
                    if ctrl.get("joint"):
                        joint_names = [str(ctrl.get("joint"))]
                    else:
                        joint_names = [str(x) for x in (ctrl.get("joints") or []) if str(x)]
                    for jn in joint_names:
                        if jn in joint_pos and jn not in moving_joint_state:
                            moving_joint_state[jn] = {
                                "q": float(joint_pos[jn]),
                                "qd": float(joint_vel.get(jn, 0.0)),
                            }
            print(
                f"[dbg t={t:.2f}s] base_pos={base_pos.tolist()} base_vel={base_vel.tolist()} joints={moving_joint_state}"
            )

    # Save trajectory (always, independent of renderer availability)
    traj_npz_path = Path(args.trajectory_npz) if args.trajectory_npz else (out_root / "trajectory.npz")
    traj_jsonl_path = Path(args.trajectory_jsonl) if args.trajectory_jsonl else (out_root / "trajectory.jsonl")
    joint_names_sorted = sorted(joint_pos.keys())
    joint_angles = np.zeros((len(trajectory_records), len(joint_names_sorted)), dtype=np.float32)
    base_translation = np.zeros((len(trajectory_records), 3), dtype=np.float32)
    base_rotation_xyzw = np.zeros((len(trajectory_records), 4), dtype=np.float32)
    times_s = np.zeros((len(trajectory_records),), dtype=np.float32)
    for i, rec in enumerate(trajectory_records):
        times_s[i] = float(rec["time_s"])
        base_translation[i] = np.asarray(rec["base_translation"], dtype=np.float32)
        base_rotation_xyzw[i] = np.asarray(rec["base_rotation_xyzw"], dtype=np.float32)
        for j, jn in enumerate(joint_names_sorted):
            joint_angles[i, j] = float(rec["joint_pos"].get(jn, 0.0))
    np.savez(
        traj_npz_path,
        joint_names=np.array(joint_names_sorted, dtype=object),
        joint_angles=joint_angles,
        base_translation=base_translation,
        base_rotation_xyzw=base_rotation_xyzw,
        time_s=times_s,
    )
    with open(traj_jsonl_path, "w", encoding="utf-8") as f:
        for rec in trajectory_records:
            row = {
                "frame_idx": int(rec["frame_idx"]),
                "time_s": float(rec["time_s"]),
                "joint_angles": {k: float(v) for k, v in rec["joint_pos"].items()},
                "base_pose": {
                    "translation": [float(x) for x in np.asarray(rec["base_translation"]).tolist()],
                    "rotation_xyzw": [float(x) for x in np.asarray(rec["base_rotation_xyzw"]).tolist()],
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    glb_scene_path = _resolve_canonical_glb_scene(asset_root, args.use_glb_scene)
    frame_dir = out_root / "plan_frames"
    render_glb_path = None
    temp_render_glb_path = None
    need_render_glb = bool(args.export_animated_glb or (not args.skip_frame_render))
    if need_render_glb:
        if args.export_animated_glb:
            glb_path = out_root / "plan_animated.glb"
        else:
            with tempfile.NamedTemporaryFile(prefix="plan_render_", suffix=".glb", delete=False) as tmp:
                glb_path = Path(tmp.name)
            temp_render_glb_path = glb_path
        if glb_scene_path is not None:
            scene = trimesh.load(glb_scene_path, force="scene", process=False)
            rest_joint_pos = {j.get("name"): 0.0 for j in joints if j.get("name")}
            link_tf0_map = compute_link_transforms(links, joints, rest_joint_pos, base_tf=np.eye(4))
            link_meshes_tmp = load_link_meshes(links, urdf_path.parent, textured=False)
            link_to_nodes = match_links_to_nodes_particulate_by_order(links, link_meshes_tmp, scene)
            if link_to_nodes is not None:
                print("[INFO] Using Particulate link->GLB node mapping by URDF link order.")
            else:
                link_bbox = compute_link_bbox(link_meshes_tmp, link_tf0_map)
                node_bbox = compute_scene_node_bbox(scene)
                link_to_nodes = match_links_to_nodes(link_bbox, node_bbox)
                print("[INFO] Using bbox-based link->GLB node mapping.")
            if not args.skip_glb_alignment_check:
                metrics = compute_mapping_alignment_metrics(link_meshes_tmp, link_tf0_map, scene, link_to_nodes)
                if metrics["matched_links"] > 0:
                    bad_center = metrics["max_center_delta"] > float(args.glb_center_tol)
                    bad_scale = metrics["max_scale_err"] > float(args.glb_scale_tol)
                    if bad_center or bad_scale:
                        raise SystemExit(
                            "Input textured GLB is not aligned with URDF rest pose. "
                            f"matched_links={metrics['matched_links']} "
                            f"max_center_delta={metrics['max_center_delta']:.6f} "
                            f"max_scale_err={metrics['max_scale_err']:.6f}. "
                            "Rebuild canonical animated_textured_<asset>.glb with tools/build_textured_animated_glb.py "
                            "or pass --skip_glb_alignment_check to override."
                        )
            node_count = len(scene.graph.nodes_geometry) if isinstance(scene, trimesh.Scene) else 0
            mesh_link_count = sum(1 for ln, meshes in link_meshes_tmp.items() if meshes)
            if node_count < mesh_link_count:
                print(
                    f"[WARN] GLB nodes ({node_count}) < mesh links ({mesh_link_count}); "
                    "falling back to URDF textured meshes for animation."
                )
                textured_meshes = load_link_meshes(links, urdf_path.parent, textured=True)
                export_animated_glb(glb_path, textured_meshes, frames, links, joints, fps)
            else:
                export_animated_glb(
                    glb_path,
                    {},
                    frames,
                    links,
                    joints,
                    fps,
                    glb_scene=scene,
                    link_to_nodes_override=link_to_nodes,
                    glb_scene_path=glb_scene_path,
                )
        else:
            textured_meshes = load_link_meshes(links, urdf_path.parent, textured=True)
            export_animated_glb(glb_path, textured_meshes, frames, links, joints, fps)
        render_glb_path = glb_path
        if args.export_animated_glb:
            print(f"Wrote animated GLB to {glb_path}")

    if args.export_mesh_sequence:
        textured_meshes = load_link_meshes(links, urdf_path.parent, textured=True)
        export_mesh_sequence(out_root, frames, textured_meshes, links, joints)

    if args.skip_frame_render:
        print("[INFO] Skipping PNG/MP4 rendering (--skip_frame_render).")
    else:
        center, radius = compute_scene_bounds(link_meshes)
        camera = compute_camera(center, radius, azim_deg=45.0, elev_deg=25.0)
        frame_dir.mkdir(parents=True, exist_ok=True)
        if render_glb_path is None:
            raise SystemExit("Blender frame rendering requires an animated GLB, but export did not produce one.")
        br.render_animation_sequence_from_glb(
            render_glb_path,
            frame_dir,
            tuple(int(x) for x in args.resolution),
            camera,
            frame_count=len(frames),
            fps=fps,
            fov_deg=45.0,
        )

        mp4_path = out_root / "plan.mp4"
        if _HAS_IMAGEIO:
            try:
                writer = imageio.get_writer(mp4_path, format="FFMPEG", fps=fps)
                for idx in range(len(frames)):
                    frame_path = frame_dir / f"frame_{idx:04d}.png"
                    writer.append_data(imageio.imread(frame_path))
                writer.close()
            except Exception:
                ffmpeg = get_ffmpeg_path()
                if ffmpeg:
                    pattern = str(frame_dir / "frame_%04d.png")
                    subprocess.run(
                        [
                            ffmpeg,
                            "-y",
                            "-framerate",
                            str(fps),
                            "-i",
                            pattern,
                            "-pix_fmt",
                            "yuv420p",
                            str(mp4_path),
                        ],
                        check=False,
                    )
                else:
                    print("[WARN] No ffmpeg available; skipping MP4 export.")
        else:
            ffmpeg = get_ffmpeg_path()
            if ffmpeg:
                pattern = str(frame_dir / "frame_%04d.png")
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-framerate",
                        str(fps),
                        "-i",
                        pattern,
                        "-pix_fmt",
                        "yuv420p",
                        str(mp4_path),
                    ],
                    check=False,
                )
            else:
                print("[WARN] No ffmpeg available; skipping MP4 export.")

    if temp_render_glb_path is not None:
        try:
            Path(temp_render_glb_path).unlink()
        except Exception:
            pass

    print(f"Wrote trajectory NPZ to {traj_npz_path}")
    print(f"Wrote trajectory JSONL to {traj_jsonl_path}")
    if not args.skip_frame_render:
        print(f"Wrote frames to {frame_dir}")


if __name__ == "__main__":
    main()
