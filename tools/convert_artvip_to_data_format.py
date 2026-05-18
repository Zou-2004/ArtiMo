#!/usr/bin/env python3
"""
Convert ArtVIP object-level articulated assets to this repo data format.

Output per asset:
  <out_root>/<asset_name>/
    mobility.urdf
    animated_textured_<asset_name>.glb
    meshes/<link_name>.obj
    sim_scripts/... (if python scripts exist)
    source_model.usd|usda|usdc

Optional:
  --sim_code_root <dir>/<asset_name>/python/... and isaac_sim_dirs/...
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


AXIS_MAP = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}

# Use t=0 as canonical rest pose for conversion. Some USD assets have
# non-rest Default values while timeline frame 0 is the intended closed/rest state.
REST_TIME = Usd.TimeCode(0.0)


@dataclass
class JointInfo:
    name: str
    jtype: str
    parent: str
    child: str
    parent_to_joint: np.ndarray
    child_to_joint: np.ndarray
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: Optional[np.ndarray]
    lower: Optional[float]
    upper: Optional[float]


def _is_identity_tf(T: np.ndarray, atol: float = 1e-8) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    I = np.eye(4, dtype=np.float64)
    return bool(np.allclose(T, I, atol=atol, rtol=0.0))


def _safe_name(s: str) -> str:
    s = s.replace(" ", "_")
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "link"


def _extract_trailing_int(name: str) -> Optional[int]:
    m = re.search(r"(\d+)$", str(name or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _find_model_usd(asset_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    candidates += sorted(asset_dir.glob("model_*.usd"))
    candidates += sorted(asset_dir.glob("model_*.usda"))
    candidates += sorted(asset_dir.glob("model_*.usdc"))
    if candidates:
        return candidates[0]
    # fallback
    generic = sorted([p for p in asset_dir.glob("*.usd")] + [p for p in asset_dir.glob("*.usda")] + [p for p in asset_dir.glob("*.usdc")])
    return generic[0] if generic else None


def _gf_to_np_mat4(m: Gf.Matrix4d) -> np.ndarray:
    # Match trimesh transform convention (points @ T.T).
    return np.array(m, dtype=np.float64).T


def _quat_to_rpy(qval) -> np.ndarray:
    # USD quaternions are (real, imagXYZ), i.e. w, (x,y,z)
    if hasattr(qval, "GetReal"):
        w = float(qval.GetReal())
        imag = qval.GetImaginary()
        x, y, z = float(imag[0]), float(imag[1]), float(imag[2])
    elif isinstance(qval, (tuple, list)) and len(qval) == 4:
        w, x, y, z = [float(v) for v in qval]
    else:
        return np.zeros(3, dtype=np.float64)
    M = trimesh.transformations.quaternion_matrix([w, x, y, z])
    rpy = np.array(trimesh.transformations.euler_from_matrix(M, axes="sxyz"), dtype=np.float64)
    return rpy


def _quat_to_mat4(qval) -> np.ndarray:
    if hasattr(qval, "GetReal"):
        w = float(qval.GetReal())
        imag = qval.GetImaginary()
        x, y, z = float(imag[0]), float(imag[1]), float(imag[2])
    elif isinstance(qval, (tuple, list)) and len(qval) == 4:
        w, x, y, z = [float(v) for v in qval]
    else:
        return np.eye(4, dtype=np.float64)
    return np.array(trimesh.transformations.quaternion_matrix([w, x, y, z]), dtype=np.float64)


def _matrix_to_xyz_rpy(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xyz = np.array(T[:3, 3], dtype=np.float64)
    rpy = np.array(trimesh.transformations.euler_from_matrix(T, axes="sxyz"), dtype=np.float64)
    return xyz, rpy


def _rigidize_transform(T: np.ndarray) -> np.ndarray:
    """Project an affine transform to the closest rigid transform (rotation + translation)."""
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = T[:3, 3]
    A = np.array(T[:3, :3], dtype=np.float64)
    U, _, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    out[:3, :3] = R
    return out


def _origin_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    return trimesh.transformations.compose_matrix(translate=xyz.tolist(), angles=rpy.tolist())


def _has_mesh_descendant(prim: Usd.Prim) -> bool:
    for p in Usd.PrimRange(prim):
        if p.GetTypeName() == "Mesh":
            return True
    return False


def _extract_body_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    out: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and _has_mesh_descendant(prim):
            out.append(prim)
    return out


def _collect_mesh_world_under_body(body_prim: Usd.Prim, body_paths: set[str], time_code: Usd.TimeCode = REST_TIME) -> Optional[trimesh.Trimesh]:
    verts_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    vert_offset = 0
    body_root = str(body_prim.GetPath())
    nested_body_prefixes = tuple(
        f"{bp}/" for bp in body_paths if bp != body_root and str(bp).startswith(f"{body_root}/")
    )

    for p in Usd.PrimRange(body_prim):
        p_path = str(p.GetPath())
        if p != body_prim and p_path in body_paths:
            continue
        if nested_body_prefixes and any(p_path.startswith(pref) for pref in nested_body_prefixes):
            continue
        if p.GetTypeName() != "Mesh":
            continue

        mesh = UsdGeom.Mesh(p)
        points = mesh.GetPointsAttr().Get(time_code)
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if points is None or face_counts is None or face_indices is None:
            continue

        xform = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(time_code)
        pts_np = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in points], dtype=np.float64)
        # robust point transform via Gf
        pts_w = np.array([[float(xform.Transform(Gf.Vec3d(*pt))[i]) for i in range(3)] for pt in pts_np], dtype=np.float64)

        idx = np.array(face_indices, dtype=np.int64)
        counts = np.array(face_counts, dtype=np.int64)
        tris: List[np.ndarray] = []
        cursor = 0
        for c in counts:
            poly = idx[cursor : cursor + c]
            cursor += c
            if c < 3:
                continue
            if c == 3:
                tris.append(poly)
            else:
                # fan triangulation
                for k in range(1, c - 1):
                    tris.append(np.array([poly[0], poly[k], poly[k + 1]], dtype=np.int64))
        if not tris:
            continue

        faces = np.vstack(tris)
        verts_all.append(pts_w)
        faces_all.append(faces + vert_offset)
        vert_offset += pts_w.shape[0]

    if not verts_all:
        return None
    V = np.vstack(verts_all)
    F = np.vstack(faces_all)
    m = trimesh.Trimesh(vertices=V, faces=F, process=False)
    return m


def _build_rest_link_tf(root_link: str, joints: List[JointInfo], link_names: List[str]) -> Dict[str, np.ndarray]:
    tree: Dict[str, List[JointInfo]] = {}
    for j in joints:
        tree.setdefault(j.parent, []).append(j)
    tf: Dict[str, np.ndarray] = {root_link: np.eye(4)}

    def dfs(parent: str):
        parent_tf = tf[parent]
        for j in tree.get(parent, []):
            T = _origin_matrix(j.origin_xyz, j.origin_rpy)
            child_tf = parent_tf @ T
            tf[j.child] = child_tf
            dfs(j.child)

    dfs(root_link)
    for ln in link_names:
        tf.setdefault(ln, np.eye(4))
    return tf


def _write_obj(mesh: trimesh.Trimesh, out_path: Path) -> None:
    out_path.write_text(trimesh.exchange.obj.export_obj(mesh), encoding="utf-8")


def _property_authored_layer_dir(attr, time_code: Usd.TimeCode = REST_TIME) -> Optional[Path]:
    if attr is None:
        return None
    for tc in (time_code, Usd.TimeCode.Default(), None):
        try:
            stack = attr.GetPropertyStack(tc) if tc is not None else attr.GetPropertyStack()
        except Exception:
            stack = []
        if not stack:
            continue
        for spec in stack:
            layer = getattr(spec, "layer", None)
            if layer is None:
                continue
            real_path = getattr(layer, "realPath", "") or ""
            if real_path:
                p = Path(real_path).resolve()
                if p.exists():
                    return p.parent
    return None


def _asset_path_to_file(ap, model_usd: Path, authored_layer_dir: Optional[Path] = None) -> Optional[Path]:
    tex_path = None
    raw_path: Optional[str] = None
    if ap is None:
        return None
    if hasattr(ap, "resolvedPath") and ap.resolvedPath:
        tex_path = Path(str(ap.resolvedPath))
        if tex_path.exists():
            return tex_path.resolve()
    elif hasattr(ap, "path") and ap.path:
        raw_path = str(ap.path)
    elif isinstance(ap, str) and ap:
        raw_path = ap

    if raw_path:
        raw_path = raw_path.replace("\\", "/")
        rp = Path(raw_path)
        if rp.is_absolute() and rp.exists():
            return rp.resolve()
        base_candidates: List[Path] = []
        if authored_layer_dir is not None:
            base_candidates.append(authored_layer_dir)
        base_candidates += [
            model_usd.parent,
            model_usd.parent / "resource",
            model_usd.parent / "resource" / "img",
        ]
        seen = set()
        for base in base_candidates:
            if base is None:
                continue
            b = base.resolve()
            if str(b) in seen:
                continue
            seen.add(str(b))
            c = (b / raw_path).resolve()
            if c.exists():
                return c
        # Last resort: basename lookup under local resource folder.
        bn = Path(raw_path).name
        if bn:
            for root in [model_usd.parent / "resource", model_usd.parent]:
                if not root.exists():
                    continue
                hit = next(root.rglob(bn), None)
                if hit is not None and hit.exists():
                    return hit.resolve()

    if tex_path is None or not tex_path.exists():
        return None
    return tex_path.resolve()


def _resolve_mesh_texture_file(
    mesh_prim: Usd.Prim, model_usd: Path, time_code: Usd.TimeCode = REST_TIME
) -> Tuple[
    str,
    Optional[Path],
    Tuple[np.ndarray, np.ndarray, float, str, str],
    Dict[str, object],
]:
    def _default_mat_props() -> Dict[str, object]:
        return {
            "kd": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "ks": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "opacity": 1.0,
        }

    def _read_color3(inp, fallback: np.ndarray) -> np.ndarray:
        if inp is None:
            return fallback
        try:
            val = inp.Get(time_code)
            if val is None:
                val = inp.Get(Usd.TimeCode.Default())
            if val is None:
                return fallback
            return np.array([float(val[0]), float(val[1]), float(val[2])], dtype=np.float64)
        except Exception:
            return fallback

    def _read_scalar(inp, fallback: float) -> float:
        if inp is None:
            return float(fallback)
        try:
            val = inp.Get(time_code)
            if val is None:
                val = inp.Get(Usd.TimeCode.Default())
            if val is None:
                return float(fallback)
            return float(val)
        except Exception:
            return float(fallback)

    def _connected_input(inp) -> bool:
        if inp is None:
            return False
        try:
            conn = inp.GetConnectedSource()
        except Exception:
            return False
        return bool(conn and conn[0])

    def _extract_mat_props(surface_shader) -> Dict[str, object]:
        props = _default_mat_props()
        if surface_shader is None:
            return props
        try:
            sh = UsdShade.Shader(surface_shader.GetPrim())
        except Exception:
            sh = None
        if sh is None or not sh.GetPrim().IsValid():
            return props

        diff_in = sh.GetInput("diffuseColor") or sh.GetInput("baseColor")
        if not _connected_input(diff_in):
            props["kd"] = _read_color3(diff_in, props["kd"])

        spec_in = sh.GetInput("specularColor")
        if spec_in is not None:
            props["ks"] = _read_color3(spec_in, props["ks"])

        op_in = sh.GetInput("opacity")
        props["opacity"] = float(np.clip(_read_scalar(op_in, float(props["opacity"])), 0.0, 1.0))
        return props

    def _to_shader(obj):
        try:
            prim = obj.GetPrim()
        except Exception:
            prim = None
        if prim is None:
            return None
        sh = UsdShade.Shader(prim)
        if not sh or not sh.GetPrim().IsValid():
            return None
        return sh

    try:
        mat, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
    except Exception:
        mat = None
    if not mat or not mat.GetPrim().IsValid():
        return (
            "default_mat",
            None,
            (
                np.array([1.0, 1.0], dtype=np.float64),
                np.array([0.0, 0.0], dtype=np.float64),
                0.0,
                "repeat",
                "repeat",
            ),
            _default_mat_props(),
        )
    mat_name = _safe_name(mat.GetPrim().GetName() or "mat")
    uv_scale = np.array([1.0, 1.0], dtype=np.float64)
    uv_translate = np.array([0.0, 0.0], dtype=np.float64)
    uv_rotate_deg = 0.0
    wrap_s = "repeat"
    wrap_t = "repeat"
    # First: try standard UsdPreviewSurface diffuseColor connection.
    try:
        shader = mat.ComputeSurfaceSource()[0]
    except Exception:
        shader = None
    mat_props = _extract_mat_props(shader)
    if shader:
        try:
            tex_input = shader.GetInput("diffuseColor") or shader.GetInput("baseColor")
            if tex_input:
                conn = tex_input.GetConnectedSource()
                if conn and conn[0]:
                    tex_shader = _to_shader(conn[0])
                    if tex_shader is not None:
                        f_in = tex_shader.GetInput("file")
                    else:
                        f_in = None
                    if f_in is not None:
                        f_attr = f_in.GetAttr()
                        ap = f_attr.Get(time_code) if f_attr else None
                        authored_dir = _property_authored_layer_dir(f_attr, time_code=time_code)
                        p = _asset_path_to_file(ap, model_usd, authored_layer_dir=authored_dir)
                        try:
                            ws_in = tex_shader.GetInput("wrapS")
                            ws_val = ws_in.Get(time_code) if ws_in else None
                            if ws_val is not None:
                                wrap_s = str(ws_val).lower()
                        except Exception:
                            pass
                        try:
                            wt_in = tex_shader.GetInput("wrapT")
                            wt_val = wt_in.Get(time_code) if wt_in else None
                            if wt_val is not None:
                                wrap_t = str(wt_val).lower()
                        except Exception:
                            pass
                        # Optional UV transform chain: UVTexture.st <- UsdTransform2d.result
                        st_in = tex_shader.GetInput("st")
                        if st_in:
                            st_conn = st_in.GetConnectedSource()
                            if st_conn and st_conn[0]:
                                st_node = _to_shader(st_conn[0])
                                if st_node is None:
                                    st_id = None
                                else:
                                    try:
                                        st_id = st_node.GetIdAttr().Get()
                                    except Exception:
                                        st_id = None
                                try:
                                    st_id_str = str(st_id) if st_id is not None else ""
                                except Exception:
                                    st_id_str = ""
                                if st_id_str == "UsdTransform2d":
                                    try:
                                        s_in = st_node.GetInput("scale")
                                        s = s_in.Get(time_code) if s_in else None
                                        if s is not None:
                                            uv_scale = np.array([float(s[0]), float(s[1])], dtype=np.float64)
                                    except Exception:
                                        pass
                                    try:
                                        t_in = st_node.GetInput("translation")
                                        t = t_in.Get(time_code) if t_in else None
                                        if t is not None:
                                            uv_translate = np.array([float(t[0]), float(t[1])], dtype=np.float64)
                                    except Exception:
                                        pass
                                    try:
                                        r_in = st_node.GetInput("rotation")
                                        r = r_in.Get(time_code) if r_in else None
                                        if r is not None:
                                            uv_rotate_deg = float(r)
                                    except Exception:
                                        pass
                        if p is not None:
                            return (
                                mat_name,
                                p,
                                (uv_scale, uv_translate, uv_rotate_deg, wrap_s, wrap_t),
                                mat_props,
                            )
        except Exception:
            pass

    # Fallback: scan all shaders under the material and pick the first existing texture file.
    try:
        mat_prim = mat.GetPrim()
        for sp in Usd.PrimRange(mat_prim):
            shader = UsdShade.Shader(sp)
            if not shader:
                continue
            for in_name in (
                "file",
                "diffuse_texture",
                "albedo_texture",
                "base_color_texture",
                "emissive_texture",
            ):
                inp = shader.GetInput(in_name)
                if not inp:
                    continue
                i_attr = inp.GetAttr()
                try:
                    ap = i_attr.Get(time_code) if i_attr else None
                except Exception:
                    ap = None
                authored_dir = _property_authored_layer_dir(i_attr, time_code=time_code)
                p = _asset_path_to_file(ap, model_usd, authored_layer_dir=authored_dir)
                if p is not None:
                    return (
                        mat_name,
                        p,
                        (uv_scale, uv_translate, uv_rotate_deg, wrap_s, wrap_t),
                        mat_props,
                    )
    except Exception:
        pass
    return mat_name, None, (uv_scale, uv_translate, uv_rotate_deg, wrap_s, wrap_t), mat_props


def _pick_uv_primvar(mesh_prim: Usd.Prim, time_code: Usd.TimeCode) -> Optional[UsdGeom.Primvar]:
    pv_api = UsdGeom.PrimvarsAPI(mesh_prim)
    primvars = list(pv_api.GetPrimvars())
    if not primvars:
        return None
    by_name = {pv.GetBaseName(): pv for pv in primvars}
    preferred = ["st", "uv", "UVMap", "map1", "st0", "uv0"]
    for name in preferred:
        pv = by_name.get(name)
        if pv and pv.HasValue():
            return pv
    for pv in primvars:
        name = (pv.GetBaseName() or "").lower()
        if not ("st" in name or "uv" in name or "map" in name):
            continue
        if pv.HasValue():
            return pv
    for pv in primvars:
        if not pv.HasValue():
            continue
        try:
            raw = pv.Get(time_code)
        except Exception:
            raw = None
        if not raw:
            continue
        try:
            if len(raw[0]) == 2:
                return pv
        except Exception:
            continue
    return None


def _export_textured_obj_for_link(
    body_prim: Usd.Prim,
    body_paths: set[str],
    model_usd: Path,
    out_obj_path: Path,
    out_tex_dir: Path,
    extra_tf: Optional[np.ndarray] = None,
    time_code: Usd.TimeCode = REST_TIME,
) -> bool:
    xcache = UsdGeom.XformCache(time_code)
    v_lines: List[str] = []
    vt_lines: List[str] = []
    f_lines: List[str] = []
    mtl_defs: Dict[str, Dict[str, object]] = {}
    copied_tex: Dict[Path, str] = {}
    v_off = 0
    cur_mat = None
    mat_uv_xform: Dict[str, Tuple[np.ndarray, np.ndarray, float, str, str]] = {}
    body_root = str(body_prim.GetPath())
    nested_body_prefixes = tuple(
        f"{bp}/" for bp in body_paths if bp != body_root and str(bp).startswith(f"{body_root}/")
    )

    disable_uv_xform = os.environ.get("CODEX_ARTVIP_DISABLE_UV_XFORM", "0") in {"1", "true", "True"}

    def transform_uv(uv: np.ndarray, xform: Tuple[np.ndarray, np.ndarray, float, str, str]) -> np.ndarray:
        if disable_uv_xform:
            return np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
        scale, trans, rot_deg, wrap_s, wrap_t = xform
        out = np.array([float(uv[0]), float(uv[1])], dtype=np.float64)
        out = out * scale
        if abs(rot_deg) > 1e-9:
            th = np.deg2rad(rot_deg)
            c = float(np.cos(th))
            s = float(np.sin(th))
            out = np.array([c * out[0] - s * out[1], s * out[0] + c * out[1]], dtype=np.float64)
        out = out + trans
        # Preserve transformed UV values as-is.
        # Do not pre-apply repeat/clamp here; baking modulo/clamp into vertices
        # breaks triangles crossing tile boundaries and causes texture drift.
        # Wrap behavior should be handled by runtime material sampler settings.
        return out

    def uv_index_for(
        interp: str,
        uv_indices: Optional[np.ndarray],
        point_index: int,
        face_vertex_flat_index: int,
        n_uv: int,
    ) -> Optional[int]:
        if n_uv <= 0:
            return None
        if interp == "faceVarying":
            if uv_indices is not None and face_vertex_flat_index < len(uv_indices):
                idx = int(uv_indices[face_vertex_flat_index])
            else:
                idx = face_vertex_flat_index
        else:  # varying/vertex/constant fallback
            if uv_indices is not None and point_index < len(uv_indices):
                idx = int(uv_indices[point_index])
            else:
                idx = point_index
        if idx < 0 or idx >= n_uv:
            return None
        return idx

    for p in Usd.PrimRange(body_prim):
        p_path = str(p.GetPath())
        if p != body_prim and p_path in body_paths:
            continue
        if nested_body_prefixes and any(p_path.startswith(pref) for pref in nested_body_prefixes):
            continue
        if p.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(p)
        pts = mesh.GetPointsAttr().Get(time_code)
        f_counts = mesh.GetFaceVertexCountsAttr().Get()
        f_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not f_counts or not f_indices:
            continue

        xf = _gf_to_np_mat4(xcache.GetLocalToWorldTransform(p))
        pts_np = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in pts], dtype=np.float64)
        pts_w = trimesh.transformations.transform_points(pts_np, xf)
        if extra_tf is not None:
            pts_w = trimesh.transformations.transform_points(pts_w, extra_tf)
        for v in pts_w:
            v_lines.append(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")

        pv = _pick_uv_primvar(p, time_code)
        uv_vals = None
        uv_indices = None
        uv_interp = "vertex"
        if pv and pv.HasValue():
            raw_uv = pv.Get(time_code)
            if raw_uv:
                uv_vals = np.array([[float(uv[0]), float(uv[1])] for uv in raw_uv], dtype=np.float64)
            uv_interp = str(pv.GetInterpolation() or "vertex")
            idx_attr = pv.GetIndicesAttr()
            if idx_attr and idx_attr.HasValue():
                raw_idx = idx_attr.Get(time_code)
                if raw_idx is not None:
                    uv_indices = np.array(raw_idx, dtype=np.int64)

        default_mat_name, default_tex_src, default_uv_xform, default_mat_props = _resolve_mesh_texture_file(
            p, model_usd, time_code=time_code
        )
        if default_mat_name not in mtl_defs:
            mtl_defs[default_mat_name] = {"texture": default_tex_src, "props": default_mat_props}
        elif mtl_defs[default_mat_name].get("texture") is None and default_tex_src is not None:
            mtl_defs[default_mat_name]["texture"] = default_tex_src
        mat_uv_xform.setdefault(default_mat_name, default_uv_xform)

        # GeomSubset support: allow per-face material assignment.
        face_mat_by_face_index: Dict[int, str] = {}
        for child in p.GetChildren():
            if child.GetTypeName() != "GeomSubset":
                continue
            subset = UsdGeom.Subset(child)
            elem_type_attr = subset.GetElementTypeAttr()
            elem_type = elem_type_attr.Get(time_code) if elem_type_attr else None
            if elem_type is not None and str(elem_type).lower() != "face":
                continue
            idx_attr = subset.GetIndicesAttr()
            subset_indices = idx_attr.Get(time_code) if idx_attr else None
            if subset_indices is None:
                continue
            sub_mat_name, sub_tex_src, sub_uv_xform, sub_mat_props = _resolve_mesh_texture_file(
                child, model_usd, time_code=time_code
            )
            if sub_mat_name not in mtl_defs:
                mtl_defs[sub_mat_name] = {"texture": sub_tex_src, "props": sub_mat_props}
            elif mtl_defs[sub_mat_name].get("texture") is None and sub_tex_src is not None:
                mtl_defs[sub_mat_name]["texture"] = sub_tex_src
            mat_uv_xform.setdefault(sub_mat_name, sub_uv_xform)
            for fi in subset_indices:
                try:
                    face_mat_by_face_index[int(fi)] = sub_mat_name
                except Exception:
                    continue

        idx = np.array(f_indices, dtype=np.int64)
        counts = np.array(f_counts, dtype=np.int64)
        vt_cache: Dict[Tuple[int, str], int] = {}
        cursor = 0
        face_idx = 0
        for c in counts:
            if c < 3:
                cursor += c
                face_idx += 1
                continue
            mat_name = face_mat_by_face_index.get(face_idx, default_mat_name)
            if cur_mat != mat_name:
                f_lines.append(f"usemtl {mat_name}")
                cur_mat = mat_name
            poly = idx[cursor : cursor + c]
            fv_base = cursor
            cursor += c
            tri_inds = []
            for k in range(1, c - 1):
                tri_inds.append((0, k, k + 1))
            for a, b, cidx in tri_inds:
                loc = [int(poly[a]), int(poly[b]), int(poly[cidx])]
                face_v = [v_off + i + 1 for i in loc]
                if uv_vals is None:
                    f_lines.append(f"f {face_v[0]} {face_v[1]} {face_v[2]}")
                    continue
                uv_loc = []
                for k_local, p_idx in zip([a, b, cidx], loc):
                    uvi = uv_index_for(uv_interp, uv_indices, p_idx, fv_base + k_local, len(uv_vals))
                    if uvi is None:
                        uv_loc = []
                        break
                    key = (int(uvi), mat_name)
                    vt_idx = vt_cache.get(key)
                    if vt_idx is None:
                        uv_t = transform_uv(
                            uv_vals[int(uvi)],
                            mat_uv_xform.get(
                                mat_name,
                                (np.array([1.0, 1.0]), np.array([0.0, 0.0]), 0.0, "repeat", "repeat"),
                            ),
                        )
                        vt_idx = len(vt_lines) + 1
                        vt_lines.append(f"vt {uv_t[0]:.9g} {uv_t[1]:.9g}")
                        vt_cache[key] = vt_idx
                    uv_loc.append(vt_idx)
                if len(uv_loc) == 3:
                    f_lines.append(
                        f"f {face_v[0]}/{uv_loc[0]} {face_v[1]}/{uv_loc[1]} {face_v[2]}/{uv_loc[2]}"
                    )
                else:
                    f_lines.append(f"f {face_v[0]} {face_v[1]} {face_v[2]}")
            face_idx += 1

        v_off += len(pts_w)

    if not v_lines:
        return False

    out_obj_path.parent.mkdir(parents=True, exist_ok=True)
    out_tex_dir.mkdir(parents=True, exist_ok=True)
    mtl_name = f"{out_obj_path.stem}.mtl"
    out_mtl_path = out_obj_path.with_suffix(".mtl")

    mtl_lines: List[str] = []
    for mat_name, entry in mtl_defs.items():
        tex_src = entry.get("texture")
        props = entry.get("props") if isinstance(entry.get("props"), dict) else {}
        kd = np.asarray(props.get("kd", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
        if kd.size < 3:
            kd = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        ks = np.asarray(props.get("ks", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
        if ks.size < 3:
            ks = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        opacity = float(np.clip(float(props.get("opacity", 1.0)), 0.0, 1.0))
        mtl_lines.extend(
            [
                f"newmtl {mat_name}",
                f"Ka {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}",
                f"Kd {kd[0]:.6f} {kd[1]:.6f} {kd[2]:.6f}",
                f"Ks {ks[0]:.6f} {ks[1]:.6f} {ks[2]:.6f}",
                f"d {opacity:.6f}",
                "illum 2",
            ]
        )
        if tex_src is not None and tex_src.exists():
            if tex_src not in copied_tex:
                dst_name = tex_src.name
                k = 2
                while (out_tex_dir / dst_name).exists():
                    dst_name = f"{tex_src.stem}_{k}{tex_src.suffix}"
                    k += 1
                shutil.copy2(tex_src, out_tex_dir / dst_name)
                copied_tex[tex_src] = dst_name
            tex_rel = f"../textures/{copied_tex[tex_src]}"
            mtl_lines.append(f"map_Kd {tex_rel}")
        mtl_lines.append("")

    obj_lines = [f"mtllib {mtl_name}"] + v_lines + vt_lines + f_lines
    out_obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")
    out_mtl_path.write_text("\n".join(mtl_lines), encoding="utf-8")
    return True


def _write_urdf(
    out_urdf: Path,
    robot_name: str,
    link_names: List[str],
    visual_origin_by_link: Dict[str, Tuple[np.ndarray, np.ndarray]],
    mesh_relpath_by_link: Dict[str, str],
    joints: List[JointInfo],
) -> None:
    lines: List[str] = []
    lines.append('<?xml version="1.0"?>')
    lines.append(f'<robot name="{robot_name}">')

    for ln in link_names:
        lines.append(f'  <link name="{ln}">')
        if ln in mesh_relpath_by_link and ln in visual_origin_by_link:
            v_xyz, v_rpy = visual_origin_by_link[ln]
            mesh_rel = mesh_relpath_by_link[ln]
            lines.append("    <visual>")
            lines.append(
                f'      <origin xyz="{v_xyz[0]:.9f} {v_xyz[1]:.9f} {v_xyz[2]:.9f}" '
                f'rpy="{v_rpy[0]:.9f} {v_rpy[1]:.9f} {v_rpy[2]:.9f}"/>'
            )
            lines.append("      <geometry>")
            lines.append(f'        <mesh filename="{mesh_rel}"/>')
            lines.append("      </geometry>")
            lines.append("    </visual>")
            lines.append("    <collision>")
            lines.append(
                f'      <origin xyz="{v_xyz[0]:.9f} {v_xyz[1]:.9f} {v_xyz[2]:.9f}" '
                f'rpy="{v_rpy[0]:.9f} {v_rpy[1]:.9f} {v_rpy[2]:.9f}"/>'
            )
            lines.append("      <geometry>")
            lines.append(f'        <mesh filename="{mesh_rel}"/>')
            lines.append("      </geometry>")
            lines.append("    </collision>")
        lines.append("  </link>")

    for j in joints:
        lines.append(f'  <joint name="{j.name}" type="{j.jtype}">')
        lines.append(f'    <parent link="{j.parent}"/>')
        lines.append(f'    <child link="{j.child}"/>')
        lines.append(
            f'    <origin xyz="{j.origin_xyz[0]:.9f} {j.origin_xyz[1]:.9f} {j.origin_xyz[2]:.9f}" '
            f'rpy="{j.origin_rpy[0]:.9f} {j.origin_rpy[1]:.9f} {j.origin_rpy[2]:.9f}"/>'
        )
        if j.axis is not None and j.jtype in ("revolute", "prismatic", "continuous"):
            lines.append(f'    <axis xyz="{j.axis[0]:.9f} {j.axis[1]:.9f} {j.axis[2]:.9f}"/>')
        if j.jtype in ("revolute", "prismatic"):
            lo = float(j.lower if j.lower is not None else 0.0)
            hi = float(j.upper if j.upper is not None else 0.0)
            lines.append(f'    <limit lower="{lo:.9f}" upper="{hi:.9f}" effort="1000" velocity="100"/>')
        lines.append("  </joint>")

    lines.append("</robot>")
    out_urdf.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _joint_axis_alignment_report(
    joints: List[JointInfo],
    root_link: str,
    body_world_tf: Dict[str, np.ndarray],
    body_world_rigid_tf: Dict[str, np.ndarray],
) -> Dict[str, object]:
    root_world = body_world_rigid_tf.get(root_link, np.eye(4, dtype=np.float64))
    link_world_tf: Dict[str, np.ndarray] = {root_link: root_world}
    unresolved = list(range(len(joints)))
    progressed = True
    while unresolved and progressed:
        progressed = False
        nxt = []
        for ji in unresolved:
            j = joints[ji]
            if j.parent not in link_world_tf:
                nxt.append(ji)
                continue
            p_w = body_world_tf.get(j.parent, link_world_tf[j.parent])
            joint_world_expected = _rigidize_transform(p_w @ j.parent_to_joint)
            parent_link_w = link_world_tf[j.parent]
            origin_m = _origin_matrix(j.origin_xyz, j.origin_rpy)
            joint_world_actual = _rigidize_transform(parent_link_w @ origin_m)
            if j.child not in link_world_tf:
                link_world_tf[j.child] = joint_world_expected
            # store on object for report
            j._joint_world_expected = joint_world_expected  # type: ignore[attr-defined]
            j._joint_world_actual = joint_world_actual  # type: ignore[attr-defined]
            progressed = True
        unresolved = nxt

    items = []
    max_pos = 0.0
    max_axis = 0.0
    for j in joints:
        ew = getattr(j, "_joint_world_expected", None)
        aw = getattr(j, "_joint_world_actual", None)
        if ew is None or aw is None:
            continue
        pos_err = float(np.linalg.norm(ew[:3, 3] - aw[:3, 3]))
        axis_err_deg = 0.0
        if j.axis is not None and np.linalg.norm(j.axis) > 0:
            a_local = np.array(j.axis, dtype=np.float64)
            a_local = a_local / np.linalg.norm(a_local)
            a_exp = ew[:3, :3] @ a_local
            a_act = aw[:3, :3] @ a_local
            a_exp = a_exp / max(1e-12, float(np.linalg.norm(a_exp)))
            a_act = a_act / max(1e-12, float(np.linalg.norm(a_act)))
            axis_err_deg = float(np.degrees(np.arccos(np.clip(np.dot(a_exp, a_act), -1.0, 1.0))))
        max_pos = max(max_pos, pos_err)
        max_axis = max(max_axis, axis_err_deg)
        items.append(
            {
                "joint": j.name,
                "type": j.jtype,
                "parent": j.parent,
                "child": j.child,
                "origin_position_error_m": pos_err,
                "axis_error_deg": axis_err_deg,
            }
        )
    return {
        "max_origin_position_error_m": max_pos,
        "max_axis_error_deg": max_axis,
        "joints": items,
    }


def _find_object_asset_dirs(artvip_root: Path) -> List[Path]:
    base = artvip_root / "Articulated_objects"
    if not base.exists():
        return []
    # Only treat folders with primary model_* USD file as object-level assets.
    # This avoids matching nested helper folders like resource/, .thumbs/, etc.
    out: set[Path] = set()
    for f in base.rglob("model_*"):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            continue
        out.add(f.parent)
    return sorted(out)


def _asset_name_from_path(asset_dir: Path, artvip_root: Path) -> str:
    rel = asset_dir.relative_to(artvip_root / "Articulated_objects")
    parts = [_safe_name(x) for x in rel.parts]
    return "__".join(parts)


def _rename_joints_compact(joints: List[JointInfo]) -> Dict[str, str]:
    """
    Rename joints to compact deterministic names: joint_0, joint_1, ...
    Returns old->new mapping.
    """
    mapping: Dict[str, str] = {}
    for i, j in enumerate(joints):
        old = str(j.name)
        new = f"joint_{i}"
        j.name = new
        mapping[old] = new
    return mapping


def _detect_behavior_runtime_needs(asset_dir: Path) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    keywords = {
        "BehaviorScript": "usd_behavior_script",
        "omni.kit.scripting": "omni_kit_scripting",
        "UsdPhysics.DriveAPI": "usd_drive_api",
        "set_visibility": "visibility_logic",
        "dynamic_control": "dynamic_control_api",
    }
    for pyf in sorted(asset_dir.rglob("*.py")):
        try:
            text = pyf.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for key, tag in keywords.items():
            if key in text and tag not in reasons:
                reasons.append(tag)
    return (len(reasons) > 0), reasons


def _convert_one(
    asset_dir: Path,
    artvip_root: Path,
    out_root: Path,
    rebuild_canonical_glb: bool,
    py_exec: str,
    keep_existing: bool,
    sim_code_root: Optional[Path],
    canonical_frames_per_joint: int,
    canonical_fps: int,
) -> Tuple[bool, str]:
    model_usd = _find_model_usd(asset_dir)
    if model_usd is None:
        return False, f"{asset_dir}: model usd/usda/usdc not found"

    stage = Usd.Stage.Open(str(model_usd))
    if not stage:
        return False, f"{asset_dir.name}: failed to open USD stage"

    body_prims = _extract_body_prims(stage)
    if not body_prims:
        return False, f"{asset_dir.name}: no rigid body prims with meshes"

    body_paths = [str(p.GetPath()) for p in body_prims]
    body_path_set = set(body_paths)

    # link names: keep legacy style (link_<number>) for compact labels.
    # Prefer trailing numeric ids from body prim names; fallback to next free id.
    path_to_link: Dict[str, str] = {}
    used_ids: set[int] = set()
    next_fallback_id = 0
    for p in body_prims:
        idx = _extract_trailing_int(p.GetName())
        if idx is None or idx in used_ids:
            while next_fallback_id in used_ids:
                next_fallback_id += 1
            idx = next_fallback_id
            next_fallback_id += 1
        used_ids.add(int(idx))
        ln = f"link_{int(idx)}"
        path_to_link[str(p.GetPath())] = ln

    joints: List[JointInfo] = []
    children_all = set()
    children_non_fixed = set()
    parents_non_fixed = set()

    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if not t.startswith("Physics") or "Joint" not in t:
            continue
        rel_body0 = prim.GetRelationship("physics:body0")
        rel_body1 = prim.GetRelationship("physics:body1")
        body0_targets = [str(x) for x in rel_body0.GetTargets()] if rel_body0 else []
        body1_targets = [str(x) for x in rel_body1.GetTargets()] if rel_body1 else []
        if not body1_targets:
            continue
        b1 = body1_targets[0]
        if b1 not in path_to_link:
            continue
        b0 = body0_targets[0] if body0_targets else None
        if b0 is not None and b0 not in path_to_link:
            b0 = None

        if t == "PhysicsRevoluteJoint":
            jtype = "revolute"
        elif t == "PhysicsPrismaticJoint":
            jtype = "prismatic"
        elif t == "PhysicsFixedJoint":
            jtype = "fixed"
        else:
            continue

        # World-anchor fixed joints (body0 empty) are not needed for URDF topology.
        if jtype == "fixed" and b0 is None:
            continue

        local_pos0 = prim.GetAttribute("physics:localPos0").Get(REST_TIME)
        local_rot0 = prim.GetAttribute("physics:localRot0").Get(REST_TIME)
        local_pos1 = prim.GetAttribute("physics:localPos1").Get(REST_TIME)
        local_rot1 = prim.GetAttribute("physics:localRot1").Get(REST_TIME)

        pos0 = np.array(
            [
                float(local_pos0[0]) if local_pos0 is not None else 0.0,
                float(local_pos0[1]) if local_pos0 is not None else 0.0,
                float(local_pos0[2]) if local_pos0 is not None else 0.0,
            ],
            dtype=np.float64,
        )
        pos1 = np.array(
            [
                float(local_pos1[0]) if local_pos1 is not None else 0.0,
                float(local_pos1[1]) if local_pos1 is not None else 0.0,
                float(local_pos1[2]) if local_pos1 is not None else 0.0,
            ],
            dtype=np.float64,
        )
        rot0 = _quat_to_mat4(local_rot0 if local_rot0 is not None else (1.0, 0.0, 0.0, 0.0))
        rot1 = _quat_to_mat4(local_rot1 if local_rot1 is not None else (1.0, 0.0, 0.0, 0.0))
        parent_to_joint = rot0.copy()
        parent_to_joint[:3, 3] = pos0
        child_to_joint = rot1.copy()
        child_to_joint[:3, 3] = pos1

        axis = None
        if jtype in ("revolute", "prismatic"):
            axis_tok = prim.GetAttribute("physics:axis").Get(REST_TIME)
            axis = AXIS_MAP.get(str(axis_tok), AXIS_MAP.get(str(axis_tok).upper(), np.array([0.0, 0.0, 1.0])))

        lower = None
        upper = None
        if jtype in ("revolute", "prismatic"):
            lo = prim.GetAttribute("physics:lowerLimit").Get(REST_TIME)
            hi = prim.GetAttribute("physics:upperLimit").Get(REST_TIME)
            if lo is not None and np.isfinite(float(lo)):
                lower = float(lo)
            if hi is not None and np.isfinite(float(hi)):
                upper = float(hi)
            if jtype == "revolute":
                # USD revolute limits are degrees.
                if lower is not None:
                    lower = math.radians(lower)
                if upper is not None:
                    upper = math.radians(upper)
                if lower is None and upper is None:
                    jtype = "continuous"

        parent_link = path_to_link[b0] if b0 is not None else ""
        child_link = path_to_link[b1]
        children_all.add(child_link)
        if jtype != "fixed":
            children_non_fixed.add(child_link)
            if parent_link:
                parents_non_fixed.add(parent_link)

        joints.append(
            JointInfo(
                name=_safe_name(prim.GetName()),
                jtype=jtype,
                parent=parent_link,
                child=child_link,
                parent_to_joint=parent_to_joint,
                child_to_joint=child_to_joint,
                origin_xyz=np.zeros(3, dtype=np.float64),
                origin_rpy=np.zeros(3, dtype=np.float64),
                axis=axis,
                lower=lower,
                upper=upper,
            )
        )

    # Absorb child joint frame offsets into the converted link frame instead of
    # introducing helper *_jf_* links. This keeps the final dataset aligned with
    # the repo's expectation that motion/render/VLM operate on visible links.
    helper_links: List[str] = []
    expanded_joints: List[JointInfo] = []
    for j in joints:
        t1_non_identity = not _is_identity_tf(j.child_to_joint)
        if t1_non_identity and j.jtype == "fixed":
            # Fixed joints can safely absorb the child frame directly.
            j.parent_to_joint = j.parent_to_joint @ np.linalg.inv(j.child_to_joint)
        j.child_to_joint = np.eye(4, dtype=np.float64)
        expanded_joints.append(j)
    joints = expanded_joints

    link_names = sorted(list(path_to_link.values()) + helper_links)
    # Recompute root candidates from expanded non-fixed joint graph.
    children_non_fixed = {j.child for j in joints if j.jtype != "fixed"}
    parents_non_fixed = {j.parent for j in joints if j.jtype != "fixed" and j.parent}
    root_candidates = [ln for ln in sorted(parents_non_fixed) if ln not in children_non_fixed]
    if not root_candidates:
        root_candidates = [ln for ln in link_names if ln not in children_non_fixed]
    root_link = root_candidates[0] if root_candidates else link_names[0]

    # fill missing parents for joints with empty body0
    for j in joints:
        if not j.parent:
            j.parent = root_link
            j.name = _safe_name(f"joint_{j.parent}_to_{j.child}")

    # USD body transform in world for each link at rest.
    body_world_tf: Dict[str, np.ndarray] = {}
    body_world_rigid_tf: Dict[str, np.ndarray] = {}
    for bp in body_prims:
        ln = path_to_link[str(bp.GetPath())]
        w = UsdGeom.Xformable(bp).ComputeLocalToWorldTransform(REST_TIME)
        body_world_tf[ln] = _gf_to_np_mat4(w)
        body_world_rigid_tf[ln] = _rigidize_transform(body_world_tf[ln])

    # add fixed joints for disconnected link bodies (e.g., nested rigid body)
    link_by_path = {v: k for k, v in path_to_link.items()}
    child_links = {j.child for j in joints}
    for ln in link_names:
        if ln == root_link or ln in child_links:
            continue
        child_path = link_by_path[ln]
        # nearest ancestor body path
        parent_ln = root_link
        cp = Path(child_path)
        for anc in cp.parents:
            anc_s = str(anc)
            if anc_s in path_to_link and path_to_link[anc_s] != ln:
                parent_ln = path_to_link[anc_s]
                break
        p_w = body_world_tf.get(parent_ln, np.eye(4, dtype=np.float64))
        c_w = body_world_tf.get(ln, np.eye(4, dtype=np.float64))
        p_to_c = np.linalg.inv(p_w) @ c_w
        joints.append(
            JointInfo(
                name=_safe_name(f"joint_{parent_ln}_to_{ln}"),
                jtype="fixed",
                parent=parent_ln,
                child=ln,
                parent_to_joint=p_to_c,
                child_to_joint=np.eye(4, dtype=np.float64),
                origin_xyz=np.zeros(3),
                origin_rpy=np.zeros(3),
                axis=None,
                lower=None,
                upper=None,
            )
        )

    # world meshes by link
    world_meshes: Dict[str, trimesh.Trimesh] = {}
    for bp in body_prims:
        ln = path_to_link[str(bp.GetPath())]
        m = _collect_mesh_world_under_body(bp, body_path_set, time_code=REST_TIME)
        if m is None or len(m.vertices) == 0:
            continue
        world_meshes[ln] = m

    # Resolve URDF link frames in world:
    # - root link frame: root rigid body frame at rest
    # - child link frame: joint frame at rest (from parent body + localPos0/localRot0),
    #   rigidized to avoid non-uniform-scale contamination from USD xforms.
    link_world_tf: Dict[str, np.ndarray] = {
        root_link: body_world_rigid_tf.get(root_link, np.eye(4, dtype=np.float64))
    }
    unresolved = list(range(len(joints)))
    progressed = True
    while unresolved and progressed:
        progressed = False
        nxt = []
        for ji in unresolved:
            j = joints[ji]
            if j.parent not in link_world_tf:
                nxt.append(ji)
                continue
            parent_body_w = body_world_tf.get(j.parent, link_world_tf[j.parent])
            joint_world = _rigidize_transform(parent_body_w @ j.parent_to_joint)
            parent_link_w = link_world_tf[j.parent]
            t_parent_link_to_joint = np.linalg.inv(parent_link_w) @ joint_world
            j.origin_xyz, j.origin_rpy = _matrix_to_xyz_rpy(_rigidize_transform(t_parent_link_to_joint))
            if j.child not in link_world_tf:
                link_world_tf[j.child] = joint_world
            progressed = True
        unresolved = nxt

    # Fallback for disconnected links/joints.
    for ln in link_names:
        link_world_tf.setdefault(ln, body_world_rigid_tf.get(ln, np.eye(4, dtype=np.float64)))
    for ji in unresolved:
        j = joints[ji]
        if j.parent not in link_world_tf:
            j.parent = root_link
        p_w = link_world_tf.get(j.parent, np.eye(4, dtype=np.float64))
        c_w = link_world_tf.get(j.child, body_world_rigid_tf.get(j.child, np.eye(4, dtype=np.float64)))
        j.origin_xyz, j.origin_rpy = _matrix_to_xyz_rpy(_rigidize_transform(np.linalg.inv(p_w) @ c_w))

    # output folder
    asset_name = _asset_name_from_path(asset_dir, artvip_root)
    out_asset = out_root / asset_name
    out_mesh = out_asset / "meshes"
    out_tex = out_asset / "textures"
    if out_asset.exists() and not keep_existing:
        shutil.rmtree(out_asset)
    out_mesh.mkdir(parents=True, exist_ok=True)
    out_tex.mkdir(parents=True, exist_ok=True)

    # write link meshes and visual origins
    local_bbox_center_norm_max = 0.0
    local_bbox_center_norm_by_link: Dict[str, float] = {}
    visual_origin_by_link: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    mesh_relpath_by_link: Dict[str, str] = {}
    body_prim_by_link = {path_to_link[str(bp.GetPath())]: bp for bp in body_prims}

    for ln in link_names:
        if ln not in world_meshes:
            continue
        world_link = _rigidize_transform(link_world_tf.get(ln, np.eye(4, dtype=np.float64)))
        world_to_link = np.linalg.inv(world_link)
        mesh_path = out_mesh / f"{ln}.obj"
        wrote_textured = False
        bp = body_prim_by_link.get(ln)
        if bp is not None:
            try:
                wrote_textured = _export_textured_obj_for_link(
                    body_prim=bp,
                    body_paths=body_path_set,
                    model_usd=model_usd,
                    out_obj_path=mesh_path,
                    out_tex_dir=out_tex,
                    extra_tf=world_to_link,
                    time_code=REST_TIME,
                )
            except Exception:
                wrote_textured = False
        if not wrote_textured:
            local_mesh = world_meshes[ln].copy()
            local_mesh.apply_transform(world_to_link)
            _write_obj(local_mesh, mesh_path)
        local_mesh_for_check = world_meshes[ln].copy()
        local_mesh_for_check.apply_transform(world_to_link)
        c = np.array(local_mesh_for_check.bounding_box.centroid, dtype=np.float64)
        c_norm = float(np.linalg.norm(c))
        local_bbox_center_norm_by_link[ln] = c_norm
        local_bbox_center_norm_max = max(local_bbox_center_norm_max, c_norm)
        mesh_relpath_by_link[ln] = f"./meshes/{ln}.obj"
        visual_origin_by_link[ln] = (np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64))

    mesh_links = [ln for ln in link_names if ln in mesh_relpath_by_link]
    helper_links_final = [ln for ln in link_names if ln not in mesh_relpath_by_link]
    link_names_final = mesh_links + helper_links_final
    joints_final = [j for j in joints if j.parent in link_names_final and j.child in link_names_final and j.parent != j.child]
    if root_link not in link_names_final and link_names_final:
        root_link = link_names_final[0]

    # ensure connectivity: if a joint parent missing, attach to root
    for j in joints_final:
        if j.parent not in link_names_final:
            j.parent = root_link

    joint_name_map = _rename_joints_compact(joints_final)

    _write_urdf(
        out_asset / "mobility.urdf",
        asset_name,
        link_names_final,
        visual_origin_by_link,
        mesh_relpath_by_link,
        joints_final,
    )

    axis_report = _joint_axis_alignment_report(joints_final, root_link, body_world_tf, body_world_rigid_tf)
    (out_asset / "axis_alignment_report.json").write_text(json.dumps(axis_report, indent=2), encoding="utf-8")

    # copy source usd and optional sim scripts
    shutil.copy2(model_usd, out_asset / f"source_model{model_usd.suffix}")
    sim_root = out_asset / "sim_scripts"
    external_sim_root = (sim_code_root / asset_name) if sim_code_root is not None else None
    py_files = sorted(asset_dir.rglob("*.py"))
    for pyf in py_files:
        rel = pyf.relative_to(asset_dir)
        dst = sim_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pyf, dst)
        if external_sim_root is not None:
            ext_dst = external_sim_root / "python" / rel
            ext_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pyf, ext_dst)
    isaac_dirs = [d for d in asset_dir.rglob("*") if d.is_dir() and "isaac" in d.name.lower()]
    for d in isaac_dirs:
        rel = d.relative_to(asset_dir)
        dst = out_asset / "isaac_sim" / rel
        if dst.exists():
            continue
        shutil.copytree(d, dst)
        if external_sim_root is not None:
            ext_dst = external_sim_root / "isaac_sim_dirs" / rel
            if not ext_dst.exists():
                shutil.copytree(d, ext_dst)
    if external_sim_root is not None:
        manifest = {
            "asset_name": asset_name,
            "source_asset_dir": str(asset_dir),
            "python_scripts_count": len(py_files),
            "isaac_dirs_count": len(isaac_dirs),
            "saved_under": str(external_sim_root),
        }
        (external_sim_root / "sim_code_manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (external_sim_root / "sim_code_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # canonical glb
    out_glb = out_asset / f"animated_textured_{asset_name}.glb"
    if rebuild_canonical_glb:
        cmd = [
            py_exec,
            "tools/build_textured_animated_glb.py",
            "--asset_root",
            str(out_asset),
            "--build_mode",
            "urdf_preview",
            "--out_glb",
            str(out_glb),
            "--fps",
            str(int(canonical_fps)),
            "--frames_per_joint",
            str(int(canonical_frames_per_joint)),
            "--initial_pose_mode",
            "zeros",
        ]
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
        if proc.returncode != 0:
            return False, f"{asset_name}: built urdf but glb build failed: {proc.stderr[:220]}"
    else:
        # fallback: copy source usd as placeholder reference (run_plan will ignore if not glb)
        pass

    needs_behavior_runtime, behavior_reasons = _detect_behavior_runtime_needs(asset_dir)
    meta = {
        "asset_name": asset_name,
        "source_asset_dir": str(asset_dir),
        "source_model": str(model_usd),
        "num_links": len(link_names_final),
        "num_joints": len(joints_final),
        "root_link": root_link,
        "joint_name_map_old_to_new": joint_name_map,
        "copied_sim_scripts": len(py_files),
        "external_sim_code_root": str(external_sim_root) if external_sim_root is not None else None,
        "axis_alignment_report": "axis_alignment_report.json",
        "max_origin_position_error_m": axis_report.get("max_origin_position_error_m"),
        "max_axis_error_deg": axis_report.get("max_axis_error_deg"),
        "mesh_localization_bbox_center_norm_max": local_bbox_center_norm_max,
        "mesh_localization_bbox_center_norm_by_link": local_bbox_center_norm_by_link,
        "needs_behavior_runtime": needs_behavior_runtime,
        "behavior_runtime_reasons": behavior_reasons,
    }
    (out_asset / "conversion_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return True, f"{asset_name}: links={len(link_names_final)} joints={len(joints_final)} scripts={len(py_files)}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert ArtVIP articulated object assets to local data format")
    ap.add_argument("--artvip_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument(
        "--sim_code_root",
        type=Path,
        default=None,
        help="Optional separate folder to store Isaac/behavior python scripts per asset",
    )
    ap.add_argument("--asset_dir", type=Path, default=None, help="Convert only one asset directory")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rebuild_canonical_glb", action="store_true")
    ap.add_argument("--keep_existing", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--canonical_frames_per_joint", type=int, default=24)
    ap.add_argument("--canonical_fps", type=int, default=30)
    args = ap.parse_args()

    artvip_root = args.artvip_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    sim_code_root = args.sim_code_root.resolve() if args.sim_code_root is not None else None
    if sim_code_root is not None:
        sim_code_root.mkdir(parents=True, exist_ok=True)

    if args.asset_dir is not None:
        asset_dirs = [args.asset_dir.resolve()]
    else:
        asset_dirs = _find_object_asset_dirs(artvip_root)

    print(f"ArtVIP root: {artvip_root}")
    print(f"Output root: {out_root}")
    if sim_code_root is not None:
        print(f"Sim code root: {sim_code_root}")
    print(f"Assets: {len(asset_dirs)}")
    print(f"Workers: {args.workers}")

    ok = 0
    fail = 0
    fails: List[str] = []

    def run_one(d: Path):
        return _convert_one(
            d,
            artvip_root,
            out_root,
            args.rebuild_canonical_glb,
            args.python,
            args.keep_existing,
            sim_code_root,
            args.canonical_frames_per_joint,
            args.canonical_fps,
        )

    if args.workers <= 1:
        for i, d in enumerate(asset_dirs, start=1):
            good, msg = run_one(d)
            print(f"[{i}/{len(asset_dirs)}] {'OK ' if good else 'ERR'} {msg}")
            if good:
                ok += 1
            else:
                fail += 1
                fails.append(msg)
    else:
        futs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, d in enumerate(asset_dirs, start=1):
                fut = ex.submit(run_one, d)
                futs[fut] = (i, d)
            for fut in concurrent.futures.as_completed(futs):
                i, d = futs[fut]
                try:
                    good, msg = fut.result()
                except Exception as e:
                    good, msg = False, f"{d}: exception {type(e).__name__}: {e}"
                print(f"[{i}/{len(asset_dirs)}] {'OK ' if good else 'ERR'} {msg}")
                if good:
                    ok += 1
                else:
                    fail += 1
                    fails.append(msg)

    report = {
        "artvip_root": str(artvip_root),
        "out_root": str(out_root),
        "num_assets": len(asset_dirs),
        "ok": ok,
        "failed": fail,
        "failures": fails,
    }
    rp = out_root / "_artvip_conversion_report.json"
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Done. ok={ok} failed={fail}")
    print(f"Report: {rp}")


if __name__ == "__main__":
    main()
