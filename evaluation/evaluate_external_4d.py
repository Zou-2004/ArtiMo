#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
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


_UFBX_SCENE_KEEPALIVE: list[Any] = []


@dataclass
class RawMeshSequence:
    path: Path
    vertices_by_frame: list[np.ndarray]
    faces: np.ndarray
    source: str
    times_s: list[float] | None = None
    component_vertex_indices: list[np.ndarray] | None = None
    component_face_indices: list[np.ndarray] | None = None
    component_names: list[str] | None = None


def _fbx_blendshape_frame_index(name: str) -> int | None:
    for prefix in ("Frame_", "Key_"):
        if not name.startswith(prefix):
            continue
        try:
            return int(name[len(prefix) :])
        except Exception:
            return None
    return None


@dataclass
class Component:
    component_id: int
    face_indices: np.ndarray
    vertex_indices: np.ndarray
    num_faces: int
    num_vertices: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    centroid: np.ndarray


@dataclass
class TargetComponent:
    component_id: int
    link: str
    node_name: str
    sample_count: int
    points: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    centroid: np.ndarray


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for k, v in list(flat.items()):
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(flat)


def _vec3_list_to_array(seq: Any, count: int | None = None) -> np.ndarray:
    if count is None:
        count = len(seq)
    out = np.empty((int(count), 3), dtype=np.float32)
    for i, v in enumerate(seq):
        if i >= count:
            break
        out[i] = (float(v.x), float(v.y), float(v.z))
    return out


def _ufbx_matrix_to_numpy(mat: Any, unit_meters: float) -> tuple[np.ndarray, np.ndarray]:
    """Return row-vector transform parts for an FBX matrix in metric units."""
    cols = []
    for name in ("c0", "c1", "c2"):
        v = getattr(mat, name)
        cols.append([float(v.x), float(v.y), float(v.z)])
    linear = np.asarray(cols, dtype=np.float32).T * float(unit_meters)
    t = getattr(mat, "c3")
    translation = np.asarray([float(t.x), float(t.y), float(t.z)], dtype=np.float32) * float(unit_meters)
    return linear, translation


def _fbx_mesh_instances(mesh: Any) -> list[Any]:
    try:
        instances = list(getattr(mesh, "instances", []) or [])
    except Exception:
        instances = []
    return instances or [None]


def _load_fbx_sequence(path: Path) -> RawMeshSequence:
    try:
        import ufbx
    except Exception as exc:
        raise RuntimeError(
            "FBX input needs the Python ufbx package in the active environment. "
            "Install with: conda run -n casual_agent python -m pip install ufbx"
        ) from exc

    scene = ufbx.load_file(str(path))
    try:
        meshes = list(scene.meshes)
        if not meshes:
            raise ValueError(f"No meshes found in FBX: {path}")

        all_shapes = list(scene.blend_shapes)
        shape_cursor = 0
        unit_meters = float(getattr(getattr(scene, "settings", None), "unit_meters", 1.0) or 1.0)
        mesh_chunks: list[tuple[np.ndarray, np.ndarray, list[tuple[int, np.ndarray, np.ndarray]]]] = []
        component_vertex_indices: list[np.ndarray] = []
        component_face_indices: list[np.ndarray] = []
        component_names: list[str] = []
        vertex_offset = 0
        face_offset = 0
        for mesh in meshes:
            base_local = _vec3_list_to_array(mesh.vertices, int(mesh.num_vertices))
            faces_local = np.asarray(list(mesh.vertex_indices), dtype=np.int64)
            if faces_local.size % 3 != 0:
                raise ValueError(f"Only triangulated FBX meshes are supported for now, got {faces_local.size} indices")
            faces_local = faces_local.reshape(-1, 3)

            frame_shapes_local: list[tuple[int, np.ndarray, np.ndarray]] = []
            while shape_cursor < len(all_shapes):
                shape = all_shapes[shape_cursor]
                name = str(getattr(shape, "name", ""))
                order = _fbx_blendshape_frame_index(name)
                if order is None:
                    break
                if frame_shapes_local and order == 0:
                    break
                # Some Animate3D FBX exports contain a malformed/sparse Key_000
                # before the real dense Key_001..Key_N shapes. Treat it as the
                # base frame instead of stopping the whole mesh sequence.
                if int(shape.num_offsets) != int(mesh.num_vertices):
                    if order == 0:
                        idx = np.zeros((0,), dtype=np.int64)
                        offsets = np.zeros((0, 3), dtype=np.float32)
                        frame_shapes_local.append((order, idx, offsets))
                        shape_cursor += 1
                        continue
                    break
                idx = np.fromiter((int(i) for i in shape.offset_vertices), dtype=np.int64, count=int(shape.num_offsets))
                offsets = _vec3_list_to_array(shape.position_offsets, int(shape.num_offsets))
                frame_shapes_local.append((order, idx, offsets))
                shape_cursor += 1
            frame_shapes_local.sort(key=lambda x: x[0])

            for inst in _fbx_mesh_instances(mesh):
                mat = getattr(inst, "geometry_to_world", None) or getattr(inst, "node_to_world", None)
                if mat is not None:
                    linear, translation = _ufbx_matrix_to_numpy(mat, unit_meters)
                else:
                    linear = np.eye(3, dtype=np.float32) * unit_meters
                    translation = np.zeros((3,), dtype=np.float32)
                base = base_local @ linear.T + translation
                frame_shapes = [(order, idx, offsets @ linear.T) for order, idx, offsets in frame_shapes_local]
                faces = faces_local.copy()

                mesh_chunks.append((base, faces, frame_shapes))
                component_vertex_indices.append(np.arange(vertex_offset, vertex_offset + len(base), dtype=np.int64))
                component_face_indices.append(np.arange(face_offset, face_offset + len(faces), dtype=np.int64))
                default_name = f"mesh_{len(component_names)}"
                mesh_name = str(getattr(mesh, "name", default_name) or default_name)
                inst_name = str(getattr(inst, "name", "") or "") if inst is not None else ""
                component_names.append(inst_name or mesh_name)
                vertex_offset += len(base)
                face_offset += len(faces)

        frame_count = max((len(shapes) for _, _, shapes in mesh_chunks), default=0)
        if frame_count <= 0:
            frame_count = 1

        vertices_by_frame: list[np.ndarray] = []
        for fi in range(frame_count):
            chunks = []
            for base, _faces, frame_shapes in mesh_chunks:
                verts = base.copy()
                if frame_shapes:
                    _order, idx, offsets = frame_shapes[min(fi, len(frame_shapes) - 1)]
                    valid = (idx >= 0) & (idx < len(verts))
                    if np.any(valid):
                        verts[idx[valid]] += offsets[valid]
                chunks.append(verts)
            vertices_by_frame.append(np.concatenate(chunks, axis=0))

        all_faces = []
        face_vertex_offset = 0
        for base, faces, _frame_shapes in mesh_chunks:
            all_faces.append(faces + face_vertex_offset)
            face_vertex_offset += len(base)
        faces = np.concatenate(all_faces, axis=0).astype(np.int64)
    finally:
        # Some AAM FBX files with many mesh-local blendshape stacks trigger a
        # native crash in ufbx.free().  Let Python/process teardown reclaim the
        # scene instead; this is much safer for evaluation, especially when the
        # caller isolates cases in short-lived processes.
        pass

    if not vertices_by_frame:
        raise ValueError(f"No vertices found in {path}")
    _UFBX_SCENE_KEEPALIVE.append(scene)
    return RawMeshSequence(
        path=path,
        vertices_by_frame=vertices_by_frame,
        faces=faces,
        source="fbx_blendshape" if len(meshes) == 1 else "fbx_blendshape_multi_mesh",
        component_vertex_indices=component_vertex_indices,
        component_face_indices=component_face_indices,
        component_names=component_names,
    )


def _gltf_node_static_tf(node: Any) -> np.ndarray:
    if getattr(node, "matrix", None) is not None:
        try:
            return np.asarray(node.matrix, dtype=np.float32).reshape(4, 4).T
        except Exception:
            pass
    return ev._trs_to_matrix(
        getattr(node, "translation", None),
        getattr(node, "rotation", None),
        getattr(node, "scale", None),
    ).astype(np.float32)


def _gltf_node_anim_tf(node: Any, anim: dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
    if not anim:
        return _gltf_node_static_tf(node)
    translation = anim.get("translation")
    rotation = anim.get("rotation")
    scale = anim.get("scale")
    any_values = next(iter(anim.values()))
    idx = min(int(frame_idx), len(any_values) - 1)
    return ev._trs_to_matrix(
        translation[idx] if translation is not None else getattr(node, "translation", None),
        rotation[idx] if rotation is not None else getattr(node, "rotation", None),
        scale[idx] if scale is not None else getattr(node, "scale", None),
    ).astype(np.float32)


def _load_glb_sequence(path: Path) -> RawMeshSequence:
    from pygltflib import GLTF2

    gltf = GLTF2().load(str(path))
    if not gltf.nodes or not gltf.meshes:
        raise ValueError(f"GLB has no mesh nodes: {path}")

    node_anim: dict[int, dict[str, np.ndarray]] = {}
    times: np.ndarray | None = None
    if gltf.animations:
        anim = gltf.animations[0]
        for ch in anim.channels or []:
            target = ch.target
            if target is None or target.node is None:
                continue
            sampler = anim.samplers[int(ch.sampler)]
            t_vals = np.asarray(ev._read_gltf_accessor(gltf, int(sampler.input)), dtype=np.float32)
            out_vals = np.asarray(ev._read_gltf_accessor(gltf, int(sampler.output)), dtype=np.float32)
            if times is None or len(t_vals) > len(times):
                times = t_vals
            node_anim.setdefault(int(target.node), {})[str(target.path)] = out_vals
    if times is None or len(times) == 0:
        times = np.asarray([0.0], dtype=np.float32)

    parent_by_child: dict[int, int] = {}
    for pi, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parent_by_child[int(child)] = int(pi)

    mesh_nodes = [i for i, n in enumerate(gltf.nodes) if n.mesh is not None]
    node_geometry: list[tuple[int, np.ndarray, np.ndarray]] = []
    vertex_face_offset = 0
    face_row_offset = 0
    vertex_offset = 0
    all_faces = []
    component_vertex_indices: list[np.ndarray] = []
    component_face_indices: list[np.ndarray] = []
    component_names: list[str] = []
    for node_idx in mesh_nodes:
        node = gltf.nodes[node_idx]
        mesh = gltf.meshes[int(node.mesh)]
        node_vertices = []
        node_faces = []
        local_offset = 0
        for prim in mesh.primitives or []:
            pos_idx = getattr(prim.attributes, "POSITION", None)
            if pos_idx is None:
                continue
            verts = np.asarray(ev._read_gltf_accessor(gltf, int(pos_idx)), dtype=np.float32)
            if prim.indices is not None:
                idx = np.asarray(ev._read_gltf_accessor(gltf, int(prim.indices)), dtype=np.int64).reshape(-1)
            else:
                idx = np.arange(len(verts), dtype=np.int64)
            if idx.size % 3 != 0:
                continue
            faces = idx.reshape(-1, 3) + local_offset
            node_vertices.append(verts)
            node_faces.append(faces)
            local_offset += len(verts)
        if not node_vertices:
            continue
        vertices = np.concatenate(node_vertices, axis=0)
        faces = np.concatenate(node_faces, axis=0)
        node_geometry.append((node_idx, vertices, faces))
        all_faces.append(faces + vertex_face_offset)
        component_face_indices.append(np.arange(face_row_offset, face_row_offset + len(faces), dtype=np.int64))
        face_row_offset += len(faces)
        vertex_face_offset += len(vertices)
        component_vertex_indices.append(np.arange(vertex_offset, vertex_offset + len(vertices), dtype=np.int64))
        component_names.append(str(node.name or f"node_{node_idx}"))
        vertex_offset += len(vertices)

    if not node_geometry:
        raise ValueError(f"Could not extract GLB primitive geometry: {path}")

    def world_tf(node_idx: int, frame_idx: int, cache: dict[int, np.ndarray]) -> np.ndarray:
        if node_idx in cache:
            return cache[node_idx]
        node = gltf.nodes[node_idx]
        local = _gltf_node_anim_tf(node, node_anim.get(node_idx, {}), frame_idx)
        parent = parent_by_child.get(node_idx)
        if parent is None:
            out = local
        else:
            out = world_tf(parent, frame_idx, cache) @ local
        cache[node_idx] = out
        return out

    vertices_by_frame: list[np.ndarray] = []
    for fi in range(len(times)):
        cache: dict[int, np.ndarray] = {}
        chunks = []
        for node_idx, vertices, _faces in node_geometry:
            chunks.append(_transform(vertices, world_tf(node_idx, fi, cache)))
        vertices_by_frame.append(np.concatenate(chunks, axis=0).astype(np.float32))

    return RawMeshSequence(
        path=path,
        vertices_by_frame=vertices_by_frame,
        faces=np.concatenate(all_faces, axis=0).astype(np.int64),
        source="glb_whole_mesh",
        times_s=[float(t) for t in times.tolist()],
        component_vertex_indices=component_vertex_indices,
        component_face_indices=component_face_indices,
        component_names=component_names,
    )


def _scene_to_single_mesh(obj: Any) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        meshes = []
        for node_name in obj.graph.nodes_geometry:
            tf, geom_name = obj.graph[node_name]
            geom = obj.geometry[geom_name]
            if not isinstance(geom, trimesh.Trimesh) or len(geom.vertices) == 0:
                continue
            mesh = geom.copy()
            mesh.apply_transform(tf)
            meshes.append(mesh)
        if meshes:
            return trimesh.util.concatenate(meshes)
    raise ValueError("Could not convert input scene to a mesh")


def _load_mesh_sequence(path: Path) -> RawMeshSequence:
    if path.suffix.lower() == ".fbx":
        return _load_fbx_sequence(path)
    if path.suffix.lower() in {".glb", ".gltf"}:
        return _load_glb_sequence(path)
    loaded = trimesh.load(path, force="scene", process=False)
    mesh = _scene_to_single_mesh(loaded)
    return RawMeshSequence(
        path=path,
        vertices_by_frame=[np.asarray(mesh.vertices, dtype=np.float32)],
        faces=np.asarray(mesh.faces, dtype=np.int64),
        source=f"static_{path.suffix.lower().lstrip('.') or 'mesh'}",
        times_s=[0.0],
    )


def _glb_frame_vertices_by_node(path: Path, frame_idx: int = 0) -> dict[str, np.ndarray]:
    from pygltflib import GLTF2

    gltf = GLTF2().load(str(path))
    if not gltf.nodes or not gltf.meshes:
        return {}

    node_anim: dict[int, dict[str, np.ndarray]] = {}
    if gltf.animations:
        anim = gltf.animations[0]
        for ch in anim.channels or []:
            target = ch.target
            if target is None or target.node is None:
                continue
            sampler = anim.samplers[int(ch.sampler)]
            out_vals = np.asarray(ev._read_gltf_accessor(gltf, int(sampler.output)), dtype=np.float32)
            node_anim.setdefault(int(target.node), {})[str(target.path)] = out_vals

    parent_by_child: dict[int, int] = {}
    for pi, node in enumerate(gltf.nodes):
        for child in node.children or []:
            parent_by_child[int(child)] = int(pi)

    def world_tf(node_idx: int, cache: dict[int, np.ndarray]) -> np.ndarray:
        if node_idx in cache:
            return cache[node_idx]
        node = gltf.nodes[node_idx]
        local = _gltf_node_anim_tf(node, node_anim.get(node_idx, {}), frame_idx)
        parent = parent_by_child.get(node_idx)
        out = local if parent is None else world_tf(parent, cache) @ local
        cache[node_idx] = out
        return out

    cache: dict[int, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    for node_idx, node in enumerate(gltf.nodes):
        if node.mesh is None:
            continue
        chunks = []
        mesh = gltf.meshes[int(node.mesh)]
        for prim in mesh.primitives or []:
            pos_idx = getattr(prim.attributes, "POSITION", None)
            if pos_idx is None:
                continue
            verts = np.asarray(ev._read_gltf_accessor(gltf, int(pos_idx)), dtype=np.float32)
            chunks.append(_transform(verts, world_tf(node_idx, cache)))
        if chunks:
            out[str(node.name or f"node_{node_idx}")] = np.concatenate(chunks, axis=0)
    return out


def _gt_glb_first_frame_targets(
    gt_glb: Path,
    asset: ev.AssetGeometry,
    num_points_per_link: int = 2048,
) -> tuple[dict[str, np.ndarray], list[TargetComponent]]:
    scene = trimesh.load(gt_glb, force="scene", process=False)
    if not isinstance(scene, trimesh.Scene):
        raise ValueError(f"GT GLB did not load as scene: {gt_glb}")
    link_to_nodes = ev._link_nodes_from_animated_scene(asset, scene)
    vertices_by_node = _glb_frame_vertices_by_node(gt_glb, frame_idx=0)
    out: dict[str, np.ndarray] = {}
    target_components: list[TargetComponent] = []
    for link in asset.visual_links:
        node_names = link_to_nodes.get(link, [])
        sample_count = max(1, int(math.ceil(max(1, int(num_points_per_link)) / max(1, len(node_names)))))
        chunks = []
        for node_name in node_names:
            pts = vertices_by_node.get(str(node_name))
            if pts is not None and len(pts):
                chunks.append(pts)
                target_components.append(
                    TargetComponent(
                        component_id=len(target_components),
                        link=link,
                        node_name=str(node_name),
                        sample_count=sample_count,
                        points=np.asarray(pts, dtype=np.float32),
                        bounds_min=np.min(pts, axis=0),
                        bounds_max=np.max(pts, axis=0),
                        centroid=np.mean(pts, axis=0),
                    )
                )
        out[link] = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3), dtype=np.float32)
    return out, target_components


def _component_split(vertices: np.ndarray, faces: np.ndarray, min_faces: int) -> list[Component]:
    parent = np.arange(len(vertices), dtype=np.int64)

    def find(x: int) -> int:
        while int(parent[x]) != int(x):
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra = find(int(a))
        rb = find(int(b))
        if ra != rb:
            parent[rb] = ra

    for f in faces:
        union(int(f[0]), int(f[1]))
        union(int(f[0]), int(f[2]))

    by_root: dict[int, list[int]] = {}
    for fi, f in enumerate(faces):
        by_root.setdefault(find(int(f[0])), []).append(fi)

    comps: list[Component] = []
    for cid, face_ids in enumerate(sorted(by_root.values(), key=len, reverse=True)):
        if len(face_ids) < int(min_faces):
            continue
        face_idx = np.asarray(face_ids, dtype=np.int64)
        vert_idx = np.unique(faces[face_idx].reshape(-1))
        pts = vertices[vert_idx]
        comps.append(
            Component(
                component_id=cid,
                face_indices=face_idx,
                vertex_indices=vert_idx,
                num_faces=int(len(face_idx)),
                num_vertices=int(len(vert_idx)),
                bounds_min=np.min(pts, axis=0),
                bounds_max=np.max(pts, axis=0),
                centroid=np.mean(pts, axis=0),
            )
        )
    return comps


def _components_from_vertex_groups(
    vertices: np.ndarray,
    groups: list[np.ndarray],
    face_groups: list[np.ndarray] | None = None,
) -> list[Component]:
    comps: list[Component] = []
    for cid, raw_idx in enumerate(groups):
        vert_idx = np.asarray(raw_idx, dtype=np.int64)
        if len(vert_idx) == 0:
            continue
        pts = vertices[vert_idx]
        face_idx = (
            np.asarray(face_groups[cid], dtype=np.int64)
            if face_groups is not None and cid < len(face_groups)
            else np.zeros((0,), dtype=np.int64)
        )
        comps.append(
            Component(
                component_id=cid,
                face_indices=face_idx,
                vertex_indices=vert_idx,
                num_faces=int(len(face_idx)),
                num_vertices=int(len(vert_idx)),
                bounds_min=np.min(pts, axis=0),
                bounds_max=np.max(pts, axis=0),
                centroid=np.mean(pts, axis=0),
            )
        )
    return comps


def _local_faces_for_component(num_vertices: int, faces: np.ndarray, comp: Component) -> np.ndarray:
    if len(comp.face_indices) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    global_faces = np.asarray(faces[comp.face_indices], dtype=np.int64)
    inv = np.full((int(num_vertices),), -1, dtype=np.int64)
    inv[np.asarray(comp.vertex_indices, dtype=np.int64)] = np.arange(len(comp.vertex_indices), dtype=np.int64)
    local = inv[global_faces]
    if local.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    local = local[np.all(local >= 0, axis=1)]
    return np.asarray(local, dtype=np.int64)


def _sample_surface_or_vertices(vertices: np.ndarray, local_faces: np.ndarray, n: int, seed_text: str) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    if len(verts) == 0 or int(n) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if local_faces is not None and len(local_faces):
        mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(local_faces, dtype=np.int64), process=False)
        return ev._deterministic_sample(mesh, int(n), seed_text)
    return _sample_rows(verts, int(n), seed=ev.stable_seed(seed_text))


def _sample_recipe(vertices: np.ndarray, local_faces: np.ndarray, n: int, seed_text: str) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32)
    if len(verts) == 0 or int(n) <= 0:
        return np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)
    state = np.random.get_state()
    np.random.seed(ev.stable_seed(seed_text))
    try:
        if local_faces is not None and len(local_faces):
            mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(local_faces, dtype=np.int64), process=False)
            pts, face_idx = trimesh.sample.sample_surface(mesh, int(n))
            face_vertices = np.asarray(local_faces, dtype=np.int64)[np.asarray(face_idx, dtype=np.int64)]
            tri = verts[face_vertices]
            a = tri[:, 0, :]
            b = tri[:, 1, :] - a
            c = tri[:, 2, :] - a
            w = np.asarray(pts, dtype=np.float32) - a
            d00 = np.einsum("ij,ij->i", b, b)
            d01 = np.einsum("ij,ij->i", b, c)
            d11 = np.einsum("ij,ij->i", c, c)
            d20 = np.einsum("ij,ij->i", w, b)
            d21 = np.einsum("ij,ij->i", w, c)
            denom = d00 * d11 - d01 * d01
            denom = np.where(np.abs(denom) < 1.0e-20, 1.0, denom)
            v = (d11 * d20 - d01 * d21) / denom
            w2 = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w2
            bary = np.stack([u, v, w2], axis=1).astype(np.float32)
            return face_vertices.astype(np.int64), bary
        if len(verts) <= int(n):
            idx = np.arange(len(verts), dtype=np.int64)
        else:
            idx = np.random.choice(len(verts), size=int(n), replace=False).astype(np.int64)
        face_vertices = np.repeat(idx[:, None], 3, axis=1)
        bary = np.zeros((len(idx), 3), dtype=np.float32)
        bary[:, 0] = 1.0
        return face_vertices, bary
    finally:
        np.random.set_state(state)


def _recipe_from_surface_samples(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_idx: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=np.float32)
    face_vertices = np.asarray(faces, dtype=np.int64)[np.asarray(face_idx, dtype=np.int64)]
    tri = verts[face_vertices]
    a = tri[:, 0, :]
    b = tri[:, 1, :] - a
    c = tri[:, 2, :] - a
    w = np.asarray(points, dtype=np.float32) - a
    d00 = np.einsum("ij,ij->i", b, b)
    d01 = np.einsum("ij,ij->i", b, c)
    d11 = np.einsum("ij,ij->i", c, c)
    d20 = np.einsum("ij,ij->i", w, b)
    d21 = np.einsum("ij,ij->i", w, c)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1.0e-20, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w2 = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w2
    bary = np.stack([u, v, w2], axis=1).astype(np.float32)
    return face_vertices.astype(np.int64), bary


def _surface_recipes_by_nearest_link(
    vertices0_aligned: np.ndarray,
    faces: np.ndarray,
    links: list[str],
    link_points_source: dict[str, np.ndarray],
    points_per_link: int,
    max_link_points: int,
    seed_text: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    verts = np.asarray(vertices0_aligned, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int64)
    recipes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    empty = (np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3), dtype=np.float32))
    if len(verts) == 0 or len(faces_arr) == 0 or int(points_per_link) <= 0:
        return {ln: empty for ln in links}

    active_links = [ln for ln in links if ln in link_points_source and len(link_points_source[ln])]
    if not active_links:
        return {ln: empty for ln in links}

    state = np.random.get_state()
    np.random.seed(ev.stable_seed(seed_text))
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces_arr, process=False)
        sample_count = max(int(points_per_link) * max(1, len(active_links)) * 8, int(points_per_link))
        sampled, face_idx = trimesh.sample.sample_surface(mesh, sample_count)
    except Exception:
        return {ln: _sample_recipe(verts, faces_arr, int(points_per_link), f"{seed_text}:{ln}") for ln in links}
    finally:
        np.random.set_state(state)

    sampled = np.asarray(sampled, dtype=np.float32)
    face_vertices, bary = _recipe_from_surface_samples(verts, faces_arr, np.asarray(face_idx, dtype=np.int64), sampled)

    dist_cols = []
    for i, ln in enumerate(active_links):
        link_pts = _sample_rows(np.asarray(link_points_source[ln], dtype=np.float32), int(max_link_points), seed=701 + i)
        vals = []
        for start in range(0, len(sampled), 2048):
            chunk = sampled[start : start + 2048]
            d2 = np.sum((chunk[:, None, :] - link_pts[None, :, :]) ** 2, axis=2)
            vals.append(np.min(d2, axis=1))
        dist_cols.append(np.concatenate(vals, axis=0))
    dist = np.stack(dist_cols, axis=1)
    labels = np.argmin(dist, axis=1)

    rng = np.random.default_rng(ev.stable_seed(f"{seed_text}:select"))
    all_idx = np.arange(len(sampled), dtype=np.int64)
    for ln in links:
        if ln not in active_links:
            recipes[ln] = empty
            continue
        li = active_links.index(ln)
        assigned = np.nonzero(labels == li)[0].astype(np.int64)
        if len(assigned) >= int(points_per_link):
            chosen = rng.choice(assigned, size=int(points_per_link), replace=False).astype(np.int64)
        else:
            nearest = np.argsort(dist[:, li]).astype(np.int64)
            if len(assigned) == 0:
                chosen = nearest[: int(points_per_link)]
            else:
                need = int(points_per_link) - len(assigned)
                supplement = [idx for idx in nearest.tolist() if idx not in set(assigned.tolist())][:need]
                chosen = np.asarray([*assigned.tolist(), *supplement], dtype=np.int64)
        recipes[ln] = (face_vertices[chosen].astype(np.int64), bary[chosen].astype(np.float32))
    return recipes


def _nearest_indices(query: np.ndarray, samples: np.ndarray, chunk: int = 1024) -> np.ndarray:
    q = np.asarray(query, dtype=np.float32)
    s = np.asarray(samples, dtype=np.float32)
    if len(q) == 0 or len(s) == 0:
        return np.zeros((0,), dtype=np.int64)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        return np.asarray(cKDTree(s).query(q, k=1)[1], dtype=np.int64)
    except Exception:
        out = []
        for start in range(0, len(q), int(chunk)):
            qq = q[start : start + int(chunk)]
            d2 = np.sum((qq[:, None, :] - s[None, :, :]) ** 2, axis=2)
            out.append(np.argmin(d2, axis=1).astype(np.int64))
        return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=np.int64)


def _surface_recipes_by_gt_nearest_pred(
    vertices0_aligned: np.ndarray,
    faces: np.ndarray,
    links: list[str],
    link_points_source: dict[str, np.ndarray],
    points_per_link: int,
    seed_text: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create pred-surface sample recipes by querying from GT link samples.

    Vertex-level predictions often arrive as a single unsegmented surface.  A
    forward Voronoi split of prediction points leaves GT areas uncovered when a
    boundary is wrong.  For evaluation we instead ask: for every sampled GT link
    point, which prediction surface point is closest in the aligned first frame?
    The stored prediction face/barycentric recipe is then replayed over time.
    """
    empty = (
        np.zeros((0, 3), dtype=np.int64),
        np.zeros((0, 3), dtype=np.float32),
    )
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices0_aligned, dtype=np.float32), faces=np.asarray(faces), process=False)
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        return {ln: empty for ln in links}

    active_links = [ln for ln in links if ln in link_points_source and len(link_points_source[ln])]
    if not active_links:
        return {ln: empty for ln in links}

    # Oversample the prediction surface so nearest-neighbor queries approximate
    # closest-point correspondences without depending on optional proximity deps.
    sample_count = max(int(points_per_link) * max(1, len(active_links)) * 16, int(points_per_link) * 32, 20000)
    sampled, face_idx = trimesh.sample.sample_surface(mesh, int(sample_count), seed=ev.stable_seed(f"{seed_text}:pred_surface"))
    face_vertices, bary = _recipe_from_surface_samples(np.asarray(vertices0_aligned, dtype=np.float32), np.asarray(faces), face_idx, sampled)

    rng = np.random.default_rng(ev.stable_seed(f"{seed_text}:gt_select"))
    recipes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for li, ln in enumerate(links):
        if ln not in active_links:
            recipes[ln] = empty
            continue
        gt_pts = np.asarray(link_points_source[ln], dtype=np.float32)
        if len(gt_pts) == 0:
            recipes[ln] = empty
            continue
        if len(gt_pts) >= int(points_per_link):
            gt_idx = rng.choice(len(gt_pts), size=int(points_per_link), replace=False)
        else:
            gt_idx = rng.choice(len(gt_pts), size=int(points_per_link), replace=True)
        query = gt_pts[np.asarray(gt_idx, dtype=np.int64)]
        nearest = _nearest_indices(query, sampled)
        recipes[ln] = (face_vertices[nearest].astype(np.int64), bary[nearest].astype(np.float32))
    return recipes


def _apply_sample_recipe(vertices: np.ndarray, recipe: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    face_vertices, bary = recipe
    if len(face_vertices) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    tri = np.asarray(vertices, dtype=np.float32)[face_vertices]
    return np.einsum("nij,ni->nj", tri, np.asarray(bary, dtype=np.float32)).astype(np.float32)


def _sample_rows(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= int(n):
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), size=int(n), replace=False)
    return pts[idx]


def _mean_nearest(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    vals = []
    b = np.asarray(b, dtype=np.float32)
    for i in range(0, len(a), chunk):
        aa = np.asarray(a[i : i + chunk], dtype=np.float32)
        d2 = np.sum((aa[:, None, :] - b[None, :, :]) ** 2, axis=2)
        vals.append(np.sqrt(np.min(d2, axis=1)))
    return float(np.mean(np.concatenate(vals)))


def _whole_points(link_points: dict[str, np.ndarray], links: list[str]) -> np.ndarray:
    chunks = [np.asarray(link_points[ln], dtype=np.float32) for ln in links if ln in link_points and len(link_points[ln])]
    return np.concatenate(chunks, axis=0)


def _alignment_matrix(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    mode: str,
    sample_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return np.eye(4, dtype=np.float32), {
            "mode": "none",
            "perm": [0, 1, 2],
            "signs": [1.0, 1.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "chamfer": 0.5 * (_mean_nearest(_sample_rows(src_points, int(sample_points), seed=13), _sample_rows(dst_points, int(sample_points), seed=17))
                              + _mean_nearest(_sample_rows(dst_points, int(sample_points), seed=17), _sample_rows(src_points, int(sample_points), seed=13))),
        }
    src = _sample_rows(src_points, int(sample_points), seed=13)
    dst = _sample_rows(dst_points, int(sample_points), seed=17)
    src_min, src_max = np.min(src, axis=0), np.max(src, axis=0)
    dst_min, dst_max = np.min(dst, axis=0), np.max(dst, axis=0)
    src_center = 0.5 * (src_min + src_max)
    dst_center = 0.5 * (dst_min + dst_max)
    src_extent = np.maximum(src_max - src_min, 1.0e-9)
    dst_extent = np.maximum(dst_max - dst_min, 1.0e-9)

    if mode == "scale_translate_3d":
        scale = float(np.linalg.norm(dst_extent) / max(float(np.linalg.norm(src_extent)), 1.0e-9))
        scale_vec = np.asarray([scale, scale, scale], dtype=np.float32)
        linear = np.eye(3, dtype=np.float32) * scale
        transformed = (src - src_center) @ linear.T + dst_center
        score = 0.5 * (_mean_nearest(transformed, dst) + _mean_nearest(dst, transformed))
        tf = np.eye(4, dtype=np.float32)
        tf[:3, :3] = linear
        tf[:3, 3] = dst_center - src_center * scale
        return tf, {
            "mode": mode,
            "perm": [0, 1, 2],
            "signs": [1.0, 1.0, 1.0],
            "scale": scale_vec.tolist(),
            "chamfer": score,
            "src_bounds": [src_min.tolist(), src_max.tolist()],
            "dst_bounds": [dst_min.tolist(), dst_max.tolist()],
        }

    best: tuple[float, np.ndarray, dict[str, Any]] | None = None
    for perm in itertools.permutations(range(3)):
        perm_mat = np.eye(3, dtype=np.float32)[:, list(perm)]
        p_extent = src_extent[list(perm)]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            rot = perm_mat @ np.diag(np.asarray(signs, dtype=np.float32))
            if mode == "axis_extent":
                scale_vec = dst_extent / np.maximum(p_extent, 1.0e-9)
            elif mode == "similarity":
                scale = float(np.linalg.norm(dst_extent) / max(float(np.linalg.norm(p_extent)), 1.0e-9))
                scale_vec = np.asarray([scale, scale, scale], dtype=np.float32)
            else:
                raise ValueError(f"Unknown alignment mode: {mode}")
            linear = np.diag(scale_vec) @ rot
            transformed = (src - src_center) @ linear.T + dst_center
            score = 0.5 * (_mean_nearest(transformed, dst) + _mean_nearest(dst, transformed))
            tf = np.eye(4, dtype=np.float32)
            tf[:3, :3] = linear
            tf[:3, 3] = dst_center - src_center @ linear.T
            info = {
                "mode": mode,
                "perm": list(perm),
                "signs": list(signs),
                "scale": scale_vec.tolist(),
                "chamfer": score,
                "src_bounds": [src_min.tolist(), src_max.tolist()],
                "dst_bounds": [dst_min.tolist(), dst_max.tolist()],
            }
            if best is None or score < best[0]:
                best = (score, tf, info)
    if best is None:
        raise ValueError("Could not estimate alignment")
    return best[1], best[2]


def _transform(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    hom = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    return (hom @ np.asarray(tf, dtype=np.float32).T)[:, :3]


def _bbox_center(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((3,), dtype=np.float32)
    return 0.5 * (np.min(pts, axis=0) + np.max(pts, axis=0))


def _alignment_rotation_and_scale(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    perm = [int(x) for x in info.get("perm", [0, 1, 2])]
    signs = np.asarray(info.get("signs", [1.0, 1.0, 1.0]), dtype=np.float32)
    scale = np.asarray(info.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
    rot = np.eye(3, dtype=np.float32)[:, perm] @ np.diag(signs)
    return rot, scale


def _transform_external_frame(
    vertices: np.ndarray,
    align_tf: np.ndarray,
    align_info: dict[str, Any],
    motion_scale_mode: str,
) -> np.ndarray:
    if str(motion_scale_mode) != "preserve_center_trajectory" or str(align_info.get("mode")) == "none":
        return _transform(vertices, align_tf)
    src_bounds = align_info.get("src_bounds")
    dst_bounds = align_info.get("dst_bounds")
    if not src_bounds or not dst_bounds:
        return _transform(vertices, align_tf)
    rot, scale = _alignment_rotation_and_scale(align_info)
    shape_linear = np.diag(scale) @ rot
    src_center0 = 0.5 * (np.asarray(src_bounds[0], dtype=np.float32) + np.asarray(src_bounds[1], dtype=np.float32))
    dst_center0 = 0.5 * (np.asarray(dst_bounds[0], dtype=np.float32) + np.asarray(dst_bounds[1], dtype=np.float32))
    frame_center = _bbox_center(vertices)
    # Scale the predicted shape into GT units, but keep the object-center
    # trajectory displacement in the prediction's original numeric units.
    center_aligned = dst_center0 + (frame_center - src_center0) @ rot.T
    return (np.asarray(vertices, dtype=np.float32) - frame_center) @ shape_linear.T + center_aligned


def _assign_components_to_links(
    links: list[str],
    link_points_source: dict[str, np.ndarray],
    vertices0_aligned: np.ndarray,
    faces: np.ndarray,
    components: list[Component],
    max_component_points: int,
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    link_points = {
        ln: _sample_rows(np.asarray(link_points_source[ln], dtype=np.float32), max_component_points, seed=101 + i)
        for i, ln in enumerate(links)
        if ln in link_points_source and len(link_points_source[ln])
    }
    assignments: dict[str, list[int]] = {ln: [] for ln in links}
    rows: list[dict[str, Any]] = []
    obj_diag = float(np.linalg.norm(np.max(vertices0_aligned, axis=0) - np.min(vertices0_aligned, axis=0)))
    for comp in components:
        pts = vertices0_aligned[comp.vertex_indices]
        pts_s = _sample_rows(pts, max_component_points, seed=1000 + comp.component_id)
        cb0, cb1 = np.min(pts, axis=0), np.max(pts, axis=0)
        cc = 0.5 * (cb0 + cb1)
        ce = cb1 - cb0
        best_link = None
        best_score = None
        score_by_link = {}
        for ln in links:
            if ln not in link_points:
                continue
            lp = link_points[ln]
            lb0, lb1 = np.min(lp, axis=0), np.max(lp, axis=0)
            lc = 0.5 * (lb0 + lb1)
            le = np.maximum(lb1 - lb0, 1.0e-9)
            nn = _mean_nearest(pts_s, lp)
            center = float(np.linalg.norm(cc - lc))
            extent = float(np.linalg.norm((np.sort(ce) - np.sort(le)) / np.maximum(np.sort(le), 1.0e-6)))
            score = nn + 0.10 * center + 0.01 * obj_diag * extent
            score_by_link[ln] = float(score)
            if best_score is None or score < best_score:
                best_score = float(score)
                best_link = ln
        if best_link is None:
            continue
        assignments[best_link].append(comp.component_id)
        rows.append(
            {
                "component_id": comp.component_id,
                "assigned_link": best_link,
                "num_faces": comp.num_faces,
                "num_vertices": comp.num_vertices,
                "bounds_min": cb0.tolist(),
                "bounds_max": cb1.tolist(),
                "score": best_score,
                "score_by_link": score_by_link,
            }
        )
    return assignments, rows


def _assign_components_to_target_components(
    links: list[str],
    vertices0_aligned: np.ndarray,
    components: list[Component],
    target_components: list[TargetComponent],
    max_component_points: int,
) -> tuple[dict[str, list[int]], list[dict[str, Any]]]:
    assignments: dict[str, list[int]] = {ln: [] for ln in links}
    rows: list[dict[str, Any]] = []
    target_samples = [
        _sample_rows(t.points, max_component_points, seed=501 + i)
        for i, t in enumerate(target_components)
    ]
    obj_diag = float(np.linalg.norm(np.max(vertices0_aligned, axis=0) - np.min(vertices0_aligned, axis=0)))

    comp_infos: list[dict[str, Any]] = []
    cost = np.full((len(target_components), len(components)), np.inf, dtype=np.float64)
    for ci, comp in enumerate(components):
        pts = vertices0_aligned[comp.vertex_indices]
        pts_s = _sample_rows(pts, max_component_points, seed=1500 + comp.component_id)
        cb0, cb1 = np.min(pts, axis=0), np.max(pts, axis=0)
        cc = 0.5 * (cb0 + cb1)
        ce = np.maximum(cb1 - cb0, 1.0e-9)
        score_by_target: dict[int, float] = {}
        for ti, target in enumerate(target_components):
            tb0, tb1 = target.bounds_min, target.bounds_max
            tc = target.centroid
            te = np.maximum(tb1 - tb0, 1.0e-9)
            nn = 0.5 * (_mean_nearest(pts_s, target_samples[ti]) + _mean_nearest(target_samples[ti], pts_s))
            center = float(np.linalg.norm(cc - tc))
            extent = float(np.linalg.norm((np.sort(ce) - np.sort(te)) / np.maximum(np.sort(te), 1.0e-6)))
            score = nn + 0.05 * center + 0.02 * obj_diag * extent
            score_by_target[ti] = float(score)
            cost[ti, ci] = float(score)
        comp_infos.append(
            {
                "component": comp,
                "bounds_min": cb0,
                "bounds_max": cb1,
                "score_by_target": score_by_target,
            }
        )

    if not target_components or not components:
        return assignments, rows

    pairs: list[tuple[int, int]] = []
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        target_idx, comp_idx = linear_sum_assignment(cost)
        pairs = [(int(ti), int(ci)) for ti, ci in zip(target_idx, comp_idx) if np.isfinite(cost[int(ti), int(ci)])]
    except Exception:
        used_targets: set[int] = set()
        used_components: set[int] = set()
        for ti, ci in sorted(
            ((ti, ci) for ti in range(cost.shape[0]) for ci in range(cost.shape[1])),
            key=lambda p: float(cost[p[0], p[1]]),
        ):
            if ti in used_targets or ci in used_components or not np.isfinite(cost[ti, ci]):
                continue
            pairs.append((int(ti), int(ci)))
            used_targets.add(int(ti))
            used_components.add(int(ci))

    def add_assignment(ti: int, ci: int, source: str) -> None:
        target = target_components[int(ti)]
        info = comp_infos[int(ci)]
        comp = info["component"]
        assignments[target.link].append(comp.component_id)
        rows.append(
            {
                "component_id": comp.component_id,
                "assigned_link": target.link,
                "target_component_id": int(ti),
                "assignment_source": source,
                "num_faces": comp.num_faces,
                "num_vertices": comp.num_vertices,
                "bounds_min": info["bounds_min"].tolist(),
                "bounds_max": info["bounds_max"].tolist(),
                "score": float(cost[int(ti), int(ci)]),
                "score_by_target": {str(k): float(v) for k, v in info["score_by_target"].items()},
            }
        )

    for ti, ci in pairs:
        add_assignment(ti, ci, "hungarian")
    return assignments, rows


def _assign_vertices_to_links(
    links: list[str],
    link_points_source: dict[str, np.ndarray],
    vertices0_aligned: np.ndarray,
    max_link_points: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    pts = np.asarray(vertices0_aligned, dtype=np.float32)
    if len(pts) == 0:
        return {ln: np.zeros((0,), dtype=np.int64) for ln in links}, []

    dist_cols = []
    active_links = [ln for ln in links if ln in link_points_source and len(link_points_source[ln])]
    for i, ln in enumerate(active_links):
        link_pts = _sample_rows(np.asarray(link_points_source[ln], dtype=np.float32), max_link_points, seed=301 + i)
        vals = []
        for start in range(0, len(pts), 2048):
            chunk = pts[start : start + 2048]
            d2 = np.sum((chunk[:, None, :] - link_pts[None, :, :]) ** 2, axis=2)
            vals.append(np.min(d2, axis=1))
        dist_cols.append(np.concatenate(vals, axis=0))
    dist = np.stack(dist_cols, axis=1)
    labels = np.argmin(dist, axis=1)

    by_link: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for ln in links:
        if ln in active_links:
            li = active_links.index(ln)
            idx = np.nonzero(labels == li)[0].astype(np.int64)
        else:
            li = -1
            idx = np.zeros((0,), dtype=np.int64)
        by_link[ln] = idx
        if len(idx):
            bounds = [np.min(pts[idx], axis=0).tolist(), np.max(pts[idx], axis=0).tolist()]
            mean_dist = float(np.mean(np.sqrt(np.maximum(dist[idx, li], 0.0))))
        else:
            bounds = None
            mean_dist = None
        rows.append(
            {
                "link": ln,
                "num_vertices": int(len(idx)),
                "mean_nearest_distance": mean_dist,
                "bounds": bounds,
            }
        )
    return by_link, rows


def _sequence_from_external(
    raw: RawMeshSequence,
    asset: ev.AssetGeometry,
    match_points_by_link: dict[str, np.ndarray],
    target_components: list[TargetComponent],
    args: argparse.Namespace,
) -> tuple[ev.SequenceData, dict[str, Any]]:
    vertices0 = raw.vertices_by_frame[0]
    faces = raw.faces
    if raw.component_vertex_indices:
        components0 = _components_from_vertex_groups(vertices0, raw.component_vertex_indices, raw.component_face_indices)
    else:
        components0 = _component_split(vertices0, faces, int(args.min_component_faces))
    if not components0:
        raise ValueError("No connected components survived --min_component_faces")

    ext_mesh0 = trimesh.Trimesh(vertices=vertices0, faces=faces, process=False)
    try:
        ext_points0 = np.asarray(ext_mesh0.sample(int(args.align_sample_points)), dtype=np.float32)
    except Exception:
        ext_points0 = _sample_rows(vertices0, int(args.align_sample_points), seed=7)
    gt_points = _whole_points(match_points_by_link, asset.visual_links)
    align_tf, align_info = _alignment_matrix(ext_points0, gt_points, str(args.alignment_mode), int(args.align_sample_points))
    effective_motion_scale_mode = str(getattr(args, "motion_scale_mode", "scale_motion"))
    vertices0_aligned = _transform(vertices0, align_tf)

    if target_components:
        component_assignments, component_rows = _assign_components_to_target_components(
            asset.visual_links,
            vertices0_aligned,
            components0,
            target_components,
            int(args.max_component_points),
        )
    else:
        component_assignments, component_rows = _assign_components_to_links(
            asset.visual_links,
            match_points_by_link,
            vertices0_aligned,
            faces,
            components0,
            int(args.max_component_points),
        )
    vertex_rows: list[dict[str, Any]] = []
    if str(args.assignment_mode) == "component":
        comp_by_id = {c.component_id: c for c in components0}
        link_vertex_indices: dict[str, np.ndarray] = {}
        for ln, comp_ids in component_assignments.items():
            if not comp_ids:
                link_vertex_indices[ln] = np.zeros((0,), dtype=np.int64)
                continue
            chunks = [comp_by_id[cid].vertex_indices for cid in comp_ids if cid in comp_by_id]
            link_vertex_indices[ln] = np.unique(np.concatenate(chunks, axis=0)) if chunks else np.zeros((0,), dtype=np.int64)
    elif str(args.assignment_mode) == "vertex":
        link_vertex_indices, vertex_rows = _assign_vertices_to_links(
            asset.visual_links,
            match_points_by_link,
            vertices0_aligned,
            int(args.max_component_points),
        )
    else:
        raise ValueError(f"Unknown assignment mode: {args.assignment_mode}")

    comp_by_id = {c.component_id: c for c in components0}
    target_by_component_id: dict[int, TargetComponent] = {}
    for row in component_rows:
        if row.get("target_component_id") is None:
            continue
        try:
            cid = int(row["component_id"])
            tid = int(row["target_component_id"])
        except Exception:
            continue
        if 0 <= tid < len(target_components):
            target_by_component_id[cid] = target_components[tid]

    component_local_faces: dict[int, np.ndarray] = {
        cid: _local_faces_for_component(len(vertices0), faces, comp)
        for cid, comp in comp_by_id.items()
    }
    component_sample_recipes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    component_fallback_count: dict[int, int] = {}
    link_component_sample_recipes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if str(args.assignment_mode) == "component":
        for ln, comp_ids in component_assignments.items():
            fallback_count = max(1, int(math.ceil(max(1, int(args.num_points_per_link)) / max(1, len(comp_ids)))))
            for cid in comp_ids:
                comp = comp_by_id.get(cid)
                if comp is None:
                    continue
                target = target_by_component_id.get(cid)
                # Keep the prediction density comparable to GT at the link level.
                # target.sample_count is per GT node; using it here undersamples a
                # predicted component that covers a whole link but matched one GT
                # node of a multi-node link (e.g. microwave body).
                count = fallback_count
                seed = (
                    f"{asset.asset_root}:{target.node_name}:{int(args.num_points_per_link)}"
                    if target is not None
                    else f"{raw.path}:{ln}:{cid}:{int(args.num_points_per_link)}"
                )
                local_vertices0 = vertices0_aligned[comp.vertex_indices]
                local_faces = component_local_faces.get(cid, np.zeros((0, 3), dtype=np.int64))
                component_sample_recipes[cid] = _sample_recipe(local_vertices0, local_faces, count, seed)
                component_fallback_count[cid] = count
            face_chunks = [
                np.asarray(comp_by_id[cid].face_indices, dtype=np.int64)
                for cid in comp_ids
                if cid in comp_by_id and len(comp_by_id[cid].face_indices)
            ]
            if face_chunks:
                link_face_indices = np.unique(np.concatenate(face_chunks, axis=0))
                link_faces = np.asarray(faces, dtype=np.int64)[link_face_indices]
                link_component_sample_recipes[ln] = _sample_recipe(
                    vertices0_aligned,
                    link_faces,
                    int(args.num_points_per_link),
                    f"{raw.path}:{ln}:component_union:{int(args.num_points_per_link)}",
                )

    face_indices_by_link: dict[str, np.ndarray] = {}
    local_faces_by_link: dict[str, np.ndarray] = {}
    link_sample_recipes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if str(args.assignment_mode) == "vertex":
        link_sample_recipes = _surface_recipes_by_gt_nearest_pred(
            vertices0_aligned,
            faces,
            asset.visual_links,
            match_points_by_link,
            int(args.num_points_per_link),
            f"{raw.path}:gt_to_pred_surface:{int(args.num_points_per_link)}",
        )

    fps = float(args.prediction_fps)
    if fps <= 0:
        fps = max(1.0, (len(raw.vertices_by_frame) - 1) / max(float(args.prediction_duration_s), 1.0e-9))
    frames: list[ev.TrajectoryFrame] = []
    for fi, verts in enumerate(raw.vertices_by_frame):
        aligned = _transform_external_frame(
            verts,
            align_tf,
            align_info,
            effective_motion_scale_mode,
        )
        vertices_by_link: dict[str, np.ndarray] = {}
        points_by_link: dict[str, np.ndarray] = {}
        for ln in asset.visual_links:
            idx = link_vertex_indices.get(ln, np.zeros((0,), dtype=np.int64))
            pts = aligned[idx] if len(idx) else np.zeros((0, 3), dtype=np.float32)
            vertices_by_link[ln] = pts
            if str(args.assignment_mode) == "component":
                recipe = link_component_sample_recipes.get(ln)
                points_by_link[ln] = _apply_sample_recipe(aligned, recipe) if recipe is not None else _sample_rows(pts, int(args.num_points_per_link), seed=fi * 1009 + len(ln))
            else:
                recipe = link_sample_recipes.get(ln)
                points_by_link[ln] = _apply_sample_recipe(aligned, recipe) if recipe is not None else _sample_rows(pts, int(args.num_points_per_link), seed=fi * 1009 + len(ln))
        frames.append(
            ev.TrajectoryFrame(
                frame_idx=fi,
                time_s=float(raw.times_s[fi]) if raw.times_s is not None and fi < len(raw.times_s) else float(fi) / fps,
                joint_angles={},
                base_tf=np.eye(4, dtype=float),
                mesh_vertices_by_link=vertices_by_link,
                mesh_points_by_link=points_by_link,
            )
        )

    diagnose = {
        "prediction_file": str(raw.path),
        "prediction_source": raw.source,
        "num_prediction_frames": len(frames),
        "num_components": len(components0),
        "assignment_mode": str(args.assignment_mode),
        "alignment": align_info,
        "motion_scale_mode": str(getattr(args, "motion_scale_mode", "scale_motion")),
        "effective_motion_scale_mode": effective_motion_scale_mode,
        "point_sampling_mode": "gt_to_pred_nearest_surface" if str(args.assignment_mode) == "vertex" else "component_surface",
        "component_assignments": component_rows,
        "components_by_link": component_assignments,
        "vertex_assignments_by_link": vertex_rows,
    }
    return ev.SequenceData(path=raw.path, frames=frames, source=raw.source), diagnose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an external single-mesh 4D animation against a benchmark case.")
    parser.add_argument("--prediction_file", type=Path, default=REPO_ROOT / "car.fbx")
    parser.add_argument("--asset_root", type=Path, default=REPO_ROOT / "data" / "causal_data" / "trolley11")
    parser.add_argument(
        "--annotation_path",
        type=Path,
        default=REPO_ROOT / "benchmark_annotations" / "trolley_constraint_templates" / "cases" / "casual_output__trolley11__push.json",
    )
    parser.add_argument(
        "--gt_trajectory",
        type=Path,
        default=REPO_ROOT / "benchmark_annotations" / "compiled_from_benchmark" / "trolley" / "trolley11" / "push" / "animation" / "trajectory.jsonl",
    )
    parser.add_argument(
        "--gt_glb",
        type=Path,
        default=REPO_ROOT / "benchmark_annotations" / "compiled_from_benchmark" / "trolley" / "trolley11" / "push" / "animation" / "plan_animated.glb",
        help="Benchmark animated mesh used both as GT geometry and as the first-frame matching target.",
    )
    parser.add_argument(
        "--gt_plan_json",
        type=Path,
        default=REPO_ROOT / "benchmark_annotations" / "compiled_from_benchmark" / "trolley" / "trolley11" / "push" / "plan.json",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--variant_name", default="external")
    parser.add_argument("--num_points_per_link", type=int, default=2048)
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--scale_floor_ratio", type=float, default=0.05)
    parser.add_argument("--dynamic_weight", type=float, default=0.8)
    parser.add_argument("--static_weight", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--pc_backend", choices=["numpy", "torch", "pytorch3d", "auto"], default="numpy")
    parser.add_argument("--gpu_devices", default="")
    parser.add_argument("--torch_chunk", type=int, default=1024)
    parser.add_argument("--pytorch3d_chunk", type=int, default=4096)
    parser.add_argument("--pc_fallback_numpy", action="store_true", default=True)
    parser.add_argument("--allow_equal_frames", action="store_true")
    parser.add_argument("--disable_terminal_state_check", action="store_true")
    parser.add_argument("--terminal_score_policy", choices=["min", "average"], default="min")
    parser.add_argument("--gt_phase_static_manifest", type=Path, default=None)
    parser.add_argument(
        "--alignment_mode",
        choices=["none", "axis_extent", "similarity", "scale_translate_3d"],
        default="axis_extent",
        help="scale_translate_3d preserves visual axes and only normalizes global scale/translation.",
    )
    parser.add_argument("--align_sample_points", type=int, default=3000)
    parser.add_argument("--max_component_points", type=int, default=256)
    parser.add_argument("--min_component_faces", type=int, default=1)
    parser.add_argument(
        "--assignment_mode",
        choices=["vertex", "component"],
        default="vertex",
        help="vertex is better for remeshed single-mesh outputs; component is useful when parts remain disconnected.",
    )
    parser.add_argument("--prediction_fps", type=float, default=24.0)
    parser.add_argument("--prediction_duration_s", type=float, default=0.625)
    parser.add_argument(
        "--motion_scale_mode",
        choices=["scale_motion", "preserve_center_trajectory"],
        default="scale_motion",
        help=(
            "scale_motion applies alignment scale to full animated coordinates. "
            "preserve_center_trajectory scales each frame's shape but leaves object-center displacement unscaled."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    diagnose_dir = out_dir / "diagnose"
    asset = ev.load_asset_geometry(Path(args.asset_root).resolve(), int(args.num_points_per_link), float(args.scale_floor_ratio))
    row, annotation = ev.parse_annotation_case(Path(args.annotation_path).resolve())
    gt_glb_path = Path(args.gt_glb).resolve() if args.gt_glb else None
    if gt_glb_path is not None and gt_glb_path.exists():
        gt_seq = ev.load_glb_sequence(gt_glb_path, asset, int(args.num_points_per_link))
        match_points_by_link, target_components = _gt_glb_first_frame_targets(
            gt_glb_path,
            asset,
            int(args.num_points_per_link),
        )
    else:
        gt_seq = ev.load_trajectory(Path(args.gt_trajectory).resolve())
        target_components = []
    gt_plan = json.loads(Path(args.gt_plan_json).read_text(encoding="utf-8"))
    if gt_glb_path is None or not gt_glb_path.exists():
        first_gt = gt_seq.frames[0] if gt_seq.frames else ev.TrajectoryFrame(0, 0.0, {}, np.eye(4))
        match_points_by_link = {
            ln: ev.transformed_points(asset, first_gt, ln)
            for ln in asset.visual_links
        }
    meta = {
        "case_id": row.get("case_id") or "external:trolley11:push",
        "class": "trolley",
        "asset_name": row.get("asset_name") or "trolley11",
        "action_name": row.get("action_name") or "push",
    }
    static_rows = None
    if args.gt_phase_static_manifest:
        static_rows = ev._phase_static_manifest_by_case(Path(args.gt_phase_static_manifest)).get(str(meta.get("case_id") or ""))
        if static_rows:
            gt_seq = ev.load_phase_static_sequence(static_rows, asset, int(args.num_points_per_link))
        else:
            raise FileNotFoundError(f"No static phase rows for case_id={meta.get('case_id')} in {args.gt_phase_static_manifest}")
    raw = _load_mesh_sequence(Path(args.prediction_file).resolve())
    pred_seq, match_diagnose = _sequence_from_external(raw, asset, match_points_by_link, target_components, args)
    per_case, per_phase, per_link, matched, _ = ev.evaluate_variant(
        meta,
        row,
        annotation,
        asset,
        gt_seq,
        pred_seq,
        gt_plan,
        str(args.variant_name),
        args,
        static_rows,
    )
    per_case["prediction_file"] = str(Path(args.prediction_file).resolve())
    per_case["prediction_sequence_source"] = pred_seq.source
    per_case["gt_sequence_source"] = gt_seq.source
    per_case["prediction_num_frames"] = len(pred_seq.frames)
    per_case["prediction_last_time_s"] = float(pred_seq.frames[-1].time_s) if pred_seq.frames else None
    per_case_rows = [per_case]

    _write_json(diagnose_dir / "external_matching.json", match_diagnose)
    _write_json(diagnose_dir / "per_case_metrics.json", per_case_rows)
    _write_json(diagnose_dir / "per_phase_metrics.json", per_phase)
    _write_json(diagnose_dir / "per_link_metrics.json", per_link)
    _write_json(diagnose_dir / "matched_frames.json", matched)
    _write_csv(
        out_dir / "external_metrics.csv",
        [
            {
                "case_id": r.get("case_id"),
                "class": r.get("class"),
                "asset_name": r.get("asset_name"),
                "action_name": r.get("action_name"),
                "variant": r.get("variant"),
                "PN_gIoU": r.get("PN_gIoU"),
                "PN_PC": r.get("PN_PC"),
                "PN_OC": r.get("PN_OC"),
            }
        for r in per_case_rows
        ],
        ["case_id", "class", "asset_name", "action_name", "variant", "PN_gIoU", "PN_PC", "PN_OC"],
    )
    print(f"[INFO] wrote {out_dir / 'external_metrics.csv'}")
    print(f"[INFO] diagnose={diagnose_dir}")
    print(json.dumps(per_case_rows, ensure_ascii=False, indent=2))
    # ufbx may crash during Python teardown after FBX results are already
    # written. Flush and exit directly so standalone runs report success.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
