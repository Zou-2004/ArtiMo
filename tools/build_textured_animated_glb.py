#!/usr/bin/env python3
import argparse
import copy
import itertools
import json
import struct
import sys
from pathlib import Path

import numpy as np
import trimesh
from pygltflib import GLTF2
from trimesh import registration as treg
from trimesh.visual.material import PBRMaterial, SimpleMaterial


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import gen_overlays_and_prompts as gop  # noqa: E402
import run_plan as rp  # noqa: E402
import torch_accel as tacc  # noqa: E402


def _iter_urdf_visual_mesh_paths(links, urdf_dir: Path):
    for visuals in links.values():
        for visual in visuals:
            mesh_path = rp._resolve_mesh_path(visual.get("filename"), urdf_dir)
            if mesh_path is not None:
                yield Path(mesh_path)


def _summarize_mesh_path_health(links, urdf_dir: Path) -> dict:
    paths = []
    seen = set()
    for mesh_path in _iter_urdf_visual_mesh_paths(links, urdf_dir):
        key = str(mesh_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(mesh_path)
    missing = []
    empty = []
    nonempty = []
    for p in paths:
        if not p.exists():
            missing.append(str(p))
            continue
        try:
            sz = int(p.stat().st_size)
        except Exception:
            sz = -1
        if sz <= 0:
            empty.append(str(p))
        else:
            nonempty.append((str(p), sz))
    return {
        "total_paths": len(paths),
        "missing": missing,
        "empty": empty,
        "nonempty": nonempty,
    }


def _format_mesh_health_error(prefix: str, health: dict) -> str:
    msg = [
        prefix,
        f"URDF referenced mesh files: total={int(health.get('total_paths', 0))}, "
        f"nonempty={len(health.get('nonempty', []))}, empty={len(health.get('empty', []))}, "
        f"missing={len(health.get('missing', []))}.",
    ]
    empty = list(health.get("empty", []))
    missing = list(health.get("missing", []))
    if empty:
        msg.append("Sample empty mesh files:")
        msg.extend(f"- {p}" for p in empty[:8])
    if missing:
        msg.append("Sample missing mesh files:")
        msg.extend(f"- {p}" for p in missing[:8])
    return "\n".join(msg)


def _glb_length_diagnostic(path: Path) -> str | None:
    try:
        raw = path.read_bytes()[:12]
        if len(raw) < 12 or raw[:4] != b"glTF":
            return None
        declared = struct.unpack("<I", raw[8:12])[0]
        actual = int(path.stat().st_size)
        if declared != actual:
            return (
                f"GLB appears truncated or corrupted: header declares {declared} bytes, "
                f"but file size is {actual} bytes ({path})."
            )
    except Exception:
        return None
    return None


def _mtl_opacity_from_kwargs(mat: SimpleMaterial) -> float:
    kwargs = getattr(mat, "kwargs", {}) or {}
    raw = kwargs.get("d", 1.0)
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    try:
        return float(np.clip(float(raw), 0.0, 1.0))
    except Exception:
        return 1.0


def _normalize_simple_material_for_glb(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    trimesh OBJ->GLB export may collapse non-textured SimpleMaterial colors to one value.
    Promote SimpleMaterial to explicit PBRMaterial per submesh to preserve per-material colors.
    """
    if mesh is None or not isinstance(mesh, trimesh.Trimesh):
        return mesh
    vis = getattr(mesh, "visual", None)
    if not isinstance(vis, trimesh.visual.texture.TextureVisuals):
        return mesh
    mat = getattr(vis, "material", None)
    if not isinstance(mat, SimpleMaterial):
        return mesh

    diffuse = np.asarray(getattr(mat, "diffuse", [255, 255, 255, 255]), dtype=np.uint8).reshape(-1)
    if diffuse.size < 4:
        rgba = np.array([255, 255, 255, 255], dtype=np.uint8)
        rgba[: min(3, diffuse.size)] = diffuse[: min(3, diffuse.size)]
    else:
        rgba = diffuse[:4].copy()
    rgba[3] = int(round(_mtl_opacity_from_kwargs(mat) * 255.0))
    base = [int(rgba[0]), int(rgba[1]), int(rgba[2]), int(rgba[3])]

    alpha_mode = "BLEND" if int(rgba[3]) < 255 else None
    pbr = PBRMaterial(
        baseColorFactor=base,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode=alpha_mode,
        doubleSided=True,
    )
    img = getattr(mat, "image", None)
    if img is not None:
        pbr.baseColorTexture = img
    vis.material = pbr
    return mesh


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    """
    Estimate similarity transform s,R,t such that dst ~= s * R @ src + t.
    src, dst: (N,3)
    """
    return tacc.umeyama_similarity(src, dst)


def _collect_urdf_link_meshes(asset_root: Path, urdf_path: Path):
    links, joints = gop.parse_urdf(urdf_path)
    link_tfs = gop.compute_link_transforms(links, joints)
    out = {}
    ordered_nonempty = []
    for link_name, visuals in links.items():
        submeshes = []
        for visual in visuals:
            mesh_path = gop._resolve_mesh_path(visual["filename"], urdf_path.parent)
            if mesh_path is None or not mesh_path.exists():
                continue
            mesh_obj = gop._load_mesh_textured(mesh_path)
            scale = visual["scale"] or [1.0, 1.0, 1.0]
            scale_mat = np.eye(4)
            scale_mat[0, 0], scale_mat[1, 1], scale_mat[2, 2] = scale
            visual_mat = gop._origin_to_matrix(
                visual.get("origin_xyz"), visual.get("origin_rpy"), visual.get("origin_quat")
            )
            link_tf = link_tfs.get(link_name, np.eye(4))
            if isinstance(mesh_obj, trimesh.Scene):
                for node_name in mesh_obj.graph.nodes_geometry:
                    node_tf, geom_name = mesh_obj.graph[node_name]
                    geom = mesh_obj.geometry[geom_name].copy()
                    geom.apply_transform(scale_mat)
                    geom.apply_transform(node_tf)
                    geom.apply_transform(visual_mat)
                    geom.apply_transform(link_tf)
                    _normalize_simple_material_for_glb(geom)
                    submeshes.append(geom)
            else:
                geom = mesh_obj.copy()
                geom.apply_transform(scale_mat)
                geom.apply_transform(visual_mat)
                geom.apply_transform(link_tf)
                _normalize_simple_material_for_glb(geom)
                submeshes.append(geom)
        if submeshes:
            # trimesh.concatenate preserves texture visuals in many OBJ/MTL cases and yields one mesh per part.
            merged = trimesh.util.concatenate(submeshes)
            out[link_name] = merged
            ordered_nonempty.append(link_name)
    return out, ordered_nonempty, joints


def _merge_fixed_joint_links(link_meshes_world: dict, ordered_nonempty: list[str], joints: list[dict]):
    """
    Merge links connected by fixed joints into their nearest non-fixed ancestor, similar to
    Particulate's preprocessing convention. Meshes are assumed already transformed to URDF rest
    world coordinates, so merging is a simple concatenation at this stage.
    """
    parent_joint_by_child = {}
    for j in joints:
        c = j.get("child")
        if c:
            parent_joint_by_child[c] = j

    def representative(link_name: str) -> str:
        cur = link_name
        seen = set()
        while True:
            if cur in seen:
                return cur
            seen.add(cur)
            j = parent_joint_by_child.get(cur)
            if not j or str(j.get("type") or "").lower() != "fixed":
                return cur
            p = j.get("parent")
            if not p:
                return cur
            cur = p

    merged = {}
    order = []
    rep_members = {}
    for ln in ordered_nonempty:
        rep = representative(ln)
        rep_members.setdefault(rep, []).append(ln)
        if rep not in merged:
            merged[rep] = []
            order.append(rep)
        merged[rep].extend([m.copy() for m in link_meshes_world.get(ln, [])])

    merged_meshes = {}
    for rep in order:
        meshes = [m for m in merged.get(rep, []) if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        if not meshes:
            continue
        merged_meshes[rep] = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    return merged_meshes, [ln for ln in order if ln in merged_meshes], rep_members


def _align4(n: int) -> int:
    return (n + 3) & ~3


def _graft_animations_from_source_glb(src_glb: Path, dst_glb: Path):
    src = GLTF2().load(str(src_glb))
    if not (src.animations or []):
        return {"copied": False, "animations": 0}
    dst = GLTF2().load(str(dst_glb))

    src_bin = src.binary_blob() or b""
    dst_bin = bytearray(dst.binary_blob() or b"")
    if dst.buffers is None or len(dst.buffers) == 0:
        raise RuntimeError("Destination GLB has no buffers; cannot graft animations")

    bv_map = {}
    acc_map = {}

    def clone_buffer_view(src_bv_idx: int) -> int:
        if src_bv_idx in bv_map:
            return bv_map[src_bv_idx]
        sbv = src.bufferViews[src_bv_idx]
        start = int(sbv.byteOffset or 0)
        end = start + int(sbv.byteLength)
        payload = src_bin[start:end]
        pad_to = _align4(len(dst_bin))
        if pad_to > len(dst_bin):
            dst_bin.extend(b"\x00" * (pad_to - len(dst_bin)))
        new_off = len(dst_bin)
        dst_bin.extend(payload)
        dbv = copy.deepcopy(sbv)
        dbv.buffer = 0
        dbv.byteOffset = new_off
        dbv.byteLength = len(payload)
        dst.bufferViews.append(dbv)
        new_idx = len(dst.bufferViews) - 1
        bv_map[src_bv_idx] = new_idx
        return new_idx

    def clone_accessor(src_acc_idx: int) -> int:
        if src_acc_idx in acc_map:
            return acc_map[src_acc_idx]
        sa = src.accessors[src_acc_idx]
        da = copy.deepcopy(sa)
        if da.bufferView is not None:
            da.bufferView = clone_buffer_view(int(da.bufferView))
        dst.accessors.append(da)
        new_idx = len(dst.accessors) - 1
        acc_map[src_acc_idx] = new_idx
        return new_idx

    if dst.animations is None:
        dst.animations = []
    if dst.bufferViews is None:
        dst.bufferViews = []
    if dst.accessors is None:
        dst.accessors = []

    copied = 0
    for anim in src.animations or []:
        da = copy.deepcopy(anim)
        for s in da.samplers or []:
            if s.input is not None:
                s.input = clone_accessor(int(s.input))
            if s.output is not None:
                s.output = clone_accessor(int(s.output))
        # Node indices are assumed preserved because we build the same world+part_node_i layout.
        dst.animations.append(da)
        copied += 1

    dst.buffers[0].byteLength = len(dst_bin)
    dst.set_binary_blob(bytes(dst_bin))
    dst.save(str(dst_glb))
    return {"copied": True, "animations": copied, "buffer_bytes": len(dst_bin)}


def _load_white_part_nodes(mobility_glb: Path):
    sc = trimesh.load(mobility_glb, force="scene", process=False)
    part_nodes = []
    for node_name in sc.graph.nodes_geometry:
        if not str(node_name).startswith("part_node_"):
            continue
        try:
            idx = int(str(node_name).split("_")[-1])
        except Exception:
            continue
        tf, geom_name = sc.graph[node_name]
        geom = sc.geometry[geom_name].copy()
        geom_world = geom.copy()
        geom_world.apply_transform(tf)
        part_nodes.append(
            {
                "idx": idx,
                "node_name": str(node_name),
                "transform": np.array(tf, dtype=float),
                "geom_name": str(geom_name),
                "geom_world": geom_world,
                "geom_local": geom,
            }
        )
    part_nodes = sorted(part_nodes, key=lambda x: x["idx"])
    return sc, part_nodes


def _bounds_center(mesh: trimesh.Trimesh):
    b = mesh.bounds
    return (b[0] + b[1]) * 0.5


def _bounds_extent(mesh: trimesh.Trimesh):
    b = mesh.bounds
    return b[1] - b[0]


def _sample_surface_points(mesh: trimesh.Trimesh, n: int = 3000) -> np.ndarray:
    if mesh is None:
        return np.zeros((0, 3), dtype=float)
    try:
        if len(mesh.faces) > 0:
            count = int(np.clip(n, 512, max(512, len(mesh.faces) * 3)))
            pts, _ = trimesh.sample.sample_surface(mesh, count)
            return np.asarray(pts, dtype=float)
    except Exception:
        pass
    v = np.asarray(mesh.vertices, dtype=float)
    if len(v) == 0:
        return np.zeros((0, 3), dtype=float)
    if len(v) <= n:
        return v
    idx = np.linspace(0, len(v) - 1, n).astype(int)
    return v[idx]


def _local_rigid_part_refine(mesh_local_src: trimesh.Trimesh, mesh_local_ref: trimesh.Trimesh):
    """
    Refine a part in the *source part local frame* with:
    1) conservative uniform scale from bbox volume ratio
    2) rigid ICP (no additional scaling / no reflection)
    This avoids skewing geometry while matching pivots/local orientation better.
    """
    bs = mesh_local_src.bounds
    bd = mesh_local_ref.bounds
    cs = (bs[0] + bs[1]) * 0.5
    cd = (bd[0] + bd[1]) * 0.5
    es = np.maximum(bs[1] - bs[0], 1e-8)
    ed = np.maximum(bd[1] - bd[0], 1e-8)

    vol_s = float(np.prod(es))
    vol_d = float(np.prod(ed))
    if vol_s > 1e-12 and vol_d > 1e-12:
        u = float(np.cbrt(vol_d / vol_s))
    else:
        u = 1.0
    u = float(np.clip(u, 0.25, 4.0))

    T0 = np.eye(4)
    T0[:3, :3] = np.eye(3) * u
    T0[:3, 3] = cd - (u * cs)

    src_pts = _sample_surface_points(mesh_local_src, n=2500)
    if len(src_pts) < 32:
        return T0, u


def _choose_link_to_part_mapping(urdf_meshes_by_link: dict, ordered_links: list[str], part_nodes: list[dict]):
    """
    For mobility assets, URDF non-empty link order is often but not always identical to part_node_i order.
    Choose a permutation by minimizing center alignment error under a global similarity transform plus a weak
    shape-signature penalty (sorted bbox extents).
    """
    n = len(ordered_links)
    if n <= 1:
        return ordered_links, {"mode": "trivial", "perm": list(range(n)), "score": 0.0}

    src_centers = np.stack([_bounds_center(urdf_meshes_by_link[ln]) for ln in ordered_links], axis=0)
    src_sig = np.stack([np.sort(_bounds_extent(urdf_meshes_by_link[ln])) for ln in ordered_links], axis=0)
    dst_centers = np.stack([_bounds_center(p["geom_world"]) for p in part_nodes], axis=0)
    dst_sig = np.stack([np.sort(_bounds_extent(p["geom_world"])) for p in part_nodes], axis=0)

    def perm_score(perm):
        dstc = dst_centers[list(perm)]
        T, _s, _R, _t = _umeyama_similarity(src_centers, dstc)
        src_h = np.concatenate([src_centers, np.ones((n, 1))], axis=1)
        src_fit = (src_h @ T.T)[:, :3]
        center_err = float(np.mean(np.linalg.norm(src_fit - dstc, axis=1)))
        # Shape penalty only on sorted extents (rotation-invariant-ish, weakly scale normalized)
        srcs = src_sig.copy()
        dsts = dst_sig[list(perm)].copy()
        srcs = srcs / np.maximum(np.linalg.norm(srcs, axis=1, keepdims=True), 1e-8)
        dsts = dsts / np.maximum(np.linalg.norm(dsts, axis=1, keepdims=True), 1e-8)
        shape_err = float(np.mean(np.linalg.norm(srcs - dsts, axis=1)))
        return center_err + 0.2 * shape_err

    best_perm = tuple(range(n))
    best_score = float("inf")
    # n is usually small (<=16); exhaustive for <=8 is fine and solves symmetric drawers robustly.
    if n <= 8:
        perms = itertools.permutations(range(n))
    else:
        # Greedy fallback by nearest center after rough normalization if large asset.
        perms = [tuple(range(n))]
    for perm in perms:
        try:
            score = perm_score(perm)
        except Exception:
            continue
        if score < best_score:
            best_score = score
            best_perm = tuple(perm)

    mapped_links = [ordered_links[i] for i in range(n)]  # links remain in source order
    # We reorder *links* so zip(ordered_links_mapped, part_nodes) matches part_node_i order.
    inverse = [None] * n
    for src_i, dst_i in enumerate(best_perm):
        inverse[dst_i] = src_i
    ordered_links_mapped = [ordered_links[src_i] for src_i in inverse]
    return ordered_links_mapped, {"mode": "perm_search" if n <= 8 else "greedy_identity", "perm_src_to_dst": list(best_perm), "score": float(best_score)}
    try:
        T_icp, _transformed, _cost = treg.icp(
            src_pts,
            mesh_local_ref,
            initial=T0,
            max_iterations=40,
            threshold=1e-7,
            scale=False,
            reflection=False,
        )
        # Sanitize to rigid + uniform scale (protect against numeric drift)
        A = np.asarray(T_icp[:3, :3], dtype=float)
        col_norms = np.linalg.norm(A, axis=0)
        s = float(np.mean(col_norms)) if np.all(np.isfinite(col_norms)) else 1.0
        s = float(np.clip(s, 0.25, 4.0))
        R = A / max(s, 1e-12)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        T = np.eye(4)
        T[:3, :3] = s * R
        T[:3, 3] = T_icp[:3, 3]
        return T, s
    except Exception:
        return T0, u


def build_textured_animated_glb(
    asset_root: Path,
    out_glb: Path,
    mobility_glb: Path | None = None,
    part_refine_mode: str = "none",
):
    urdf_path = gop.find_urdf(asset_root)
    if urdf_path is None:
        raise FileNotFoundError(f"No URDF found under {asset_root}")
    if mobility_glb is None:
        mobility_glb = asset_root / "mobility_animated.glb"
    if not mobility_glb.exists():
        raise FileNotFoundError(f"mobility animated glb not found: {mobility_glb}")
    glb_diag = _glb_length_diagnostic(mobility_glb)
    if glb_diag is not None:
        raise RuntimeError(glb_diag)

    urdf_link_meshes_raw, ordered_links_raw, joints = _collect_urdf_link_meshes(asset_root, urdf_path)
    urdf_link_meshes, ordered_links, fixed_merge_members = _merge_fixed_joint_links(urdf_link_meshes_raw, ordered_links_raw, joints)
    if not urdf_link_meshes:
        health = _summarize_mesh_path_health(gop.parse_urdf(urdf_path)[0], urdf_path.parent)
        raise RuntimeError(_format_mesh_health_error("No textured URDF link meshes found.", health))

    _white_scene, part_nodes = _load_white_part_nodes(mobility_glb)
    if len(part_nodes) != len(ordered_links):
        raise RuntimeError(
            f"part count mismatch: mobility_animated has {len(part_nodes)} part_node_i, URDF nonempty links={len(ordered_links)} ({ordered_links})"
        )

    # Choose mapping from URDF links to part_node_i order. This avoids swapped trajectories on symmetric parts
    # (e.g. double drawers) where naive link-order == part-node-order assumptions can be wrong.
    ordered_links_mapped, mapping_solver = _choose_link_to_part_mapping(urdf_link_meshes, ordered_links, part_nodes)
    urdf_meshes_ordered = [urdf_link_meshes[ln] for ln in ordered_links_mapped]
    src_centers = np.stack([_bounds_center(m) for m in urdf_meshes_ordered], axis=0)
    dst_centers = np.stack([_bounds_center(p["geom_world"]) for p in part_nodes], axis=0)
    T_sim, scale, _, _ = _umeyama_similarity(src_centers, dst_centers)

    # Build new textured scene with same part_node_i transforms as mobility_animated.
    out_scene = trimesh.Scene()
    mapping = []
    for link_name, part in zip(ordered_links_mapped, part_nodes):
        # Global similarity gets us close in world coordinates. For trajectory fidelity we default to *no*
        # per-part local refinement (Particulate-style: keep node local frame untouched, only swap geometry).
        # Optional local rigid ICP exists for assets that need tighter static alignment.
        mesh_world = urdf_link_meshes[link_name].copy()
        mesh_world.apply_transform(T_sim)
        if part_refine_mode == "local_rigid_icp":
            try:
                mesh_local_tmp = mesh_world.copy()
                mesh_local_tmp.apply_transform(np.linalg.inv(part["transform"]))
                ref_local = part["geom_local"]
                A_local, part_scale = _local_rigid_part_refine(mesh_local_tmp, ref_local)
                mesh_local = mesh_local_tmp.copy()
                mesh_local.apply_transform(A_local)
                mesh_world = mesh_local.copy()
                mesh_world.apply_transform(part["transform"])
                T_part = A_local
                refine_mode_used = "local_rigid_icp"
            except Exception:
                T_part = np.eye(4)
                part_scale = 1.0
                mesh_local = mesh_world.copy()
                mesh_local.apply_transform(np.linalg.inv(part["transform"]))
                refine_mode_used = "none_fallback"
        else:
            T_part = np.eye(4)
            part_scale = 1.0
            mesh_local = mesh_world.copy()
            mesh_local.apply_transform(np.linalg.inv(part["transform"]))
            refine_mode_used = "none"
        geom_name = f"geometry_{part['idx']}"
        out_scene.add_geometry(
            mesh_local,
            geom_name=geom_name,
            node_name=part["node_name"],
            transform=part["transform"],
        )
        mapping.append(
            {
                "part_node": part["node_name"],
                "link_name": link_name,
                "white_bbox_extent": _bounds_extent(part["geom_world"]).tolist(),
                "textured_bbox_extent_after_align": _bounds_extent(mesh_world).tolist(),
                "part_refine_transform": T_part.tolist(),
                "part_refine_scale": float(part_scale),
                "part_refine_mode": refine_mode_used,
            }
        )

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    out_scene.export(out_glb)
    anim_copy_info = _graft_animations_from_source_glb(mobility_glb, out_glb)

    # Emit a small report next to the GLB for debugging.
    report = {
        "asset_root": str(asset_root),
        "urdf": str(urdf_path),
        "mobility_animated_glb": str(mobility_glb),
        "output_glb": str(out_glb),
        "num_parts": len(part_nodes),
        "ordered_links_original": ordered_links_raw,
        "ordered_links_after_fixed_merge": ordered_links,
        "ordered_links_mapped_to_part_nodes": ordered_links_mapped,
        "fixed_merge_members": fixed_merge_members,
        "mapping_solver": mapping_solver,
        "similarity_transform": T_sim.tolist(),
        "similarity_scale": float(scale),
        "part_refine_mode": part_refine_mode,
        "mapping": mapping,
        "joints_count": len(joints),
        "animation_graft": anim_copy_info,
    }
    report_path = out_glb.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_glb, report_path


def _concat_meshes_preserve_visual(meshes):
    meshes = [m.copy() for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    return trimesh.util.concatenate(meshes)


def _choose_joint_preview_target(joint):
    jtype = (joint.get("type") or "").lower()
    lim = joint.get("limit") or {}
    lo = lim.get("lower")
    hi = lim.get("upper")
    if jtype == "fixed":
        return 0.0
    if jtype == "continuous":
        return float(np.pi * 0.75)
    candidates = [0.0]
    if lo is not None:
        candidates.append(float(lo))
    if hi is not None:
        candidates.append(float(hi))
    # Prefer the farther limit from zero.
    best = max(candidates, key=lambda x: abs(x))
    if abs(best) < 1e-6 and hi is not None:
        best = float(hi)
    return float(best)


def _initial_joint_positions_for_preview(joints, mode: str = "zeros"):
    joint_pos0 = {}
    for j in joints:
        jn = j.get("name")
        if not jn:
            continue
        jtype = str(j.get("type") or "").lower()
        lim = j.get("limit") or {}
        q0 = 0.0
        if mode == "prismatic_lower" and jtype == "prismatic":
            lo = lim.get("lower")
            if lo is not None:
                q0 = float(lo)
        elif mode == "lower" and jtype in ("prismatic", "revolute"):
            lo = lim.get("lower")
            if lo is not None:
                q0 = float(lo)
        joint_pos0[jn] = float(q0)
    return joint_pos0


def _build_urdf_preview_frames(
    links,
    joints,
    fps=30,
    frames_per_joint=36,
    hold_frames=6,
    initial_pose_mode: str = "zeros",
    max_preview_joints: int | None = None,
):
    joint_pos0 = _initial_joint_positions_for_preview(joints, mode=initial_pose_mode)
    movable = [j for j in joints if (j.get("type") or "").lower() not in ("", "fixed")]
    if max_preview_joints is not None and int(max_preview_joints) > 0:
        movable = movable[: int(max_preview_joints)]
    frames = []

    def add_frame(joint_pos):
        frames.append((dict(joint_pos), np.eye(4)))

    if int(frames_per_joint) <= 0:
        add_frame(joint_pos0)
        return frames

    # Start hold
    for _ in range(max(1, hold_frames)):
        add_frame(joint_pos0)

    if not movable:
        for _ in range(max(1, fps)):
            add_frame(joint_pos0)
        return frames

    # Sequentially animate each joint open->close to create a canonical preview animation.
    for joint in movable:
        jn = joint.get("name")
        if not jn:
            continue
        q_target = _choose_joint_preview_target(joint)
        q_start = joint_pos0[jn]
        n_open = max(4, frames_per_joint // 2)
        n_close = max(4, frames_per_joint - n_open)

        for i in range(n_open):
            a = (i + 1) / float(n_open)
            jp = dict(joint_pos0)
            jp[jn] = (1.0 - a) * q_start + a * q_target
            add_frame(jp)
        for _ in range(max(1, hold_frames // 2)):
            jp = dict(joint_pos0)
            jp[jn] = q_target
            add_frame(jp)
        for i in range(n_close):
            a = (i + 1) / float(n_close)
            jp = dict(joint_pos0)
            jp[jn] = (1.0 - a) * q_target + a * q_start
            add_frame(jp)

    # End hold
    for _ in range(max(1, hold_frames)):
        add_frame(joint_pos0)
    return frames


def build_textured_animated_glb_from_urdf(
    asset_root: Path,
    out_glb: Path,
    fps: int = 30,
    frames_per_joint: int = 36,
    initial_pose_mode: str = "zeros",
    max_preview_joints: int | None = None,
):
    urdf_path = gop.find_urdf(asset_root)
    if urdf_path is None:
        raise FileNotFoundError(f"No URDF found under {asset_root}")

    links, joints = rp.parse_urdf(urdf_path)
    link_meshes = rp.load_link_meshes(links, urdf_path.parent, textured=True)
    # Keep URDF link granularity in preview mode so downstream link->node mapping
    # stays consistent with run_plan's URDF-order assumptions.
    link_tf0_full = rp.compute_link_transforms(links, joints, _initial_joint_positions_for_preview(joints, mode=initial_pose_mode), base_tf=np.eye(4))
    link_meshes_world = {}
    ordered_nonempty = []
    for ln in links.keys():
        meshes = link_meshes.get(ln) or []
        if not meshes:
            continue
        tf = np.array(link_tf0_full.get(ln, np.eye(4)), dtype=float)
        world_meshes = []
        for m in meshes:
            mm = m.copy()
            mm.apply_transform(tf)
            _normalize_simple_material_for_glb(mm)
            world_meshes.append(mm)
        if world_meshes:
            link_meshes_world[ln] = world_meshes
            ordered_nonempty.append(ln)
    ordered_links = list(ordered_nonempty)
    fixed_merge_members = {ln: [ln] for ln in ordered_links}
    if not ordered_links:
        health = _summarize_mesh_path_health(links, urdf_path.parent)
        raise RuntimeError(_format_mesh_health_error("No textured link meshes found from URDF.", health))

    # Rest transforms at q=0 from URDF FK (this is the calibration the user requested).
    joint_pos0 = _initial_joint_positions_for_preview(joints, mode=initial_pose_mode)
    link_tf0 = rp.compute_link_transforms(links, joints, joint_pos0, base_tf=np.eye(4))

    scene = trimesh.Scene()
    link_to_nodes = {}
    report_parts = []
    for idx, ln in enumerate(ordered_links):
        # Keep per-link submeshes separate so multi-material OBJ links don't collapse to
        # a single texture/material during canonical GLB export.
        meshes_world = [m for m in (link_meshes_world.get(ln) or []) if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0]
        if not meshes_world:
            continue
        rest_tf = np.array(link_tf0.get(ln, np.eye(4)), dtype=float)
        rest_inv = np.linalg.inv(rest_tf)
        node_names = []
        local_for_report = []
        for mi, mesh_world in enumerate(meshes_world):
            mesh_local = mesh_world.copy()
            mesh_local.apply_transform(rest_inv)
            _normalize_simple_material_for_glb(mesh_local)
            node_name = f"part_node_{idx}" if mi == 0 else f"part_node_{idx}__sub_{mi:02d}"
            geom_name = f"geometry_{idx}_{mi:02d}"
            scene.add_geometry(mesh_local, node_name=node_name, geom_name=geom_name, transform=rest_tf)
            node_names.append(node_name)
            local_for_report.append(mesh_local.copy())
        link_to_nodes[ln] = node_names
        merged_local = local_for_report[0] if len(local_for_report) == 1 else trimesh.util.concatenate(local_for_report)
        report_parts.append(
            {
                "part_node": node_names[0],
                "part_node_submeshes": node_names,
                "link_name": ln,
                "rest_transform": rest_tf.tolist(),
                "bbox_extent_local": _bounds_extent(merged_local).tolist(),
                "submesh_count": len(node_names),
            }
        )

    frames = _build_urdf_preview_frames(
        links,
        joints,
        fps=fps,
        frames_per_joint=frames_per_joint,
        initial_pose_mode=initial_pose_mode,
        max_preview_joints=max_preview_joints,
    )
    if not frames:
        frames = [(joint_pos0, np.eye(4))]

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    rp.export_animated_glb(
        str(out_glb),
        link_meshes={},  # glb_scene path used
        frames=frames,
        links=links,
        joints=joints,
        fps=fps,
        glb_scene=scene,
        link_to_nodes_override=link_to_nodes,
    )

    report = {
        "asset_root": str(asset_root),
        "urdf": str(urdf_path),
        "output_glb": str(out_glb),
        "build_mode": "urdf_preview",
        "fps": int(fps),
        "frames_per_joint": int(frames_per_joint),
        "initial_pose_mode": str(initial_pose_mode),
        "max_preview_joints": None if max_preview_joints is None else int(max_preview_joints),
        "preview_joint_count": int(min(len([j for j in joints if (j.get("type") or "").lower() not in ("", "fixed")]), int(max_preview_joints)))
        if max_preview_joints is not None and int(max_preview_joints) > 0
        else int(len([j for j in joints if (j.get("type") or "").lower() not in ("", "fixed")])),
        "num_frames": len(frames),
        "ordered_links_mapped_to_part_nodes": ordered_links,
        "fixed_merge_members": fixed_merge_members,
        "joints_count": len(joints),
        "movable_joints": [j.get("name") for j in joints if (j.get("type") or "").lower() not in ("", "fixed")],
        "parts": report_parts,
    }
    report_path = out_glb.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_glb, report_path


def main():
    ap = argparse.ArgumentParser(description="Build animated_textured_<asset>.glb from URDF + textured part meshes")
    ap.add_argument("--asset_root", required=True)
    ap.add_argument("--out_glb", default=None, help="Default: <asset_root>/animated_textured_<asset_name>.glb")
    ap.add_argument("--mobility_glb", default=None, help="Legacy mobility_align mode only")
    ap.add_argument("--make_symlink", action="store_true", help="Also create textured_<asset_name>.glb symlink to out_glb")
    ap.add_argument("--build_mode", choices=["mobility_align", "urdf_preview"], default="urdf_preview")
    ap.add_argument("--part_refine_mode", choices=["none", "local_rigid_icp"], default="none")
    ap.add_argument("--fps", type=int, default=30, help="URDF preview mode only")
    ap.add_argument("--frames_per_joint", type=int, default=36, help="URDF preview mode only")
    ap.add_argument("--initial_pose_mode", choices=["zeros", "prismatic_lower", "lower"], default="zeros", help="URDF preview mode only")
    ap.add_argument("--max_preview_joints", type=int, default=None, help="URDF preview mode only")
    args = ap.parse_args()

    asset_root = Path(args.asset_root).absolute()
    asset_name = asset_root.name
    out_glb = Path(args.out_glb).absolute() if args.out_glb else (asset_root / f"animated_textured_{asset_name}.glb").absolute()
    mobility_glb = Path(args.mobility_glb).absolute() if args.mobility_glb else None

    if args.build_mode == "urdf_preview":
        out_glb, report = build_textured_animated_glb_from_urdf(
            asset_root,
            out_glb,
            fps=args.fps,
            frames_per_joint=args.frames_per_joint,
            initial_pose_mode=args.initial_pose_mode,
            max_preview_joints=args.max_preview_joints,
        )
    else:
        out_glb, report = build_textured_animated_glb(
            asset_root, out_glb, mobility_glb=mobility_glb, part_refine_mode=args.part_refine_mode
        )
    print(f"Wrote {out_glb}")
    print(f"Wrote {report}")
    if args.make_symlink:
        link_path = asset_root / f"textured_{asset_name}.glb"
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(out_glb.name)
            print(f"Linked {link_path} -> {out_glb.name}")
        except Exception as exc:
            print(f"[WARN] Failed to create symlink {link_path}: {exc}")


if __name__ == "__main__":
    main()
