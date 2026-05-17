#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np


SUPPORTED_CONTROL_MODES = {
    "base_velocity",
    "base_velocity_decay",
    "joint_velocity",
    "joint_position",
    "hold_position",
    "spring_return",
    "mode_set",
}


def _curve_alpha(curve: str | None, t: float) -> float:
    u = float(max(0.0, min(1.0, t)))
    curve_name = str(curve or "linear").strip().lower()
    if curve_name == "ease_in_out":
        return u * u * (3.0 - 2.0 * u)
    if curve_name == "ease_out":
        return 1.0 - (1.0 - u) * (1.0 - u)
    if curve_name == "ease_in":
        return u * u
    return u


def _ensure_float3(values: list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"Expected 3 values, got {arr.tolist()}")
    return arr


def _normalize_axis(axis_world: list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    axis = _ensure_float3(axis_world)
    n = float(np.linalg.norm(axis))
    if n <= 1.0e-8:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return (axis / n).astype(np.float32)


def _parse_floats(text: str | None, default=None):
    if text is None:
        return default
    parts = [p for p in str(text).replace(",", " ").split() if p]
    if not parts:
        return default
    out: list[float] = []
    for token in parts:
        token_norm = str(token).strip().lower()
        if token_norm in {"none", "null", "nan"}:
            out.append(0.0)
        else:
            out.append(float(token))
    return out


def _write_rgba_png(image_rgba: np.ndarray, path: Path) -> None:
    from PIL import Image

    img = Image.fromarray(np.asarray(image_rgba, dtype=np.uint8), mode="RGBA")
    img.save(path)


def _capture_runtime_record(articulation, frame_idx: int, time_s: float, joint_names: list[str]) -> dict[str, Any]:
    root_positions, root_orientations = articulation.get_world_poses()
    joint_positions = np.asarray(articulation.get_joint_positions(), dtype=np.float32).reshape(-1)
    joint_pos_map = {
        str(name): float(val)
        for name, val in zip(list(joint_names), joint_positions.tolist())
    }
    return {
        "frame_idx": int(frame_idx),
        "time_s": float(time_s),
        "joint_pos": joint_pos_map,
        "base_translation": np.asarray(root_positions[0], dtype=np.float32),
        "base_rotation_xyzw": np.asarray(root_orientations[0], dtype=np.float32),
    }


def _write_runtime_trajectory(
    out_dir: Path,
    records: list[dict[str, Any]],
    joint_names: list[str],
    npz_name: str,
    jsonl_name: str,
) -> tuple[Path, Path]:
    traj_npz_path = (out_dir / str(npz_name or "trajectory.npz")).resolve()
    traj_jsonl_path = (out_dir / str(jsonl_name or "trajectory.jsonl")).resolve()
    joint_names_sorted = list(joint_names)
    joint_angles = np.zeros((len(records), len(joint_names_sorted)), dtype=np.float32)
    base_translation = np.zeros((len(records), 3), dtype=np.float32)
    base_rotation_xyzw = np.zeros((len(records), 4), dtype=np.float32)
    times_s = np.zeros((len(records),), dtype=np.float32)
    for idx, rec in enumerate(records):
        times_s[idx] = float(rec["time_s"])
        base_translation[idx] = np.asarray(rec["base_translation"], dtype=np.float32)
        base_rotation_xyzw[idx] = np.asarray(rec["base_rotation_xyzw"], dtype=np.float32)
        for j_idx, joint_name in enumerate(joint_names_sorted):
            joint_angles[idx, j_idx] = float((rec.get("joint_pos") or {}).get(joint_name, 0.0))
    np.savez(
        traj_npz_path,
        joint_names=np.array(joint_names_sorted, dtype=object),
        joint_angles=joint_angles,
        base_translation=base_translation,
        base_rotation_xyzw=base_rotation_xyzw,
        time_s=times_s,
    )
    with traj_jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            row = {
                "frame_idx": int(rec["frame_idx"]),
                "time_s": float(rec["time_s"]),
                "joint_angles": {k: float(v) for k, v in (rec.get("joint_pos") or {}).items()},
                "base_pose": {
                    "translation": [float(x) for x in np.asarray(rec["base_translation"]).tolist()],
                    "rotation_xyzw": [float(x) for x in np.asarray(rec["base_rotation_xyzw"]).tolist()],
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return traj_npz_path, traj_jsonl_path


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    x, y, z, w = q.tolist()
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rot = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    return tf


def _rpy_to_matrix(rpy: list[float] | tuple[float, float, float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64).reshape(3).tolist()
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    return tf


def _origin_to_matrix(origin_xyz, origin_rpy) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    if origin_rpy is not None:
        tf = _rpy_to_matrix(origin_rpy)
    if origin_xyz is not None:
        tf[:3, 3] = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
    return tf


def _scale_to_matrix(scale_xyz) -> np.ndarray:
    scale = np.asarray(scale_xyz if scale_xyz is not None else [1.0, 1.0, 1.0], dtype=np.float64).reshape(3)
    tf = np.eye(4, dtype=np.float64)
    tf[0, 0] = float(scale[0])
    tf[1, 1] = float(scale[1])
    tf[2, 2] = float(scale[2])
    return tf


def _axis_angle_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12 or abs(float(angle_rad)) <= 1.0e-12:
        return np.eye(4, dtype=np.float64)
    x, y, z = (axis / norm).tolist()
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    C = 1.0 - c
    rot = np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ],
        dtype=np.float64,
    )
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    return tf


def _resolve_mesh_path(mesh_filename: str | None, urdf_dir: Path) -> Path | None:
    if mesh_filename is None:
        return None
    mesh_filename = str(mesh_filename).strip()
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


def _parse_urdf(urdf_path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    root = ET.parse(urdf_path).getroot()
    links: dict[str, list[dict[str, Any]]] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name:
            visuals: list[dict[str, Any]] = []
            for visual in link.findall("visual"):
                geom = visual.find("geometry")
                if geom is None:
                    continue
                mesh_tag = geom.find("mesh")
                if mesh_tag is None:
                    continue
                filename = mesh_tag.get("filename") or mesh_tag.get("file")
                origin = visual.find("origin")
                visuals.append(
                    {
                        "name": visual.get("name"),
                        "filename": filename,
                        "scale": _parse_floats(mesh_tag.get("scale"), default=[1.0, 1.0, 1.0]),
                        "origin_xyz": _parse_floats(origin.get("xyz")) if origin is not None else None,
                        "origin_rpy": _parse_floats(origin.get("rpy")) if origin is not None else None,
                    }
                )
            links[str(name)] = visuals

    joints: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        origin = joint.find("origin")
        joints.append(
            {
                "name": joint.get("name"),
                "type": joint.get("type"),
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


def _compute_link_transforms(
    links: dict[str, list[dict[str, Any]]],
    joints: list[dict[str, Any]],
    joint_positions: dict[str, float],
    base_tf: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    joint_tree: dict[str, list[dict[str, Any]]] = {}
    for joint in joints:
        parent = joint.get("parent")
        child = joint.get("child")
        if not parent or not child:
            continue
        joint_tree.setdefault(str(parent), []).append(joint)

    child_links = {str(j.get("child")) for j in joints if j.get("child")}
    root_links = [str(link_name) for link_name in links.keys() if str(link_name) not in child_links]
    if not root_links:
        root_links = [str(link_name) for link_name in links.keys()]

    link_tf: dict[str, np.ndarray] = {name: np.eye(4, dtype=np.float64) for name in root_links}

    def walk(parent_link: str) -> None:
        parent_tf = link_tf[parent_link]
        for joint in joint_tree.get(parent_link, []):
            origin = joint.get("origin") or {}
            origin_tf = _origin_to_matrix(origin.get("xyz"), origin.get("rpy"))
            motion_tf = np.eye(4, dtype=np.float64)
            q = float(joint_positions.get(str(joint.get("name")), 0.0))
            axis = np.asarray(joint.get("axis") or [0.0, 0.0, 1.0], dtype=np.float64)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1.0e-12:
                axis = axis / axis_norm
            if joint.get("type") in {"revolute", "continuous"}:
                motion_tf = _axis_angle_matrix(axis, q)
            elif joint.get("type") == "prismatic":
                motion_tf[:3, 3] = axis * q
            child_link = joint.get("child")
            if not child_link:
                continue
            link_tf[str(child_link)] = parent_tf @ origin_tf @ motion_tf
            walk(str(child_link))

    for root_link in root_links:
        link_tf[root_link] = np.eye(4, dtype=np.float64)
        walk(root_link)

    if base_tf is not None:
        base_tf64 = np.asarray(base_tf, dtype=np.float64)
        for link_name in list(link_tf.keys()):
            link_tf[link_name] = base_tf64 @ link_tf[link_name]
    return link_tf


def _look_at_transform(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray([0.0, 0.0, 1.0] if up_hint is None else up_hint, dtype=np.float64).reshape(3)
    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1.0e-8:
        forward = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        forward = forward / forward_norm
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1.0e-8:
        up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up)
        right_norm = float(np.linalg.norm(right))
    right = right / max(right_norm, 1.0e-8)
    true_up = np.cross(right, forward)
    tf = np.eye(4, dtype=np.float64)
    tf[:3, 0] = right
    tf[:3, 1] = true_up
    tf[:3, 2] = -forward
    tf[:3, 3] = eye
    return tf


def _set_transform_prim(prim, world_tf: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    if prim is None or not prim.IsValid():
        return
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    op = None
    for cand in ops:
        if cand.GetOpType() == UsdGeom.XformOp.TypeTransform:
            op = cand
            break
    if op is None:
        op = xf.AddTransformOp()
    op.Set(Gf.Matrix4d(np.asarray(world_tf, dtype=np.float64).tolist()))


def _get_transform_prim_world(prim) -> np.ndarray:
    from pxr import UsdGeom

    if prim is None or not prim.IsValid():
        return np.eye(4, dtype=np.float64)
    try:
        cache = UsdGeom.XformCache()
        return np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
    except Exception:
        return np.eye(4, dtype=np.float64)


def _initial_root_translation(spec: dict[str, Any]) -> np.ndarray:
    placement = spec.get("placement") or {}
    values = placement.get("initial_root_translation")
    if values is None:
        return np.zeros(3, dtype=np.float64)
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(3)
    except Exception:
        return np.zeros(3, dtype=np.float64)
    return arr


def _author_root_translation(stage, base_path: str, translation: np.ndarray) -> dict[str, Any]:
    delta = np.asarray(translation, dtype=np.float64).reshape(3)
    out: dict[str, Any] = {
        "requested_translation": [float(x) for x in delta.tolist()],
        "applied": False,
        "base_path": str(base_path),
    }
    if float(np.linalg.norm(delta)) <= 1.0e-9:
        return out
    base_prim = stage.GetPrimAtPath(str(base_path))
    if base_prim is None or not base_prim.IsValid():
        out["error"] = "invalid_base_prim"
        return out
    world_tf = _get_transform_prim_world(base_prim)
    world_tf[:3, 3] += delta
    _set_transform_prim(base_prim, world_tf)
    out["applied"] = True
    out["authored_world_translation"] = [float(x) for x in world_tf[:3, 3].tolist()]
    return out


def _apply_root_pose_translation(articulation, translation: np.ndarray) -> dict[str, Any]:
    target = np.asarray(translation, dtype=np.float64).reshape(3)
    out: dict[str, Any] = {
        "requested_translation": [float(x) for x in target.tolist()],
        "applied": False,
    }
    if float(np.linalg.norm(target)) <= 1.0e-9:
        return out
    root_positions, root_orientations = articulation.get_world_poses()
    pos = np.asarray(root_positions, dtype=np.float32).copy()
    orn = np.asarray(root_orientations, dtype=np.float32).copy()
    pos[0] = target.astype(np.float32)
    set_world_poses = getattr(articulation, "set_world_poses", None)
    if callable(set_world_poses):
        try:
            set_world_poses(positions=pos, orientations=orn)
            out["applied"] = True
            out["method"] = "set_world_poses"
        except TypeError:
            set_world_poses(pos, orn)
            out["applied"] = True
            out["method"] = "set_world_poses"
        except Exception as exc:
            out["error"] = repr(exc)
    if not out["applied"]:
        set_world_pose = getattr(articulation, "set_world_pose", None)
        if callable(set_world_pose):
            try:
                set_world_pose(position=pos[0], orientation=orn[0])
                out["applied"] = True
                out["method"] = "set_world_pose"
            except TypeError:
                set_world_pose(pos[0], orn[0])
                out["applied"] = True
                out["method"] = "set_world_pose"
            except Exception as exc:
                out["error"] = repr(exc)
    out["result_root_translation"] = [float(x) for x in pos[0].tolist()]
    return out


def _set_camera_fov(camera_prim, fov_deg: float | None, aspect_ratio: float) -> None:
    if camera_prim is None or not camera_prim.IsValid() or fov_deg is None:
        return
    fov = float(fov_deg)
    if not (1.0 < fov < 179.0):
        return
    horiz_ap_attr = camera_prim.GetAttribute("horizontalAperture")
    vert_ap_attr = camera_prim.GetAttribute("verticalAperture")
    horiz_ap = horiz_ap_attr.Get() if horiz_ap_attr.IsValid() else None
    vert_ap = vert_ap_attr.Get() if vert_ap_attr.IsValid() else None
    if vert_ap is None:
        if horiz_ap is not None and aspect_ratio > 1.0e-6:
            vert_ap = float(horiz_ap) / float(aspect_ratio)
        else:
            vert_ap = 15.2908
    if horiz_ap_attr.IsValid() and aspect_ratio > 1.0e-6:
        horiz_ap_attr.Set(float(vert_ap) * float(aspect_ratio))
    if vert_ap_attr.IsValid():
        vert_ap_attr.Set(float(vert_ap))
    focal_length = 0.5 * float(vert_ap) / math.tan(math.radians(fov) * 0.5)
    focal_attr = camera_prim.GetAttribute("focalLength")
    if focal_attr.IsValid():
        focal_attr.Set(float(focal_length))


def _control_joint_names(ctrl: dict[str, Any]) -> list[str]:
    joints = [str(j) for j in list(ctrl.get("joints") or []) if str(j)]
    joint_name = str(ctrl.get("joint") or "").strip()
    if joint_name:
        joints = [joint_name]
    return list(dict.fromkeys(joints))


def _planned_initial_joint_positions(timeline: list[dict[str, Any]]) -> dict[str, float]:
    initial: dict[str, float] = {}
    for seg in timeline:
        for ctrl in list(seg.get("controls") or []):
            for joint_name in _control_joint_names(ctrl):
                initial.setdefault(joint_name, 0.0)
    for seg in timeline:
        if float(seg.get("t0", 0.0)) > 1.0e-9:
            continue
        for ctrl in list(seg.get("controls") or []):
            mode = str(ctrl.get("mode") or "").strip()
            if mode != "joint_position":
                continue
            joint_name = str(ctrl.get("joint") or "").strip()
            if not joint_name:
                continue
            q_start = ctrl.get("q_start_rad")
            if q_start is not None:
                initial[joint_name] = float(q_start)
    return initial


class FollowCameraRig:
    def __init__(self, articulation, camera_root_prim, camera_spec: dict[str, Any]) -> None:
        self.articulation = articulation
        self.camera_root_prim = camera_root_prim
        root_positions, _root_orientations = articulation.get_world_poses()
        self.base0 = np.asarray(root_positions[0], dtype=np.float64)
        # Preserve the camera rig that Replicator authored for the initial
        # eye/target pair and only translate it with the moving base. Rewriting
        # a fresh look-at transform every frame can fight the internal camera
        # hierarchy and produces the "sudden zoom / white screen" failure seen
        # on trolley assets.
        self.camera_world_tf0 = _get_transform_prim_world(camera_root_prim)

    def update(self) -> None:
        root_positions, _root_orientations = self.articulation.get_world_poses()
        base = np.asarray(root_positions[0], dtype=np.float64)
        delta = base - self.base0
        world_tf = np.array(self.camera_world_tf0, dtype=np.float64, copy=True)
        world_tf[:3, 3] += delta
        _set_transform_prim(self.camera_root_prim, world_tf)


class TimelinePhysicsExecutor:
    def __init__(self, articulation, spec: dict[str, Any], fps: int, visual_sync: Any | None = None) -> None:
        self.art = articulation
        self.spec = spec
        self.timeline = list(spec.get("timeline") or [])
        self.duration_s = float(spec.get("meta", {}).get("duration_s", 0.0))
        self.dt_nominal = 1.0 / float(max(1, fps))
        self.sim_time = 0.0
        self.segment_entry_positions: dict[int, dict[str, float]] = {}
        self.segment_hold_positions: dict[int, dict[str, float]] = {}
        self.mode_flags: dict[str, bool] = {}
        self.finished = False
        self.has_base_motion = bool(spec.get("has_base_motion", False))
        self.joint_name_set = set(self.art.dof_names or [])
        self.all_joint_names = list(self.art.dof_names or [])
        self.dof_index_by_name = {name: idx for idx, name in enumerate(self.all_joint_names)}
        self.visual_sync = visual_sync
        self.plan_initial_positions = {
            str(name): float(val)
            for name, val in _planned_initial_joint_positions(self.timeline).items()
            if str(name) in self.joint_name_set
        }
        if not self.all_joint_names:
            raise RuntimeError("Imported articulation exposes no DOFs; executor cannot run.")

    def _active_segments(self, t: float) -> list[tuple[int, dict[str, Any]]]:
        out: list[tuple[int, dict[str, Any]]] = []
        for idx, seg in enumerate(self.timeline):
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", t0))
            if t0 <= t < t1:
                out.append((idx, seg))
        return out

    def _joint_positions(self, joint_names: list[str]) -> np.ndarray:
        if not joint_names:
            return np.zeros((0,), dtype=np.float32)
        joint_indices = np.asarray([self.dof_index_by_name[name] for name in joint_names], dtype=np.int64)
        data = self.art.get_joint_positions(joint_indices=joint_indices)
        return np.asarray(data, dtype=np.float32).reshape(-1)

    def _joint_velocities(self, joint_names: list[str]) -> np.ndarray:
        if not joint_names:
            return np.zeros((0,), dtype=np.float32)
        joint_indices = np.asarray([self.dof_index_by_name[name] for name in joint_names], dtype=np.int64)
        data = self.art.get_joint_velocities(joint_indices=joint_indices)
        return np.asarray(data, dtype=np.float32).reshape(-1)

    def _remember_segment_entry_positions(self, seg_idx: int, joint_names: list[str]) -> None:
        if seg_idx in self.segment_entry_positions:
            return
        positions = self._joint_positions(joint_names)
        self.segment_entry_positions[seg_idx] = {
            str(name): float(pos) for name, pos in zip(joint_names, positions.tolist())
        }

    def _remember_hold_positions(self, seg_idx: int, joint_names: list[str]) -> None:
        if seg_idx in self.segment_hold_positions:
            return
        positions = self._joint_positions(joint_names)
        self.segment_hold_positions[seg_idx] = {
            str(name): float(pos) for name, pos in zip(joint_names, positions.tolist())
        }

    def _set_joint_gains(self, joints: list[str], stiffness: float, damping: float) -> None:
        if not joints:
            return
        joint_indices = np.asarray([self.dof_index_by_name[name] for name in joints], dtype=np.int64)
        kps = np.full((1, len(joints)), float(stiffness), dtype=np.float32)
        kds = np.full((1, len(joints)), float(damping), dtype=np.float32)
        self.art.set_gains(kps=kps, kds=kds, joint_indices=joint_indices)

    def _set_joint_position_targets(self, position_targets: dict[str, float]) -> None:
        if not position_targets:
            return
        joints = list(position_targets.keys())
        joint_indices = np.asarray([self.dof_index_by_name[name] for name in joints], dtype=np.int64)
        targets = np.asarray([[float(position_targets[jn]) for jn in joints]], dtype=np.float32)
        self.art.set_joint_position_targets(targets, joint_indices=joint_indices)

    def _set_joint_positions_immediate(self, position_targets: dict[str, float]) -> None:
        if not position_targets:
            return
        joints = list(position_targets.keys())
        joint_indices = np.asarray([self.dof_index_by_name[name] for name in joints], dtype=np.int64)
        targets = np.asarray([[float(position_targets[jn]) for jn in joints]], dtype=np.float32)
        self.art.set_joint_positions(targets, joint_indices=joint_indices)

    def _set_joint_velocity_targets(self, velocity_targets: dict[str, float]) -> None:
        if not velocity_targets:
            return
        joints = list(velocity_targets.keys())
        joint_indices = np.asarray([self.dof_index_by_name[name] for name in joints], dtype=np.int64)
        targets = np.asarray([[float(velocity_targets[jn]) for jn in joints]], dtype=np.float32)
        self.art.set_joint_velocity_targets(targets, joint_indices=joint_indices)

    def apply_plan_initial_pose(self) -> None:
        if self.plan_initial_positions:
            self._set_joint_positions_immediate(self.plan_initial_positions)
        if self.all_joint_names:
            self._set_joint_velocity_targets({str(name): 0.0 for name in self.all_joint_names})

    def _set_root_velocity(self, linear_xyz: np.ndarray, angular_xyz: np.ndarray | None = None) -> None:
        ang = np.zeros(3, dtype=np.float32) if angular_xyz is None else _ensure_float3(angular_xyz)
        lin = _ensure_float3(linear_xyz)
        vel = np.asarray([[lin[0], lin[1], lin[2], ang[0], ang[1], ang[2]]], dtype=np.float32)
        self.art.set_velocities(vel)

    def _zero_motion(self) -> None:
        if self.has_base_motion:
            self._set_root_velocity(np.zeros(3, dtype=np.float32))
        if self.all_joint_names:
            joint_indices = np.asarray([self.dof_index_by_name[name] for name in self.all_joint_names], dtype=np.int64)
            self._set_joint_gains(self.all_joint_names, stiffness=0.0, damping=600.0)
            self.art.set_joint_velocity_targets(
                np.zeros((1, len(self.all_joint_names)), dtype=np.float32),
                joint_indices=joint_indices,
            )

    def _apply_mode_set(self, ctrl: dict[str, Any]) -> None:
        name = str(ctrl.get("name") or ctrl.get("mode") or "").strip()
        if name:
            self.mode_flags[name] = bool(ctrl.get("set", True))

    def _apply_joint_velocity_control(
        self,
        ctrl: dict[str, Any],
        seg_t: float,
        local_t: float,
        velocity_targets: dict[str, float],
    ) -> None:
        omega = float(ctrl.get("omega_radps", 0.0))
        if ctrl.get("ramp_to_omega_radps") is not None:
            omega1 = float(ctrl.get("ramp_to_omega_radps", omega))
            omega = omega + (omega1 - omega) * float(max(0.0, min(1.0, local_t)))
        decay = ctrl.get("decay")
        if isinstance(decay, dict) and str(decay.get("type") or "").strip().lower() == "exponential":
            tau = max(1.0e-6, float(decay.get("tau_s", 1.0)))
            min_omega = float(decay.get("min_omega_radps", 0.0))
            omega = max(min_omega, omega * math.exp(-seg_t / tau))
        joints = list(ctrl.get("joints") or [])
        if ctrl.get("joint"):
            joints = [str(ctrl["joint"])]
        for joint_name in joints:
            if joint_name in self.joint_name_set:
                velocity_targets[joint_name] = float(omega)

    def _apply_joint_position_control(
        self,
        seg_idx: int,
        ctrl: dict[str, Any],
        local_t: float,
        position_targets: dict[str, float],
    ) -> None:
        joint_name = str(ctrl.get("joint") or "").strip()
        if not joint_name or joint_name not in self.joint_name_set:
            return
        self._remember_segment_entry_positions(seg_idx, [joint_name])
        entry_map = self.segment_entry_positions.get(seg_idx, {})
        q_start = ctrl.get("q_start_rad")
        if q_start is None:
            q_start = entry_map.get(joint_name, 0.0)
        q_target = float(ctrl.get("q_target_rad", q_start))
        alpha = _curve_alpha(str(ctrl.get("curve") or "linear"), local_t)
        q_des = (1.0 - alpha) * float(q_start) + alpha * q_target
        position_targets[joint_name] = float(q_des)

    def _apply_hold_position_control(
        self,
        seg_idx: int,
        ctrl: dict[str, Any],
        position_targets: dict[str, float],
    ) -> None:
        joints = list(ctrl.get("joints") or [])
        if ctrl.get("joint"):
            joints = [str(ctrl["joint"])]
        joints = [str(j) for j in joints if str(j) in self.joint_name_set]
        if not joints:
            return
        self._remember_hold_positions(seg_idx, joints)
        for joint_name, q_hold in self.segment_hold_positions.get(seg_idx, {}).items():
            position_targets[joint_name] = float(q_hold)

    def _apply_spring_return_control(
        self,
        ctrl: dict[str, Any],
        step_dt: float,
        velocity_targets: dict[str, float],
    ) -> None:
        joint_name = str(ctrl.get("joint") or "").strip()
        if not joint_name or joint_name not in self.joint_name_set:
            return
        q = float(self._joint_positions([joint_name])[0])
        qd = float(self._joint_velocities([joint_name])[0])
        k = float(ctrl.get("spring_k", 4.0))
        c = float(ctrl.get("damping_c", 0.6))
        rest = float(ctrl.get("rest_position", ctrl.get("target_rad", 0.0)))
        qdd = -k * (q - rest) - c * qd
        velocity_targets[joint_name] = float(qd + qdd * step_dt)

    def _apply_base_control(
        self,
        ctrl: dict[str, Any],
        seg_t: float,
        base_linear: np.ndarray,
    ) -> None:
        mode = str(ctrl.get("mode") or "").strip()
        axis = _normalize_axis(ctrl.get("axis_world") or [1.0, 0.0, 0.0])
        if mode == "base_velocity":
            v_mps = float(ctrl.get("v_mps", ctrl.get("linear_velocity_mps", 0.0)))
            base_linear[:] = base_linear + axis * v_mps
            return
        if mode == "base_velocity_decay":
            v0 = float(ctrl.get("v0_mps", ctrl.get("v_mps", 0.0)))
            tau = max(1.0e-6, float(ctrl.get("tau_s", 1.0)))
            base_linear[:] = base_linear + axis * float(v0 * math.exp(-seg_t / tau))

    def _apply_default_hold(
        self,
        position_targets: dict[str, float],
        velocity_targets: dict[str, float],
    ) -> None:
        idle_joints = [
            str(name)
            for name in self.all_joint_names
            if str(name) not in position_targets and str(name) not in velocity_targets
        ]
        if not idle_joints:
            return
        q_idle = self._joint_positions(idle_joints)
        for joint_name, q_val in zip(idle_joints, q_idle.tolist()):
            position_targets[joint_name] = float(q_val)

    def on_physics_step(self, step_size: float) -> None:
        if self.finished:
            return
        dt = float(step_size if step_size and step_size > 0.0 else self.dt_nominal)
        t = float(self.sim_time)
        if t >= self.duration_s:
            self._zero_motion()
            self.finished = True
            return

        active_segments = self._active_segments(t)
        position_targets: dict[str, float] = {}
        velocity_targets: dict[str, float] = {}
        base_linear = np.zeros(3, dtype=np.float32)
        has_base_control = False

        for seg_idx, seg in active_segments:
            t0 = float(seg.get("t0", 0.0))
            t1 = float(seg.get("t1", t0))
            seg_duration = max(1.0e-6, t1 - t0)
            seg_t = max(0.0, t - t0)
            local_t = seg_t / seg_duration
            for ctrl in list(seg.get("controls") or []):
                mode = str(ctrl.get("mode") or "").strip()
                if mode not in SUPPORTED_CONTROL_MODES:
                    raise RuntimeError(f"Unsupported control mode during execution: {mode}")
                if mode == "mode_set":
                    self._apply_mode_set(ctrl)
                elif mode == "joint_velocity":
                    self._apply_joint_velocity_control(ctrl, seg_t, local_t, velocity_targets)
                elif mode == "joint_position":
                    self._apply_joint_position_control(seg_idx, ctrl, local_t, position_targets)
                elif mode == "hold_position":
                    self._apply_hold_position_control(seg_idx, ctrl, position_targets)
                elif mode == "spring_return":
                    self._apply_spring_return_control(ctrl, dt, velocity_targets)
                elif mode in {"base_velocity", "base_velocity_decay"}:
                    self._apply_base_control(ctrl, seg_t, base_linear)
                    has_base_control = True

        for joint_name in list(position_targets.keys()):
            velocity_targets.pop(joint_name, None)
        self._apply_default_hold(position_targets, velocity_targets)

        if position_targets:
            self._set_joint_gains(list(position_targets.keys()), stiffness=1400.0, damping=180.0)
            self._set_joint_positions_immediate(position_targets)
        if velocity_targets:
            self._set_joint_gains(list(velocity_targets.keys()), stiffness=0.0, damping=600.0)
            self._set_joint_velocity_targets(velocity_targets)
        if self.has_base_motion:
            if has_base_control:
                self._set_root_velocity(base_linear)
            else:
                self._set_root_velocity(np.zeros(3, dtype=np.float32))
        if self.visual_sync is not None:
            self.visual_sync.sync_from_articulation(self.art)

        self.sim_time = t + dt
        if self.sim_time >= self.duration_s:
            self._zero_motion()
            self.finished = True


def _hide_articulation_subtree(stage, articulation_root_path: str) -> None:
    from pxr import UsdGeom

    root = stage.GetPrimAtPath(articulation_root_path)
    if not root.IsValid():
        return
    try:
        UsdGeom.Imageable(root).MakeInvisible()
    except Exception:
        pass
    for prim in stage.Traverse():
        path_str = str(prim.GetPath())
        if path_str != articulation_root_path and not path_str.startswith(f"{articulation_root_path}/"):
            continue
        try:
            UsdGeom.Imageable(prim).MakeInvisible()
        except Exception:
            continue


def _collect_import_tree_debug(stage, articulation_root_path: str, link_names: list[str] | None = None) -> dict[str, Any]:
    from pxr import Usd

    root = stage.GetPrimAtPath(articulation_root_path)
    out: dict[str, Any] = {
        "articulation_root_path": str(articulation_root_path),
        "articulation_root_valid": bool(root.IsValid()),
        "top_level_children": [],
        "prim_count": 0,
        "mesh_count": 0,
        "mesh_paths_sample": [],
        "link_mesh_counts": {},
        "link_prim_samples": {},
    }
    if not root.IsValid():
        return out

    top_level_children = [str(child.GetName()) for child in root.GetChildren()]
    out["top_level_children"] = top_level_children[:128]
    tracked_links = [str(name) for name in (link_names or []) if str(name)]
    link_mesh_counts: dict[str, int] = {name: 0 for name in tracked_links}
    link_prim_samples: dict[str, list[str]] = {name: [] for name in tracked_links}
    mesh_paths_sample: list[str] = []

    for prim in Usd.PrimRange(root):
        out["prim_count"] = int(out["prim_count"]) + 1
        prim_path = str(prim.GetPath())
        prim_type = str(prim.GetTypeName() or "")
        path_segments = set(seg for seg in prim_path.split("/") if seg)
        if prim_type == "Mesh":
            out["mesh_count"] = int(out["mesh_count"]) + 1
            if len(mesh_paths_sample) < 128:
                mesh_paths_sample.append(prim_path)
        for link_name in tracked_links:
            if link_name not in path_segments:
                continue
            if len(link_prim_samples[link_name]) < 24:
                link_prim_samples[link_name].append(f"{prim_type}:{prim_path}")
            if prim_type == "Mesh":
                link_mesh_counts[link_name] = int(link_mesh_counts.get(link_name, 0)) + 1

    out["mesh_paths_sample"] = mesh_paths_sample
    out["link_mesh_counts"] = link_mesh_counts
    out["link_prim_samples"] = link_prim_samples
    return out


class URDFMeshVisualSync:
    def __init__(self, spec: dict[str, Any]) -> None:
        visual = spec.get("custom_visuals") or {}
        self.enabled = str(visual.get("mode") or "").strip().lower() == "urdf_meshes"
        self.urdf_path = Path(spec["import"]["urdf_path"]).resolve()
        self.urdf_dir = self.urdf_path.parent
        self.cache_dir = Path(visual.get("cache_dir") or (self.urdf_dir / ".isaac_mesh_visual_cache")).resolve()
        self.hide_imported_visuals = bool(visual.get("hide_imported_visuals", True))
        self.skip_empty_links = bool(visual.get("skip_empty_links", True))
        self.links, self.joints = _parse_urdf(self.urdf_path)
        self.link_root_paths: dict[str, str] = {}
        self.visual_count = sum(len(v) for v in self.links.values())
        self.renderable_links = sorted([str(name) for name, visuals in self.links.items() if visuals])
        self.empty_links = sorted([str(name) for name, visuals in self.links.items() if not visuals])
        self.converted_meshes: dict[str, str] = {}

    def _converted_mesh_usd_path(self, mesh_path: Path) -> Path:
        rel_key = str(mesh_path.resolve())
        mesh_hash = hashlib.sha1(rel_key.encode("utf-8")).hexdigest()[:12]
        safe_stem = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in mesh_path.stem)
        return self.cache_dir / f"{safe_stem}_{mesh_hash}.usd"

    def _ensure_mesh_usd(self, converter_mgr, mesh_path: Path) -> Path:
        import omni.kit.asset_converter as converter

        usd_path = self._converted_mesh_usd_path(mesh_path)
        if usd_path.exists():
            self.converted_meshes[str(mesh_path)] = str(usd_path)
            return usd_path
        usd_path.parent.mkdir(parents=True, exist_ok=True)
        ctx = converter.AssetConverterContext()
        task = converter_mgr.create_converter_task(str(mesh_path), str(usd_path), None, ctx)
        ok_convert = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
        if not ok_convert or not usd_path.exists():
            raise RuntimeError(f"Failed to convert URDF visual mesh to USD: {mesh_path}")
        self.converted_meshes[str(mesh_path)] = str(usd_path)
        return usd_path

    def attach_visuals(self, stage, converter_mgr, root_prim_path: str = "/World/CustomVisuals") -> None:
        if not self.enabled:
            return
        root = stage.DefinePrim(root_prim_path, "Xform")
        self.link_root_paths.clear()
        for link_name, visuals in self.links.items():
            if not visuals and self.skip_empty_links:
                continue
            link_root_path = f"{root_prim_path}/{link_name}"
            stage.DefinePrim(link_root_path, "Xform")
            self.link_root_paths[link_name] = link_root_path
            for vis_idx, visual in enumerate(visuals):
                mesh_path = _resolve_mesh_path(visual.get("filename"), self.urdf_dir)
                if mesh_path is None or not mesh_path.exists():
                    continue
                usd_path = self._ensure_mesh_usd(converter_mgr, mesh_path)
                vis_prim = stage.DefinePrim(f"{link_root_path}/visual_{vis_idx:03d}", "Xform")
                vis_prim.GetReferences().AddReference(str(usd_path))
                local_tf = _origin_to_matrix(visual.get("origin_xyz"), visual.get("origin_rpy")) @ _scale_to_matrix(visual.get("scale"))
                _set_transform_prim(vis_prim, local_tf)

    def hide_articulation_meshes(self, stage, articulation_root_path: str) -> None:
        if self.hide_imported_visuals:
            _hide_articulation_subtree(stage, articulation_root_path)

    def _current_joint_pos_map(self, articulation) -> dict[str, float]:
        out: dict[str, float] = {}
        if articulation.dof_names:
            q = np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
            for name, val in zip(list(articulation.dof_names), q.tolist()):
                out[str(name)] = float(val)
        return out

    def sync_from_articulation(self, articulation) -> None:
        import omni.usd

        if not self.enabled or not self.link_root_paths:
            return
        q_map = self._current_joint_pos_map(articulation)
        root_positions, root_orientations = articulation.get_world_poses()
        base_tf = np.eye(4, dtype=np.float64)
        base_tf[:3, :3] = _quat_xyzw_to_matrix(np.asarray(root_orientations[0], dtype=np.float64))[:3, :3]
        base_tf[:3, 3] = np.asarray(root_positions[0], dtype=np.float64)
        link_tf = _compute_link_transforms(self.links, self.joints, q_map, base_tf=base_tf)
        stage = omni.usd.get_context().get_stage()
        for link_name, prim_path in self.link_root_paths.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            world_tf = link_tf.get(link_name, base_tf)
            _set_transform_prim(prim, world_tf)

    def debug_state(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": "urdf_meshes",
            "link_count": int(len(self.links)),
            "renderable_link_count": int(len(self.renderable_links)),
            "empty_link_count": int(len(self.empty_links)),
            "visual_count": int(self.visual_count),
            "attached_link_roots": int(len(self.link_root_paths)),
            "converted_mesh_count": int(len(self.converted_meshes)),
            "renderable_links": list(self.renderable_links),
            "empty_links": list(self.empty_links),
            "skip_empty_links": bool(self.skip_empty_links),
            "cache_dir": str(self.cache_dir),
        }


class VisualSync:
    def __init__(self, spec: dict[str, Any]) -> None:
        visual = spec.get("visual_asset") or {}
        self.enabled = bool(visual)
        self.report_json_path = str(visual.get("report_json_path") or "")
        parts = list(visual.get("parts") or [])
        if (not parts) and self.report_json_path:
            try:
                report = json.loads(Path(self.report_json_path).read_text(encoding="utf-8"))
            except Exception:
                report = {}
            parts = list(report.get("parts") or [])
        self.parts = parts
        self.part_prim_paths: dict[str, str] = {}
        self.link_node_names: dict[str, list[str]] = {}
        self.requested_node_names: set[str] = set()
        self.rest_link_tf: dict[str, np.ndarray] = {}
        self.node_rest_world_tf: dict[str, np.ndarray] = {}
        self.visual_root_path: str | None = None
        self.urdf_path = Path(spec["import"]["urdf_path"]).resolve()
        self.links, self.joints = _parse_urdf(self.urdf_path)
        self.rest_link_tf = _compute_link_transforms(self.links, self.joints, {})
        for row in self.parts:
            link_name = str(row.get("link_name") or "")
            node_name = str(row.get("part_node") or "")
            submesh_names = [str(n) for n in list(row.get("part_node_submeshes") or []) if str(n)]
            node_names = []
            if node_name:
                node_names.append(node_name)
            node_names.extend(submesh_names)
            node_names = list(dict.fromkeys(node_names))
            if not link_name or not node_names:
                continue
            self.link_node_names[link_name] = node_names
            self.requested_node_names.update(node_names)
            if row.get("rest_transform") is not None:
                rest_tf = np.asarray(row.get("rest_transform"), dtype=np.float64)
                for item_node_name in node_names:
                    self.node_rest_world_tf[item_node_name] = rest_tf

    def attach_visual_usd(self, stage, usd_path: Path, root_prim_path: str = "/World/VisualAsset") -> None:
        root = stage.DefinePrim(root_prim_path, "Xform")
        root.GetReferences().AddReference(str(usd_path))
        self.visual_root_path = str(root.GetPath())

    def _discover_part_prims(self, stage) -> None:
        if len(self.part_prim_paths) >= len(self.requested_node_names):
            return
        from pxr import UsdGeom

        xform_cache = UsdGeom.XformCache()
        for prim in stage.Traverse():
            name = prim.GetName()
            if name in self.requested_node_names:
                self.part_prim_paths[name] = str(prim.GetPath())
                self.node_rest_world_tf.setdefault(
                    name,
                    np.asarray(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float64),
                )

    def hide_articulation_meshes(self, stage, articulation_root_path: str) -> None:
        _hide_articulation_subtree(stage, articulation_root_path)

    def _current_joint_pos_map(self, articulation) -> dict[str, float]:
        out: dict[str, float] = {}
        if articulation.dof_names:
            q = np.asarray(articulation.get_joint_positions(), dtype=np.float64).reshape(-1)
            for name, val in zip(list(articulation.dof_names), q.tolist()):
                out[str(name)] = float(val)
        return out

    def sync_from_articulation(self, articulation) -> None:
        import omni.usd

        q_map = self._current_joint_pos_map(articulation)
        root_positions, root_orientations = articulation.get_world_poses()
        base_tf = np.eye(4, dtype=np.float64)
        base_tf[:3, :3] = _quat_xyzw_to_matrix(np.asarray(root_orientations[0], dtype=np.float64))[:3, :3]
        base_tf[:3, 3] = np.asarray(root_positions[0], dtype=np.float64)
        link_tf = _compute_link_transforms(self.links, self.joints, q_map, base_tf=base_tf)
        stage = omni.usd.get_context().get_stage()
        self._discover_part_prims(stage)
        if not self.part_prim_paths:
            return
        from pxr import UsdGeom

        xform_cache = UsdGeom.XformCache()
        for row in self.parts:
            link_name = str(row.get("link_name") or "")
            rest_link = self.rest_link_tf.get(link_name)
            cur_link = link_tf.get(link_name)
            if rest_link is None or cur_link is None:
                continue
            delta = np.asarray(cur_link, dtype=np.float64) @ np.linalg.inv(np.asarray(rest_link, dtype=np.float64))
            for node_name in self.link_node_names.get(link_name, []):
                prim_path = self.part_prim_paths.get(node_name)
                rest_node = self.node_rest_world_tf.get(node_name)
                if prim_path is None or rest_node is None:
                    continue
                prim = stage.GetPrimAtPath(prim_path)
                world_tf = delta @ np.asarray(rest_node, dtype=np.float64)
                parent = prim.GetParent()
                parent_world_tf = np.eye(4, dtype=np.float64)
                if parent is not None and parent.IsValid():
                    parent_world_tf = np.asarray(xform_cache.GetLocalToWorldTransform(parent), dtype=np.float64)
                local_tf = np.linalg.inv(parent_world_tf) @ world_tf
                _set_transform_prim(prim, local_tf)

    def debug_state(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "report_json_path": self.report_json_path,
            "requested_node_count": int(len(self.requested_node_names)),
            "discovered_node_count": int(len(self.part_prim_paths)),
            "requested_nodes": sorted(self.requested_node_names),
            "discovered_nodes": sorted(self.part_prim_paths.keys()),
            "link_node_counts": {str(k): int(len(v)) for k, v in self.link_node_names.items()},
        }


class BakedGLBRenderVisual:
    def __init__(self, spec: dict[str, Any]) -> None:
        visual = spec.get("render_visual") or {}
        self.enabled = str(visual.get("mode") or "").strip().lower() == "baked_glb"
        self.glb_path = Path(visual.get("glb_path") or "").resolve() if visual.get("glb_path") else None
        self.usd_path = Path(visual.get("usd_path") or "").resolve() if visual.get("usd_path") else None
        self.hide_imported_urdf_visuals = bool(visual.get("hide_imported_urdf_visuals", True))
        self.root_path: str | None = None
        self.converted = False
        self.last_time_s = 0.0
        self.time_update_errors: list[str] = []

    def ensure_usd(self, converter_mgr) -> Path:
        import omni.kit.asset_converter as converter

        if self.glb_path is None or not self.glb_path.exists():
            raise FileNotFoundError(f"Missing baked GLB render visual: {self.glb_path}")
        if self.usd_path is None:
            self.usd_path = self.glb_path.with_suffix(".usd")
        if self.usd_path.exists():
            return self.usd_path
        self.usd_path.parent.mkdir(parents=True, exist_ok=True)
        ctx = converter.AssetConverterContext()
        task = converter_mgr.create_converter_task(str(self.glb_path), str(self.usd_path), None, ctx)
        ok_convert = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
        if not ok_convert or not self.usd_path.exists():
            raise RuntimeError(f"Failed to convert baked GLB render visual to USD: {self.glb_path}")
        self.converted = True
        return self.usd_path

    def attach(self, stage, converter_mgr, root_prim_path: str = "/World/BakedRenderVisual") -> None:
        if not self.enabled:
            return
        usd_path = self.ensure_usd(converter_mgr)
        root = stage.DefinePrim(root_prim_path, "Xform")
        root.GetReferences().AddReference(str(usd_path))
        self.root_path = str(root.GetPath())

    def hide_articulation_meshes(self, stage, articulation_root_path: str) -> None:
        if self.hide_imported_urdf_visuals:
            _hide_articulation_subtree(stage, articulation_root_path)

    def sync_from_articulation(self, articulation) -> None:
        return

    def set_time(self, timeline, time_s: float) -> None:
        self.last_time_s = float(time_s)
        if timeline is None:
            return
        try:
            if hasattr(timeline, "set_current_time"):
                timeline.set_current_time(float(time_s))
            elif hasattr(timeline, "set_current_timecode"):
                timeline.set_current_timecode(float(time_s))
        except Exception as exc:
            if len(self.time_update_errors) < 8:
                self.time_update_errors.append(repr(exc))

    def debug_state(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mode": "baked_glb",
            "glb_path": str(self.glb_path) if self.glb_path is not None else "",
            "usd_path": str(self.usd_path) if self.usd_path is not None else "",
            "root_path": self.root_path,
            "converted": bool(self.converted),
            "hide_imported_urdf_visuals": bool(self.hide_imported_urdf_visuals),
            "last_time_s": float(self.last_time_s),
            "time_update_errors": list(self.time_update_errors),
        }


def _locate_png_sequence(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("frame_*.png"))


def _encode_video(frames_dir: Path, fps: int, video_path: Path) -> bool:
    pngs = _locate_png_sequence(frames_dir)
    if not pngs:
        return False
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "format=yuv420p",
        str(video_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return False
    return video_path.exists()


def run_spec(spec_path: Path, *, headless: bool = True, resolution: tuple[int, int] | None = None, max_frames: int | None = None) -> dict[str, Any]:
    from isaacsim import SimulationApp

    launch_cfg = {"headless": bool(headless), "renderer": "RaytracedLighting"}
    simulation_app = SimulationApp(launch_config=launch_cfg)
    try:
        import carb.settings
        import omni.kit.commands
        import omni.kit.asset_converter as converter
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.prims import get_articulation_root_api_prim_path
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Sdf, UsdLux

        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        fps = int(spec.get("meta", {}).get("fps", 30))
        duration_s = float(spec.get("meta", {}).get("duration_s", 4.0))
        capture_resolution = tuple(int(x) for x in (resolution or tuple(spec.get("camera", {}).get("resolution", [1280, 720]))))
        out_dir = Path(spec.get("outputs", {}).get("out_dir") or Path(spec_path).parent / "outputs").resolve()
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_path = out_dir / str(spec.get("outputs", {}).get("video_name") or f"{spec.get('asset_name', 'asset')}_isaac.mp4")
        trajectory_npz_name = str(spec.get("outputs", {}).get("trajectory_npz_name") or "trajectory.npz")
        trajectory_jsonl_name = str(spec.get("outputs", {}).get("trajectory_jsonl_name") or "trajectory.jsonl")
        debug_path = out_dir / "runtime_debug.json"
        debug_events_path = out_dir / "runtime_debug_events.jsonl"
        if debug_events_path.exists():
            debug_events_path.unlink()

        def write_debug(payload: dict[str, Any]) -> None:
            try:
                debug_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                with debug_events_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            except Exception:
                pass

        write_debug(
            {
                "stage": "startup",
                "spec_path": str(Path(spec_path).resolve()),
                "asset_name": spec.get("asset_name"),
            }
        )

        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
        carb.settings.get_settings().set("/omni/replicator/asyncRendering", False)
        carb.settings.get_settings().set("/app/asyncRendering", False)
        carb.settings.get_settings().set("/rtx/post/backgroundZeroAlpha/enabled", True)
        carb.settings.get_settings().set("/rtx/post/backgroundZeroAlpha/backgroundComposite", False)
        carb.settings.get_settings().set("/rtx/post/backgroundZeroAlpha/outputAlphaInComposite", True)
        carb.settings.get_settings().set("/rtx/post/backgroundZeroAlpha/blackBackgroundInComposite", False)
        carb.settings.get_settings().set("/rtx/post/backgroundZeroAlpha/backgroundDefaultColor", [1.0, 1.0, 1.0])

        create_new_stage()
        write_debug({"stage": "stage_created"})
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / fps, rendering_dt=1.0 / fps)
        ground_plane = world.scene.add_ground_plane(size=5000.0, color=np.asarray([1.0, 1.0, 1.0], dtype=np.float32))
        try:
            ground_plane.set_visibility(False)
        except Exception:
            pass
        stage = omni.usd.get_context().get_stage()
        timeline = omni.timeline.get_timeline_interface()
        try:
            stage.SetTimeCodesPerSecond(float(fps))
            stage.SetFramesPerSecond(float(fps))
            stage.SetStartTimeCode(0.0)
            stage.SetEndTimeCode(float(round(duration_s * fps)))
            timeline.set_current_time(0.0)
        except Exception as exc:
            write_debug({"stage": "timeline_setup_failed", "error": repr(exc)})
        dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
        dome.CreateIntensityAttr(1500.0)
        dome.CreateColorAttr().Set((1.0, 1.0, 1.0))
        light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
        light.CreateIntensityAttr(2200.0)

        urdf_path = str(Path(spec["import"]["urdf_path"]).resolve())
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        import_config.merge_fixed_joints = False
        import_config.fix_base = bool(spec.get("import", {}).get("fix_base", False))
        import_config.make_default_prim = True
        import_config.create_physics_scene = False
        if hasattr(import_config, "self_collision"):
            import_config.self_collision = False
        ok, base_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=urdf_path,
            import_config=import_config,
            get_articulation_root=False,
        )
        if not ok:
            raise RuntimeError(f"URDF import failed for {urdf_path}")
        art_root_path = get_articulation_root_api_prim_path(str(base_path))
        write_debug({"stage": "urdf_imported", "base_path": str(base_path), "art_root_path": str(art_root_path)})
        placement_delta = _initial_root_translation(spec)
        write_debug(
            {
                "stage": "initial_root_placement_requested",
                "placement_translation": [float(x) for x in placement_delta.tolist()],
                "placement": spec.get("placement"),
            }
        )
        authored_placement = _author_root_translation(stage, str(base_path), placement_delta)
        write_debug({"stage": "initial_root_placement_authored", "placement_result": authored_placement})
        try:
            import_tree_debug = _collect_import_tree_debug(
                stage,
                art_root_path,
                list((spec.get("import") or {}).get("renderable_links") or []),
            )
            base_tree_debug = _collect_import_tree_debug(
                stage,
                str(base_path),
                list((spec.get("import") or {}).get("renderable_links") or []),
            )
            write_debug(
                {
                    "stage": "urdf_import_tree",
                    "import_tree": import_tree_debug,
                    "base_tree": base_tree_debug,
                }
            )
        except BaseException as exc:
            write_debug({"stage": "urdf_import_tree_failed", "error": repr(exc)})
        write_debug({"stage": "before_articulation_ctor"})
        articulation = Articulation(prim_paths_expr=art_root_path, name=f"{spec.get('asset_name', 'asset')}_articulation")
        write_debug({"stage": "after_articulation_ctor"})
        try:
            render_visual_mode = str((spec.get("render_visual") or {}).get("mode") or "").strip().lower()
            custom_visual_mode = str((spec.get("custom_visuals") or {}).get("mode") or "").strip().lower()
            if render_visual_mode == "baked_glb":
                visual_sync = BakedGLBRenderVisual(spec)
            elif custom_visual_mode == "urdf_meshes":
                visual_sync = URDFMeshVisualSync(spec)
            else:
                visual_sync = VisualSync(spec) if spec.get("visual_asset") else None
        except BaseException as exc:
            write_debug({"stage": "visual_sync_ctor_failed", "error": repr(exc)})
            raise
        write_debug(
            {
                "stage": "visual_sync_created",
                "visual_sync_enabled": bool(visual_sync is not None and visual_sync.enabled),
                "visual": visual_sync.debug_state() if visual_sync is not None else None,
            }
        )

        if visual_sync is not None and visual_sync.enabled:
            if isinstance(visual_sync, BakedGLBRenderVisual):
                write_debug(
                    {
                        "stage": "render_visual_pre_attach",
                        "mode": "baked_glb",
                        "glb_path": str(visual_sync.glb_path),
                        "usd_path": str(visual_sync.usd_path),
                    }
                )
                visual_sync.attach(stage, converter.get_instance())
                visual_sync.hide_articulation_meshes(stage, str(base_path))
                visual_sync.set_time(timeline, 0.0)
                write_debug({"stage": "render_visual_attached", "visual": visual_sync.debug_state()})
            elif isinstance(visual_sync, URDFMeshVisualSync):
                write_debug(
                    {
                        "stage": "visual_sync_pre_attach",
                        "custom_visual_mode": "urdf_meshes",
                        "cache_dir": str(visual_sync.cache_dir),
                    }
                )
                visual_sync.attach_visuals(stage, converter.get_instance())
                write_debug({"stage": "visual_sync_attached", "visual": visual_sync.debug_state()})
                visual_sync.hide_articulation_meshes(stage, art_root_path)
                write_debug({"stage": "visual_attached", "visual": visual_sync.debug_state()})
            else:
                usd_path = Path(spec["visual_asset"]["usd_path"]).resolve()
                write_debug({"stage": "visual_sync_pre_attach", "visual_usd_exists": bool(usd_path.exists()), "visual_usd_path": str(usd_path)})
                if not usd_path.exists():
                    mgr = converter.get_instance()
                    ctx = converter.AssetConverterContext()
                    task = mgr.create_converter_task(str(Path(spec["visual_asset"]["glb_path"]).resolve()), str(usd_path), None, ctx)
                    ok_convert = asyncio.get_event_loop().run_until_complete(task.wait_until_finished())
                    if not ok_convert:
                        raise RuntimeError(f"Failed to convert visual GLB to USD: {spec['visual_asset']['glb_path']}")
                    write_debug({"stage": "visual_sync_converted", "visual_usd_path": str(usd_path)})
                visual_sync.attach_visual_usd(stage, usd_path)
                write_debug({"stage": "visual_sync_attached"})
                visual_sync.hide_articulation_meshes(stage, art_root_path)
                write_debug(
                    {
                        "stage": "visual_attached",
                        "visual_usd_path": str(usd_path),
                        "visual": visual_sync.debug_state(),
                    }
                )

        world.reset()
        write_debug({"stage": "world_reset"})
        if isinstance(visual_sync, BakedGLBRenderVisual) and visual_sync.enabled:
            visual_sync.hide_articulation_meshes(stage, str(base_path))
            visual_sync.set_time(timeline, 0.0)
            write_debug({"stage": "render_visual_after_reset", "visual": visual_sync.debug_state()})
        world.play()
        world.step(render=False)
        write_debug({"stage": "world_first_step"})
        articulation.initialize()
        if not articulation.is_physics_handle_valid():
            world.step(render=False)
            articulation.initialize()
        if not articulation.is_physics_handle_valid():
            raise RuntimeError(f"Failed to initialize articulation physics handle at {art_root_path}")
        write_debug({"stage": "articulation_initialized", "dof_names": list(articulation.dof_names or [])})
        runtime_placement = _apply_root_pose_translation(articulation, placement_delta)
        write_debug({"stage": "initial_root_placement_runtime", "placement_result": runtime_placement})
        if runtime_placement.get("applied"):
            world.step(render=False)
            write_debug({"stage": "initial_root_placement_runtime_step"})
        executor = TimelinePhysicsExecutor(articulation, spec, fps=fps, visual_sync=visual_sync)
        executor.apply_plan_initial_pose()
        write_debug({"stage": "plan_initial_pose_applied", "joint_positions": executor.plan_initial_positions})

        camera_spec = spec.get("camera") or {}
        camera = rep.create.camera(
            position=tuple(float(x) for x in camera_spec.get("eye", [3.0, -3.0, 2.0])),
            look_at=tuple(float(x) for x in camera_spec.get("target", [0.0, 0.0, 0.0])),
        )
        camera_root_prim = camera.get_output_prims()["prims"][0]
        camera_prim = None
        for child in camera_root_prim.GetChildren():
            if child.GetTypeName() == "Camera":
                camera_prim = child
                break
        if camera_prim is None and camera_root_prim.GetTypeName() == "Camera":
            camera_prim = camera_root_prim
        aspect_ratio = float(capture_resolution[0]) / max(1.0, float(capture_resolution[1]))
        view_cfg = camera_spec.get("view") or {}
        _set_camera_fov(camera_prim, view_cfg.get("fov_deg"), aspect_ratio)
        render_product = rep.create.render_product(camera, capture_resolution, force_new=True, name=f"{spec.get('asset_name', 'asset')}_rp")
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach(render_product)
        rep.orchestrator.preview()
        write_debug({"stage": "camera_and_annotator_ready"})

        for _ in range(4):
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
        write_debug({"stage": "render_warmup_done"})

        if visual_sync is not None:
            visual_sync.sync_from_articulation(articulation)
            if isinstance(visual_sync, BakedGLBRenderVisual):
                visual_sync.set_time(timeline, 0.0)
            write_debug({"stage": "visual_first_sync", "visual": visual_sync.debug_state()})

        world.add_physics_callback("timeline_executor", executor.on_physics_step)
        follow_camera = FollowCameraRig(articulation, camera_root_prim, camera_spec) if bool(camera_spec.get("follow_base")) else None
        if follow_camera is not None:
            follow_camera.update()
        write_debug(
            {
                "stage": "ready_to_capture",
                "follow_camera": bool(follow_camera is not None),
                "camera": camera_spec,
                "visual": visual_sync.debug_state() if visual_sync is not None else None,
            }
        )

        total_frames = int(round(duration_s * fps)) + 1
        if max_frames is not None:
            total_frames = min(total_frames, int(max_frames))

        trajectory_records: list[dict[str, Any]] = []
        if isinstance(visual_sync, BakedGLBRenderVisual):
            visual_sync.set_time(timeline, 0.0)
        rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
        _write_rgba_png(rgb_annot.get_data(), frames_dir / "frame_0000.png")
        trajectory_records.append(_capture_runtime_record(articulation, 0, 0.0, executor.all_joint_names))
        frame_count = 1
        while simulation_app.is_running() and frame_count < total_frames:
            world.step(render=False)
            if follow_camera is not None:
                follow_camera.update()
            if isinstance(visual_sync, BakedGLBRenderVisual):
                visual_sync.set_time(timeline, float(frame_count) / float(max(fps, 1)))
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)
            _write_rgba_png(rgb_annot.get_data(), frames_dir / f"frame_{frame_count:04d}.png")
            trajectory_records.append(
                _capture_runtime_record(
                    articulation,
                    frame_count,
                    min(float(duration_s), float(frame_count) / float(max(fps, 1))),
                    executor.all_joint_names,
                )
            )
            frame_count += 1

        executor._zero_motion()
        world.step(render=False)
        trajectory_npz_path, trajectory_jsonl_path = _write_runtime_trajectory(
            out_dir,
            trajectory_records,
            executor.all_joint_names,
            trajectory_npz_name,
            trajectory_jsonl_name,
        )

        report = {
            "spec_path": str(Path(spec_path).resolve()),
            "asset_name": spec.get("asset_name"),
            "articulation_root_path": str(art_root_path),
            "frames_dir": str(frames_dir),
            "video_path": str(video_path),
            "trajectory_npz_path": str(trajectory_npz_path),
            "trajectory_jsonl_path": str(trajectory_jsonl_path),
            "frame_count": int(frame_count),
            "fps": int(fps),
            "duration_s": float(duration_s),
            "headless": bool(headless),
            "resolution": list(capture_resolution),
            "video_encoded": bool(_encode_video(frames_dir, fps, video_path)),
            "follow_camera": bool(follow_camera is not None),
            "visual": visual_sync.debug_state() if visual_sync is not None else None,
        }
        (out_dir / "execution_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        simulation_app.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a compiled Isaac Sim timeline executor bundle.")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--no-headless", dest="headless", action="store_false")
    ap.add_argument("--resolution", type=int, nargs=2, default=None)
    ap.add_argument("--max_frames", type=int, default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_spec(
        Path(args.spec),
        headless=bool(args.headless),
        resolution=tuple(args.resolution) if args.resolution is not None else None,
        max_frames=args.max_frames,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
