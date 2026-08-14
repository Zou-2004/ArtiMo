#!/usr/bin/env python3
"""Visual and numeric diagnostics for one declared link-local contact pose.

This tool never accepts/rejects a rollout and never changes source inputs.  It
helps an agent see whether a proposed point is on the declared collision link,
which link is actually nearest, and how to move the point onto the surface.
Contact local +Z is the outward surface normal; positive grasp/precontact
offsets are rendered into free space and robot approach is local -Z.  The
shared harness maps Panda grasptarget +Z (palm to fingertips) to contact -Z.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

import run_artimo_physics as ph


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]


def _resolve(value: Path) -> Path:
    path = value.expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _quat(values: list[float]) -> list[float]:
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (4,) or float(np.linalg.norm(q)) < 1e-9:
        raise ValueError(f"Invalid quaternion: {values}")
    return (q / np.linalg.norm(q)).tolist()


def _maps(body: int, client: int) -> tuple[dict[str, int], dict[int, str]]:
    base_name = p.getBodyInfo(body, physicsClientId=client)[0].decode("utf-8")
    by_name: dict[str, int] = {base_name: -1}
    by_index: dict[int, str] = {-1: base_name}
    for index in range(p.getNumJoints(body, physicsClientId=client)):
        info = p.getJointInfo(body, index, physicsClientId=client)
        by_name[info[12].decode("utf-8")] = index
        by_index[index] = info[12].decode("utf-8")
    return by_name, by_index


def _joint_map(body: int, client: int) -> dict[str, int]:
    return {
        p.getJointInfo(body, index, physicsClientId=client)[1].decode("utf-8"): index
        for index in range(p.getNumJoints(body, physicsClientId=client))
    }


def _link_pose(body: int, link: int, client: int) -> tuple[list[float], list[float]]:
    if link == -1:
        position, rotation = p.getBasePositionAndOrientation(body, physicsClientId=client)
    else:
        state = p.getLinkState(body, link, computeForwardKinematics=True, physicsClientId=client)
        position, rotation = state[4], state[5]
    return list(position), list(rotation)


def _global_aabb(body: int, client: int) -> tuple[list[float], list[float]]:
    boxes = [p.getAABB(body, -1, physicsClientId=client)]
    boxes.extend(
        p.getAABB(body, index, physicsClientId=client)
        for index in range(p.getNumJoints(body, physicsClientId=client))
    )
    return (
        [min(box[0][axis] for box in boxes) for axis in range(3)],
        [max(box[1][axis] for box in boxes) for axis in range(3)],
    )


def _initial_joints(trajectory: Path | None) -> dict[str, float]:
    if trajectory is None:
        return {}
    for line in trajectory.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            values = row.get("joint_angles", {})
            return {str(name): float(value) for name, value in values.items()}
    raise ValueError(f"Empty trajectory: {trajectory}")


def _parse_joint_overrides(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--joint expects name=value, got {value!r}")
        name, numeric = value.split("=", 1)
        result[name] = float(numeric)
    return result


def _point_to_link(
    probe: int,
    body: int,
    link: int,
    probe_radius: float,
    search_distance: float,
    client: int,
) -> dict[str, Any] | None:
    points = p.getClosestPoints(
        probe,
        body,
        search_distance,
        linkIndexB=link,
        physicsClientId=client,
    )
    if not points:
        return None
    point = min(points, key=lambda item: abs(float(item[8]) + probe_radius))
    signed = float(point[8]) + probe_radius
    link_position, link_rotation = _link_pose(body, link, client)
    inverse_position, inverse_rotation = p.invertTransform(link_position, link_rotation)
    closest_local, _ = p.multiplyTransforms(
        inverse_position,
        inverse_rotation,
        point[6],
        [0.0, 0.0, 0.0, 1.0],
    )
    return {
        "signed_point_to_surface_distance_m": signed,
        "closest_surface_point_world_m": list(point[6]),
        "closest_surface_point_link_m": list(closest_local),
        "normal_on_link_world": list(point[7]),
    }


def _marker(position: list[float], radius: float, color: list[float], client: int) -> int:
    visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color,
        physicsClientId=client,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=position,
        physicsClientId=client,
    )


def _render_view(
    target: list[float],
    distance: float,
    yaw: float,
    pitch: float,
    label: str,
    client: int,
) -> Image.Image:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=client,
    )
    projection = p.computeProjectionMatrixFOV(55.0, 4.0 / 3.0, 0.01, 20.0)
    _, _, rgba, _, _ = p.getCameraImage(
        640,
        480,
        viewMatrix=view,
        projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 640, 28), fill=(18, 22, 29))
    draw.text((10, 8), label, fill=(245, 248, 252), font=ImageFont.load_default())
    return image


def inspect(args: argparse.Namespace) -> dict[str, Any]:
    task = ph._read_json(_resolve(args.task_spec))
    source_urdf = ph._resolve(task["inputs"]["urdf"])
    urdf = ph.resolve_simulation_urdf(task, {}, source_urdf)
    trajectory_value = task["inputs"].get("trajectory")
    trajectory = ph._resolve(trajectory_value) if trajectory_value else None
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet")
    try:
        requested_base = [float(value) for value in args.object_base_translation_m]
        base_rotation = _quat([float(value) for value in args.object_base_rotation_xyzw])
        body = p.loadURDF(
            str(urdf),
            basePosition=requested_base,
            baseOrientation=base_rotation,
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
            physicsClientId=client,
        )
        joints = _joint_map(body, client)
        joint_values = _initial_joints(trajectory)
        joint_values.update(_parse_joint_overrides(args.joint))
        for name, value in joint_values.items():
            if name in joints:
                p.resetJointState(body, joints[name], value, physicsClientId=client)
        effective_base = requested_base.copy()
        if args.auto_ground:
            for _ in range(4):
                p.performCollisionDetection(physicsClientId=client)
                minimum_z = _global_aabb(body, client)[0][2]
                correction = float(args.support_top_z_m) + float(args.ground_clearance_m) - minimum_z
                if abs(correction) <= 1e-8:
                    break
                effective_base[2] += correction
                p.resetBasePositionAndOrientation(
                    body,
                    effective_base,
                    base_rotation,
                    physicsClientId=client,
                )

        links, link_names = _maps(body, client)
        if args.contact_link not in links:
            raise KeyError(f"Unknown contact link {args.contact_link!r}; available: {sorted(links)}")
        target_link = links[args.contact_link]
        link_position, link_rotation = _link_pose(body, target_link, client)
        local_point = [float(value) for value in args.point_link_m]
        local_rotation, contact_frame = ph._canonical_contact_rotation_xyzw(
            task,
            {
                "contact_link": args.contact_link,
                "contact_pose_link": {"translation_m": local_point},
            },
            0.0,
        )
        world_point, world_rotation = p.multiplyTransforms(
            link_position,
            link_rotation,
            local_point,
            local_rotation,
            physicsClientId=client,
        )
        outward_world = list(p.rotateVector(world_rotation, [0.0, 0.0, 1.0]))
        grasp_depth_adjustment = float(args.grasp_depth_m)
        grasp_depth = (
            ph.PANDA_CENTERED_GRASP_BASELINE_M + grasp_depth_adjustment
        )
        precontact_offset = float(args.precontact_offset_m)
        if precontact_offset < 0.0:
            raise ValueError("precontact offset must be non-negative")
        grasp_target_world = [
            world_point[axis] + outward_world[axis] * grasp_depth
            for axis in range(3)
        ]
        precontact_world = [
            world_point[axis] + outward_world[axis] * (grasp_depth + precontact_offset)
            for axis in range(3)
        ]

        probe_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=float(args.probe_radius_m),
            physicsClientId=client,
        )
        probe = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=probe_shape,
            basePosition=world_point,
            physicsClientId=client,
        )
        distances: dict[str, dict[str, Any] | None] = {}
        for index, name in link_names.items():
            distances[name] = _point_to_link(
                probe,
                body,
                index,
                float(args.probe_radius_m),
                float(args.search_distance_m),
                client,
            )
        target = distances[args.contact_link]
        available = [(name, value) for name, value in distances.items() if value is not None]
        nearest_name, nearest = min(
            available,
            key=lambda item: abs(float(item[1]["signed_point_to_surface_distance_m"])),
        ) if available else (None, None)
        on_target = bool(
            target is not None
            and abs(float(target["signed_point_to_surface_distance_m"])) <= float(args.surface_tolerance_m)
        )
        correction_local = None
        if target is not None:
            correction_local = [
                float(target["closest_surface_point_link_m"][axis]) - local_point[axis]
                for axis in range(3)
            ]

        # Visual overlay: target link yellow, proposed point green/red, closest
        # target surface blue, grasp-target standoff cyan, and outward
        # precontact samples magenta.
        p.changeVisualShape(
            body,
            target_link,
            rgbaColor=[1.0, 0.68, 0.08, 1.0],
            physicsClientId=client,
        )
        marker_radius = 0.008
        _marker(list(world_point), marker_radius, [0.15, 0.9, 0.25, 1.0] if on_target else [0.95, 0.12, 0.12, 1.0], client)
        if target is not None:
            _marker(target["closest_surface_point_world_m"], 0.006, [0.1, 0.45, 1.0, 1.0], client)
        _marker(grasp_target_world, 0.006, [0.05, 0.85, 0.9, 1.0], client)
        sample_limit = max(0.08, max(0.0, grasp_depth) + precontact_offset)
        for distance in np.linspace(0.02, sample_limit, 4):
            position = [
                world_point[axis] + outward_world[axis] * float(distance)
                for axis in range(3)
            ]
            _marker(position, 0.0035, [0.72, 0.25, 0.92, 1.0], client)

        global_min, global_max = _global_aabb(body, client)
        center = [(global_min[axis] + global_max[axis]) * 0.5 for axis in range(3)]
        extent = max(global_max[axis] - global_min[axis] for axis in range(3))
        overview_distance = max(0.6, extent * 1.8)
        close_distance = max(0.22, min(0.65, extent * 0.45))
        views = [
            _render_view(center, overview_distance, 45, -22, "overview", client),
            _render_view(list(world_point), close_distance, 45, -18, "contact close-up A", client),
            _render_view(list(world_point), close_distance, 135, -18, "contact close-up B", client),
            _render_view(list(world_point), close_distance, 45, -70, "contact top view", client),
        ]
        mosaic = Image.new("RGB", (1280, 960), (20, 24, 31))
        for index, image in enumerate(views):
            mosaic.paste(image, ((index % 2) * 640, (index // 2) * 480))
        mosaic.save(out / "diagnostic.png")

        result = {
            "schema_version": 1,
            "urdf": str(urdf),
            "contact_link": args.contact_link,
            "declared_point_link_m": local_point,
            "declared_rotation_link_xyzw": local_rotation,
            "contact_frame_source": contact_frame["source"],
            "contact_frame_surface_normal_link": contact_frame["surface_normal_link"],
            "contact_frame_principal_tangent_link": contact_frame[
                "principal_tangent_link"
            ],
            "point_world_m": list(world_point),
            "surface_outward_axis_link": [0.0, 0.0, 1.0],
            "surface_outward_axis_world": outward_world,
            "panda_grasptarget_forward_world": [-value for value in outward_world],
            "robot_approach_direction_world": [-value for value in outward_world],
            "grasp_depth_m": grasp_depth_adjustment,
            "effective_robot_contact_offset_m": grasp_depth,
            "centered_grasp_zero_baseline_m": ph.PANDA_CENTERED_GRASP_BASELINE_M,
            "precontact_offset_m": precontact_offset,
            "grasp_target_world_m": grasp_target_world,
            "precontact_world_m": precontact_world,
            "requested_object_base_translation_m": requested_base,
            "effective_object_base_translation_m": effective_base,
            "target_link_aabb_world": [
                list(p.getAABB(body, target_link, physicsClientId=client)[0]),
                list(p.getAABB(body, target_link, physicsClientId=client)[1]),
            ],
            "object_global_collision_aabb_world": [global_min, global_max],
            "target_link_surface": target,
            "on_declared_link_surface": on_target,
            "surface_tolerance_m": float(args.surface_tolerance_m),
            "suggested_correction_link_m": correction_local,
            "nearest_object_link": nearest_name,
            "nearest_object_link_surface": nearest,
            "nearest_link_matches_declared": nearest_name == args.contact_link,
            "all_link_distances": distances,
            "visual_legend": {
                "yellow": "declared contact link",
                "green": "candidate point within tolerance",
                "red": "candidate point off surface",
                "blue": "nearest point on declared link surface",
                "cyan": "EEF grasp target after outward grasp-depth offset",
                "magenta": "free-space samples along outward +Z",
            },
        }
        (out / "diagnostic.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        p.disconnect(client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--contact-link", required=True)
    parser.add_argument("--point-link-m", type=float, nargs=3, required=True)
    parser.add_argument(
        "--grasp-depth-m",
        type=float,
        default=0.0,
        help="Signed adjustment around Panda's -0.015 m centered-grasp baseline.",
    )
    parser.add_argument("--precontact-offset-m", type=float, default=0.08)
    parser.add_argument("--object-base-translation-m", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--object-base-rotation-xyzw", type=float, nargs=4, default=[0.0, 0.0, 0.0, 1.0])
    parser.add_argument("--joint", action="append", default=[], help="Override an initial joint as name=value")
    parser.add_argument("--auto-ground", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--support-top-z-m", type=float, default=-0.001)
    parser.add_argument("--ground-clearance-m", type=float, default=0.002)
    parser.add_argument("--probe-radius-m", type=float, default=0.0005)
    parser.add_argument("--surface-tolerance-m", type=float, default=0.002)
    parser.add_argument("--search-distance-m", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = inspect(args)
        print(
            json.dumps(
                {
                    "on_declared_link_surface": result["on_declared_link_surface"],
                    "nearest_object_link": result["nearest_object_link"],
                    "nearest_link_matches_declared": result["nearest_link_matches_declared"],
                    "suggested_correction_link_m": result["suggested_correction_link_m"],
                    "diagnostic_json": str(args.out.expanduser().resolve() / "diagnostic.json"),
                    "diagnostic_png": str(args.out.expanduser().resolve() / "diagnostic.png"),
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"Contact-pose inspection failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
