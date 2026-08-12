#!/usr/bin/env python3
"""Solve where to put the object and the robot so the whole plan is reachable.

This runs after a candidate contact surface has been declared. It fixes a stable,
link-centered robot stance before contact-pose or controller tuning can be blamed
for reach failures. An arm reaching around a moving link or standing inside its
swept volume produces IK residuals and body-vs-object overlap that are placement
failures.

The search is asset-agnostic.  It reads the driver joint, its path, and the
contacted link from the ArtiMo plan and the execution stages, then scores candidate
(object pose, robot base pose) pairs on whether *every* sample of *every* stage's
driver path admits a collision-free IK solution.  Nothing here knows an asset name;
bounds and seeds are inputs and every rejected candidate is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pybullet as p

APP_ROOT = Path(__file__).resolve().parent
REPO = APP_ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT))

import run_artimo_physics as ph  # noqa: E402
from artimo_ik import BulletIK, set_fingers, set_robot_arm  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_orientation_gate(
    config: dict[str, Any], template_path: Path | None, template: dict[str, Any]
) -> dict[str, Any] | None:
    required_stage_ids = {
        str(stage["id"]) for stage in template.get("stages", [])
    }
    if not required_stage_ids:
        return None
    gate_value = config.get("orientation_gate_path")
    if not isinstance(gate_value, str) or not gate_value:
        raise ValueError(
            "Placement of robot-contact stages requires orientation_gate_path; "
            "render separate views and hard-filter visual-invalid rolls first"
        )
    if template_path is None:
        raise ValueError(
            "A visually gated placement must use execution_template_path so its "
            "exact selected bytes can be verified"
        )
    gate_path = ph._resolve(gate_value)
    gate = ph._read_json(gate_path)
    if int(gate.get("schema_version", 0)) != 1:
        raise ValueError("Orientation gate must use schema_version 1")
    if gate.get("policy") != "visual_invalid_candidates_hard_excluded_before_planning":
        raise ValueError("Orientation gate does not enforce hard visual exclusion")
    if gate.get("execution_sha256") != _sha256(template_path):
        raise ValueError(
            "Orientation gate execution hash does not match execution_template_path"
        )
    gated_ids = [str(item.get("stage_id")) for item in gate.get("stages", [])]
    if len(gated_ids) != len(set(gated_ids)):
        raise ValueError("Orientation gate contains duplicate stage ids")
    missing = sorted(required_stage_ids - set(gated_ids))
    if missing:
        raise ValueError(f"Orientation gate does not cover grasp stages {missing}")
    return {**gate, "path": str(gate_path), "sha256": _sha256(gate_path)}


def _yaw_quat(yaw_deg: float) -> list[float]:
    half = math.radians(yaw_deg) / 2.0
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def _tilted_approach(
    tilt_deg: float, spin_deg: float, roll_deg: float = 0.0
) -> list[float]:
    """Quaternion whose local +Z is the approach axis, tilted off the link's +Z.

    ``tilt_deg`` leans the axis away from the link frame's +Z; ``spin_deg`` rotates
    which way it leans.  Sweeping both is what lets the wrist find a lean that
    keeps the bulky hand out of the object while the fingertip stays on target.
    """
    tilt = math.radians(tilt_deg)
    spin = math.radians(spin_deg)
    axis = np.array([
        math.sin(tilt) * math.cos(spin),
        math.sin(tilt) * math.sin(spin),
        math.cos(tilt),
    ])
    reference = np.array([1.0, 0.0, 0.0])
    x = reference - axis * float(np.dot(reference, axis))
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - axis * axis[1]
    x /= np.linalg.norm(x)
    y = np.cross(axis, x)
    # Roll changes only the gripper tangent axes; the local +Z surface normal
    # and therefore the contact-facing robot placement remain byte-identical.
    roll = math.radians(roll_deg)
    original_x = x.copy()
    original_y = y.copy()
    x = math.cos(roll) * original_x + math.sin(roll) * original_y
    y = -math.sin(roll) * original_x + math.cos(roll) * original_y
    m = np.column_stack([x, y, axis])
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
        q = [0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[j] = (m[j, i] + m[i, j]) / s
        q[k] = (m[k, i] + m[i, k]) / s
        w = (m[k, j] - m[j, k]) / s
    v = np.asarray([*q, w], dtype=np.float64)
    return (v / np.linalg.norm(v)).tolist()


def _driver_path(start: float, target: float, samples: int) -> np.ndarray:
    u = np.linspace(0.0, 1.0, samples)
    return start + (target - start) * (3.0 * u * u - 2.0 * u * u * u)


def _contact_facing_base_pose(
    link_center_world_m: Iterable[float],
    outward_axis_world: Iterable[float],
    distance_m: float,
    base_z_m: float,
    yaw_offset_deg: float = 0.0,
    lateral_offset_m: float = 0.0,
) -> tuple[list[float], float]:
    """Put the robot on the target link's outward ray, facing its center.

    Only the horizontal projection is used because the Panda base stays upright.
    The target link's collision AABB supplies the placement center, while the
    declared contact frame supplies the outward direction.  Keeping these roles
    separate prevents an off-center handle contact from shifting the entire robot
    stance before reachability search has even begun.
    """
    point = np.asarray(list(link_center_world_m), dtype=np.float64)
    outward = np.asarray(list(outward_axis_world), dtype=np.float64)
    if point.shape != (3,) or outward.shape != (3,):
        raise ValueError("Link center and outward axis must both contain three values")
    if float(distance_m) <= 0.0:
        raise ValueError("contact_facing_distance_m values must be positive")
    horizontal = outward[:2]
    norm = float(np.linalg.norm(horizontal))
    if norm < 1e-6:
        raise ValueError(
            "Contact outward normal is vertical; a horizontal contact-facing robot base is undefined"
        )
    horizontal /= norm
    # After fixing the link-centered facing direction, a single tangent offset is the
    # minimal extra degree of freedom needed to cover a long articulated arc
    # (for example the far end of an opening door).  It does not change the
    # object pose or invent independent world x/y tuning.
    tangent = np.asarray([-horizontal[1], horizontal[0]], dtype=np.float64)
    base_xy = (
        point[:2]
        + horizontal * float(distance_m)
        + tangent * float(lateral_offset_m)
    )
    # Panda base +X is the forward direction used by the placement convention.
    # Point it from the base back toward the target link center.
    facing = math.degrees(
        math.atan2(float(point[1] - base_xy[1]), float(point[0] - base_xy[0]))
    )
    yaw = facing + float(yaw_offset_deg)
    return [float(base_xy[0]), float(base_xy[1]), float(base_z_m)], yaw


def _center_first_lateral_offsets(values: Iterable[float]) -> list[float]:
    """Return the exact centerline first, then declared lateral refinements."""
    ordered: list[float] = [0.0]
    for raw in values:
        value = float(raw)
        if abs(value) < 1e-12:
            continue
        if not any(abs(value - existing) < 1e-12 for existing in ordered):
            ordered.append(value)
    return ordered


def _contact_frame_world(
    simulation_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
    stage: dict[str, Any],
) -> dict[str, list[float]]:
    """Measure the selected initial contact frame after deterministic grounding."""
    grounding = ph._ground_execution_scene(
        simulation_urdf, robot_urdf, json.loads(json.dumps(execution)), initial
    )
    grounded = grounding["execution"]
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet contact-frame client")
    try:
        body = p.loadURDF(
            str(simulation_urdf),
            basePosition=grounded["scene"]["object_base_translation_m"],
            baseOrientation=ph._quat(grounded["scene"]["object_base_rotation_xyzw"]),
            useFixedBase=True,
            physicsClientId=client,
        )
        joints, links = ph._maps(body, client)
        for name, value in initial.items():
            if name in joints:
                p.resetJointState(body, joints[name], float(value), physicsClientId=client)
        link = links[stage["contact_link"]]
        link_position, link_rotation = ph.link_world_pose(body, link, client)
        contact_position, contact_rotation = p.multiplyTransforms(
            link_position,
            link_rotation,
            stage["contact_pose_link"]["translation_m"],
            ph._quat(stage["contact_pose_link"]["rotation_xyzw"]),
            physicsClientId=client,
        )
        outward = p.rotateVector(contact_rotation, [0.0, 0.0, 1.0])
        aabb_min, aabb_max = p.getAABB(body, link, physicsClientId=client)
        link_center = (
            0.5
            * (
                np.asarray(aabb_min, dtype=np.float64)
                + np.asarray(aabb_max, dtype=np.float64)
            )
        )
        return {
            "contact_world_m": [float(v) for v in contact_position],
            "link_center_world_m": [float(v) for v in link_center],
            "outward_axis_world": [float(v) for v in outward],
        }
    finally:
        p.disconnect(client)


def discover_contact_points(
    simulation_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
    stage: dict[str, Any],
    step_m: float,
    minimum_neighbour_clearance_m: float,
    limit: int,
    robot_position: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Return link-frame points that lie on the contacted link's actual surface.

    A hand-guessed contact point is the single most common cause of a "placement"
    failure: if the point floats inside the link, IK happily drives the gripper
    into the object and every candidate reports the same constant penetration.

    The candidates come from the link's own collision mesh vertices, decimated on
    a grid of ``step_m``.  Sampling geometry that already exists is far cheaper
    than sweeping a volume, and every returned point is on the surface by
    construction.  Points near another link are dropped so the fingers do not also
    strike a neighbouring part such as a door leaf or a lid.
    """
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet probe client")
    try:
        # Load the object at the placement being evaluated, grounded the same way the
        # harness will.  Probing it at the URDF origin instead reports points below
        # the floor and makes the reachability test meaningless.
        grounding = ph._ground_execution_scene(
            simulation_urdf, simulation_urdf, json.loads(json.dumps(execution)), initial
        )
        grounded = grounding.pop("execution")
        body = p.loadURDF(
            str(simulation_urdf),
            basePosition=grounded["scene"]["object_base_translation_m"],
            baseOrientation=ph._quat(grounded["scene"]["object_base_rotation_xyzw"]),
            useFixedBase=True,
            physicsClientId=client,
        )
        joints, links = ph._maps(body, client)
        for name, value in initial.items():
            if name in joints:
                p.resetJointState(body, joints[name], float(value), physicsClientId=client)
        target_link = links[stage["contact_link"]]
        others = [index for index in links.values() if index != target_link]
        other_names = {index: name for name, index in links.items() if index != target_link}
        robot_position = list(robot_position or grounded["robot"]["base_translation_m"])

        # Read the collision meshes and their origins straight from the URDF.
        # getCollisionShapeData reports frames relative to the *inertial* frame,
        # which is offset from the link frame the contact pose is expressed in.
        vertices: list[np.ndarray] = []
        tree = ET.parse(simulation_urdf)
        for link in tree.getroot().iter("link"):
            if link.attrib.get("name") != stage["contact_link"]:
                continue
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None or not mesh.attrib.get("filename"):
                    continue
                reference = mesh.attrib["filename"]
                mesh_path = Path(reference)
                if not mesh_path.is_absolute():
                    mesh_path = simulation_urdf.parent / reference
                if not mesh_path.is_file():
                    continue
                origin_element = collision.find("origin")
                offset = np.zeros(3)
                if origin_element is not None and origin_element.attrib.get("xyz"):
                    offset = np.asarray([float(v) for v in origin_element.attrib["xyz"].split()])
                for line in mesh_path.read_text(errors="replace").splitlines():
                    if line.startswith("v "):
                        vertices.append(np.asarray([float(v) for v in line.split()[1:4]]) + offset)
        if not vertices:
            return []
        array = np.asarray(vertices)
        # Decimate to one representative per grid cell to bound the candidate count.
        keyed: dict[tuple[int, int, int], np.ndarray] = {}
        for point in array:
            cell = tuple(int(math.floor(point[a] / step_m)) for a in range(3))
            keyed.setdefault(cell, point)

        origin, rotation = ph.link_world_pose(body, target_link, client)
        radius = 0.0005
        shape_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client)
        probe = p.createMultiBody(
            baseMass=0.0, baseCollisionShapeIndex=shape_id, basePosition=[0.0, 0.0, 0.0],
            physicsClientId=client,
        )
        found: list[dict[str, Any]] = []
        for point in keyed.values():
            world, _ = p.multiplyTransforms(origin, rotation, point.tolist(), [0.0, 0.0, 0.0, 1.0])
            p.resetBasePositionAndOrientation(probe, world, [0.0, 0.0, 0.0, 1.0], physicsClientId=client)
            clearance = 1e9
            for other in others:
                near = p.getClosestPoints(probe, body, minimum_neighbour_clearance_m, linkIndexB=other,
                                          physicsClientId=client)
                if near:
                    clearance = min(clearance, min(float(h[8]) for h in near) + radius)
            if clearance < minimum_neighbour_clearance_m:
                continue
            # A point on the far face is unreachable no matter how the arm is
            # placed: the object's own body is in the way.  Cast from the robot's
            # base at the point's own height and require the target link to be the
            # first thing hit.
            ray_from = [float(robot_position[0]), float(robot_position[1]), float(world[2])]
            hits = p.rayTest(ray_from, list(world), physicsClientId=client)
            if hits and hits[0][0] == body and int(hits[0][1]) != target_link:
                continue
            exposure = float(np.linalg.norm(np.asarray(world) - np.asarray(ray_from)))
            found.append({
                "point_link_m": [round(float(v), 4) for v in point],
                "point_world_m": [round(float(v), 4) for v in world],
                "distance_from_robot_m": round(exposure, 4),
                "neighbour_clearance_m": None if clearance > 1e8 else round(clearance, 4),
                "source": "collision_mesh_vertex",
            })
        # Prefer points nearest the robot -- those are on the face it can actually
        # reach -- then spread the candidates out so they are not all clustered on
        # one corner of the part.
        found.sort(key=lambda item: (item["distance_from_robot_m"],
                                     -(item["neighbour_clearance_m"] or 1.0)))
        chosen: list[dict[str, Any]] = []
        for item in found:
            location = np.asarray(item["point_link_m"])
            if all(float(np.linalg.norm(location - np.asarray(k["point_link_m"]))) > step_m * 2 for k in chosen):
                chosen.append(item)
            if len(chosen) >= limit:
                break
        return chosen
    finally:
        p.disconnect(client)


def _score_candidate(
    simulation_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
    samples: int,
    allowed_penetration_m: float,
    maximum_grasp_gap_m: float = 0.006,
) -> dict[str, Any]:
    """Return reachability and clearance evidence for one full placement.

    Every stage is swept over its whole driver path.  A candidate only counts as
    feasible if the arm can hold the contact pose at each sample without any body
    link other than the nominated contact links overlapping the object.
    """
    grounding = ph._ground_execution_scene(simulation_urdf, robot_urdf, execution, initial)
    grounded = grounding.pop("execution")
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet placement client")
    try:
        object_body, robot_body, robot_support_body = ph._load_scene(
            simulation_urdf, robot_urdf, grounded, client, False
        )
        object_joints, object_links = ph._maps(object_body, client)
        robot_joints, robot_links = ph._maps(robot_body, client)
        robot_link_names = {index: name for name, index in robot_links.items()}
        object_link_names = {index: name for name, index in object_links.items()}
        for name, value in initial.items():
            if name in object_joints:
                p.resetJointState(object_body, object_joints[name], value, physicsClientId=client)
        spec = grounded["robot"]
        arm = [robot_joints[name] for name in spec["arm_joint_names"]]
        fingers = [robot_joints[name] for name in spec["finger_joint_names"]]
        eef = robot_links[spec["end_effector_link"]]
        home = np.asarray(spec["home_joint_positions"], dtype=np.float64)
        ik_config = grounded.get("ik", {})

        stage_reports: list[dict[str, Any]] = []
        feasible = True
        current = dict(initial)
        reference = home.copy()
        for stage_index, stage in enumerate(grounded["stages"]):
            continues_from_previous = (
                stage_index > 0
                and ph._same_contact_sequence(grounded["stages"][stage_index - 1], stage)
            )
            for name, value in current.items():
                if name in object_joints:
                    p.resetJointState(
                        object_body, object_joints[name], float(value), physicsClientId=client
                    )
            if not continues_from_previous:
                reference = home.copy()
            opening = float(stage["finger_opening_m"])
            set_fingers(robot_body, fingers, opening, client)
            solver = BulletIK(
                robot_body, arm, eef, fingers, opening,
                {
                    "random_seed": int(grounded["seeds"].get("ik", 0)) + stage_index,
                    "random_restarts": int(ik_config.get("random_restarts", 96)),
                    "max_iterations": int(ik_config.get("max_iterations", 2000)),
                    "position_tolerance_m": float(ik_config.get("position_tolerance_m", 0.001)),
                    "orientation_tolerance_deg": float(ik_config.get("orientation_tolerance_deg", 1.0)),
                    "max_joint_step_rad": float(ik_config.get("max_joint_step_rad", 1.2)),
                },
                client,
            )
            driver = object_joints[stage["driver_joint"]]
            contact_link = object_links[stage["contact_link"]]
            allowed_by_name = {
                name: robot_links[name] for name in stage["allowed_robot_contact_links"]
            }
            allowed = set(allowed_by_name.values())
            start = float(
                current.get(
                    stage["driver_joint"],
                    p.getJointState(object_body, driver, physicsClientId=client)[0],
                )
            )
            target = float(
                stage.get("command_joint_position", stage["target_joint_position"])
            )
            path = _driver_path(
                start,
                target,
                samples,
            )
            solved = 0
            worst_error = 0.0
            deepest = 0.0
            grasp_reach_limit = 0.05
            worst_gap_by_link = {
                name: -float("inf") for name in allowed_by_name
            }
            offenders: dict[str, float] = {}
            required_clearance = float(stage.get("minimum_swept_clearance_m", 0.0))
            forbidden = {
                name: object_links[name]
                for name in stage.get("forbidden_contact_links", [])
            }
            unknown_forbidden = set(stage.get("forbidden_contact_links", [])) - set(object_links)
            if unknown_forbidden:
                raise KeyError(
                    f"Stage {stage['id']} names unknown forbidden_contact_links: "
                    f"{sorted(unknown_forbidden)}"
                )
            minimum_forbidden_clearance: float | None = None
            clearance_pairs: dict[str, float] = {}
            # Match the physics entry point. Grasp depth remains at the final
            # manipulation pose, so an excessive value appears here as a real
            # per-finger gap instead of being confused with precontact clearance.
            grasp_depth = float(stage.get("grasp_depth_m", 0.0))
            for value in path:
                p.resetJointState(object_body, driver, float(value), physicsClientId=client)
                position, rotation = ph._target_pose(
                    object_body,
                    contact_link,
                    stage["contact_pose_link"],
                    grasp_depth,
                    client,
                    stage.get("robot_tool_contact_offset_eef_m"),
                )
                answer = solver.solve(position, rotation, reference, enforce_step=False)
                if not answer["success"]:
                    break
                reference = np.asarray(answer["q"], dtype=np.float64)
                worst_error = max(worst_error, float(answer["position_error_m"]))
                set_robot_arm(robot_body, arm, reference, client)
                p.performCollisionDetection(physicsClientId=client)
                # The gripper must actually reach the driven link.  Without this
                # test a candidate that never touches anything scores perfectly on
                # collision and wins, which is how an arm that plainly misses the
                # handle can look like the best placement.
                for link_name, link_index in allowed_by_name.items():
                    near = p.getClosestPoints(
                        robot_body, object_body, grasp_reach_limit,
                        linkIndexA=link_index, linkIndexB=contact_link,
                        physicsClientId=client,
                    )
                    gap = (
                        min(float(h[8]) for h in near)
                        if near
                        else grasp_reach_limit
                    )
                    worst_gap_by_link[link_name] = max(
                        worst_gap_by_link[link_name], gap
                    )
                clearance_query = max(required_clearance, 0.02)
                for object_name, object_index in forbidden.items():
                    for point in p.getClosestPoints(
                        robot_body,
                        object_body,
                        clearance_query,
                        linkIndexB=object_index,
                        physicsClientId=client,
                    ):
                        distance = float(point[8])
                        if (
                            minimum_forbidden_clearance is None
                            or distance < minimum_forbidden_clearance
                        ):
                            minimum_forbidden_clearance = distance
                        key = (
                            f"{robot_link_names.get(int(point[3]), point[3])}|"
                            f"{object_name}"
                        )
                        clearance_pairs[key] = min(
                            clearance_pairs.get(key, float("inf")), distance
                        )
                    if robot_support_body is not None:
                        for point in p.getClosestPoints(
                            robot_support_body,
                            object_body,
                            clearance_query,
                            linkIndexB=object_index,
                            physicsClientId=client,
                        ):
                            distance = float(point[8])
                            if (
                                minimum_forbidden_clearance is None
                                or distance < minimum_forbidden_clearance
                            ):
                                minimum_forbidden_clearance = distance
                            key = f"robot_support|{object_name}"
                            clearance_pairs[key] = min(
                                clearance_pairs.get(key, float("inf")), distance
                            )
                for point in p.getClosestPoints(robot_body, object_body, 0.0, physicsClientId=client):
                    robot_link, object_link = int(point[3]), int(point[4])
                    depth = float(point[8])
                    if robot_link in allowed and object_link == contact_link:
                        continue
                    if depth < deepest:
                        deepest = depth
                    key = (
                        f"{robot_link_names.get(robot_link, robot_link)}|"
                        f"{object_link_names.get(object_link, object_link)}"
                    )
                    offenders[key] = min(offenders.get(key, 0.0), depth)
                if robot_support_body is not None:
                    for point in p.getClosestPoints(
                        robot_support_body,
                        object_body,
                        0.0,
                        physicsClientId=client,
                    ):
                        object_link = int(point[4])
                        depth = float(point[8])
                        deepest = min(deepest, depth)
                        key = (
                            f"robot_support|"
                            f"{object_link_names.get(object_link, object_link)}"
                        )
                        offenders[key] = min(offenders.get(key, 0.0), depth)
                solved += 1
            p.resetJointState(
                object_body, driver, start, physicsClientId=client
            )
            set_robot_arm(robot_body, arm, home, client)
            normalized_gaps = {
                name: (
                    grasp_reach_limit
                    if not math.isfinite(gap)
                    else float(gap)
                )
                for name, gap in worst_gap_by_link.items()
            }
            near_link_count = sum(
                gap <= float(maximum_grasp_gap_m)
                for gap in normalized_gaps.values()
            )
            required_near_links = min(
                2 if stage["interaction"] == "explicit_ideal_feasibility" else 1,
                len(normalized_gaps),
            )
            # An open-then-close grasp needs opposing gripper geometry close to the
            # target throughout the path. One nearby finger and one finger
            # centimetres away is a miss, even though the EEF IK is perfect.
            grasped = (
                solved == samples
                and required_near_links > 0
                and near_link_count >= required_near_links
            )
            clearance_passed = (
                minimum_forbidden_clearance is None
                or minimum_forbidden_clearance >= required_clearance
            )
            stage_ok = (
                solved == samples
                and deepest >= -abs(allowed_penetration_m)
                and grasped
                and clearance_passed
            )
            feasible = feasible and stage_ok
            stage_reports.append({
                "stage_id": stage["id"],
                "samples_solved": solved,
                "samples_required": samples,
                "maximum_ik_position_error_m": round(worst_error, 6),
                "deepest_body_penetration_m": round(deepest, 5),
                "target_link_gap_by_allowed_robot_link_m": {
                    name: round(gap, 5) for name, gap in sorted(normalized_gaps.items())
                },
                "maximum_target_link_gap_m": round(
                    max(normalized_gaps.values(), default=grasp_reach_limit), 5
                ),
                "maximum_allowed_target_gap_m": float(maximum_grasp_gap_m),
                "required_near_contact_links": required_near_links,
                "near_contact_link_count": near_link_count,
                "target_actually_gripped": bool(grasped),
                "required_forbidden_clearance_m": required_clearance,
                "minimum_forbidden_clearance_m": (
                    None
                    if minimum_forbidden_clearance is None
                    else round(minimum_forbidden_clearance, 5)
                ),
                "forbidden_clearance_passed": bool(clearance_passed),
                "tightest_forbidden_pairs": dict(
                    sorted(clearance_pairs.items(), key=lambda item: item[1])[:5]
                ),
                "worst_offenders": dict(sorted(offenders.items(), key=lambda kv: kv[1])[:5]),
                "feasible": bool(stage_ok),
            })
            current[stage["driver_joint"]] = target
            next_stage = (
                grounded["stages"][stage_index + 1]
                if stage_index + 1 < len(grounded["stages"])
                else None
            )
            if next_stage is None or not ph._same_contact_sequence(stage, next_stage):
                reference = home.copy()
        # A manipulation-only placement is not sufficient: the same generic
        # planner used by rollout must also clear home->first approach,
        # approach, manipulation, release/retreat, every direct inter-stage
        # transit, and the final return home with the appropriate finger width.
        # Run this dense confirmation only for otherwise feasible candidates so
        # placement search remains bounded.
        if feasible:
            try:
                # World-frame release waypoints are solved only after the robot
                # base is fixed.  A waypoint produced for an earlier stance is
                # stale by construction and must not reject a new placement.
                # Keep the declared clearance requirement in execution data,
                # but remove stale waypoints from the emitted candidate and let
                # the dedicated release solver regenerate them next.
                placement_execution = json.loads(json.dumps(grounded))
                for planned_stage in placement_execution["stages"]:
                    planned_stage.pop("release_retreat_waypoints_world", None)
                dense_plans = ph._plan_stages(
                    simulation_urdf,
                    robot_urdf,
                    placement_execution,
                    initial,
                    validate_release_clearance=False,
                )
            except Exception as exc:
                feasible = False
                if stage_reports:
                    stage_reports[-1]["full_path_rejected"] = str(exc)[:2000]
                    stage_reports[-1]["feasible"] = False
            else:
                by_id = {item.stage["id"]: item for item in dense_plans}
                for stage_report in stage_reports:
                    dense = by_id[stage_report["stage_id"]]
                    stage_report["dense_full_path_minimum_clearance_m"] = (
                        None
                        if dense.minimum_swept_clearance_m is None
                        else round(float(dense.minimum_swept_clearance_m), 6)
                    )
                    stage_report["dense_full_path_tightest_samples"] = (
                        dense.swept_clearance_violations[:5]
                    )
                    required_dense_clearance = float(
                        dense.stage.get("minimum_swept_clearance_m", 0.0)
                    )
                    dense_clearance_passed = (
                        dense.minimum_swept_clearance_m is None
                        or dense.minimum_swept_clearance_m
                        >= required_dense_clearance
                    )
                    dense_ik_passed = (
                        not dense.debug_truncated
                        and dense.maximum_position_error_m <= 0.004
                        and math.degrees(dense.maximum_orientation_error_rad) <= 2.0
                        and dense.minimum_joint_limit_margin_rad > 1e-4
                        and dense.maximum_adjacent_joint_step_rad <= 0.0800001
                    )
                    stage_report["dense_full_path_clearance_passed"] = bool(
                        dense_clearance_passed
                    )
                    stage_report["dense_ik_passed"] = bool(dense_ik_passed)
                    if not dense_clearance_passed or not dense_ik_passed:
                        stage_report["feasible"] = False
                        reasons = []
                        if not dense_clearance_passed:
                            reasons.append("dense_swept_clearance")
                        if not dense_ik_passed:
                            reasons.append("dense_ik_diagnostics")
                        stage_report["full_path_rejected"] = ",".join(reasons)
                        feasible = False
        output_execution = json.loads(json.dumps(grounded))
        for output_stage in output_execution["stages"]:
            output_stage.pop("release_retreat_waypoints_world", None)
        return {
            "feasible": bool(feasible),
            "stages": stage_reports,
            "grounding": grounding,
            "execution": output_execution,
        }
    finally:
        p.disconnect(client)


def _candidate_rank(report: dict[str, Any]) -> tuple:
    """Order candidates so the most complete, least-penetrating one wins."""
    stages = report["stages"]
    solved = sum(int(s["samples_solved"]) for s in stages)
    required = sum(int(s["samples_required"]) for s in stages)
    deepest = min([float(s["deepest_body_penetration_m"]) for s in stages], default=0.0)
    error = max([float(s["maximum_ik_position_error_m"]) for s in stages], default=1.0)
    gap = max([float(s.get("maximum_target_link_gap_m", 1.0)) for s in stages], default=1.0)
    gripped = all(bool(s.get("target_actually_gripped")) for s in stages)
    # Gripping the target comes first: a candidate that touches nothing collides
    # with nothing, and would otherwise rank above one that really does the task.
    return (-gripped, -(solved == required), -solved / max(required, 1), gap, -deepest, error)


def solve(config: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    inputs = task["inputs"]
    template_value = config.get("execution_template")
    resolved_template_path: Path | None = None
    if template_value is None:
        template_path = config.get("execution_template_path")
        if not isinstance(template_path, str) or not template_path:
            raise ValueError(
                "Placement config requires execution_template or execution_template_path"
            )
        resolved_template_path = ph._resolve(template_path)
        template = ph._read_json(resolved_template_path)
    elif isinstance(template_value, dict):
        template = template_value
    else:
        raise ValueError("execution_template must be an object")
    orientation_gate = _validated_orientation_gate(
        config, resolved_template_path, template
    )
    source_urdf = ph._resolve(inputs["urdf"])
    simulation_urdf = ph.resolve_simulation_urdf(
        task, template, source_urdf
    )
    ph._require_matching_mechanism(source_urdf, simulation_urdf)
    robot_urdf = ph._resolve(inputs["robot_urdf"])
    initial = ph.task_initial_joint_values(task)
    plan = ph._read_json(ph._resolve(inputs["plan"]))

    bounds = config["bounds"]
    samples = int(config.get("path_samples", 13))
    allowed_penetration = float(config.get("allowed_body_penetration_m", 0.002))
    # How close an allowed contact link must come to the driven link for the grasp
    # to count as real rather than a near miss.
    maximum_grasp_gap = float(config.get("maximum_grasp_gap_m", 0.006))

    object_yaws = bounds.get("object_yaw_deg", [0.0])
    placement_mode = str(
        config.get(
            "placement_mode",
            "contact_facing" if "contact_facing_distance_m" in bounds else "cartesian",
        )
    )
    if placement_mode not in {"contact_facing", "cartesian"}:
        raise ValueError("placement_mode must be 'contact_facing' or 'cartesian'")
    # A pedestal height is part of placement: a contact well above or below the
    # arm's shoulder forces the forearm to reach through the object, which shows up
    # as body-vs-object overlap that no contact-pose change can fix.
    robot_z = bounds.get("robot_base_z_m", [0.0])
    orientation_bound_keys = {
        "approach_tilt_deg", "approach_spin_deg", "approach_roll_deg"
    }
    if orientation_gate is not None:
        forbidden_orientation_bounds = orientation_bound_keys & set(bounds)
        if forbidden_orientation_bounds:
            raise ValueError(
                "Visually gated contact rotations are frozen; placement bounds must "
                f"not contain {sorted(forbidden_orientation_bounds)}"
            )
        orientations: list[tuple[float, float, float] | None] = [None]
    else:
        tilts = bounds.get("approach_tilt_deg", [0.0])
        spins = bounds.get("approach_spin_deg", [0.0])
        rolls = bounds.get("approach_roll_deg", [0.0])
        orientations = list(itertools.product(tilts, spins, rolls))

    if placement_mode == "contact_facing":
        distances = bounds.get("contact_facing_distance_m")
        if not isinstance(distances, list) or not distances:
            raise ValueError(
                "contact_facing placement requires non-empty bounds.contact_facing_distance_m"
            )
        yaw_offsets = bounds.get("contact_facing_yaw_offset_deg", [0.0])
        lateral_offsets = _center_first_lateral_offsets(
            bounds.get("contact_facing_lateral_offset_m", [0.0])
        )
        forbidden_cartesian = {
            "robot_base_x_m", "robot_base_y_m", "robot_base_yaw_deg"
        } & set(bounds)
        if forbidden_cartesian:
            raise ValueError(
                "contact_facing placement derives robot x/y/yaw; remove independent bounds "
                f"{sorted(forbidden_cartesian)}"
            )
    else:
        robot_x = bounds["robot_base_x_m"]
        robot_y = bounds["robot_base_y_m"]
        robot_yaws = bounds["robot_base_yaw_deg"]

    # Contact points depend on the placement: which face of the part is reachable
    # changes as the object rotates and the robot moves.  They are therefore
    # rediscovered per placement rather than once up front.
    discovery = config.get("contact_point_discovery", {})
    discover = discovery.get("enabled", True) and len(template["stages"]) == 1
    declared_point = list(template["stages"][0]["contact_pose_link"]["translation_m"])

    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    index = -1
    if placement_mode == "contact_facing":
        placement_count = (
            len(object_yaws)
            * len(robot_z)
            * len(distances)
            * len(yaw_offsets)
            * len(lateral_offsets)
        )
    else:
        placement_count = len(object_yaws) * len(robot_x) * len(robot_y) * len(robot_z) * len(robot_yaws)
    contacts_per_placement = (
        max(1, int(discovery.get("limit", 6))) if discover else 1
    )
    total = placement_count * len(orientations) * contacts_per_placement
    all_discovered: list[dict[str, Any]] = []
    if placement_mode == "contact_facing":
        # Search the physically interpretable centerline first: all declared
        # normal distances are attempted with zero lateral offset.  Only if none
        # works do we add the bounded left/right offsets.  This prevents an
        # arbitrary off-center stance from becoming the default merely because
        # it happened to appear first in a Cartesian product.
        centered = list(
            itertools.product(object_yaws, robot_z, distances, yaw_offsets, [0.0])
        )
        refined = list(
            itertools.product(
                object_yaws,
                robot_z,
                distances,
                yaw_offsets,
                lateral_offsets[1:],
            )
        )
        coarse_placements = iter(centered + refined)
    else:
        coarse_placements = itertools.product(object_yaws, robot_z, robot_x, robot_y, robot_yaws)
    for placement_values in coarse_placements:
        if placement_mode == "contact_facing":
            oyaw, rz, distance, yaw_offset, lateral_offset = placement_values
            explicit_pose = None
        else:
            oyaw, rz, rx, ry, ryaw = placement_values
            distance = None
            yaw_offset = None
            lateral_offset = None
            explicit_pose = ([float(rx), float(ry), float(rz)], float(ryaw))
        for orientation in orientations:
            oriented = json.loads(json.dumps(template))
            oriented["scene"]["object_base_rotation_xyzw"] = _yaw_quat(oyaw)
            if orientation is None:
                tilt = spin = roll = None
            else:
                tilt, spin, roll = orientation
                for stage in oriented["stages"]:
                    stage["contact_pose_link"]["rotation_xyzw"] = _tilted_approach(
                        tilt, spin, roll
                    )

            if placement_mode == "contact_facing":
                frame = _contact_frame_world(
                    simulation_urdf, robot_urdf, oriented, initial, oriented["stages"][0]
                )
                robot_base, ryaw = _contact_facing_base_pose(
                    frame["link_center_world_m"], frame["outward_axis_world"],
                    float(distance), float(rz), float(yaw_offset),
                    float(lateral_offset),
                )
                rx, ry, _ = robot_base
            else:
                frame = None
                robot_base, ryaw = explicit_pose
                rx, ry, _ = robot_base

            placed = json.loads(json.dumps(oriented))
            placed["robot"]["base_translation_m"] = robot_base
            placed["robot"]["base_rotation_xyzw"] = _yaw_quat(ryaw)
            if float(rz) > 0.0:
                placed["robot"]["base_support_height_m"] = float(rz)
            elif "base_support_height_m" in placed["robot"]:
                placed["robot"]["base_support_height_m"] = 0.0

            if discover:
                try:
                    points_here = discover_contact_points(
                        simulation_urdf, placed, initial, placed["stages"][0],
                        float(discovery.get("step_m", 0.01)),
                        float(discovery.get("minimum_neighbour_clearance_m", 0.004)),
                        int(discovery.get("limit", 6)),
                        robot_position=robot_base,
                    )
                except Exception:
                    points_here = []
                contact_points = [item["point_link_m"] for item in points_here] or [declared_point]
                all_discovered.extend(
                    {**item, "for_object_yaw_deg": oyaw, "for_robot_base_m": robot_base}
                    for item in points_here
                )
            else:
                contact_points = [declared_point]

            for point in contact_points:
                index += 1
                candidate = json.loads(json.dumps(placed))
                candidate["stages"][0]["contact_pose_link"]["translation_m"] = [float(v) for v in point]
                try:
                    ph._validate_execution_against_plan(plan, candidate)
                    report = _score_candidate(
                        simulation_urdf, robot_urdf, candidate, initial, samples,
                        allowed_penetration, maximum_grasp_gap,
                    )
                except Exception as exc:
                    attempts.append({
                        "index": index,
                        "object_yaw_deg": oyaw, "robot_base_m": [rx, ry, rz], "robot_yaw_deg": ryaw,
                        "placement_mode": placement_mode,
                        "contact_facing_distance_m": distance,
                        "contact_facing_lateral_offset_m": lateral_offset,
                        "approach_tilt_deg": tilt, "approach_spin_deg": spin,
                        "approach_roll_deg": roll,
                        "contact_point_link_m": point, "rejected": str(exc)[:200],
                    })
                    continue
                record = {
                    "index": index,
                    "object_yaw_deg": oyaw, "robot_base_m": [rx, ry, rz], "robot_yaw_deg": ryaw,
                    "placement_mode": placement_mode,
                    "contact_facing_distance_m": distance,
                    "contact_facing_lateral_offset_m": lateral_offset,
                    "contact_frame_world": frame,
                    "approach_tilt_deg": tilt, "approach_spin_deg": spin,
                    "approach_roll_deg": roll,
                    "contact_point_link_m": point,
                    "feasible": report["feasible"], "stages": report["stages"],
                }
                attempts.append(record)
                scored = {**record, "execution": report["execution"], "grounding": report["grounding"]}
                if best is None or _candidate_rank(scored) < _candidate_rank(best):
                    best = scored
                print(
                    f"[{index + 1}/~{total}] obj_yaw={oyaw:3.0f} base=({rx:+.2f},{ry:+.2f},{rz:.2f}) "
                    f"yaw={ryaw:+.0f} "
                    f"orientation={'frozen-visual-gate' if orientation is None else f'tilt={tilt:.0f} spin={spin:.0f} roll={roll:.0f}'} "
                    f"pt={point} -> "
                    f"solved={[s['samples_solved'] for s in report['stages']]} "
                    f"pen={[s['deepest_body_penetration_m'] for s in report['stages']]} "
                    f"{'FEASIBLE' if report['feasible'] else ''}",
                    flush=True,
                )
                if report["feasible"] and config.get("stop_on_first_feasible", True):
                    break
            if best is not None and best["feasible"] and config.get("stop_on_first_feasible", True):
                break
        if best is not None and best["feasible"] and config.get("stop_on_first_feasible", True):
            break
    discovered = all_discovered

    if best is None:
        raise RuntimeError("No placement candidate could be evaluated; check bounds and template")
    return {
        "schema_version": 1,
        "feasible": bool(best["feasible"]),
        "chosen": {
            "object_yaw_deg": best["object_yaw_deg"],
            "placement_mode": best["placement_mode"],
            "contact_facing_distance_m": best.get("contact_facing_distance_m"),
            "contact_facing_lateral_offset_m": best.get(
                "contact_facing_lateral_offset_m"
            ),
            "contact_frame_world": best.get("contact_frame_world"),
            "robot_base_m": best["robot_base_m"],
            "robot_yaw_deg": best["robot_yaw_deg"],
            "approach_tilt_deg": best["approach_tilt_deg"],
            "approach_spin_deg": best["approach_spin_deg"],
            "approach_roll_deg": best["approach_roll_deg"],
            "contact_point_link_m": best["contact_point_link_m"],
            "stages": best["stages"],
        },
        # Never emit a runnable execution from a merely "best available"
        # placement.  Every coarse sample and the dense 65-point planner must
        # pass first; otherwise only rejection diagnostics are returned.
        "execution": best["execution"] if best["feasible"] else None,
        "search": {
            "orientation_gate": orientation_gate,
            "path_samples": samples,
            "allowed_body_penetration_m": allowed_penetration,
            "maximum_grasp_gap_m": maximum_grasp_gap,
            "candidates_evaluated": len(attempts),
            "discovered_contact_points": discovered,
            "bounds": bounds,
            "attempts": attempts,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Placement search bounds + execution template")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        task = ph._read_json(args.task_spec.expanduser().resolve())
        config = ph._read_json(args.config.expanduser().resolve())
        answer = solve(config, task)
        out = args.out.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "placement.json").write_text(
            json.dumps(answer, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Only a genuinely feasible placement becomes runnable execution data.
        # An infeasible search keeps placement.json diagnostics but cannot leak
        # a partial candidate into the physics/video pipeline.
        if answer["execution"] is not None:
            (out / "execution.json").write_text(
                json.dumps(answer["execution"], indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps({
            "feasible": answer["feasible"],
            "chosen": {k: v for k, v in answer["chosen"].items() if k != "stages"},
            "execution": (
                str(out / "execution.json")
                if answer["execution"] is not None
                else None
            ),
            "placement": str(out / "placement.json"),
        }, indent=2, ensure_ascii=False, sort_keys=True))
        return 0 if answer["feasible"] else 2
    except Exception as exc:
        print(f"Placement solve failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
