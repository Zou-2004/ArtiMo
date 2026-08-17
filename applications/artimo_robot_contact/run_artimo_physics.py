#!/usr/bin/env python3
"""Execute one declarative, asset-agnostic ArtiMo robot-contact plan.

The module contains no object registry or calibrated task values.  It reads all
geometry, robot, contact, causal, camera, and seed values from an execution JSON
conforming to applications/artimo_robot_contact/schemas/artimo_robot_execution.schema.json.  Panda arm
actuation and grasp/push interaction policy are benchmark invariants, not
per-asset tuning knobs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pkgutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

import artimo_plan
import artimo_video
from artimo_curobo import create_curobo_backend
from artimo_ik import BulletIK, link_world_pose, quat_angle_rad, set_fingers, set_robot_arm


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
EXECUTION_SCHEMA = APP_ROOT / "schemas" / "artimo_robot_execution.schema.json"
DT = 1.0 / 240.0
# Panda ``panda_grasptarget`` is located at the fingertips (0.105 m from the
# hand), while the useful opposed-contact region lies behind that frame. Make
# execution depth zero a conservative 15 mm inward inset: deep enough to put a
# compact control feature between the fingers without needlessly sweeping the
# palm into nearby structure. This is a robot/gripper-frame invariant, not a
# task/asset calibration. Signed execution depth remains a continuous
# adjustment around this zero point.
PANDA_CENTERED_GRASP_BASELINE_M = -0.015
# Preserve simulation-time playback while reducing expensive PyBullet CPU
# rasterization. Ten frames per second is still twice the contract's default
# 5 fps review rate; 640x480 plus the same-frame contact inset keeps the grasp
# legible while cutting TinyRenderer pixel work by more than 4x versus the old
# 20 fps 960x720 stream. NVENC still handles H.264 encoding when available.
VIDEO_FPS = 10
CAPTURE_EVERY = 24
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
# A small, generic separation from the support surface prevents numerical
# penetration while keeping the placement independent of asset identity.
GROUND_CLEARANCE_M = 0.002
# This benchmark evaluates contact-pose/IK feasibility rather than actuator
# limits.  Keep one invariant stiff Panda servo for every asset so a new agent
# cannot "repair" geometry by tuning torque or gravity sag.
PANDA_ARM_FORCE_N = 1000.0
PANDA_ARM_FORCE_SCALE = 1.0
PANDA_ARM_POSITION_GAIN = 0.2
PANDA_FINGER_FORCE_N = 20.0
# A physically verified grasp gates a plan-authoritative object-joint actuator.
# The actuator target is tied to measured dense-sample robot progress along the
# current IK path, never elapsed time. This deliberately evaluates contact/plan causality
# rather than frictional force closure or Bullet constraint propagation.
CONTACT_GATED_ACTUATION_FORCE_OR_TORQUE = 1000.0
CONTACT_GATED_ACTUATION_POSITION_GAIN = 1.0
CONTACT_GATED_PROGRESS_MAXIMUM_RESIDUAL_RAD = 0.10
CONTACT_GATED_OBJECT_TRACKING_TOLERANCE_RAD = 0.01
GRASP_ACQUISITION_DWELL_S = 0.05
GRASP_PRE_CLOSE_SETTLE_S = 0.25
GRASP_VERIFICATION_STABILIZE_S = 0.15
GRASP_CONTACT_NORMAL_MAXIMUM_DOT = -0.25
GRASP_FINGER_MAXIMUM_SPEED_M_S = 0.01
GRASP_MINIMUM_CLOSURE_FRACTION = 0.25
DEFAULT_OBJECT_HOLD_FORCE_OR_TORQUE = 1000.0
OBJECT_JOINT_STABILITY_TOLERANCE_M_OR_RAD = 1e-4
# A release retreat is a real motion segment, not a zero-time phase boundary.
# Hold the solved safe endpoint long enough to produce multiple physics/video
# samples before any dependent mechanism motion or passive return is enabled.
# This is harness policy rather than a per-asset timing knob.
RELEASE_RETREAT_SETTLE_S = 0.10

# Application-owned Panda and planner defaults.  These values describe the
# fixed robot/harness installed with this application; they are deliberately
# not task knobs.  Object-specific execution data supplies only semantic
# contact choices (link, point, interaction and visual roll/depth decisions).
PANDA_ARM_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
PANDA_FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
PANDA_END_EFFECTOR_LINK = "panda_grasptarget"
PANDA_HOME_JOINT_POSITIONS = [
    0.0,
    -math.pi / 4.0,
    0.0,
    -3.0 * math.pi / 4.0,
    0.0,
    math.pi / 2.0,
    math.pi / 4.0,
]
DEFAULT_CONTACT_ROLL_DEGREES = (0.0, 45.0, 90.0, 135.0, 180.0)
DENSE_MANIPULATION_PATH_SAMPLES = 257
DEFAULT_PRECONTACT_OFFSET_M = 0.10
DEFAULT_FINAL_FINGER_OPENING_M = 0.0064
# ``finger_opening_m`` is one Panda finger-joint position, so the physical jaw
# aperture is twice this value.  Transit/approach needs only a small clearance
# around the width selected for the final grasp; opening every grasp to the
# Panda's 80 mm maximum makes the outside of a finger sweep nearby housing long
# before the selected feature reaches the pads.  Add 5 mm to the *total* jaw
# aperture (2.5 mm per finger) instead of using one fixed maximum-width pose.
DEFAULT_APPROACH_TOTAL_JAW_CLEARANCE_M = 0.005
DEFAULT_CONTACT_CLOSE_S = 0.50
DEFAULT_CONTACT_SETTLE_S = 0.30
DEFAULT_CONTACT_RELEASE_S = 0.50
DEFAULT_MANIPULATION_SAMPLE_HOLD_S = 0.05
DEFAULT_IK_RANDOM_RESTARTS = 64
DEFAULT_IK_MAX_ITERATIONS = 4000
DEFAULT_SEARCH_SEED = 27024
DEFAULT_PHYSICS_SEED = 1101
# Missing URDF inertials make Bullet assign unit mass to articulated links. A
# passive return must therefore overcome at least a few unit-mass gravity
# loads in any orientation; the former 5 N/Nm cap could saturate before a
# declared spring return moved at all. Keep one bounded application value well
# below the task/hold actuator while preserving the plan-owned return target.
DEFAULT_PASSIVE_RETURN_FORCE = 50.0
DEFAULT_PASSIVE_RETURN_POSITION_GAIN = 0.5


def _validate_execution_schema(execution: dict[str, Any]) -> None:
    """Reject malformed execution data with one readable message.

    Without this, a mistyped or misplaced field surfaces much later as an opaque
    ``KeyError`` deep inside planning, which is expensive for an agent to
    diagnose.  Schema validation turns that into a pointed error naming the exact
    JSON path.
    """
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "jsonschema is required to validate execution data; "
            "install it from requirements.txt"
        ) from exc
    schema = _read_json(EXECUTION_SCHEMA)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(execution), key=lambda item: list(item.path))
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise ValueError(f"Execution data violates execution.schema.json -> {details}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Invalid or empty JSONL: {path}")
    return rows


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _coordinate_frame_signature(execution: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable world-frame fields for one task search directory."""
    scene = execution["scene"]
    robot = execution["robot"]
    return {
        "object_base_translation_m": list(scene["object_base_translation_m"]),
        "object_base_rotation_xyzw": list(scene["object_base_rotation_xyzw"]),
        "support_surface": scene.get("support_surface"),
        "robot_base_translation_m": list(robot["base_translation_m"]),
        "robot_base_rotation_xyzw": list(robot["base_rotation_xyzw"]),
    }


def _enforce_coordinate_frame_lock(execution_path: Path, execution: dict[str, Any]) -> dict[str, Any]:
    """Keep all search attempts in one world frame.

    The sidecar lives beside per-task search execution data (never in the
    published three-file directory).  It intentionally excludes only the
    auto-grounded effective z values; raw scene fields remain locked too, so
    a candidate cannot move or rotate the object/robot to manufacture contact.
    """
    signature = _coordinate_frame_signature(execution)
    lock_path = execution_path.parent / ".coordinate-frame-lock.json"
    if lock_path.exists():
        locked = _read_json(lock_path)
        if locked.get("signature") != signature:
            changed = sorted(
                key for key in set(signature) | set(locked.get("signature", {}))
                if signature.get(key) != locked.get("signature", {}).get(key)
            )
            raise RuntimeError(
                "Coordinate frame changed between search attempts; "
                f"locked fields differ: {changed}"
            )
    else:
        lock_path.write_text(
            json.dumps(
                {"schema_version": 1, "signature": signature, "sha256": _canonical_hash(signature)},
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {"path": str(lock_path), "sha256": _canonical_hash(signature), "signature": signature}


def _maps(body: int, client: int) -> tuple[dict[str, int], dict[str, int]]:
    joints: dict[str, int] = {}
    base_name = p.getBodyInfo(body, physicsClientId=client)[0].decode("utf-8")
    links: dict[str, int] = {base_name: -1}
    for index in range(p.getNumJoints(body, physicsClientId=client)):
        info = p.getJointInfo(body, index, physicsClientId=client)
        joints[info[1].decode("utf-8")] = index
        links[info[12].decode("utf-8")] = index
    return joints, links


@dataclass(frozen=True)
class _StaticConcaveOverlay:
    body: int
    source_link_index: int
    source_link_name: str


# PyBullet imports a mesh attached to an articulated link as a convex collision
# shape, even when that link is rigidly fixed to a fixed base.  That can fill a
# doorway, cabinet opening, or handle recess with nonexistent solid volume.  A
# static concave copy is therefore installed for mesh-only links in the fixed
# base subtree.  The original link remains the visual/kinematic authority; only
# its collision response is replaced.  Registry keys are overwritten on every
# scene load because PyBullet reuses body ids after disconnect.
_STATIC_CONCAVE_OVERLAYS: dict[
    tuple[int, int], list[_StaticConcaveOverlay]
] = {}


def _fixed_base_subtree_links(object_urdf: Path) -> set[str]:
    import xml.etree.ElementTree as ET

    root = ET.parse(object_urdf).getroot()
    link_names = {
        str(link.get("name")) for link in root.findall("link") if link.get("name")
    }
    child_names = {
        str(child.get("link"))
        for joint in root.findall("joint")
        for child in [joint.find("child")]
        if child is not None and child.get("link")
    }
    roots = sorted(link_names - child_names)
    if len(roots) != 1:
        return set()
    fixed = {roots[0]}
    changed = True
    while changed:
        changed = False
        for joint in root.findall("joint"):
            if str(joint.get("type")) != "fixed":
                continue
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is None or child is None:
                continue
            parent_name = str(parent.get("link"))
            child_name = str(child.get("link"))
            if parent_name in fixed and child_name not in fixed:
                fixed.add(child_name)
                changed = True
    return fixed


def _install_static_concave_overlays(
    object_urdf: Path,
    object_body: int,
    client: int,
) -> list[_StaticConcaveOverlay]:
    """Replace convexified fixed-link meshes with static source trimeshes."""
    _, links = _maps(object_body, client)
    installed: list[_StaticConcaveOverlay] = []
    for link_name in sorted(_fixed_base_subtree_links(object_urdf)):
        if link_name not in links:
            continue
        link_index = int(links[link_name])
        shapes = list(
            p.getCollisionShapeData(
                object_body, link_index, physicsClientId=client
            )
            or []
        )
        # Primitive shapes are already exact. Mixed links remain untouched so
        # no primitive is accidentally removed or duplicated.
        if not shapes or any(int(shape[2]) != p.GEOM_MESH for shape in shapes):
            continue
        link_position, link_rotation = link_world_pose(
            object_body, link_index, client
        )
        pending: list[_StaticConcaveOverlay] = []
        try:
            for shape in shapes:
                filename = (
                    shape[4].decode("utf-8")
                    if isinstance(shape[4], bytes)
                    else str(shape[4])
                )
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    mesh_path = object_urdf.parent / mesh_path
                if not mesh_path.is_file():
                    raise FileNotFoundError(
                        f"Static collision mesh does not exist: {mesh_path}"
                    )
                world_position, world_rotation = p.multiplyTransforms(
                    link_position,
                    link_rotation,
                    list(shape[5]),
                    list(shape[6]),
                    physicsClientId=client,
                )
                collision_shape = p.createCollisionShape(
                    p.GEOM_MESH,
                    fileName=str(mesh_path.resolve()),
                    meshScale=[float(value) for value in shape[3]],
                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                    physicsClientId=client,
                )
                overlay_body = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision_shape,
                    baseVisualShapeIndex=-1,
                    basePosition=world_position,
                    baseOrientation=world_rotation,
                    physicsClientId=client,
                )
                # The overlay is a collision-only replacement for one fixed
                # link of this same articulated object.  As a separate Bullet
                # body it would otherwise collide with movable sibling links,
                # reintroducing self-collisions that the source multibody does
                # not have and mechanically locking doors/drawers.  Disable
                # only these internal object-overlay pairs; robot-overlay
                # collision remains enabled and is queried explicitly below.
                for source_object_link in range(
                    -1, p.getNumJoints(object_body, physicsClientId=client)
                ):
                    p.setCollisionFilterPair(
                        object_body,
                        overlay_body,
                        source_object_link,
                        -1,
                        0,
                        physicsClientId=client,
                    )
                pending.append(
                    _StaticConcaveOverlay(
                        body=int(overlay_body),
                        source_link_index=link_index,
                        source_link_name=link_name,
                    )
                )
        except Exception:
            for overlay in pending:
                p.removeBody(overlay.body, physicsClientId=client)
            raise
        # Disable the convexified source only after every source mesh has a
        # successful exact replacement.
        p.setCollisionFilterGroupMask(
            object_body,
            link_index,
            collisionFilterGroup=0,
            collisionFilterMask=0,
            physicsClientId=client,
        )
        installed.extend(pending)
    _STATIC_CONCAVE_OVERLAYS[(int(client), int(object_body))] = installed
    return installed


def _overlay_rows(
    body_a: int,
    object_body: int,
    client: int,
    *,
    distance: float | None,
    link_index_a: int | None,
    link_index_b: int | None,
    contacts: bool,
) -> list[tuple[Any, ...]]:
    overlays = _STATIC_CONCAVE_OVERLAYS.get(
        (int(client), int(object_body)), []
    )
    overlay_indices = {item.source_link_index for item in overlays}
    rows: list[tuple[Any, ...]] = []
    # An overlaid source link is intentionally absent from the original query;
    # explicit closest-point calls ignore broadphase collision masks on some
    # Bullet builds, so filtering only by masks is insufficient.
    query_original = link_index_b is None or link_index_b not in overlay_indices
    if query_original:
        kwargs: dict[str, Any] = {"physicsClientId": client}
        if link_index_a is not None:
            kwargs["linkIndexA"] = int(link_index_a)
        if link_index_b is not None:
            kwargs["linkIndexB"] = int(link_index_b)
        original = (
            p.getContactPoints(body_a, object_body, **kwargs)
            if contacts
            else p.getClosestPoints(body_a, object_body, float(distance), **kwargs)
        )
        rows.extend(
            tuple(point)
            for point in original
            if int(point[4]) not in overlay_indices
        )
    for overlay in overlays:
        if (
            link_index_b is not None
            and overlay.source_link_index != int(link_index_b)
        ):
            continue
        kwargs = {"physicsClientId": client}
        if link_index_a is not None:
            kwargs["linkIndexA"] = int(link_index_a)
        overlay_points = (
            p.getContactPoints(body_a, overlay.body, **kwargs)
            if contacts
            else p.getClosestPoints(
                body_a, overlay.body, float(distance), **kwargs
            )
        )
        for point in overlay_points:
            normalized = list(point)
            normalized[2] = int(object_body)
            normalized[4] = int(overlay.source_link_index)
            rows.append(tuple(normalized))
    return rows


def object_closest_points(
    body_a: int,
    object_body: int,
    distance: float,
    client: int,
    *,
    link_index_a: int | None = None,
    link_index_b: int | None = None,
) -> list[tuple[Any, ...]]:
    return _overlay_rows(
        body_a,
        object_body,
        client,
        distance=float(distance),
        link_index_a=link_index_a,
        link_index_b=link_index_b,
        contacts=False,
    )


def object_contact_points(
    body_a: int,
    object_body: int,
    client: int,
    *,
    link_index_a: int | None = None,
    link_index_b: int | None = None,
) -> list[tuple[Any, ...]]:
    return _overlay_rows(
        body_a,
        object_body,
        client,
        distance=None,
        link_index_a=link_index_a,
        link_index_b=link_index_b,
        contacts=True,
    )


def _body_min_z(body: int, client: int) -> float:
    """Return the lowest collision AABB point for a loaded body."""
    p.performCollisionDetection(physicsClientId=client)
    mins = [float(p.getAABB(body, -1, physicsClientId=client)[0][2])]
    mins.extend(
        float(p.getAABB(body, link, physicsClientId=client)[0][2])
        for link in range(p.getNumJoints(body, physicsClientId=client))
    )
    return min(mins)


def _support_top_z(scene: dict[str, Any]) -> float:
    support = scene.get("support_surface")
    if isinstance(support, dict):
        center = support.get("center_m", [0.0, 0.0, -0.011])
        half = support.get("half_extents_m", [1.0, 1.0, 0.01])
        return float(center[2]) + float(half[2])
    # The renderer's generic floor is centered at -0.011 with half-height
    # 0.01, hence its visible top is -0.001 m.
    return -0.001


def _ground_execution_scene(
    object_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
) -> dict[str, Any]:
    """Ground both bodies from their initial geometry, without asset branches.

    The returned execution copy is the one used by planning, rollout, and
    grasp.json.  This makes placement deterministic and visible in the
    published execution data instead of silently relying on a camera or mesh
    convention.
    """
    effective = copy.deepcopy(execution)
    scene = effective["scene"]
    robot_spec = effective["robot"]
    support_top = _support_top_z(scene)
    clearance = float(scene.get("ground_clearance_m", GROUND_CLEARANCE_M))
    if clearance < 0.0:
        raise ValueError("scene.ground_clearance_m must be non-negative")

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet grounding client")
    try:
        object_body = p.loadURDF(
            str(object_urdf),
            basePosition=scene["object_base_translation_m"],
            baseOrientation=_quat(scene["object_base_rotation_xyzw"]),
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
            physicsClientId=client,
        )
        object_joints, _ = _maps(object_body, client)
        for name, value in initial.items():
            if name in object_joints:
                p.resetJointState(object_body, object_joints[name], float(value), physicsClientId=client)
        requested_object_min = _body_min_z(object_body, client)
        object_shift = support_top + clearance - requested_object_min
        object_position = list(scene["object_base_translation_m"])
        object_position[2] += object_shift
        scene["object_base_translation_m"] = object_position

        robot_body = p.loadURDF(
            str(robot_urdf),
            basePosition=robot_spec["base_translation_m"],
            baseOrientation=_quat(robot_spec["base_rotation_xyzw"]),
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
            physicsClientId=client,
        )
        robot_joints, _ = _maps(robot_body, client)
        home = robot_spec.get("home_joint_positions", [])
        for name, value in zip(robot_spec.get("arm_joint_names", []), home):
            if name in robot_joints:
                p.resetJointState(robot_body, robot_joints[name], float(value), physicsClientId=client)
        requested_robot_min = _body_min_z(robot_body, client)
        # A declared pedestal raises the robot's own support, so ground the base to
        # the top of that plinth instead of the floor.  Reaching a contact that sits
        # well above the shoulder otherwise forces the forearm through the object.
        robot_support_top = support_top + float(robot_spec.get("base_support_height_m", 0.0))
        robot_shift = robot_support_top + clearance - requested_robot_min
        robot_position = list(robot_spec["base_translation_m"])
        robot_position[2] += robot_shift
        robot_spec["base_translation_m"] = robot_position

        # Re-check after applying both deterministic shifts.  The booleans are
        # carried into native result.json so a run cannot pass with an
        # underground initial scene hidden behind a video crop.
        p.resetBasePositionAndOrientation(
            object_body,
            scene["object_base_translation_m"],
            _quat(scene["object_base_rotation_xyzw"]),
            physicsClientId=client,
        )
        p.resetBasePositionAndOrientation(
            robot_body,
            robot_spec["base_translation_m"],
            _quat(robot_spec["base_rotation_xyzw"]),
            physicsClientId=client,
        )
        # PyBullet's fixed-joint/inertial frame can make a base reset move a
        # configured arm by a slightly different z offset.  Correct that
        # generic numerical effect by re-measuring after each reset; no asset
        # or robot name is involved.  The same loop also protects against a
        # small initial AABB update lag.
        for _ in range(4):
            current_object_min = _body_min_z(object_body, client)
            current_robot_min = _body_min_z(robot_body, client)
            object_correction = support_top + clearance - current_object_min
            robot_correction = robot_support_top + clearance - current_robot_min
            if abs(object_correction) > 1e-7:
                scene["object_base_translation_m"][2] += object_correction
                p.resetBasePositionAndOrientation(
                    object_body,
                    scene["object_base_translation_m"],
                    _quat(scene["object_base_rotation_xyzw"]),
                    physicsClientId=client,
                )
            if abs(robot_correction) > 1e-7:
                robot_spec["base_translation_m"][2] += robot_correction
                p.resetBasePositionAndOrientation(
                    robot_body,
                    robot_spec["base_translation_m"],
                    _quat(robot_spec["base_rotation_xyzw"]),
                    physicsClientId=client,
                )
        final_object_min = _body_min_z(object_body, client)
        final_robot_min = _body_min_z(robot_body, client)
        passed = final_object_min >= support_top - 1e-7 and final_robot_min >= robot_support_top - 1e-7
        return {
            "support_top_z_m": support_top,
            "robot_support_top_z_m": robot_support_top,
            "robot_base_support_height_m": float(robot_spec.get("base_support_height_m", 0.0)),
            "clearance_m": clearance,
            "requested_object_min_z_m": requested_object_min,
            "requested_robot_min_z_m": requested_robot_min,
            "object_shift_z_m": object_shift,
            "robot_shift_z_m": robot_shift,
            "effective_object_base_translation_m": list(scene["object_base_translation_m"]),
            "effective_robot_base_translation_m": list(robot_spec["base_translation_m"]),
            "final_object_min_z_m": final_object_min,
            "final_robot_min_z_m": final_robot_min,
            "passed": passed,
            "execution": effective,
        }
    finally:
        p.disconnect(client)


def _quat(value: Iterable[float]) -> list[float]:
    array = np.asarray(list(value), dtype=np.float64)
    if array.shape != (4,) or float(np.linalg.norm(array)) < 1e-9:
        raise ValueError(f"Invalid quaternion: {value}")
    return (array / np.linalg.norm(array)).tolist()


def _initial_joint_values(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = rows[0].get("joint_angles")
    if not isinstance(values, dict):
        raise ValueError("First trajectory row has no joint_angles object")
    return {str(name): float(value) for name, value in values.items()}


def task_initial_joint_values(task: dict[str, Any]) -> dict[str, float]:
    """Return the optional captured initial state, otherwise URDF defaults.

    ArtiMo trajectories are never replayed by the robot-contact application;
    when supplied, only their first frame is used to recover a non-zero initial
    object configuration.  A plan-only handoff intentionally starts every URDF
    joint at its declared/default zero configuration.
    """
    trajectory = task.get("inputs", {}).get("trajectory")
    if trajectory is None:
        return {}
    return _initial_joint_values(_read_jsonl(_resolve(str(trajectory))))


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert one proper 3x3 rotation matrix to a canonical XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("Expected a 3x3 rotation matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
        w = 0.25 * scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    # q and -q encode the same rotation.  Fix the sign so hashes and fresh
    # agents cannot create two representations of one application-owned frame.
    first = next((value for value in quaternion if abs(float(value)) > 1e-12), 1.0)
    if first < 0.0:
        quaternion *= -1.0
    quaternion[np.abs(quaternion) < 1e-12] = 0.0
    return [float(value) for value in quaternion]


def _quaternion_multiply_xyzw(left: Iterable[float], right: Iterable[float]) -> list[float]:
    lx, ly, lz, lw = _quat(left)
    rx, ry, rz, rw = _quat(right)
    return _quat(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def _smoothstep01(value: float | np.ndarray) -> float | np.ndarray:
    """The single object-trajectory parameterization used by plan and rollout."""
    clipped = np.clip(value, 0.0, 1.0)
    return 3.0 * clipped * clipped - 2.0 * clipped * clipped * clipped


def _canonical_contact_frame(
    task: dict[str, Any], stage: dict[str, Any]
) -> dict[str, Any]:
    """Derive a deterministic link-local contact frame from collision geometry.

    Local +Z is the measured outward surface normal.  Local +X is the contacted
    link's longest principal tangent direction, with a deterministic sign.  An
    agent therefore chooses only the point and one of four visual wrist rolls;
    it cannot silently replace the common roll-zero frame with another IK branch.
    """
    source_urdf = _resolve(task["inputs"]["urdf"])
    initial = task_initial_joint_values(task)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect contact-frame geometry client")
    try:
        body = p.loadURDF(
            str(source_urdf),
            useFixedBase=True,
            flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
            physicsClientId=client,
        )
        joints, links = _maps(body, client)
        for name, value in initial.items():
            if name in joints:
                p.resetJointState(body, joints[name], float(value), physicsClientId=client)
        contact_link = str(stage["contact_link"])
        if contact_link not in links:
            raise KeyError(f"Unknown contact link {contact_link!r}")
        link_index = links[contact_link]
        point_link = np.asarray(
            stage["contact_pose_link"]["translation_m"], dtype=np.float64
        )
        link_position, link_rotation = link_world_pose(body, link_index, client)
        point_world, _ = p.multiplyTransforms(
            link_position,
            link_rotation,
            point_link.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client,
        )
        probe_radius = 0.0005
        probe_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=probe_radius, physicsClientId=client
        )
        probe = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=probe_shape,
            basePosition=point_world,
            physicsClientId=client,
        )
        hits = p.getClosestPoints(
            probe,
            body,
            0.25,
            linkIndexB=link_index,
            physicsClientId=client,
        )
        if not hits:
            raise ValueError(
                f"Contact point for {contact_link!r} has no nearby collision surface"
            )
        hit = min(hits, key=lambda item: abs(float(item[8]) + probe_radius))
        inverse_link_rotation = p.invertTransform(
            [0.0, 0.0, 0.0], link_rotation
        )[1]
        normal_link = np.asarray(
            p.rotateVector(inverse_link_rotation, hit[7]), dtype=np.float64
        )
        normal_link /= np.linalg.norm(normal_link)

        _, vertices = p.getMeshData(
            body,
            link_index,
            flags=p.MESH_DATA_SIMULATION_MESH,
            physicsClientId=client,
        )
        points = np.asarray(vertices, dtype=np.float64)
        tangent: np.ndarray | None = None
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 3:
            covariance = np.cov((points - points.mean(axis=0)).T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            for eigen_index in np.argsort(eigenvalues)[::-1]:
                candidate = eigenvectors[:, int(eigen_index)]
                candidate = candidate - normal_link * float(np.dot(candidate, normal_link))
                length = float(np.linalg.norm(candidate))
                if length > 1e-6:
                    tangent = candidate / length
                    break
        if tangent is None:
            axes = np.eye(3, dtype=np.float64)
            reference = min(axes, key=lambda axis: abs(float(np.dot(axis, normal_link))))
            tangent = reference - normal_link * float(np.dot(reference, normal_link))
            tangent /= np.linalg.norm(tangent)
        sign_index = int(np.argmax(np.abs(tangent)))
        if tangent[sign_index] < 0.0:
            tangent *= -1.0
        bitangent = np.cross(normal_link, tangent)
        bitangent /= np.linalg.norm(bitangent)
        tangent = np.cross(bitangent, normal_link)
        tangent /= np.linalg.norm(tangent)
        base_rotation = _matrix_to_quaternion_xyzw(
            np.column_stack((tangent, bitangent, normal_link))
        )
        return {
            "rotation_xyzw": base_rotation,
            "surface_normal_link": normal_link.tolist(),
            "principal_tangent_link": tangent.tolist(),
            "point_to_surface_distance_m": float(hit[8]) + probe_radius,
            "source": "application_collision_surface_normal_and_principal_tangent",
        }
    finally:
        p.disconnect(client)


def _canonical_contact_rotation_xyzw(
    task: dict[str, Any], stage: dict[str, Any], roll_degrees: float = 0.0
) -> tuple[list[float], dict[str, Any]]:
    frame = _canonical_contact_frame(task, stage)
    if float(roll_degrees) not in DEFAULT_CONTACT_ROLL_DEGREES:
        raise ValueError(
            "contact_roll_deg must be one of the application-owned five rolls "
            f"{DEFAULT_CONTACT_ROLL_DEGREES}"
        )
    half_angle = math.radians(float(roll_degrees)) * 0.5
    roll = [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
    return _quaternion_multiply_xyzw(frame["rotation_xyzw"], roll), frame


def _urdf_robot_contact_links(
    robot_urdf: Path, finger_joint_names: Iterable[str]
) -> list[str]:
    wanted = set(str(value) for value in finger_joint_names)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect robot-default geometry client")
    try:
        body = p.loadURDF(str(robot_urdf), useFixedBase=True, physicsClientId=client)
        links = []
        for index in range(p.getNumJoints(body, physicsClientId=client)):
            info = p.getJointInfo(body, index, physicsClientId=client)
            if info[1].decode("utf-8") in wanted:
                links.append(info[12].decode("utf-8"))
        if len(links) != len(wanted):
            raise ValueError(
                "Fixed Panda finger joints do not map uniquely to robot contact links"
            )
        return links
    finally:
        p.disconnect(client)


def _urdf_object_link_names(object_urdf: Path) -> list[str]:
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect object-default geometry client")
    try:
        body = p.loadURDF(str(object_urdf), useFixedBase=True, physicsClientId=client)
        return list(_maps(body, client)[1])
    finally:
        p.disconnect(client)


def _application_planning_backend() -> dict[str, Any]:
    """Return the installed machine backend without consulting task/agent data."""
    curobo_python = Path("C:/ProgramData/miniforge3/envs/artimo-curobo/python.exe")
    if curobo_python.is_file():
        return {
            "name": "curobo",
            "python_executable": str(curobo_python),
            "device": "cuda:0",
            "num_seeds": 64,
            # Dense continuity needs the complete seed bank. Keeping only the
            # best 16 pose-wise solutions can discard the one redundant-arm
            # branch that connects to the adjacent sample even though every
            # individual pose remains feasible.
            "return_seeds": 64,
            "cuda_graph": True,
            "self_collision": True,
            "environment_collision": True,
            "collision_sphere_buffer_m": 0.0,
            "allow_bullet_fallback": False,
            "motion_num_graph_seeds": 4,
            "motion_num_trajopt_seeds": 4,
            "motion_timeout_s": 10.0,
            "motion_max_attempts": 6,
        }
    return {"name": "bullet"}


def materialize_execution_defaults(
    task: dict[str, Any], execution_input: dict[str, Any]
) -> dict[str, Any]:
    """Materialize application-owned execution fields before validation/use.

    The input may omit robot constants, contact-frame rotation, plan projection,
    forbidden links, acquisition timing, backend and numerical budgets.  Values
    supplied for those fields are overwritten, so a fresh agent cannot change
    search semantics by guessing a different quaternion or tolerance bundle.
    """
    execution = copy.deepcopy(execution_input)
    execution["schema_version"] = 2
    # Collision representation is application-owned.  Ignore legacy agent-authored
    # task-local proxies so sparse, dense, transit, release, and rollout all use
    # the same deterministic model derived from the frozen source input.
    execution.pop("physics_urdf", None)
    scene = execution.setdefault("scene", {})
    scene.setdefault("object_base_translation_m", [0.0, 0.0, 0.0])
    scene.setdefault("object_base_rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
    scene.setdefault("ground_clearance_m", GROUND_CLEARANCE_M)

    robot = execution.setdefault("robot", {})
    robot.setdefault("base_translation_m", [0.0, 0.0, 0.0])
    robot.setdefault("base_rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
    robot["arm_joint_names"] = list(PANDA_ARM_JOINT_NAMES)
    robot["finger_joint_names"] = list(PANDA_FINGER_JOINT_NAMES)
    robot["end_effector_link"] = PANDA_END_EFFECTOR_LINK
    robot["home_joint_positions"] = list(PANDA_HOME_JOINT_POSITIONS)
    robot.setdefault("base_support_height_m", 0.0)

    execution["planning_ik_backend"] = _application_planning_backend()
    execution["seeds"] = {
        "search": DEFAULT_SEARCH_SEED,
        "physics": DEFAULT_PHYSICS_SEED,
    }
    execution["ik"] = {
        "random_restarts": DEFAULT_IK_RANDOM_RESTARTS,
        "max_iterations": DEFAULT_IK_MAX_ITERATIONS,
    }
    execution.setdefault("causal_rules", [])
    execution.setdefault("control_execution", [])
    execution.setdefault("stages", [])
    execution.setdefault("settle_s", 2.0)
    execution.setdefault(
        "camera",
        {"eye_m": [2.0, -2.0, 1.45], "target_m": [0.0, 0.0, 0.8], "fov_deg": 48.0},
    )

    plan = _read_json(_resolve(task["inputs"]["plan"]))
    phases = artimo_plan.phases_by_name(plan)
    phase_order = {
        str(phase["name"]): index for index, phase in enumerate(plan.get("timeline", []))
    }
    stages_by_id = {str(stage.get("id", "")): stage for stage in execution["stages"]}
    robot_rows = []
    for row in execution["control_execution"]:
        if row.get("motion_owner") != "robot_contact":
            continue
        stage_id = str(row.get("stage_id", ""))
        if stage_id not in stages_by_id:
            raise ValueError(
                f"robot_contact ownership references missing contact stage {stage_id!r}"
            )
        phase_name = str(row["source_phase"])
        control_index = int(row["source_control_index"])
        controls = phases[phase_name].get("controls", [])
        if not 0 <= control_index < len(controls):
            raise ValueError(f"Invalid plan control {phase_name!r}[{control_index}]")
        control = controls[control_index]
        target = artimo_plan.control_target(control)
        if target is None:
            raise ValueError(
                f"robot_contact control {phase_name!r}[{control_index}] has no target"
            )
        stage = stages_by_id[stage_id]
        stage["source_phase"] = phase_name
        stage["source_control_index"] = control_index
        stage["driver_joint"] = str(control["joint"])
        stage["target_joint_position"] = float(target)
        stage.pop("command_joint_position", None)
        robot_rows.append((phase_order[phase_name], control_index, stage))

    robot_rows.sort(key=lambda item: (item[0], item[1]))
    groups: list[list[tuple[int, int, dict[str, Any]]]] = []
    for row in robot_rows:
        if not groups:
            groups.append([row])
            continue
        prior_phase = groups[-1][-1][0]
        current_phase = row[0]
        contact_link_changed = str(groups[-1][-1][2]["contact_link"]) != str(
            row[2]["contact_link"]
        )
        released = any(
            str(plan["timeline"][index].get("phase_type", "")) == "control_release"
            for index in range(prior_phase + 1, current_phase + 1)
        )
        if released or contact_link_changed:
            groups.append([row])
        else:
            groups[-1].append(row)
    for group_index, group in enumerate(groups):
        sequence = f"contact_sequence_{group_index:03d}"
        contact_links = {str(item[2].get("contact_link", "")) for item in group}
        if len(contact_links) != 1:
            raise ValueError(
                "Plan has no release between robot controls, so application-owned "
                f"sequence {sequence!r} requires one contact link; got {sorted(contact_links)}"
            )
        for _, _, stage in group:
            stage["contact_sequence"] = sequence
            stage.pop("release_before_phase", None)
        final_phase_index = group[-1][0]
        # Materialize the earliest release boundary implied by plan ownership.
        # A later robot-contact group requires release/reacquisition before its
        # first phase.  A causal mechanism requires the triggering contact
        # sequence to clear before its effect starts.  Passive return and an
        # explicit control_release are also plan-derived release boundaries.
        # This is deliberately expressed only in generic plan semantics; asset
        # identity never participates in execution materialization.
        next_group = groups[group_index + 1] if group_index + 1 < len(groups) else None
        group_stage_ids = {str(item[2]["id"]) for item in group}
        group_driver_joints = {str(item[2]["driver_joint"]) for item in group}
        release_candidates: set[str] = set()
        if next_group is not None:
            release_candidates.add(str(next_group[0][2]["source_phase"]))
        release_candidates.update(
            str(rule["source_effect_phase"])
            for rule in execution.get("causal_rules", [])
            if str(rule.get("trigger_stage", "")) in group_stage_ids
        )
        for row in execution["control_execution"]:
            if row.get("motion_owner") != "passive_return":
                continue
            phase_name = str(row["source_phase"])
            control_index = int(row["source_control_index"])
            control = phases[phase_name]["controls"][control_index]
            if str(control.get("joint", "")) in group_driver_joints:
                release_candidates.add(phase_name)
        release_candidates.update(
            str(phase["name"])
            for phase in plan["timeline"][final_phase_index + 1 :]
            if str(phase.get("phase_type", "")) == "control_release"
        )
        release_phase = min(
            release_candidates,
            key=phase_order.__getitem__,
            default=None,
        )
        if release_phase is not None:
            group[-1][2]["release_before_phase"] = release_phase

    object_urdf = _resolve(task["inputs"]["urdf"])
    object_links = _urdf_object_link_names(object_urdf)
    robot_urdf = _resolve(task["inputs"]["robot_urdf"])
    finger_links = _urdf_robot_contact_links(robot_urdf, PANDA_FINGER_JOINT_NAMES)
    for stage in execution["stages"]:
        interaction = str(stage.get("interaction", "explicit_ideal_feasibility"))
        if interaction not in {"explicit_ideal_feasibility", "physical_push"}:
            raise ValueError(f"Unknown contact interaction {interaction!r}")
        stage["interaction"] = interaction
        mode = "open_then_close" if interaction == "explicit_ideal_feasibility" else "maintain_width"
        final_opening = float(stage.get("finger_opening_m", DEFAULT_FINAL_FINGER_OPENING_M))
        stage["finger_opening_m"] = final_opening
        stage["contact_acquisition"] = {
            "mode": mode,
            "approach_finger_opening_m": (
                final_opening + 0.5 * DEFAULT_APPROACH_TOTAL_JAW_CLEARANCE_M
                if mode == "open_then_close"
                else final_opening
            ),
            "close_s": DEFAULT_CONTACT_CLOSE_S if mode == "open_then_close" else 0.0,
            "settle_s": DEFAULT_CONTACT_SETTLE_S,
            "release_s": DEFAULT_CONTACT_RELEASE_S if mode == "open_then_close" else 0.0,
        }
        stage.setdefault("precontact_offset_m", DEFAULT_PRECONTACT_OFFSET_M)
        stage.setdefault("grasp_depth_m", 0.0)
        stage["manipulation_sample_hold_s"] = DEFAULT_MANIPULATION_SAMPLE_HOLD_S
        stage.setdefault("minimum_swept_clearance_m", 0.0)
        contact_link = str(stage["contact_link"])
        stage["forbidden_contact_links"] = [
            link for link in object_links if link != contact_link
        ]
        if mode == "open_then_close":
            stage["allowed_robot_contact_links"] = list(finger_links)
            stage.pop("robot_tool_contact_offset_eef_m", None)
        elif not stage.get("allowed_robot_contact_links"):
            # A physical push may intentionally nominate one of the two fingers;
            # only that semantic tool choice remains agent-owned.
            stage["allowed_robot_contact_links"] = [finger_links[0]]
        roll_degrees = float(stage.get("contact_roll_deg", 0.0))
        rotation, frame = _canonical_contact_rotation_xyzw(
            task, stage, roll_degrees
        )
        stage["contact_roll_deg"] = roll_degrees
        stage["contact_frame_source"] = frame["source"]
        stage["contact_pose_link"]["rotation_xyzw"] = rotation

    passive = []
    for row in execution["control_execution"]:
        if row.get("motion_owner") != "passive_return":
            continue
        phase = phases[str(row["source_phase"])]
        control = phase["controls"][int(row["source_control_index"])]
        passive.append(
            {
                "joint": str(control["joint"]),
                "rest_position": float(artimo_plan.control_target(control)),
                "maximum_force_or_torque": DEFAULT_PASSIVE_RETURN_FORCE,
                "position_gain": DEFAULT_PASSIVE_RETURN_POSITION_GAIN,
            }
        )
    execution["passive_joints"] = passive
    return execution


def _plan_requests(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the plan's joint requests through the one canonical parser.

    Sharing :mod:`artimo_plan` with the delivery verifier is what keeps the
    measured manifest and the acceptance gate from disagreeing about which
    extrema the plan asked for.
    """
    return artimo_plan.phase_targets(plan)


def _validate_execution_against_plan(
    plan: dict[str, Any],
    execution: dict[str, Any],
    *,
    require_release_route: bool = True,
) -> None:
    """Ensure every plan control has exactly one disclosed physical executor.

    Matching joint targets is not enough: without explicit ownership a contact
    on a control can silently trigger a motor on an unrelated effect joint.  In
    a compound task that can make a parent body move after the robot operates
    only its latch. ``control_execution`` closes that loophole by exhaustively
    assigning each timeline control to robot contact, a declared internal
    mechanism, a plan-declared passive return, or a no-new-motion hold.  The
    assignment is per control because one phase may combine different owners.
    """
    phases = artimo_plan.phases_by_name(plan)
    expected_keys = {
        (str(row["source_phase"]), int(row["source_control_index"]))
        for row in artimo_plan.timeline_controls(plan)
    }
    ownership_rows = execution.get("control_execution", [])
    ownership_keys = [
        (str(row.get("source_phase", "")), int(row.get("source_control_index", -1)))
        for row in ownership_rows
    ]
    if len(set(ownership_keys)) != len(ownership_keys):
        raise ValueError("control_execution contains duplicate phase/control entries")
    if set(ownership_keys) != expected_keys:
        missing = sorted(expected_keys - set(ownership_keys))
        extra = sorted(set(ownership_keys) - expected_keys)
        raise ValueError(
            "control_execution must cover every plan control exactly once; "
            f"missing={missing}, extra={extra}"
        )
    ownership = {key: row for key, row in zip(ownership_keys, ownership_rows)}

    stages = execution.get("stages", [])
    stage_ids = [str(stage.get("id", "")) for stage in stages]
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError("Execution contains duplicate stage ids")
    stages_by_id = {str(stage["id"]): stage for stage in stages}
    for stage in stages:
        _validate_contact_acquisition(stage)
        source_phase = str(stage.get("source_phase", ""))
        if source_phase not in phases:
            raise ValueError(f"Execution stage {stage.get('id')} has unknown source_phase {source_phase!r}")
        control_index = int(stage.get("source_control_index", -1))
        controls = phases[source_phase].get("controls", [])
        if not 0 <= control_index < len(controls):
            raise ValueError(
                f"Execution stage {stage.get('id')} has invalid source_control_index {control_index}"
            )
        control = controls[control_index]
        driver = str(stage.get("driver_joint"))
        expected = artimo_plan.control_target(control)
        if expected is None or control.get("joint") != driver:
            raise ValueError(
                f"Execution stage {stage.get('id')} does not project a target control "
                f"from {source_phase!r}[{control_index}]"
            )
        if abs(float(stage.get("target_joint_position")) - expected) > 1e-6:
            raise ValueError(
                f"Execution stage {stage.get('id')} target disagrees with "
                f"plan control {source_phase!r}[{control_index}]"
            )
        row = ownership[(source_phase, control_index)]
        if row.get("motion_owner") != "robot_contact" or row.get("stage_id") != stage["id"]:
            raise ValueError(
                f"Stage {stage['id']!r} drives {source_phase!r}[{control_index}], but that control is not assigned "
                "to this robot-contact stage"
            )

    rules = execution.get("causal_rules", [])
    rule_ids = [str(rule.get("id", "")) for rule in rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise ValueError("Execution contains duplicate causal-rule ids")
    rules_by_id = {str(rule["id"]): rule for rule in rules}
    stage_id_set = set(stage_ids)
    for rule in rules:
        if str(rule.get("trigger_stage")) not in stage_id_set:
            raise ValueError(
                f"Causal rule references unknown trigger_stage {rule.get('trigger_stage')!r}"
            )
        _validate_effect_group(plan, phases, rule, "source_effect_phase", "effects")
        for effect in rule["effects"]:
            effect_key = (str(rule["source_effect_phase"]), int(effect["source_control_index"]))
            effect_owner = ownership[effect_key]
            if (
                effect_owner.get("motion_owner") != "internal_mechanism"
                or effect_owner.get("causal_rule_id") != rule["id"]
            ):
                raise ValueError(
                    f"Causal rule {rule['id']!r} motors {effect_key}, but control_execution does "
                    "not assign that control to this internal mechanism"
                )
        release = rule.get("release")
        if release is not None:
            _validate_effect_group(plan, phases, release, "source_effect_phase", "effects")
            for effect in release["effects"]:
                release_key = (str(release["source_effect_phase"]), int(effect["source_control_index"]))
                release_owner = ownership[release_key]
                if (
                    release_owner.get("motion_owner") != "internal_mechanism"
                    or release_owner.get("causal_rule_id") != rule["id"]
                ):
                    raise ValueError(
                        f"Causal rule {rule['id']!r} release motors {release_key}, but "
                        "control_execution does not assign it to this internal mechanism"
                    )

    passive = {str(item["joint"]): item for item in execution.get("passive_joints", [])}
    assigned_stage_ids: set[str] = set()
    assigned_rule_ids: set[str] = set()
    assigned_passive_joints: set[str] = set()
    last_targets: dict[str, float] = {}
    for phase in plan.get("timeline", []):
        phase_name = str(phase["name"])
        for control_index, control in enumerate(phase.get("controls", [])):
            row = ownership[(phase_name, control_index)]
            owner = str(row["motion_owner"])
            mode = str(control.get("mode", ""))
            target = artimo_plan.control_target(control)
            joint = str(control.get("joint", ""))
            is_new = target is not None and (
                joint not in last_targets or abs(last_targets[joint] - float(target)) > 1e-12
            )

            if owner == "robot_contact":
                stage_id = str(row.get("stage_id", ""))
                if target is None or not is_new or stage_id not in stages_by_id:
                    raise ValueError(
                        f"robot_contact requires a new numeric target and known stage at {phase_name!r}[{control_index}]"
                    )
                stage = stages_by_id[stage_id]
                if (
                    stage.get("source_phase") != phase_name
                    or int(stage.get("source_control_index", -1)) != control_index
                    or stage.get("driver_joint") != joint
                    or abs(float(stage.get("target_joint_position")) - float(target)) > 1e-6
                ):
                    raise ValueError(f"Stage {stage_id!r} does not exactly implement {phase_name!r}[{control_index}]")
                assigned_stage_ids.add(stage_id)
            elif owner == "internal_mechanism":
                rule_id = str(row.get("causal_rule_id", ""))
                if target is None or not is_new or rule_id not in rules_by_id:
                    raise ValueError(
                        f"internal_mechanism requires a new numeric target and known causal rule at "
                        f"{phase_name!r}[{control_index}]"
                    )
                rule = rules_by_id[rule_id]
                groups = [rule] + ([rule["release"]] if isinstance(rule.get("release"), dict) else [])
                matches = [
                    effect
                    for group in groups
                    if group.get("source_effect_phase") == phase_name
                    for effect in group.get("effects", [])
                    if int(effect.get("source_control_index", -1)) == control_index
                ]
                if len(matches) != 1 or matches[0].get("joint") != joint or abs(float(matches[0]["target"]) - float(target)) > 1e-6:
                    raise ValueError(f"Causal rule {rule_id!r} does not exactly implement {phase_name!r}[{control_index}]")
                assigned_rule_ids.add(rule_id)
            elif owner == "passive_return":
                if mode != "spring_return" or target is None:
                    raise ValueError(
                        f"passive_return is allowed only for a plan-declared spring_return control: "
                        f"{phase_name!r}[{control_index}]"
                    )
                if joint not in passive or abs(float(passive[joint]["rest_position"]) - target) > 1e-6:
                    raise ValueError(
                        f"Passive control {phase_name!r}[{control_index}] lacks a matching rest target for {joint!r}"
                    )
                assigned_passive_joints.add(joint)
            elif owner == "hold":
                if mode != "hold_position" and is_new:
                    raise ValueError(
                        f"Hold control {phase_name!r}[{control_index}] may only retain an already reached target"
                    )
                if mode != "hold_position" and target is None:
                    raise ValueError(f"Hold control {phase_name!r}[{control_index}] has no hold semantics")
            else:  # schema validation should make this unreachable
                raise ValueError(f"Unknown motion_owner {owner!r}")

            forbidden_fields = {
                "robot_contact": {"causal_rule_id", "energy_source", "justification"},
                "internal_mechanism": {"stage_id"},
                "passive_return": {"stage_id", "causal_rule_id", "energy_source", "justification"},
                "hold": {"stage_id", "causal_rule_id", "energy_source", "justification"},
            }[owner]
            present_forbidden = sorted(forbidden_fields & set(row))
            if present_forbidden:
                raise ValueError(
                    f"Control owner {owner!r} at {phase_name!r}[{control_index}] has incompatible fields "
                    f"{present_forbidden}"
                )
            if target is not None:
                last_targets[joint] = float(target)

    if assigned_stage_ids != stage_id_set:
        raise ValueError(
            f"Every stage must be owned exactly once; unassigned={sorted(stage_id_set - assigned_stage_ids)}"
        )
    if assigned_rule_ids != set(rule_ids):
        raise ValueError(
            f"Every causal rule must implement an internal phase; unassigned={sorted(set(rule_ids) - assigned_rule_ids)}"
        )
    if assigned_passive_joints != set(passive):
        raise ValueError(
            "Every passive joint must implement a plan-declared passive_return; "
            f"unassigned={sorted(set(passive) - assigned_passive_joints)}"
        )

    _validate_contact_sequences(
        stages, plan, require_release_route=require_release_route
    )
    _validate_release_boundaries(plan, execution)


def _validate_release_boundaries(
    plan: dict[str, Any], execution: dict[str, Any]
) -> None:
    """Keep triggered object motion behind a completed robot retreat.

    ``release_before_phase`` used to be accepted at any later plan phase.  That
    allowed an execution to keep the robot at the contact pose while a causal
    lid/door/panel moved, then retreat only before a still-later control return.
    The earliest dependent moving phase is the actual safety boundary.
    """
    timeline = list(plan.get("timeline", []))
    phase_order = {
        str(phase["name"]): index for index, phase in enumerate(timeline)
    }
    ownership = {
        (str(row["source_phase"]), int(row["source_control_index"])): row
        for row in execution.get("control_execution", [])
    }
    stages = list(execution.get("stages", []))
    stages_by_id = {str(stage["id"]): stage for stage in stages}

    # A causal rule may be triggered by the acquisition stage of a continuous
    # contact sequence.  Release belongs to the final stage of that sequence.
    release_stage_by_trigger: dict[str, dict[str, Any]] = {}
    for stage in stages:
        sequence = stage.get("contact_sequence")
        release_stage = stage
        if sequence is not None:
            release_stage = next(
                candidate
                for candidate in reversed(stages)
                if candidate.get("contact_sequence") == sequence
            )
        release_stage_by_trigger[str(stage["id"])] = release_stage

    required: dict[str, list[tuple[int, str, str]]] = {}
    for rule in execution.get("causal_rules", []):
        trigger_id = str(rule["trigger_stage"])
        release_stage = release_stage_by_trigger[trigger_id]
        phase_name = str(rule["source_effect_phase"])
        required.setdefault(str(release_stage["id"]), []).append(
            (phase_order[phase_name], phase_name, "causal mechanism motion")
        )

    for stage in stages:
        source_index = phase_order[str(stage["source_phase"])]
        driver_joint = str(stage["driver_joint"])
        for phase_index in range(source_index + 1, len(timeline)):
            phase = timeline[phase_index]
            phase_name = str(phase["name"])
            if any(
                str(control.get("joint", "")) == driver_joint
                and ownership[(phase_name, control_index)]["motion_owner"]
                == "passive_return"
                for control_index, control in enumerate(phase.get("controls", []))
            ):
                release_stage = release_stage_by_trigger[str(stage["id"])]
                required.setdefault(str(release_stage["id"]), []).append(
                    (phase_index, phase_name, "passive return")
                )
                break

    for stage_id, boundaries in required.items():
        stage = stages_by_id[stage_id]
        source_index = phase_order[str(stage["source_phase"])]
        required_index, required_phase, reason = min(boundaries)
        if required_index <= source_index:
            raise ValueError(
                f"Stage {stage_id!r} triggers {reason} in phase "
                f"{required_phase!r}, but that phase is not later than its "
                "robot-contact phase; the plan provides no legal interval for "
                "release and retreat"
            )
        declared = stage.get("release_before_phase")
        if declared is None:
            raise ValueError(
                f"Stage {stage_id!r} requires release_before_phase at or before "
                f"{required_phase!r} because that is the earliest dependent "
                f"{reason}"
            )
        declared_index = phase_order[str(declared)]
        if declared_index > required_index:
            raise ValueError(
                f"Stage {stage_id!r} release_before_phase {declared!r} is too "
                f"late: {required_phase!r} begins {reason} first. The robot must "
                "release, retreat, and settle before that earlier phase."
            )

        # An immediately following plan-owned hold preserves the physical
        # contact.  Do not satisfy clearance by silently releasing before it.
        held_through = source_index
        driver_joint = str(stage["driver_joint"])
        for phase_index in range(source_index + 1, len(timeline)):
            phase = timeline[phase_index]
            phase_name = str(phase["name"])
            retains_driver = any(
                str(control.get("mode", "")) == "hold_position"
                and str(control.get("joint", "")) == driver_joint
                and ownership[(phase_name, control_index)]["motion_owner"] == "hold"
                for control_index, control in enumerate(phase.get("controls", []))
            )
            if not retains_driver:
                break
            held_through = phase_index
        if declared_index <= held_through:
            raise ValueError(
                f"Stage {stage_id!r} release_before_phase {declared!r} would "
                "release before its plan-owned hold completes"
            )


def _validate_contact_sequences(
    stages: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    require_release_route: bool = True,
) -> None:
    """Require a continued grasp to be adjacent and geometrically identical."""
    phase_order = {str(phase["name"]): index for index, phase in enumerate(plan.get("timeline", []))}
    previous_order = (-1, -1)
    sequence_indices: dict[str, list[int]] = {}
    for index, stage in enumerate(stages):
        current_order = (phase_order[str(stage["source_phase"])], int(stage["source_control_index"]))
        if current_order < previous_order:
            raise ValueError("Robot stages must follow plan control order")
        previous_order = current_order
        sequence = stage.get("contact_sequence")
        if sequence is not None:
            sequence_indices.setdefault(str(sequence), []).append(index)
    invariant_keys = (
        "interaction",
        "contact_link",
        "contact_pose_link",
        "contact_roll_deg",
        "contact_frame_source",
        "allowed_robot_contact_links",
        "finger_opening_m",
        "grasp_depth_m",
        "robot_tool_contact_offset_eef_m",
        "contact_acquisition",
    )
    for sequence, indices in sequence_indices.items():
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"contact_sequence {sequence!r} must occupy adjacent stages")
        first = stages[indices[0]]
        for index in indices[1:]:
            changed = [key for key in invariant_keys if stages[index].get(key) != first.get(key)]
            if changed:
                raise ValueError(
                    f"contact_sequence {sequence!r} cannot preserve one grasp; changed fields={changed}"
                )
    timeline = list(plan.get("timeline", []))
    for left, right in zip(stages, stages[1:]):
        left_index = phase_order[str(left["source_phase"])]
        right_index = phase_order[str(right["source_phase"])]
        left_driver = str(left["driver_joint"])
        intervening = timeline[left_index + 1 : right_index + 1]
        plan_releases_contact = any(
            str(phase.get("phase_type", "")) == "control_release"
            or any(
                str(control.get("joint", "")) == left_driver
                and str(control.get("mode", "")) in {"spring_return", "release"}
                for control in phase.get("controls", [])
            )
            for phase in intervening
        )

        same_link = str(left["contact_link"]) == str(right["contact_link"])
        same_sequence = _same_contact_sequence(left, right)
        if not same_link:
            if same_sequence:
                raise ValueError(
                    f"Stages {left['id']!r} and {right['id']!r} change contact_link "
                    "and therefore cannot preserve one grasp"
                )
            release_phase = left.get("release_before_phase")
            if release_phase is None or phase_order[str(release_phase)] > right_index:
                raise ValueError(
                    f"Changing contact_link between {left['id']!r} and {right['id']!r} "
                    f"requires release and retreat before phase {right['source_phase']!r}"
                )
            continue
        if not same_sequence and not plan_releases_contact:
            raise ValueError(
                "Adjacent robot-contact stages on the same contact_link must "
                "preserve one contact_sequence until a plan-declared release."
            )
    for index, stage in enumerate(stages):
        release_phase = stage.get("release_before_phase")
        if release_phase is None:
            continue
        if str(release_phase) not in phase_order:
            raise ValueError(
                f"Stage {stage['id']!r} release_before_phase {release_phase!r} is not in plan"
            )
        if phase_order[str(release_phase)] <= phase_order[str(stage["source_phase"])]:
            raise ValueError(
                f"Stage {stage['id']!r} release_before_phase must follow its source phase"
            )
        sequence = stage.get("contact_sequence")
        if sequence is not None and index != sequence_indices[str(sequence)][-1]:
            raise ValueError(
                f"Only the final stage of contact_sequence {sequence!r} may declare release_before_phase"
            )
        if stage["contact_acquisition"]["mode"] not in {
            "open_then_close",
            "maintain_width",
        }:
            raise ValueError(
                f"Stage {stage['id']!r} release_before_phase requires a supported "
                "contact acquisition mode"
            )
        if require_release_route and not stage.get("release_retreat_waypoints_world"):
            raise ValueError(
                f"Stage {stage['id']!r} release_before_phase requires a planned "
                "release_retreat_waypoints_world outside the released mechanism sweep"
            )
        if require_release_route and float(
            stage.get("minimum_release_swept_clearance_m", 0.0)
        ) <= 0.0:
            raise ValueError(
                f"Stage {stage['id']!r} release_before_phase requires strictly "
                "positive measured minimum_release_swept_clearance_m"
            )


def _validate_contact_acquisition(stage: dict[str, Any]) -> None:
    """Validate the agent-declared way contact is acquired.

    A generic harness cannot infer from an asset name whether the fingers must
    surround a handle before closing or remain pre-shaped to press a button.
    That decision is execution data; the engine only enforces its mechanics.
    """
    acquisition = stage["contact_acquisition"]
    mode = str(acquisition["mode"])
    approach = float(acquisition["approach_finger_opening_m"])
    final = float(stage["finger_opening_m"])
    close_s = float(acquisition["close_s"])
    release_s = float(acquisition["release_s"])
    tool_offset = stage.get("robot_tool_contact_offset_eef_m")
    if tool_offset is not None and (
        mode != "maintain_width" or stage["interaction"] != "physical_push"
    ):
        raise ValueError(
            f"Stage {stage['id']!r} robot_tool_contact_offset_eef_m is only "
            "valid for maintain_width physical_push"
        )
    if mode == "open_then_close":
        if stage["interaction"] != "explicit_ideal_feasibility":
            raise ValueError(
                f"Stage {stage['id']!r} open_then_close uses the invariant "
                "explicit_ideal_feasibility grasp; frictional grasp tuning is disabled"
            )
        if approach <= final:
            raise ValueError(
                f"Stage {stage['id']!r} open_then_close requires "
                "approach_finger_opening_m > finger_opening_m"
            )
        if close_s <= 0.0 or release_s <= 0.0:
            raise ValueError(
                f"Stage {stage['id']!r} open_then_close requires positive close_s and release_s"
            )
        if float(acquisition["settle_s"]) <= 0.0:
            raise ValueError(
                f"Stage {stage['id']!r} open_then_close requires positive settle_s "
                "so grasp acquisition completes before manipulation"
            )
    elif mode == "maintain_width":
        if stage["interaction"] != "physical_push":
            raise ValueError(
                f"Stage {stage['id']!r} maintain_width uses the invariant "
                "physical_push interaction"
            )
        if abs(approach - final) > 1e-9:
            raise ValueError(
                f"Stage {stage['id']!r} maintain_width requires "
                "approach_finger_opening_m == finger_opening_m"
            )
        if close_s != 0.0 or release_s != 0.0:
            raise ValueError(
                f"Stage {stage['id']!r} maintain_width requires close_s=release_s=0"
            )
    else:  # schema validation should make this unreachable
        raise ValueError(f"Unknown contact acquisition mode {mode!r}")


def _validate_effect_group(
    plan: dict[str, Any],
    phases: dict[str, dict[str, Any]],
    group: dict[str, Any],
    phase_key: str,
    effects_key: str,
) -> None:
    """Require every declared effect target to be copied from its plan phase."""
    phase_name = str(group.get(phase_key, ""))
    if phase_name not in phases:
        raise ValueError(f"Causal rule has unknown {phase_key} {phase_name!r}")
    controls = phases[phase_name].get("controls", [])
    seen_indices: set[int] = set()
    for effect in group.get(effects_key, []):
        control_index = int(effect.get("source_control_index", -1))
        if not 0 <= control_index < len(controls) or control_index in seen_indices:
            raise ValueError(
                f"Causal effect has invalid or duplicate control index {control_index} in phase {phase_name!r}"
            )
        seen_indices.add(control_index)
        control = controls[control_index]
        joint = str(effect.get("joint"))
        expected = artimo_plan.control_target(control)
        if expected is None or control.get("joint") != joint:
            raise ValueError(
                f"Causal effect {joint!r} does not match plan control {phase_name!r}[{control_index}]"
            )
        if abs(float(effect.get("target")) - expected) > 1e-6:
            raise ValueError(
                f"Causal effect target disagrees with plan control {phase_name!r}[{control_index}]"
            )


def _load_scene(
    object_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    client: int,
    visuals: bool,
) -> tuple[int, int, int | None]:
    scene = execution["scene"]
    robot = execution["robot"]
    object_body = p.loadURDF(
        str(object_urdf),
        basePosition=scene["object_base_translation_m"],
        baseOrientation=_quat(scene["object_base_rotation_xyzw"]),
        useFixedBase=True,
        flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        physicsClientId=client,
    )
    _install_static_concave_overlays(
        object_urdf, object_body, client
    )
    robot_body = p.loadURDF(
        str(robot_urdf),
        basePosition=robot["base_translation_m"],
        baseOrientation=_quat(robot["base_rotation_xyzw"]),
        useFixedBase=True,
        flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL,
        physicsClientId=client,
    )
    robot_support_body: int | None = None
    plinth = float(robot.get("base_support_height_m", 0.0))
    if plinth > 0.0:
        # The support shown in the video must be the same body used by planning
        # and physics. Derive its footprint from the grounded robot base link
        # instead of drawing a task-tuned fixed-size cube.
        base_aabb = p.getAABB(robot_body, -1, physicsClientId=client)
        support_bottom = _support_top_z(scene)
        support_top = float(base_aabb[0][2])
        support_height = max(0.0, support_top - support_bottom)
        if support_height > 1e-6:
            half_extents = [
                max(0.02, 0.5 * (float(base_aabb[1][0]) - float(base_aabb[0][0]))),
                max(0.02, 0.5 * (float(base_aabb[1][1]) - float(base_aabb[0][1]))),
                0.5 * support_height,
            ]
            center = [
                0.5 * (float(base_aabb[0][0]) + float(base_aabb[1][0])),
                0.5 * (float(base_aabb[0][1]) + float(base_aabb[1][1])),
                support_bottom + 0.5 * support_height,
            ]
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=half_extents,
                physicsClientId=client,
            )
            visual = -1
            if visuals:
                visual = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=[0.22, 0.24, 0.28, 1.0],
                    physicsClientId=client,
                )
            robot_support_body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=center,
                physicsClientId=client,
            )
    if visuals:
        floor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[2.0, 2.0, 0.01],
            rgbaColor=[0.76, 0.79, 0.83, 1.0],
            physicsClientId=client,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=floor_visual,
            basePosition=[0.0, 0.0, -0.011],
            physicsClientId=client,
        )
        support = scene.get("support_surface")
        if isinstance(support, dict):
            shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=support["half_extents_m"],
                rgbaColor=[0.28, 0.31, 0.35, 1.0],
                physicsClientId=client,
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=shape,
                basePosition=support["center_m"],
                physicsClientId=client,
            )
    return object_body, robot_body, robot_support_body


def _target_pose(
    object_body: int,
    object_link: int,
    pose: dict[str, Any],
    offset_m: float,
    client: int,
    tool_contact_offset_eef_m: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Return a Panda EEF pose from the surface-contact convention.

    Contact-frame local +Z is the outward surface normal and ``offset_m`` is a
    signed final offset: positive is a standoff into free space and negative
    moves inward to straddle protruding contact geometry. Panda ``grasptarget`` local +Z runs
    from palm to fingertips, so it must point *toward* the surface.  The fixed
    180-degree local-X frame conversion preserves the declared tangent X/roll
    while mapping EEF +Z to contact -Z.  Task data never has to encode this
    robot-model convention or guess a sign.
    """
    rotation = _quat(pose["rotation_xyzw"])
    translation = np.asarray(pose["translation_m"], dtype=np.float64)
    if offset_m:
        translation += np.asarray(
            p.rotateVector(rotation, [0.0, 0.0, offset_m]), dtype=np.float64
        )
    link_position, link_rotation = link_world_pose(object_body, object_link, client)
    position, contact_quaternion = p.multiplyTransforms(
        link_position,
        link_rotation,
        translation.tolist(),
        rotation,
        physicsClientId=client,
    )
    _, eef_quaternion = p.multiplyTransforms(
        [0.0, 0.0, 0.0],
        contact_quaternion,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        physicsClientId=client,
    )
    if tool_contact_offset_eef_m is not None:
        tool_offset_world = np.asarray(
            p.rotateVector(eef_quaternion, list(tool_contact_offset_eef_m)),
            dtype=np.float64,
        )
        position = (
            np.asarray(position, dtype=np.float64) - tool_offset_world
        ).tolist()
    return list(position), list(eef_quaternion)


def _effective_grasp_depth(stage: dict[str, Any]) -> float:
    """Map execution depth adjustment to the robot's actual contact-frame offset."""
    adjustment = float(stage.get("grasp_depth_m", 0.0))
    acquisition = stage.get("contact_acquisition", {})
    centered_parallel_jaw = (
        acquisition.get("mode") == "open_then_close"
        and stage.get("robot_tool_contact_offset_eef_m") is None
    )
    return (
        PANDA_CENTERED_GRASP_BASELINE_M + adjustment
        if centered_parallel_jaw
        else adjustment
    )


@dataclass(frozen=True)
class StagePlan:
    stage: dict[str, Any]
    approach: np.ndarray
    manipulation: np.ndarray
    retreat: np.ndarray
    object_path: np.ndarray
    maximum_position_error_m: float
    maximum_orientation_error_rad: float
    minimum_swept_clearance_m: float | None
    swept_clearance_violations: list[dict[str, Any]]
    minimum_joint_limit_margin_rad: float = math.inf
    minimum_joint_limit_margin_sample: int | None = None
    minimum_joint_limit_margin_joint: int | None = None
    maximum_adjacent_joint_step_rad: float = 0.0
    ik_backend: str = "pybullet"
    ik_backend_fallback_reason: str | None = None
    transit_planner_backend: str = "direct_interpolation"
    transit_planner_evidence: list[dict[str, Any]] = field(default_factory=list)
    debug_truncated: bool = False
    debug_failure: dict[str, Any] | None = None
    # Fully serialized joint path from the actual preceding robot state through
    # any declared world-frame obstacle-avoidance poses to approach[0].  Keeping
    # this in the plan makes planning clearance checks and rollout execute the
    # exact same transit instead of independently reconstructing a straight line.
    transit_in: np.ndarray | None = None


def _curobo_collision_obstacles(
    object_body: int,
    object_link_names: dict[int, str],
    included_link_indices: Iterable[int],
    simulation_urdf: Path,
    client: int,
) -> list[dict[str, Any]]:
    """Serialize selected PyBullet object collision shapes for cuRobo.

    The geometry stays in the current kinematic shadow-world state.  cuRobo
    consumes meshes/cuboids on the GPU; PyBullet later checks the returned path
    against the exact same selected links and remains the final authority.
    """
    obstacles: list[dict[str, Any]] = []
    for object_index in sorted(set(int(value) for value in included_link_indices)):
        if object_index == -1:
            link_position, link_rotation = p.getBasePositionAndOrientation(
                object_body, physicsClientId=client
            )
        else:
            link_position, link_rotation = link_world_pose(
                object_body, object_index, client
            )
        shapes = p.getCollisionShapeData(
            object_body, object_index, physicsClientId=client
        )
        for shape_index, shape in enumerate(shapes):
            geometry_type = int(shape[2])
            world_position, world_rotation = p.multiplyTransforms(
                link_position,
                link_rotation,
                list(shape[5]),
                list(shape[6]),
            )
            obstacle: dict[str, Any] = {
                "name": f"object_{object_index}_shape_{shape_index}",
                "position_world_m": list(world_position),
                "rotation_xyzw_world": list(world_rotation),
                "object_link": object_link_names.get(object_index, "base"),
            }
            if geometry_type == p.GEOM_MESH:
                filename = shape[4].decode() if isinstance(shape[4], bytes) else str(shape[4])
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    mesh_path = simulation_urdf.parent / mesh_path
                if not mesh_path.is_file():
                    raise FileNotFoundError(
                        f"Collision mesh reported by PyBullet does not exist: {mesh_path}"
                    )
                obstacle.update(
                    {
                        "geometry_type": "mesh",
                        "file_path": str(mesh_path.resolve()),
                        "scale": [float(component) for component in shape[3]],
                    }
                )
            elif geometry_type == p.GEOM_BOX:
                obstacle.update(
                    {
                        "geometry_type": "cuboid",
                        "dims_m": [float(component) for component in shape[3]],
                    }
                )
            else:
                lower, upper = p.getAABB(
                    object_body,
                    linkIndex=object_index,
                    physicsClientId=client,
                )
                obstacle.update(
                    {
                        "geometry_type": "cuboid",
                        "position_world_m": (
                            (np.asarray(lower) + np.asarray(upper)) * 0.5
                        ).tolist(),
                        "rotation_xyzw_world": [0.0, 0.0, 0.0, 1.0],
                        "dims_m": (np.asarray(upper) - np.asarray(lower)).tolist(),
                    }
                )
            obstacles.append(obstacle)
    return obstacles


def _swept_clearance(
    robot_body: int,
    object_body: int,
    arm: list[int],
    robot_link_names: dict[int, str],
    robot_support_body: int | None,
    forbidden: dict[str, int],
    target_contact: tuple[str, int, set[int]] | None,
    configurations: Iterable[np.ndarray],
    phase: str,
    search_distance: float,
    client: int,
) -> tuple[float | None, list[dict[str, Any]]]:
    """Measure the closest approach to forbidden links over a joint path.

    Running this at planning time is what makes a bad contact candidate cheap to
    reject: planning a stage costs a few seconds, whereas rendering a rollout to
    discover the same collision costs an order of magnitude more.
    """
    if not forbidden and target_contact is None:
        return None, []
    minimum: float | None = None
    violations: list[dict[str, Any]] = []
    for sample_index, q in enumerate(configurations):
        set_robot_arm(robot_body, arm, q, client)
        for link_name, link_index in sorted(forbidden.items()):
            points = object_closest_points(
                robot_body,
                object_body,
                search_distance,
                client,
                link_index_b=link_index,
            )
            for point in points:
                distance = float(point[8])
                if minimum is None or distance < minimum:
                    minimum = distance
                violations.append(
                    {
                        "phase": phase,
                        "sample": sample_index,
                        "robot_link": robot_link_names.get(int(point[3]), str(point[3])),
                        "object_link": link_name,
                        "distance_m": distance,
                    }
                )
            if robot_support_body is not None:
                for point in object_closest_points(
                    robot_support_body,
                    object_body,
                    search_distance,
                    client,
                    link_index_b=link_index,
                ):
                    distance = float(point[8])
                    if minimum is None or distance < minimum:
                        minimum = distance
                    violations.append(
                        {
                            "phase": phase,
                            "sample": sample_index,
                            "robot_link": "robot_support",
                            "object_link": link_name,
                            "distance_m": distance,
                        }
                    )
        if target_contact is not None:
            target_name, target_index, allowed_robot_links = target_contact
            if target_name not in forbidden:
                for point in object_closest_points(
                    robot_body,
                    object_body,
                    search_distance,
                    client,
                    link_index_b=target_index,
                ):
                    robot_link = int(point[3])
                    if robot_link in allowed_robot_links:
                        continue
                    distance = float(point[8])
                    if minimum is None or distance < minimum:
                        minimum = distance
                    violations.append(
                        {
                            "phase": phase,
                            "sample": sample_index,
                            "robot_link": robot_link_names.get(
                                robot_link, str(robot_link)
                            ),
                            "object_link": target_name,
                            "distance_m": distance,
                            "reason": "unauthorized_robot_link_on_contact_target",
                        }
                    )
                if robot_support_body is not None:
                    for point in object_closest_points(
                        robot_support_body,
                        object_body,
                        search_distance,
                        client,
                        link_index_b=target_index,
                    ):
                        distance = float(point[8])
                        if minimum is None or distance < minimum:
                            minimum = distance
                        violations.append(
                            {
                                "phase": phase,
                                "sample": sample_index,
                                "robot_link": "robot_support",
                                "object_link": target_name,
                                "distance_m": distance,
                                "reason": "unauthorized_robot_link_on_contact_target",
                            }
                        )
    # Keep only the tightest few, so debug evidence stays readable while still
    # naming which link and which path sample was too close.
    violations.sort(key=lambda item: item["distance_m"])
    return minimum, violations[:12]


def _plan_stages(
    object_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
    allow_partial_debug: bool = False,
    validate_release_clearance: bool = True,
    object_plan: dict[str, Any] | None = None,
    ik_path_solver: Any | None = None,
) -> list[StagePlan]:
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet planning client")
    try:
        object_body, robot_body, robot_support_body = _load_scene(
            object_urdf, robot_urdf, execution, client, False
        )
        object_joints, object_links = _maps(object_body, client)
        robot_joints, robot_links = _maps(robot_body, client)
        object_link_names = {index: name for name, index in object_links.items()}
        robot_link_names = {index: name for name, index in robot_links.items()}
        robot_spec = execution["robot"]
        arm = [robot_joints[name] for name in robot_spec["arm_joint_names"]]
        fingers = [robot_joints[name] for name in robot_spec["finger_joint_names"]]
        eef = robot_links[robot_spec["end_effector_link"]]
        home = np.asarray(robot_spec["home_joint_positions"], dtype=np.float64)
        if len(home) != len(arm):
            raise ValueError("home_joint_positions length differs from arm_joint_names")
        for name, value in initial.items():
            if name in object_joints:
                p.resetJointState(object_body, object_joints[name], value, physicsClientId=client)
        current = dict(initial)
        plans: list[StagePlan] = []
        reference = home.copy()
        for stage_index, stage in enumerate(execution["stages"]):
            if object_plan is not None:
                # Planning happens in a kinematic shadow world.  Project every
                # authoritative control that precedes this robot control,
                # including internal mechanisms, passive returns, and earlier
                # controls in the same phase.  Otherwise a later manipulation
                # block is scored against the initial asset geometry instead of
                # the geometry it will actually encounter at runtime.
                current = _object_joint_state_before_control(
                    object_plan,
                    initial,
                    str(stage["source_phase"]),
                    int(stage["source_control_index"]),
                )
            stage_entry_reference = reference.copy()
            continues_from_previous = (
                stage_index > 0
                and _same_contact_sequence(execution["stages"][stage_index - 1], stage)
            )
            terminal_plan_hold = bool(
                object_plan is not None
                and stage_index + 1 == len(execution["stages"])
                and _terminal_plan_hold_phase_index(
                    object_plan, execution, stage
                )
                is not None
            )
            # Planning is sequential in object state.  A later robot-owned phase
            # must see every endpoint already reached by earlier phases (for
            # example, a turned handle remains turned while the same grasp pulls
            # the door).  Resetting only the current driver would erase that
            # cross-joint state and plan the wrong contact trajectory.
            for name, value in current.items():
                if name in object_joints:
                    p.resetJointState(object_body, object_joints[name], float(value), physicsClientId=client)
            driver_name = stage["driver_joint"]
            if driver_name not in object_joints or stage["contact_link"] not in object_links:
                raise KeyError(f"Unknown driver/contact in stage {stage['id']}")
            driver = object_joints[driver_name]
            contact_link = object_links[stage["contact_link"]]
            allowed_target_robot_links = {
                robot_links[name]
                for name in stage["allowed_robot_contact_links"]
            }
            target_contact = (
                str(stage["contact_link"]),
                contact_link,
                allowed_target_robot_links,
            )
            target_contact_during_transit = (
                str(stage["contact_link"]),
                contact_link,
                set(),
            )
            required_clearance = stage.get("minimum_swept_clearance_m")
            forbidden = {
                name: object_links[name]
                for name in stage.get("forbidden_contact_links", [])
            }
            unknown = set(stage.get("forbidden_contact_links", [])) - set(object_links)
            if unknown:
                raise KeyError(
                    f"Stage {stage['id']} names unknown forbidden_contact_links: {sorted(unknown)}"
                )
            search_distance = float(required_clearance) if required_clearance else 0.0
            acquisition = stage["contact_acquisition"]
            approach_opening = float(acquisition["approach_finger_opening_m"])
            start = float(current.get(driver_name, p.getJointState(object_body, driver, physicsClientId=client)[0]))
            target = float(stage.get("command_joint_position", stage["target_joint_position"]))
            # Dense Cartesian tracking needs enough object-side samples that a
            # valid Panda branch can remain inside the fixed 0.08 rad local IK
            # trust region through high-curvature/singular portions of a link
            # sweep.  Endpoints and the smoothstep object path are unchanged;
            # the extra samples only halve the maximum object-side increment.
            # The dense execution path must remain inside the 0.08 rad local
            # trust region even near high-curvature/singular portions of an
            # articulated contact sweep. 257 object samples halve the step of
            # the prior 129-point proof without weakening that invariant.
            u = np.linspace(0.0, 1.0, DENSE_MANIPULATION_PATH_SAMPLES)
            object_path = start + (target - start) * _smoothstep01(u)
            opening = float(stage["finger_opening_m"])
            set_fingers(robot_body, fingers, opening, client)
            # The IK budget is execution data: a contact pose near the arm's
            # workspace boundary converges only intermittently under a small
            # restart count, and that is a property of the task's geometry rather
            # than something the harness can pick for every asset.
            ik_settings = execution.get("ik", {})
            solver = BulletIK(
                robot_body,
                arm,
                eef,
                fingers,
                opening,
                {
                    "random_seed": int(execution["seeds"].get("ik", 0)) + stage_index,
                    "random_restarts": int(ik_settings.get("random_restarts", 48)),
                    "max_iterations": int(ik_settings.get("max_iterations", 1800)),
                    # Residual normalization only; none is an accept/reject
                    # threshold for the dense trajectory.
                    "position_tolerance_m": 0.001,
                    "orientation_tolerance_deg": 1.0,
                    # Fixed local trust-region radius for adjacent dense path
                    # samples.  This is a trajectory construction invariant,
                    # not an agent-tunable pass/fail threshold.
                    "max_joint_step_rad": 0.08,
                },
                client,
            )
            debug_failure: dict[str, Any] | None = None
            # Grasp depth is the final manipulation standoff, not approach
            # clearance.  Its value must be justified by measured finger-to-target
            # gaps; precontact_offset_m is the separate transient approach distance.
            grasp_depth = _effective_grasp_depth(stage)
            curobo_path: list[list[float]] | None = None
            curobo_evidence: dict[str, Any] | None = None
            curobo_fallback_reason: str | None = None
            if ik_path_solver is not None:
                dense_target_poses: list[tuple[list[float], list[float]]] = []
                dense_obstacle_worlds: list[list[dict[str, Any]]] = []
                for value in object_path:
                    p.resetJointState(
                        object_body, driver, float(value), physicsClientId=client
                    )
                    dense_target_poses.append(_target_pose(
                        object_body,
                        contact_link,
                        stage["contact_pose_link"],
                        grasp_depth,
                        client,
                        stage.get("robot_tool_contact_offset_eef_m"),
                    ))
                    if ik_path_solver.environment_collision:
                        dense_obstacle_worlds.append(_curobo_collision_obstacles(
                            object_body,
                            {index: name for name, index in object_links.items()},
                            forbidden.values(),
                            object_urdf,
                            client,
                        ))
                try:
                    curobo_evidence = ik_path_solver.solve_path(
                        [pose[0] for pose in dense_target_poses],
                        [pose[1] for pose in dense_target_poses],
                        execution["robot"]["base_translation_m"],
                        execution["robot"]["base_rotation_xyzw"],
                        reference,
                        0.08,
                        continues_from_previous,
                        dense_obstacle_worlds,
                        sequential=True,
                    )
                    if curobo_evidence.get("success"):
                        curobo_path = curobo_evidence["path"]
                    else:
                        curobo_fallback_reason = (
                            "no_continuous_gpu_branch_at_sample_"
                            f"{curobo_evidence.get('failed_sample')}"
                        )
                except Exception as exc:
                    curobo_fallback_reason = f"curobo_worker_error: {exc}"
                # A redundant-arm singularity can leave every pose individually
                # valid while cuRobo's returned seed set contains no adjacent
                # branch inside 0.08 rad. In that specific case the sequential
                # Bullet IK below may construct the kinematic path, but source-
                # mesh collision remains GPU-authoritative afterward. This is
                # not a collision fallback.

            def trace_manipulation_branch(
                first_answer: dict[str, Any] | None,
                precomputed_path: list[list[float]] | None = None,
            ) -> dict[str, Any]:
                branch_reference = reference.copy()
                branch_path: list[np.ndarray] = []
                branch_failures: list[dict[str, Any]] = []
                branch_max_position = 0.0
                branch_max_rotation = 0.0
                branch_minimum_clearance: float | None = None
                for sample_index, value in enumerate(object_path):
                    p.resetJointState(
                        object_body, driver, float(value), physicsClientId=client
                    )
                    target_position, target_rotation = _target_pose(
                        object_body,
                        contact_link,
                        stage["contact_pose_link"],
                        grasp_depth,
                        client,
                        stage.get("robot_tool_contact_offset_eef_m"),
                    )
                    try:
                        if precomputed_path is not None:
                            q = np.asarray(precomputed_path[sample_index], dtype=np.float64)
                            set_robot_arm(robot_body, arm, q, client)
                            actual_position, actual_rotation = link_world_pose(
                                robot_body, eef, client
                            )
                            answer = {
                                "success": True,
                                "q": q,
                                "position_error_m": float(np.linalg.norm(
                                    np.asarray(actual_position)
                                    - np.asarray(target_position)
                                )),
                                "orientation_error_rad": quat_angle_rad(
                                    actual_rotation, target_rotation
                                ),
                                "solver": "curobo_batch_ik_pybullet_verified",
                            }
                        elif sample_index == 0 and first_answer is not None:
                            answer = dict(first_answer)
                            answer.setdefault("success", True)
                            answer.setdefault(
                                "solver", "collision_ranked_global_branch"
                            )
                        else:
                            answer = solver.solve_continuous(
                                target_position,
                                target_rotation,
                                branch_reference,
                            )
                    except RuntimeError as exc:
                        answer = {
                            "success": False,
                            "q": branch_reference.copy(),
                            "position_error_m": 1.0e9,
                            "orientation_error_rad": math.pi,
                            "error": str(exc),
                        }
                    if not answer["success"]:
                        branch_failures.append(
                            {
                                "segment": "manipulation",
                                "sample": int(sample_index),
                                "position_error_m": float(
                                    answer["position_error_m"]
                                ),
                                "orientation_error_deg": math.degrees(
                                    float(answer["orientation_error_rad"])
                                ),
                                "max_joint_step_rad": float(
                                    answer.get("max_joint_step_rad", 0.0)
                                ),
                                "joint_l2_step_rad": float(
                                    answer.get("joint_l2_step_rad", 0.0)
                                ),
                                "solver": answer.get("solver"),
                                "continuous_refinement": answer.get(
                                    "continuous_refinement"
                                ),
                                "solver_error": answer.get("error"),
                            }
                        )
                    else:
                        branch_reference = np.asarray(
                            answer["q"], dtype=np.float64
                        )
                    branch_path.append(branch_reference.copy())
                    branch_max_position = max(
                        branch_max_position,
                        float(answer["position_error_m"]),
                    )
                    branch_max_rotation = max(
                        branch_max_rotation,
                        float(answer["orientation_error_rad"]),
                    )
                    sample_minimum, _ = _swept_clearance(
                        robot_body,
                        object_body,
                        arm,
                        robot_link_names,
                        robot_support_body,
                        (
                            {}
                            if ik_path_solver is not None
                            and ik_path_solver.environment_collision
                            else forbidden
                        ),
                        target_contact,
                        [branch_reference],
                        "manipulation_branch",
                        search_distance,
                        client,
                    )
                    if sample_minimum is not None and (
                        branch_minimum_clearance is None
                        or sample_minimum < branch_minimum_clearance
                    ):
                        branch_minimum_clearance = float(sample_minimum)
                branch_array = np.asarray(branch_path, dtype=np.float64)
                margins = np.minimum(
                    branch_array - solver.arm_lower,
                    solver.arm_upper - branch_array,
                )
                steps = np.abs(np.diff(branch_array, axis=0))
                minimum_margin = float(np.min(margins))
                maximum_step = float(np.max(steps)) if steps.size else 0.0
                ik_passed = (
                    not branch_failures
                    and branch_max_position <= 0.004
                    and branch_max_rotation <= math.radians(2.0)
                    and minimum_margin > 1.0e-4
                    and maximum_step <= 0.0800001
                )
                clearance_passed = (
                    branch_minimum_clearance is None
                    or branch_minimum_clearance
                    >= float(required_clearance or 0.0)
                )
                return {
                    "path": branch_path,
                    "failures": branch_failures,
                    "maximum_position_error_m": branch_max_position,
                    "maximum_orientation_error_rad": branch_max_rotation,
                    "minimum_joint_limit_margin_rad": minimum_margin,
                    "maximum_adjacent_joint_step_rad": maximum_step,
                    "minimum_swept_clearance_m": branch_minimum_clearance,
                    "ik_passed": ik_passed,
                    "clearance_passed": clearance_passed,
                }

            branch_traces: list[dict[str, Any]] = []
            if curobo_path is not None:
                branch_traces.append(trace_manipulation_branch(None, curobo_path))
            elif continues_from_previous:
                # The preceding stage already ends at this exact link-relative
                # contact pose and object state. Re-solving sample zero lets a
                # redundant arm move to another nearby IK solution even though
                # the grasp did not change, which appears as a twitch at the
                # phase boundary. Inherit the exact final joint vector instead;
                # continuous IK resumes only once the contacted link moves.
                branch_traces.append(
                    trace_manipulation_branch(
                        _inherited_contact_sequence_answer(reference)
                    )
                )
            else:
                p.resetJointState(
                    object_body, driver, float(object_path[0]), physicsClientId=client
                )
                first_position, first_rotation = _target_pose(
                    object_body,
                    contact_link,
                    stage["contact_pose_link"],
                    grasp_depth,
                    client,
                    stage.get("robot_tool_contact_offset_eef_m"),
                )
                first_candidates = [
                    candidate
                    for candidate in solver.solve_candidates(
                        first_position, first_rotation, reference
                    )
                    if float(candidate["position_error_m"]) <= 0.004
                    and float(candidate["orientation_error_rad"])
                    <= math.radians(2.0)
                    and float(candidate["minimum_joint_limit_margin_rad"])
                    > 1.0e-4
                ]
                if not first_candidates:
                    first_candidates = [
                        solver.solve(
                            first_position,
                            first_rotation,
                            reference,
                            enforce_step=False,
                        )
                    ]
                for candidate in first_candidates:
                    branch_traces.append(trace_manipulation_branch(candidate))

            def manipulation_branch_rank(item: dict[str, Any]) -> tuple[Any, ...]:
                clearance = item["minimum_swept_clearance_m"]
                numeric_clearance = math.inf if clearance is None else float(clearance)
                return (
                    not (item["ik_passed"] and item["clearance_passed"]),
                    not item["ik_passed"],
                    -numeric_clearance,
                    float(item["maximum_position_error_m"]),
                    float(item["maximum_orientation_error_rad"]),
                    -float(item["minimum_joint_limit_margin_rad"]),
                    float(item["maximum_adjacent_joint_step_rad"]),
                )

            chosen_branch = min(branch_traces, key=manipulation_branch_rank)
            manipulation_gpu_evidence: dict[str, Any] | None = curobo_evidence
            if (
                curobo_path is None
                and ik_path_solver is not None
                and ik_path_solver.environment_collision
                and chosen_branch["path"]
            ):
                manipulation_gpu_evidence = ik_path_solver.check_joint_path(
                    chosen_branch["path"],
                    execution["robot"]["base_translation_m"],
                    execution["robot"]["base_rotation_xyzw"],
                    dense_obstacle_worlds,
                    required_clearance_m=float(required_clearance or 0.0),
                )
                gpu_minimum = manipulation_gpu_evidence.get(
                    "minimum_environment_clearance_m"
                )
                if gpu_minimum is not None:
                    existing_minimum = chosen_branch["minimum_swept_clearance_m"]
                    chosen_branch["minimum_swept_clearance_m"] = (
                        float(gpu_minimum)
                        if existing_minimum is None
                        else min(float(existing_minimum), float(gpu_minimum))
                    )
                chosen_branch["clearance_passed"] = bool(
                    chosen_branch["clearance_passed"]
                    and manipulation_gpu_evidence.get("success")
                )

            def exact_path_clearance(
                configurations: Iterable[np.ndarray],
                included_object_links: Iterable[int],
                target_semantics: tuple[str, int, set[int]] | None,
                phase: str,
                obstacle_worlds: list[list[dict[str, Any]]] | None = None,
            ) -> tuple[float | None, list[dict[str, Any]], dict[str, Any] | None]:
                """Use GPU screening plus exact PyBullet path authority."""
                rows = [np.asarray(row, dtype=np.float64) for row in configurations]
                included_indices = [int(value) for value in included_object_links]
                if not rows:
                    return None, [], None
                if ik_path_solver is None or not ik_path_solver.environment_collision:
                    minimum, found = _swept_clearance(
                        robot_body,
                        object_body,
                        arm,
                        robot_link_names,
                        robot_support_body,
                        {
                            object_link_names[index]: index
                            for index in included_indices
                        },
                        target_semantics,
                        rows,
                        phase,
                        search_distance,
                        client,
                    )
                    return minimum, found, None
                if obstacle_worlds is None:
                    one_world = _curobo_collision_obstacles(
                        object_body,
                        object_link_names,
                        included_indices,
                        object_urdf,
                        client,
                    )
                    obstacle_worlds = [one_world for _ in rows]
                evidence = ik_path_solver.check_joint_path(
                    rows,
                    execution["robot"]["base_translation_m"],
                    execution["robot"]["base_rotation_xyzw"],
                    obstacle_worlds,
                    required_clearance_m=float(required_clearance or 0.0),
                )
                minimum = evidence.get("minimum_environment_clearance_m")
                found: list[dict[str, Any]] = []
                if not evidence.get("success"):
                    found.append({
                        "phase": phase,
                        "sample": evidence.get("failed_sample"),
                        "robot_link": "curobo_collision_spheres",
                        "object_link": "source_mesh_environment",
                        "distance_m": float(minimum or 0.0),
                        "reason": "gpu_source_mesh_collision",
                    })
                # GPU source-mesh checking uses the configured robot collision
                # spheres and remains the high-throughput planner screen.  The
                # exact PyBullet robot geometry is still the final dense/path
                # authority because sphere models may under-cover a real link.
                bullet_minimum, bullet_found = _swept_clearance(
                    robot_body,
                    object_body,
                    arm,
                    robot_link_names,
                    robot_support_body,
                    {
                        object_link_names[index]: index
                        for index in included_indices
                    },
                    target_semantics,
                    rows,
                    phase + "_pybullet_exact",
                    search_distance,
                    client,
                )
                if bullet_minimum is not None:
                    minimum = (
                        float(bullet_minimum)
                        if minimum is None
                        else min(float(minimum), float(bullet_minimum))
                    )
                found.extend(bullet_found)
                return (
                    None if minimum is None else float(minimum),
                    found,
                    evidence,
                )
            manipulation = chosen_branch["path"]
            ik_failures = chosen_branch["failures"]
            max_position = float(chosen_branch["maximum_position_error_m"])
            max_rotation = float(chosen_branch["maximum_orientation_error_rad"])
            reference = manipulation[-1].copy()
            continuation_reference = manipulation[-1].copy()
            for name, value in current.items():
                if name in object_joints:
                    p.resetJointState(object_body, object_joints[name], float(value), physicsClientId=client)
            p.resetJointState(object_body, driver, start, physicsClientId=client)
            reverse = [manipulation[0].copy()]
            reference = manipulation[0].copy()
            # Contact +Z is outward. Retreat continues farther along +Z from
            # the grasp pose, so all offsets remain non-negative distances.
            for offset in grasp_depth + np.linspace(
                0.0, float(stage["precontact_offset_m"]), 17
            )[1:]:
                position, rotation = _target_pose(
                    object_body,
                    contact_link,
                    stage["contact_pose_link"],
                    float(offset),
                    client,
                    stage.get("robot_tool_contact_offset_eef_m"),
                )
                answer = solver.solve_continuous(position, rotation, reference)
                if not answer["success"]:
                    ik_failures.append(
                        {
                            "segment": "precontact",
                            "offset_m": float(offset),
                            "position_error_m": float(answer["position_error_m"]),
                            "orientation_error_deg": math.degrees(
                                float(answer["orientation_error_rad"])
                            ),
                        }
                    )
                    reference = reference.copy()
                else:
                    reference = np.asarray(answer["q"], dtype=np.float64)
                reverse.append(reference.copy())
                max_position = max(max_position, float(answer["position_error_m"]))
                max_rotation = max(max_rotation, float(answer["orientation_error_rad"]))
            # Optional via-points continue outward from the precontact pose, so
            # the wrist can be routed around an intervening link rather than
            # driven straight at it.
            for waypoint_index, waypoint in enumerate(
                stage.get("approach_waypoints_link_m", [])
            ):
                pose = {
                    "translation_m": list(waypoint),
                    "rotation_xyzw": stage["contact_pose_link"]["rotation_xyzw"],
                }
                position, rotation = _target_pose(
                    object_body,
                    contact_link,
                    pose,
                    0.0,
                    client,
                    stage.get("robot_tool_contact_offset_eef_m"),
                )
                answer = solver.solve_continuous(position, rotation, reference)
                if not answer["success"]:
                    ik_failures.append(
                        {
                            "segment": "approach_waypoint",
                            "waypoint": int(waypoint_index),
                            "position_error_m": float(answer["position_error_m"]),
                            "orientation_error_deg": math.degrees(
                                float(answer["orientation_error_rad"])
                            ),
                        }
                    )
                    reference = reference.copy()
                else:
                    reference = np.asarray(answer["q"], dtype=np.float64)
                reverse.append(reference.copy())
                max_position = max(max_position, float(answer["position_error_m"]))
                max_rotation = max(max_rotation, float(answer["orientation_error_rad"]))

            approach = np.asarray(list(reversed(reverse)))
            if continues_from_previous:
                # This stage begins with the exact grasp and arm command left
                # by the preceding stage in the same contact sequence. There is
                # no second approach, open, close, or reacquisition to plan or
                # collision-check.
                approach = np.asarray([manipulation[0].copy()])
            # Retreat is not the reverse of approach when the contacted object
            # link has moved.  Solve it in the stage's final object state so the
            # arm leaves the actual final contact point rather than snapping
            # toward the link's initial pose.
            for name, value in current.items():
                if name in object_joints:
                    p.resetJointState(object_body, object_joints[name], float(value), physicsClientId=client)
            p.resetJointState(object_body, driver, target, physicsClientId=client)
            retreat_configurations = [manipulation[-1].copy()]
            reference = manipulation[-1].copy()
            declared_release_route = list(
                stage.get("release_retreat_waypoints_world", [])
            )
            # A solver-authored release route starts at the exact grasp-release
            # joint vector. Do not first replay the generic link-normal retreat:
            # after a door/panel has moved, that old local direction can sweep
            # the forearm back through the moved geometry before reaching the
            # safe world-frame waypoint.
            if not declared_release_route:
                for offset in grasp_depth + np.linspace(
                    0.0, float(stage["precontact_offset_m"]), 17
                )[1:]:
                    position, rotation = _target_pose(
                        object_body,
                        contact_link,
                        stage["contact_pose_link"],
                        float(offset),
                        client,
                        stage.get("robot_tool_contact_offset_eef_m"),
                    )
                    answer = solver.solve_continuous(position, rotation, reference)
                    if not answer["success"]:
                        ik_failures.append(
                            {
                                "segment": "retreat",
                                "offset_m": float(offset),
                                "position_error_m": float(answer["position_error_m"]),
                                "orientation_error_deg": math.degrees(
                                    float(answer["orientation_error_rad"])
                                ),
                            }
                        )
                        reference = reference.copy()
                    else:
                        reference = np.asarray(answer["q"], dtype=np.float64)
                    retreat_configurations.append(reference.copy())
                    max_position = max(max_position, float(answer["position_error_m"]))
                    max_rotation = max(max_rotation, float(answer["orientation_error_rad"]))
                for waypoint_index, waypoint in enumerate(
                    stage.get("approach_waypoints_link_m", [])
                ):
                    pose = {
                        "translation_m": list(waypoint),
                        "rotation_xyzw": stage["contact_pose_link"]["rotation_xyzw"],
                    }
                    position, rotation = _target_pose(
                        object_body,
                        contact_link,
                        pose,
                        0.0,
                        client,
                        stage.get("robot_tool_contact_offset_eef_m"),
                    )
                    answer = solver.solve_continuous(position, rotation, reference)
                    if not answer["success"]:
                        ik_failures.append(
                            {
                                "segment": "retreat_waypoint",
                                "waypoint_index": int(waypoint_index),
                                "position_error_m": float(answer["position_error_m"]),
                                "orientation_error_deg": math.degrees(
                                    float(answer["orientation_error_rad"])
                                ),
                            }
                        )
                        reference = reference.copy()
                    else:
                        reference = np.asarray(answer["q"], dtype=np.float64)
                    retreat_configurations.append(reference.copy())
                    max_position = max(max_position, float(answer["position_error_m"]))
                    max_rotation = max(max_rotation, float(answer["orientation_error_rad"]))
            release_waypoints = (
                declared_release_route
                if validate_release_clearance
                else []
            )
            for waypoint_index, pose in enumerate(release_waypoints):
                set_robot_arm(robot_body, arm, reference, client)
                start_position, start_rotation = link_world_pose(
                    robot_body, eef, client
                )
                target_position = np.asarray(
                    pose["translation_m"], dtype=np.float64
                )
                target_rotation = list(pose["rotation_xyzw"])
                distance = float(
                    np.linalg.norm(target_position - np.asarray(start_position))
                )
                sample_count = max(2, int(math.ceil(distance / 0.01)) + 1)
                for path_sample, alpha in enumerate(
                    np.linspace(0.0, 1.0, sample_count)[1:], start=1
                ):
                    position = (
                        (1.0 - alpha) * np.asarray(start_position)
                        + alpha * target_position
                    )
                    rotation = p.getQuaternionSlerp(
                        start_rotation,
                        target_rotation,
                        float(alpha),
                    )
                    answer = solver.solve_continuous(
                        position.tolist(), list(rotation), reference
                    )
                    if not answer["success"]:
                        ik_failures.append(
                            {
                                "segment": "release_retreat_waypoint",
                                "waypoint_index": int(waypoint_index),
                                "path_sample": int(path_sample),
                                "position_error_m": float(answer["position_error_m"]),
                                "orientation_error_deg": math.degrees(
                                    float(answer["orientation_error_rad"])
                                ),
                            }
                        )
                        reference = reference.copy()
                    else:
                        reference = np.asarray(answer["q"], dtype=np.float64)
                    retreat_configurations.append(reference.copy())
                    max_position = max(
                        max_position, float(answer["position_error_m"])
                    )
                    max_rotation = max(
                        max_rotation, float(answer["orientation_error_rad"])
                    )
            retreat = np.asarray(retreat_configurations)
            continues_to_next = bool(
                stage_index + 1 < len(execution["stages"])
                and _same_contact_sequence(
                    stage, execution["stages"][stage_index + 1]
                )
            )
            if continues_to_next or terminal_plan_hold:
                # Preserve the final grasp command into the next same-sequence
                # manipulation stage. A generated withdrawal here is a
                # manufactured release and must not participate in planning.
                # The same rule applies when the authoritative plan ends in a
                # hold_position: rollout keeps the disclosed grasp through the
                # terminal hold, so planning must not invent a release,
                # withdrawal, or return-home path after that endpoint.
                retreat = np.asarray([manipulation[-1].copy()])
            # A contact change may need to route around geometry moved by an
            # earlier plan stage (for example an opened panel).  These waypoints
            # belong only to the incoming transit of this stage: unlike
            # approach_waypoints_link_m they must not be replayed on retreat.
            transit_waypoints = list(stage.get("transit_waypoints_world", []))
            if continues_from_previous and transit_waypoints:
                raise ValueError(
                    f"Stage {stage['id']} continues contact_sequence but declares "
                    "transit_waypoints_world"
                )
            transit_in: np.ndarray | None = None
            transit_planner_backend = "direct_interpolation"
            transit_planner_evidence: list[dict[str, Any]] = []
            if not continues_from_previous:
                for name, value in current.items():
                    if name in object_joints:
                        p.resetJointState(
                            object_body,
                            object_joints[name],
                            float(value),
                            physicsClientId=client,
                        )
                p.resetJointState(
                    object_body, driver, start, physicsClientId=client
                )
                transit_gpu_obstacles = (
                    _curobo_collision_obstacles(
                        object_body,
                        {index: name for name, index in object_links.items()},
                        [*forbidden.values(), contact_link],
                        object_urdf,
                        client,
                    )
                    if ik_path_solver is not None
                    and ik_path_solver.environment_collision
                    else []
                )
                anchors = [stage_entry_reference.copy()]
                waypoint_reference = stage_entry_reference.copy()
                for waypoint_index, pose in enumerate(transit_waypoints):
                    target_position = list(pose["translation_m"])
                    target_rotation = list(pose["rotation_xyzw"])
                    waypoint_candidates = [
                        candidate
                        for candidate in solver.solve_candidates(
                            target_position,
                            target_rotation,
                            waypoint_reference,
                        )
                        if float(candidate["position_error_m"]) <= 0.004
                        and float(candidate["orientation_error_rad"])
                        <= math.radians(2.0)
                        and float(candidate["minimum_joint_limit_margin_rad"])
                        > 1.0e-4
                    ]
                    set_fingers(robot_body, fingers, approach_opening, client)
                    ranked_waypoints: list[
                        tuple[float, float, float, dict[str, Any]]
                    ] = []
                    for candidate in waypoint_candidates:
                        candidate_q = np.asarray(candidate["q"], dtype=np.float64)
                        scored_path = _interpolate(
                            waypoint_reference, candidate_q, 90
                        )
                        if waypoint_index == len(transit_waypoints) - 1:
                            scored_path.extend(
                                _interpolate(candidate_q, approach[0], 90)[1:]
                            )
                        waypoint_minimum, _ = _swept_clearance(
                            robot_body,
                            object_body,
                            arm,
                            robot_link_names,
                            robot_support_body,
                            forbidden,
                            target_contact_during_transit,
                            scored_path,
                            "transit_waypoint_branch",
                            search_distance,
                            client,
                        )
                        ranked_waypoints.append(
                            (
                                math.inf
                                if waypoint_minimum is None
                                else float(waypoint_minimum),
                                float(candidate["error_score"]),
                                float(candidate["joint_l2_step_rad"]),
                                candidate,
                            )
                        )
                    if ranked_waypoints:
                        answer = min(
                            ranked_waypoints,
                            key=lambda item: (-item[0], item[1], item[2]),
                        )[3]
                        answer["success"] = True
                        answer["pose_residual_only"] = True
                        answer["solver"] = "collision_ranked_transit_branch"
                    else:
                        answer = solver.solve(
                            target_position,
                            target_rotation,
                            waypoint_reference,
                            enforce_step=False,
                        )
                    max_position = max(
                        max_position, float(answer["position_error_m"])
                    )
                    max_rotation = max(
                        max_rotation, float(answer["orientation_error_rad"])
                    )
                    if not answer["success"]:
                        ik_failures.append(
                            {
                                "segment": "transit_waypoint",
                                "waypoint_index": int(waypoint_index),
                                "position_error_m": float(
                                    answer["position_error_m"]
                                ),
                                "orientation_error_deg": math.degrees(
                                    float(answer["orientation_error_rad"])
                                ),
                                "solver_error": answer.get("error"),
                            }
                        )
                        continue
                    waypoint_reference = np.asarray(
                        answer["q"], dtype=np.float64
                    )
                    anchors.append(waypoint_reference.copy())
                anchors.append(approach[0].copy())
                transit_configurations: list[np.ndarray] = []
                for anchor_index, (left, right) in enumerate(
                    zip(anchors, anchors[1:])
                ):
                    segment = _interpolate(left, right, 90)
                    direct_minimum, direct_violations, direct_gpu = exact_path_clearance(
                        segment,
                        [*forbidden.values(), contact_link],
                        None,
                        "transit_direct_probe",
                    )
                    direct_clear = (
                        not direct_violations
                        and (
                            direct_minimum is None
                            or direct_minimum >= float(required_clearance or 0.0)
                        )
                    )
                    if not direct_clear:
                        planned: list[np.ndarray] | None = None
                        gpu_evidence: dict[str, Any] | None = None
                        if ik_path_solver is not None:
                            try:
                                gpu_evidence = ik_path_solver.plan_joint_path(
                                    left,
                                    right,
                                    execution["robot"]["base_translation_m"],
                                    execution["robot"]["base_rotation_xyzw"],
                                    transit_gpu_obstacles,
                                    maximum_joint_step_rad=0.08,
                                    required_clearance_m=float(required_clearance or 0.0),
                                )
                                if gpu_evidence.get("success"):
                                    gpu_candidate = [
                                        np.asarray(row, dtype=np.float64)
                                        for row in gpu_evidence["path"]
                                    ]
                                    gpu_minimum, gpu_violations, gpu_verify = exact_path_clearance(
                                        gpu_candidate,
                                        [*forbidden.values(), contact_link],
                                        None,
                                        "transit_curobo_pybullet_verify",
                                    )
                                    gpu_passed = bool(
                                        not gpu_violations
                                        and (
                                            gpu_minimum is None
                                            or gpu_minimum
                                            >= float(required_clearance or 0.0)
                                        )
                                    )
                                    gpu_evidence["pybullet_verified"] = gpu_passed
                                    gpu_evidence["pybullet_minimum_clearance_m"] = gpu_minimum
                                    gpu_evidence["source_mesh_gpu_verify"] = gpu_verify
                                    if gpu_passed:
                                        planned = gpu_candidate
                                        transit_planner_backend = (
                                            "curobo_motion_gen_gpu_pybullet_verified"
                                        )
                            except Exception as exc:
                                gpu_evidence = {
                                    "success": False,
                                    "backend": "curobo_motion_gen_gpu",
                                    "error": str(exc)[:2000],
                                }
                            transit_planner_evidence.append(
                                {
                                    "anchor_index": int(anchor_index),
                                    **(gpu_evidence or {}),
                                }
                            )

                        # CPU RRT is compatibility-only. Selecting cuRobo with
                        # allow_bullet_fallback=false makes a GPU failure visible
                        # instead of silently moving the expensive search to CPU.
                        if planned is None and (
                            ik_path_solver is None
                            or ik_path_solver.allow_bullet_fallback
                        ):
                            def configuration_is_clear(q: np.ndarray) -> bool:
                                probe_minimum, probe_violations = _swept_clearance(
                                    robot_body,
                                    object_body,
                                    arm,
                                    robot_link_names,
                                    robot_support_body,
                                    forbidden,
                                    target_contact_during_transit,
                                    [q],
                                    "transit_rrt_probe",
                                    search_distance,
                                    client,
                                )
                                return bool(
                                    not probe_violations
                                    and (
                                        probe_minimum is None
                                        or probe_minimum
                                        >= float(required_clearance or 0.0)
                                    )
                                )

                            planned = _rrt_connect(
                                left,
                                right,
                                solver.arm_lower,
                                solver.arm_upper,
                                configuration_is_clear,
                                np.random.default_rng(
                                    int(execution["seeds"].get("search", 0))
                                    + 1009 * stage_index
                                    + anchor_index
                                ),
                            )
                            if planned is not None:
                                transit_planner_backend = "pybullet_cpu_rrt_fallback"
                        if planned is not None:
                            segment = planned
                    if anchor_index:
                        segment = segment[1:]
                    transit_configurations.extend(segment)
                transit_in = np.asarray(
                    transit_configurations, dtype=np.float64
                )
            p.resetJointState(object_body, driver, start, physicsClientId=client)
            set_fingers(robot_body, fingers, approach_opening, client)
            minimum_clearance, violations, approach_gpu = exact_path_clearance(
                list(approach),
                forbidden.values(),
                target_contact,
                "approach",
            )
            # A stage begins at the actual preceding robot state.  The first
            # stage starts at home; a later, different contact starts at the
            # preceding stage's released retreat endpoint.  Returning to home
            # between contacts manufactures two unnecessary transits and can
            # drive the arm back through an already moved door or lid.
            if not continues_from_previous:
                transit_minimum, transit_violations, transit_gpu = exact_path_clearance(
                    list(transit_in),
                    [*forbidden.values(), contact_link],
                    None,
                    "transit_in",
                )
                if transit_minimum is not None and (
                    minimum_clearance is None or transit_minimum < minimum_clearance
                ):
                    minimum_clearance = transit_minimum
                violations.extend(transit_violations)
            # The manipulation path is swept with the object at the states it will
            # actually pass through, since a forbidden link can move too.
            set_fingers(robot_body, fingers, opening, client)
            manipulation_gpu_minimum = (
                None
                if manipulation_gpu_evidence is None
                else manipulation_gpu_evidence.get(
                    "minimum_environment_clearance_m"
                )
            )
            if manipulation_gpu_minimum is not None and (
                minimum_clearance is None
                or float(manipulation_gpu_minimum) < minimum_clearance
            ):
                minimum_clearance = float(manipulation_gpu_minimum)
            if (
                manipulation_gpu_evidence is not None
                and not manipulation_gpu_evidence.get("success")
            ):
                violations.append({
                    "phase": "manipulate",
                    "sample": manipulation_gpu_evidence.get("failed_sample"),
                    "robot_link": "curobo_collision_spheres",
                    "object_link": "source_mesh_environment",
                    "distance_m": float(manipulation_gpu_minimum or 0.0),
                    "reason": "gpu_source_mesh_collision",
                })
            for sample_index, value in enumerate(object_path):
                p.resetJointState(object_body, driver, float(value), physicsClientId=client)
                sample_minimum, sample_violations = _swept_clearance(
                    robot_body,
                    object_body,
                    arm,
                    robot_link_names,
                    robot_support_body,
                    forbidden,
                    target_contact,
                    [manipulation[sample_index]],
                    "manipulate",
                    search_distance,
                    client,
                )
                if sample_minimum is not None and (
                    minimum_clearance is None or sample_minimum < minimum_clearance
                ):
                    minimum_clearance = sample_minimum
                violations.extend(sample_violations)
            for name, value in current.items():
                if name in object_joints:
                    p.resetJointState(object_body, object_joints[name], float(value), physicsClientId=client)
            p.resetJointState(object_body, driver, target, physicsClientId=client)
            retreat_opening = (
                approach_opening
                if acquisition["mode"] == "open_then_close"
                else opening
            )
            set_fingers(robot_body, fingers, retreat_opening, client)
            retreat_minimum, retreat_violations, retreat_gpu = exact_path_clearance(
                list(retreat),
                forbidden.values(),
                target_contact,
                "retreat",
            )
            if retreat_minimum is not None and (
                minimum_clearance is None or retreat_minimum < minimum_clearance
            ):
                minimum_clearance = retreat_minimum
            violations.extend(retreat_violations)
            release_required = stage.get("minimum_release_swept_clearance_m")
            release_clearance_pending = bool(
                not validate_release_clearance
                and stage.get("release_before_phase") is not None
            )
            if validate_release_clearance and release_required is not None:
                # The serialized value is measured evidence, not a requested
                # comfort margin. Strict validation requires only positive
                # separation over the complete release and later object sweep.
                release_required = 0.0
                if len(retreat) < 2:
                    ik_failures.append(
                        {
                            "segment": "release_clearance",
                            "reason": "no_solved_clearance_waypoint",
                        }
                    )
                if object_plan is None:
                    raise ValueError(
                        "Strict release-clearance validation requires the "
                        "authoritative object plan"
                    )
                final_state = _object_joint_state_before_phase(
                    object_plan,
                    initial,
                    str(stage["release_before_phase"]),
                )
                for name, value in final_state.items():
                    if name in object_joints:
                        p.resetJointState(
                            object_body,
                            object_joints[name],
                            float(value),
                            targetVelocity=0.0,
                            physicsClientId=client,
                        )
                set_fingers(robot_body, fingers, retreat_opening, client)
                release_minimum = 0.25
                # Initial proximity to the just-released contact link is
                # unavoidable; after the first quarter of the withdrawal it is
                # treated like every other object link.  Other links must be
                # clear from the first sample.
                # Validate the exact dense joint path serialized into the stage
                # plan and consumed by rollout, not a fresh endpoint chord.
                release_path = list(retreat)
                # The first retreat element is byte-identical to the final
                # manipulation command and was already validated under the
                # manipulation collision tolerance. Release clearance begins
                # at the first newly commanded withdrawal sample.
                for sample_index, q in enumerate(release_path[1:], start=1):
                    set_robot_arm(robot_body, arm, q, client)
                    points = object_closest_points(
                        robot_body,
                        object_body,
                        0.25,
                        client,
                    )
                    distances = [
                        float(point[8])
                        for point in points
                        if int(point[4]) != contact_link
                        or sample_index >= len(release_path) // 4
                    ]
                    if robot_support_body is not None:
                        distances.extend(
                            float(point[8])
                            for point in object_closest_points(
                                robot_support_body,
                                object_body,
                                0.25,
                                client,
                            )
                        )
                    if distances:
                        release_minimum = min(release_minimum, min(distances))
                set_robot_arm(robot_body, arm, retreat[-1], client)
                next_contact_phase = (
                    str(execution["stages"][stage_index + 1]["source_phase"])
                    if stage_index + 1 < len(execution["stages"])
                    else None
                )
                for transition in _object_joint_transitions_from_phase(
                    object_plan,
                    final_state,
                    str(stage["release_before_phase"]),
                    next_contact_phase,
                ):
                    name = str(transition["joint"])
                    if name not in object_joints:
                        continue
                    for value in np.linspace(
                        float(transition["start"]),
                        float(transition["target"]),
                        41,
                    ):
                        p.resetJointState(
                            object_body,
                            object_joints[str(name)],
                            float(value),
                            targetVelocity=0.0,
                            physicsClientId=client,
                        )
                        distances = [
                            float(point[8])
                            for point in object_closest_points(
                                robot_body,
                                object_body,
                                0.25,
                                client,
                            )
                        ]
                        if robot_support_body is not None:
                            distances.extend(
                                float(point[8])
                                for point in object_closest_points(
                                    robot_support_body,
                                    object_body,
                                    0.25,
                                    client,
                                )
                            )
                        if distances:
                            release_minimum = min(release_minimum, min(distances))
                for name, value in final_state.items():
                    if name in object_joints:
                        p.resetJointState(
                            object_body,
                            object_joints[name],
                            float(value),
                            targetVelocity=0.0,
                            physicsClientId=client,
                        )
                if minimum_clearance is None or release_minimum < minimum_clearance:
                    minimum_clearance = release_minimum
                violations.append(
                    {
                        "phase": "release_then_plan_sweep",
                        "distance_m": float(release_minimum),
                        "required_m": float(release_required),
                    }
                )
            next_stage = (
                execution["stages"][stage_index + 1]
                if stage_index + 1 < len(execution["stages"])
                else None
            )
            if (
                not release_clearance_pending
                # The schedule holds at the declared release-clearance waypoint
                # while the passive phase runs; it does not append an invented
                # return-to-home motion at that semantic boundary.
                and stage.get("release_before_phase") is None
                and next_stage is None
                and not terminal_plan_hold
            ):
                transit_out_minimum, transit_out_violations, transit_out_gpu = exact_path_clearance(
                    _interpolate(retreat[-1], home, 90),
                    [*forbidden.values(), contact_link],
                    None,
                    "transit_out",
                )
                if transit_out_minimum is not None and (
                    minimum_clearance is None
                    or transit_out_minimum < minimum_clearance
                ):
                    minimum_clearance = transit_out_minimum
                violations.extend(transit_out_violations)
            set_robot_arm(robot_body, arm, retreat[-1], client)
            violations.sort(key=lambda item: item["distance_m"])
            if ik_failures:
                debug_failure = {
                    "kind": "best_effort_ik_diagnostics",
                    "stage_id": str(stage["id"]),
                    "requested_samples": int(len(object_path)),
                    "executed_samples": int(len(manipulation)),
                    "failures": ik_failures,
                    "rollout_continued": True,
                }
            manipulation_array = np.asarray(manipulation)
            joint_limit_distances = np.minimum(
                manipulation_array - solver.arm_lower,
                solver.arm_upper - manipulation_array,
            )
            limit_sample, limit_joint = np.unravel_index(
                int(np.argmin(joint_limit_distances)),
                joint_limit_distances.shape,
            )
            adjacent_steps = np.abs(np.diff(manipulation_array, axis=0))
            plans.append(
                StagePlan(
                    stage=stage,
                    approach=approach,
                    manipulation=manipulation_array,
                    retreat=retreat,
                    object_path=np.asarray(object_path),
                    maximum_position_error_m=max_position,
                    maximum_orientation_error_rad=max_rotation,
                    minimum_swept_clearance_m=minimum_clearance,
                    swept_clearance_violations=violations[:12],
                    minimum_joint_limit_margin_rad=float(
                        joint_limit_distances[limit_sample, limit_joint]
                    ),
                    minimum_joint_limit_margin_sample=int(limit_sample),
                    minimum_joint_limit_margin_joint=int(limit_joint),
                    maximum_adjacent_joint_step_rad=(
                        float(np.max(adjacent_steps))
                        if adjacent_steps.size
                        else 0.0
                    ),
                    ik_backend=(
                        "curobo_batch_ik_pybullet_verified"
                        if curobo_path is not None
                        else "pybullet"
                    ),
                    ik_backend_fallback_reason=curobo_fallback_reason,
                    transit_planner_backend=transit_planner_backend,
                    transit_planner_evidence=transit_planner_evidence,
                    debug_truncated=False,
                    debug_failure=debug_failure,
                    transit_in=transit_in,
                )
            )
            current[driver_name] = target
            next_stage = execution["stages"][stage_index + 1] if stage_index + 1 < len(execution["stages"]) else None
            reference = (
                continuation_reference
                if next_stage is not None and _same_contact_sequence(stage, next_stage)
                else retreat[-1].copy()
            )
        return plans
    finally:
        p.disconnect(client)


def _interpolate(a: np.ndarray, b: np.ndarray, count: int) -> list[np.ndarray]:
    return [(1.0 - t) * a + t * b for t in np.linspace(0.0, 1.0, count)]


def _rrt_connect(
    start: np.ndarray,
    goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    is_clear: Any,
    rng: np.random.Generator,
    *,
    step_rad: float = 0.04,
    maximum_iterations: int = 4000,
) -> list[np.ndarray] | None:
    """Plan a deterministic collision-free joint path with RRT-Connect.

    Cartesian waypoints constrain where the gripper should pass, but straight
    interpolation between their IK solutions can still sweep an elbow or wrist
    through an obstacle.  This generic joint-space fallback keeps every edge
    below the rollout's adjacent-command limit and delegates all scene policy
    to the supplied whole-robot collision predicate.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if start.shape != goal.shape or lower.shape != start.shape or upper.shape != start.shape:
        raise ValueError("RRT joint vectors and bounds must have identical shapes")
    if step_rad <= 0.0 or maximum_iterations <= 0:
        raise ValueError("RRT step and iteration budget must be positive")
    if not is_clear(start) or not is_clear(goal):
        return None

    def edge_is_clear(left: np.ndarray, right: np.ndarray) -> bool:
        distance = float(np.max(np.abs(right - left)))
        count = max(2, int(math.ceil(distance / (0.5 * step_rad))) + 1)
        return all(is_clear(q) for q in _interpolate(left, right, count)[1:])

    direct_distance = float(np.max(np.abs(goal - start)))
    direct_count = max(2, int(math.ceil(direct_distance / step_rad)) + 1)
    direct = _interpolate(start, goal, direct_count)
    if all(is_clear(q) for q in direct[1:]):
        return direct

    def nearest(nodes: list[np.ndarray], target: np.ndarray) -> int:
        return min(
            range(len(nodes)),
            key=lambda index: float(np.linalg.norm(nodes[index] - target)),
        )

    def extend(
        nodes: list[np.ndarray],
        parents: list[int],
        target: np.ndarray,
    ) -> tuple[str, int | None]:
        near_index = nearest(nodes, target)
        delta = target - nodes[near_index]
        distance = float(np.max(np.abs(delta)))
        if distance <= 1e-9:
            return "reached", near_index
        candidate = nodes[near_index] + delta * min(1.0, step_rad / distance)
        candidate = np.clip(candidate, lower, upper)
        if not edge_is_clear(nodes[near_index], candidate):
            return "trapped", None
        nodes.append(candidate)
        parents.append(near_index)
        return ("reached" if distance <= step_rad else "advanced"), len(nodes) - 1

    def trace(nodes: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
        path = []
        while index >= 0:
            path.append(nodes[index])
            index = parents[index]
        return list(reversed(path))

    tree_a = [start.copy()]
    parent_a = [-1]
    tree_b = [goal.copy()]
    parent_b = [-1]
    tree_a_starts_at_start = True
    for iteration in range(maximum_iterations):
        if iteration % 5 == 0:
            sample = tree_b[0]
        else:
            sample = rng.uniform(lower, upper)
        status_a, index_a = extend(tree_a, parent_a, sample)
        if status_a != "trapped" and index_a is not None:
            target = tree_a[index_a]
            while True:
                status_b, index_b = extend(tree_b, parent_b, target)
                if status_b == "trapped":
                    break
                if status_b == "reached" and index_b is not None:
                    path_a = trace(tree_a, parent_a, index_a)
                    path_b = trace(tree_b, parent_b, index_b)
                    if tree_a_starts_at_start:
                        answer = path_a + list(reversed(path_b))[1:]
                    else:
                        answer = path_b + list(reversed(path_a))[1:]
                    return [np.asarray(q, dtype=np.float64) for q in answer]
        tree_a, tree_b = tree_b, tree_a
        parent_a, parent_b = parent_b, parent_a
        tree_a_starts_at_start = not tree_a_starts_at_start
    return None


def _same_contact_sequence(left: dict[str, Any], right: dict[str, Any]) -> bool:
    sequence = left.get("contact_sequence")
    return (
        sequence is not None
        and sequence == right.get("contact_sequence")
        and str(left.get("contact_link")) == str(right.get("contact_link"))
    )


def _inherited_contact_sequence_answer(reference: np.ndarray) -> dict[str, Any]:
    """Return sample-zero evidence without re-solving an unchanged grasp pose."""
    return {
        "success": True,
        "q": np.asarray(reference, dtype=np.float64).copy(),
        "position_error_m": 0.0,
        "orientation_error_rad": 0.0,
        "solver": "exact_contact_sequence_inheritance",
    }


def _grasp_verification_key(stage: dict[str, Any], stage_index: int) -> str:
    """Identify one persistent verified contact gate across a contact sequence."""
    sequence = stage.get("contact_sequence")
    return f"sequence:{sequence}" if sequence is not None else f"stage:{stage_index}"


def _verified_gripper_collision_links(
    robot_body: int,
    verified_finger_links: set[int],
    client: int,
) -> set[int]:
    """Return verified fingers plus their nearest common rigid palm parent.

    After bilateral verification, target-link response for this compact
    gripper assembly is replaced by the semantic joint actuator. Keeping the
    palm/target pair active can still close a solver loop even when the two
    fingertip pairs are disabled. Arm links above the nearest common ancestor
    are deliberately excluded and remain collision-authoritative.
    """
    fingers = sorted(int(index) for index in verified_finger_links)
    if not fingers:
        return set()

    def ancestor_chain(link_index: int) -> list[int]:
        chain: list[int] = []
        current = int(link_index)
        while current >= 0:
            chain.append(current)
            current = int(
                p.getJointInfo(
                    robot_body, current, physicsClientId=client
                )[16]
            )
        return chain

    chains = [ancestor_chain(index) for index in fingers]
    common = set(chains[0])
    for chain in chains[1:]:
        common.intersection_update(chain)
    nearest_common = next(
        (index for index in chains[0] if index in common),
        None,
    )
    result = set(fingers)
    if nearest_common is not None:
        result.add(int(nearest_common))
    return result


def _bilateral_grasp_sample(
    contact_points: Iterable[tuple[Any, ...]],
    required_robot_links: set[int],
    target_object_link: int,
    finger_joint_states: Iterable[tuple[Any, ...]],
    approach_opening_m: float,
    commanded_opening_m: float,
) -> dict[str, Any]:
    """Measure one real opposed-finger acquisition sample.

    The agent does not nominate collision links or claim that a grasp worked.
    ``required_robot_links`` is the application-derived pair of Panda finger
    collision links.  PyBullet contact points, contact normals and measured
    finger joint states are the only acquisition authority.
    """
    required = sorted(int(index) for index in required_robot_links)
    contacts_by_link: dict[int, list[tuple[Any, ...]]] = {
        index: [] for index in required
    }
    unexpected_contact_pairs: dict[str, int] = {}
    for point in contact_points:
        robot_link = int(point[3])
        object_link = int(point[4])
        if robot_link in contacts_by_link and object_link == int(target_object_link):
            contacts_by_link[robot_link].append(point)
        else:
            key = f"{robot_link}|{object_link}"
            unexpected_contact_pairs[key] = unexpected_contact_pairs.get(key, 0) + 1

    both_links_contact = (
        len(required) == 2
        and all(bool(contacts_by_link[index]) for index in required)
    )
    minimum_normal_dot: float | None = None
    if both_links_contact:
        for left in contacts_by_link[required[0]]:
            left_normal = np.asarray(left[7], dtype=np.float64)
            left_norm = float(np.linalg.norm(left_normal))
            if left_norm <= 1e-12:
                continue
            left_normal /= left_norm
            for right in contacts_by_link[required[1]]:
                right_normal = np.asarray(right[7], dtype=np.float64)
                right_norm = float(np.linalg.norm(right_normal))
                if right_norm <= 1e-12:
                    continue
                right_normal /= right_norm
                dot = float(np.dot(left_normal, right_normal))
                minimum_normal_dot = (
                    dot
                    if minimum_normal_dot is None
                    else min(minimum_normal_dot, dot)
                )
    opposed_contact = bool(
        minimum_normal_dot is not None
        and minimum_normal_dot <= GRASP_CONTACT_NORMAL_MAXIMUM_DOT
    )

    states = list(finger_joint_states)
    positions = [float(state[0]) for state in states]
    speeds = [abs(float(state[1])) for state in states]
    maximum_speed = max(speeds, default=float("inf"))
    fingers_settled = bool(
        len(states) == 2 and maximum_speed <= GRASP_FINGER_MAXIMUM_SPEED_M_S
    )
    closure_span = max(
        float(approach_opening_m) - float(commanded_opening_m), 1e-9
    )
    closure_fractions = [
        (float(approach_opening_m) - position) / closure_span
        for position in positions
    ]
    minimum_closure_fraction = min(closure_fractions, default=-float("inf"))
    sufficiently_closed = bool(
        len(states) == 2
        and minimum_closure_fraction >= GRASP_MINIMUM_CLOSURE_FRACTION
    )
    passed = bool(
        both_links_contact
        and opposed_contact
        and fingers_settled
        and sufficiently_closed
        and not unexpected_contact_pairs
    )
    return {
        "passed": passed,
        "required_robot_links": required,
        "contact_count_by_robot_link": {
            str(index): len(contacts_by_link[index]) for index in required
        },
        "unexpected_contact_pairs": unexpected_contact_pairs,
        "non_target_contact_zero": not unexpected_contact_pairs,
        "both_finger_links_contact_target": both_links_contact,
        "minimum_opposed_contact_normal_dot": minimum_normal_dot,
        "opposed_contact_normals": opposed_contact,
        "finger_joint_positions_m": positions,
        "maximum_finger_speed_m_s": maximum_speed,
        "fingers_settled": fingers_settled,
        "minimum_closure_fraction": minimum_closure_fraction,
        "sufficiently_closed": sufficiently_closed,
    }


def _schedule(
    plans: list[StagePlan], execution: dict[str, Any], plan: dict[str, Any]
) -> list[dict[str, Any]]:
    object_plan = plan
    home = np.asarray(execution["robot"]["home_joint_positions"], dtype=np.float64)
    # The settle tail is the only window in which a released effect (a lid
    # closing after the pedal returns) can finish, so its length is execution
    # data rather than a fixed constant.
    settle_ticks = int(round(float(execution.get("settle_s", 1.0)) / DT))
    timeline = list(plan.get("timeline", []))
    phase_order = {str(phase["name"]): index for index, phase in enumerate(timeline)}
    ownership = {
        (str(row["source_phase"]), int(row["source_control_index"])): row
        for row in execution["control_execution"]
    }
    passive_start: dict[str, int] = {}
    for phase_index, phase in enumerate(timeline):
        for control_index, control in enumerate(phase.get("controls", [])):
            row = ownership[(str(phase["name"]), control_index)]
            if row["motion_owner"] == "passive_return":
                passive_start[str(control["joint"])] = phase_index

    commands: list[dict[str, Any]] = []

    def append_command(
        phase: str,
        stage: int | None,
        arm_q: np.ndarray,
        finger: float,
        finger_force_n: float,
        timeline_phase_index: int,
        sample: int | None = None,
    ) -> None:
        command: dict[str, Any] = {
            "phase": phase,
            "stage": stage,
            "arm": arm_q.tolist(),
            "finger": float(finger),
            "finger_force_n": float(finger_force_n),
            "timeline_phase_index": int(timeline_phase_index),
            "active_passive_joints": sorted(
                joint
                for joint, start_index in passive_start.items()
                if start_index <= timeline_phase_index
            ),
        }
        if sample is not None:
            command["sample"] = int(sample)
        commands.append(command)

    def append_plan_boundaries(
        start: int,
        stop: int,
        arm_q: np.ndarray,
        finger: float,
        finger_force_n: float,
    ) -> None:
        """Preserve semantic ordering without replaying ArtiMo timestamps.

        ``t0``/``t1`` describe the source animation and must never determine
        robot rollout duration.  One command tick is sufficient to cross an
        owner/phase boundary; physical dwell belongs to contact stages or the
        explicit final settle window.
        """
        for phase_index in range(start, stop):
            append_command(
                "plan_boundary",
                None,
                arm_q,
                finger,
                finger_force_n,
                phase_index,
            )

    current = home.copy()
    current_finger = 0.04
    current_finger_force = PANDA_FINGER_FORCE_N
    last_timeline_phase = -1
    for index, plan in enumerate(plans):
        continues_from_previous = index > 0 and _same_contact_sequence(plans[index - 1].stage, plan.stage)
        continues_to_next = index + 1 < len(plans) and _same_contact_sequence(plan.stage, plans[index + 1].stage)
        opening = float(plan.stage["finger_opening_m"])
        acquisition = plan.stage["contact_acquisition"]
        acquisition_mode = str(acquisition["mode"])
        approach_opening = float(acquisition["approach_finger_opening_m"])
        finger_force = PANDA_FINGER_FORCE_N
        timeline_phase = phase_order[str(plan.stage["source_phase"])]
        stage_last_timeline_phase = timeline_phase
        held_through_timeline_phase = timeline_phase
        driver_joint = str(plan.stage["driver_joint"])
        for following_phase_index in range(timeline_phase + 1, len(timeline)):
            following_phase = timeline[following_phase_index]
            following_name = str(following_phase["name"])
            retains_driver = any(
                str(control.get("mode", "")) == "hold_position"
                and str(control.get("joint", "")) == driver_joint
                and ownership[(following_name, control_index)]["motion_owner"] == "hold"
                for control_index, control in enumerate(following_phase.get("controls", []))
            )
            if not retains_driver:
                break
            held_through_timeline_phase = following_phase_index
        if not continues_from_previous:
            append_plan_boundaries(
                last_timeline_phase + 1,
                timeline_phase,
                current,
                current_finger,
                current_finger_force,
            )
        if not continues_from_previous:
            transit_path = (
                plan.transit_in
                if plan.transit_in is not None
                else np.asarray(
                    _interpolate(current, plan.approach[0], 90),
                    dtype=np.float64,
                )
            )
            for q in transit_path:
                append_command(
                    "transit", index, q, approach_opening, finger_force, timeline_phase
                )
            for q in plan.approach:
                for _ in range(4):
                    append_command(
                        "approach", index, q, approach_opening, finger_force, timeline_phase
                    )
            if acquisition_mode == "open_then_close":
                # Reach the final contact-pose IK with the fingers still open
                # before closure begins.  Closing while the arm is still
                # traversing precontact -> contact can brush one side of a
                # held feature and finish with both fingers beside it.  This
                # deterministic convergence dwell is application policy, not
                # an agent-supplied timing knob.
                preclose_ticks = max(
                    1, int(round(GRASP_PRE_CLOSE_SETTLE_S / DT))
                )
                for _ in range(preclose_ticks):
                    append_command(
                        "contact_preclose_settle",
                        index,
                        plan.manipulation[0],
                        approach_opening,
                        finger_force,
                        timeline_phase,
                        0,
                    )
                close_ticks = max(1, int(round(float(acquisition["close_s"]) / DT)))
                for finger in np.linspace(approach_opening, opening, close_ticks):
                    append_command(
                        "contact_acquire",
                        index,
                        plan.manipulation[0],
                        float(finger),
                        finger_force,
                        timeline_phase,
                        0,
                    )
                settle_ticks_for_grasp = int(round(float(acquisition["settle_s"]) / DT))
                for _ in range(settle_ticks_for_grasp):
                    append_command(
                        "contact_settle",
                        index,
                        plan.manipulation[0],
                        opening,
                        finger_force,
                        timeline_phase,
                        0,
                    )
            if plan.stage["interaction"] == "explicit_ideal_feasibility":
                # Acquisition is deliberately split into three ordered pieces:
                # close the fingers, let the real robot converge while the
                # object driver remains locked, then verify the single disclosed
                # contact gate and stabilize it before manipulation.  The
                # legacy phase label is kept for serialized-command compatibility;
                # no Bullet attachment is created.  A gate on the first
                # contact-settle tick could let a
                # lagging arm could therefore drag an unlatched door before the
                # commanded grasp pose had actually settled.
                append_command(
                    "contact_attach",
                    index,
                    plan.manipulation[0],
                    opening,
                    finger_force,
                    timeline_phase,
                    0,
                )
                verification_settle_ticks = max(
                    1, int(round(GRASP_VERIFICATION_STABILIZE_S / DT))
                )
                for _ in range(verification_settle_ticks):
                    append_command(
                        "grasp_stabilize",
                        index,
                        plan.manipulation[0],
                        opening,
                        finger_force,
                        timeline_phase,
                        0,
                    )
        sample_ticks = max(
            1,
            int(
                round(
                    float(plan.stage.get("manipulation_sample_hold_s", 8 * DT))
                    / DT
                )
            ),
        )
        previous_manipulation_q = plan.manipulation[0].copy()
        for sample, q in enumerate(plan.manipulation):
            for tick_index in range(sample_ticks):
                alpha = float(tick_index + 1) / float(sample_ticks)
                continuous_q = (
                    (1.0 - alpha) * previous_manipulation_q + alpha * q
                )
                append_command(
                    "manipulate",
                    index,
                    continuous_q,
                    opening,
                    finger_force,
                    timeline_phase,
                    sample,
                )
            previous_manipulation_q = q.copy()
        for _ in range(int(round(float(plan.stage.get("hold_s", 0.25)) / DT))):
            append_command(
                "hold",
                index,
                plan.manipulation[-1],
                opening,
                finger_force,
                timeline_phase,
            )
        if not continues_to_next:
            retreat_opening = opening
            release_before_phase = plan.stage.get("release_before_phase")
            release_timeline_phase = (
                phase_order[str(release_before_phase)]
                if release_before_phase is not None
                else timeline_phase
            )
            release_command_timeline_phase = (
                max(timeline_phase, release_timeline_phase - 1)
                if release_before_phase is not None
                else timeline_phase
            )
            if release_before_phase is not None:
                # Preserve the acquired grasp through all complete intervening
                # plan phases.  The named phase is the explicit semantic
                # boundary at which release becomes allowed.
                append_plan_boundaries(
                    timeline_phase + 1,
                    release_timeline_phase,
                    plan.manipulation[-1],
                    opening,
                    finger_force,
                )
                stage_last_timeline_phase = release_timeline_phase - 1
            elif held_through_timeline_phase > timeline_phase:
                # A plan-owned hold_position retains the endpoint reached by
                # this robot contact.  Keep the physical grasp through every
                # immediately following hold phase instead of releasing first
                # and letting an unpowered joint drift during the declared
                # hold.  One boundary tick preserves ordering; the stage dwell
                # and final settle window remain the only wall-clock timing.
                append_plan_boundaries(
                    timeline_phase + 1,
                    held_through_timeline_phase + 1,
                    plan.manipulation[-1],
                    opening,
                    finger_force,
                )
                stage_last_timeline_phase = held_through_timeline_phase
            terminal_plan_hold = bool(
                release_before_phase is None
                and index + 1 == len(plans)
                and _terminal_plan_hold_phase_index(
                    object_plan, execution, plan.stage
                )
                is not None
            )
            if terminal_plan_hold:
                # There is no later semantic release boundary.  End the
                # rollout at the held object endpoint with the disclosed ideal
                # grasp still active; simulator cleanup removes it only after
                # all commands and rendered frames have completed.
                current = plan.manipulation[-1].copy()
                current_finger = opening
                current_finger_force = finger_force
                last_timeline_phase = stage_last_timeline_phase
                continue
            if acquisition_mode == "open_then_close":
                release_ticks = max(1, int(round(float(acquisition["release_s"]) / DT)))
                for finger in np.linspace(opening, approach_opening, release_ticks):
                    append_command(
                        "contact_release",
                        index,
                        plan.manipulation[-1],
                        float(finger),
                        finger_force,
                        release_command_timeline_phase,
                    )
                retreat_opening = approach_opening
            if release_before_phase is not None:
                # Dependent mechanism motion or a passive return may not start
                # with the hand left in its swept volume. Release first,
                # execute only the explicitly planned clearance retreat, then
                # visibly settle at its safe endpoint before crossing the plan
                # boundary that enables object motion.
                for q in plan.retreat[1:]:
                    for _ in range(3):
                        append_command(
                            "retreat",
                            index,
                            q,
                            retreat_opening,
                            finger_force,
                            release_command_timeline_phase,
                        )
                current = plan.retreat[-1].copy()
                release_settle_ticks = max(
                    1, int(round(RELEASE_RETREAT_SETTLE_S / DT))
                )
                for _ in range(release_settle_ticks):
                    append_command(
                        "release_retreat_settle",
                        index,
                        current,
                        retreat_opening,
                        finger_force,
                        release_command_timeline_phase,
                    )
            elif index + 1 == len(plans):
                for q in plan.retreat:
                    for _ in range(3):
                        append_command(
                            "retreat",
                            index,
                            q,
                            retreat_opening,
                            finger_force,
                            timeline_phase,
                        )
                for q in _interpolate(plan.retreat[-1], home, 90):
                    append_command(
                        "transit",
                        index,
                        q,
                        retreat_opening,
                        finger_force,
                        timeline_phase,
                    )
                current = home.copy()
            else:
                # A later plan-owned robot stage is already the reason for a
                # new transit.  Preserve the released retreat endpoint and let
                # the next stage interpolate directly from it to its own
                # approach instead of using home as an unrelated hub pose.
                for q in plan.retreat:
                    for _ in range(3):
                        append_command(
                            "retreat",
                            index,
                            q,
                            retreat_opening,
                            finger_force,
                            timeline_phase,
                        )
                current = plan.retreat[-1].copy()
            current_finger = retreat_opening
            current_finger_force = finger_force
        else:
            current = plan.manipulation[-1].copy()
            current_finger = opening
            current_finger_force = finger_force
        last_timeline_phase = stage_last_timeline_phase
    append_plan_boundaries(
        last_timeline_phase + 1,
        len(timeline),
        current,
        current_finger,
        current_finger_force,
    )
    for _ in range(settle_ticks):
        append_command(
            "settle",
            None,
            current,
            current_finger,
            current_finger_force,
            len(timeline),
        )
    return commands


def _robot_path_sample_progress(
    path: np.ndarray,
    configuration: np.ndarray,
    maximum_sample: int | None = None,
) -> tuple[float, float]:
    """Return dense-sample progress and residual to a robot joint path.

    The nearest projected point is capped by the currently commanded sample,
    preventing a self-intersecting path from jumping ahead.  Rollout keeps the
    returned progress monotonic and accepts it only when ``residual`` is within
    the application-owned tracking bound. Progress is the fractional sample
    index, not joint-space arc length: dense IK is generated at uniform values
    of the object trajectory parameter. The caller applies the same smoothstep
    used during planning to recover the exact ArtiMo joint target.
    """
    rows = np.asarray(path, dtype=np.float64)
    q = np.asarray(configuration, dtype=np.float64)
    if rows.ndim != 2 or len(rows) == 0 or rows.shape[1] != len(q):
        raise ValueError("Robot progress path/configuration shape mismatch")
    if len(rows) == 1:
        return 1.0, float(np.max(np.abs(q - rows[0])))
    segment_vectors = rows[1:] - rows[:-1]
    if float(np.max(np.abs(segment_vectors))) <= 1e-12:
        return 1.0, float(np.max(np.abs(q - rows[-1])))
    last_segment = len(segment_vectors) - 1
    if maximum_sample is not None:
        last_segment = min(last_segment, max(0, int(maximum_sample) - 1))
    best_distance = float("inf")
    best_progress = 0.0
    for index in range(last_segment + 1):
        vector = segment_vectors[index]
        squared_length = float(np.dot(vector, vector))
        alpha = 0.0 if squared_length <= 1e-18 else float(
            np.clip(np.dot(q - rows[index], vector) / squared_length, 0.0, 1.0)
        )
        projected = rows[index] + alpha * vector
        residual = float(np.max(np.abs(q - projected)))
        if residual < best_distance:
            best_distance = residual
            best_progress = float((index + alpha) / (len(rows) - 1))
    if maximum_sample is not None:
        cap_index = min(max(0, int(maximum_sample)), len(rows) - 1)
        best_progress = min(best_progress, float(cap_index / (len(rows) - 1)))
    return float(np.clip(best_progress, 0.0, 1.0)), best_distance


def _project_pixel(
    world: list[float], view: list[float], projection: list[float], width: int, height: int
) -> tuple[int, int] | None:
    view_matrix = np.asarray(view, dtype=np.float64).reshape(4, 4, order="F")
    projection_matrix = np.asarray(projection, dtype=np.float64).reshape(4, 4, order="F")
    clip = projection_matrix @ view_matrix @ np.asarray([*world, 1.0], dtype=np.float64)
    if clip[3] <= 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    if np.any(np.abs(ndc[:2]) > 1.2):
        return None
    return (
        int(round((ndc[0] + 1.0) * 0.5 * width)),
        int(round((1.0 - ndc[1]) * 0.5 * height)),
    )


def _render_frame(
    client: int,
    execution: dict[str, Any],
    status: str,
    phase: str,
    target_world: list[float] | None,
    renderer: int,
) -> np.ndarray:
    camera = execution.get("camera", {})
    eye = camera.get("eye_m", [1.2, -1.2, 0.9])
    target = camera.get("target_m", [0.0, 0.0, 0.35])
    view = p.computeViewMatrix(eye, target, [0.0, 0.0, 1.0])
    projection = p.computeProjectionMatrixFOV(
        float(camera.get("fov_deg", 50.0)),
        VIDEO_WIDTH / VIDEO_HEIGHT,
        0.02,
        6.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        VIDEO_WIDTH,
        VIDEO_HEIGHT,
        view,
        projection,
        renderer=renderer,
        physicsClientId=client,
    )
    image = Image.fromarray(
        np.asarray(rgba, dtype=np.uint8).reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 4),
        "RGBA",
    ).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, VIDEO_WIDTH, 28), fill=(18, 22, 29))
    draw.text((9, 8), status, fill=(245, 248, 252), font=ImageFont.load_default())
    if phase in {
        "approach",
        "contact_acquire",
        "contact_settle",
        "manipulate",
        "hold",
        "contact_release",
        "retreat",
    } and target_world is not None:
        pixel = _project_pixel(
            target_world, view, projection, VIDEO_WIDTH, VIDEO_HEIGHT
        )
        if pixel is not None:
            half_width, half_height = 80, 60
            left = max(
                0,
                min(VIDEO_WIDTH - 2 * half_width, pixel[0] - half_width),
            )
            top = max(
                28,
                min(VIDEO_HEIGHT - 2 * half_height, pixel[1] - half_height),
            )
            inset = image.crop((left, top, left + 2 * half_width, top + 2 * half_height))
            inset_width, inset_height = 240, 180
            inset_x, inset_y = 8, VIDEO_HEIGHT - inset_height - 8
            inset = inset.resize(
                (inset_width, inset_height), Image.Resampling.BICUBIC
            )
            image.paste(inset, (inset_x, inset_y))
            draw = ImageDraw.Draw(image)
            label_top = inset_y - 22
            draw.rectangle(
                (
                    inset_x - 2,
                    label_top,
                    inset_x + inset_width + 2,
                    inset_y + inset_height + 2,
                ),
                outline=(240, 117, 30),
                width=2,
            )
            draw.rectangle(
                (inset_x, label_top + 2, inset_x + inset_width, inset_y),
                fill=(18, 22, 29),
            )
            draw.text(
                (inset_x + 6, label_top + 6),
                "TARGET CONTACT - SAME FRAME",
                fill=(255, 181, 92),
                font=ImageFont.load_default(),
            )
    return np.asarray(image, dtype=np.uint8)


def _select_rollout_frame_renderer(client: int) -> tuple[int, dict[str, Any]]:
    """Use EGL hardware OpenGL when this PyBullet build actually provides it.

    Merely passing ``ER_BULLET_HARDWARE_OPENGL`` to a DIRECT client on Windows
    silently runs TinyRenderer.  Probe for the real EGL plugin instead and
    record an explicit fallback.  ``ARTIMO_PYBULLET_RENDERER=gpu`` makes the
    absence of a hardware context an error rather than an undisclosed fallback.
    """
    requested = os.environ.get("ARTIMO_PYBULLET_RENDERER", "auto").strip().lower()
    if requested not in {"auto", "gpu", "egl", "cpu", "tiny"}:
        raise ValueError(
            "ARTIMO_PYBULLET_RENDERER must be auto, gpu, egl, cpu, or tiny"
        )
    if requested in {"auto", "gpu", "egl"}:
        loader = pkgutil.get_loader("eglRenderer")
        if loader is not None:
            try:
                plugin_id = p.loadPlugin(
                    loader.get_filename(), "_eglRendererPlugin", physicsClientId=client
                )
            except Exception as exc:  # pragma: no cover - platform/plugin guard
                plugin_id = -1
                failure = str(exc)
            else:
                failure = None
            if plugin_id >= 0:
                return p.ER_BULLET_HARDWARE_OPENGL, {
                    "backend": "egl_hardware_opengl",
                    "hardware_accelerated": True,
                    "requested": requested,
                    "plugin_id": int(plugin_id),
                }
        else:
            failure = "eglRenderer plugin is not installed in this PyBullet build"
        if requested in {"gpu", "egl"}:
            raise RuntimeError(
                "GPU PyBullet rendering was requested but unavailable: " + str(failure)
            )
    return p.ER_TINY_RENDERER, {
        "backend": "tiny_renderer",
        "hardware_accelerated": False,
        "requested": requested,
        "fallback_reason": (
            "eglRenderer plugin is not installed in this PyBullet build"
            if requested == "auto"
            else None
        ),
    }


def _rollout(
    object_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
    object_plan: dict[str, Any],
    plans: list[StagePlan],
    commands: list[dict[str, Any]],
    condition: str,
    video_path: Path | None,
    debug_partial: bool = False,
    maximum_command_count: int | None = None,
) -> dict[str, Any]:
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet rollout client")
    writer = None
    video_encoding = None
    frame_renderer, frame_rendering = _select_rollout_frame_renderer(client)
    try:
        p.setPhysicsEngineParameter(
            fixedTimeStep=DT,
            numSolverIterations=180,
            deterministicOverlappingPairs=1,
            physicsClientId=client,
        )
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)
        object_body, robot_body, robot_support_body = _load_scene(
            object_urdf, robot_urdf, execution, client, True
        )
        object_joints, object_links = _maps(object_body, client)
        robot_joints, robot_links = _maps(robot_body, client)
        # Index -> name, so an unexpected contact can be reported by link name
        # instead of an opaque PyBullet index.
        object_link_names = {index: name for name, index in object_links.items()}
        robot_link_names = {index: name for name, index in robot_links.items()}
        robot_spec = execution["robot"]
        arm = [robot_joints[name] for name in robot_spec["arm_joint_names"]]
        fingers = [robot_joints[name] for name in robot_spec["finger_joint_names"]]
        # The Panda is a stiff position-controlled trajectory executor in this
        # benchmark.  Do not silently cap it to the robot URDF's nominal effort
        # fields. These application-owned constants are deliberately absent
        # from per-task execution data.
        arm_effort_limits = [
            PANDA_ARM_FORCE_N * PANDA_ARM_FORCE_SCALE for _ in arm
        ]
        finger_effort_limits = [PANDA_FINGER_FORCE_N for _ in fingers]
        eef = robot_links[robot_spec["end_effector_link"]]
        for name, value in initial.items():
            if name in object_joints:
                p.resetJointState(object_body, object_joints[name], value, targetVelocity=0.0, physicsClientId=client)
        movable = list(object_joints.values())
        p.setJointMotorControlArray(
            object_body, movable, p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(movable), forces=[0.0] * len(movable), physicsClientId=client
        )
        declared_object_joints = _declared_object_joint_names(execution, plans)
        undeclared_object_joints = sorted(
            set(object_joints) - declared_object_joints
        )
        # Joints absent from plan ownership are scene structure, not passive
        # animation channels.  Leaving every URDF joint at zero motor force made
        # unrequested kettle bases/buttons drift under gravity or incidental
        # contact.  Hold only undeclared joints at their frozen initial state;
        # robot drivers, causal effects, and passive returns retain their
        # declared physical policies below.
        for name in undeclared_object_joints:
            joint = object_joints[name]
            maximum_force = max(
                DEFAULT_OBJECT_HOLD_FORCE_OR_TORQUE,
                float(p.getJointInfo(object_body, joint, physicsClientId=client)[10]),
            )
            p.setJointMotorControl2(
                object_body,
                joint,
                p.POSITION_CONTROL,
                targetPosition=float(initial.get(name, 0.0)),
                force=maximum_force,
                positionGain=1.0,
                velocityGain=1.0,
                physicsClientId=client,
            )
        set_robot_arm(robot_body, arm, np.asarray(robot_spec["home_joint_positions"], dtype=np.float64), client)
        set_fingers(robot_body, fingers, 0.04, client)

        allowed_by_stage: list[tuple[set[int], int]] = []
        effect_links: set[int] = set()
        effect_joint_names = {
            effect["joint"]
            for rule in execution.get("causal_rules", [])
            for group in (rule, rule.get("release"))
            if isinstance(group, dict)
            for effect in group.get("effects", [])
        }
        for name in effect_joint_names:
            if name in object_joints:
                effect_links.add(object_joints[name])
        for plan in plans:
            allowed_robot = {robot_links[name] for name in plan.stage["allowed_robot_contact_links"]}
            allowed_object = object_links[plan.stage["contact_link"]]
            allowed_by_stage.append((allowed_robot, allowed_object))
            if condition == "contact_disabled":
                for robot_link in allowed_robot:
                    p.setCollisionFilterPair(
                        robot_body, object_body, robot_link, allowed_object, 0, physicsClientId=client
                    )

        if video_path is not None:
            writer, video_encoding = artimo_video.open_h264_writer(
                video_path, fps=VIDEO_FPS, macro_block_size=1
            )
            video_encoding["frame_rendering"] = frame_rendering
            video_encoding["fps"] = VIDEO_FPS
            video_encoding["resolution"] = [VIDEO_WIDTH, VIDEO_HEIGHT]
        stage_metrics = [
            {
                "stage_id": plan.stage["id"],
                "interaction": plan.stage["interaction"],
                "target_contact_observations": 0,
                "non_target_contact_observations": 0,
                "effect_link_contact_observations": 0,
                "maximum_driver_displacement": 0.0,
                "maximum_absolute_driver_motor_torque": 0.0,
                "driver_motor_torque_by_phase": {},
                "maximum_robot_execution_progress": 0.0,
                "last_contact_gated_actuation_target": None,
                "maximum_contact_gated_tracking_error": 0.0,
                "maximum_continuous_contact_ticks": 0,
                "first_target_contact_tick": None,
                "unexpected_contact_pairs": {},
            }
            for plan in plans
        ]
        current_contact_ticks = [0] * len(plans)
        # Instantaneous (not peak) driver offset from its initial value, so a
        # release rule can observe the control actually coming back.
        current_driver_displacement = [0.0] * len(plans)
        histories: dict[str, list[float]] = {name: [] for name in object_joints}
        previous_command_q = np.asarray(
            robot_spec["home_joint_positions"], dtype=np.float64
        )
        previous_actual_q = previous_command_q.copy()
        robot_tracking = {
            "maximum_command_tracking_error_rad": 0.0,
            "maximum_command_tracking_error_tick": 0,
            "maximum_command_tracking_error_joint": None,
            "maximum_commanded_joint_step_rad": 0.0,
            "maximum_commanded_joint_step_tick": 0,
            "maximum_actual_joint_step_rad": 0.0,
            "maximum_actual_joint_step_tick": 0,
        }
        rule_states = [
            {
                "latched": False,
                "dwell": 0,
                "triggers": 0,
                "effects_enabled": False,
                "latched_tick": None,
                "effects_enabled_tick": None,
                "released": False,
                "released_tick": None,
                "release_ticks": 0,
                "release_triggers": 0,
            }
            for _ in execution.get("causal_rules", [])
        ]
        # Verification belongs to the uninterrupted contact sequence, not to
        # each object-motion stage.  Once established, it gates object-joint
        # actuation until the explicit release boundary.  No runtime constraint
        # or hidden attachment is created.
        verified_grasps: set[str] = set()
        verified_disabled_collision_pairs: dict[
            str, list[tuple[int, int]]
        ] = {}
        required_grasp_keys = {
            _grasp_verification_key(plan.stage, index)
            for index, plan in enumerate(plans)
            if plan.stage["interaction"] == "explicit_ideal_feasibility"
        }
        acquisition_required_ticks = max(
            1, int(round(GRASP_ACQUISITION_DWELL_S / DT))
        )
        acquisition_states = {
            key: {
                "consecutive_pass_ticks": 0,
                "maximum_consecutive_pass_ticks": 0,
                "acquired": False,
                "verified_tick": None,
                "failed_verification_tick": None,
                "last_sample": None,
            }
            for key in required_grasp_keys
        }
        aborted_after_failed_acquisition = False
        executed_command_count = 0
        initial_driver = {
            plan.stage["id"]: float(initial.get(plan.stage["driver_joint"], 0.0)) for plan in plans
        }
        precontact_driver_holds = _precontact_driver_hold_targets(plans, initial)
        contact_released_driver_joints: set[str] = set()
        completed_driver_holds: dict[str, float] = {}
        actuation_start_positions: dict[int, float] = {}
        actuation_progress: dict[int, float] = {
            index: 0.0 for index in range(len(plans))
        }
        actuation_targets: dict[int, float] = {}
        timeline_phase_order = _timeline_phase_order(object_plan)

        def release_verified_grasp(grasp_key: str) -> None:
            verified_grasps.discard(grasp_key)
            for robot_link, object_link in verified_disabled_collision_pairs.pop(
                grasp_key, []
            ):
                p.setCollisionFilterPair(
                    robot_body,
                    object_body,
                    robot_link,
                    object_link,
                    1,
                    physicsClientId=client,
                )

        scheduled_commands = (
            commands
            if maximum_command_count is None
            else commands[: int(maximum_command_count)]
        )
        for tick, command in enumerate(scheduled_commands):
            executed_command_count = tick + 1
            stage_index = command.get("stage")
            phase = str(command["phase"])
            active_robot_driver_joint: str | None = None
            if stage_index is not None:
                active_plan_for_control = plans[int(stage_index)]
                active_grasp_key_for_control = _grasp_verification_key(
                    active_plan_for_control.stage, int(stage_index)
                )
                candidate_driver_joint = str(
                    active_plan_for_control.stage["driver_joint"]
                )
                if _robot_driver_actuation_is_enabled(
                    interaction=str(
                        active_plan_for_control.stage["interaction"]
                    ),
                    phase=phase,
                    condition=condition,
                    grasp_verified=(
                        active_grasp_key_for_control in verified_grasps
                    ),
                    target_contact_observed=(
                        candidate_driver_joint
                        in contact_released_driver_joints
                    ),
                ):
                    active_robot_driver_joint = candidate_driver_joint
                    contact_released_driver_joints.add(active_robot_driver_joint)
            p.setJointMotorControlArray(
                robot_body,
                arm,
                p.POSITION_CONTROL,
                targetPositions=command["arm"],
                forces=arm_effort_limits,
                positionGains=[PANDA_ARM_POSITION_GAIN] * len(arm),
                velocityGains=[1.0] * len(arm),
                physicsClientId=client,
            )
            active_passive_joints = set(command.get("active_passive_joints", []))
            # A robot-owned object joint is free only while its own contact
            # stage is actively manipulating it.  Before acquisition it holds
            # the initial state; after reaching an endpoint it holds that
            # endpoint until a declared passive-return controller takes over.
            # This keeps earlier joints in a multi-joint contact sequence from
            # drifting without making the current driver fight the robot.
            for name, target in precontact_driver_holds.items():
                joint = object_joints[name]
                if condition == "physical" and name == active_robot_driver_joint:
                    # The contact-gated actuator is written after every other
                    # object controller immediately before the physics step.
                    # Clear the prior hold here so no stale POSITION controller
                    # survives if progress validation rejects this tick.
                    p.setJointMotorControl2(
                        object_body,
                        joint,
                        p.VELOCITY_CONTROL,
                        targetVelocity=0.0,
                        force=0.0,
                        physicsClientId=client,
                    )
                    continue
                hold_target = completed_driver_holds.get(name, float(target))
                maximum_force = max(
                    DEFAULT_OBJECT_HOLD_FORCE_OR_TORQUE,
                    float(
                        p.getJointInfo(
                            object_body, joint, physicsClientId=client
                        )[10]
                    ),
                )
                p.setJointMotorControl2(
                    object_body,
                    joint,
                    p.POSITION_CONTROL,
                    targetPosition=float(hold_target),
                    force=maximum_force,
                    positionGain=1.0,
                    velocityGain=1.0,
                    physicsClientId=client,
                )
            for passive in execution.get("passive_joints", []):
                if passive["joint"] not in active_passive_joints:
                    continue
                if not _passive_driver_is_enabled(
                    str(passive["joint"]),
                    precontact_driver_holds,
                    contact_released_driver_joints,
                ):
                    # The contact-disabled control never acquired the driver,
                    # so passive return cannot create motion without the push.
                    continue
                p.setJointMotorControl2(
                    object_body,
                    object_joints[passive["joint"]],
                    p.POSITION_CONTROL,
                    targetPosition=float(passive["rest_position"]),
                    force=float(passive["maximum_force_or_torque"]),
                    positionGain=float(passive["position_gain"]),
                    velocityGain=1.0,
                    physicsClientId=client,
                )
            p.setJointMotorControlArray(
                robot_body,
                fingers,
                p.POSITION_CONTROL,
                targetPositions=[float(command["finger"])] * len(fingers),
                forces=[
                    min(float(command["finger_force_n"]), effort)
                    for effort in finger_effort_limits
                ],
                physicsClientId=client,
            )
            if stage_index is not None:
                active_stage_plan = plans[int(stage_index)]
                grasp_key = _grasp_verification_key(
                    active_stage_plan.stage, int(stage_index)
                )
                if phase in {"contact_release", "retreat"}:
                    release_verified_grasp(grasp_key)

            for rule_index, rule in enumerate(execution.get("causal_rules", [])):
                state = rule_states[rule_index]
                trigger_index = next(
                    index for index, plan in enumerate(plans) if plan.stage["id"] == rule["trigger_stage"]
                )
                metric = stage_metrics[trigger_index]
                if (
                    not state["latched"]
                    and float(metric["maximum_driver_displacement"]) >= float(rule["minimum_displacement"])
                    and current_contact_ticks[trigger_index] >= int(round(float(rule["minimum_dwell_s"]) / DT))
                ):
                    state["latched"] = True
                    state["triggers"] += 1
                    state["latched_tick"] = int(tick)
                trigger_sequence = plans[trigger_index].stage.get("contact_sequence")
                release_boundary = plans[trigger_index].stage.get("release_before_phase")
                if trigger_sequence is not None:
                    release_boundary = next(
                        item.stage.get("release_before_phase")
                        for item in reversed(plans)
                        if item.stage.get("contact_sequence") == trigger_sequence
                    )
                if release_boundary is not None:
                    # A retreat command is not proof that the robot is already
                    # clear.  For an explicit release boundary, causal motion
                    # becomes eligible only after the whole retreat and its
                    # endpoint settle have completed and the scheduler crosses
                    # into the declared later phase.
                    trigger_stage_done = (
                        int(command["timeline_phase_index"])
                        >= int(timeline_phase_order[str(release_boundary)])
                        and phase
                        not in {
                            "contact_release",
                            "retreat",
                            "release_retreat_settle",
                        }
                    )
                else:
                    trigger_stage_done = (
                        stage_index is None
                        or int(stage_index) > trigger_index
                        or phase in {"retreat", "transit", "settle"}
                    )
                source_effect_phase_reached = int(
                    command["timeline_phase_index"]
                ) >= int(timeline_phase_order[str(rule["source_effect_phase"])])
                minimum_clearance = float(rule.get("minimum_clearance_m", 0.0))
                clearance_passed = True
                if minimum_clearance > 0.0:
                    for effect in rule["effects"]:
                        effect_link = object_joints[effect["joint"]]
                        nearby = object_closest_points(
                            robot_body,
                            object_body,
                            minimum_clearance,
                            client,
                            link_index_b=effect_link,
                        )
                        if nearby:
                            clearance_passed = False
                            break
                if (
                    state["latched"]
                    and trigger_stage_done
                    and source_effect_phase_reached
                    and clearance_passed
                ):
                    if not state["effects_enabled"]:
                        state["effects_enabled_tick"] = int(tick)
                    state["effects_enabled"] = True

                # An ArtiMo plan may drive the same effect joint to a second
                # endpoint once the control returns toward rest (a lid closing
                # after a pedal is released).  That later phase is gated on the
                # *instantaneous* driver displacement falling back below a
                # declared bound, so it still follows from measured object state
                # rather than from elapsed time.
                release = rule.get("release")
                if release is not None and state["effects_enabled"]:
                    returned = current_driver_displacement[trigger_index] <= float(
                        release["maximum_driver_displacement"]
                    )
                    state["release_ticks"] = state["release_ticks"] + 1 if returned else 0
                    if state["release_ticks"] >= int(
                        round(float(release.get("minimum_dwell_s", 0.0)) / DT)
                    ):
                        if not state["released"]:
                            state["released"] = True
                            state["released_tick"] = int(tick)
                            state["release_triggers"] += 1
                # Later phases override earlier ones per joint, so a release that
                # names only some effect joints leaves the rest holding their
                # first endpoint instead of silently going uncommanded.
                active: dict[str, dict[str, Any]] = {
                    str(effect["joint"]): effect for effect in rule["effects"]
                }
                if release is not None and state["released"]:
                    active.update(
                        {str(effect["joint"]): effect for effect in release["effects"]}
                    )
                for effect in active.values():
                    joint_index = object_joints[effect["joint"]]
                    target = float(effect["target"]) if state["effects_enabled"] else float(initial.get(effect["joint"], 0.0))
                    p.setJointMotorControl2(
                        object_body,
                        joint_index,
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=float(effect["maximum_force_or_torque"]),
                        positionGain=0.08,
                        velocityGain=1.0,
                        physicsClientId=client,
                    )

            # This is intentionally the final object-controller write before
            # stepping physics. The interaction-specific real-contact gate
            # enables it, and its target is determined by the measured
            # dense-sample parameter along the current IK path, not elapsed
            # schedule time or joint-space arc length. If the arm stalls away
            # from the path, progress and therefore object motion stop.
            if active_robot_driver_joint is not None and stage_index is not None:
                active_index = int(stage_index)
                active_plan = plans[active_index]
                actual_arm_q = np.asarray(
                    [
                        p.getJointState(
                            robot_body, joint, physicsClientId=client
                        )[0]
                        for joint in arm
                    ],
                    dtype=np.float64,
                )
                maximum_sample = int(
                    command.get("sample", len(active_plan.manipulation) - 1)
                )
                measured_progress, progress_residual = _robot_path_sample_progress(
                    active_plan.manipulation,
                    actual_arm_q,
                    maximum_sample,
                )
                if (
                    progress_residual
                    <= CONTACT_GATED_PROGRESS_MAXIMUM_RESIDUAL_RAD
                ):
                    actuation_progress[active_index] = max(
                        float(actuation_progress[active_index]),
                        float(measured_progress),
                    )
                driver_index = object_joints[active_robot_driver_joint]
                start_position = actuation_start_positions.setdefault(
                    active_index,
                    float(
                        p.getJointState(
                            object_body, driver_index, physicsClientId=client
                        )[0]
                    ),
                )
                target_position = float(
                    start_position
                    + float(_smoothstep01(actuation_progress[active_index]))
                    * (
                        float(active_plan.stage["target_joint_position"])
                        - start_position
                    )
                )
                actuation_targets[active_index] = target_position
                p.setJointMotorControl2(
                    object_body,
                    driver_index,
                    p.POSITION_CONTROL,
                    targetPosition=target_position,
                    force=CONTACT_GATED_ACTUATION_FORCE_OR_TORQUE,
                    positionGain=CONTACT_GATED_ACTUATION_POSITION_GAIN,
                    velocityGain=1.0,
                    physicsClientId=client,
                )
                if phase == "hold":
                    completed_driver_holds[active_robot_driver_joint] = (
                        target_position
                    )

            p.stepSimulation(physicsClientId=client)
            p.performCollisionDetection(physicsClientId=client)
            all_object_contacts = object_contact_points(
                robot_body, object_body, client
            )
            if stage_index is not None:
                acquisition_plan = plans[int(stage_index)]
                acquisition_key = _grasp_verification_key(
                    acquisition_plan.stage, int(stage_index)
                )
                if (
                    condition == "physical"
                    and acquisition_plan.stage["interaction"]
                    == "explicit_ideal_feasibility"
                    and acquisition_key not in verified_grasps
                    and phase in {"contact_settle", "contact_attach"}
                ):
                    allowed_robot, allowed_object = allowed_by_stage[
                        int(stage_index)
                    ]
                    sample = _bilateral_grasp_sample(
                        all_object_contacts,
                        allowed_robot,
                        allowed_object,
                        [
                            p.getJointState(
                                robot_body, joint, physicsClientId=client
                            )
                            for joint in fingers
                        ],
                        float(
                            acquisition_plan.stage["contact_acquisition"][
                                "approach_finger_opening_m"
                            ]
                        ),
                        float(command["finger"]),
                    )
                    state = acquisition_states[acquisition_key]
                    state["last_sample"] = sample
                    state["consecutive_pass_ticks"] = (
                        int(state["consecutive_pass_ticks"]) + 1
                        if sample["passed"]
                        else 0
                    )
                    state["maximum_consecutive_pass_ticks"] = max(
                        int(state["maximum_consecutive_pass_ticks"]),
                        int(state["consecutive_pass_ticks"]),
                    )
                    if (
                        int(state["consecutive_pass_ticks"])
                        >= acquisition_required_ticks
                    ):
                        # The real bilateral dwell is the only transition into
                        # semantic actuation. No attachment is created. The
                        # verified fingers and their nearest common rigid palm
                        # parent stop responding only to the target link, so
                        # the task actuator and contact solver do not form a
                        # redundant closed loop. The rest of the arm and every
                        # non-target object link remain collision-authoritative.
                        verified_grasps.add(acquisition_key)
                        disabled_pairs: list[tuple[int, int]] = []
                        verified_gripper_links = _verified_gripper_collision_links(
                            robot_body,
                            allowed_robot,
                            client,
                        )
                        for robot_link in sorted(verified_gripper_links):
                            p.setCollisionFilterPair(
                                robot_body,
                                object_body,
                                robot_link,
                                allowed_object,
                                0,
                                physicsClientId=client,
                            )
                            disabled_pairs.append((robot_link, allowed_object))
                        verified_disabled_collision_pairs[acquisition_key] = (
                            disabled_pairs
                        )
                        state["acquired"] = True
                        state["verified_tick"] = int(tick)
                    elif phase == "contact_attach":
                            # Do not execute even one manipulation command after
                            # a failed physical acquisition.  The negative
                            # control later receives this exact command prefix.
                            state["failed_verification_tick"] = int(tick)
                            aborted_after_failed_acquisition = True
            command_q = np.asarray(command["arm"], dtype=np.float64)
            actual_q = np.asarray(
                [
                    p.getJointState(
                        robot_body, joint, physicsClientId=client
                    )[0]
                    for joint in arm
                ],
                dtype=np.float64,
            )
            tracking_error = np.abs(actual_q - command_q)
            worst_tracking_joint = int(np.argmax(tracking_error))
            worst_tracking_error = float(tracking_error[worst_tracking_joint])
            if worst_tracking_error > float(
                robot_tracking["maximum_command_tracking_error_rad"]
            ):
                robot_tracking["maximum_command_tracking_error_rad"] = (
                    worst_tracking_error
                )
                robot_tracking["maximum_command_tracking_error_tick"] = int(tick)
                robot_tracking["maximum_command_tracking_error_joint"] = str(
                    robot_spec["arm_joint_names"][worst_tracking_joint]
                )
            commanded_step = float(np.max(np.abs(command_q - previous_command_q)))
            if commanded_step > float(
                robot_tracking["maximum_commanded_joint_step_rad"]
            ):
                robot_tracking["maximum_commanded_joint_step_rad"] = commanded_step
                robot_tracking["maximum_commanded_joint_step_tick"] = int(tick)
            actual_step = float(np.max(np.abs(actual_q - previous_actual_q)))
            if actual_step > float(robot_tracking["maximum_actual_joint_step_rad"]):
                robot_tracking["maximum_actual_joint_step_rad"] = actual_step
                robot_tracking["maximum_actual_joint_step_tick"] = int(tick)
            previous_command_q = command_q
            previous_actual_q = actual_q
            for name, index in object_joints.items():
                histories[name].append(float(p.getJointState(object_body, index, physicsClientId=client)[0]))

            # Every stage's driver offset is refreshed each tick, including while
            # no stage is active: a release rule fires during retreat/settle, so
            # tying this to the active stage would make it unobservable.
            for index, plan in enumerate(plans):
                current_driver_displacement[index] = abs(
                    histories[plan.stage["driver_joint"]][-1] - initial_driver[plan.stage["id"]]
                )

            if stage_index is not None:
                index = int(stage_index)
                allowed_robot, allowed_object = allowed_by_stage[index]
                metric = stage_metrics[index]
                metric["maximum_driver_displacement"] = max(
                    float(metric["maximum_driver_displacement"]),
                    current_driver_displacement[index],
                )
                metric["maximum_robot_execution_progress"] = max(
                    float(metric["maximum_robot_execution_progress"]),
                    float(actuation_progress[index]),
                )
                if index in actuation_targets:
                    metric["last_contact_gated_actuation_target"] = float(
                        actuation_targets[index]
                    )
                driver_state = p.getJointState(
                    object_body,
                    object_joints[plans[index].stage["driver_joint"]],
                    physicsClientId=client,
                )
                metric["maximum_absolute_driver_motor_torque"] = max(
                    float(metric["maximum_absolute_driver_motor_torque"]),
                    abs(float(driver_state[3])),
                )
                phase_torque = metric["driver_motor_torque_by_phase"].setdefault(
                    phase,
                    {"maximum_absolute": 0.0, "last": 0.0, "samples": 0},
                )
                phase_torque["maximum_absolute"] = max(
                    float(phase_torque["maximum_absolute"]),
                    abs(float(driver_state[3])),
                )
                phase_torque["last"] = float(driver_state[3])
                phase_torque["samples"] = int(phase_torque["samples"]) + 1
                if (
                    stage_index is not None
                    and int(stage_index) == index
                    and active_robot_driver_joint
                    == str(plans[index].stage["driver_joint"])
                    and index in actuation_targets
                ):
                    metric["maximum_contact_gated_tracking_error"] = max(
                        float(metric["maximum_contact_gated_tracking_error"]),
                        abs(float(driver_state[0]) - float(actuation_targets[index])),
                    )
                grasp_key = _grasp_verification_key(
                    plans[index].stage, index
                )
                verified_contact_active = bool(
                    plans[index].stage["interaction"]
                    == "explicit_ideal_feasibility"
                    and grasp_key in verified_grasps
                    and condition == "physical"
                )
                target_observed = verified_contact_active
                if verified_contact_active:
                    # Once the physical dwell gate passes, the uninterrupted
                    # grasp remains the disclosed causal gate until release.
                    metric["target_contact_observations"] += 1
                for point in all_object_contacts:
                    robot_link, object_link = int(point[3]), int(point[4])
                    if robot_link in allowed_robot and object_link == allowed_object:
                        if not verified_contact_active:
                            metric["target_contact_observations"] += 1
                        target_observed = True
                    else:
                        metric["non_target_contact_observations"] += 1
                        if object_link in effect_links:
                            metric["effect_link_contact_observations"] += 1
                        # Name the offending pair.  A bare count tells an agent
                        # that something collided but not what, which is not
                        # enough to repair a contact candidate.
                        key = (
                            robot_link_names.get(robot_link, str(robot_link)),
                            object_link_names.get(object_link, str(object_link)),
                            phase,
                        )
                        record = metric["unexpected_contact_pairs"].setdefault(
                            key, {"observations": 0, "deepest_penetration_m": 0.0}
                        )
                        record["observations"] += 1
                        record["deepest_penetration_m"] = min(
                            float(record["deepest_penetration_m"]), float(point[8])
                        )
                if robot_support_body is not None:
                    for point in object_contact_points(
                        robot_support_body, object_body, client
                    ):
                        object_link = int(point[4])
                        metric["non_target_contact_observations"] += 1
                        if object_link in effect_links:
                            metric["effect_link_contact_observations"] += 1
                        key = (
                            "robot_support",
                            object_link_names.get(object_link, str(object_link)),
                            phase,
                        )
                        record = metric["unexpected_contact_pairs"].setdefault(
                            key,
                            {
                                "observations": 0,
                                "deepest_penetration_m": 0.0,
                            },
                        )
                        record["observations"] += 1
                        record["deepest_penetration_m"] = min(
                            float(record["deepest_penetration_m"]), float(point[8])
                        )
                if target_observed:
                    if metric["first_target_contact_tick"] is None:
                        metric["first_target_contact_tick"] = int(tick)
                    # A raw fingertip contact during closing is not permission
                    # to unlatch an explicit ideal interaction.  Release that
                    # object driver only once the contact gate is active and the
                    # first manipulation command has begun.  This prevents a
                    # door from moving backwards before the grasp is complete.
                    if (
                        plans[index].stage["interaction"]
                        != "explicit_ideal_feasibility"
                        or (verified_contact_active and phase == "manipulate")
                    ):
                        contact_released_driver_joints.add(
                            str(plans[index].stage["driver_joint"])
                        )
                    current_contact_ticks[index] += 1
                    metric["maximum_continuous_contact_ticks"] = max(
                        int(metric["maximum_continuous_contact_ticks"]), current_contact_ticks[index]
                    )
                else:
                    current_contact_ticks[index] = 0

            if writer is not None and tick % CAPTURE_EVERY == 0:
                target_world = None
                if stage_index is not None:
                    active_plan = plans[int(stage_index)]
                    target_world, _ = _target_pose(
                        object_body,
                        object_links[active_plan.stage["contact_link"]],
                        active_plan.stage["contact_pose_link"],
                        0.0,
                        client,
                    )
                writer.append_data(
                    _render_frame(
                        client,
                        execution,
                        (
                            (
                                "FULL CONTACT-GATED ROLLOUT (IK DIAGNOSTICS RECORDED)"
                                if debug_partial
                                else "VERIFIED CONTACT-GATED IK/TRAJECTORY FEASIBILITY"
                            )
                            if any(
                                item.stage["interaction"] == "explicit_ideal_feasibility"
                                for item in plans
                            )
                            else (
                                "FULL PHYSICAL ROLLOUT (IK DIAGNOSTICS RECORDED)"
                                if debug_partial
                                else "PHYSICAL"
                            )
                        )
                        + f" | phase={phase} | stage={stage_index}",
                        phase,
                        target_world,
                        frame_renderer,
                    )
                )

            if aborted_after_failed_acquisition:
                break

        for grasp_key in list(verified_grasps):
            release_verified_grasp(grasp_key)
        summarized = []
        diagnostics = []
        for metric in stage_metrics:
            ticks = int(metric.pop("maximum_continuous_contact_ticks"))
            # The offending pairs are diagnostics, not acceptance metrics: they
            # stay out of `contacts`, whose stable fields describe target-contact
            # evidence consumed by the delivery verifier.
            pairs = metric.pop("unexpected_contact_pairs")
            metric["continuous_contact_s"] = ticks * DT
            summarized.append(metric)
            diagnostics.append(
                {
                    "stage_id": metric["stage_id"],
                    "unexpected_contact_pairs": sorted(
                        (
                            {
                                "robot_link": robot_link,
                                "object_link": object_link,
                                "phase": pair_phase,
                                **record,
                            }
                            for (robot_link, object_link, pair_phase), record in pairs.items()
                        ),
                        key=lambda item: -int(item["observations"]),
                    ),
                }
            )
        first_motion_ticks = _joint_first_motion_ticks(histories, initial)
        maximum_initial_displacements = _joint_maximum_initial_displacements(
            histories, initial
        )
        causal_timing = []
        for rule, state in zip(execution.get("causal_rules", []), rule_states):
            trigger_index = next(
                index
                for index, plan in enumerate(plans)
                if plan.stage["id"] == rule["trigger_stage"]
            )
            first_contact_tick = stage_metrics[trigger_index][
                "first_target_contact_tick"
            ]
            driver_joint = str(plans[trigger_index].stage["driver_joint"])
            driver_motion_tick = first_motion_ticks.get(driver_joint)
            effect_motion_ticks = {
                str(effect["joint"]): first_motion_ticks.get(str(effect["joint"]))
                for effect in rule["effects"]
            }
            effect_enable_tick = state["effects_enabled_tick"]
            causal_timing.append(
                {
                    "rule_id": str(rule["id"]),
                    "trigger_stage": str(rule["trigger_stage"]),
                    "driver_joint": driver_joint,
                    "first_target_contact_tick": first_contact_tick,
                    "driver_first_motion_tick": driver_motion_tick,
                    "latched_tick": state["latched_tick"],
                    "effects_enabled_tick": effect_enable_tick,
                    "effect_first_motion_ticks": effect_motion_ticks,
                    "effect_enabled_after_target_contact": (
                        effect_enable_tick is None
                        or (
                            first_contact_tick is not None
                            and int(effect_enable_tick) > int(first_contact_tick)
                        )
                    ),
                    "effect_motion_after_target_contact": all(
                        motion_tick is None
                        or (
                            first_contact_tick is not None
                            and int(motion_tick) > int(first_contact_tick)
                        )
                        for motion_tick in effect_motion_ticks.values()
                    ),
                    "driver_motion_after_target_contact": (
                        driver_motion_tick is None
                        or (
                            first_contact_tick is not None
                            and int(driver_motion_tick) >= int(first_contact_tick)
                        )
                    ),
                }
            )
        return {
            "condition": condition,
            "contacts": summarized,
            "contact_diagnostics": diagnostics,
            "joint_history": histories,
            "causal_triggers": sum(int(state["triggers"]) for state in rule_states),
            "causal_rule_states": [
                {
                    "rule_id": str(rule["id"]),
                    "trigger_stage": str(rule["trigger_stage"]),
                    "source_effect_phase": str(rule["source_effect_phase"]),
                    "source_effect_controls": [
                        int(effect["source_control_index"]) for effect in rule["effects"]
                    ],
                    "triggered": bool(state["latched"]),
                    "latched_tick": state["latched_tick"],
                    "effects_enabled": bool(state["effects_enabled"]),
                    "effects_enabled_tick": state["effects_enabled_tick"],
                    "release_declared": rule.get("release") is not None,
                    "release_source_effect_phase": (
                        str(rule["release"]["source_effect_phase"])
                        if rule.get("release") is not None
                        else None
                    ),
                    "released": bool(state["released"]),
                    "released_tick": state["released_tick"],
                }
                for rule, state in zip(execution.get("causal_rules", []), rule_states)
            ],
            "causal_timing": causal_timing,
            "object_joint_first_motion_ticks": first_motion_ticks,
            "object_joint_maximum_initial_displacements": maximum_initial_displacements,
            "created_fixed_constraints": 0,
            "verified_grasp_count": sum(
                bool(state["acquired"]) for state in acquisition_states.values()
            ),
            "required_verified_grasps": len(required_grasp_keys),
            "grasp_acquisition": [
                {
                    "grasp_key": key,
                    "required_consecutive_ticks": acquisition_required_ticks,
                    "required_dwell_s": GRASP_ACQUISITION_DWELL_S,
                    **state,
                }
                for key, state in sorted(acquisition_states.items())
            ],
            "aborted_after_failed_grasp_acquisition": bool(
                aborted_after_failed_acquisition
            ),
            "executed_command_count": int(executed_command_count),
            "maximum_runtime_constraint_count": 0,
            "object_joint_resets_after_initialization": 0,
            "undeclared_object_joints": undeclared_object_joints,
            "robot_command_schedule_sha256": _canonical_hash(
                scheduled_commands[:executed_command_count]
            ),
            "robot_tracking": robot_tracking,
            "video_encoding": video_encoding,
        }
    finally:
        if writer is not None:
            writer.close()
        p.disconnect(client)


def _joint_motion(
    requests: list[dict[str, Any]], initial: dict[str, float], histories: dict[str, list[float]]
) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for request in requests:
        grouped.setdefault(request["joint"], []).append(float(request["target"]))
    result: dict[str, Any] = {}
    for joint, targets in grouped.items():
        history = histories.get(joint, [])
        start = float(initial.get(joint, 0.0))
        cursor = 0
        ratios: list[float] = []
        observed: list[float] = []
        order_passed = True
        for target in targets:
            segment = history[cursor:]
            if not segment:
                ratios.append(0.0)
                observed.append(start)
                order_passed = False
                continue
            denominator = abs(target - start)
            if denominator <= 1e-9:
                errors = [abs(value - target) for value in segment]
                best_offset = int(np.argmin(errors))
                ratio = 1.0 if errors[best_offset] <= 1e-3 else 0.0
            else:
                progress = [1.0 - abs(value - target) / denominator for value in segment]
                best_offset = int(np.argmax(progress))
                ratio = float(np.clip(progress[best_offset], 0.0, 1.0))
            best = float(segment[best_offset])
            ratios.append(ratio)
            observed.append(best)
            if ratio < 0.9:
                order_passed = False
            cursor += best_offset
            start = target
        result[joint] = {
            "requested_extrema": targets,
            "observed_extrema": observed,
            "minimum_progress_ratio": min(ratios) if ratios else 0.0,
            "order_passed": order_passed,
        }
    return result


def _declared_object_joint_names(
    execution: dict[str, Any],
    plans: list[StagePlan],
) -> set[str]:
    """Return object joints whose motion has an explicit execution owner."""
    declared = {str(plan.stage["driver_joint"]) for plan in plans}
    declared.update(
        str(item["joint"]) for item in execution.get("passive_joints", [])
    )
    declared.update(
        str(effect["joint"])
        for rule in execution.get("causal_rules", [])
        for group in (rule, rule.get("release"))
        if isinstance(group, dict)
        for effect in group.get("effects", [])
    )
    return declared


def _precontact_driver_hold_targets(
    plans: list[StagePlan], initial: dict[str, float]
) -> dict[str, float]:
    """Freeze robot-owned object drivers until nominated contact is observed."""
    return {
        str(plan.stage["driver_joint"]): float(
            initial.get(str(plan.stage["driver_joint"]), 0.0)
        )
        for plan in plans
    }


def _passive_driver_is_enabled(
    joint: str,
    precontact_driver_holds: dict[str, float],
    contact_released_driver_joints: set[str],
) -> bool:
    """Permit return only after the robot physically released its driver."""
    return (
        joint not in precontact_driver_holds
        or joint in contact_released_driver_joints
    )


def _robot_driver_actuation_is_enabled(
    *,
    interaction: str,
    phase: str,
    condition: str,
    grasp_verified: bool,
    target_contact_observed: bool,
) -> bool:
    """Gate robot-owned object motion by the declared contact semantics."""
    if condition != "physical" or phase not in {"manipulate", "hold"}:
        return False
    if interaction == "explicit_ideal_feasibility":
        return bool(grasp_verified)
    if interaction == "physical_push":
        return bool(target_contact_observed)
    return False


def _object_joint_state_before_control(
    plan: dict[str, Any],
    initial: dict[str, float],
    stop_phase: str,
    stop_control_index: int,
) -> dict[str, float]:
    """Project plan endpoints before one exact timeline control.

    This is the shadow-world state used by whole-task placement.  It includes
    controls from earlier phases and earlier controls in ``stop_phase``, but it
    intentionally excludes the nominated control itself.
    """
    if stop_control_index < 0:
        raise ValueError("stop_control_index must be non-negative")
    state = {str(name): float(value) for name, value in initial.items()}
    for phase in plan.get("timeline", []):
        phase_name = str(phase["name"])
        controls = list(phase.get("controls", []))
        if phase_name == str(stop_phase):
            if stop_control_index > len(controls):
                raise ValueError(
                    f"Control index {stop_control_index} is outside phase "
                    f"{stop_phase!r} with {len(controls)} controls"
                )
            controls = controls[:stop_control_index]
            for control in controls:
                joint = str(control.get("joint", ""))
                target = artimo_plan.control_target(control)
                if joint and target is not None:
                    state[joint] = float(target)
            return state
        for control in controls:
            joint = str(control.get("joint", ""))
            target = artimo_plan.control_target(control)
            if joint and target is not None:
                state[joint] = float(target)
    raise ValueError(f"Unknown phase {stop_phase!r} in plan timeline")


def _object_joint_state_before_phase(
    plan: dict[str, Any],
    initial: dict[str, float],
    stop_before_phase: str,
) -> dict[str, float]:
    """Project every authoritative plan endpoint before a timeline boundary."""
    return _object_joint_state_before_control(
        plan, initial, stop_before_phase, 0
    )


def _object_joint_transitions_from_phase(
    plan: dict[str, Any],
    start_state: dict[str, float],
    start_phase: str,
    stop_before_phase: str | None = None,
) -> list[dict[str, Any]]:
    """Return plan sweeps after release, optionally stopping at reacquisition.

    ``stop_before_phase`` is exclusive.  A cross-contact release whose boundary
    is the next robot-contact phase therefore has no stationary post-release
    object sweep: its subsequent motion belongs to that next manipulation
    block, while the path between contacts belongs to the transit planner.
    """
    state = {str(name): float(value) for name, value in start_state.items()}
    transitions: list[dict[str, Any]] = []
    timeline = list(plan.get("timeline", []))
    phase_indices = {
        str(phase["name"]): index for index, phase in enumerate(timeline)
    }
    if str(start_phase) not in phase_indices:
        raise ValueError(f"Unknown release phase {start_phase!r} in plan timeline")
    start_index = phase_indices[str(start_phase)]
    stop_index = len(timeline)
    if stop_before_phase is not None:
        if str(stop_before_phase) not in phase_indices:
            raise ValueError(
                f"Unknown post-release stop phase {stop_before_phase!r} in plan timeline"
            )
        stop_index = phase_indices[str(stop_before_phase)]
        if stop_index < start_index:
            raise ValueError(
                f"Post-release stop phase {stop_before_phase!r} precedes "
                f"release phase {start_phase!r}"
            )
    for phase in timeline[start_index:stop_index]:
        phase_name = str(phase["name"])
        for control_index, control in enumerate(phase.get("controls", [])):
            joint = str(control.get("joint", ""))
            target = artimo_plan.control_target(control)
            if not joint or target is None:
                continue
            start = float(state.get(joint, 0.0))
            target = float(target)
            if abs(target - start) > 1e-12:
                transitions.append(
                    {
                        "phase": phase_name,
                        "control_index": int(control_index),
                        "joint": joint,
                        "start": start,
                        "target": target,
                    }
                )
            state[joint] = target
    return transitions


def _timeline_phase_order(object_plan: dict[str, Any]) -> dict[str, int]:
    """Map authoritative phase names without relying on rollout local names."""
    return {
        str(item["name"]): index
        for index, item in enumerate(object_plan.get("timeline", []))
    }


def _terminal_plan_hold_phase_index(
    object_plan: dict[str, Any],
    execution: dict[str, Any],
    stage: dict[str, Any],
) -> int | None:
    """Return the terminal hold phase retained by ``stage``, if any.

    A terminal plan-owned ``hold_position`` has no release boundary.  The
    robot therefore remains at the final manipulation command with the grasp
    active.  This helper is shared by path planning and scheduling so the
    planner cannot reject a candidate for a release/retreat/home segment that
    the scheduler will never execute.
    """
    timeline = list(object_plan.get("timeline", []))
    phase_order = _timeline_phase_order(object_plan)
    source_phase = str(stage.get("source_phase", ""))
    if source_phase not in phase_order:
        return None
    ownership = {
        (str(row["source_phase"]), int(row["source_control_index"])): row
        for row in execution.get("control_execution", [])
    }
    driver_joint = str(stage.get("driver_joint", ""))
    source_index = int(phase_order[source_phase])
    held_through = source_index
    for phase_index in range(source_index + 1, len(timeline)):
        phase = timeline[phase_index]
        phase_name = str(phase.get("name", ""))
        retains_driver = any(
            str(control.get("mode", "")) == "hold_position"
            and str(control.get("joint", "")) == driver_joint
            and ownership.get((phase_name, control_index), {}).get("motion_owner")
            == "hold"
            for control_index, control in enumerate(phase.get("controls", []))
        )
        if not retains_driver:
            break
        held_through = phase_index
    if held_through == len(timeline) - 1 and held_through > source_index:
        return held_through
    return None


def _joint_first_motion_ticks(
    histories: dict[str, list[float]],
    initial: dict[str, float],
    tolerance: float = OBJECT_JOINT_STABILITY_TOLERANCE_M_OR_RAD,
) -> dict[str, int | None]:
    """Return the first measured tick each object joint left its initial state."""
    answer: dict[str, int | None] = {}
    for name, history in histories.items():
        start = float(initial.get(name, 0.0))
        answer[name] = next(
            (
                int(tick)
                for tick, value in enumerate(history)
                if abs(float(value) - start) > float(tolerance)
            ),
            None,
        )
    return answer


def _joint_maximum_initial_displacements(
    histories: dict[str, list[float]], initial: dict[str, float]
) -> dict[str, float]:
    return {
        name: max(
            (abs(float(value) - float(initial.get(name, 0.0))) for value in history),
            default=0.0,
        )
        for name, history in histories.items()
    }


def _mechanism_signature(urdf: Path) -> dict[str, Any]:
    """Describe a URDF's kinematic tree, ignoring geometry entirely."""
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet mechanism client")
    try:
        body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=client)
        joints = []
        for index in range(p.getNumJoints(body, physicsClientId=client)):
            info = p.getJointInfo(body, index, physicsClientId=client)
            joints.append(
                {
                    "joint": info[1].decode("utf-8"),
                    "type": int(info[2]),
                    "child_link": info[12].decode("utf-8"),
                    "parent_index": int(info[16]),
                    "axis": [round(float(value), 9) for value in info[13]],
                    "lower": round(float(info[8]), 9),
                    "upper": round(float(info[9]), 9),
                }
            )
        return {"joints": joints}
    finally:
        p.disconnect(client)


def _require_matching_mechanism(source_urdf: Path, simulation_urdf: Path) -> None:
    """Refuse a physics URDF that alters the mechanism, not just its collisions.

    A collision proxy is only legitimate if the joints, axes, limits, and parent
    relationships still match the source the ArtiMo plan was authored against.
    Otherwise a "proxy" could quietly become an easier object to manipulate.
    """
    if source_urdf == simulation_urdf:
        return
    source = _mechanism_signature(source_urdf)
    simulation = _mechanism_signature(simulation_urdf)
    if source == simulation:
        return
    source_joints = {item["joint"]: item for item in source["joints"]}
    simulation_joints = {item["joint"]: item for item in simulation["joints"]}
    differences: list[str] = []
    for name in sorted(set(source_joints) | set(simulation_joints)):
        expected = source_joints.get(name)
        actual = simulation_joints.get(name)
        if expected is None:
            differences.append(f"{name}: absent from source URDF")
        elif actual is None:
            differences.append(f"{name}: absent from physics URDF")
        elif expected != actual:
            fields = sorted(k for k in expected if expected[k] != actual[k])
            differences.append(f"{name}: differs in {fields}")
    raise ValueError(
        "physics_urdf must preserve the source mechanism (joints, axes, limits, "
        f"parents); differences: {differences}"
    )


def resolve_simulation_urdf(
    task: dict[str, Any], execution: dict[str, Any], source_urdf: Path
) -> Path:
    """Return the locked source collision model (or an explicit task input).

    Collision representation is not agent-authored execution data.  In
    particular, do not automatically replace source meshes with convex hulls or
    V-HACD parts: those representations are not geometrically equivalent and
    can close real free space.  A separately locked task input remains valid,
    but ordinary runs use the source URDF byte-for-byte.
    """
    task_value = task["inputs"].get("physics_urdf")
    # Deliberately ignore a legacy execution.physics_urdf value.  Old debug
    # executions remain readable, but they cannot change collision semantics.
    simulation_urdf = _resolve(task_value) if task_value else source_urdf.resolve()
    _require_matching_mechanism(source_urdf.resolve(), simulation_urdf)
    return simulation_urdf


def run(
    task_spec_path: Path,
    execution_path: Path,
    out: Path,
    render: bool,
    allow_partial_debug: bool = False,
) -> dict[str, Any]:
    task = _read_json(task_spec_path)
    execution_input = _read_json(execution_path)
    execution = materialize_execution_defaults(task, execution_input)
    if task.get("schema_version") != 2 or execution.get("schema_version") != 2:
        raise ValueError("Expected task schema v2 and execution schema v2")
    inputs = task["inputs"]
    object_urdf = _resolve(inputs["urdf"])
    robot_urdf = _resolve(inputs["robot_urdf"])
    plan_path = _resolve(inputs["plan"])
    _validate_execution_schema(execution)
    # One application-owned collision model is shared by planning and rollout;
    # the source URDF remains the mechanism and visual contract.
    simulation_urdf = resolve_simulation_urdf(task, execution, object_urdf)
    for path in (object_urdf, simulation_urdf, robot_urdf, plan_path, execution_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan = _read_json(plan_path)
    initial = task_initial_joint_values(task)
    _require_matching_mechanism(object_urdf, simulation_urdf)
    coordinate_lock = _enforce_coordinate_frame_lock(execution_path, execution)
    grounding = _ground_execution_scene(simulation_urdf, robot_urdf, execution, initial)
    execution = grounding.pop("execution")
    requests = _plan_requests(plan)
    _validate_execution_against_plan(plan, execution)
    ik_path_solver = create_curobo_backend(execution, robot_urdf, execution["robot"])
    try:
        plans = _plan_stages(
            simulation_urdf,
            robot_urdf,
            execution,
            initial,
            allow_partial_debug=allow_partial_debug,
            object_plan=plan,
            ik_path_solver=ik_path_solver,
        )
    finally:
        if ik_path_solver is not None:
            ik_path_solver.close()
    ik_diagnostics = [
        plan.debug_failure for plan in plans if plan.debug_failure is not None
    ]
    commands = _schedule(plans, execution, plan)
    out.mkdir(parents=True, exist_ok=True)
    video_path = out / "video.mp4" if render else None
    physical = _rollout(
        simulation_urdf,
        robot_urdf,
        execution,
        initial,
        plan,
        plans,
        commands,
        "physical",
        video_path,
        debug_partial=bool(ik_diagnostics),
    )
    control = _rollout(
        simulation_urdf,
        robot_urdf,
        execution,
        initial,
        plan,
        plans,
        commands,
        "contact_disabled",
        None,
        debug_partial=bool(ik_diagnostics),
        maximum_command_count=int(physical["executed_command_count"]),
    )
    motion = _joint_motion(requests, initial, physical["joint_history"])
    requested_joints = {request["joint"] for request in requests}
    negative_remained_initial = all(
        max(
            (abs(value - float(initial.get(joint, 0.0))) for value in control["joint_history"].get(joint, [])),
            default=0.0,
        )
        <= 0.001
        for joint in requested_joints
    )
    negative_all_object_joints_initial = all(
        float(displacement) <= OBJECT_JOINT_STABILITY_TOLERANCE_M_OR_RAD
        for displacement in control[
            "object_joint_maximum_initial_displacements"
        ].values()
    )
    undeclared_joint_motion_zero = all(
        float(rollout["object_joint_maximum_initial_displacements"].get(joint, 0.0))
        <= OBJECT_JOINT_STABILITY_TOLERANCE_M_OR_RAD
        for rollout in (physical, control)
        for joint in rollout["undeclared_object_joints"]
    )
    minimum_ratio = float(task["acceptance"]["minimum_joint_motion_ratio"])
    zero_constraints_required = bool(task["acceptance"]["require_zero_fixed_constraints"])
    robot_owned_stage_ids = {
        str(row["stage_id"])
        for row in execution["control_execution"]
        if row["motion_owner"] == "robot_contact"
    }
    contacted_stage_ids = {
        str(item["stage_id"])
        for item in physical["contacts"]
        if int(item["target_contact_observations"]) > 0
    }
    checks = {
        "control_execution_ownership": True,
        "passive_return_after_release_retreat": all(
            not command.get("active_passive_joints")
            for command in commands
            if command["phase"]
            in {"contact_release", "retreat", "release_retreat_settle"}
            and command.get("stage") is not None
            and plans[int(command["stage"])].stage.get("release_before_phase")
        ),
        "release_retreat_has_nonzero_settle": all(
            any(
                command["phase"] == "release_retreat_settle"
                and command.get("stage") == stage_index
                for command in commands
            )
            for stage_index, item in enumerate(plans)
            if item.stage.get("release_before_phase") is not None
        ),
        "release_passive_swept_clearance": all(
            any(
                sample.get("phase") in {
                    "release_then_passive_sweep",
                    "release_then_plan_sweep",
                }
                and float(sample.get("distance_m", -math.inf)) > 0.0
                for sample in item.swept_clearance_violations
            )
            for item in plans
            if item.stage.get("release_before_phase") is not None
        ),
        "every_robot_owned_stage_contact": robot_owned_stage_ids == contacted_stage_ids,
        "ik_all_stages": all(
            not plan.debug_truncated
            and
            plan.maximum_position_error_m <= 0.004
            and math.degrees(plan.maximum_orientation_error_rad) <= 2.0
            for plan in plans
        ),
        "ik_joint_limits_not_saturated": all(
            plan.minimum_joint_limit_margin_rad > 1e-4
            for plan in plans
        ),
        "ik_command_continuity": all(
            plan.maximum_adjacent_joint_step_rad <= 0.0800001
            for plan in plans
        ),
        "planning_swept_clearance": all(
            plan.minimum_swept_clearance_m is None
            or plan.minimum_swept_clearance_m
            >= float(plan.stage.get("minimum_swept_clearance_m", 0.0))
            for plan in plans
        ),
        "all_requested_joint_motion": all(
            item["minimum_progress_ratio"] >= minimum_ratio and item["order_passed"]
            for item in motion.values()
        ),
        "target_contact_positive": all(item["target_contact_observations"] > 0 for item in physical["contacts"]),
        "target_contact_duration": all(
            item["continuous_contact_s"] >= float(task["acceptance"].get("minimum_continuous_contact_s", 0.0))
            for item in physical["contacts"]
        ),
        "physical_grasp_acquisition": (
            not physical["aborted_after_failed_grasp_acquisition"]
            and all(
                bool(item["acquired"])
                for item in physical["grasp_acquisition"]
            )
            and
            int(physical["verified_grasp_count"])
            == int(physical["required_verified_grasps"])
            and int(physical["maximum_runtime_constraint_count"]) == 0
        ),
        "contact_gated_object_tracking": all(
            float(metric["maximum_contact_gated_tracking_error"])
            <= CONTACT_GATED_OBJECT_TRACKING_TOLERANCE_RAD
            for metric, plan in zip(physical["contacts"], plans)
            if plan.stage["interaction"] == "explicit_ideal_feasibility"
        ),
        "non_target_contact_zero": all(item["non_target_contact_observations"] == 0 for item in physical["contacts"]),
        "effect_link_contact_zero": all(item["effect_link_contact_observations"] == 0 for item in physical["contacts"]),
        "negative_target_contact_zero": all(item["target_contact_observations"] == 0 for item in control["contacts"]),
        "negative_no_causal_trigger": control["causal_triggers"] == 0,
        "negative_requested_motion_initial": negative_remained_initial,
        "negative_all_object_joints_initial": negative_all_object_joints_initial,
        "undeclared_object_joint_motion_zero": undeclared_joint_motion_zero,
        "causal_effect_after_target_contact": all(
            bool(item["effect_enabled_after_target_contact"])
            and bool(item["effect_motion_after_target_contact"])
            and bool(item["driver_motion_after_target_contact"])
            for item in physical["causal_timing"]
        ),
        "same_robot_command_schedule": physical["robot_command_schedule_sha256"] == control["robot_command_schedule_sha256"],
        "no_runtime_object_resets": physical["object_joint_resets_after_initialization"] == 0 and control["object_joint_resets_after_initialization"] == 0,
        "constraint_policy": (
            physical["maximum_runtime_constraint_count"] == 0
            if zero_constraints_required
            else True
        ),
        "scene_grounded": bool(grounding["passed"]),
    }
    grasp = dict(execution)
    grasp_path = out / "grasp.json"
    grasp_path.write_text(json.dumps(grasp, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    diagnostics_passed = not ik_diagnostics and all(checks.values())
    result = {
        "schema_version": 2,
        "passed": bool(
            video_path is not None
            and video_path.is_file()
            and diagnostics_passed
        ),
        "video_exported": bool(video_path is not None and video_path.is_file()),
        "rollout_completed": bool(
            not physical["aborted_after_failed_grasp_acquisition"]
            and int(physical["executed_command_count"]) == len(commands)
        ),
        "diagnostics_passed": diagnostics_passed,
        "debug_partial_rollout": {
            "enabled": False,
            "physical_contact_only": True,
            "object_trajectory_replay": False,
            "truncations": [],
        },
        "ik_diagnostics": ik_diagnostics,
        "checks": checks,
        "physics_engine": "PyBullet",
        "object_trajectory_replay": False,
        "execution_plan_sha256": _sha256(grasp_path),
        "ik": [
            {
                "stage_id": plan.stage["id"],
                "maximum_position_error_m": plan.maximum_position_error_m,
                "maximum_orientation_error_deg": math.degrees(plan.maximum_orientation_error_rad),
                "minimum_swept_clearance_m": plan.minimum_swept_clearance_m,
                "minimum_joint_limit_margin_rad": plan.minimum_joint_limit_margin_rad,
                "minimum_joint_limit_margin_sample": plan.minimum_joint_limit_margin_sample,
                "minimum_joint_limit_margin_joint": plan.minimum_joint_limit_margin_joint,
                "maximum_adjacent_joint_step_rad": plan.maximum_adjacent_joint_step_rad,
                "ik_backend": plan.ik_backend,
                "ik_backend_fallback_reason": plan.ik_backend_fallback_reason,
                "transit_planner_backend": plan.transit_planner_backend,
                "transit_planner_evidence": plan.transit_planner_evidence,
                "tightest_swept_samples": plan.swept_clearance_violations,
                "debug_truncated": plan.debug_truncated,
                "debug_failure": plan.debug_failure,
            }
            for plan in plans
        ],
        "physical": {**{key: value for key, value in physical.items() if key != "joint_history"}, "joint_motion": motion},
        "negative_control": {
            **{key: value for key, value in control.items() if key != "joint_history"},
            "requested_joint_motion_remained_initial": negative_remained_initial,
        },
        "inputs": {
            "task_spec_sha256": _sha256(task_spec_path),
            "execution_input_sha256": _sha256(execution_path),
            "object_urdf_sha256": _sha256(object_urdf),
            "simulated_urdf_sha256": _sha256(simulation_urdf),
            "collision_proxy_used": simulation_urdf != object_urdf,
            "collision_model_policy": (
                "task_locked_collision_model"
                if simulation_urdf != object_urdf
                else "source_urdf_meshes"
            ),
            "robot_urdf_sha256": _sha256(robot_urdf),
            "initial_joint_state_source": (
                "trajectory_first_frame"
                if inputs.get("trajectory") is not None
                else "urdf_default_zero"
            ),
        },
        "scene_grounding": grounding,
        "coordinate_frame_lock": coordinate_lock,
    }
    (out / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--allow-partial-debug-rollout",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    try:
        result = run(
            args.task_spec.expanduser().resolve(),
            args.execution.expanduser().resolve(),
            args.out.expanduser().resolve(),
            args.render,
            args.allow_partial_debug_rollout,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Generic ArtiMo physics failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
