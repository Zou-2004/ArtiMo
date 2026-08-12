#!/usr/bin/env python3
"""Render the full grounded scene for one execution JSON.

Shows where the object is, where the robot is, the declared contact point, the
approach axis, and the solved arm configuration at several points along the
stage's object path.  This is the picture you need before believing any IK
number: a candidate that "fails IK" usually looks obviously wrong on screen.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

APP_ROOT = Path(__file__).resolve().parent
REPO = APP_ROOT.parents[1]
sys.path.insert(0, str(REPO / "tools"))

import run_artimo_physics as ph  # noqa: E402
from artimo_ik import BulletIK, set_fingers, set_robot_arm  # noqa: E402


def marker(position, radius, color, client):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    return p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=list(position), physicsClientId=client)


def arrow(start, direction, length, color, client, count=9):
    for i in range(1, count + 1):
        t = length * i / count
        marker([start[a] + direction[a] * t for a in range(3)], 0.006, color, client)


def view(target, distance, yaw, pitch, label, client, width=760, height=580):
    v = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0.0, 2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(52.0, width / height, 0.02, 12.0)
    _, _, rgba, _, _ = p.getCameraImage(width, height, viewMatrix=v, projectionMatrix=proj,
                                        renderer=p.ER_TINY_RENDERER, physicsClientId=client)
    img = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3])
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width, 26), fill=(16, 20, 27))
    d.text((8, 7), label, fill=(240, 245, 250), font=ImageFont.load_default())
    return img


def look_at_view(
    target,
    eye_direction,
    distance,
    label,
    client,
    width=760,
    height=580,
):
    """Render a contact-frame view without assuming an asset/world axis."""
    target_array = np.asarray(target, dtype=np.float64)
    direction = np.asarray(eye_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        raise ValueError("eye_direction must be non-zero")
    direction /= norm
    eye = target_array + direction * float(distance)
    forward = target_array - eye
    forward /= np.linalg.norm(forward)
    up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.92:
        up = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    v = p.computeViewMatrix(
        cameraEyePosition=eye.tolist(),
        cameraTargetPosition=target_array.tolist(),
        cameraUpVector=up.tolist(),
        physicsClientId=client,
    )
    proj = p.computeProjectionMatrixFOV(48.0, width / height, 0.01, 4.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width,
        height,
        viewMatrix=v,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=client,
    )
    img = Image.fromarray(
        np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    )
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, width, 26), fill=(16, 20, 27))
    d.text((8, 7), label, fill=(240, 245, 250), font=ImageFont.load_default())
    return img


def _set_body_visibility(body, visible_links, client):
    """Hide every visual except selected links and return colors for restore."""
    colors = {}
    for shape in p.getVisualShapeData(body, physicsClientId=client) or []:
        link = int(shape[1])
        colors.setdefault(link, list(shape[7]))
    for link, rgba in colors.items():
        color = rgba if link in visible_links else [rgba[0], rgba[1], rgba[2], 0.0]
        p.changeVisualShape(body, link, rgbaColor=color, physicsClientId=client)
    return colors


def _restore_body_visibility(body, colors, client):
    for link, rgba in colors.items():
        p.changeVisualShape(body, link, rgbaColor=rgba, physicsClientId=client)


def _create_parallel_jaw_proxy(target_position, target_quaternion, opening, client):
    """Draw a kinematic-free Panda-like jaw proxy at one declared grasp pose."""
    target = np.asarray(target_position, dtype=np.float64)
    closing_axis = np.asarray(
        p.rotateVector(target_quaternion, [0.0, 1.0, 0.0]), dtype=np.float64
    )
    forward = np.asarray(
        p.rotateVector(target_quaternion, [0.0, 0.0, 1.0]), dtype=np.float64
    )
    finger_half = [0.006, 0.005, 0.035]
    finger_center = target - forward * 0.035
    separation = float(opening) + finger_half[1]
    bodies = []
    for sign, color in (
        (-1.0, [0.1, 0.85, 1.0, 1.0]),
        (1.0, [1.0, 0.2, 0.75, 1.0]),
    ):
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=finger_half,
            rgbaColor=color,
            physicsClientId=client,
        )
        bodies.append(
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=(finger_center + sign * separation * closing_axis).tolist(),
                baseOrientation=list(target_quaternion),
                physicsClientId=client,
            )
        )
    return bodies


def _single_tool_push_preview(stage):
    """Return whether visual-only QA must show one nominated robot tool link."""
    acquisition = stage.get("contact_acquisition", {})
    return (
        acquisition.get("mode") == "maintain_width"
        and stage.get("interaction") == "physical_push"
        and len(stage.get("allowed_robot_contact_links", [])) == 1
        and stage.get("robot_tool_contact_offset_eef_m") is not None
    )


def _tool_com_pose_at_target_eef(
    robot,
    end_effector_link,
    tool_link,
    target_position,
    target_quaternion,
    client,
):
    """Transport a robot link COM's current EEF-relative pose to a target EEF."""
    eef_state = p.getLinkState(
        robot, end_effector_link, computeForwardKinematics=True, physicsClientId=client
    )
    tool_state = p.getLinkState(
        robot, tool_link, computeForwardKinematics=True, physicsClientId=client
    )
    inverse_eef = p.invertTransform(eef_state[4], eef_state[5])
    eef_to_tool_com = p.multiplyTransforms(
        inverse_eef[0], inverse_eef[1], tool_state[0], tool_state[1]
    )
    return p.multiplyTransforms(
        list(target_position),
        list(target_quaternion),
        eef_to_tool_com[0],
        eef_to_tool_com[1],
    )


def _collision_shape_as_visual(shape, color, client):
    """Clone one PyBullet collision-shape description as visual-only geometry."""
    geometry_type = int(shape[2])
    dimensions = list(shape[3])
    filename = shape[4]
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8")
    common = {"shapeType": geometry_type, "rgbaColor": color, "physicsClientId": client}
    if geometry_type == p.GEOM_MESH:
        return p.createVisualShape(
            **common, fileName=filename, meshScale=dimensions
        )
    if geometry_type == p.GEOM_BOX:
        return p.createVisualShape(**common, halfExtents=dimensions)
    if geometry_type == p.GEOM_SPHERE:
        return p.createVisualShape(**common, radius=float(dimensions[0]))
    if geometry_type in (p.GEOM_CAPSULE, p.GEOM_CYLINDER):
        return p.createVisualShape(
            **common, radius=float(dimensions[1]), length=float(dimensions[0])
        )
    raise ValueError(f"Unsupported robot tool collision geometry type {geometry_type}")


def _create_single_robot_tool_proxy(
    robot,
    end_effector_link,
    tool_link,
    target_position,
    target_quaternion,
    tool_contact_offset_eef_m,
    client,
):
    """Draw the nominated robot collision link at a target EEF pose, without IK."""
    # getCollisionShapeData local frames are relative to the link inertial/COM
    # frame, not the URDF link frame.  Transport the COM pose before applying
    # each returned shape-local transform; using getLinkState()[4:6] here
    # would apply the inertial offset twice and visibly detach the tool face.
    tool_position, tool_quaternion = _tool_com_pose_at_target_eef(
        robot,
        end_effector_link,
        tool_link,
        target_position,
        target_quaternion,
        client,
    )
    color = [0.1, 0.85, 1.0, 1.0]
    bodies = []
    shapes = p.getCollisionShapeData(robot, tool_link, physicsClientId=client) or []
    if not shapes:
        raise ValueError("Nominated robot tool link has no collision geometry to preview")
    for shape in shapes:
        visual = _collision_shape_as_visual(shape, color, client)
        shape_position, shape_quaternion = p.multiplyTransforms(
            tool_position,
            tool_quaternion,
            shape[5],
            shape[6],
        )
        bodies.append(
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual,
                basePosition=shape_position,
                baseOrientation=shape_quaternion,
                physicsClientId=client,
            )
        )
    offset_world = p.rotateVector(target_quaternion, tool_contact_offset_eef_m)
    contact_world = np.asarray(target_position, dtype=np.float64) + np.asarray(
        offset_world, dtype=np.float64
    )
    bodies.append(marker(contact_world, 0.003, color, client))
    return bodies


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-spec", type=Path, required=True)
    ap.add_argument("--execution", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stage", type=int, default=0)
    ap.add_argument(
        "--video",
        type=Path,
        help="Optional labelled turntable MP4 for placement QA; this is not a physical rollout",
    )
    ap.add_argument(
        "--maximum-target-gap-m",
        type=float,
        default=0.006,
        help="Largest per-contact-link gap that still counts as grasp geometry",
    )
    ap.add_argument(
        "--orientation-only-no-ik",
        action="store_true",
        help=(
            "Render separate target+jaw-proxy orientation views without running "
            "IK or numerical contact/clearance probes."
        ),
    )
    args = ap.parse_args()

    task = json.loads(args.task_spec.read_text())
    ex = json.loads(args.execution.read_text())
    inp = task["inputs"]
    source_urdf = ph._resolve(inp["urdf"])
    sim_urdf = ph.resolve_simulation_urdf(task, ex, source_urdf)
    ph._require_matching_mechanism(source_urdf, sim_urdf)
    robot_urdf = ph._resolve(inp["robot_urdf"])
    init = ph.task_initial_joint_values(task)

    g = ph._ground_execution_scene(sim_urdf, robot_urdf, ex, init)
    ex = g.pop("execution")
    args.out.mkdir(parents=True, exist_ok=True)

    stage = ex["stages"][args.stage]
    stage_start_state = dict(init)
    for prior in ex["stages"][: args.stage]:
        stage_start_state[prior["driver_joint"]] = float(
            prior.get("command_joint_position", prior["target_joint_position"])
        )
    report = {
        "grounded_object_base_m": ex["scene"]["object_base_translation_m"],
        "grounded_robot_base_m": ex["robot"]["base_translation_m"],
        "robot_base_rotation_xyzw": ex["robot"]["base_rotation_xyzw"],
        "contact_link": stage["contact_link"],
        "contact_point_link_m": stage["contact_pose_link"]["translation_m"],
        "driver_joint": stage["driver_joint"],
        "contact_acquisition": stage["contact_acquisition"],
        "prior_stage_joint_state": stage_start_state,
        "maximum_target_gap_m": float(args.maximum_target_gap_m),
        "samples": [],
    }

    client = p.connect(p.DIRECT)
    ob, rb, robot_support = ph._load_scene(sim_urdf, robot_urdf, ex, client, True)
    oj, ol = ph._maps(ob, client)
    rj, rl = ph._maps(rb, client)
    inv_r = {v: k for k, v in rl.items()}
    inv_o = {v: k for k, v in ol.items()}
    for n, v in stage_start_state.items():
        if n in oj:
            p.resetJointState(ob, oj[n], v, physicsClientId=client)
    arm = [rj[n] for n in ex["robot"]["arm_joint_names"]]
    fingers = [rj[n] for n in ex["robot"]["finger_joint_names"]]
    opening = float(stage["finger_opening_m"])
    set_fingers(rb, fingers, opening, client)
    home = np.asarray(ex["robot"]["home_joint_positions"], dtype=np.float64)
    set_robot_arm(rb, arm, home, client)

    # Highlight the contacted link so it is unmistakable on screen.
    p.changeVisualShape(ob, ol[stage["contact_link"]], rgbaColor=[1.0, 0.7, 0.1, 1.0], physicsClientId=client)

    solver = BulletIK(rb, arm, rl[ex["robot"]["end_effector_link"]], fingers, opening,
                      {"random_seed": int(ex["seeds"].get("ik", 0)),
                       "random_restarts": int(ex.get("ik", {}).get("random_restarts", 256)),
                       "max_iterations": int(ex.get("ik", {}).get("max_iterations", 3000)),
                       "position_tolerance_m": 0.001, "orientation_tolerance_deg": 1.0,
                       "max_joint_step_rad": 1.2}, client)

    driver = oj[stage["driver_joint"]]
    start = float(stage_start_state.get(stage["driver_joint"], 0.0))
    target = float(stage.get("command_joint_position", stage["target_joint_position"]))
    u = np.linspace(0.0, 1.0, 65)
    path = start + (target - start) * (3 * u * u - 2 * u * u * u)

    # Produce separate grasp-orientation views before adding per-sample
    # diagnostic markers.  Do not tile these views: a combined card makes a
    # parallel-jaw closing direction unnecessarily easy to misread.
    p.resetJointState(ob, driver, start, physicsClientId=client)
    initial_surface_wp, initial_wq = ph._target_pose(
        ob, ol[stage["contact_link"]], stage["contact_pose_link"], 0.0, client
    )
    initial_wp, initial_wq = ph._target_pose(
        ob,
        ol[stage["contact_link"]],
        stage["contact_pose_link"],
        float(stage.get("grasp_depth_m", 0.0)),
        client,
        stage.get("robot_tool_contact_offset_eef_m"),
    )
    if args.orientation_only_no_ik:
        robot_colors = _set_body_visibility(rb, set(), client)
        single_tool_preview = _single_tool_push_preview(stage)
        if single_tool_preview:
            tool_name = stage["allowed_robot_contact_links"][0]
            if tool_name not in rl:
                raise ValueError(f"Unknown nominated robot tool link {tool_name!r}")
            _create_single_robot_tool_proxy(
                rb,
                rl[ex["robot"]["end_effector_link"]],
                rl[tool_name],
                initial_wp,
                initial_wq,
                stage["robot_tool_contact_offset_eef_m"],
                client,
            )
            proxy_label = "cyan = nominated robot collision tool"
            isolated_label = "ISOLATED ONE-TOOL PUSH | NO IK"
            proxy_geometry = "nominated_robot_collision_link_visual_proxy"
            colour_legend = {
                "nominated_robot_tool_link": "cyan",
                "tool_contact_point": "cyan marker",
                "target_link": "orange",
            }
        else:
            _create_parallel_jaw_proxy(initial_wp, initial_wq, opening, client)
            proxy_label = "cyan/magenta = jaw proxy"
            isolated_label = "ISOLATED JAW PROXY | NO IK"
            proxy_geometry = "parallel_jaw_visual_proxy"
            colour_legend = {
                "negative_closing_axis_jaw": "cyan",
                "positive_closing_axis_jaw": "magenta",
                "target_link": "orange",
            }
        outward = np.asarray(
            p.rotateVector(initial_wq, [0.0, 0.0, -1.0]), dtype=np.float64
        )
        tangent = np.cross(
            outward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        )
        if float(np.linalg.norm(tangent)) < 1e-6:
            tangent = np.asarray(
                p.rotateVector(initial_wq, [1.0, 0.0, 0.0]), dtype=np.float64
            )
        orientation_images = {
            "full_object_robot_oblique": view(
                list(initial_wp),
                1.05,
                52,
                -18,
                f"VISUAL ORIENTATION ONLY | NO IK | {proxy_label}",
                client,
            ),
            "full_object_robot_top": view(
                list(initial_wp),
                0.95,
                35,
                -74,
                "VISUAL ORIENTATION ONLY TOP | NO IK",
                client,
            ),
        }
        object_colors = _set_body_visibility(
            ob, {ol[stage["contact_link"]]}, client
        )
        orientation_images["isolated_target_gripper_surface_normal"] = look_at_view(
            initial_surface_wp,
            outward,
            0.42,
            f"{isolated_label} | surface-normal view",
            client,
        )
        orientation_images["isolated_target_gripper_tangent"] = look_at_view(
            initial_surface_wp,
            tangent,
            0.42,
            f"{isolated_label} | tangent view",
            client,
        )
        orientation_files = {}
        for panel_name, orientation_image in orientation_images.items():
            filename = f"orientation__{panel_name}.png"
            orientation_image.save(args.out / filename)
            orientation_files[panel_name] = filename
        orientation_images["full_object_robot_oblique"].save(args.out / "scene.png")
        report["orientation_preview"] = {
            "rendering_policy": "one_candidate_one_view_per_file_no_composite",
            "images": orientation_files,
            "ik_was_run": False,
            "proxy_geometry": proxy_geometry,
            "jaw_colour_legend": colour_legend,
        }
        (args.out / "scene.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        _restore_body_visibility(ob, object_colors, client)
        _restore_body_visibility(rb, robot_colors, client)
        p.disconnect(client)
        print(json.dumps(report, indent=2))
        return 0
    initial_answer = solver.solve(initial_wp, initial_wq, home, enforce_step=False)
    set_robot_arm(rb, arm, np.asarray(initial_answer["q"], dtype=np.float64), client)
    p.performCollisionDetection(physicsClientId=client)
    orientation_tag = (
        f"initial contact | IK={'OK' if initial_answer['success'] else 'FAIL'} "
        f"err={initial_answer['position_error_m'] * 1000.0:.2f}mm"
    )
    orientation_images = {
        "full_object_robot_oblique": view(
            list(initial_wp), 1.05, 52, -18,
            "FULL OBJECT + ROBOT | " + orientation_tag, client
        ),
        "full_object_robot_top": view(
            list(initial_wp), 0.95, 35, -74,
            "FULL OBJECT + ROBOT TOP | " + orientation_tag, client
        ),
    }
    visible_robot_links = {
        rl[name]
        for name in stage["allowed_robot_contact_links"]
        if name in rl
    }
    visible_robot_links.add(rl[ex["robot"]["end_effector_link"]])
    frontier = list(visible_robot_links)
    for _ in range(2):
        next_frontier = []
        for link in frontier:
            if link < 0:
                continue
            parent = int(p.getJointInfo(rb, link, physicsClientId=client)[16])
            if parent not in visible_robot_links:
                visible_robot_links.add(parent)
                next_frontier.append(parent)
        frontier = next_frontier
    object_colors = _set_body_visibility(
        ob, {ol[stage["contact_link"]]}, client
    )
    robot_colors = _set_body_visibility(rb, visible_robot_links, client)
    outward = np.asarray(p.rotateVector(initial_wq, [0.0, 0.0, -1.0]), dtype=np.float64)
    tangent = np.cross(outward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    if float(np.linalg.norm(tangent)) < 1e-6:
        tangent = np.asarray(p.rotateVector(initial_wq, [1.0, 0.0, 0.0]), dtype=np.float64)
    # Colour the two opposing contact links differently so their separation
    # (the jaw-closing axis) is visible without inferring it from wrist shape.
    allowed_visible = [
        rl[name] for name in stage["allowed_robot_contact_links"] if name in rl
    ]
    if allowed_visible:
        p.changeVisualShape(
            rb, allowed_visible[0], rgbaColor=[0.1, 0.85, 1.0, 1.0],
            textureUniqueId=-1,
            physicsClientId=client,
        )
    if len(allowed_visible) > 1:
        p.changeVisualShape(
            rb, allowed_visible[1], rgbaColor=[1.0, 0.2, 0.75, 1.0],
            textureUniqueId=-1,
            physicsClientId=client,
        )
    orientation_images["isolated_target_gripper_surface_normal"] = look_at_view(
        initial_surface_wp,
        outward,
        0.42,
        "ISOLATED | cyan/magenta = opposing jaws | surface-normal view",
        client,
    )
    orientation_images["isolated_target_gripper_tangent"] = look_at_view(
        initial_surface_wp,
        tangent,
        0.42,
        "ISOLATED | cyan/magenta = opposing jaws | tangent view",
        client,
    )
    _restore_body_visibility(ob, object_colors, client)
    _restore_body_visibility(rb, robot_colors, client)
    orientation_files = {}
    for panel_name, orientation_image in orientation_images.items():
        filename = f"orientation__{panel_name}.png"
        orientation_image.save(args.out / filename)
        orientation_files[panel_name] = filename
    report["orientation_preview"] = {
        "rendering_policy": "one_candidate_one_view_per_file_no_composite",
        "images": orientation_files,
        "initial_ik_success": bool(initial_answer["success"]),
        "initial_ik_position_error_m": float(initial_answer["position_error_m"]),
        "initial_ik_orientation_error_deg": math.degrees(
            float(initial_answer["orientation_error_rad"])
        ),
        "jaw_colour_legend": {
            "first_allowed_contact_link": "cyan",
            "second_allowed_contact_link": "magenta",
            "target_link": "orange",
        },
    }

    images = []
    ref = home.copy()
    for idx in (0, 16, 32, 48, 64):
        p.resetJointState(ob, driver, float(path[idx]), physicsClientId=client)
        surface_wp, wq = ph._target_pose(
            ob, ol[stage["contact_link"]], stage["contact_pose_link"], 0.0, client
        )
        grasp_depth = float(stage.get("grasp_depth_m", 0.0))
        wp, wq = ph._target_pose(
            ob,
            ol[stage["contact_link"]],
            stage["contact_pose_link"],
            grasp_depth,
            client,
            stage.get("robot_tool_contact_offset_eef_m"),
        )
        ans = solver.solve(wp, wq, ref, enforce_step=False)
        if ans["success"]:
            ref = np.asarray(ans["q"], dtype=np.float64)
        set_robot_arm(rb, arm, np.asarray(ans["q"], dtype=np.float64), client)
        p.performCollisionDetection(physicsClientId=client)
        touching = {}
        for pt in p.getClosestPoints(rb, ob, 0.0, physicsClientId=client):
            key = (inv_r.get(pt[3], pt[3]), inv_o.get(pt[4], pt[4]))
            touching[key] = min(touching.get(key, 0.0), float(pt[8]))
        if robot_support is not None:
            for pt in p.getClosestPoints(robot_support, ob, 0.0, physicsClientId=client):
                key = ("robot_support", inv_o.get(pt[4], pt[4]))
                touching[key] = min(touching.get(key, 0.0), float(pt[8]))
        target_gaps = {}
        for robot_link_name in stage["allowed_robot_contact_links"]:
            near = p.getClosestPoints(
                rb,
                ob,
                0.10,
                linkIndexA=rl[robot_link_name],
                linkIndexB=ol[stage["contact_link"]],
                physicsClientId=client,
            )
            target_gaps[robot_link_name] = (
                min(float(point[8]) for point in near) if near else 0.10
            )
        target_contacting_robot_links = {
            inv_r.get(int(point[3]), str(point[3]))
            for point in p.getContactPoints(
                rb, ob, physicsClientId=client
            )
            if int(point[4]) == ol[stage["contact_link"]]
            and int(point[3]) in {
                rl[name] for name in stage["allowed_robot_contact_links"]
            }
        }
        required_physical_contact_links = (
            set(stage["allowed_robot_contact_links"])
            if stage["interaction"] == "explicit_ideal_feasibility"
            else set()
        )
        bilateral_physical_contact = bool(required_physical_contact_links) and (
            required_physical_contact_links <= target_contacting_robot_links
        )
        required_near_links = min(
            2 if stage["interaction"] == "explicit_ideal_feasibility" else 1,
            len(target_gaps),
        )
        near_link_count = sum(
            gap <= float(args.maximum_target_gap_m) for gap in target_gaps.values()
        )
        forbidden_clearances = {}
        required_clearance = float(stage.get("minimum_swept_clearance_m", 0.0))
        for object_link_name in stage.get("forbidden_contact_links", []):
            for point in p.getClosestPoints(
                rb,
                ob,
                max(required_clearance, 0.10),
                linkIndexB=ol[object_link_name],
                physicsClientId=client,
            ):
                key = f"{inv_r.get(int(point[3]), point[3])}|{object_link_name}"
                forbidden_clearances[key] = min(
                    forbidden_clearances.get(key, float("inf")), float(point[8])
                )
            if robot_support is not None:
                for point in p.getClosestPoints(
                    robot_support,
                    ob,
                    max(required_clearance, 0.10),
                    linkIndexB=ol[object_link_name],
                    physicsClientId=client,
                ):
                    key = f"robot_support|{object_link_name}"
                    forbidden_clearances[key] = min(
                        forbidden_clearances.get(key, float("inf")), float(point[8])
                    )
        minimum_forbidden_clearance = min(
            forbidden_clearances.values(), default=None
        )
        surface_outward = list(p.rotateVector(wq, [0, 0, -1]))
        eef_forward = list(p.rotateVector(wq, [0, 0, 1]))
        base_from_contact = np.asarray(ex["robot"]["base_translation_m"], dtype=np.float64) - np.asarray(surface_wp, dtype=np.float64)
        base_distance = float(np.linalg.norm(base_from_contact))
        outward_halfspace = float(np.dot(np.asarray(surface_outward), base_from_contact))
        outward_alignment = outward_halfspace / base_distance if base_distance > 1e-9 else 0.0
        base_from_contact_xy = base_from_contact[:2]
        surface_outward_xy = np.asarray(surface_outward, dtype=np.float64)[:2]
        horizontal_denominator = float(
            np.linalg.norm(base_from_contact_xy) * np.linalg.norm(surface_outward_xy)
        )
        horizontal_outward_alignment = (
            float(np.dot(base_from_contact_xy, surface_outward_xy)) / horizontal_denominator
            if horizontal_denominator > 1e-9
            else 0.0
        )
        robot_forward_xy = np.asarray(
            p.rotateVector(ex["robot"]["base_rotation_xyzw"], [1.0, 0.0, 0.0]),
            dtype=np.float64,
        )[:2]
        robot_facing_denominator = float(
            np.linalg.norm(robot_forward_xy) * np.linalg.norm(base_from_contact_xy)
        )
        robot_facing_contact = (
            float(np.dot(robot_forward_xy, -base_from_contact_xy)) / robot_facing_denominator
            if robot_facing_denominator > 1e-9
            else 0.0
        )
        eef_contact_alignment = float(np.dot(np.asarray(eef_forward), -np.asarray(surface_outward)))
        report["samples"].append({
            "sample": idx, "driver_q": float(path[idx]),
            "contact_point_world_m": [round(float(v), 4) for v in surface_wp],
            "grasp_target_world_m": [round(float(v), 4) for v in wp],
            "surface_outward_axis_world": [round(float(v), 3) for v in surface_outward],
            "panda_grasptarget_forward_world": [round(float(v), 3) for v in eef_forward],
            "robot_approach_direction_world": [round(float(v), 3) for v in eef_forward],
            "robot_base_outward_halfspace_m": round(outward_halfspace, 4),
            "robot_base_outward_alignment_cosine": round(outward_alignment, 4),
            "robot_base_horizontal_outward_alignment_cosine": round(
                horizontal_outward_alignment, 4
            ),
            "robot_base_facing_contact_cosine": round(robot_facing_contact, 4),
            "panda_eef_to_contact_inward_alignment_cosine": round(eef_contact_alignment, 4),
            "ik_success": bool(ans["success"]),
            "ik_position_error_m": round(float(ans["position_error_m"]), 6),
            "ik_orientation_error_deg": round(math.degrees(float(ans["orientation_error_rad"])), 3),
            "target_link_gap_by_allowed_robot_link_m": {
                name: round(gap, 5) for name, gap in sorted(target_gaps.items())
            },
            "required_near_contact_links": required_near_links,
            "near_contact_link_count": near_link_count,
            "target_contact_geometry_ready": bool(
                ans["success"]
                and required_near_links > 0
                and near_link_count >= required_near_links
            ),
            "target_contacting_robot_links": sorted(target_contacting_robot_links),
            "required_physical_contact_links": sorted(required_physical_contact_links),
            "bilateral_physical_contact": bool(bilateral_physical_contact),
            "required_forbidden_clearance_m": required_clearance,
            "minimum_forbidden_clearance_m": (
                None
                if minimum_forbidden_clearance is None
                else round(minimum_forbidden_clearance, 5)
            ),
            "forbidden_clearance_passed": bool(
                minimum_forbidden_clearance is None
                or minimum_forbidden_clearance >= required_clearance
            ),
            "tightest_forbidden_pairs": {
                name: round(distance, 5)
                for name, distance in sorted(
                    forbidden_clearances.items(), key=lambda item: item[1]
                )[:6]
            },
            "penetrating_pairs": {f"{a}|{b}": round(d, 4) for (a, b), d in sorted(touching.items(), key=lambda kv: kv[1])[:6]},
        })
        # Markers must be rebuilt per frame because the contact point moves.
        marker(surface_wp, 0.012, [0.1, 0.95, 0.25, 1.0], client)
        marker(wp, 0.014, [0.05, 0.85, 0.9, 1.0] if ans["success"] else [0.95, 0.1, 0.1, 1.0], client)
        arrow(wp, eef_forward, 0.10, [0.75, 0.25, 0.95, 1.0], client)
        tag = f"sample {idx}  q={path[idx]:.2f}  IK={'OK' if ans['success'] else 'FAIL'} err={ans['position_error_m']*1000:.2f}mm"
        images.append(view(list(wp), 1.15, 52, -16, tag, client))
        images.append(view(list(wp), 0.55, 128, -12, tag + "  [close]", client))

    # Whole-scene overview, so object/robot placement is visible at a glance.
    boxes = [p.getAABB(ob, -1, physicsClientId=client)] + [p.getAABB(ob, j, physicsClientId=client) for j in range(p.getNumJoints(ob, physicsClientId=client))]
    boxes += [p.getAABB(rb, -1, physicsClientId=client)] + [p.getAABB(rb, j, physicsClientId=client) for j in range(p.getNumJoints(rb, physicsClientId=client))]
    if robot_support is not None:
        boxes.append(p.getAABB(robot_support, -1, physicsClientId=client))
    lo = [min(b[0][a] for b in boxes) for a in range(3)]
    hi = [max(b[1][a] for b in boxes) for a in range(3)]
    centre = [(lo[a] + hi[a]) / 2 for a in range(3)]
    span = max(hi[a] - lo[a] for a in range(3))
    overview = [view(centre, span * 1.6, yaw, -18, f"OVERVIEW yaw={yaw}", client) for yaw in (35, 125, 215)]
    overview.append(view(centre, span * 1.6, 35, -80, "OVERVIEW top", client))

    if args.video is not None:
        # A placement preview is deliberately separate from the physical video.
        # It freezes the object at the selected stage's initial state and shows
        # the first contact-pose IK from all azimuths, so front/back placement is
        # visually auditable even when a later manipulation waypoint is infeasible.
        p.resetJointState(ob, driver, start, physicsClientId=client)
        initial_wp, initial_wq = ph._target_pose(
            ob,
            ol[stage["contact_link"]],
            stage["contact_pose_link"],
            float(stage.get("grasp_depth_m", 0.0)),
            client,
            stage.get("robot_tool_contact_offset_eef_m"),
        )
        initial_answer = solver.solve(initial_wp, initial_wq, home, enforce_step=False)
        set_robot_arm(rb, arm, np.asarray(initial_answer["q"], dtype=np.float64), client)
        p.performCollisionDetection(physicsClientId=client)
        preview_frames = []
        for yaw in np.linspace(20.0, 380.0, 120, endpoint=False):
            frame = view(
                centre,
                span * 1.55,
                float(yaw),
                -18,
                "PLACEMENT QA PREVIEW - NOT A PHYSICAL ROLLOUT",
                client,
                width=960,
                height=720,
            )
            preview_frames.append(np.asarray(frame, dtype=np.uint8))
        video_path = args.video.expanduser().resolve()
        video_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(
            video_path,
            preview_frames,
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )
        report["placement_preview_video"] = str(video_path)
        report["placement_preview_is_physical_rollout"] = False

    cols, rows_n = 2, (len(images) + len(overview) + 1) // 2
    sheet = Image.new("RGB", (760 * cols, 580 * rows_n), (18, 22, 30))
    for i, img in enumerate(overview + images):
        sheet.paste(img, ((i % cols) * 760, (i // cols) * 580))
    sheet.save(args.out / "scene.png")
    (args.out / "scene.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    p.disconnect(client)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
