#!/usr/bin/env python3
"""Render the object and isolated robot-contact links before choosing a pose.

Run this after assigning motion ownership and before selecting a contact point.
The overview establishes the grounded scene and free side.  A separate four-view
reference image isolates every nominated robot-contact link, because a small
handle, button, rim, or lip can be visually lost inside the full asset even when
it belongs to the same URDF link as a much larger panel.  Contact selection must
use that semantic feature, not default to the link AABB centre.

Colour code:
  red     the driver/control link the robot must operate
  orange  plan-motion links whose physical executor is still unassigned
  grey    static structure

Alongside the images it reports the measured facts contact and placement need:
the isolated-reference path, target extent and centre, link axes, and free-space
map.  No asset name appears anywhere; every label comes from plan ownership and
URDF geometry.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

APP_ROOT = Path(__file__).resolve().parent
REPO = APP_ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT))

import artimo_plan  # noqa: E402
import run_artimo_physics as ph  # noqa: E402

DRIVER_COLOR = [0.92, 0.12, 0.12, 1.0]
EFFECT_COLOR = [1.0, 0.62, 0.05, 1.0]
STATIC_COLOR = [0.72, 0.74, 0.78, 1.0]
ISOLATED_COLOR = [0.72, 0.80, 0.90, 1.0]


def _view(target, distance, yaw, pitch, label, client, width=760, height=580, subtitle=None):
    view = p.computeViewMatrixFromYawPitchRoll(target, distance, yaw, pitch, 0.0, 2, physicsClientId=client)
    projection = p.computeProjectionMatrixFOV(50.0, width / height, 0.02, 14.0)
    _, _, rgba, _, _ = p.getCameraImage(
        width, height, viewMatrix=view, projectionMatrix=projection,
        renderer=p.ER_TINY_RENDERER, physicsClientId=client,
    )
    image = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)[:, :, :3])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 44), fill=(16, 20, 27))
    font = ImageFont.load_default()
    draw.text((8, 6), label, fill=(240, 245, 250), font=font)
    draw.text(
        (8, 24),
        subtitle or "RED = link the robot must drive   ORANGE = effect link, do not touch",
        fill=(255, 190, 120),
        font=font,
    )
    return image


def _marker(position, radius, color, client):
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color, physicsClientId=client)
    return p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=list(position),
                             physicsClientId=client)


def _safe_component(value: str) -> str:
    """Return a deterministic filename component for a declared joint/link name."""
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _axis_markers(origin, rotation, length, client):
    """Create small RGB cylinders showing the selected link's local XYZ axes."""
    bodies = []
    radius = max(0.0015, float(length) * 0.018)
    local_axes = (
        ([1.0, 0.0, 0.0], [0.92, 0.12, 0.12, 1.0]),
        ([0.0, 1.0, 0.0], [0.12, 0.85, 0.18, 1.0]),
        ([0.0, 0.0, 1.0], [0.12, 0.35, 0.95, 1.0]),
    )
    for local_axis, colour in local_axes:
        direction = np.asarray(p.rotateVector(rotation, local_axis), dtype=np.float64)
        midpoint = np.asarray(origin, dtype=np.float64) + direction * float(length) * 0.5
        cylinder_rotation = p.getQuaternionFromEuler(
            [0.0, np.pi / 2.0, 0.0]
            if local_axis[0]
            else ([-np.pi / 2.0, 0.0, 0.0] if local_axis[1] else [0.0, 0.0, 0.0])
        )
        shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=float(length),
            rgbaColor=colour,
            physicsClientId=client,
        )
        bodies.append(
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=shape,
                basePosition=midpoint.tolist(),
                baseOrientation=cylinder_rotation,
                physicsClientId=client,
            )
        )
    return bodies


def _salient_extreme_features(body, link_index, client):
    """Find small projections that protrude from either side of a thin link.

    This is deliberately geometric rather than semantic.  A handle or button
    often occupies a small footprint at one extreme of a panel's thinnest axis.
    The returned crops are only visual references; the agent still decides what
    the feature means from the task and image.
    """
    _, vertices = p.getMeshData(
        body, link_index, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=client
    )
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 8 or points.shape[1] != 3:
        return []
    low = points.min(axis=0)
    high = points.max(axis=0)
    extent = high - low
    thin_axis = int(np.argmin(extent))
    other_axes = [axis for axis in range(3) if axis != thin_axis]
    full_footprint = float(np.prod(np.maximum(extent[other_axes], 1e-6)))
    band_width = max(0.002, float(extent[thin_axis]) * 0.16)
    features = []
    for side, extreme in (("negative", low[thin_axis]), ("positive", high[thin_axis])):
        mask = (
            points[:, thin_axis] <= extreme + band_width
            if side == "negative"
            else points[:, thin_axis] >= extreme - band_width
        )
        band = points[mask]
        # A collision hull can represent a thin protrusion with only four extreme
        # vertices, so do not require a dense tessellation for a visual crop.
        if band.shape[0] < 4:
            continue
        band_low = band.min(axis=0)
        band_high = band.max(axis=0)
        band_extent = band_high - band_low
        footprint_ratio = float(
            np.prod(np.maximum(band_extent[other_axes], 1e-6)) / full_footprint
        )
        # A broad panel face is not a feature crop.  Preserve only compact
        # extreme geometry that would disappear in a whole-link rendering.
        if footprint_ratio >= 0.45:
            continue
        features.append({
            "thin_axis": "xyz"[thin_axis],
            "side": side,
            "footprint_ratio": round(footprint_ratio, 6),
            "centre_link_m": ((band_low + band_high) * 0.5).tolist(),
            "extent_link_m": band_extent.tolist(),
        })
    return features


def _free_space_map(body, client, height, links, span=0.6, steps=13, probe_radius=0.05):
    """ASCII occupancy at one height, so the open side is obvious in text too."""
    shape = p.createCollisionShape(p.GEOM_SPHERE, radius=probe_radius, physicsClientId=client)
    probe = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape,
                              basePosition=[0.0, 0.0, -50.0], physicsClientId=client)
    xs = np.linspace(-span, span, steps)
    ys = np.linspace(-span, span, steps)
    rows = []
    for x in xs:
        row = ""
        for y in ys:
            p.resetBasePositionAndOrientation(probe, [float(x), float(y), height], [0, 0, 0, 1],
                                              physicsClientId=client)
            blocked = False
            for index in links:
                hits = p.getClosestPoints(probe, body, 0.0, linkIndexB=index, physicsClientId=client)
                if hits and min(float(h[8]) for h in hits) < 0.0:
                    blocked = True
                    break
            row += "#" if blocked else "."
        rows.append(row)
    p.removeBody(probe, physicsClientId=client)
    return {
        "height_m": round(float(height), 4),
        "probe_radius_m": probe_radius,
        "x_axis_m": [round(float(v), 3) for v in xs],
        "y_axis_m": [round(float(v), 3) for v in ys],
        "rows_are_x_columns_are_y": rows,
        "legend": "# = a robot-sized probe collides here, . = free space",
    }


def survey(task_spec: Path, out: Path, driver_links: dict[str, str], samples: int) -> dict[str, Any]:
    task = ph._read_json(task_spec)
    inputs = task["inputs"]
    simulation_urdf = ph._resolve(inputs.get("physics_urdf") or inputs["urdf"])
    plan = artimo_plan.read_plan(ph._resolve(inputs["plan"]))
    initial = ph.task_initial_joint_values(task)
    requested = artimo_plan.requested_extrema(plan)

    out.mkdir(parents=True, exist_ok=True)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet survey client")
    try:
        # Ground the object exactly as the harness will, so the reported heights are
        # the ones a placement will actually face.
        probe_execution = {
            "schema_version": 2,
            "scene": {"object_base_translation_m": [0.0, 0.0, 0.0],
                      "object_base_rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            "robot": {"base_translation_m": [0.0, 0.0, 0.0],
                      "base_rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                      "arm_joint_names": [], "finger_joint_names": [],
                      "end_effector_link": "", "home_joint_positions": []},
            "control_execution": [], "stages": [], "causal_rules": [], "seeds": {"ik": 0},
        }
        grounding = ph._ground_execution_scene(simulation_urdf, simulation_urdf, probe_execution, initial)
        grounded = grounding.pop("execution")
        base_z = grounded["scene"]["object_base_translation_m"]

        body = p.loadURDF(str(simulation_urdf), basePosition=base_z,
                          baseOrientation=[0.0, 0.0, 0.0, 1.0], useFixedBase=True,
                          flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, physicsClientId=client)
        joints, links = ph._maps(body, client)
        for name, value in initial.items():
            if name in joints:
                p.resetJointState(body, joints[name], float(value), physicsClientId=client)
        p.performCollisionDetection(physicsClientId=client)

        # The caller nominates robot-owned controls only after completing the
        # per-control ownership table.  Any other moving joint is deliberately
        # labelled unassigned here: geometry alone cannot prove that it is an
        # internally actuated effect rather than additional robot work.
        driver_joints = set(driver_links)
        unassigned_joints = set(requested) - driver_joints
        joint_child = {}
        for index in range(p.getNumJoints(body, physicsClientId=client)):
            info = p.getJointInfo(body, index, physicsClientId=client)
            joint_child[info[1].decode("utf-8")] = (index, info[12].decode("utf-8"))

        classified = {"driver": [], "unassigned_plan_motion": [], "static": []}
        for name, (index, child) in joint_child.items():
            if name in driver_joints:
                classified["driver"].append((name, index, child))
            elif name in unassigned_joints:
                classified["unassigned_plan_motion"].append((name, index, child))
        moving = {
            index
            for group in ("driver", "unassigned_plan_motion")
            for _, index, _ in classified[group]
        }
        for index in range(-1, p.getNumJoints(body, physicsClientId=client)):
            if index not in moving:
                classified["static"].append(("", index, ""))

        def apply_overview_colours():
            for visual_index in range(-1, p.getNumJoints(body, physicsClientId=client)):
                p.changeVisualShape(body, visual_index, rgbaColor=STATIC_COLOR, physicsClientId=client)
            for _, visual_index, _ in classified["unassigned_plan_motion"]:
                p.changeVisualShape(body, visual_index, rgbaColor=EFFECT_COLOR, physicsClientId=client)
            for _, visual_index, _ in classified["driver"]:
                p.changeVisualShape(body, visual_index, rgbaColor=DRIVER_COLOR, physicsClientId=client)

        apply_overview_colours()

        report: dict[str, Any] = {
            "object_base_translation_m": [round(float(v), 4) for v in base_z],
            "plan_requested_extrema": requested,
            "unassigned_plan_motion_joints": sorted(unassigned_joints),
            "ownership_warning": (
                "Unassigned moving joints are not automatically internal effects; "
                "classify every plan control in control_execution before rollout."
            ),
            "targets": [],
        }
        target_centres = []
        for joint, contact_link in driver_links.items():
            if joint not in joint_child:
                raise KeyError(f"URDF has no joint named {joint!r}")
            index = links[contact_link]
            low, high = p.getAABB(body, index, physicsClientId=client)
            extent = [round(float(high[a] - low[a]), 4) for a in range(3)]
            centre = [round(float((high[a] + low[a]) * 0.5), 4) for a in range(3)]
            origin, rotation = ph.link_world_pose(body, index, client)
            axes = {
                f"local_{name}_in_world": [round(float(v), 3) for v in p.rotateVector(rotation, vector)]
                for name, vector in (("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]))
            }
            # The thinnest world axis of a lever/handle is usually the door or lid
            # face normal; the longest is the direction it protrudes.
            protrusion_axis = int(np.argmax(extent))
            thin_axis = int(np.argmin(extent))
            target_centres.append(centre)
            reference_name = (
                "contact_link_reference__"
                f"{_safe_component(joint)}__{_safe_component(contact_link)}.png"
            )
            report["targets"].append({
                "driver_joint": joint,
                "contact_link": contact_link,
                "colour": "red",
                "isolated_reference_image": reference_name,
                "world_aabb_low_m": [round(float(v), 4) for v in low],
                "world_aabb_high_m": [round(float(v), 4) for v in high],
                "world_centre_m": centre,
                "world_extent_m": extent,
                "longest_world_axis": "xyz"[protrusion_axis],
                "thinnest_world_axis": "xyz"[thin_axis],
                "link_frame_origin_world_m": [round(float(v), 4) for v in origin],
                "link_frame_axes": axes,
                "working_height_m": centre[2],
            })

        # Produce one dedicated image per robot-contact link.  Other object links
        # are fully hidden, rather than merely recoloured: this makes a small
        # handle/button/lip visible even when its link is surrounded by a large
        # appliance body in the overview.  The RGB triad is oriented in the
        # contacted link frame and anchored at the link's collision-AABB centre.
        all_links = list(range(-1, p.getNumJoints(body, physicsClientId=client)))
        for target in report["targets"]:
            target_index = links[target["contact_link"]]
            for visual_index in all_links:
                p.changeVisualShape(
                    body, visual_index, rgbaColor=[0.75, 0.78, 0.82, 0.0],
                    physicsClientId=client,
                )
            p.changeVisualShape(
                body, target_index, rgbaColor=ISOLATED_COLOR, physicsClientId=client
            )
            link_origin, link_rotation = ph.link_world_pose(body, target_index, client)
            target_span = max(float(value) for value in target["world_extent_m"])
            axis_bodies = _axis_markers(
                target["world_centre_m"], link_rotation,
                max(0.035, min(0.10, target_span * 0.22)), client,
            )
            distance = max(0.20, target_span * 1.35)
            reference_views = [
                _view(
                    target["world_centre_m"], distance, yaw, pitch,
                    f"ISOLATED CONTACT LINK {target['contact_link']}  yaw={yaw}deg",
                    client,
                    subtitle=(
                        f"driver={target['driver_joint']} | only selected link visible | "
                        "RGB axes = link XYZ"
                    ),
                )
                for yaw, pitch in ((0, -15), (90, -15), (180, -15), (270, -15))
            ]
            reference_sheet = Image.new("RGB", (1520, 1160), (18, 22, 30))
            for image_index, image in enumerate(reference_views):
                reference_sheet.paste(
                    image, ((image_index % 2) * 760, (image_index // 2) * 580)
                )
            reference_sheet.save(out / target["isolated_reference_image"])
            for marker_body in axis_bodies:
                p.removeBody(marker_body, physicsClientId=client)

            target["salient_feature_references"] = []
            for feature_index, feature in enumerate(
                _salient_extreme_features(body, target_index, client)
            ):
                feature_world, _ = p.multiplyTransforms(
                    link_origin,
                    link_rotation,
                    feature["centre_link_m"],
                    [0.0, 0.0, 0.0, 1.0],
                    physicsClientId=client,
                )
                feature_span = max(float(value) for value in feature["extent_link_m"])
                # This is intentionally a tight crop.  A long, thin handle can
                # span most of a door's height; fitting its full length would
                # again reduce its graspable cross-section to a few pixels.
                # Cropping the ends is preferable to hiding the interface.
                feature_distance = max(0.12, min(0.24, feature_span * 0.65))
                feature_name = (
                    "contact_feature_reference__"
                    f"{_safe_component(target['driver_joint'])}__"
                    f"{_safe_component(target['contact_link'])}__"
                    f"{feature['thin_axis']}_{feature['side']}_{feature_index}.png"
                )
                marker_bodies = [
                    _marker(feature_world, 0.008, [0.1, 0.95, 0.3, 1.0], client)
                ]
                feature_low = (
                    np.asarray(feature["centre_link_m"], dtype=np.float64)
                    - np.asarray(feature["extent_link_m"], dtype=np.float64) * 0.5
                )
                feature_high = (
                    np.asarray(feature["centre_link_m"], dtype=np.float64)
                    + np.asarray(feature["extent_link_m"], dtype=np.float64) * 0.5
                )
                for corner_index in range(8):
                    corner_local = [
                        float(feature_high[axis] if corner_index & (1 << axis) else feature_low[axis])
                        for axis in range(3)
                    ]
                    corner_world, _ = p.multiplyTransforms(
                        link_origin,
                        link_rotation,
                        corner_local,
                        [0.0, 0.0, 0.0, 1.0],
                        physicsClientId=client,
                    )
                    marker_bodies.append(
                        _marker(corner_world, 0.0045, [0.95, 0.08, 0.75, 1.0], client)
                    )
                crop_views = [
                    _view(
                        feature_world,
                        feature_distance,
                        yaw,
                        pitch,
                        f"SALIENT PROTRUSION ON {target['contact_link']}  yaw={yaw}deg",
                        client,
                        subtitle=(
                            f"thin-axis={feature['thin_axis']} {feature['side']} | "
                            f"footprint={feature['footprint_ratio']:.3f} | green=center, magenta=bounds"
                        ),
                    )
                    for yaw, pitch in ((0, -8), (180, -8), (45, -22), (225, -22))
                ]
                crop_sheet = Image.new("RGB", (1520, 1160), (18, 22, 30))
                for image_index, image in enumerate(crop_views):
                    crop_sheet.paste(
                        image, ((image_index % 2) * 760, (image_index // 2) * 580)
                    )
                crop_sheet.save(out / feature_name)
                for marker_body in marker_bodies:
                    p.removeBody(marker_body, physicsClientId=client)
                target["salient_feature_references"].append({
                    **feature,
                    "centre_world_m": [round(float(value), 6) for value in feature_world],
                    "image": feature_name,
                })

        apply_overview_colours()
        for target in report["targets"]:
            _marker(target["world_centre_m"], 0.02, [0.1, 0.95, 0.3, 1.0], client)

        # Free space at each target's working height decides which side to stand on.
        report["free_space"] = [
            _free_space_map(body, client, target["working_height_m"], all_links)
            for target in report["targets"]
        ]

        boxes = [p.getAABB(body, index, physicsClientId=client) for index in all_links]
        low = [min(b[0][a] for b in boxes) for a in range(3)]
        high = [max(b[1][a] for b in boxes) for a in range(3)]
        report["object_world_aabb_low_m"] = [round(float(v), 4) for v in low]
        report["object_world_aabb_high_m"] = [round(float(v), 4) for v in high]
        report["contact_selection_rule"] = (
            "Open each target's isolated_reference_image and every listed "
            "salient_feature_references image before proposing a contact. "
            "Select a task-semantic handle, button, rim, lip, or other control feature "
            "visible on that link; do not default to the link AABB centre or a broad "
            "panel unless direct panel pushing is the declared interaction."
        )
        centre = [(low[a] + high[a]) * 0.5 for a in range(3)]
        span = max(high[a] - low[a] for a in range(3))

        images = [
            _view(centre, span * 1.5, yaw, -20, f"OBJECT ONLY  yaw={yaw}deg", client)
            for yaw in (0, 90, 180, 270)
        ]
        images.append(_view(centre, span * 1.5, 45, -80, "OBJECT ONLY  top-down", client))
        for target in report["targets"]:
            images.append(_view(
                target["world_centre_m"], max(0.35, span * 0.45), 45, -15,
                f"TARGET {target['contact_link']} close-up (red)", client,
            ))
            images.append(_view(
                target["world_centre_m"], max(0.35, span * 0.45), 225, -15,
                f"TARGET {target['contact_link']} opposite side", client,
            ))
        columns = 2
        rows_n = (len(images) + columns - 1) // columns
        sheet = Image.new("RGB", (760 * columns, 580 * rows_n), (18, 22, 30))
        for i, image in enumerate(images):
            sheet.paste(image, ((i % columns) * 760, (i // columns) * 580))
        sheet.save(out / "object_survey.png")
        (out / "object_survey.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
        return report
    finally:
        p.disconnect(client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument(
        "--driver", action="append", required=True, metavar="JOINT=LINK",
        help="Control joint the robot drives and the link it contacts, e.g. joint_2=link_2",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--path-samples", type=int, default=5)
    args = parser.parse_args()
    try:
        drivers: dict[str, str] = {}
        for item in args.driver:
            if "=" not in item:
                raise ValueError(f"--driver expects JOINT=LINK, got {item!r}")
            joint, link = item.split("=", 1)
            drivers[joint] = link
        report = survey(
            args.task_spec.expanduser().resolve(), args.out.expanduser().resolve(),
            drivers, int(args.path_samples),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Object survey failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
