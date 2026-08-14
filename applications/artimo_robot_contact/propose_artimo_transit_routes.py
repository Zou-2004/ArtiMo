#!/usr/bin/env python3
"""Derive one bounded Cartesian route batch around a moved plan obstacle.

The tool reads the actual preceding stage endpoint and incoming approach from
the generic planner, applies all prior robot-owned object endpoints, identifies
the forbidden link blocking their direct Cartesian segment, measures that
link's world AABB, and emits four lateral routes plus three top-corner routes.
It never knows an asset or task name and never runs a physical rollout.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

import run_artimo_physics as ph


AXES = ("x", "y", "z")


def _normalized_quaternion(values: list[float]) -> list[float]:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or float(np.linalg.norm(quaternion)) < 1e-12:
        raise ValueError(f"Invalid XYZW quaternion: {values!r}")
    return (quaternion / np.linalg.norm(quaternion)).tolist()


def generate_face_routes(
    start_position: list[float],
    start_rotation: list[float],
    end_position: list[float],
    end_rotation: list[float],
    expanded_low: list[float],
    expanded_high: list[float],
    face_offset_m: float,
    top_pivot_fractions: tuple[float, ...] = (0.55, 0.70, 0.85),
) -> list[dict[str, Any]]:
    """Generate one orientation-aware bridge outside each usable AABB face."""
    if face_offset_m <= 0.0:
        raise ValueError("face_offset_m must be positive")
    start = np.asarray(start_position, dtype=np.float64)
    end = np.asarray(end_position, dtype=np.float64)
    low = np.asarray(expanded_low, dtype=np.float64)
    high = np.asarray(expanded_high, dtype=np.float64)
    if any(array.shape != (3,) for array in (start, end, low, high)):
        raise ValueError("Positions and AABB bounds must be three-vectors")
    if np.any(low >= high):
        raise ValueError("expanded AABB must satisfy low < high")
    if not top_pivot_fractions or any(
        not 0.0 < fraction < 1.0 for fraction in top_pivot_fractions
    ):
        raise ValueError("top pivot fractions must lie strictly between zero and one")
    start_q = _normalized_quaternion(start_rotation)
    end_q = _normalized_quaternion(end_rotation)
    routes = []
    # The simulator has a ground plane.  A -Z route would pass underneath it,
    # so only +Z and both signs of the horizontal world axes are admissible.
    for axis in range(2):
        signs = (-1, 1)
        for sign in signs:
            face = (
                float(low[axis] - face_offset_m)
                if sign < 0
                else float(high[axis] + face_offset_m)
            )
            first = start.copy()
            second = end.copy()
            first[axis] = face
            second[axis] = face
            side = f"{AXES[axis]}-{'low' if sign < 0 else 'high'}"
            routes.append(
                {
                    "id": f"auto-{side}",
                    "side": side,
                    "waypoints_world": [
                        {
                            "translation_m": first.tolist(),
                            "rotation_xyzw": start_q,
                        },
                        {
                            "translation_m": second.tolist(),
                            "rotation_xyzw": start_q,
                        },
                        {
                            "translation_m": second.tolist(),
                            "rotation_xyzw": end_q,
                        },
                    ],
                }
            )
    top_face = float(high[2] + face_offset_m)
    first_top = start.copy()
    first_top[2] = top_face
    horizontal_exits = []
    for axis in (0, 1):
        if end[axis] < low[axis]:
            horizontal_exits.append((0, float(low[axis] - end[axis]), axis, -1))
        elif end[axis] > high[axis]:
            horizontal_exits.append((0, float(end[axis] - high[axis]), axis, 1))
        else:
            horizontal_exits.extend(
                [
                    (1, abs(float(end[axis] - low[axis])), axis, -1),
                    (1, abs(float(end[axis] - high[axis])), axis, 1),
                ]
            )
    _, _, exit_axis, exit_sign = min(horizontal_exits)
    exit_face = (
        float(low[exit_axis] - face_offset_m)
        if exit_sign < 0
        else float(high[exit_axis] + face_offset_m)
    )
    rotation_height = float(high[2] + min(0.002, 0.1 * face_offset_m))
    exit_side = f"{AXES[exit_axis]}-{'low' if exit_sign < 0 else 'high'}"
    for fraction in top_pivot_fractions:
        pivot = (1.0 - fraction) * start + fraction * end
        pivot[exit_axis] = exit_face
        pivot[2] = top_face
        rotation_point = pivot.copy()
        rotation_point[2] = rotation_height
        fraction_token = int(round(100.0 * fraction))
        routes.append(
            {
                "id": f"auto-z-high-via-{exit_side}-p{fraction_token:02d}",
                "side": f"z-high-via-{exit_side}",
                "waypoints_world": [
                    {
                        "translation_m": first_top.tolist(),
                        "rotation_xyzw": start_q,
                    },
                    {
                        "translation_m": pivot.tolist(),
                        "rotation_xyzw": start_q,
                    },
                    {
                        "translation_m": rotation_point.tolist(),
                        "rotation_xyzw": end_q,
                    },
                ],
            }
        )
    return routes


def segment_intersects_aabb(
    start_position: list[float],
    end_position: list[float],
    low: list[float],
    high: list[float],
) -> bool:
    """Return whether a finite Cartesian segment crosses a closed AABB."""
    start = np.asarray(start_position, dtype=np.float64)
    end = np.asarray(end_position, dtype=np.float64)
    lower = np.asarray(low, dtype=np.float64)
    upper = np.asarray(high, dtype=np.float64)
    if any(array.shape != (3,) for array in (start, end, lower, upper)):
        raise ValueError("Segment and AABB values must be three-vectors")
    if np.any(lower >= upper):
        raise ValueError("AABB must satisfy low < high")
    direction = end - start
    t_low, t_high = 0.0, 1.0
    for axis in range(3):
        if abs(float(direction[axis])) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        left = float((lower[axis] - start[axis]) / direction[axis])
        right = float((upper[axis] - start[axis]) / direction[axis])
        if left > right:
            left, right = right, left
        t_low = max(t_low, left)
        t_high = min(t_high, right)
        if t_low > t_high:
            return False
    return True


def _outer_contact_pose(
    object_body: int,
    object_links: dict[str, int],
    stage: dict[str, Any],
    client: int,
) -> tuple[list[float], list[float]]:
    """Return the Cartesian endpoint farthest from contact for approach/retreat."""
    waypoints = list(stage.get("approach_waypoints_link_m", []))
    if waypoints:
        pose = {
            "translation_m": list(waypoints[-1]),
            "rotation_xyzw": stage["contact_pose_link"]["rotation_xyzw"],
        }
        return ph._target_pose(
            object_body,
            object_links[stage["contact_link"]],
            pose,
            0.0,
            client,
            stage.get("robot_tool_contact_offset_eef_m"),
        )
    return ph._target_pose(
        object_body,
        object_links[stage["contact_link"]],
        stage["contact_pose_link"],
        ph._effective_grasp_depth(stage)
        + float(stage["precontact_offset_m"]),
        client,
        stage.get("robot_tool_contact_offset_eef_m"),
    )


def propose(
    task: dict[str, Any],
    execution: dict[str, Any],
    execution_path: Path,
    incoming_stage_id: str,
    obstacle_link_override: str | None,
    expansion_margin_m: float,
    face_offset_m: float,
    top_pivot_fractions: tuple[float, ...],
) -> dict[str, Any]:
    if expansion_margin_m <= 0.0:
        raise ValueError("expansion_margin_m must be positive")
    execution = ph.materialize_execution_defaults(task, execution)
    ph._validate_execution_schema(execution)
    plan_json = ph._read_json(ph._resolve(task["inputs"]["plan"]))
    ph._validate_execution_against_plan(plan_json, execution)
    matches = [
        index
        for index, stage in enumerate(execution["stages"])
        if stage["id"] == incoming_stage_id
    ]
    if len(matches) != 1 or matches[0] == 0:
        raise ValueError("incoming_stage_id must match one non-first stage")
    stage_index = matches[0]
    if ph._same_contact_sequence(
        execution["stages"][stage_index - 1], execution["stages"][stage_index]
    ):
        raise ValueError("continuous contact_sequence cannot declare a transit")

    source_urdf = ph._resolve(task["inputs"]["urdf"])
    simulation_urdf = ph.resolve_simulation_urdf(task, execution, source_urdf)
    ph._require_matching_mechanism(source_urdf, simulation_urdf)
    robot_urdf = ph._resolve(task["inputs"]["robot_urdf"])
    initial = ph.task_initial_joint_values(task)
    grounded = ph._ground_execution_scene(
        simulation_urdf, robot_urdf, copy.deepcopy(execution), initial
    )["execution"]
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet route-proposal client")
    try:
        object_body, robot_body, robot_support = ph._load_scene(
            simulation_urdf, robot_urdf, grounded, client, False
        )
        object_joints, object_links = ph._maps(object_body, client)
        prior_state = dict(initial)
        for prior_stage in grounded["stages"][:stage_index]:
            prior_state[str(prior_stage["driver_joint"])] = float(
                prior_stage.get(
                    "command_joint_position", prior_stage["target_joint_position"]
                )
            )
        for joint_name, value in prior_state.items():
            if joint_name in object_joints:
                p.resetJointState(
                    object_body,
                    object_joints[joint_name],
                    float(value),
                    physicsClientId=client,
                )
        previous = grounded["stages"][stage_index - 1]
        incoming = grounded["stages"][stage_index]
        if previous.get("release_retreat_waypoints_world"):
            last_release_pose = previous["release_retreat_waypoints_world"][-1]
            start_pose = (
                list(last_release_pose["translation_m"]),
                _normalized_quaternion(last_release_pose["rotation_xyzw"]),
            )
        else:
            start_pose = _outer_contact_pose(
                object_body, object_links, previous, client
            )
        end_pose = _outer_contact_pose(object_body, object_links, incoming, client)
        forbidden = {
            name: object_links[name]
            for name in incoming.get("forbidden_contact_links", [])
        }
        unknown = set(incoming.get("forbidden_contact_links", [])) - set(
            object_links
        )
        if unknown:
            raise KeyError(f"Unknown forbidden links: {sorted(unknown)}")
        required_clearance = float(incoming.get("minimum_swept_clearance_m", 0.0))
        measured_obstacles = []
        for link_name, link_index in sorted(forbidden.items()):
            raw_low, raw_high = p.getAABB(
                object_body, link_index, physicsClientId=client
            )
            expanded_low = [
                float(value) - expansion_margin_m for value in raw_low
            ]
            expanded_high = [
                float(value) + expansion_margin_m for value in raw_high
            ]
            measured_obstacles.append(
                {
                    "link": link_name,
                    "measured_collision_aabb_world_m": {
                        "low": list(raw_low),
                        "high": list(raw_high),
                    },
                    "expanded_aabb_world_m": {
                        "low": expanded_low,
                        "high": expanded_high,
                    },
                    "direct_eef_segment_intersects_expanded_aabb": segment_intersects_aabb(
                        list(start_pose[0]),
                        list(end_pose[0]),
                        expanded_low,
                        expanded_high,
                    ),
                }
            )
        blocking_obstacles = [
            item
            for item in measured_obstacles
            if item["direct_eef_segment_intersects_expanded_aabb"]
        ]
        route_required = bool(blocking_obstacles)
        if obstacle_link_override is not None:
            if obstacle_link_override not in forbidden:
                raise ValueError(
                    "--obstacle-link must be declared in incoming "
                    "forbidden_contact_links"
                )
            obstacle_link = obstacle_link_override
            route_required = True
        elif blocking_obstacles:
            # Prefer the smallest blocking volume.  It is the least-cost local
            # detour and avoids routing around a larger parent link when a
            # compact declared obstacle already explains the intersection.
            obstacle_link = min(
                blocking_obstacles,
                key=lambda item: float(
                    np.prod(
                        np.asarray(item["expanded_aabb_world_m"]["high"])
                        - np.asarray(item["expanded_aabb_world_m"]["low"])
                    )
                ),
            )["link"]
        else:
            obstacle_link = None

        report: dict[str, Any] = {
            "schema_version": 1,
            "incoming_stage_id": incoming_stage_id,
            "route_required": route_required,
            "required_clearance_m": required_clearance,
            "direct_geometry_probe": measured_obstacles,
            "transit_endpoints_world_m": {
                "start": list(start_pose[0]),
                "end": list(end_pose[0]),
            },
            "transit_endpoint_rotations_xyzw": {
                "start": list(start_pose[1]),
                "end": list(end_pose[1]),
            },
            "prior_plan_joint_state": prior_state,
            "obstacle": None,
            "routes_config": None,
        }
        if obstacle_link is None:
            return report

        measured = next(
            item for item in measured_obstacles if item["link"] == obstacle_link
        )
        raw_low = measured["measured_collision_aabb_world_m"]["low"]
        raw_high = measured["measured_collision_aabb_world_m"]["high"]
        expanded_low = measured["expanded_aabb_world_m"]["low"]
        expanded_high = measured["expanded_aabb_world_m"]["high"]
        candidates = generate_face_routes(
            list(start_pose[0]),
            list(start_pose[1]),
            list(end_pose[0]),
            list(end_pose[1]),
            expanded_low,
            expanded_high,
            face_offset_m,
            top_pivot_fractions,
        )
        obstacle = {
            "link": obstacle_link,
            "state": prior_state,
            "measured_collision_aabb_world_m": {
                "low": list(raw_low),
                "high": list(raw_high),
            },
            "expansion_margin_m": expansion_margin_m,
            "expanded_aabb_world_m": {
                "low": expanded_low,
                "high": expanded_high,
            },
        }
        report["obstacle"] = obstacle
        report["routes_config"] = {
            "schema_version": 1,
            "execution_template_path": str(execution_path),
            "incoming_stage_id": incoming_stage_id,
            "transit_endpoints_world_m": report["transit_endpoints_world_m"],
            "transit_endpoint_rotations_xyzw": report[
                "transit_endpoint_rotations_xyzw"
            ],
            "obstacle": obstacle,
            "generation": {
                "policy": "four_lateral_faces_plus_three_top_corner_pivots",
                "face_offset_m": face_offset_m,
                "top_pivot_fractions": list(top_pivot_fractions),
                "candidate_count": len(candidates),
            },
            "candidates": candidates,
        }
        return report
    finally:
        p.disconnect(client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--incoming-stage", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--obstacle-link")
    parser.add_argument("--expansion-margin-m", type=float, default=0.06)
    parser.add_argument("--face-offset-m", type=float, default=0.02)
    parser.add_argument(
        "--top-pivot-fractions",
        type=float,
        nargs="+",
        default=[0.55, 0.70, 0.85],
    )
    args = parser.parse_args()

    task_path = args.task_spec.expanduser().resolve()
    execution_path = args.execution.expanduser().resolve()
    answer = propose(
        ph._read_json(task_path),
        ph._read_json(execution_path),
        execution_path,
        str(args.incoming_stage),
        args.obstacle_link,
        float(args.expansion_margin_m),
        float(args.face_offset_m),
        tuple(float(value) for value in args.top_pivot_fractions),
    )
    output = args.out.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "proposal.json").write_text(
        json.dumps(answer, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    routes = answer.get("routes_config")
    if routes is not None:
        (output / "routes.json").write_text(
            json.dumps(routes, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "route_required": answer["route_required"],
                "obstacle": answer["obstacle"],
                "proposal": str(output / "proposal.json"),
                "routes": str(output / "routes.json") if routes is not None else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
