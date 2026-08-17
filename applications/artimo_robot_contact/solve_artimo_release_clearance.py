#!/usr/bin/env python3
"""Search an asset-agnostic robot retreat outside a pre-reacquisition sweep.

The input is the same task/execution pair used by the physics harness.  The
tool starts at the final executable grasp pose, opens the fingers, searches
nearby world-frame end-effector poses, and scores both the retreat path and the
stationary robot against every sampled state of dependent plan motion before
the next robot-contact acquisition, including internal mechanisms and passive
returns. Later manipulation belongs to that next block, and motion between
contacts belongs to the transit planner. It writes diagnostics only;
the agent copies the chosen pose into
``release_retreat_waypoints_world`` in execution data.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p

import run_artimo_physics as ph
from artimo_ik import BulletIK, link_world_pose, set_fingers, set_robot_arm


def _minimum_body_distance(
    robot: int,
    obj: int,
    robot_support: int | None,
    client: int,
) -> float:
    distances = [
        float(point[8])
        for point in ph.object_closest_points(robot, obj, 0.25, client)
    ]
    if robot_support is not None:
        distances.extend(
            float(point[8])
            for point in ph.object_closest_points(
                robot_support, obj, 0.25, client
            )
        )
    return min(distances, default=0.25)


def _candidate_sort_key(
    item: dict[str, Any], minimum_required: float
) -> tuple[bool, float, float, float]:
    """Rank the full retreat and the later plan sweep before route length."""
    return (
        float(item["minimum_clearance_m"]) <= minimum_required,
        -float(item["minimum_clearance_m"]),
        -float(item["post_release_plan_sweep_minimum_clearance_m"]),
        float(np.linalg.norm(item["offset_world_m"])),
    )


def solve(
    task_path: Path | dict[str, Any],
    execution_path: Path | dict[str, Any],
    *,
    planned_stages: list[ph.StagePlan] | None = None,
) -> dict[str, Any]:
    """Solve release clearance from files or an in-memory dense candidate.

    Placement already owns a fully planned dense trajectory.  Passing that
    trajectory here avoids planning the same manipulation a second time merely
    to recover its final arm command.  The path-based CLI remains unchanged for
    standalone diagnostics and older callers.
    """
    task = (
        task_path
        if isinstance(task_path, dict)
        else ph._read_json(task_path)
    )
    execution_input = (
        execution_path
        if isinstance(execution_path, dict)
        else ph._read_json(execution_path)
    )
    execution = ph.materialize_execution_defaults(task, execution_input)
    inputs = task["inputs"]
    source_urdf = ph._resolve(inputs["urdf"])
    robot_urdf = ph._resolve(inputs["robot_urdf"])
    plan = ph._read_json(ph._resolve(inputs["plan"]))
    initial = ph.task_initial_joint_values(task)
    simulation_urdf = ph.resolve_simulation_urdf(task, execution, source_urdf)
    grounded = ph._ground_execution_scene(
        simulation_urdf, robot_urdf, execution, initial
    )["execution"]
    plans = planned_stages
    if plans is None:
        plans = ph._plan_stages(
            simulation_urdf,
            robot_urdf,
            grounded,
            initial,
            allow_partial_debug=True,
            # Placement fixes the base first; this solver is responsible for
            # replacing any stale world-frame release waypoint from an older base.
            validate_release_clearance=False,
            object_plan=plan,
        )
    release_indices = [
        index
        for index, item in enumerate(plans)
        if item.stage.get("release_before_phase") is not None
    ]
    if len(release_indices) != 1:
        raise ValueError(
            "Release-clearance solver currently requires exactly one declared "
            f"release boundary; found {len(release_indices)}"
        )
    release_index = release_indices[0]
    release_plan = plans[release_index]
    next_contact_phase = (
        str(plans[release_index + 1].stage["source_phase"])
        if release_index + 1 < len(plans)
        else None
    )
    # Release clearance is a non-penetration proof, not a hidden comfort
    # margin. Any strictly positive whole-route separation is acceptable.
    minimum_required = 0.0

    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet release-clearance client")
    try:
        object_body, robot_body, robot_support_body = ph._load_scene(
            simulation_urdf, robot_urdf, grounded, client, False
        )
        object_joints, object_links = ph._maps(object_body, client)
        robot_joints, robot_links = ph._maps(robot_body, client)
        robot_spec = grounded["robot"]
        arm = [robot_joints[name] for name in robot_spec["arm_joint_names"]]
        fingers = [robot_joints[name] for name in robot_spec["finger_joint_names"]]
        eef = robot_links[robot_spec["end_effector_link"]]

        # The release scene is the complete authoritative object state at this
        # timeline boundary, not merely the endpoints of preceding robot stages.
        # Internal effects (for example an opened lid) can materially change the
        # retreat collision geometry and must be present during route scoring.
        final_joint_state = ph._object_joint_state_before_phase(
            plan,
            initial,
            str(release_plan.stage["release_before_phase"]),
        )
        for name, value in final_joint_state.items():
            if name in object_joints:
                p.resetJointState(
                    object_body,
                    object_joints[name],
                    float(value),
                    targetVelocity=0.0,
                    physicsClientId=client,
                )

        # Release routing starts at the exact final grasp command. A moved door
        # or panel can make the generic link-normal withdrawal unsafe, so a
        # solver-authored world route replaces that default retreat entirely.
        start_q = release_plan.manipulation[-1].copy()
        set_robot_arm(robot_body, arm, start_q, client)
        opening = float(
            release_plan.stage["contact_acquisition"]["approach_finger_opening_m"]
        )
        set_fingers(robot_body, fingers, opening, client)
        start_position, start_rotation = link_world_pose(robot_body, eef, client)
        ik_settings = grounded.get("ik", {})
        solver = BulletIK(
            robot_body,
            arm,
            eef,
            fingers,
            opening,
            {
                "random_seed": int(grounded["seeds"].get("ik", 0)) + 991,
                "random_restarts": int(ik_settings.get("random_restarts", 96)),
                "max_iterations": int(ik_settings.get("max_iterations", 2400)),
                "position_tolerance_m": float(
                    ik_settings.get("position_tolerance_m", 0.002)
                ),
                "orientation_tolerance_deg": float(
                    ik_settings.get("orientation_tolerance_deg", 1.0)
                ),
                "max_joint_step_rad": float(
                    ik_settings.get("max_joint_step_rad", 0.8)
                ),
            },
            client,
        )

        contact_link_index = object_links[release_plan.stage["contact_link"]]
        directions = []
        for raw in itertools.product((-1.0, 0.0, 1.0), repeat=3):
            vector = np.asarray(raw, dtype=np.float64)
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                directions.append(vector / norm)
        grasp_depth = ph._effective_grasp_depth(release_plan.stage)
        outward_position, _ = ph._target_pose(
            object_body,
            contact_link_index,
            release_plan.stage["contact_pose_link"],
            grasp_depth + 0.06,
            client,
            release_plan.stage.get("robot_tool_contact_offset_eef_m"),
        )
        preferred_direction = np.asarray(outward_position) - np.asarray(
            start_position
        )
        preferred_norm = float(np.linalg.norm(preferred_direction))
        if preferred_norm > 1e-9:
            preferred_direction /= preferred_norm
            directions.sort(
                key=lambda direction: -float(
                    np.dot(direction, preferred_direction)
                )
            )
        candidates: list[dict[str, Any]] = []
        chosen: dict[str, Any] | None = None
        strict_checked = 0
        for distance in (0.06, 0.10, 0.14, 0.18, 0.22):
            for direction in directions:
                target = np.asarray(start_position) + distance * direction
                sample_count = max(2, int(np.ceil(distance / 0.01)) + 1)
                reference = start_q.copy()
                release_path: list[np.ndarray] = []
                path_ik_failed = False
                for alpha in np.linspace(0.0, 1.0, sample_count)[1:]:
                    position = (
                        (1.0 - alpha) * np.asarray(start_position)
                        + alpha * target
                    )
                    rotation = p.getQuaternionSlerp(
                        start_rotation, start_rotation, float(alpha)
                    )
                    answer = solver.solve_continuous(
                        position.tolist(), list(rotation), reference
                    )
                    if not answer["success"]:
                        path_ik_failed = True
                        break
                    reference = np.asarray(answer["q"], dtype=np.float64)
                    release_path.append(reference.copy())
                if path_ik_failed or not release_path:
                    continue
                candidate_q = release_path[-1]

                # The start command belongs to the already validated final
                # manipulation sample. Validate every newly commanded dense
                # release sample. During the first quarter of withdrawal,
                # proximity to the just-released link is expected; every other
                # object link and the robot support must already be clear.
                path_minimum = 0.25
                path_collision = False
                for sample, q in enumerate(release_path):
                    set_robot_arm(robot_body, arm, q, client)
                    points = ph.object_closest_points(
                        robot_body, object_body, 0.25, client
                    )
                    near_distances = [
                        float(point[8])
                        for point in points
                        if int(point[4]) != contact_link_index
                        or sample >= len(release_path) // 4
                    ]
                    if robot_support_body is not None:
                        near_distances.extend(
                            float(point[8])
                            for point in ph.object_closest_points(
                                robot_support_body,
                                object_body,
                                0.25,
                                client,
                            )
                        )
                    if near_distances:
                        path_minimum = min(path_minimum, min(near_distances))
                    if path_minimum <= minimum_required:
                        path_collision = True
                        break
                if path_collision:
                    continue

                # Hold the robot at the candidate only through object motion
                # before the next robot-contact acquisition.  The incoming
                # manipulation and cross-contact transit have their own
                # collision checks and must not be folded into release scoring.
                set_robot_arm(robot_body, arm, candidate_q, client)
                post_release_minimum = 0.25
                for transition in ph._object_joint_transitions_from_phase(
                    plan,
                    final_joint_state,
                    str(release_plan.stage["release_before_phase"]),
                    next_contact_phase,
                ):
                    joint_name = str(transition["joint"])
                    if joint_name not in object_joints:
                        continue
                    for value in np.linspace(
                        float(transition["start"]),
                        float(transition["target"]),
                        41,
                    ):
                        p.resetJointState(
                            object_body,
                            object_joints[joint_name],
                            float(value),
                            targetVelocity=0.0,
                            physicsClientId=client,
                        )
                        post_release_minimum = min(
                            post_release_minimum,
                            _minimum_body_distance(
                                robot_body,
                                object_body,
                                robot_support_body,
                                client,
                            ),
                        )
                for name, value in final_joint_state.items():
                    if name in object_joints:
                        p.resetJointState(
                            object_body,
                            object_joints[name],
                            float(value),
                            targetVelocity=0.0,
                            physicsClientId=client,
                        )
                clearance = min(path_minimum, post_release_minimum)
                preflight_passed = bool(clearance > minimum_required)
                candidate = {
                    "translation_m": [float(value) for value in target],
                    "rotation_xyzw": [float(value) for value in start_rotation],
                    "offset_world_m": [float(value) for value in distance * direction],
                    "path_minimum_clearance_m": float(path_minimum),
                    "post_release_plan_sweep_minimum_clearance_m": float(
                        post_release_minimum
                    ),
                    "minimum_clearance_m": float(clearance),
                    "preflight_passed": preflight_passed,
                    "dense_release_samples": len(release_path),
                    "passed": False,
                }
                candidates.append(candidate)
                if not preflight_passed:
                    candidate["strict_validation_rejected"] = (
                        "release_or_passive_clearance_below_requirement"
                    )
                    continue
                # This is the exact dense Cartesian path consumed by rollout,
                # plus the complete post-release mechanism sweep. One strict
                # pass is sufficient; do not re-plan prior manipulation stages
                # or keep searching for a larger margin.
                strict_checked += 1
                candidate["strict_validation_passed"] = True
                candidate["passed"] = True
                chosen = candidate
                break
            if chosen is not None:
                break
        return {
            "schema_version": 1,
            "stage_id": release_plan.stage["id"],
            "release_before_phase": release_plan.stage["release_before_phase"],
            "post_release_sweep_stop_before_phase": next_contact_phase,
            "start_pose_world": {
                "translation_m": start_position,
                "rotation_xyzw": start_rotation,
            },
            "minimum_required_clearance_m": minimum_required,
            "object_joint_state_at_release_boundary": final_joint_state,
            "chosen": chosen,
            "candidates_evaluated": len(candidates),
            "strict_candidates_checked": strict_checked,
            "candidate_order": "distance_then_direction_first_strict_pass",
            "top_candidates": candidates[:20],
        }
    finally:
        p.disconnect(client)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = solve(args.task_spec, args.execution)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if (result.get("chosen") or {}).get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
