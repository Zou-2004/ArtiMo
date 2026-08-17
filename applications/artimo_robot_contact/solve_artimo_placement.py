#!/usr/bin/env python3
"""Solve where to put the object and the robot so the whole plan is reachable.

This runs after candidate contact surfaces have been declared. It builds a
whole-task reference from every robot-contact stage, then searches a stable base
with deterministic center-out GPU batches before contact-pose or controller
tuning can be blamed for reach failures. An arm reaching around a moving link or
standing inside its swept volume produces IK residuals and body-vs-object overlap
that are placement failures.

The search is asset-agnostic. It merges uninterrupted contacts into manipulation
blocks, projects the authoritative future object state into a kinematic shadow
world, and tests base poses plus visual-valid contact choices in declared
center-out order. A candidate is accepted only
when every sample of every block admits a continuous collision-free solution.
Nothing here knows an asset name; bounds and seeds are inputs and every rejected
candidate is reported.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import itertools
import json
import math
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pybullet as p

APP_ROOT = Path(__file__).resolve().parent
REPO = APP_ROOT.parents[1]
sys.path.insert(0, str(APP_ROOT))

import run_artimo_physics as ph  # noqa: E402
from artimo_curobo import create_curobo_backend  # noqa: E402
from artimo_ik import (  # noqa: E402
    BulletIK,
    link_world_pose,
    quat_angle_rad,
    set_fingers,
    set_robot_arm,
)


# The visual agent supplies only the center of a bounded depth search. Dense
# validation expands symmetrically from that seed; it never treats the seed as
# acceptance evidence. The effective Panda depth is
# PANDA_CENTERED_GRASP_BASELINE_M (-0.015 m) plus the selected adjustment.
RULE_BASED_GRASP_DEPTH_OFFSETS_M = (
    0.000,
    0.0025,
    -0.0025,
    0.005,
    -0.005,
    0.0075,
    -0.0075,
    0.010,
    -0.010,
    0.0125,
    -0.0125,
    0.015,
    -0.015,
)
RULE_BASED_GRASP_DEPTH_ADJUSTMENT_BOUNDS_M = (-0.015, 0.015)
SPARSE_GPU_BASE_BATCH_SIZE = 16


def _target_gap_policy(
    interaction: str,
    validation_tier: str,
    near_link_count: int,
    required_near_links: int,
) -> tuple[bool, bool, str]:
    """Apply exact target-gap evidence only where planning owns that check.

    A physical push is validated by measured target contact and dwell during
    rollout, so placement must not turn an unqueried target gap into a failure.
    A bilateral grasp defers the same exact query only until dense validation.
    """
    if interaction == "physical_push":
        return True, True, "deferred_to_rollout_contact_gate"
    if validation_tier != "dense":
        return True, True, "deferred_to_dense_pybullet_exact_contact_query"
    return (
        near_link_count >= required_near_links,
        False,
        "pybullet_exact_contact_query",
    )


class _SparseBatchIKProxy:
    """Coalesce concurrent sparse candidate paths into one cuRobo request."""

    def __init__(self, backend: Any, maximum_batch_size: int) -> None:
        self.backend = backend
        self.maximum_batch_size = int(maximum_batch_size)
        self.allow_bullet_fallback = backend.allow_bullet_fallback
        self.environment_collision = backend.environment_collision
        self.self_collision = backend.self_collision
        self._condition = threading.Condition()
        self._queue: list[dict[str, Any]] = []
        self._running = True
        self._dispatcher = threading.Thread(
            target=self._dispatch, name="artimo-sparse-gpu-batcher", daemon=True
        )
        self._dispatcher.start()

    def solve_path(
        self,
        positions_world: Any,
        quaternions_xyzw_world: Any,
        robot_base_position_world: Any,
        robot_base_quaternion_xyzw_world: Any,
        reference: Any,
        maximum_joint_step_rad: float | None,
        enforce_start_step: bool,
        obstacle_worlds_by_sample: Any = None,
        sequential: bool = False,
    ) -> dict[str, Any]:
        pending = {
            "request": {
                "positions_world": positions_world,
                "quaternions_xyzw_world": quaternions_xyzw_world,
                "robot_base_position_world": robot_base_position_world,
                "robot_base_quaternion_xyzw_world": robot_base_quaternion_xyzw_world,
                "reference": list(map(float, reference)),
                "maximum_joint_step_rad": maximum_joint_step_rad,
                "enforce_start_step": bool(enforce_start_step),
                "sequential": bool(sequential),
                "obstacle_worlds_by_sample": (
                    []
                    if obstacle_worlds_by_sample is None
                    else obstacle_worlds_by_sample
                ),
            },
            "event": threading.Event(),
            "response": None,
            "error": None,
        }
        with self._condition:
            if not self._running:
                raise RuntimeError("Sparse GPU batcher is closed")
            self._queue.append(pending)
            self._condition.notify_all()
        pending["event"].wait()
        if pending["error"] is not None:
            raise pending["error"]
        return pending["response"]

    def _dispatch(self) -> None:
        while True:
            with self._condition:
                while self._running and not self._queue:
                    self._condition.wait()
                if not self._running and not self._queue:
                    return
                deadline = time.perf_counter() + 0.10
                while self._running and len(self._queue) < self.maximum_batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        break
                    self._condition.wait(timeout=remaining)
                batch = self._queue[: self.maximum_batch_size]
                del self._queue[: self.maximum_batch_size]
            try:
                responses = self.backend.solve_paths_batch(
                    [item["request"] for item in batch]
                )
                for item, response in zip(batch, responses):
                    item["response"] = response
            except Exception as exc:
                for item in batch:
                    item["error"] = exc
            finally:
                for item in batch:
                    item["event"].set()

    def close(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._dispatcher.join()


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
    for gated_stage in gate.get("stages", []):
        if "inherited_from_stage_id" in gated_stage:
            continue
        if gated_stage.get("agent_decision_scope") != "wrist_angle_only":
            raise ValueError(
                "Orientation gate predates angle-only agent decisions; rerender "
                "the five-roll batch and apply schema-v4 decisions"
            )
        if gated_stage.get("grasp_depth_owner") != "application_rule_based_dense_search":
            raise ValueError("Orientation gate gives grasp depth to the wrong owner")
        if gated_stage.get("selected_angle_status") != "valid":
            raise ValueError("Orientation gate selected angle did not pass visual review")
    groups = gate.get("placement_candidate_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError(
            "Orientation gate lacks placement_candidate_groups; rerun visual "
            "decisions so IK can be deferred to the actual placement base"
        )
    grouped_stage_ids: set[str] = set()
    for group in groups:
        stage_ids = group.get("stage_ids")
        candidates = group.get("candidates")
        if (
            not isinstance(stage_ids, list)
            or not stage_ids
            or not isinstance(candidates, list)
            or not candidates
        ):
            raise ValueError("Every placement candidate group needs stages and candidates")
        overlap = grouped_stage_ids & {str(value) for value in stage_ids}
        if overlap:
            raise ValueError(
                f"Orientation placement candidate groups overlap stages {sorted(overlap)}"
            )
        grouped_stage_ids.update(str(value) for value in stage_ids)
        priorities = [int(candidate.get("visual_priority", -1)) for candidate in candidates]
        if priorities != list(range(1, len(candidates) + 1)):
            raise ValueError(
                "Orientation placement candidates must remain in contiguous visual priority order"
            )
        for candidate in candidates:
            candidate_path = Path(str(candidate.get("execution", ""))).expanduser().resolve()
            if not candidate_path.is_file():
                raise ValueError(f"Missing orientation candidate execution {candidate_path}")
            if candidate.get("execution_sha256") != _sha256(candidate_path):
                raise ValueError(
                    f"Orientation candidate {candidate.get('id')!r} execution changed"
                )
    if grouped_stage_ids != required_stage_ids:
        raise ValueError(
            "Orientation placement candidate groups must cover every robot stage; "
            f"missing={sorted(required_stage_ids - grouped_stage_ids)}"
        )
    return {**gate, "path": str(gate_path), "sha256": _sha256(gate_path)}


def _gated_orientation_options(
    template: dict[str, Any], gate: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build visual-priority orientation combinations without running IK.

    Candidate files are immutable evidence from the no-IK render pass.  Their
    agent-reviewed contact translations and stage rotations are projected
    together onto the placement template; base position remains a placement
    variable.  Priority orders deterministic evaluation and breaks exact
    geometric ties; it never commits a contact choice before all manipulation
    blocks have been scored together.
    """
    stages_by_id = {str(stage["id"]): index for index, stage in enumerate(template["stages"])}
    groups = list(gate["placement_candidate_groups"])
    combinations = list(itertools.product(*(group["candidates"] for group in groups)))
    combinations.sort(
        key=lambda choices: (
            sum(int(choice["visual_priority"]) - 1 for choice in choices),
            tuple(int(choice["visual_priority"]) for choice in choices),
        )
    )
    options: list[dict[str, Any]] = []
    for choices in combinations:
        oriented = json.loads(json.dumps(template))
        ids: list[str] = []
        priorities: list[int] = []
        for group, choice in zip(groups, choices):
            source = ph._read_json(Path(str(choice["execution"])).resolve())
            source_by_id = {str(stage["id"]): stage for stage in source["stages"]}
            for stage_id in group["stage_ids"]:
                target_stage = oriented["stages"][stages_by_id[str(stage_id)]]
                source_stage = source_by_id[str(stage_id)]
                if "contact_roll_deg" in source_stage:
                    target_stage["contact_roll_deg"] = float(
                        source_stage["contact_roll_deg"]
                    )
                if "contact_frame_source" in source_stage:
                    target_stage["contact_frame_source"] = str(
                        source_stage["contact_frame_source"]
                    )
                if "translation_m" in source_stage["contact_pose_link"]:
                    target_stage["contact_pose_link"]["translation_m"] = list(
                        source_stage["contact_pose_link"]["translation_m"]
                    )
                target_stage["contact_pose_link"]["rotation_xyzw"] = list(
                    source_stage["contact_pose_link"]["rotation_xyzw"]
                )
            ids.append(str(choice["id"]))
            priorities.append(int(choice["visual_priority"]))
        options.append(
            {
                "execution": oriented,
                "candidate_ids": ids,
                "visual_priorities": priorities,
                "angles": None,
            }
        )
    return options


def _rule_based_grasp_depth_groups(
    execution: dict[str, Any],
) -> list[tuple[str, list[int]]]:
    """Group stages that must preserve one application-selected grasp depth."""
    grouped: dict[str, list[int]] = {}
    for index, stage in enumerate(execution.get("stages", [])):
        if stage.get("interaction") != "explicit_ideal_feasibility":
            continue
        sequence = stage.get("contact_sequence")
        key = (
            f"sequence:{sequence}"
            if sequence is not None
            else f"stage:{stage['id']}"
        )
        grouped.setdefault(key, []).append(index)
    return list(grouped.items())


def _failed_depth_group_indices(
    execution: dict[str, Any],
    groups: list[tuple[str, list[int]]],
    report: dict[str, Any],
    attempted: set[int] | None = None,
) -> list[int]:
    """Return only untried depth groups containing a currently failed stage.

    A depth coordinate cannot repair a different contact sequence.  In
    particular, once the tray stage already passes, enumerating every tray
    depth while the door remains the sole failure only repeats expensive dense
    IK work and can never improve the failing door block.
    """
    failed_stage_ids = {
        str(stage["stage_id"])
        for stage in report.get("stages", [])
        if not bool(stage.get("feasible"))
    }
    already_attempted = attempted or set()
    answer: list[int] = []
    for group_index, (_, indices) in enumerate(groups):
        if group_index in already_attempted:
            continue
        if any(
            str(execution["stages"][stage_index]["id"]) in failed_stage_ids
            for stage_index in indices
        ):
            answer.append(group_index)
    return answer


def _seed_rule_based_grasp_depths(execution: dict[str, Any]) -> dict[str, Any]:
    """Preserve agent depth only as the center of rule-based validation."""
    answer = copy.deepcopy(execution)
    for _, indices in _rule_based_grasp_depth_groups(answer):
        center = float(answer["stages"][indices[0]].get("grasp_depth_m", 0.0))
        for index in indices:
            answer["stages"][index]["grasp_depth_m"] = center
    return answer


def _ordered_rule_based_grasp_depths(center_m: float) -> list[float]:
    """Expand center-out without leaving the generic Panda depth envelope."""
    lower, upper = RULE_BASED_GRASP_DEPTH_ADJUSTMENT_BOUNDS_M
    values: list[float] = []
    for offset in RULE_BASED_GRASP_DEPTH_OFFSETS_M:
        value = float(center_m) + float(offset)
        if value < lower - 1e-12 or value > upper + 1e-12:
            continue
        value = round(value, 9)
        if value not in values:
            values.append(value)
    return values


def _expand_rule_based_grasp_depths(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compatibility helper exposing the declared dense depth combinations.

    The live placement funnel uses center-out coordinate search and does not
    evaluate this Cartesian expansion.
    """
    expanded: list[dict[str, Any]] = []
    for record in records:
        execution = record["_candidate_execution"]
        groups = _rule_based_grasp_depth_groups(execution)
        if not groups:
            expanded.append(record)
            continue
        centers = [
            float(execution["stages"][indices[0]].get("grasp_depth_m", 0.0))
            for _, indices in groups
        ]
        for values in itertools.product(
            *[_ordered_rule_based_grasp_depths(center) for center in centers]
        ):
            candidate_record = copy.deepcopy(record)
            candidate_execution = candidate_record["_candidate_execution"]
            selected: dict[str, float] = {}
            for (key, indices), depth in zip(groups, values):
                selected[key] = float(depth)
                for index in indices:
                    candidate_execution["stages"][index]["grasp_depth_m"] = float(
                        depth
                    )
            candidate_record["rule_based_grasp_depth_m_by_group"] = selected
            expanded.append(candidate_record)
    return expanded


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


def _smooth_driver_value(start: float, target: float, fraction: float) -> float:
    u = float(np.clip(fraction, 0.0, 1.0))
    smooth = 3.0 * u * u - 2.0 * u * u * u
    return float(start + (target - start) * smooth)


def _driver_path(
    start: float,
    target: float,
    samples: int,
    extra_fractions: Iterable[float] = (),
) -> np.ndarray:
    fractions = {
        round(float(value), 12)
        for value in itertools.chain(
            np.linspace(0.0, 1.0, samples), extra_fractions
        )
        if 0.0 <= float(value) <= 1.0
    }
    return np.asarray(
        [
            _smooth_driver_value(start, target, fraction)
            for fraction in sorted(fractions)
        ],
        dtype=np.float64,
    )


def _driver_fraction(start: float, target: float, value: float) -> float:
    """Invert the monotonic smoothstep used by object-side planning."""
    if abs(float(target) - float(start)) < 1e-12:
        return 0.0
    normalized = float(np.clip((value - start) / (target - start), 0.0, 1.0))
    low, high = 0.0, 1.0
    for _ in range(48):
        middle = 0.5 * (low + high)
        smooth = 3.0 * middle * middle - 2.0 * middle * middle * middle
        if smooth < normalized:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def _adaptive_driver_path(
    start: float,
    target: float,
    pose_at_value: Any,
    *,
    initial_samples: int = 5,
    maximum_samples: int = 65,
    maximum_position_step_m: float = 0.01,
    maximum_rotation_step_deg: float = 4.0,
) -> np.ndarray:
    """Adapt object samples to Cartesian contact-pose curvature.

    Search tiers use fixed sparse fractions.  Only shortlisted candidates call
    this helper.  It begins with five representative states and recursively
    bisects a segment when the contacted pose moves too far in translation or
    rotation.  The final generic trajectory planner still performs its own
    strict dense IK and swept-collision certification afterward.
    """
    initial_samples = max(2, int(initial_samples))
    maximum_samples = max(initial_samples, int(maximum_samples))
    rotation_limit = math.radians(float(maximum_rotation_step_deg))
    fractions = list(np.linspace(0.0, 1.0, initial_samples))
    pose_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def value_at(fraction: float) -> float:
        smooth = 3.0 * fraction * fraction - 2.0 * fraction * fraction * fraction
        return float(start + (target - start) * smooth)

    def pose(fraction: float) -> tuple[np.ndarray, np.ndarray]:
        key = round(float(fraction), 15)
        if key not in pose_cache:
            position, rotation = pose_at_value(value_at(float(fraction)))
            pose_cache[key] = (
                np.asarray(position, dtype=np.float64),
                np.asarray(rotation, dtype=np.float64),
            )
        return pose_cache[key]

    while len(fractions) < maximum_samples:
        additions: list[float] = []
        for left, right in zip(fractions[:-1], fractions[1:]):
            left_position, left_rotation = pose(left)
            right_position, right_rotation = pose(right)
            position_step = float(np.linalg.norm(right_position - left_position))
            dot = float(
                np.clip(
                    abs(np.dot(left_rotation, right_rotation))
                    / max(
                        np.linalg.norm(left_rotation) * np.linalg.norm(right_rotation),
                        1e-12,
                    ),
                    -1.0,
                    1.0,
                )
            )
            rotation_step = 2.0 * math.acos(dot)
            if (
                position_step > float(maximum_position_step_m)
                or rotation_step > rotation_limit
            ):
                additions.append(0.5 * (left + right))
        if not additions:
            break
        available = maximum_samples - len(fractions)
        fractions = sorted(set(fractions + additions[:available]))
    return np.asarray([value_at(fraction) for fraction in fractions], dtype=np.float64)


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


def _center_out_numeric_values(
    values: Iterable[float], *, center: float | None = None
) -> list[float]:
    """Order a declared numeric axis from its center toward both extremes."""
    unique: list[float] = []
    for raw in values:
        value = float(raw)
        if not any(abs(value - existing) < 1e-12 for existing in unique):
            unique.append(value)
    if not unique:
        return []
    target = (
        0.5 * (min(unique) + max(unique))
        if center is None
        else float(center)
    )
    return sorted(
        unique,
        key=lambda value: (
            round(abs(value - target), 12),
            value > target,
            value,
        ),
    )


def _center_first_lateral_offsets(values: Iterable[float]) -> list[float]:
    """Return centerline, then the nearest alternating lateral refinements."""
    return _center_out_numeric_values(values, center=0.0)


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


def _equal_weight_moving_link_centers(
    centers_by_moving_link: dict[str, list[list[float]]],
) -> tuple[list[dict[str, Any]], list[float]]:
    """Collapse sampled centers without letting repeated links dominate."""
    if not centers_by_moving_link:
        raise ValueError("Whole-task placement found no plan-driven moving links")
    moving_links = [
        {
            "moving_link": moving_link,
            "sample_count": len(centers),
            "center_world_m": [
                float(value)
                for value in np.mean(
                    np.asarray(centers, dtype=np.float64), axis=0
                )
            ],
        }
        for moving_link, centers in sorted(centers_by_moving_link.items())
        if centers
    ]
    if not moving_links:
        raise ValueError("Whole-task placement moving links have no samples")
    reference = np.mean(
        np.asarray(
            [item["center_world_m"] for item in moving_links],
            dtype=np.float64,
        ),
        axis=0,
    )
    return moving_links, [float(value) for value in reference]


def _whole_task_moving_link_reference_world(
    simulation_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
    object_plan: dict[str, Any],
    *,
    samples_per_stage: int = 5,
) -> dict[str, Any]:
    """Average every plan-driven moving link over the whole robot task.

    Each authoritative ``driver_joint`` identifies one moving child link.  We
    sample that link over the sparse states in every stage, average samples per
    unique link first, and then average links.  Consequently a repeated stage
    cannot silently bias placement back toward the first contact or toward a
    link that happens to have more samples.
    """
    if int(samples_per_stage) < 2:
        raise ValueError("whole-task reference requires at least two samples per stage")
    grounding = ph._ground_execution_scene(
        simulation_urdf, robot_urdf, json.loads(json.dumps(execution)), initial
    )
    grounded = grounding["execution"]
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect whole-task reference client")
    try:
        body = p.loadURDF(
            str(simulation_urdf),
            basePosition=grounded["scene"]["object_base_translation_m"],
            baseOrientation=ph._quat(
                grounded["scene"]["object_base_rotation_xyzw"]
            ),
            useFixedBase=True,
            physicsClientId=client,
        )
        joints, links = ph._maps(body, client)
        keyframes: list[dict[str, Any]] = []
        centers_by_moving_link: dict[str, list[list[float]]] = {}
        outward_axis_world: list[float] | None = None
        for stage_index, stage in enumerate(grounded["stages"]):
            state = ph._object_joint_state_before_control(
                object_plan,
                initial,
                str(stage["source_phase"]),
                int(stage["source_control_index"]),
            )
            for name, value in state.items():
                if name in joints:
                    p.resetJointState(
                        body, joints[name], float(value), physicsClientId=client
                    )
            driver = joints[str(stage["driver_joint"])]
            contact_link = links[str(stage["contact_link"])]
            joint_info = p.getJointInfo(body, driver, physicsClientId=client)
            moving_link = joint_info[12].decode("utf-8")
            start = float(
                state.get(
                    str(stage["driver_joint"]),
                    p.getJointState(body, driver, physicsClientId=client)[0],
                )
            )
            target = float(
                stage.get("command_joint_position", stage["target_joint_position"])
            )
            fractions = np.linspace(0.0, 1.0, int(samples_per_stage))
            for fraction in fractions:
                value = _smooth_driver_value(start, target, float(fraction))
                p.resetJointState(body, driver, value, physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                aabb_min, aabb_max = p.getAABB(
                    body, driver, physicsClientId=client
                )
                moving_center = 0.5 * (
                    np.asarray(aabb_min, dtype=np.float64)
                    + np.asarray(aabb_max, dtype=np.float64)
                )
                if not np.all(np.isfinite(moving_center)):
                    link_state = p.getLinkState(
                        body,
                        driver,
                        computeForwardKinematics=True,
                        physicsClientId=client,
                    )
                    moving_center = np.asarray(link_state[4], dtype=np.float64)
                center_world_m = [
                    float(component) for component in moving_center
                ]
                centers_by_moving_link.setdefault(moving_link, []).append(
                    center_world_m
                )
                keyframes.append(
                    {
                        "stage_id": str(stage["id"]),
                        "stage_index": int(stage_index),
                        "driver_joint": str(stage["driver_joint"]),
                        "moving_link": moving_link,
                        "driver_fraction": float(fraction),
                        "driver_value": float(value),
                        "position_world_m": center_world_m,
                        "reference_source": "driver_child_link_aabb_center",
                    }
                )
                if outward_axis_world is None:
                    link_position, link_rotation = ph.link_world_pose(
                        body, contact_link, client
                    )
                    _, contact_rotation = p.multiplyTransforms(
                        link_position,
                        link_rotation,
                        stage["contact_pose_link"]["translation_m"],
                        ph._quat(stage["contact_pose_link"]["rotation_xyzw"]),
                        physicsClientId=client,
                    )
                    outward_axis_world = [
                        float(component)
                        for component in p.rotateVector(
                            contact_rotation, [0.0, 0.0, 1.0]
                        )
                    ]
        if not keyframes or outward_axis_world is None:
            raise ValueError("Execution contains no robot-contact keyframes")
        moving_links, reference = _equal_weight_moving_link_centers(
            centers_by_moving_link
        )
        return {
            "reference_world_m": reference,
            "outward_axis_world": outward_axis_world,
            "keyframes": keyframes,
            "moving_links": moving_links,
            "samples_per_stage": int(samples_per_stage),
            "weighting": "equal_unique_driver_child_links_after_fraction_average",
        }
    finally:
        p.disconnect(client)


def _contact_frame_cache_key(
    execution: dict[str, Any], stage: dict[str, Any]
) -> tuple[Any, ...]:
    """Return the placement-invariant inputs that determine a contact frame.

    Contact-facing distance, lateral offset, robot yaw, robot pedestal height,
    and later contact stages do not affect the first contacted object's world
    frame. Keeping them out of this key prevents the sparse base matrix from
    repeatedly loading the same object and robot URDFs in PyBullet merely to
    recover an identical link center and outward axis.
    """
    scene = execution["scene"]
    pose = stage["contact_pose_link"]
    return (
        tuple(float(value) for value in scene["object_base_translation_m"]),
        tuple(float(value) for value in scene["object_base_rotation_xyzw"]),
        float(scene.get("ground_clearance_m", ph.GROUND_CLEARANCE_M)),
        json.dumps(scene.get("support_surface"), sort_keys=True),
        str(stage["contact_link"]),
        tuple(float(value) for value in pose["translation_m"]),
        tuple(float(value) for value in pose["rotation_xyzw"]),
    )


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


def _tier_ik_budget(
    ik_config: dict[str, Any],
    validation_tier: str,
    *,
    random_restarts_override: int | None = None,
    max_iterations_override: int | None = None,
) -> tuple[int, int]:
    """Return a bounded IK budget appropriate for one validation tier.

    Search screening must not silently inherit the expensive final-validation
    budget from execution data.  Overrides are supplied by placement data; the
    defaults deliberately reserve the full execution budget for dense proof.
    """
    if validation_tier not in {"coarse", "sparse", "dense"}:
        raise ValueError(f"Unknown placement validation tier {validation_tier!r}")
    final_restarts = int(ik_config.get("random_restarts", 96))
    final_iterations = int(ik_config.get("max_iterations", 2000))
    default_restarts = {
        "coarse": min(final_restarts, 4),
        "sparse": min(final_restarts, 12),
        "dense": final_restarts,
    }[validation_tier]
    default_iterations = {
        "coarse": min(final_iterations, 500),
        "sparse": min(final_iterations, 1000),
        "dense": final_iterations,
    }[validation_tier]
    restarts = (
        default_restarts
        if random_restarts_override is None
        else int(random_restarts_override)
    )
    iterations = (
        default_iterations
        if max_iterations_override is None
        else int(max_iterations_override)
    )
    if restarts < 0:
        raise ValueError("IK random restart budget must be nonnegative")
    if iterations <= 0:
        raise ValueError("IK iteration budget must be positive")
    return min(restarts, final_restarts), min(iterations, final_iterations)


def _score_candidate(
    simulation_urdf: Path,
    robot_urdf: Path,
    execution: dict[str, Any],
    initial: dict[str, float],
    object_plan: dict[str, Any],
    samples: int,
    allowed_penetration_m: float,
    maximum_grasp_gap_m: float = 0.006,
    *,
    validation_tier: str = "dense",
    run_full_path_confirmation: bool = True,
    maximum_screening_joint_step_rad: float = 0.5,
    ik_random_restarts: int | None = None,
    ik_max_iterations: int | None = None,
    ik_backend: Any | None = None,
    sparse_extra_fractions_by_stage: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    """Return tiered reachability and clearance evidence for one placement.

    ``coarse`` and ``sparse`` are rejection-only search tiers.  ``dense`` uses
    adaptive manipulation samples and may run the strict full-path planner.
    Only that last form can authorize a runnable placement execution.
    """
    if validation_tier not in {"coarse", "sparse", "dense"}:
        raise ValueError(f"Unknown placement validation tier {validation_tier!r}")
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
        tier_random_restarts, tier_max_iterations = _tier_ik_budget(
            ik_config,
            validation_tier,
            random_restarts_override=ik_random_restarts,
            max_iterations_override=ik_max_iterations,
        )

        stage_reports: list[dict[str, Any]] = []
        feasible = True
        pybullet_contact_collision_queries_ran = False
        current = dict(initial)
        reference = home.copy()
        for stage_index, stage in enumerate(grounded["stages"]):
            current = ph._object_joint_state_before_control(
                object_plan,
                initial,
                str(stage["source_phase"]),
                int(stage["source_control_index"]),
            )
            object_state_before = dict(current)
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
                    "random_restarts": tier_random_restarts,
                    "max_iterations": tier_max_iterations,
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
            if validation_tier == "dense":
                def pose_at_value(value: float) -> tuple[list[float], list[float]]:
                    p.resetJointState(
                        object_body, driver, float(value), physicsClientId=client
                    )
                    return ph._target_pose(
                        object_body,
                        contact_link,
                        stage["contact_pose_link"],
                        ph._effective_grasp_depth(stage),
                        client,
                        stage.get("robot_tool_contact_offset_eef_m"),
                    )

                path = _adaptive_driver_path(
                    start,
                    target,
                    pose_at_value,
                    initial_samples=min(65, samples),
                    maximum_samples=samples,
                    maximum_position_step_m=0.004,
                    maximum_rotation_step_deg=1.0,
                )
            else:
                path = _driver_path(
                    start,
                    target,
                    samples,
                    (sparse_extra_fractions_by_stage or {}).get(
                        str(stage["id"]), []
                    ),
                )
            required_samples = int(len(path))
            solved = 0
            worst_error = 0.0
            worst_orientation_error = 0.0
            maximum_adjacent_joint_step = 0.0
            maximum_entry_joint_step = 0.0
            minimum_joint_limit_margin = float("inf")
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
            grasp_depth = ph._effective_grasp_depth(stage)
            target_poses: list[tuple[list[float], list[float]]] = []
            gpu_obstacle_worlds: list[list[dict[str, Any]]] = []
            for value in path:
                p.resetJointState(object_body, driver, float(value), physicsClientId=client)
                target_poses.append(ph._target_pose(
                    object_body,
                    contact_link,
                    stage["contact_pose_link"],
                    grasp_depth,
                    client,
                    stage.get("robot_tool_contact_offset_eef_m"),
                ))
                if (
                    validation_tier in {"sparse", "dense"}
                    and ik_backend is not None
                    and ik_backend.environment_collision
                ):
                    # Sparse and dense collision use the object's actual
                    # collision meshes on the GPU. The
                    # nominated target link is omitted because contact with it
                    # is intentional and its angle/offset geometry was already
                    # frozen by the mandatory visual gate. Every forbidden
                    # object link is represented by its current source shapes.
                    gpu_obstacle_worlds.append(ph._curobo_collision_obstacles(
                        object_body,
                        object_link_names,
                        forbidden.values(),
                        simulation_urdf,
                        client,
                    ))
            gpu_path: list[list[float]] | None = None
            gpu_evidence: dict[str, Any] | None = None
            gpu_fallback_reason: str | None = None
            if ik_backend is not None:
                try:
                    maximum_gpu_step = (
                        None
                        if validation_tier == "coarse"
                        else float(maximum_screening_joint_step_rad)
                        if validation_tier == "dense"
                        else float(maximum_screening_joint_step_rad)
                    )
                    gpu_evidence = ik_backend.solve_path(
                        [pose[0] for pose in target_poses],
                        [pose[1] for pose in target_poses],
                        grounded["robot"]["base_translation_m"],
                        grounded["robot"]["base_rotation_xyzw"],
                        reference,
                        maximum_gpu_step,
                        continues_from_previous,
                        (
                            gpu_obstacle_worlds
                            if validation_tier in {"sparse", "dense"}
                            else None
                        ),
                        sequential=validation_tier == "dense",
                    )
                    if gpu_evidence.get("success"):
                        gpu_path = gpu_evidence["path"]
                    else:
                        gpu_fallback_reason = (
                            "no_continuous_gpu_branch_at_sample_"
                            f"{gpu_evidence.get('failed_sample')}"
                        )
                except Exception as exc:
                    gpu_fallback_reason = f"curobo_worker_error: {exc}"
                # Sparse matrix semantics are GPU-only. A worker/collision
                # failure rejects the cell visibly; it may not silently turn a
                # 336-cell GPU search back into the old CPU bottleneck.
                if gpu_path is None and (
                    validation_tier == "sparse"
                    or not ik_backend.allow_bullet_fallback
                ):
                    feasible = False
            gpu_source_mesh_environment_checked = bool(
                gpu_path is not None
                and validation_tier in {"sparse", "dense"}
                and gpu_evidence is not None
                and gpu_evidence.get("gpu_environment_collision_checked")
            )
            if gpu_source_mesh_environment_checked:
                gpu_minimum = gpu_evidence.get("minimum_environment_clearance_m")
                if gpu_minimum is not None:
                    minimum_forbidden_clearance = float(gpu_minimum)
                    clearance_pairs["gpu_source_mesh_environment"] = float(gpu_minimum)
                    deepest = min(deepest, float(gpu_minimum))
            for sample_index, value in enumerate(path):
                p.resetJointState(object_body, driver, float(value), physicsClientId=client)
                position, rotation = target_poses[sample_index]
                if gpu_path is not None:
                    q = np.asarray(gpu_path[sample_index], dtype=np.float64)
                    set_robot_arm(robot_body, arm, q, client)
                    actual_position, actual_rotation = link_world_pose(robot_body, eef, client)
                    answer = {
                        "success": True,
                        "q": q,
                        "position_error_m": float(np.linalg.norm(
                            np.asarray(actual_position) - np.asarray(position)
                        )),
                        "orientation_error_rad": quat_angle_rad(actual_rotation, rotation),
                        "minimum_joint_limit_margin_rad": float(np.min(np.minimum(
                            q - solver.arm_lower, solver.arm_upper - q
                        ))),
                        "solver": "curobo_batch_ik_pybullet_verified",
                    }
                elif ik_backend is not None and (
                    validation_tier == "sparse"
                    or not ik_backend.allow_bullet_fallback
                ):
                    break
                elif validation_tier == "coarse" or solved == 0:
                    answer = solver.solve(
                        position, rotation, reference, enforce_step=False
                    )
                else:
                    answer = solver.solve_continuous(position, rotation, reference)
                if not answer["success"]:
                    break
                previous_reference = reference.copy()
                reference = np.asarray(answer["q"], dtype=np.float64)
                worst_error = max(worst_error, float(answer["position_error_m"]))
                worst_orientation_error = max(
                    worst_orientation_error,
                    float(answer["orientation_error_rad"]),
                )
                joint_step = float(np.max(np.abs(reference - previous_reference)))
                if sample_index == 0 and not continues_from_previous:
                    # Home/previous-retreat -> approach is a transit planning
                    # problem. Rejecting a placement because the first grasp IK
                    # is more than the manipulation trust radius from home
                    # prevents the dedicated interpolation/RRT planner from ever
                    # seeing an otherwise continuous manipulation path.
                    maximum_entry_joint_step = max(maximum_entry_joint_step, joint_step)
                else:
                    maximum_adjacent_joint_step = max(
                        maximum_adjacent_joint_step, joint_step
                    )
                minimum_joint_limit_margin = min(
                    minimum_joint_limit_margin,
                    float(
                        answer.get(
                            "minimum_joint_limit_margin_rad",
                            np.min(
                                np.minimum(
                                    reference - solver.arm_lower,
                                    solver.arm_upper - reference,
                                )
                            ),
                        )
                    ),
                )
                set_robot_arm(robot_body, arm, reference, client)
                # Full-matrix sparse search has already checked robot self and
                # non-target environment collision in cuRobo on GPU. Nominal
                # target contact was frozen by the visual angle/offset gate.
                # Do not repeat PyBullet contact/collision queries for hundreds
                # of base cells; exact queries are reserved for dense Top-K.
                if validation_tier == "sparse" and gpu_path is not None:
                    solved += 1
                    continue
                pybullet_contact_collision_queries_ran = True
                p.performCollisionDetection(physicsClientId=client)
                # The gripper must actually reach the driven link.  Without this
                # test a candidate that never touches anything scores perfectly on
                # collision and wins, which is how an arm that plainly misses the
                # handle can look like the best placement.
                for link_name, link_index in allowed_by_name.items():
                    near = ph.object_closest_points(
                        robot_body,
                        object_body,
                        grasp_reach_limit,
                        client,
                        link_index_a=link_index,
                        link_index_b=contact_link,
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
                # cuRobo's collision spheres are the fast sparse screen, not
                # the final geometry authority.  Dense candidates must also be
                # checked against PyBullet's exact robot collision bodies: a
                # sphere model can under-cover a link and otherwise report
                # positive source-mesh clearance for a physically penetrating
                # path.  Keep sparse fully GPU batched; pay this exact query
                # only for cost-ordered dense candidates.
                if validation_tier == "dense" or not gpu_source_mesh_environment_checked:
                    for object_name, object_index in forbidden.items():
                        for point in ph.object_closest_points(
                            robot_body,
                            object_body,
                            clearance_query,
                            client,
                            link_index_b=object_index,
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
                            for point in ph.object_closest_points(
                                robot_support_body,
                                object_body,
                                clearance_query,
                                client,
                                link_index_b=object_index,
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
                collision_points = ph.object_closest_points(
                    robot_body, object_body, 0.0, client
                )
                for point in collision_points:
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
                    for point in ph.object_closest_points(
                        robot_support_body,
                        object_body,
                        0.0,
                        client,
                        link_index_b=None,
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
            # An open-then-close grasp needs both application-owned finger links
            # physically near the target throughout the dense path. A physical
            # push instead measures target contact and dwell during rollout.
            # Sparse GPU screening therefore never turns an unqueried gap into
            # a sentinel failure.
            (
                target_gap_passed,
                target_gap_deferred,
                target_contact_geometry_source,
            ) = _target_gap_policy(
                str(stage["interaction"]),
                validation_tier,
                near_link_count,
                required_near_links,
            )
            grasped = bool(
                solved == required_samples
                and required_near_links > 0
                and target_gap_passed
            )
            effective_required_clearance = (
                required_clearance - abs(float(allowed_penetration_m))
            )
            clearance_passed = (
                minimum_forbidden_clearance is None
                or minimum_forbidden_clearance >= effective_required_clearance
            )
            pose_residual_passed = (
                worst_error <= 0.004
                and worst_orientation_error <= math.radians(2.0)
            )
            joint_limit_passed = minimum_joint_limit_margin > 1e-4
            continuity_passed = (
                validation_tier == "coarse"
                or maximum_adjacent_joint_step
                <= float(maximum_screening_joint_step_rad)
            )
            failed_sample: int | None = None
            if gpu_path is None and gpu_evidence is not None:
                raw_failed_sample = gpu_evidence.get("failed_sample")
                if raw_failed_sample is not None:
                    failed_sample = int(raw_failed_sample)
            if failed_sample is None and solved < required_samples:
                failed_sample = int(solved)
            if failed_sample is not None and target_poses:
                failed_sample = max(0, min(failed_sample, len(target_poses) - 1))
                failed_target_position = np.asarray(
                    target_poses[failed_sample][0], dtype=np.float64
                )
                base_position = np.asarray(
                    grounded["robot"]["base_translation_m"], dtype=np.float64
                )
                failed_target_vector = failed_target_position - base_position
                failed_driver_fraction = _driver_fraction(
                    start, target, float(path[failed_sample])
                )
            else:
                failed_target_position = None
                failed_target_vector = None
                failed_driver_fraction = None
            stage_ok = (
                solved == required_samples
                and deepest >= -abs(allowed_penetration_m)
                and grasped
                and clearance_passed
                and pose_residual_passed
                and joint_limit_passed
                and continuity_passed
            )
            feasible = feasible and stage_ok
            stage_reports.append({
                "stage_id": stage["id"],
                "object_state_before": object_state_before,
                "object_state_after": {
                    **object_state_before,
                    str(stage["driver_joint"]): float(target),
                },
                "samples_solved": solved,
                "samples_required": required_samples,
                "validation_tier": validation_tier,
                "ik_random_restarts": tier_random_restarts,
                "ik_max_iterations": tier_max_iterations,
                "ik_backend": (
                    (
                        (
                            "curobo_gpu_ik_source_mesh_collision_screened_"
                            "pybullet_dense_verified"
                            if validation_tier == "dense"
                            else "curobo_gpu_ik_source_mesh_collision_screened"
                        )
                        if validation_tier in {"sparse", "dense"}
                        else "curobo_batch_ik_pybullet_verified"
                    )
                    if gpu_path is not None
                    else (
                        "curobo_gpu_no_valid_solution"
                        if validation_tier == "sparse" and ik_backend is not None
                        else "pybullet"
                    )
                ),
                "ik_backend_fallback_reason": gpu_fallback_reason,
                "curobo_solve_time_s": (
                    None if gpu_evidence is None else gpu_evidence.get("solve_time_s")
                ),
                "curobo_failed_sample": (
                    None if gpu_evidence is None else gpu_evidence.get("failed_sample")
                ),
                "failure_guidance_driver_fraction": failed_driver_fraction,
                "failure_guidance_target_position_world_m": (
                    None
                    if failed_target_position is None
                    else [float(value) for value in failed_target_position]
                ),
                "failure_guidance_vector_from_base_world_m": (
                    None
                    if failed_target_vector is None
                    else [float(value) for value in failed_target_vector]
                ),
                "failure_guidance_target_distance_m": (
                    None
                    if failed_target_vector is None
                    else float(np.linalg.norm(failed_target_vector))
                ),
                "curobo_failure_attribution": (
                    None
                    if gpu_path is not None or gpu_evidence is None
                    else "unattributed_no_valid_solution_not_collision_evidence"
                ),
                "curobo_valid_candidates_per_sample": (
                    []
                    if gpu_evidence is None
                    else gpu_evidence.get("valid_candidates_per_sample", [])
                ),
                "curobo_gpu_collision_obstacles_per_sample": (
                    []
                    if gpu_evidence is None
                    else gpu_evidence.get("gpu_collision_obstacles_per_sample", [])
                ),
                "curobo_minimum_environment_clearance_m": (
                    None
                    if gpu_evidence is None
                    else gpu_evidence.get("minimum_environment_clearance_m")
                ),
                "curobo_environment_clearance_by_sample_m": (
                    []
                    if gpu_evidence is None
                    else gpu_evidence.get("environment_clearance_by_sample_m", [])
                ),
                "maximum_ik_position_error_m": round(worst_error, 6),
                "maximum_ik_orientation_error_deg": round(
                    math.degrees(worst_orientation_error), 6
                ),
                "maximum_adjacent_joint_step_rad": round(
                    maximum_adjacent_joint_step, 6
                ),
                "maximum_entry_joint_step_rad": round(maximum_entry_joint_step, 6),
                "entry_step_deferred_to_transit_planner": bool(
                    not continues_from_previous
                ),
                "minimum_joint_limit_margin_rad": (
                    None
                    if not math.isfinite(minimum_joint_limit_margin)
                    else round(minimum_joint_limit_margin, 6)
                ),
                "pose_residual_passed": bool(pose_residual_passed),
                "joint_limit_passed": bool(joint_limit_passed),
                "continuity_passed": bool(continuity_passed),
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
                "target_contact_geometry_source": target_contact_geometry_source,
                "pybullet_contact_collision_queries_ran": bool(
                    pybullet_contact_collision_queries_ran
                ),
                "gpu_self_collision_checked": bool(
                    gpu_evidence is not None
                    and gpu_evidence.get("gpu_self_collision_checked")
                ),
                "gpu_environment_collision_checked": bool(
                    gpu_evidence is not None
                    and gpu_evidence.get("gpu_environment_collision_checked")
                ),
                "required_forbidden_clearance_m": round(
                    effective_required_clearance, 5
                ),
                "nominal_minimum_swept_clearance_m": round(
                    required_clearance, 5
                ),
                "allowed_body_penetration_m": round(
                    abs(float(allowed_penetration_m)), 5
                ),
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
        screening_passed = bool(feasible)
        transit_route_repairs: list[dict[str, Any]] = []
        dense_plans: list[ph.StagePlan] | None = None
        if feasible and run_full_path_confirmation:
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
                    object_plan=object_plan,
                    ik_path_solver=ik_backend,
                )
            except Exception as exc:
                feasible = False
                if stage_reports:
                    stage_reports[-1]["full_path_rejected"] = str(exc)[:2000]
                    stage_reports[-1]["feasible"] = False
            else:
                by_id = {item.stage["id"]: item for item in dense_plans}
                for stage_index, stage_report in enumerate(stage_reports):
                    dense = by_id[stage_report["stage_id"]]
                    stage_report["dense_full_path_minimum_clearance_m"] = (
                        None
                        if dense.minimum_swept_clearance_m is None
                        else round(float(dense.minimum_swept_clearance_m), 6)
                    )
                    stage_report["dense_full_path_tightest_samples"] = (
                        dense.swept_clearance_violations[:5]
                    )
                    stage_report["dense_full_path_ik_backend"] = dense.ik_backend
                    stage_report["dense_full_path_ik_backend_fallback_reason"] = (
                        dense.ik_backend_fallback_reason
                    )
                    dense_counterexample_samples: set[int] = set()
                    for violation in dense.swept_clearance_violations:
                        sample = violation.get("sample")
                        if (
                            sample is not None
                            and "manipulation" in str(violation.get("phase", ""))
                        ):
                            dense_counterexample_samples.add(int(sample))
                    if dense.minimum_joint_limit_margin_sample is not None:
                        dense_counterexample_samples.add(
                            int(dense.minimum_joint_limit_margin_sample)
                        )
                    if dense.debug_failure is not None:
                        for failure in dense.debug_failure.get("failures", []):
                            sample = failure.get("sample")
                            if sample is not None:
                                dense_counterexample_samples.add(int(sample))
                    dense_counterexample_fractions: list[float] = []
                    stage_start = float(
                        stage_report.get("object_state_before", {}).get(
                            str(dense.stage["driver_joint"]),
                            dense.object_path[0],
                        )
                    )
                    stage_target = float(
                        dense.stage.get(
                            "command_joint_position",
                            dense.stage["target_joint_position"],
                        )
                    )
                    for sample in sorted(dense_counterexample_samples):
                        if 0 <= sample < len(dense.object_path):
                            dense_counterexample_fractions.append(
                                _driver_fraction(
                                    stage_start,
                                    stage_target,
                                    float(dense.object_path[sample]),
                                )
                            )
                    stage_report["dense_counterexample_driver_fractions"] = sorted(
                        {
                            round(float(value), 12)
                            for value in dense_counterexample_fractions
                        }
                    )
                    required_dense_clearance = float(
                        dense.stage.get("minimum_swept_clearance_m", 0.0)
                    )
                    effective_dense_clearance = (
                        required_dense_clearance
                        - abs(float(allowed_penetration_m))
                    )
                    dense_clearance_passed = (
                        dense.minimum_swept_clearance_m is None
                        or dense.minimum_swept_clearance_m
                        >= effective_dense_clearance
                    )
                    stage_report["dense_full_path_required_clearance_m"] = round(
                        effective_dense_clearance, 6
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
                        route_repair = None
                        if dense_ik_passed and not dense_clearance_passed:
                            route_repair = _transit_route_repair(
                                simulation_urdf,
                                grounded["stages"],
                                stage_index,
                                dense,
                            )
                        if route_repair is not None:
                            stage_report["transit_route_repair_required"] = True
                            stage_report["transit_route_repair"] = route_repair
                            transit_route_repairs.append(route_repair)
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
            "screening_passed": screening_passed,
            "validation_tier": validation_tier,
            "full_path_confirmation_ran": bool(run_full_path_confirmation),
            "transit_route_repair_required": bool(transit_route_repairs),
            "transit_route_repairs": transit_route_repairs,
            "stages": stage_reports,
            "manipulation_blocks": _summarize_manipulation_blocks(
                grounded["stages"], stage_reports
            ),
            "grounding": grounding,
            "execution": output_execution,
            # Private in-memory handoff to the release gate.  This is stripped
            # from placement.json and prevents release checking from replanning
            # the just-validated dense manipulation trajectory.
            "_dense_plans": dense_plans,
        }
    finally:
        p.disconnect(client)


def _manipulation_block_stage_ids(
    stages: list[dict[str, Any]],
) -> list[list[str]]:
    """Merge adjacent stages that preserve one uninterrupted contact."""
    blocks: list[list[str]] = []
    previous: dict[str, Any] | None = None
    for stage in stages:
        stage_id = str(stage["id"])
        if (
            blocks
            and previous is not None
            and stage.get("contact_sequence") is not None
            and stage.get("contact_sequence")
            == previous.get("contact_sequence")
        ):
            blocks[-1].append(stage_id)
        else:
            blocks.append([stage_id])
        previous = stage
    return blocks


def _joint_moved_link_sets(urdf: Path) -> dict[str, set[str]]:
    """Map every articulated joint to the complete link subtree it moves."""
    root = ET.parse(urdf).getroot()
    child_by_joint: dict[str, str] = {}
    children_by_link: dict[str, set[str]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        parent = joint.find("parent")
        child = joint.find("child")
        if not name or parent is None or child is None:
            continue
        parent_name = parent.attrib.get("link")
        child_name = child.attrib.get("link")
        if not parent_name or not child_name:
            continue
        child_by_joint[name] = child_name
        children_by_link.setdefault(parent_name, set()).add(child_name)

    def subtree(link: str) -> set[str]:
        answer = {link}
        pending = [link]
        while pending:
            for child in children_by_link.get(pending.pop(), set()):
                if child not in answer:
                    answer.add(child)
                    pending.append(child)
        return answer

    return {joint: subtree(link) for joint, link in child_by_joint.items()}


def _transit_route_repair(
    simulation_urdf: Path,
    stages: list[dict[str, Any]],
    stage_index: int,
    dense_plan: ph.StagePlan,
) -> dict[str, Any] | None:
    """Classify a failed direct transit that must enter bounded route search.

    Only a collision confined to ``transit_in`` and caused by a link moved by
    an earlier plan-owned stage is repairable here. Manipulation, approach,
    release, static-body, and continuous-grasp collisions remain ordinary hard
    failures and must never be hidden behind a waypoint search.
    """
    if stage_index <= 0 or ph._same_contact_sequence(
        stages[stage_index - 1], stages[stage_index]
    ):
        return None
    required = float(stages[stage_index].get("minimum_swept_clearance_m", 0.0))
    failing = [
        item
        for item in dense_plan.swept_clearance_violations
        if item.get("object_link") is not None
        and float(item.get("distance_m", math.inf)) < required
    ]
    if not failing or any(item.get("phase") != "transit_in" for item in failing):
        return None
    moved_by_joint = _joint_moved_link_sets(simulation_urdf)
    prior_moved_links: set[str] = set()
    source_joints: dict[str, list[str]] = {}
    for prior in stages[:stage_index]:
        joint = str(prior["driver_joint"])
        links = moved_by_joint.get(joint, set())
        prior_moved_links.update(links)
        for link in links:
            source_joints.setdefault(link, []).append(joint)
    blocking_links = sorted({str(item["object_link"]) for item in failing})
    if any(link not in prior_moved_links for link in blocking_links):
        return None
    tightest = min(failing, key=lambda item: float(item["distance_m"]))
    obstacle_link = str(tightest["object_link"])
    return {
        "incoming_stage_id": str(stages[stage_index]["id"]),
        "incoming_stage_index": int(stage_index),
        "classification": "prior_plan_moved_link_blocks_transit",
        "blocking_object_links": blocking_links,
        "primary_obstacle_link": obstacle_link,
        "moved_by_prior_driver_joints": sorted(set(source_joints[obstacle_link])),
        "minimum_direct_transit_clearance_m": float(tightest["distance_m"]),
        "required_clearance_m": required,
        "route_solver_required_before_rejection": True,
    }


def _summarize_manipulation_blocks(
    stages: list[dict[str, Any]], stage_reports: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Expose whole-block feasibility and the shadow states used to score it."""
    reports = {str(item["stage_id"]): item for item in stage_reports}
    stage_defs = {str(item["id"]): item for item in stages}
    summaries: list[dict[str, Any]] = []
    for block_index, stage_ids in enumerate(_manipulation_block_stage_ids(stages)):
        rows = [reports[stage_id] for stage_id in stage_ids]
        sequence = stage_defs[stage_ids[0]].get("contact_sequence")
        summaries.append(
            {
                "block_id": f"block_{block_index}",
                "contact_sequence": sequence,
                "stage_ids": stage_ids,
                "object_state_before": rows[0].get("object_state_before", {}),
                "object_state_after": rows[-1].get("object_state_after", {}),
                "feasible": all(bool(row.get("feasible")) for row in rows),
                "minimum_sample_completion_ratio": min(
                    int(row.get("samples_solved", 0))
                    / max(int(row.get("samples_required", 1)), 1)
                    for row in rows
                ),
                "target_actually_gripped": all(
                    bool(row.get("target_actually_gripped")) for row in rows
                ),
                "deepest_body_penetration_m": min(
                    float(row.get("deepest_body_penetration_m", 0.0))
                    for row in rows
                ),
                "maximum_target_link_gap_m": max(
                    float(row.get("maximum_target_link_gap_m", 1.0))
                    for row in rows
                ),
                "maximum_ik_position_error_m": max(
                    float(row.get("maximum_ik_position_error_m", 1.0))
                    for row in rows
                ),
                "minimum_gpu_environment_clearance_m": min(
                    (
                        float(row["curobo_minimum_environment_clearance_m"])
                        for row in rows
                        if row.get("curobo_minimum_environment_clearance_m") is not None
                    ),
                    default=None,
                ),
            }
        )
    return summaries


def _candidate_rank(report: dict[str, Any]) -> tuple:
    """Rank by the worst manipulation block before aggregate quality.

    A base that is excellent for the first block and unusable for a later block
    must rank below a balanced base.  Visual priority remains only a final
    tie-break among already visual-valid candidates.
    """
    blocks = report.get("manipulation_blocks")
    if not isinstance(blocks, list) or not blocks:
        blocks = [
            {
                "feasible": bool(stage.get("feasible", False)),
                "minimum_sample_completion_ratio": int(stage.get("samples_solved", 0))
                / max(int(stage.get("samples_required", 1)), 1),
                "target_actually_gripped": bool(stage.get("target_actually_gripped")),
                "deepest_body_penetration_m": float(
                    stage.get("deepest_body_penetration_m", 0.0)
                ),
                "maximum_target_link_gap_m": float(
                    stage.get("maximum_target_link_gap_m", 1.0)
                ),
                "maximum_ik_position_error_m": float(
                    stage.get("maximum_ik_position_error_m", 1.0)
                ),
                "minimum_gpu_environment_clearance_m": stage.get(
                    "curobo_minimum_environment_clearance_m"
                ),
            }
            for stage in report["stages"]
        ]

    def block_rank(block: dict[str, Any]) -> tuple:
        gpu_clearance = block.get("minimum_gpu_environment_clearance_m")
        return (
            not bool(block.get("feasible", False)),
            not bool(block.get("target_actually_gripped", False)),
            1.0 - float(block.get("minimum_sample_completion_ratio", 0.0)),
            float("inf") if gpu_clearance is None else -float(gpu_clearance),
            max(0.0, -float(block.get("deepest_body_penetration_m", 0.0))),
            float(block.get("maximum_target_link_gap_m", 1.0)),
            float(block.get("maximum_ik_position_error_m", 1.0)),
        )

    worst_block = max(
        (block_rank(block) for block in blocks),
        default=(True, True, 1.0, float("inf"), 1.0, 1.0, 1.0),
    )
    all_blocks_feasible = all(bool(block.get("feasible")) for block in blocks)
    visual_penalty = sum(
        max(0, int(priority) - 1)
        for priority in report.get("orientation_visual_priorities", [])
    )
    lateral_cost = abs(float(report.get("contact_facing_lateral_offset_m") or 0.0))
    return (
        not all_blocks_feasible,
        worst_block,
        lateral_cost,
        visual_penalty,
    )


def _contact_facing_sparse_matrix(
    object_yaws: list[Any],
    robot_z: list[Any],
    distances: list[Any],
    yaw_offsets: list[Any],
    lateral_offsets: list[Any],
) -> list[tuple[Any, ...]]:
    """Return every declared contact-facing placement cell exactly once."""
    return list(
        itertools.product(
            object_yaws, robot_z, distances, yaw_offsets, lateral_offsets
        )
    )


def _sparse_survivors(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows eligible for the dense top-K budget."""
    return [record for record in records if record.get("sparse_screening_passed")]


def _dense_shortlist(
    sparse_survivors: list[dict[str, Any]], top_k: int
) -> list[dict[str, Any]]:
    """Cost-order the bounded set allowed to spend dense planning work."""
    if int(top_k) <= 0:
        raise ValueError("dense top-k must be positive")
    return sorted(sparse_survivors, key=_candidate_rank)[: int(top_k)]


def _bounded_batches(
    records: Iterable[dict[str, Any]], batch_size: int
) -> Iterable[list[dict[str, Any]]]:
    """Lazily yield deterministic consecutive sparse batches."""
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    iterator = iter(records)
    while True:
        batch = list(itertools.islice(iterator, int(batch_size)))
        if not batch:
            return
        yield batch


def _run_ordered_until_accepted(
    records: list[dict[str, Any]],
    evaluate: Callable[[dict[str, Any]], dict[str, Any]],
    accept: Callable[[int, dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any] | None:
    """Evaluate cost-ordered rows until one passes every downstream gate."""
    for position, record in enumerate(records, start=1):
        report = evaluate(record)
        if accept(position, record, report):
            return record
    return None


def _release_gate_dense_candidate(
    task: dict[str, Any],
    record: dict[str, Any],
    report: dict[str, Any],
    *,
    release_solve: Callable[..., dict[str, Any]] | None = None,
) -> bool:
    """Require a dense candidate's release route before accepting placement.

    A dense manipulation pass is only an intermediate result.  If its execution
    declares a release boundary, solve that boundary immediately, write the
    measured route into the candidate, and return true only when the route also
    passes.  A failure is retained as diagnostics so the caller can continue to
    the next dense row instead of terminating the entire task.
    """
    if not bool(report.get("feasible")):
        return False
    execution = report.get("execution")
    if not isinstance(execution, dict):
        record["full_chain_feasible"] = False
        record["release_clearance_error"] = "dense_report_missing_execution"
        return False
    release_stages = [
        stage
        for stage in execution.get("stages", [])
        if stage.get("release_before_phase") is not None
    ]
    record["dense_feasible"] = True
    if not release_stages:
        record["release_clearance_required"] = False
        record["release_clearance_passed"] = True
        record["full_chain_feasible"] = True
        record["_output_execution"] = execution
        return True

    record["release_clearance_required"] = True
    if release_solve is None:
        import solve_artimo_release_clearance as release_solver

        release_solve = release_solver.solve
    try:
        release_report = release_solve(
            task,
            execution,
            planned_stages=report.get("_dense_plans"),
        )
    except Exception as exc:
        record["release_clearance_passed"] = False
        record["full_chain_feasible"] = False
        record["feasible"] = False
        record["release_clearance_error"] = str(exc)[:2000]
        record.pop("_output_execution", None)
        return False

    record["release_clearance"] = release_report
    chosen = release_report.get("chosen")
    if not isinstance(chosen, dict) or not bool(chosen.get("passed")):
        record["release_clearance_passed"] = False
        record["full_chain_feasible"] = False
        record["feasible"] = False
        record.pop("_output_execution", None)
        return False

    stage_id = str(release_report.get("stage_id", ""))
    merged = copy.deepcopy(execution)
    matching = [stage for stage in merged["stages"] if str(stage["id"]) == stage_id]
    if len(matching) != 1:
        record["release_clearance_passed"] = False
        record["full_chain_feasible"] = False
        record["feasible"] = False
        record["release_clearance_error"] = (
            f"release solver stage {stage_id!r} did not match exactly one execution stage"
        )
        record.pop("_output_execution", None)
        return False
    matching[0]["release_retreat_waypoints_world"] = [
        {
            "translation_m": [float(value) for value in chosen["translation_m"]],
            "rotation_xyzw": [float(value) for value in chosen["rotation_xyzw"]],
        }
    ]
    matching[0]["minimum_release_swept_clearance_m"] = float(
        chosen["minimum_clearance_m"]
    )
    record["release_clearance_passed"] = True
    record["full_chain_feasible"] = True
    record["feasible"] = True
    record["_output_execution"] = merged
    return True


def _feasible_region_summary(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-block regions and their whole-task intersection."""
    block_regions: dict[str, set[tuple[float, float, float, float]]] = {}
    task_region: set[tuple[float, float, float, float]] = set()
    choices_by_base: dict[
        tuple[float, float, float, float], set[tuple[str, ...]]
    ] = {}
    for attempt in attempts:
        base = attempt.get("robot_base_m")
        if not isinstance(base, list) or len(base) != 3:
            continue
        key = (
            round(float(base[0]), 9),
            round(float(base[1]), 9),
            round(float(base[2]), 9),
            round(float(attempt.get("robot_yaw_deg", 0.0)), 9),
        )
        for block in attempt.get("manipulation_blocks", []):
            if bool(block.get("feasible")):
                block_regions.setdefault(str(block["block_id"]), set()).add(key)
        if bool(attempt.get("feasible")):
            task_region.add(key)
            choices_by_base.setdefault(key, set()).add(
                tuple(
                    str(value)
                    for value in attempt.get("orientation_candidate_ids", [])
                )
            )

    def row(key: tuple[float, float, float, float]) -> dict[str, Any]:
        return {
            "robot_base_m": list(key[:3]),
            "robot_yaw_deg": key[3],
        }

    return {
        "block_feasible_base_regions": {
            block_id: [row(key) for key in sorted(keys)]
            for block_id, keys in sorted(block_regions.items())
        },
        "whole_task_feasible_base_region": [
            {
                **row(key),
                "orientation_candidate_combinations": [
                    list(choice) for choice in sorted(choices_by_base.get(key, set()))
                ],
            }
            for key in sorted(task_region)
        ],
    }


def solve(
    config: dict[str, Any],
    task: dict[str, Any],
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    solve_started_at = time.perf_counter()

    def emit_progress(event: str, **fields: Any) -> None:
        row = {
            "schema_version": 1,
            "event": str(event),
            "elapsed_s": round(time.perf_counter() - solve_started_at, 6),
            **fields,
        }
        if progress is not None:
            progress(row)
        print(
            "PLACEMENT_PROGRESS "
            + json.dumps(row, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

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
    template = _seed_rule_based_grasp_depths(
        ph.materialize_execution_defaults(task, template)
    )
    planning_backend = ph._application_planning_backend()
    if planning_backend.get("name", "bullet") == "curobo":
        # Carry the selected generic accelerator into the runnable execution so
        # final dense planning and any moved-obstacle transit use the same GPU
        # backend instead of silently rebuilding those paths on CPU.
        planning_backend.setdefault("allow_bullet_fallback", False)
        template["planning_ik_backend"] = planning_backend
    source_urdf = ph._resolve(inputs["urdf"])
    simulation_urdf = ph.resolve_simulation_urdf(
        task, template, source_urdf
    )
    ph._require_matching_mechanism(source_urdf, simulation_urdf)
    robot_urdf = ph._resolve(inputs["robot_urdf"])
    initial = ph.task_initial_joint_values(task)
    plan = ph._read_json(ph._resolve(inputs["plan"]))
    ik_backend = create_curobo_backend(
        {"planning_ik_backend": planning_backend}, robot_urdf, template["robot"]
    )

    bounds = _application_contact_facing_bounds(task)
    # Match the final rollout exactly. A 129-point non-sequential DP can accept
    # one IK branch while the 257-point sequential execution selects or falls
    # back to another branch that intersects the object.
    dense_samples = ph.DENSE_MANIPULATION_PATH_SAMPLES
    # Five object-side fractions per contact stage provide the initial whole-task
    # keyframes. Dense counterexamples are promoted into this set on demand.
    sparse_samples = 5
    # Every sparse survivor in the bounded GPU batch is eligible for ordered
    # dense verification.  Dense remains serial and stops at the first full
    # manipulation + release pass, but failure of the cheapest survivor must
    # not silently discard the other feasible bases from the same batch.
    dense_top_k = SPARSE_GPU_BASE_BATCH_SIZE
    # Dense rows are already deterministically ordered by base cost and then by
    # shallow-to-deep rule-based grasp depth. Run exactly one at a time and stop
    # on the first full-path pass; speculative parallel dense work can return a
    # later/deeper row first and wastes GPU time after an earlier solution.
    dense_candidate_jobs = 1
    sparse_ik_restarts = 12
    sparse_ik_iterations = 1000
    sparse_maximum_joint_step = 1.2
    if sparse_samples < 3:
        raise ValueError("sparse_path_samples must be at least 3")
    if dense_samples < sparse_samples:
        raise ValueError("path_samples must be at least sparse_path_samples")
    if dense_top_k <= 0:
        raise ValueError("dense_top_k must be positive")
    if dense_candidate_jobs <= 0:
        raise ValueError("dense_candidate_jobs must be positive")
    if dense_candidate_jobs > 1:
        backend = planning_backend
        if not isinstance(backend, dict) or backend.get("name", "bullet") != "curobo":
            raise ValueError(
                "dense_candidate_jobs greater than one requires the cuRobo backend"
            )
    if sparse_ik_restarts < 0:
        raise ValueError("sparse IK restart budget must be nonnegative")
    if sparse_ik_iterations <= 0:
        raise ValueError("sparse IK iteration budget must be positive")
    allowed_penetration = 0.002
    # How close an allowed contact link must come to the driven link for the grasp
    # to count as real rather than a near miss.
    maximum_grasp_gap = 0.006

    object_yaws = bounds.get("object_yaw_deg", [0.0])
    placement_mode = "contact_facing"
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
        orientation_options = _gated_orientation_options(template, orientation_gate)
    else:
        tilts = bounds.get("approach_tilt_deg", [0.0])
        spins = bounds.get("approach_spin_deg", [0.0])
        rolls = bounds.get("approach_roll_deg", [0.0])
        orientation_options = [
            {
                "execution": template,
                "candidate_ids": [],
                "visual_priorities": [],
                "angles": values,
            }
            for values in itertools.product(tilts, spins, rolls)
        ]

    if placement_mode == "contact_facing":
        distances = bounds.get("contact_facing_distance_m")
        if not isinstance(distances, list) or not distances:
            raise ValueError(
                "contact_facing placement requires non-empty bounds.contact_facing_distance_m"
            )
        distances = _center_out_numeric_values(distances)
        robot_z = _center_out_numeric_values(robot_z)
        yaw_offsets = _center_out_numeric_values(
            bounds.get("contact_facing_yaw_offset_deg", [0.0]), center=0.0
        )
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

    if placement_mode != "contact_facing":
        raise ValueError("Center-out placement currently requires contact_facing bounds")
    if len(object_yaws) != 1:
        raise ValueError("Object yaw must be frozen before center-out placement")

    attempts: list[dict[str, Any]] = []
    all_discovered: list[dict[str, Any]] = []
    index = -1
    candidate_construction_elapsed_s = 0.0
    contact_frame_cache_hits = 0
    contact_frame_cache_misses = 0
    oyaw = float(object_yaws[0])
    oriented_cache: dict[int, tuple[dict[str, Any], Any, Any, Any]] = {}

    reference_execution = json.loads(json.dumps(orientation_options[0]["execution"]))
    reference_execution["scene"]["object_base_rotation_xyzw"] = _yaw_quat(oyaw)
    whole_task_reference = _whole_task_moving_link_reference_world(
        simulation_urdf,
        robot_urdf,
        reference_execution,
        initial,
        plan,
        samples_per_stage=sparse_samples,
    )
    task_reference = whole_task_reference["reference_world_m"]
    outward_axis = whole_task_reference["outward_axis_world"]
    sparse_placements = _contact_facing_sparse_matrix(
        object_yaws, robot_z, distances, yaw_offsets, lateral_offsets
    )
    sparse_candidate_total = len(sparse_placements) * len(orientation_options)
    sparse_batches_total = max(
        1,
        int(math.ceil(sparse_candidate_total / SPARSE_GPU_BASE_BATCH_SIZE)),
    )
    emit_progress(
        "search_domain_ready",
        search_mode="whole_task_center_out_stream",
        whole_task_reference_world_m=task_reference,
        whole_task_keyframe_count=len(whole_task_reference["keyframes"]),
        whole_task_moving_link_count=len(whole_task_reference["moving_links"]),
        whole_task_moving_links=[
            item["moving_link"] for item in whole_task_reference["moving_links"]
        ],
        orientation_combinations=len(orientation_options),
        sparse_candidate_total=sparse_candidate_total,
        sparse_batch_count=sparse_batches_total,
        maximum_gpu_batch_size=int(SPARSE_GPU_BASE_BATCH_SIZE),
    )

    def make_center_out_record(
        state: dict[str, Any], orientation_index: int, iteration: int
    ) -> dict[str, Any]:
        nonlocal index, candidate_construction_elapsed_s
        construction_started_at = time.perf_counter()
        orientation_option = orientation_options[int(orientation_index)]
        oriented_row = oriented_cache.get(int(orientation_index))
        if oriented_row is None:
            orientation = orientation_option["angles"]
            oriented = json.loads(json.dumps(orientation_option["execution"]))
            oriented["scene"]["object_base_rotation_xyzw"] = _yaw_quat(oyaw)
            if orientation is None:
                tilt = spin = roll = None
            else:
                tilt, spin, roll = orientation
                for stage in oriented["stages"]:
                    stage["contact_pose_link"]["rotation_xyzw"] = _tilted_approach(
                        tilt, spin, roll
                    )
            oriented_row = (oriented, tilt, spin, roll)
            oriented_cache[int(orientation_index)] = oriented_row
        oriented, tilt, spin, roll = oriented_row
        robot_base = [float(value) for value in state["base_translation_m"]]
        facing = math.degrees(
            math.atan2(
                float(task_reference[1] - robot_base[1]),
                float(task_reference[0] - robot_base[0]),
            )
        )
        ryaw = facing + float(state.get("yaw_offset_deg", 0.0))
        placed = json.loads(json.dumps(oriented))
        placed["robot"]["base_translation_m"] = robot_base
        placed["robot"]["base_rotation_xyzw"] = _yaw_quat(ryaw)
        placed["robot"]["base_support_height_m"] = max(0.0, robot_base[2])
        index += 1
        record = {
            "index": index,
            "search_iteration": int(iteration),
            "search_move": "declared_center_out_grid",
            "object_yaw_deg": oyaw,
            "robot_base_m": robot_base,
            "robot_yaw_deg": ryaw,
            "placement_mode": "whole_task_center_out_stream",
            "whole_task_reference_world_m": list(task_reference),
            "contact_facing_distance_m": float(
                state["contact_facing_distance_m"]
            ),
            "contact_facing_lateral_offset_m": float(
                state["contact_facing_lateral_offset_m"]
            ),
            "contact_facing_yaw_offset_deg": float(
                state.get("yaw_offset_deg", 0.0)
            ),
            "approach_tilt_deg": tilt,
            "approach_spin_deg": spin,
            "approach_roll_deg": roll,
            "orientation_index": int(orientation_index),
            "orientation_candidate_ids": list(orientation_option["candidate_ids"]),
            "orientation_visual_priorities": list(
                orientation_option["visual_priorities"]
            ),
            "contact_point_link_m": list(
                placed["stages"][0]["contact_pose_link"]["translation_m"]
            ),
            "feasible": False,
            "_candidate_execution": placed,
        }
        try:
            ph._validate_execution_against_plan(
                plan, placed, require_release_route=False
            )
        except Exception as exc:
            record["rejected"] = str(exc)[:200]
        candidate_construction_elapsed_s += (
            time.perf_counter() - construction_started_at
        )
        return record

    def score_sparse(record: dict[str, Any], backend: Any) -> dict[str, Any]:
        return _score_candidate(
            simulation_urdf,
            robot_urdf,
            record["_candidate_execution"],
            initial,
            plan,
            sparse_samples,
            allowed_penetration,
            maximum_grasp_gap,
            validation_tier="sparse",
            run_full_path_confirmation=False,
            ik_random_restarts=sparse_ik_restarts,
            ik_max_iterations=sparse_ik_iterations,
            maximum_screening_joint_step_rad=sparse_maximum_joint_step,
            ik_backend=backend,
        )

    sparse_backend: Any = ik_backend
    sparse_batch_proxy: _SparseBatchIKProxy | None = None
    if ik_backend is not None:
        sparse_batch_proxy = _SparseBatchIKProxy(
            ik_backend, SPARSE_GPU_BASE_BATCH_SIZE
        )
        sparse_backend = sparse_batch_proxy

    def run_sparse(record: dict[str, Any]) -> tuple[dict[str, Any] | None, Exception | None]:
        try:
            return score_sparse(record, sparse_backend), None
        except Exception as exc:
            return None, exc

    def refine_records(
        records: list[dict[str, Any]],
        *,
        tier: str,
        sample_count: int,
        full_path: bool,
        ik_restarts: int | None = None,
        ik_iterations: int | None = None,
    ) -> list[dict[str, Any]]:
        passed: list[dict[str, Any]] = []

        def score(record: dict[str, Any], backend: Any) -> dict[str, Any]:
            if tier == "dense":
                emit_progress(
                    "dense_candidate_started",
                    candidate_index=int(record["index"]),
                    rule_based_grasp_depth_m_by_group=record.get(
                        "rule_based_grasp_depth_m_by_group", {}
                    ),
                )
            return _score_candidate(
                simulation_urdf,
                robot_urdf,
                record["_candidate_execution"],
                initial,
                plan,
                sample_count,
                allowed_penetration,
                maximum_grasp_gap,
                validation_tier=tier,
                run_full_path_confirmation=full_path,
                ik_random_restarts=ik_restarts,
                ik_max_iterations=ik_iterations,
                ik_backend=backend,
            )

        def apply_report(
            position: int, record: dict[str, Any], report: dict[str, Any]
        ) -> None:
            record["latest_validation_tier"] = tier
            record["stages"] = report["stages"]
            record["manipulation_blocks"] = report["manipulation_blocks"]
            record[f"{tier}_screening_passed"] = bool(
                report["screening_passed"]
            )
            if full_path:
                record["dense_full_path_confirmation_ran"] = True
                record["feasible"] = bool(report["feasible"])
                record["_output_execution"] = report["execution"]
                record["transit_route_repair_required"] = bool(
                    report.get("transit_route_repair_required")
                )
                record["transit_route_repairs"] = report.get(
                    "transit_route_repairs", []
                )
                if record["transit_route_repair_required"]:
                    # This execution is input only to the bounded transit route
                    # solver. It is deliberately kept separate from runnable
                    # placement execution until that solver succeeds.
                    record["_transit_route_repair_execution"] = report["execution"]
            print(
                f"[{tier} {position}/{len(records)}] "
                f"base={record['robot_base_m']} "
                f"orientation={'/'.join(record['orientation_candidate_ids'])} "
                f"solved={[stage['samples_solved'] for stage in report['stages']]} "
                f"required={[stage['samples_required'] for stage in report['stages']]} "
                f"{'PASS' if report['feasible'] else 'REJECT'}",
                flush=True,
            )

        def accept_report(
            position: int, record: dict[str, Any], report: dict[str, Any]
        ) -> bool:
            apply_report(position, record, report)
            accepted = bool(report["feasible"])
            if tier == "dense" and accepted:
                accepted = _release_gate_dense_candidate(task, record, report)
                if accepted:
                    try:
                        ph._validate_execution_against_plan(
                            plan,
                            record["_output_execution"],
                            require_release_route=True,
                        )
                    except Exception as exc:
                        accepted = False
                        record["release_clearance_passed"] = False
                        record["full_chain_feasible"] = False
                        record["feasible"] = False
                        record["release_clearance_error"] = str(exc)[:2000]
                        record.pop("_output_execution", None)
            if accepted:
                passed.append(record)
                print(
                    "first full-chain candidate passed sparse, dense, and release; "
                    "stopping search",
                    flush=True,
                )
            elif tier == "dense" and bool(report["feasible"]):
                print(
                    f"[dense {position}/{len(records)}] manipulation passed but "
                    "release failed; continuing to the next candidate",
                    flush=True,
                )
            if tier == "dense":
                emit_progress(
                    "dense_candidate_finished",
                    candidate_index=int(record["index"]),
                    dense_position=position,
                    dense_candidate_count=len(records),
                    manipulation_feasible=bool(report["feasible"]),
                    failed_stage_ids=[
                        str(stage["stage_id"])
                        for stage in report.get("stages", [])
                        if not bool(stage.get("feasible"))
                    ],
                    stage_sample_completion=[
                        {
                            "stage_id": str(stage["stage_id"]),
                            "solved": int(stage.get("samples_solved", 0)),
                            "required": int(stage.get("samples_required", 0)),
                            "full_path_rejected": stage.get(
                                "full_path_rejected"
                            ),
                        }
                        for stage in report.get("stages", [])
                    ],
                    release_clearance_passed=bool(
                        record.get("release_clearance_passed", False)
                    ),
                    full_chain_accepted=bool(accepted),
                )
            return accepted

        if tier != "dense" or dense_candidate_jobs == 1 or len(records) <= 1:
            _run_ordered_until_accepted(
                records,
                lambda record: score(record, ik_backend),
                accept_report,
            )
            return passed

        # Dense candidates are independent. Give each executor thread its own
        # persistent cuRobo subprocess: sharing one line-delimited worker would
        # serialize requests and corrupt request/response ownership. PyBullet
        # scoring already creates an isolated DIRECT client per candidate.
        worker_local = threading.local()
        worker_lock = threading.Lock()
        worker_backends: list[Any] = []

        def thread_backend() -> Any:
            backend = getattr(worker_local, "backend", None)
            if backend is None:
                backend = create_curobo_backend(
                    config, robot_urdf, template["robot"]
                )
                worker_local.backend = backend
                with worker_lock:
                    worker_backends.append(backend)
            return backend

        def parallel_score(record: dict[str, Any]) -> dict[str, Any]:
            return score(record, thread_backend())

        try:
            worker_count = min(dense_candidate_jobs, len(records))
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="artimo-dense",
            )
            pending_records = iter(enumerate(records, start=1))
            futures: dict[Any, tuple[int, dict[str, Any]]] = {}

            def submit_next() -> bool:
                try:
                    position, record = next(pending_records)
                except StopIteration:
                    return False
                futures[executor.submit(parallel_score, record)] = (position, record)
                return True

            for _ in range(worker_count):
                submit_next()
            found = False
            while futures and not found:
                done, _ = concurrent.futures.wait(
                    futures,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    position, record = futures.pop(future)
                    report = future.result()
                    if accept_report(position, record, report):
                        found = True
                        break
                    submit_next()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            for backend in worker_backends:
                if backend is not None:
                    backend.close()
        return passed

    dense_depth_attempts: list[dict[str, Any]] = []

    def run_funnel(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal ik_backend
        sparse_survivors = _sparse_survivors(records)
        base_shortlist = sorted(
            sparse_survivors, key=lambda record: int(record["index"])
        )[:dense_top_k]
        if dense_candidate_jobs > 1 and ik_backend is not None:
            # Do not retain the sparse worker while the configured dense worker
            # pool allocates its own models and CUDA graphs.
            ik_backend.close()
            ik_backend = None
        if not base_shortlist:
            return []
        for base_record in base_shortlist:
            groups = _rule_based_grasp_depth_groups(
                base_record["_candidate_execution"]
            )
            if not groups:
                dense_depth_attempts.append(base_record)
                passed = refine_records(
                    [base_record],
                    tier="dense",
                    sample_count=dense_samples,
                    full_path=True,
                )
                if passed:
                    return passed
                continue

            def depth_record(depths: list[float]) -> dict[str, Any]:
                candidate_record = copy.deepcopy(base_record)
                selected: dict[str, float] = {}
                for (key, indices), depth in zip(groups, depths):
                    selected[str(key)] = float(depth)
                    for stage_index in indices:
                        candidate_record["_candidate_execution"]["stages"][
                            stage_index
                        ]["grasp_depth_m"] = float(depth)
                candidate_record["rule_based_grasp_depth_m_by_group"] = selected
                candidate_record["depth_search_policy"] = (
                    "center_out_rule_based_coordinate_search"
                )
                return candidate_record

            seen_depths: set[tuple[float, ...]] = set()

            def evaluate_depths(
                depths: list[float],
            ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
                key = tuple(round(float(value), 9) for value in depths)
                if key in seen_depths:
                    raise RuntimeError(
                        "Dense depth coordinate search repeated a point"
                    )
                seen_depths.add(key)
                candidate = depth_record(depths)
                dense_depth_attempts.append(candidate)
                passed = refine_records(
                    [candidate],
                    tier="dense",
                    sample_count=dense_samples,
                    full_path=True,
                )
                return candidate, passed

            depth_centers = [
                float(
                    base_record["_candidate_execution"]["stages"][indices[0]].get(
                        "grasp_depth_m", 0.0
                    )
                )
                for _, indices in groups
            ]
            selected_depths = list(depth_centers)
            current_best, passed = evaluate_depths(selected_depths)
            if passed:
                return passed

            attempted_groups: set[int] = set()
            while True:
                failed_groups = _failed_depth_group_indices(
                    base_record["_candidate_execution"],
                    groups,
                    current_best,
                    attempted_groups,
                )
                if not failed_groups:
                    break
                group_index = failed_groups[0]
                attempted_groups.add(group_index)
                coordinate_candidates = [current_best]
                for depth in _ordered_rule_based_grasp_depths(
                    depth_centers[group_index]
                ):
                    if abs(float(depth) - selected_depths[group_index]) < 1e-12:
                        continue
                    trial_depths = list(selected_depths)
                    trial_depths[group_index] = float(depth)
                    candidate, passed = evaluate_depths(trial_depths)
                    if passed:
                        return passed
                    coordinate_candidates.append(candidate)
                current_best = min(coordinate_candidates, key=_candidate_rank)
                selected = current_best.get(
                    "rule_based_grasp_depth_m_by_group", {}
                )
                selected_depths = [
                    float(selected[str(key)]) for key, _ in groups
                ]
                # If the group whose complete one-dimensional lattice was just
                # exhausted is still failed, no depth belonging to another
                # contact sequence can repair it.  Abandon this base and let
                # the funnel evaluate the next sparse survivor.  Advance to a
                # different group only when the current group has become
                # feasible and the failure has genuinely moved downstream.
                still_failed_groups = set(
                    _failed_depth_group_indices(
                        base_record["_candidate_execution"],
                        groups,
                        current_best,
                    )
                )
                if group_index in still_failed_groups:
                    break
        return []

    sparse_matrix_elapsed_s = 0.0
    dense_elapsed_s = 0.0
    final_feasible: list[dict[str, Any]] = []
    sparse_batches_evaluated = 0
    progressive_early_stop = False

    def ordered_specifications() -> Iterable[tuple[dict[str, Any], int]]:
        """Yield each declared center-out base/orientation cell exactly once."""
        for placement_values in sparse_placements:
            _, rz, distance, yaw_offset, lateral_offset = placement_values
            robot_base, _ = _contact_facing_base_pose(
                task_reference,
                outward_axis,
                float(distance),
                float(rz),
                float(yaw_offset),
                float(lateral_offset),
            )
            state = {
                "base_translation_m": robot_base,
                "yaw_offset_deg": float(yaw_offset),
                "source": "declared_center_out_grid",
                "contact_facing_distance_m": float(distance),
                "contact_facing_lateral_offset_m": float(lateral_offset),
            }
            for orientation_index in range(len(orientation_options)):
                yield state, orientation_index

    try:
        for iteration, specifications in enumerate(
            _bounded_batches(
                ordered_specifications(), SPARSE_GPU_BASE_BATCH_SIZE
            ),
            start=1,
        ):
            sparse_batches_evaluated = iteration
            construction_before = candidate_construction_elapsed_s
            sparse_batch = [
                make_center_out_record(state, orientation_index, iteration)
                for state, orientation_index in specifications
            ]
            construction_delta_s = (
                candidate_construction_elapsed_s - construction_before
            )
            emit_progress(
                "sparse_batch_constructed",
                batch_index=iteration,
                batch_count=sparse_batches_total,
                candidate_count=len(sparse_batch),
                first_candidate_index=int(sparse_batch[0]["index"]),
                last_candidate_index=int(sparse_batch[-1]["index"]),
                construction_elapsed_s=round(construction_delta_s, 6),
                first_robot_base_world_m=list(sparse_batch[0]["robot_base_m"]),
                last_robot_base_world_m=list(sparse_batch[-1]["robot_base_m"]),
                search_mode="whole_task_center_out_stream",
            )
            valid_batch = [
                record for record in sparse_batch if "rejected" not in record
            ]
            emit_progress(
                "sparse_batch_started",
                batch_index=iteration,
                batch_count=sparse_batches_total,
                valid_candidate_count=len(valid_batch),
            )
            sparse_started_at = time.perf_counter()
            if not valid_batch:
                sparse_results: list[
                    tuple[dict[str, Any] | None, Exception | None]
                ] = []
            elif sparse_batch_proxy is None or len(valid_batch) == 1:
                sparse_results = [run_sparse(record) for record in valid_batch]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(valid_batch),
                    thread_name_prefix="artimo-sparse-base",
                ) as executor:
                    sparse_results = list(executor.map(run_sparse, valid_batch))
            batch_sparse_elapsed_s = time.perf_counter() - sparse_started_at
            sparse_matrix_elapsed_s += batch_sparse_elapsed_s
            result_by_index = {
                int(record["index"]): result
                for record, result in zip(valid_batch, sparse_results)
            }
            for record in sparse_batch:
                report: dict[str, Any] | None = None
                if "rejected" not in record:
                    report, error = result_by_index[int(record["index"])]
                    if error is not None or report is None:
                        record["rejected"] = str(error)[:200]
                    else:
                        record.update(
                            {
                                "sparse_screening_passed": bool(
                                    report["screening_passed"]
                                ),
                                "latest_validation_tier": "sparse",
                                "stages": report["stages"],
                                "manipulation_blocks": report[
                                    "manipulation_blocks"
                                ],
                                "_grounding": report["grounding"],
                            }
                        )
                attempts.append(record)
                if report is not None:
                    orientation_label = (
                        "/".join(record["orientation_candidate_ids"])
                        if record["approach_tilt_deg"] is None
                        else (
                            f"tilt={record['approach_tilt_deg']:.0f} "
                            f"spin={record['approach_spin_deg']:.0f} "
                            f"roll={record['approach_roll_deg']:.0f}"
                        )
                    )
                    print(
                        f"[center-out {iteration}:{record['index']}] "
                        f"obj_yaw={record['object_yaw_deg']:3.0f} "
                        f"base=({record['robot_base_m'][0]:+.2f},"
                        f"{record['robot_base_m'][1]:+.2f},"
                        f"{record['robot_base_m'][2]:.2f}) "
                        f"yaw={record['robot_yaw_deg']:+.0f} "
                        f"orientation={orientation_label} "
                        f"pt={record['contact_point_link_m']} -> "
                        f"solved={[s['samples_solved'] for s in report['stages']]} "
                        f"pen={[s['deepest_body_penetration_m'] for s in report['stages']]} "
                        f"{'SPARSE_PASS' if report['screening_passed'] else ''}",
                        flush=True,
                    )

            batch_survivors = _sparse_survivors(sparse_batch)
            ranked_batch = [
                record for record in sparse_batch if record.get("stages")
            ]
            if not ranked_batch:
                continue
            best_sparse = min(ranked_batch, key=_candidate_rank)
            best_stages = best_sparse.get("stages", [])
            emit_progress(
                "sparse_batch_finished",
                batch_index=iteration,
                batch_count=sparse_batches_total,
                sparse_elapsed_s=round(batch_sparse_elapsed_s, 6),
                survivor_count=len(batch_survivors),
                best_candidate_index=int(best_sparse["index"]),
                best_complete_blocks=sum(
                    bool(block.get("feasible"))
                    for block in best_sparse.get("manipulation_blocks", [])
                ),
                best_samples_solved=sum(
                    int(stage.get("samples_solved", 0)) for stage in best_stages
                ),
                best_samples_required=sum(
                    int(stage.get("samples_required", 0)) for stage in best_stages
                ),
            )
            dense_candidates = sorted(
                batch_survivors, key=lambda record: int(record["index"])
            )[:dense_top_k]
            if dense_candidates:
                emit_progress(
                    "dense_scan_started",
                    batch_index=iteration,
                    sparse_survivor_count=len(batch_survivors),
                    dense_candidate_count=len(dense_candidates),
                )
                dense_started_at = time.perf_counter()
                batch_feasible = run_funnel(dense_candidates)
                batch_dense_elapsed_s = time.perf_counter() - dense_started_at
                dense_elapsed_s += batch_dense_elapsed_s
                emit_progress(
                    "dense_scan_finished",
                    batch_index=iteration,
                    dense_elapsed_s=round(batch_dense_elapsed_s, 6),
                    full_chain_candidate_count=len(batch_feasible),
                )
                if batch_feasible:
                    final_feasible = batch_feasible
                    progressive_early_stop = True
                    emit_progress(
                        "full_chain_accepted",
                        batch_index=iteration,
                        candidate_index=int(batch_feasible[0]["index"]),
                    )
                    break
    finally:
        if sparse_batch_proxy is not None:
            sparse_batch_proxy.close()

    dense_evaluated = [
        record
        for record in dense_depth_attempts
        if record.get("dense_full_path_confirmation_ran")
    ]
    route_repair_records = [
        record
        for record in dense_evaluated
        if record.get("transit_route_repair_required")
    ]
    ranked_pool = final_feasible or route_repair_records or dense_evaluated or attempts
    best = min(ranked_pool, key=_candidate_rank) if ranked_pool else None
    discovered = all_discovered
    feasible_regions = _feasible_region_summary(attempts)

    if best is None:
        raise RuntimeError("No placement candidate could be evaluated; check bounds and template")
    best_execution = best.get("_output_execution") if best["feasible"] else None
    best_route_repair = (
        {
            "required": True,
            "candidate_index": int(best["index"]),
            "repairs": best.get("transit_route_repairs", []),
            "execution": best.get("_transit_route_repair_execution"),
            "runnable_before_route_solve": False,
        }
        if best.get("transit_route_repair_required")
        else None
    )
    public_attempts = [
        {
            key: value
            for key, value in record.items()
            if not key.startswith("_")
        }
        for record in attempts + dense_depth_attempts
    ]
    feedback_execution = (
        best.get("_output_execution")
        or best.get("_candidate_execution")
        or template
    )
    collision_feedback = _collision_rejection_feedback(best, feedback_execution)
    backend_report = json.loads(json.dumps(planning_backend))
    if ik_backend is not None:
        ik_backend.close()
    emit_progress(
        "solve_finished",
        feasible=bool(best["feasible"]),
        sparse_batches_evaluated=sparse_batches_evaluated,
        sparse_batches_total=sparse_batches_total,
        progressive_early_stop=progressive_early_stop,
        sparse_elapsed_s=round(sparse_matrix_elapsed_s, 6),
        dense_elapsed_s=round(dense_elapsed_s, 6),
        candidate_construction_elapsed_s=round(
            candidate_construction_elapsed_s, 6
        ),
    )
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
            "orientation_candidate_ids": best.get("orientation_candidate_ids", []),
            "orientation_visual_priorities": best.get(
                "orientation_visual_priorities", []
            ),
            "rule_based_grasp_depth_m_by_group": best.get(
                "rule_based_grasp_depth_m_by_group", {}
            ),
            "contact_point_link_m": best["contact_point_link_m"],
            "stages": best["stages"],
            "manipulation_blocks": best.get("manipulation_blocks", []),
        },
        # Never emit a runnable execution from a merely "best available"
        # placement. One candidate must pass sparse screening, dense full-path
        # planning, and every declared release boundary first.
        "execution": best_execution,
        "transit_route_repair": best_route_repair,
        "agent_feedback": {
            "collision_and_depth": collision_feedback,
        },
        "search": {
            "planning_ik_backend": backend_report,
            "orientation_gate": orientation_gate,
            "sparse_path_samples": sparse_samples,
            "adaptive_dense_max_samples": dense_samples,
            "dense_top_k": dense_top_k,
            "dense_candidate_jobs": dense_candidate_jobs,
            "progressive_sparse_batches_evaluated": sparse_batches_evaluated,
            "progressive_sparse_batches_total": sparse_batches_total,
            "progressive_early_stop": progressive_early_stop,
            "search_mode": "whole_task_center_out_stream",
            "maximum_sparse_candidates": sparse_candidate_total,
            "whole_task_reference": whole_task_reference,
            "sparse_gpu_base_batch_size": (
                SPARSE_GPU_BASE_BATCH_SIZE if ik_backend is not None else 1
            ),
            "sparse_gpu_batching": (
                "multi_base_single_worker_solve_batch_env"
                if ik_backend is not None
                else "disabled_for_bullet_backend"
            ),
            "sparse_matrix_elapsed_s": round(sparse_matrix_elapsed_s, 6),
            "dense_elapsed_s": round(dense_elapsed_s, 6),
            "candidate_construction_elapsed_s": round(
                candidate_construction_elapsed_s, 6
            ),
            "total_solve_elapsed_s": round(
                time.perf_counter() - solve_started_at, 6
            ),
            "sparse_ik_random_restarts": sparse_ik_restarts,
            "sparse_ik_max_iterations": sparse_ik_iterations,
            "allowed_body_penetration_m": allowed_penetration,
            "maximum_grasp_gap_m": maximum_grasp_gap,
            "grasp_depth_policy": {
                "owner": "application_rule_based_dense_search",
                "agent_supplied_depth_is_search_center_only": True,
                "ordered_offsets_from_agent_center_m": list(
                    RULE_BASED_GRASP_DEPTH_OFFSETS_M
                ),
                "adjustment_bounds_m": list(
                    RULE_BASED_GRASP_DEPTH_ADJUSTMENT_BOUNDS_M
                ),
                "acceptance": (
                    "both_application_owned_finger_links_within_target_gap_"
                    "and_no_forbidden_collision_over_dense_full_path"
                ),
            },
            "candidates_evaluated": len(attempts),
            "sparse_candidates_evaluated": sum(
                "sparse_screening_passed" in record for record in attempts
            ),
            "sparse_candidates_passed": sum(
                bool(record.get("sparse_screening_passed")) for record in attempts
            ),
            "dense_candidates_evaluated": len(dense_evaluated),
            "dense_candidates_manipulation_feasible": sum(
                bool(record.get("dense_feasible")) for record in dense_evaluated
            ),
            "dense_candidates_release_passed": sum(
                bool(record.get("release_clearance_passed"))
                for record in dense_evaluated
            ),
            "transit_route_repair_candidates": len(route_repair_records),
            "discovered_contact_points": discovered,
            "bounds": bounds,
            "attempts": public_attempts,
            "selection_policy": (
                "whole_task_center_out_declared_grid__"
                "bounded_gpu_sparse_batches__"
                "first_sparse_survivor_to_ordered_dense__"
                "release_failure_falls_through__stop_at_first_full_chain_pass"
            ),
            **feasible_regions,
        },
    }


def _collision_rejection_feedback(
    best: dict[str, Any], execution: dict[str, Any]
) -> dict[str, Any]:
    """Explain collision-owned rejection without accepting agent link policy.

    ``materialize_execution_defaults`` derives every stage's forbidden object
    links from the URDF (all object links except ``contact_link``) and derives
    the allowed Panda links from the interaction.  This report deliberately
    echoes those application-owned sets so a task agent only has to revise
    contact geometry such as ``grasp_depth_m``; it can neither hide an offender
    by editing a blacklist nor widen target-contact permission.
    """
    stage_definitions = {
        str(stage["id"]): stage for stage in execution.get("stages", [])
    }
    rejected_stages: list[dict[str, Any]] = []
    for report in best.get("stages", []):
        stage_id = str(report.get("stage_id", ""))
        stage = stage_definitions.get(stage_id)
        if stage is None:
            continue
        contact_link = str(stage["contact_link"])
        forbidden = {str(name) for name in stage.get("forbidden_contact_links", [])}
        allowed_robot = {
            str(name) for name in stage.get("allowed_robot_contact_links", [])
        }
        violations: list[dict[str, Any]] = []

        for pair, distance_value in report.get("worst_offenders", {}).items():
            distance = float(distance_value)
            robot_link, separator, object_link = str(pair).partition("|")
            if not separator:
                continue
            if object_link in forbidden:
                collision_class = "non_target_object_link"
            elif object_link == contact_link and robot_link not in allowed_robot:
                collision_class = "non_allowed_robot_link_on_target"
            else:
                continue
            violations.append({
                "collision_class": collision_class,
                "phase": "manipulation_screen",
                "robot_link": robot_link,
                "object_link": object_link,
                "distance_m": distance,
            })

        required_dense = float(
            report.get(
                "dense_full_path_required_clearance_m",
                report.get("required_forbidden_clearance_m", 0.0),
            )
        )
        for row in report.get("dense_full_path_tightest_samples", []):
            if not isinstance(row, dict) or row.get("object_link") is None:
                continue
            distance = float(row.get("distance_m", 0.0))
            if distance >= required_dense:
                continue
            robot_link = str(row.get("robot_link", "unknown"))
            object_link = str(row["object_link"])
            # cuRobo checks the application-owned forbidden source meshes as a
            # batch and may therefore report their collective obstacle label.
            # It is still collision evidence, but never an agent-authored link.
            if object_link == "source_mesh_environment":
                collision_class = "application_forbidden_source_mesh"
            elif object_link in forbidden:
                collision_class = "non_target_object_link"
            elif object_link == contact_link and robot_link not in allowed_robot:
                collision_class = "non_allowed_robot_link_on_target"
            else:
                continue
            violation = {
                "collision_class": collision_class,
                "phase": str(row.get("phase", "dense_full_path")),
                "robot_link": robot_link,
                "object_link": object_link,
                "distance_m": distance,
            }
            if row.get("sample") is not None:
                violation["sample"] = int(row["sample"])
            if row.get("reason") is not None:
                violation["reason"] = str(row["reason"])
            violations.append(violation)

        # A GPU batch can reject the forbidden world without an exact Bullet
        # pair. Preserve that conservative result rather than asking the agent
        # to guess which link should be ignored.
        if (
            report.get("forbidden_clearance_passed") is False
            and not violations
            and report.get("minimum_forbidden_clearance_m") is not None
        ):
            violations.append({
                "collision_class": "application_forbidden_source_mesh",
                "phase": "manipulation_screen",
                "robot_link": "curobo_collision_spheres",
                "object_link": "source_mesh_environment",
                "distance_m": float(report["minimum_forbidden_clearance_m"]),
            })

        if violations:
            rejected_stages.append({
                "stage_id": stage_id,
                "contact_link": contact_link,
                "grasp_depth_m": float(stage.get("grasp_depth_m", 0.0)),
                "effective_robot_contact_offset_m": ph._effective_grasp_depth(stage),
                "application_forbidden_contact_links": sorted(forbidden),
                "application_allowed_robot_contact_links": sorted(allowed_robot),
                "violations": violations,
            })

    rejected = bool(rejected_stages)
    return {
        "policy": (
            "application_derives_all_object_links_except_contact_link_as_forbidden; "
            "agent-supplied forbidden/allowed collision lists are ignored"
        ),
        "agent_collision_link_input_accepted": False,
        "depth_adjustment_valid": not rejected,
        "status": (
            "rejected_collision_at_proposed_depth"
            if rejected
            else "no_collision_attributed_depth_rejection"
        ),
        "rejected_stages": rejected_stages,
        "next_action": (
            "revise only task-local contact geometry (normally a shallower "
            "grasp_depth_m), rerender the complete five-roll visual batch, and "
            "rerun placement; do not edit collision-link lists"
            if rejected
            else "inspect the non-collision placement diagnostics before changing depth"
        ),
    }


def _application_contact_facing_bounds(task: dict[str, Any]) -> dict[str, Any]:
    """Build the full sparse base matrix from geometry and harness constants.

    Lateral coverage is the widest initial horizontal object-link extent, with
    a robot-independent minimum half-span.  The 5 cm grid and Panda working
    distance range are application policy; an agent cannot narrow them because
    a prior centered row failed or a command is taking too long.
    """
    object_urdf = ph._resolve(task["inputs"]["urdf"])
    initial = ph.task_initial_joint_values(task)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect placement-bounds geometry client")
    try:
        body = p.loadURDF(str(object_urdf), useFixedBase=True, physicsClientId=client)
        joints, _ = ph._maps(body, client)
        for name, value in initial.items():
            if name in joints:
                p.resetJointState(body, joints[name], float(value), physicsClientId=client)
        p.performCollisionDetection(physicsClientId=client)
        horizontal_extents = []
        for link_index in range(-1, p.getNumJoints(body, physicsClientId=client)):
            low, high = p.getAABB(body, link_index, physicsClientId=client)
            horizontal_extents.extend(
                [float(high[0] - low[0]), float(high[1] - low[1])]
            )
        widest = max(horizontal_extents, default=0.5)
    finally:
        p.disconnect(client)
    step = 0.05
    lateral_half_span = max(0.5, round(widest / step) * step)
    lateral_count = int(round(lateral_half_span / step))
    return {
        "object_yaw_deg": [0.0],
        "robot_base_z_m": [0.0],
        "contact_facing_distance_m": [
            round(0.35 + step * index, 10) for index in range(16)
        ],
        "contact_facing_lateral_offset_m": [
            round(step * index, 10)
            for index in range(-lateral_count, lateral_count + 1)
        ],
        "contact_facing_yaw_offset_deg": [0.0],
    }


def _run_transit_route_repair(
    task: dict[str, Any], answer: dict[str, Any], out: Path
) -> None:
    """Run the one bounded moved-obstacle repair batch selected by placement."""
    repair = answer.get("transit_route_repair")
    if not isinstance(repair, dict) or not repair.get("required"):
        return
    execution = repair.get("execution")
    repairs = repair.get("repairs", [])
    if not isinstance(execution, dict) or not repairs:
        return
    chosen_repair = min(
        repairs,
        key=lambda item: (
            int(item.get("incoming_stage_index", 1 << 30)),
            float(item.get("minimum_direct_transit_clearance_m", math.inf)),
        ),
    )
    repair_root = out / "transit-route-repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    candidate_execution_path = repair_root / "candidate-execution.json"
    candidate_execution_path.write_text(
        json.dumps(execution, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    repair["candidate_execution_path"] = str(candidate_execution_path)
    repair["selected_repair"] = chosen_repair
    repair.pop("execution", None)

    # Import lazily to keep ordinary placement startup independent of the
    # conditional route machinery.
    import propose_artimo_transit_routes as route_proposer
    import solve_artimo_transit_clearance as route_solver

    proposal = route_proposer.propose(
        task,
        execution,
        candidate_execution_path,
        str(chosen_repair["incoming_stage_id"]),
        str(chosen_repair["primary_obstacle_link"]),
        0.06,
        0.02,
        (0.55, 0.70, 0.85),
    )
    (repair_root / "proposal.json").write_text(
        json.dumps(proposal, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    routes = proposal.get("routes_config")
    if routes is None:
        repair["solved"] = False
        repair["failure"] = "route_proposer_emitted_no_bounded_routes"
        return
    (repair_root / "routes.json").write_text(
        json.dumps(routes, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transit = route_solver.solve(routes, task, jobs=4)
    (repair_root / "transit.json").write_text(
        json.dumps(transit, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    repair["solved"] = bool(transit["feasible"])
    repair["transit_report_path"] = str(repair_root / "transit.json")
    if transit["execution"] is not None:
        release_record: dict[str, Any] = {"feasible": True}
        release_report = {
            "feasible": True,
            "execution": transit["execution"],
            "_dense_plans": None,
        }
        release_passed = _release_gate_dense_candidate(
            task, release_record, release_report
        )
        repair["release_clearance_passed"] = bool(release_passed)
        if release_record.get("release_clearance") is not None:
            repair["release_clearance"] = release_record["release_clearance"]
        if release_record.get("release_clearance_error") is not None:
            repair["release_clearance_error"] = release_record[
                "release_clearance_error"
            ]
        if release_passed:
            object_plan = ph._read_json(ph._resolve(task["inputs"]["plan"]))
            try:
                ph._validate_execution_against_plan(
                    object_plan,
                    release_record["_output_execution"],
                    require_release_route=True,
                )
            except Exception as exc:
                release_passed = False
                repair["release_clearance_passed"] = False
                repair["release_clearance_error"] = str(exc)[:2000]
            else:
                answer["execution"] = release_record["_output_execution"]
                answer["feasible"] = True
        if not release_passed:
            answer["execution"] = None
            answer["feasible"] = False
            repair["solved"] = False
            repair["failure"] = "transit_passed_but_release_clearance_failed"
        repair["runnable_before_route_solve"] = False
        repair["runnable_after_route_solve"] = bool(release_passed)
        repair["chosen_route"] = transit["chosen"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Placement search bounds + execution template")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    def write_progress(row: dict[str, Any]) -> None:
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            stream.flush()

    try:
        task = ph._read_json(args.task_spec.expanduser().resolve())
        config = ph._read_json(args.config.expanduser().resolve())
        answer = solve(config, task, progress=write_progress)
        if answer["execution"] is None and answer.get("transit_route_repair"):
            _run_transit_route_repair(task, answer, out)
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
        write_progress(
            {
                "schema_version": 1,
                "event": "solver_failed",
                "error": str(exc)[:2000],
            }
        )
        print(f"Placement solve failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
