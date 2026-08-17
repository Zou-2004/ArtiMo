#!/usr/bin/env python3
"""Persistent cuRobo IK and source-mesh collision worker for ArtiMo.

The normal application environment deliberately does not import CUDA libraries.
This process runs in a separate, explicitly selected Python environment and
communicates through line-delimited JSON. Sparse and dense object collision are
screened against source collision meshes on the GPU. PyBullet remains
authoritative only for target-contact semantics and physical rollout.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

# cuRobo 0.7.x uses the pre-Warp-1.10 ``wp.torch`` namespace.  Current Warp
# exposes the same adapters at module level.  Keep the compatibility local to
# this isolated worker instead of modifying either installed package.
if not hasattr(wp, "torch"):
    wp.torch = wp

from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)


RESPONSE_PREFIX = "ARTIMO_CUROBO_RESPONSE "


def _local_poses(
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    base_position: np.ndarray,
    base_quaternion_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    base_rotation = Rotation.from_quat(base_quaternion_xyzw)
    local_positions = base_rotation.inv().apply(positions - base_position[None, :])
    local_rotations = base_rotation.inv() * Rotation.from_quat(quaternions_xyzw)
    xyzw = local_rotations.as_quat()
    wxyz = xyzw[:, [3, 0, 1, 2]]
    return np.ascontiguousarray(local_positions), np.ascontiguousarray(wxyz)


def _select_continuous_branch(
    candidates: np.ndarray,
    valid: np.ndarray,
    reference: np.ndarray,
    maximum_step: float | None,
    enforce_start_step: bool,
) -> tuple[list[int] | None, int | None]:
    count, seeds, _ = candidates.shape
    costs = np.full((count, seeds), np.inf, dtype=np.float64)
    parents = np.full((count, seeds), -1, dtype=np.int64)
    for seed in range(seeds):
        if not valid[0, seed]:
            continue
        delta = np.abs(candidates[0, seed] - reference)
        if enforce_start_step and maximum_step is not None and np.max(delta) > maximum_step:
            continue
        costs[0, seed] = float(np.linalg.norm(delta))
    if not np.isfinite(costs[0]).any():
        return None, 0
    for sample in range(1, count):
        for seed in range(seeds):
            if not valid[sample, seed]:
                continue
            delta = np.abs(candidates[sample - 1] - candidates[sample, seed][None, :])
            allowed = valid[sample - 1] & np.isfinite(costs[sample - 1])
            if maximum_step is not None:
                allowed &= np.max(delta, axis=1) <= maximum_step
            indices = np.flatnonzero(allowed)
            if not len(indices):
                continue
            edge_costs = costs[sample - 1, indices] + np.linalg.norm(delta[indices], axis=1)
            best_offset = int(np.argmin(edge_costs))
            parents[sample, seed] = int(indices[best_offset])
            costs[sample, seed] = float(edge_costs[best_offset])
        if not np.isfinite(costs[sample]).any():
            return None, sample
    selected = [int(np.argmin(costs[-1]))]
    for sample in range(count - 1, 0, -1):
        selected.append(int(parents[sample, selected[-1]]))
    selected.reverse()
    return selected, None


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.tensor_args = TensorDeviceType(device=torch.device(args.device), dtype=torch.float32)
        # ``RobotConfig.from_basic`` intentionally has no collision model. The
        # application robot is Panda, so reuse cuRobo's maintained Panda sphere
        # decomposition while keeping this repository's authoritative URDF and
        # end-effector link. This enables both self and world collision on GPU.
        robot_dict = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))[
            "robot_cfg"
        ]
        kinematics = robot_dict["kinematics"]
        kinematics["urdf_path"] = str(Path(args.robot_urdf).resolve())
        kinematics["asset_root_path"] = str(Path(args.robot_urdf).resolve().parent)
        kinematics["base_link"] = args.base_link
        kinematics["ee_link"] = args.end_effector_link
        kinematics["collision_link_names"] = [
            name
            for name in kinematics["collision_link_names"]
            if name != "attached_object"
        ]
        kinematics["extra_collision_spheres"] = None
        kinematics["extra_links"] = None
        # The maintained cuRobo Panda profile adds a 4 mm optimization buffer.
        # ArtiMo uses this worker as a sparse feasibility screen, not as a
        # clearance optimizer, so the task-declared buffer is explicit and zero
        # by default. Exact dense PyBullet validation remains authoritative.
        kinematics["collision_sphere_buffer"] = float(args.collision_sphere_buffer_m)
        # The second Panda finger is a mimic joint in this URDF. Locking its
        # driver fixes both collision geometries at the manipulation aperture.
        kinematics["lock_joints"] = {"panda_finger_joint1": 0.005}
        for key in ("joint_names", "retract_config", "null_space_weight", "cspace_distance_weight"):
            kinematics["cspace"][key] = kinematics["cspace"][key][:7]
        # Keep a task-neutral template so path-clearance requests can use the
        # physical gripper aperture for their phase.  The old worker locked the
        # fingers at 5 mm for every request, so an open 40 mm approach could
        # miss a cabinet strike by more than 3 cm.
        self.robot_dict_template = copy.deepcopy(robot_dict)
        robot = RobotConfig.from_dict(robot_dict, self.tensor_args)
        self.robot_config = robot
        self.robot_configs_by_finger_joint_value: dict[int, RobotConfig] = {
            5000: robot
        }
        self.arm_joint_names = list(args.arm_joint_names)
        model_names = list(robot.kinematics.kinematics_config.joint_names)
        missing = set(self.arm_joint_names) - set(model_names)
        if missing:
            raise ValueError(f"cuRobo robot model is missing arm joints {sorted(missing)}")
        self.output_indices = [model_names.index(name) for name in self.arm_joint_names]
        self.num_seeds = int(args.num_seeds)
        self.return_seeds = int(args.return_seeds)
        self.use_cuda_graph = bool(args.cuda_graph)
        self.self_collision = not bool(args.disable_self_collision)
        self.solvers: dict[tuple[int, int, int, int], IKSolver] = {}
        self.motion_num_graph_seeds = int(args.motion_num_graph_seeds)
        self.motion_num_trajopt_seeds = int(args.motion_num_trajopt_seeds)
        self.motion_timeout_s = float(args.motion_timeout_s)
        self.motion_max_attempts = int(args.motion_max_attempts)
        self.motion_generators: dict[tuple[int, int], MotionGen] = {}

    def _worlds(
        self, request: dict[str, Any], count: int | None = None
    ) -> list[WorldConfig]:
        if "obstacle_world" in request:
            rows = [request.get("obstacle_world", [])]
        else:
            rows = request.get(
                "obstacle_worlds_by_sample",
                request.get("obstacle_cuboids_world_by_sample", []),
            )
        if count is None:
            count = len(request["positions_world"])
        if not rows:
            rows = [[] for _ in range(count)]
        if len(rows) != count:
            raise ValueError("Obstacle-world samples must align with IK pose samples")
        base_position = np.asarray(request["robot_base_position_world"], dtype=np.float64)
        base_rotation = Rotation.from_quat(request["robot_base_quaternion_xyzw_world"])
        worlds: list[WorldConfig] = []
        for sample_index, obstacles in enumerate(rows):
            cuboids: list[Cuboid] = []
            meshes: list[Mesh] = []
            for obstacle_index, obstacle in enumerate(obstacles):
                world_position = np.asarray(
                    obstacle.get("position_world_m", obstacle.get("center_world_m")),
                    dtype=np.float64,
                )
                local_position = base_rotation.inv().apply(world_position - base_position)
                local_rotation = base_rotation.inv() * Rotation.from_quat(
                    obstacle.get("rotation_xyzw_world", [0.0, 0.0, 0.0, 1.0])
                )
                xyzw = local_rotation.as_quat()
                pose = [
                    *local_position.astype(float).tolist(),
                    float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]),
                ]
                name = str(obstacle.get("name", f"obstacle_{obstacle_index}"))
                if obstacle.get("geometry_type") == "mesh":
                    meshes.append(Mesh(
                        name=name,
                        pose=pose,
                        file_path=str(obstacle["file_path"]),
                        scale=[float(value) for value in obstacle.get("scale", [1.0, 1.0, 1.0])],
                    ))
                else:
                    cuboids.append(Cuboid(
                        name=name,
                        pose=pose,
                        dims=[float(value) for value in obstacle["dims_m"]],
                    ))
            worlds.append(WorldConfig(cuboid=cuboids, mesh=meshes))
        return worlds

    def _motion_generator(self, world: WorldConfig) -> MotionGen:
        obstacle_capacity = len(world.cuboid)
        mesh_capacity = len(world.mesh)
        key = (obstacle_capacity, mesh_capacity)
        motion_gen = self.motion_generators.get(key)
        if motion_gen is not None:
            motion_gen.update_world(world)
            return motion_gen
        config = MotionGenConfig.load_from_robot_config(
            self.robot_config,
            world,
            tensor_args=self.tensor_args,
            num_ik_seeds=self.num_seeds,
            num_graph_seeds=self.motion_num_graph_seeds,
            num_trajopt_seeds=self.motion_num_trajopt_seeds,
            self_collision_check=self.self_collision,
            self_collision_opt=self.self_collision,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": max(1, obstacle_capacity),
                "mesh": max(1, mesh_capacity),
            },
            use_cuda_graph=self.use_cuda_graph,
            interpolation_dt=0.02,
        )
        motion_gen = MotionGen(config)
        # Warm both graph and joint-space trajectory optimization once. Later
        # requests reuse the CUDA allocations and only replace the collision
        # world, which is the important latency property for a placement matrix.
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=True)
        self.motion_generators[key] = motion_gen
        return motion_gen

    def _robot_config_for_finger_opening(
        self, finger_opening_m: float | None
    ) -> tuple[int, RobotConfig]:
        """Return a Panda sphere model at the requested total finger opening."""
        if finger_opening_m is None:
            return 5000, self.robot_config
        opening = min(0.08, max(0.0, float(finger_opening_m)))
        finger_joint_value = 0.5 * opening
        key = int(round(finger_joint_value * 1_000_000.0))
        cached = self.robot_configs_by_finger_joint_value.get(key)
        if cached is not None:
            return key, cached
        robot_dict = copy.deepcopy(self.robot_dict_template)
        robot_dict["kinematics"]["lock_joints"] = {
            "panda_finger_joint1": finger_joint_value
        }
        config = RobotConfig.from_dict(robot_dict, self.tensor_args)
        self.robot_configs_by_finger_joint_value[key] = config
        return key, config

    def _solver(
        self,
        batch_size: int,
        worlds: list[WorldConfig],
        finger_opening_m: float | None = None,
    ) -> IKSolver:
        obstacle_capacity = max((len(world.cuboid) for world in worlds), default=0)
        mesh_capacity = max((len(world.mesh) for world in worlds), default=0)
        finger_key, robot_config = self._robot_config_for_finger_opening(
            finger_opening_m
        )
        key = (finger_key, batch_size, obstacle_capacity, mesh_capacity)
        solver = self.solvers.get(key)
        if solver is not None:
            solver.world_coll_checker.load_batch_collision_model(worlds)
            return solver
        config = IKSolverConfig.load_from_robot_config(
            robot_config,
            worlds,
            tensor_args=self.tensor_args,
            num_seeds=self.num_seeds,
            position_threshold=0.004,
            rotation_threshold=math.radians(2.0),
            self_collision_check=self.self_collision,
            self_collision_opt=self.self_collision,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": max(1, obstacle_capacity),
                "mesh": max(1, mesh_capacity),
            },
            n_collision_envs=batch_size,
            use_cuda_graph=self.use_cuda_graph,
            seed=0,
        )
        solver = IKSolver(config)
        self.solvers[key] = solver
        return solver

    def _environment_clearance_m(
        self,
        solver: IKSolver,
        path_full: torch.Tensor,
        worlds: list[WorldConfig],
        environment_indices: torch.Tensor | None = None,
    ) -> tuple[float | None, list[float]]:
        """Measure selected-path clearance for every environment on the GPU.

        cuRobo ESDF is positive inside an obstacle and negative outside. The
        negated maximum sphere ESDF is therefore the minimum robot/world
        clearance; a negative result means penetration.
        """
        if not any(len(world.cuboid) + len(world.mesh) for world in worlds):
            return None, []
        with torch.no_grad():
            spheres = solver.fk(path_full).link_spheres_tensor.unsqueeze(1).contiguous()
            query_buffer = CollisionQueryBuffer.initialize_from_shape(
                spheres.shape,
                self.tensor_args,
                solver.world_coll_checker.collision_types,
            )
            env_query_idx = (
                torch.arange(
                    spheres.shape[0],
                    device=self.tensor_args.device,
                    dtype=torch.int32,
                )
                if environment_indices is None
                else environment_indices.to(
                    device=self.tensor_args.device, dtype=torch.int32
                )
            )
            esdf = solver.world_coll_checker.get_sphere_distance(
                spheres,
                query_buffer,
                weight=self.tensor_args.to_device([1.0]),
                activation_distance=self.tensor_args.to_device(
                    [float(solver.world_coll_checker.max_distance)]
                ),
                env_query_idx=env_query_idx,
                return_loss=False,
                sum_collisions=False,
                compute_esdf=True,
            )
            per_sample = -torch.amax(esdf.reshape(spheres.shape[0], -1), dim=1)
        values = per_sample.detach().cpu().to(dtype=torch.float64).tolist()
        return float(min(values)), [float(value) for value in values]

    def solve_path(self, request: dict[str, Any]) -> dict[str, Any]:
        positions = np.asarray(request["positions_world"], dtype=np.float64)
        rotations = np.asarray(request["quaternions_xyzw_world"], dtype=np.float64)
        reference = np.asarray(request["reference"], dtype=np.float64)
        local_positions, local_wxyz = _local_poses(
            positions,
            rotations,
            np.asarray(request["robot_base_position_world"], dtype=np.float64),
            np.asarray(request["robot_base_quaternion_xyzw_world"], dtype=np.float64),
        )
        goal = Pose(
            position=self.tensor_args.to_device(local_positions),
            quaternion=self.tensor_args.to_device(local_wxyz),
        )
        worlds = self._worlds(request)
        if bool(request.get("sequential", False)):
            return self._solve_path_sequential(
                goal, worlds, reference, request
            )
        solver = self._solver(len(positions), worlds)
        retract = self.tensor_args.to_device(np.repeat(reference[None, :], len(positions), axis=0))
        # Use the same seed bank for every waypoint. Independent random banks
        # make identical neighboring poses converge to unrelated redundant-arm
        # branches, leaving no 0.08-rad edge even though a continuous path exists.
        seed_bank = solver.sample_configs(self.num_seeds)
        seed_bank[0] = self.tensor_args.to_device(reference)
        seed_config = seed_bank.unsqueeze(0).repeat(len(positions), 1, 1).contiguous()
        result = solver.solve_batch_env(
            goal,
            retract_config=retract,
            seed_config=seed_config,
            return_seeds=min(self.return_seeds, self.num_seeds),
        )
        q = result.solution.detach().cpu().numpy()[:, :, self.output_indices]
        valid = result.success.detach().cpu().numpy().astype(bool)
        selected, failed_sample = _select_continuous_branch(
            q,
            valid,
            reference,
            request.get("maximum_joint_step_rad"),
            bool(request.get("enforce_start_step", False)),
        )
        response: dict[str, Any] = {
            "success": selected is not None,
            "failed_sample": failed_sample,
            "valid_candidates_per_sample": valid.sum(axis=1).astype(int).tolist(),
            "solve_time_s": float(result.solve_time),
            "backend": "curobo_batch_ik_continuous_dp",
            "gpu_self_collision_checked": self.self_collision,
            "gpu_environment_collision_checked": any(
                len(world.cuboid) + len(world.mesh) for world in worlds
            ),
            "gpu_collision_obstacles_per_sample": [
                len(world.cuboid) + len(world.mesh) for world in worlds
            ],
        }
        if selected is not None:
            selected_rows = torch.as_tensor(
                selected, device=result.solution.device, dtype=torch.long
            )
            sample_rows = torch.arange(
                len(selected), device=result.solution.device, dtype=torch.long
            )
            path_full = result.solution[sample_rows, selected_rows].detach()
            path = path_full[:, self.output_indices].cpu().numpy()
            minimum_clearance, clearance_by_sample = self._environment_clearance_m(
                solver, path_full, worlds
            )
            response["path"] = path.tolist()
            response["selected_seed_indices"] = selected
            response["minimum_environment_clearance_m"] = minimum_clearance
            response["environment_clearance_by_sample_m"] = clearance_by_sample
            response["maximum_position_error_m"] = float(
                np.max(result.position_error.detach().cpu().numpy()[
                    np.arange(len(selected)), selected
                ])
            )
            response["maximum_orientation_error_rad"] = float(
                np.max(result.rotation_error.detach().cpu().numpy()[
                    np.arange(len(selected)), selected
                ])
            )
            if len(path) > 1:
                response["maximum_adjacent_joint_step_rad"] = float(
                    np.max(np.abs(np.diff(path, axis=0)))
                )
            else:
                response["maximum_adjacent_joint_step_rad"] = 0.0
        return response

    def solve_paths_batch(self, request: dict[str, Any]) -> dict[str, Any]:
        """Solve independent base/path candidates in one cuRobo GPU call.

        Each pose sample remains its own collision environment.  Flattening
        ``candidate x sample`` therefore preserves the exact per-base source
        mesh world while allowing cuRobo to solve all sparse candidates as one
        tensor batch.  Continuity selection is still performed independently
        for each candidate after the GPU result returns.
        """
        paths = request.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("solve_paths_batch requires a non-empty paths list")

        flat_positions: list[np.ndarray] = []
        flat_rotations_wxyz: list[np.ndarray] = []
        flat_references: list[np.ndarray] = []
        worlds: list[WorldConfig] = []
        spans: list[tuple[int, int]] = []
        references: list[np.ndarray] = []
        cursor = 0
        for path_index, row in enumerate(paths):
            positions = np.asarray(row["positions_world"], dtype=np.float64)
            rotations = np.asarray(
                row["quaternions_xyzw_world"], dtype=np.float64
            )
            reference = np.asarray(row["reference"], dtype=np.float64)
            if (
                positions.ndim != 2
                or positions.shape[1] != 3
                or rotations.shape != (len(positions), 4)
                or not len(positions)
            ):
                raise ValueError(
                    f"Batch path {path_index} has invalid pose array shapes"
                )
            local_positions, local_wxyz = _local_poses(
                positions,
                rotations,
                np.asarray(row["robot_base_position_world"], dtype=np.float64),
                np.asarray(
                    row["robot_base_quaternion_xyzw_world"], dtype=np.float64
                ),
            )
            flat_positions.extend(local_positions)
            flat_rotations_wxyz.extend(local_wxyz)
            flat_references.extend(
                np.repeat(reference[None, :], len(positions), axis=0)
            )
            path_worlds = self._worlds(row)
            worlds.extend(path_worlds)
            spans.append((cursor, cursor + len(positions)))
            references.append(reference)
            cursor += len(positions)

        goal = Pose(
            position=self.tensor_args.to_device(np.asarray(flat_positions)),
            quaternion=self.tensor_args.to_device(
                np.asarray(flat_rotations_wxyz)
            ),
        )
        solver = self._solver(cursor, worlds)
        retract = self.tensor_args.to_device(np.asarray(flat_references))
        seed_bank = solver.sample_configs(self.num_seeds)
        seed_config = seed_bank.unsqueeze(0).repeat(cursor, 1, 1).contiguous()
        seed_config[:, 0, :] = retract
        result = solver.solve_batch_env(
            goal,
            retract_config=retract,
            seed_config=seed_config,
            return_seeds=min(self.return_seeds, self.num_seeds),
        )
        q = result.solution.detach().cpu().numpy()[:, :, self.output_indices]
        valid = result.success.detach().cpu().numpy().astype(bool)
        position_error = result.position_error.detach().cpu().numpy()
        rotation_error = result.rotation_error.detach().cpu().numpy()
        responses: list[dict[str, Any]] = []
        for row, reference, (start, stop) in zip(paths, references, spans):
            selected, failed_sample = _select_continuous_branch(
                q[start:stop],
                valid[start:stop],
                reference,
                row.get("maximum_joint_step_rad"),
                bool(row.get("enforce_start_step", False)),
            )
            path_worlds = worlds[start:stop]
            response: dict[str, Any] = {
                "success": selected is not None,
                "failed_sample": failed_sample,
                "valid_candidates_per_sample": valid[start:stop]
                .sum(axis=1)
                .astype(int)
                .tolist(),
                "solve_time_s": float(result.solve_time),
                "batch_size": len(paths),
                "batch_pose_environments": cursor,
                "backend": "curobo_multi_base_batch_ik_continuous_dp",
                "gpu_self_collision_checked": self.self_collision,
                "gpu_environment_collision_checked": any(
                    len(world.cuboid) + len(world.mesh) for world in path_worlds
                ),
                "gpu_collision_obstacles_per_sample": [
                    len(world.cuboid) + len(world.mesh) for world in path_worlds
                ],
            }
            if selected is not None:
                selected_rows = torch.as_tensor(
                    selected, device=result.solution.device, dtype=torch.long
                )
                sample_rows = torch.arange(
                    start, stop, device=result.solution.device, dtype=torch.long
                )
                path_full = result.solution[sample_rows, selected_rows].detach()
                path = path_full[:, self.output_indices].cpu().numpy()
                minimum_clearance, clearance_by_sample = (
                    self._environment_clearance_m(
                        solver,
                        path_full,
                        path_worlds,
                        environment_indices=sample_rows,
                    )
                )
                local_rows = np.arange(stop - start)
                response["path"] = path.tolist()
                response["selected_seed_indices"] = selected
                response["minimum_environment_clearance_m"] = minimum_clearance
                response["environment_clearance_by_sample_m"] = clearance_by_sample
                response["maximum_position_error_m"] = float(
                    np.max(position_error[start:stop][local_rows, selected])
                )
                response["maximum_orientation_error_rad"] = float(
                    np.max(rotation_error[start:stop][local_rows, selected])
                )
                response["maximum_adjacent_joint_step_rad"] = (
                    0.0
                    if len(path) < 2
                    else float(np.max(np.abs(np.diff(path, axis=0))))
                )
            responses.append(response)
        return {
            "results": responses,
            "batch_size": len(paths),
            "batch_pose_environments": cursor,
            "solve_time_s": float(result.solve_time),
            "backend": "curobo_multi_base_batch_ik",
        }

    def _solve_path_sequential(
        self,
        goal: Pose,
        worlds: list[WorldConfig],
        reference: np.ndarray,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Track a dense pose path while propagating the previous IK branch.

        A single batch solves every pose independently. On a redundant arm that
        can discard the locally continuous branch even when all poses have many
        valid solutions. Dense execution instead seeds each sample from the
        selected predecessor and enforces the same joint trust region online.
        """
        maximum_step = request.get("maximum_joint_step_rad")
        enforce_start = bool(request.get("enforce_start_step", False))
        current = np.asarray(reference, dtype=np.float64)
        selected_path: list[np.ndarray] = []
        selected_full: list[torch.Tensor] = []
        selected_indices: list[int] = []
        valid_counts: list[int] = []
        position_errors: list[float] = []
        rotation_errors: list[float] = []
        solve_time = 0.0

        for sample_index, world in enumerate(worlds):
            solver = self._solver(1, [world])
            sample_goal = Pose(
                position=goal.position[sample_index : sample_index + 1],
                quaternion=goal.quaternion[sample_index : sample_index + 1],
            )
            retract = self.tensor_args.to_device(current[None, :])
            seed_bank = solver.sample_configs(self.num_seeds)
            seed_bank[0] = self.tensor_args.to_device(current)
            result = solver.solve_batch_env(
                sample_goal,
                retract_config=retract,
                seed_config=seed_bank.unsqueeze(0).contiguous(),
                return_seeds=min(self.return_seeds, self.num_seeds),
            )
            solve_time += float(result.solve_time)
            candidates = result.solution.detach().cpu().numpy()[0][
                :, self.output_indices
            ]
            valid = result.success.detach().cpu().numpy()[0].astype(bool)
            valid_counts.append(int(valid.sum()))
            deltas = np.max(np.abs(candidates - current[None, :]), axis=1)
            allowed = valid.copy()
            if maximum_step is not None and (sample_index > 0 or enforce_start):
                allowed &= deltas <= float(maximum_step)
            indices = np.flatnonzero(allowed)
            if not len(indices):
                return {
                    "success": False,
                    "failed_sample": int(sample_index),
                    "valid_candidates_per_sample": valid_counts,
                    "solve_time_s": solve_time,
                    "backend": "curobo_sequential_seeded_ik",
                    "gpu_self_collision_checked": self.self_collision,
                    "gpu_environment_collision_checked": any(
                        len(row.cuboid) + len(row.mesh) for row in worlds
                    ),
                    "gpu_collision_obstacles_per_sample": [
                        len(row.cuboid) + len(row.mesh) for row in worlds
                    ],
                }
            # Prefer the smallest infinity-norm step, then L2, then seed index.
            best = min(
                (int(index) for index in indices),
                key=lambda index: (
                    float(deltas[index]),
                    float(np.linalg.norm(candidates[index] - current)),
                    index,
                ),
            )
            current = np.asarray(candidates[best], dtype=np.float64)
            selected_path.append(current.copy())
            selected_full.append(result.solution[0, best].detach())
            selected_indices.append(best)
            position_errors.append(float(result.position_error[0, best].item()))
            rotation_errors.append(float(result.rotation_error[0, best].item()))

        path_full = torch.stack(selected_full, dim=0)
        clearance_solver = self._solver(len(worlds), worlds)
        minimum_clearance, clearance_by_sample = self._environment_clearance_m(
            clearance_solver, path_full, worlds
        )
        path = np.asarray(selected_path, dtype=np.float64)
        return {
            "success": True,
            "failed_sample": None,
            "valid_candidates_per_sample": valid_counts,
            "solve_time_s": solve_time,
            "backend": "curobo_sequential_seeded_ik",
            "gpu_self_collision_checked": self.self_collision,
            "gpu_environment_collision_checked": any(
                len(row.cuboid) + len(row.mesh) for row in worlds
            ),
            "gpu_collision_obstacles_per_sample": [
                len(row.cuboid) + len(row.mesh) for row in worlds
            ],
            "path": path.tolist(),
            "selected_seed_indices": selected_indices,
            "minimum_environment_clearance_m": minimum_clearance,
            "environment_clearance_by_sample_m": clearance_by_sample,
            "maximum_position_error_m": max(position_errors, default=0.0),
            "maximum_orientation_error_rad": max(rotation_errors, default=0.0),
            "maximum_adjacent_joint_step_rad": (
                0.0
                if len(path) < 2
                else float(np.max(np.abs(np.diff(path, axis=0))))
            ),
        }

    def plan_joint_path(self, request: dict[str, Any]) -> dict[str, Any]:
        start = np.asarray(request["start"], dtype=np.float32)
        goal = np.asarray(request["goal"], dtype=np.float32)
        if start.shape != (len(self.arm_joint_names),) or goal.shape != start.shape:
            raise ValueError("Joint-path start/goal must match the configured arm joints")
        world = self._worlds(request, count=1)[0]
        motion_gen = self._motion_generator(world)
        start_state = JointState.from_position(
            self.tensor_args.to_device(start[None, :]),
            joint_names=self.arm_joint_names,
        )
        goal_state = JointState.from_position(
            self.tensor_args.to_device(goal[None, :]),
            joint_names=self.arm_joint_names,
        )
        result = motion_gen.plan_single_js(
            start_state,
            goal_state,
            MotionGenPlanConfig(
                enable_graph=True,
                enable_opt=True,
                max_attempts=self.motion_max_attempts,
                timeout=self.motion_timeout_s,
                check_start_validity=True,
                num_graph_seeds=self.motion_num_graph_seeds,
                num_trajopt_seeds=self.motion_num_trajopt_seeds,
            ),
        )
        success = bool(result.success.item())
        response: dict[str, Any] = {
            "success": success,
            "backend": "curobo_motion_gen_gpu",
            "status": None if result.status is None else str(result.status),
            "solve_time_s": float(result.solve_time),
            "total_time_s": float(result.total_time),
            "graph_time_s": float(result.graph_time),
            "trajopt_time_s": float(result.trajopt_time),
            "finetune_time_s": float(result.finetune_time),
            "attempts": int(result.attempts),
            "used_graph": bool(result.used_graph),
            "gpu_self_collision_checked": self.self_collision,
            "gpu_environment_collision_checked": bool(
                len(world.cuboid) + len(world.mesh)
            ),
            "gpu_collision_obstacles": len(world.cuboid) + len(world.mesh),
            "required_clearance_m": float(request.get("required_clearance_m", 0.0)),
        }
        if not success:
            return response
        trajectory = result.get_interpolated_plan()
        joint_names = list(trajectory.joint_names or self.arm_joint_names)
        missing = set(self.arm_joint_names) - set(joint_names)
        if missing:
            raise ValueError(
                f"MotionGen trajectory is missing configured arm joints {sorted(missing)}"
            )
        indices = [joint_names.index(name) for name in self.arm_joint_names]
        path = trajectory.position.detach().cpu().numpy()
        if path.ndim == 3 and path.shape[0] == 1:
            path = path[0]
        path = np.asarray(path[:, indices], dtype=np.float64)
        # MotionGen already interpolates on the GPU, but enforce the execution
        # contract even if a future cuRobo configuration chooses a coarser dt.
        maximum_step = float(request.get("maximum_joint_step_rad", 0.08))
        refined: list[np.ndarray] = [start.astype(np.float64)]
        for target in path:
            delta = float(np.max(np.abs(target - refined[-1])))
            subdivisions = max(1, int(math.ceil(delta / maximum_step)))
            left = refined[-1].copy()
            for alpha in np.linspace(0.0, 1.0, subdivisions + 1)[1:]:
                refined.append((1.0 - alpha) * left + alpha * target)
        if float(np.max(np.abs(refined[-1] - goal))) > 1.0e-5:
            delta = float(np.max(np.abs(goal - refined[-1])))
            subdivisions = max(1, int(math.ceil(delta / maximum_step)))
            left = refined[-1].copy()
            for alpha in np.linspace(0.0, 1.0, subdivisions + 1)[1:]:
                refined.append((1.0 - alpha) * left + alpha * goal)
        output = np.asarray(refined, dtype=np.float64)
        response["path"] = output.tolist()
        response["path_samples"] = int(len(output))
        response["maximum_adjacent_joint_step_rad"] = (
            0.0
            if len(output) < 2
            else float(np.max(np.abs(np.diff(output, axis=0))))
        )
        return response

    def check_joint_path(self, request: dict[str, Any]) -> dict[str, Any]:
        """Measure source-mesh clearance for an existing joint path on GPU."""
        path = np.asarray(request["joint_path"], dtype=np.float32)
        expected = len(self.arm_joint_names)
        if path.ndim != 2 or path.shape[1] != expected or len(path) == 0:
            raise ValueError(
                "joint_path must be a non-empty matrix matching configured arm joints"
            )
        worlds = self._worlds(request, count=len(path))
        solver = self._solver(
            len(path), worlds, request.get("finger_opening_m")
        )
        path_full = self.tensor_args.to_device(path)
        minimum, by_sample = self._environment_clearance_m(
            solver, path_full, worlds
        )
        required = float(request.get("required_clearance_m", 0.0))
        return {
            "success": minimum is None or minimum >= required,
            "backend": "curobo_gpu_source_mesh_path_check",
            "gpu_self_collision_checked": False,
            "gpu_environment_collision_checked": any(
                len(world.cuboid) + len(world.mesh) for world in worlds
            ),
            "gpu_collision_obstacles_per_sample": [
                len(world.cuboid) + len(world.mesh) for world in worlds
            ],
            "minimum_environment_clearance_m": minimum,
            "environment_clearance_by_sample_m": by_sample,
            "required_clearance_m": required,
            "finger_opening_m": request.get("finger_opening_m"),
            "failed_sample": (
                None
                if minimum is None or minimum >= required
                else int(np.argmin(np.asarray(by_sample, dtype=np.float64)))
            ),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-urdf", required=True)
    parser.add_argument("--base-link", required=True)
    parser.add_argument("--end-effector-link", required=True)
    parser.add_argument("--arm-joint-names", nargs="+", required=True)
    parser.add_argument("--num-seeds", type=int, default=32)
    parser.add_argument("--return-seeds", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collision-sphere-buffer-m", type=float, default=0.0)
    parser.add_argument("--motion-num-graph-seeds", type=int, default=4)
    parser.add_argument("--motion-num-trajopt-seeds", type=int, default=4)
    parser.add_argument("--motion-timeout-s", type=float, default=10.0)
    parser.add_argument("--motion-max-attempts", type=int, default=6)
    parser.add_argument("--disable-self-collision", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    worker = Worker(parser.parse_args())
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                break
            if request.get("command") == "solve_path":
                response = worker.solve_path(request)
            elif request.get("command") == "solve_paths_batch":
                response = worker.solve_paths_batch(request)
            elif request.get("command") == "plan_joint_path":
                response = worker.plan_joint_path(request)
            elif request.get("command") == "check_joint_path":
                response = worker.check_joint_path(request)
            else:
                raise ValueError(f"Unknown cuRobo worker command {request.get('command')!r}")
            response["id"] = request.get("id")
        except Exception as exc:  # protocol errors must reach the parent process
            response = {
                "id": None,
                "success": False,
                "worker_error": repr(exc),
                "worker_traceback": traceback.format_exc(),
            }
        print(RESPONSE_PREFIX + json.dumps(response, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
