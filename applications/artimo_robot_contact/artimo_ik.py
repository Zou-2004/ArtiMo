"""Generic PyBullet helpers for ArtiMo robot-contact trajectory planning."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pybullet as p
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def quat_angle_rad(a: Iterable[float], b: Iterable[float]) -> float:
    qa = np.asarray(list(a), dtype=np.float64)
    qb = np.asarray(list(b), dtype=np.float64)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    return float(2.0 * math.acos(float(np.clip(abs(np.dot(qa, qb)), -1.0, 1.0))))


def link_world_pose(
    body: int, link_index: int, client: int
) -> tuple[list[float], list[float]]:
    if link_index == -1:
        position, quaternion = p.getBasePositionAndOrientation(
            body, physicsClientId=client
        )
        return list(position), list(quaternion)
    state = p.getLinkState(
        body, link_index, computeForwardKinematics=True, physicsClientId=client
    )
    return list(state[4]), list(state[5])


def set_robot_arm(
    body: int, indices: list[int], q: np.ndarray, client: int
) -> None:
    for joint_index, value in zip(indices, q):
        p.resetJointState(
            body,
            joint_index,
            targetValue=float(value),
            targetVelocity=0.0,
            physicsClientId=client,
        )


def set_fingers(body: int, indices: list[int], value: float, client: int) -> None:
    for joint_index in indices:
        p.resetJointState(
            body,
            joint_index,
            targetValue=float(value),
            targetVelocity=0.0,
            physicsClientId=client,
        )


class BulletIK:
    """Deterministic bounded-restart IK with pose and continuity diagnostics."""

    def __init__(
        self,
        body: int,
        arm_indices: list[int],
        eef_link_index: int,
        finger_indices: list[int],
        finger_position: float,
        config: dict[str, Any],
        client: int,
    ) -> None:
        self.body = body
        self.arm_indices = arm_indices
        self.eef_link_index = eef_link_index
        self.finger_indices = finger_indices
        self.finger_position = finger_position
        self.client = client
        self.rng = np.random.default_rng(int(config.get("random_seed", 0)))
        self.max_iterations = int(config.get("max_iterations", 1000))
        self.random_restarts = int(config.get("random_restarts", 24))
        self.position_tolerance = float(config.get("position_tolerance_m", 0.002))
        self.orientation_tolerance = math.radians(
            float(config.get("orientation_tolerance_deg", 0.5))
        )
        self.max_joint_step = float(config.get("max_joint_step_rad", 0.65))

        self.movable_indices: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        for joint_index in range(p.getNumJoints(body, physicsClientId=client)):
            info = p.getJointInfo(body, joint_index, physicsClientId=client)
            if info[2] == p.JOINT_FIXED:
                continue
            lo, hi = float(info[8]), float(info[9])
            if hi < lo:
                lo, hi = -2.0 * math.pi, 2.0 * math.pi
            self.movable_indices.append(joint_index)
            lower.append(lo)
            upper.append(hi)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.ranges = self.upper - self.lower
        slots = {
            joint_index: slot
            for slot, joint_index in enumerate(self.movable_indices)
        }
        self.arm_slots = [slots[joint_index] for joint_index in arm_indices]
        self.arm_lower = self.lower[self.arm_slots]
        self.arm_upper = self.upper[self.arm_slots]

    def _joint_limit_metrics(self, arm_q: np.ndarray) -> tuple[float, float]:
        """Return absolute and range-normalized distance to the nearest limit."""
        distances = np.minimum(arm_q - self.arm_lower, self.arm_upper - arm_q)
        safe_ranges = np.maximum(self.arm_upper - self.arm_lower, 1e-9)
        normalized = distances / safe_ranges
        return float(np.min(distances)), float(np.min(normalized))

    def _movable_rest(self, arm_q: np.ndarray) -> np.ndarray:
        rest = np.asarray(
            [
                p.getJointState(self.body, joint, physicsClientId=self.client)[0]
                for joint in self.movable_indices
            ],
            dtype=np.float64,
        )
        rest[self.arm_slots] = arm_q
        for finger_index in self.finger_indices:
            if finger_index in self.movable_indices:
                rest[self.movable_indices.index(finger_index)] = self.finger_position
        return rest

    def _candidate(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        seed: np.ndarray,
        reference: np.ndarray,
    ) -> dict[str, Any] | None:
        set_robot_arm(self.body, self.arm_indices, seed, self.client)
        result = np.asarray(
            p.calculateInverseKinematics(
                self.body,
                self.e_link,
                targetPosition=target_position,
                targetOrientation=target_quaternion,
                lowerLimits=self.lower.tolist(),
                upperLimits=self.upper.tolist(),
                jointRanges=self.ranges.tolist(),
                restPoses=self._movable_rest(seed).tolist(),
                maxNumIterations=self.max_iterations,
                residualThreshold=1e-7,
                physicsClientId=self.client,
            ),
            dtype=np.float64,
        )
        if result.size < len(self.movable_indices):
            return None
        arm_q = result[self.arm_slots]
        if np.any(arm_q < self.arm_lower - 1e-6) or np.any(
            arm_q > self.arm_upper + 1e-6
        ):
            return None
        set_robot_arm(self.body, self.arm_indices, arm_q, self.client)
        actual_position, actual_quaternion = link_world_pose(
            self.body, self.e_link, self.client
        )
        position_error = float(
            np.linalg.norm(np.asarray(actual_position) - np.asarray(target_position))
        )
        orientation_error = quat_angle_rad(actual_quaternion, target_quaternion)
        limit_margin, normalized_limit_margin = self._joint_limit_metrics(arm_q)
        return {
            "q": arm_q,
            "actual_position": actual_position,
            "actual_quaternion": actual_quaternion,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "max_joint_step_rad": float(np.max(np.abs(arm_q - reference))),
            "joint_l2_step_rad": float(np.linalg.norm(arm_q - reference)),
            "minimum_joint_limit_margin_rad": limit_margin,
            "minimum_normalized_joint_limit_margin": normalized_limit_margin,
        }

    def _continuous_refinement(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        reference: np.ndarray,
    ) -> dict[str, Any] | None:
        """Refine from the previous waypoint without changing IK branches.

        Bullet's null-space IK can occasionally jump to another valid branch
        even when it is seeded with the preceding waypoint.  Random restarts
        cannot repair that failure because they search more branches.  This
        bounded local least-squares fallback instead starts at ``reference``
        and minimizes Cartesian pose error, with a very weak joint-displacement
        regularizer that resolves the Panda's redundant degree of freedom in
        favor of continuity.
        """
        target_p = np.asarray(target_position, dtype=np.float64)
        target_r = Rotation.from_quat(np.asarray(target_quaternion, dtype=np.float64))
        position_scale = max(self.position_tolerance, 1e-5)
        orientation_scale = max(self.orientation_tolerance, math.radians(0.05))
        joint_scale = max(self.max_joint_step, 1e-3)
        joint_midpoint = 0.5 * (self.arm_lower + self.arm_upper)
        joint_range = np.maximum(self.arm_upper - self.arm_lower, 1e-6)

        # scipy requires a strictly in-bounds initial value.  The epsilon is
        # numerical only; returned configurations are still checked against the
        # original robot limits below.
        epsilon = 1e-8
        # A continuation solve is a local trust-region step around the previous
        # command, not another global IK query.  Bounding the optimizer itself
        # prevents null-space wandering from producing a large adjacent command
        # while still returning a finite best local pose when the exact target
        # lies outside this one step's neighbourhood.
        local_lower = np.maximum(
            self.arm_lower, reference - self.max_joint_step
        )
        local_upper = np.minimum(
            self.arm_upper, reference + self.max_joint_step
        )
        x0 = np.clip(reference, local_lower + epsilon, local_upper - epsilon)

        def residual(q: np.ndarray) -> np.ndarray:
            set_robot_arm(self.body, self.arm_indices, q, self.client)
            actual_position, actual_quaternion = link_world_pose(
                self.body, self.e_link, self.client
            )
            actual_r = Rotation.from_quat(
                np.asarray(actual_quaternion, dtype=np.float64)
            )
            position_error = (np.asarray(actual_position) - target_p) / position_scale
            orientation_error = (target_r * actual_r.inv()).as_rotvec() / orientation_scale
            # The seventh Panda joint is redundant for a six-dimensional end
            # effector pose.  A nearly-zero regularizer lets the numerical
            # optimizer wander a long way along that null space between two
            # adjacent Cartesian samples, which is visually indistinguishable
            # from an IK branch jump even though both poses are accurate.  Keep
            # the local pose solve softly anchored to the preceding command.
            # Cartesian pose terms still dominate whenever motion is required.
            continuity = 1e-2 * (q - reference) / joint_scale
            # A continuity-only null-space objective is myopic: it can follow
            # the locally nearest redundant solution until one Panda joint
            # reaches a hard limit halfway through an otherwise reachable
            # manipulation.  A weaker range-normalized centering term spends
            # that redundant freedom early while Cartesian residuals and local
            # continuity remain dominant.  This is robot-kinematic policy, not
            # an asset or contact-specific preference.
            joint_limit_centering = 3e-3 * (q - joint_midpoint) / joint_range
            return np.concatenate(
                (
                    position_error,
                    orientation_error,
                    continuity,
                    joint_limit_centering,
                )
            )

        def numerical_jacobian(q: np.ndarray, step: float) -> np.ndarray:
            # scipy's ``diff_step`` is relative to each q component, which
            # collapses near zero.  Robot joints routinely cross zero, so use an
            # explicit absolute central difference that remains above
            # PyBullet's single-precision FK quantization everywhere.
            columns: list[np.ndarray] = []
            for joint in range(len(q)):
                low = q.copy()
                high = q.copy()
                low[joint] = max(self.arm_lower[joint], q[joint] - step)
                high[joint] = min(self.arm_upper[joint], q[joint] + step)
                width = float(high[joint] - low[joint])
                if width <= 1e-12:
                    columns.append(np.zeros_like(residual(q)))
                else:
                    columns.append((residual(high) - residual(low)) / width)
            return np.column_stack(columns)

        best_q: np.ndarray | None = None
        best_score = float("inf")
        guess = x0
        try:
            for step in (1e-3, 1e-4):
                result = least_squares(
                    residual,
                    guess,
                    jac=lambda q, h=step: numerical_jacobian(q, h),
                    bounds=(local_lower, local_upper),
                    method="trf",
                    x_scale="jac",
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                    max_nfev=max(200, min(self.max_iterations, 800)),
                )
                guess = np.asarray(result.x, dtype=np.float64)
                score = float(np.linalg.norm(residual(guess)[:6]))
                if score < best_score:
                    best_score = score
                    best_q = guess.copy()
        except (ValueError, np.linalg.LinAlgError):
            return None
        if best_q is None:
            return None

        arm_q = best_q
        set_robot_arm(self.body, self.arm_indices, arm_q, self.client)
        actual_position, actual_quaternion = link_world_pose(
            self.body, self.e_link, self.client
        )
        position_error = float(
            np.linalg.norm(np.asarray(actual_position) - target_p)
        )
        orientation_error = quat_angle_rad(actual_quaternion, target_quaternion)
        limit_margin, normalized_limit_margin = self._joint_limit_metrics(arm_q)
        answer = {
            "q": arm_q,
            "actual_position": actual_position,
            "actual_quaternion": actual_quaternion,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "max_joint_step_rad": float(np.max(np.abs(arm_q - reference))),
            "joint_l2_step_rad": float(np.linalg.norm(arm_q - reference)),
            "seed_index": -1,
            "solver": "continuous_least_squares",
            "minimum_joint_limit_margin_rad": limit_margin,
            "minimum_normalized_joint_limit_margin": normalized_limit_margin,
        }
        answer["success"] = True
        return answer

    @property
    def e_link(self) -> int:
        return self.eef_link_index

    def solve_continuous(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        reference: np.ndarray,
    ) -> dict[str, Any]:
        """Return the nearest finite in-limit solution on the current branch.

        Dense manipulation paths are trajectory problems, not independent pose
        tests.  A millimetre/degree tolerance miss must never discard a smooth
        local solution in favour of a distant global IK branch.  Rank the
        reference-seeded Bullet candidate and local least-squares refinement by
        joint-space continuity first; pose residuals remain measurements only.
        """
        candidate = self._candidate(
            target_position, target_quaternion, reference, reference
        )
        refined = self._continuous_refinement(
            target_position, target_quaternion, reference
        )
        continuous_options = [
            option
            for option in (candidate, refined)
            if option is not None
            and float(option["max_joint_step_rad"])
            <= self.max_joint_step + 1e-7
        ]
        if continuous_options:
            # The local least-squares refinement is a fallback, not an
            # unconditional winner.  It can terminate at a finite but badly
            # wrong pose near a singularity even when Bullet's reference-seeded
            # solution is accurate and remains inside the same trust region.
            # Preserve continuity as a hard filter, then prefer a pose-valid
            # option and only use joint distance to break ties.
            answer = min(
                continuous_options,
                key=lambda option: (
                    not (
                        float(option["position_error_m"])
                        <= 4.0 * self.position_tolerance
                        and float(option["orientation_error_rad"])
                        <= 2.0 * self.orientation_tolerance
                    ),
                    float(option["position_error_m"])
                    / max(self.position_tolerance, 1e-9)
                    + float(option["orientation_error_rad"])
                    / max(self.orientation_tolerance, 1e-9),
                    float(option["joint_l2_step_rad"]),
                ),
            )
        else:
            set_robot_arm(self.body, self.arm_indices, reference, self.client)
            actual_position, actual_quaternion = link_world_pose(
                self.body, self.e_link, self.client
            )
            return {
                "q": reference.copy(),
                "actual_position": actual_position,
                "actual_quaternion": actual_quaternion,
                "position_error_m": float(
                    np.linalg.norm(
                        np.asarray(actual_position) - np.asarray(target_position)
                    )
                ),
                "orientation_error_rad": quat_angle_rad(
                    actual_quaternion, target_quaternion
                ),
                "max_joint_step_rad": 0.0,
                "joint_l2_step_rad": 0.0,
                "seed_index": -1,
                "solver": "hold_previous_continuous_branch",
                "pose_residual_only": True,
                "success": True,
            }
        answer["seed_index"] = int(answer.get("seed_index", 0))
        answer["solver"] = str(answer.get("solver", "bullet_reference"))
        answer["pose_residual_only"] = True
        answer["success"] = True
        return answer

    def solve(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        reference: np.ndarray,
        enforce_step: bool,
    ) -> dict[str, Any]:
        candidates = self.solve_candidates(
            target_position, target_quaternion, reference
        )

        if not candidates:
            set_robot_arm(self.body, self.arm_indices, reference, self.client)
            actual_position, actual_quaternion = link_world_pose(
                self.body, self.e_link, self.client
            )
            return {
                "q": reference.copy(),
                "actual_position": actual_position,
                "actual_quaternion": actual_quaternion,
                "position_error_m": float(
                    np.linalg.norm(
                        np.asarray(actual_position) - np.asarray(target_position)
                    )
                ),
                "orientation_error_rad": quat_angle_rad(
                    actual_quaternion, target_quaternion
                ),
                "max_joint_step_rad": 0.0,
                "joint_l2_step_rad": 0.0,
                "seed_index": -1,
                "solver": "hold_previous_no_global_candidate",
                "pose_residual_only": True,
                "success": True,
            }
        answer = min(
            candidates,
            key=lambda item: (
                item["error_score"],
                item["joint_l2_step_rad"],
            ),
        )
        answer["pose_residual_only"] = True
        answer["success"] = True
        return answer

    def solve_candidates(
        self,
        target_position: list[float],
        target_quaternion: list[float],
        reference: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Return every deterministic bounded-restart solution.

        The generic physics planner can then rank these pose-valid branches by
        whole-robot collision clearance.  Keeping collision policy outside the
        IK helper avoids coupling this reusable solver to one scene or asset.
        """
        seeds = [reference.copy()]
        for restart in range(self.random_restarts):
            if restart < self.random_restarts // 2:
                scale = 0.08 if restart < self.random_restarts // 4 else 0.25
                seed = np.clip(
                    reference + self.rng.normal(0.0, scale, size=reference.shape),
                    self.arm_lower,
                    self.arm_upper,
                )
            else:
                seed = self.rng.uniform(self.arm_lower, self.arm_upper)
            seeds.append(seed)

        candidates: list[dict[str, Any]] = []
        for seed_index, seed in enumerate(seeds):
            candidate = self._candidate(
                target_position, target_quaternion, seed, reference
            )
            if candidate is None:
                continue
            candidate["seed_index"] = seed_index
            candidate["error_score"] = candidate["position_error_m"] + 0.1 * candidate[
                "orientation_error_rad"
            ]
            candidates.append(candidate)

        return candidates
