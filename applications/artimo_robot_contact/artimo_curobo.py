"""Subprocess bridge from the normal ArtiMo environment to cuRobo."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence


_PREFIX = "ARTIMO_CUROBO_RESPONSE "


def _urdf_root_link(path: Path) -> str:
    root = ET.parse(path).getroot()
    links = {node.attrib["name"] for node in root.findall("link")}
    children = {
        node.find("child").attrib["link"]
        for node in root.findall("joint")
        if node.find("child") is not None
    }
    roots = sorted(links - children)
    if len(roots) != 1:
        raise ValueError(f"Robot URDF must have exactly one root link, found {roots}")
    return roots[0]


class CuroboBatchIK:
    def __init__(self, config: dict[str, Any], robot_urdf: Path, robot: dict[str, Any]) -> None:
        executable = Path(str(config["python_executable"])).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(f"cuRobo Python executable does not exist: {executable}")
        command = [
            str(executable), "-u", str(Path(__file__).with_name("artimo_curobo_worker.py")),
            "--robot-urdf", str(robot_urdf.resolve()),
            "--base-link", _urdf_root_link(robot_urdf),
            "--end-effector-link", str(robot["end_effector_link"]),
            "--arm-joint-names", *[str(value) for value in robot["arm_joint_names"]],
            "--num-seeds", str(int(config.get("num_seeds", 32))),
            "--return-seeds", str(int(config.get("return_seeds", 8))),
            "--device", str(config.get("device", "cuda:0")),
            "--collision-sphere-buffer-m", str(
                float(config.get("collision_sphere_buffer_m", 0.0))
            ),
            "--motion-num-graph-seeds", str(
                int(config.get("motion_num_graph_seeds", 4))
            ),
            "--motion-num-trajopt-seeds", str(
                int(config.get("motion_num_trajopt_seeds", 4))
            ),
            "--motion-timeout-s", str(
                float(config.get("motion_timeout_s", 10.0))
            ),
            "--motion-max-attempts", str(
                int(config.get("motion_max_attempts", 6))
            ),
        ]
        if not bool(config.get("self_collision", True)):
            command.append("--disable-self-collision")
        if bool(config.get("cuda_graph", True)):
            command.append("--cuda-graph")
        worker_env = os.environ.copy()
        environment_root = executable.parent
        worker_env["PATH"] = os.pathsep.join(
            [
                str(environment_root / "bin"),
                str(environment_root / "Library" / "bin"),
                str(environment_root / "Scripts"),
                worker_env.get("PATH", ""),
            ]
        )
        # Warp otherwise compiles new collision kernels below the user's
        # AppData cache, which is outside a repository-scoped run sandbox.
        # Keep generated CUDA cache data recoverable and task-neutral.
        warp_cache = (Path.cwd() / ".artimo-runs" / "_gpu-cache" / "warp").resolve()
        warp_cache.mkdir(parents=True, exist_ok=True)
        worker_env["WARP_CACHE_PATH"] = str(warp_cache)
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=worker_env,
        )
        self._stderr: list[str] = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._request_id = 0
        self._rpc_lock = threading.Lock()
        self.allow_bullet_fallback = bool(config.get("allow_bullet_fallback", False))
        self.environment_collision = bool(config.get("environment_collision", True))
        self.self_collision = bool(config.get("self_collision", True))
        atexit.register(self.close)

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in self.process.stderr:
            self._stderr.append(line.rstrip())
            del self._stderr[:-100]

    def solve_path(
        self,
        positions_world: Sequence[Sequence[float]],
        quaternions_xyzw_world: Sequence[Sequence[float]],
        robot_base_position_world: Sequence[float],
        robot_base_quaternion_xyzw_world: Sequence[float],
        reference: Sequence[float],
        maximum_joint_step_rad: float | None,
        enforce_start_step: bool,
        obstacle_worlds_by_sample: Sequence[Sequence[dict[str, Any]]] | None = None,
        sequential: bool = False,
    ) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise RuntimeError("cuRobo worker exited: " + "\n".join(self._stderr[-20:]))
        self._request_id += 1
        request = {
            "id": self._request_id, "command": "solve_path",
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
        }
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("cuRobo worker closed stdout: " + "\n".join(self._stderr[-20:]))
            if not line.startswith(_PREFIX):
                continue
            response = json.loads(line[len(_PREFIX):])
            if response.get("worker_error"):
                raise RuntimeError(
                    str(response["worker_error"]) + "\n" + str(response.get("worker_traceback", ""))
                )
            return response

    def solve_paths_batch(
        self, paths: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Solve several independent pose paths in one GPU environment batch."""
        if not paths:
            return []
        if self.process.poll() is not None:
            raise RuntimeError("cuRobo worker exited: " + "\n".join(self._stderr[-20:]))
        with self._rpc_lock:
            self._request_id += 1
            request = {
                "id": self._request_id,
                "command": "solve_paths_batch",
                "paths": list(paths),
            }
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        "cuRobo worker closed stdout: " + "\n".join(self._stderr[-20:])
                    )
                if not line.startswith(_PREFIX):
                    continue
                response = json.loads(line[len(_PREFIX):])
                if response.get("worker_error"):
                    raise RuntimeError(
                        str(response["worker_error"])
                        + "\n"
                        + str(response.get("worker_traceback", ""))
                    )
                results = response.get("results")
                if not isinstance(results, list) or len(results) != len(paths):
                    raise RuntimeError("cuRobo batch response does not align with requests")
                return results

    def plan_joint_path(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        robot_base_position_world: Sequence[float],
        robot_base_quaternion_xyzw_world: Sequence[float],
        obstacle_world: Sequence[dict[str, Any]],
        maximum_joint_step_rad: float = 0.08,
        required_clearance_m: float = 0.0,
    ) -> dict[str, Any]:
        """Plan one collision-aware joint transit entirely in cuRobo.

        The returned path is still verified by the ordinary PyBullet planning
        client before it can be serialized or rolled out.  This method replaces
        only the CPU RRT proposal/search, not the authoritative physics check.
        """
        if self.process.poll() is not None:
            raise RuntimeError("cuRobo worker exited: " + "\n".join(self._stderr[-20:]))
        self._request_id += 1
        request = {
            "id": self._request_id,
            "command": "plan_joint_path",
            "start": list(map(float, start)),
            "goal": list(map(float, goal)),
            "robot_base_position_world": list(map(float, robot_base_position_world)),
            "robot_base_quaternion_xyzw_world": list(
                map(float, robot_base_quaternion_xyzw_world)
            ),
            "obstacle_world": list(obstacle_world),
            "maximum_joint_step_rad": float(maximum_joint_step_rad),
            "required_clearance_m": float(required_clearance_m),
        }
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "cuRobo worker closed stdout: " + "\n".join(self._stderr[-20:])
                )
            if not line.startswith(_PREFIX):
                continue
            response = json.loads(line[len(_PREFIX):])
            if response.get("worker_error"):
                raise RuntimeError(
                    str(response["worker_error"])
                    + "\n"
                    + str(response.get("worker_traceback", ""))
                )
            return response

    def check_joint_path(
        self,
        joint_path: Sequence[Sequence[float]],
        robot_base_position_world: Sequence[float],
        robot_base_quaternion_xyzw_world: Sequence[float],
        obstacle_worlds_by_sample: Sequence[Sequence[dict[str, Any]]],
        required_clearance_m: float = 0.0,
        finger_opening_m: float | None = None,
    ) -> dict[str, Any]:
        """Check an already selected joint path against source-mesh worlds."""
        if self.process.poll() is not None:
            raise RuntimeError("cuRobo worker exited: " + "\n".join(self._stderr[-20:]))
        self._request_id += 1
        request = {
            "id": self._request_id,
            "command": "check_joint_path",
            "joint_path": [list(map(float, row)) for row in joint_path],
            "robot_base_position_world": list(map(float, robot_base_position_world)),
            "robot_base_quaternion_xyzw_world": list(
                map(float, robot_base_quaternion_xyzw_world)
            ),
            "obstacle_worlds_by_sample": obstacle_worlds_by_sample,
            "required_clearance_m": float(required_clearance_m),
            "finger_opening_m": (
                None if finger_opening_m is None else float(finger_opening_m)
            ),
        }
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    "cuRobo worker closed stdout: " + "\n".join(self._stderr[-20:])
                )
            if not line.startswith(_PREFIX):
                continue
            response = json.loads(line[len(_PREFIX):])
            if response.get("worker_error"):
                raise RuntimeError(
                    str(response["worker_error"])
                    + "\n"
                    + str(response.get("worker_traceback", ""))
                )
            return response

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None or process.poll() is not None:
            return
        try:
            assert process.stdin is not None
            process.stdin.write('{"command":"close"}\n')
            process.stdin.flush()
            process.wait(timeout=5)
        except Exception:
            process.terminate()


def create_curobo_backend(
    config: dict[str, Any], robot_urdf: Path, robot: dict[str, Any]
) -> CuroboBatchIK | None:
    backend = config.get("planning_ik_backend", {"name": "bullet"})
    if not isinstance(backend, dict):
        raise ValueError("planning_ik_backend must be an object")
    name = str(backend.get("name", "bullet"))
    if name == "bullet":
        return None
    if name != "curobo":
        raise ValueError(f"Unknown planning IK backend {name!r}")
    return CuroboBatchIK(backend, robot_urdf, robot)
