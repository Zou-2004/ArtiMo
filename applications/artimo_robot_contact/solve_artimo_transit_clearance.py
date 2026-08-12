#!/usr/bin/env python3
"""Evaluate one immutable batch of inter-stage obstacle-avoidance routes."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import run_artimo_physics as ph


POSITION_LIMIT_M = 0.004
ORIENTATION_LIMIT_DEG = 2.0
JOINT_LIMIT_MARGIN_RAD = 1e-4
ADJACENT_STEP_LIMIT_RAD = 0.0800001
CANDIDATE_WORKER = Path(__file__).with_name("artimo_transit_candidate_worker.py")


def _vec(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    answer = [float(item) for item in value]
    if not all(math.isfinite(item) for item in answer):
        raise ValueError(f"{label} must contain finite numbers")
    return answer


def _quat(value: Any, label: str) -> list[float]:
    answer = _vec(value, 4, label)
    norm = float(np.linalg.norm(answer))
    if norm < 1e-9:
        raise ValueError(f"{label} must be non-zero")
    return (np.asarray(answer, dtype=np.float64) / norm).tolist()


def _polyline_length(
    start: Iterable[float], waypoints: list[dict[str, Any]], end: Iterable[float]
) -> float:
    points = [np.asarray(list(start), dtype=np.float64)]
    points.extend(
        np.asarray(item["translation_m"], dtype=np.float64) for item in waypoints
    )
    points.append(np.asarray(list(end), dtype=np.float64))
    return float(
        sum(np.linalg.norm(right - left) for left, right in zip(points, points[1:]))
    )


def _outside_aabb(
    point: Iterable[float], low: Iterable[float], high: Iterable[float]
) -> bool:
    value = np.asarray(list(point), dtype=np.float64)
    return bool(
        np.any(value < np.asarray(list(low), dtype=np.float64))
        or np.any(value > np.asarray(list(high), dtype=np.float64))
    )


def _rank_key(report: dict[str, Any]) -> tuple[float, float, str]:
    clearance = report.get("minimum_full_robot_clearance_m")
    numeric = math.inf if clearance is None else float(clearance)
    return (-numeric, float(report["eef_polyline_length_m"]), str(report["id"]))


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Transit config must use schema_version 1")
    if not isinstance(config.get("execution_template_path"), str):
        raise ValueError("execution_template_path is required")
    if not isinstance(config.get("incoming_stage_id"), str):
        raise ValueError("incoming_stage_id is required")
    endpoints = config.get("transit_endpoints_world_m")
    if not isinstance(endpoints, dict):
        raise ValueError("transit_endpoints_world_m is required")
    start = _vec(endpoints.get("start"), 3, "transit start")
    end = _vec(endpoints.get("end"), 3, "transit end")
    obstacle = config.get("obstacle")
    if not isinstance(obstacle, dict):
        raise ValueError("obstacle is required")
    expanded = obstacle.get("expanded_aabb_world_m")
    if not isinstance(expanded, dict):
        raise ValueError("obstacle.expanded_aabb_world_m is required")
    low = _vec(expanded.get("low"), 3, "expanded AABB low")
    high = _vec(expanded.get("high"), 3, "expanded AABB high")
    if any(left >= right for left, right in zip(low, high)):
        raise ValueError("expanded AABB must satisfy low < high")
    raw_candidates = config.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidates must be a non-empty array")
    seen: set[str] = set()
    candidates = []
    for candidate_index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {candidate_index} must be an object")
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("candidate ids must be unique non-empty strings")
        seen.add(candidate_id)
        raw_waypoints = candidate.get(
            "transit_waypoints_world", candidate.get("waypoints_world")
        )
        if not isinstance(raw_waypoints, list) or not raw_waypoints:
            raise ValueError(f"candidate {candidate_id} must declare waypoints")
        waypoints = []
        for waypoint_index, waypoint in enumerate(raw_waypoints):
            if not isinstance(waypoint, dict):
                raise ValueError(f"candidate {candidate_id} waypoint must be an object")
            translation = _vec(
                waypoint.get("translation_m"),
                3,
                f"candidate {candidate_id} waypoint {waypoint_index} translation",
            )
            if not _outside_aabb(translation, low, high):
                raise ValueError(
                    f"candidate {candidate_id} waypoint {waypoint_index} lies inside "
                    "the expanded obstacle AABB"
                )
            waypoints.append(
                {
                    "translation_m": translation,
                    "rotation_xyzw": _quat(
                        waypoint.get("rotation_xyzw"),
                        f"candidate {candidate_id} waypoint {waypoint_index} rotation",
                    ),
                }
            )
        candidates.append(
            {
                "id": candidate_id,
                "side": candidate.get("side"),
                "transit_waypoints_world": waypoints,
            }
        )
    answer = copy.deepcopy(config)
    answer["transit_endpoints_world_m"] = {"start": start, "end": end}
    answer["obstacle"]["expanded_aabb_world_m"] = {"low": low, "high": high}
    answer["candidates"] = candidates
    return answer


def _stage_report(plan: ph.StagePlan) -> dict[str, Any]:
    required = float(plan.stage.get("minimum_swept_clearance_m", 0.0))
    clearance_passed = (
        plan.minimum_swept_clearance_m is None
        or plan.minimum_swept_clearance_m >= required
    )
    ik_passed = (
        not plan.debug_truncated
        and plan.maximum_position_error_m <= POSITION_LIMIT_M
        and math.degrees(plan.maximum_orientation_error_rad) <= ORIENTATION_LIMIT_DEG
        and plan.minimum_joint_limit_margin_rad > JOINT_LIMIT_MARGIN_RAD
        and plan.maximum_adjacent_joint_step_rad <= ADJACENT_STEP_LIMIT_RAD
    )
    return {
        "stage_id": plan.stage["id"],
        "feasible": bool(clearance_passed and ik_passed),
        "minimum_swept_clearance_m": plan.minimum_swept_clearance_m,
        "required_swept_clearance_m": required,
        "clearance_passed": bool(clearance_passed),
        "maximum_position_error_m": plan.maximum_position_error_m,
        "maximum_orientation_error_deg": math.degrees(
            plan.maximum_orientation_error_rad
        ),
        "minimum_joint_limit_margin_rad": plan.minimum_joint_limit_margin_rad,
        "maximum_adjacent_joint_step_rad": plan.maximum_adjacent_joint_step_rad,
        "ik_passed": bool(ik_passed),
        "tightest_swept_samples": plan.swept_clearance_violations[:5],
        "debug_failure": plan.debug_failure,
    }


def _evaluate_candidate(
    candidate: dict[str, Any],
    grounded: dict[str, Any],
    stage_index: int,
    simulation_urdf: Path,
    robot_urdf: Path,
    initial: dict[str, float],
    transit_start: list[float],
    transit_end: list[float],
) -> dict[str, Any]:
    candidate_execution = copy.deepcopy(grounded)
    candidate_execution["stages"][stage_index]["transit_waypoints_world"] = copy.deepcopy(
        candidate["transit_waypoints_world"]
    )
    length = _polyline_length(
        transit_start,
        candidate["transit_waypoints_world"],
        transit_end,
    )
    try:
        ph._validate_execution_schema(candidate_execution)
        plans = ph._plan_stages(
            simulation_urdf,
            robot_urdf,
            candidate_execution,
            initial,
            validate_release_clearance=False,
        )
    except Exception as exc:
        return {
            **candidate,
            "feasible": False,
            "eef_polyline_length_m": length,
            "error": str(exc)[:2000],
            "execution": None,
        }
    stages = [_stage_report(plan) for plan in plans]
    clearances = [
        float(plan.minimum_swept_clearance_m)
        for plan in plans
        if plan.minimum_swept_clearance_m is not None
    ]
    feasible = all(stage["feasible"] for stage in stages)
    return {
        **candidate,
        "feasible": bool(feasible),
        "minimum_full_robot_clearance_m": min(clearances)
        if clearances
        else None,
        "eef_polyline_length_m": length,
        "stages": stages,
        "execution": candidate_execution if feasible else None,
    }


def _run_candidate_worker(request_path: Path, response_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(CANDIDATE_WORKER),
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Transit candidate worker failed for {request_path.name}: "
            f"{completed.stdout[-4000:]}"
        )


def solve(
    config: dict[str, Any], task: dict[str, Any], *, jobs: int = 1
) -> dict[str, Any]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    config = _normalize_config(config)
    execution = ph._read_json(ph._resolve(config["execution_template_path"]))
    ph._validate_execution_schema(execution)
    matches = [
        index
        for index, stage in enumerate(execution["stages"])
        if stage["id"] == config["incoming_stage_id"]
    ]
    if len(matches) != 1 or matches[0] == 0:
        raise ValueError("incoming_stage_id must match one non-first stage")
    stage_index = matches[0]
    incoming = execution["stages"][stage_index]
    if ph._same_contact_sequence(execution["stages"][stage_index - 1], incoming):
        raise ValueError("continuous contact_sequence cannot declare a transit")
    obstacle_link = str(config["obstacle"].get("link", ""))
    if obstacle_link not in incoming.get("forbidden_contact_links", []):
        raise ValueError("obstacle link must be forbidden by the incoming stage")

    source_urdf = ph._resolve(task["inputs"]["urdf"])
    simulation_urdf = ph.resolve_simulation_urdf(task, execution, source_urdf)
    ph._require_matching_mechanism(source_urdf, simulation_urdf)
    robot_urdf = ph._resolve(task["inputs"]["robot_urdf"])
    initial = ph.task_initial_joint_values(task)
    grounded = ph._ground_execution_scene(
        simulation_urdf, robot_urdf, copy.deepcopy(execution), initial
    )["execution"]
    if jobs == 1 or len(config["candidates"]) == 1:
        attempts = [
            _evaluate_candidate(
                candidate,
                grounded,
                stage_index,
                simulation_urdf,
                robot_urdf,
                initial,
                config["transit_endpoints_world_m"]["start"],
                config["transit_endpoints_world_m"]["end"],
            )
            for candidate in config["candidates"]
        ]
    else:
        # Each immutable candidate owns a separate PyBullet DIRECT client.
        # Ordinary child interpreters avoid both the GIL and restricted
        # process-pool semaphore APIs. Each request is immutable and futures
        # are consumed in config order, so output remains deterministic.
        with tempfile.TemporaryDirectory(prefix="artimo-transit-candidates-") as raw_temp:
            temporary = Path(raw_temp)
            requests = []
            responses = []
            for candidate_index, candidate in enumerate(config["candidates"]):
                request_path = temporary / f"request-{candidate_index:03d}.json"
                response_path = temporary / f"response-{candidate_index:03d}.json"
                request_path.write_text(
                    json.dumps(
                        {
                            "candidate": candidate,
                            "grounded": grounded,
                            "stage_index": stage_index,
                            "simulation_urdf": str(simulation_urdf),
                            "robot_urdf": str(robot_urdf),
                            "initial": initial,
                            "transit_start": config["transit_endpoints_world_m"]["start"],
                            "transit_end": config["transit_endpoints_world_m"]["end"],
                        },
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                requests.append(request_path)
                responses.append(response_path)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(jobs, len(requests))
            ) as executor:
                futures = [
                    executor.submit(_run_candidate_worker, request, response)
                    for request, response in zip(requests, responses)
                ]
                for future in futures:
                    future.result()
            attempts = [
                json.loads(response.read_text(encoding="utf-8"))
                for response in responses
            ]
    feasible_attempts = [attempt for attempt in attempts if attempt["feasible"]]
    chosen = min(feasible_attempts, key=_rank_key) if feasible_attempts else None
    return {
        "schema_version": 1,
        "feasible": chosen is not None,
        "incoming_stage_id": config["incoming_stage_id"],
        "obstacle": config["obstacle"],
        "ranking": [
            "minimum_full_robot_clearance_m_descending",
            "eef_polyline_length_m_ascending",
            "candidate_id_ascending",
        ],
        "chosen": None
        if chosen is None
        else {key: value for key, value in chosen.items() if key != "execution"},
        "execution": None if chosen is None else chosen["execution"],
        "attempts": [
            {key: value for key, value in attempt.items() if key != "execution"}
            for attempt in attempts
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    try:
        answer = solve(
            ph._read_json(args.config.expanduser().resolve()),
            ph._read_json(args.task_spec.expanduser().resolve()),
            jobs=int(args.jobs),
        )
        out = args.out.expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "transit.json").write_text(
            json.dumps(answer, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if answer["execution"] is not None:
            (out / "execution.json").write_text(
                json.dumps(answer["execution"], indent=2, ensure_ascii=False, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        print(
            json.dumps(
                {
                    "feasible": answer["feasible"],
                    "chosen": answer["chosen"],
                    "transit": str(out / "transit.json"),
                    "execution": str(out / "execution.json")
                    if answer["execution"] is not None
                    else None,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0 if answer["feasible"] else 2
    except Exception as exc:
        print(f"Transit-clearance solve failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
