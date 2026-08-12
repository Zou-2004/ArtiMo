#!/usr/bin/env python3
"""Verify the stable agent-neutral ArtiMo three-file delivery contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import jsonschema

import artimo_video


APPLICATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APPLICATION_ROOT.parents[1]
FINAL_FILES = {"video.mp4", "grasp.json", "result.json"}
HEX64 = set("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_repo(path: Path, root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(root))) == str(root)
    except ValueError:
        return False


def _repo_path(value: str, root: Path) -> Path:
    raw = Path(value).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    if not _inside_repo(resolved, root):
        raise ValueError(f"path escapes repository root: {value}")
    return resolved


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))


def _is_hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX64)
    )


def _require(condition: bool, name: str, checks: dict[str, bool], errors: list[str]) -> None:
    checks[name] = bool(condition)
    if not condition:
        errors.append(name)


def _decode_video(path: Path) -> tuple[bool, dict[str, Any], str]:
    decode, decode_backend = artimo_video.decode_video_to_null(path)
    try:
        metadata = artimo_video.probe_video(path)
    except Exception as exc:
        return False, {}, f"video metadata probe failed: {exc}"
    metadata["artimo_decode_backend"] = decode_backend
    return decode.returncode == 0, metadata, decode.stderr


def _validate_standard_manifest(
    manifest: dict[str, Any],
    spec: dict[str, Any],
    spec_sha256: str,
    execution_sha256: str,
    requested_motion: dict[str, list[float]],
    expected_release_lock_sha256: str,
    expected_handoff_lock_sha256: str | None,
    checks: dict[str, bool],
    errors: list[str],
) -> None:
    acceptance = spec["acceptance"]
    _require(manifest.get("schema_version") == 2, "manifest_schema_v2", checks, errors)
    _require(
        manifest.get("task_spec_sha256") == spec_sha256,
        "manifest_task_spec_hash_matches",
        checks,
        errors,
    )
    _require(
        manifest.get("release_lock_sha256") == expected_release_lock_sha256,
        "manifest_release_lock_hash_matches",
        checks,
        errors,
    )
    _require(
        manifest.get("execution_plan_sha256") == execution_sha256,
        "manifest_execution_plan_hash_matches",
        checks,
        errors,
    )
    handoff_lock = manifest.get("handoff_lock_sha256")
    _require(
        (
            handoff_lock == expected_handoff_lock_sha256
            if expected_handoff_lock_sha256 is not None
            else _is_hex64(handoff_lock)
        ),
        "manifest_handoff_lock_hash_valid",
        checks,
        errors,
    )
    _require(manifest.get("physics_engine") == "PyBullet", "physics_engine_pybullet", checks, errors)
    _require(manifest.get("physical_only_video") is True, "physical_only_video", checks, errors)
    _require(manifest.get("object_trajectory_replay") is False, "no_object_replay", checks, errors)
    _require(
        manifest.get("object_joint_resets_after_initialization") == 0,
        "no_runtime_object_resets",
        checks,
        errors,
    )
    if acceptance.get("require_zero_fixed_constraints", True):
        _require(manifest.get("fixed_constraint_count") == 0, "zero_fixed_constraints", checks, errors)
    _require(_is_hex64(manifest.get("robot_command_schedule_sha256")), "command_schedule_hash", checks, errors)
    _require(isinstance(manifest.get("seeds"), dict) and bool(manifest["seeds"]), "deterministic_seeds_recorded", checks, errors)

    physical = manifest.get("physical")
    _require(isinstance(physical, dict), "physical_manifest_present", checks, errors)
    if isinstance(physical, dict):
        contacts = physical.get("contacts")
        _require(isinstance(contacts, list) and bool(contacts), "physical_contacts_present", checks, errors)
        if isinstance(contacts, list):
            for index, contact in enumerate(contacts):
                prefix = f"physical_contact_{index}"
                _require(int(contact.get("target_contact_observations", 0)) > 0, f"{prefix}_target_positive", checks, errors)
                _require(contact.get("non_target_contact_observations") == 0, f"{prefix}_non_target_zero", checks, errors)
                _require(contact.get("effect_link_contact_observations") == 0, f"{prefix}_effect_zero", checks, errors)
                duration = float(contact.get("continuous_contact_s", math.nan))
                _require(
                    math.isfinite(duration)
                    and duration >= float(acceptance.get("minimum_continuous_contact_s", 0.0)),
                    f"{prefix}_duration",
                    checks,
                    errors,
                )
        joint_motion = physical.get("joint_motion")
        _require(isinstance(joint_motion, dict), "physical_joint_motion_present", checks, errors)
        if isinstance(joint_motion, dict):
            for joint, extrema in requested_motion.items():
                measured = joint_motion.get(joint)
                _require(isinstance(measured, dict), f"joint_{joint}_present", checks, errors)
                if not isinstance(measured, dict):
                    continue
                reported = measured.get("requested_extrema")
                targets_match = (
                    isinstance(reported, list)
                    and len(reported) == len(extrema)
                    and all(abs(float(a) - float(b)) <= 1e-6 for a, b in zip(reported, extrema))
                )
                _require(targets_match, f"joint_{joint}_targets_match_plan", checks, errors)
                _require(
                    float(measured.get("minimum_progress_ratio", -1.0))
                    >= float(acceptance["minimum_joint_motion_ratio"]),
                    f"joint_{joint}_progress",
                    checks,
                    errors,
                )
                _require(measured.get("order_passed") is True, f"joint_{joint}_order", checks, errors)

    control = manifest.get("negative_control")
    _require(isinstance(control, dict), "negative_control_present", checks, errors)
    if isinstance(control, dict):
        _require(control.get("same_robot_command_schedule") is True, "negative_same_schedule", checks, errors)
        _require(control.get("target_contact_observations") == 0, "negative_target_contact_zero", checks, errors)
        _require(control.get("causal_triggers") == 0, "negative_not_triggered", checks, errors)
        _require(control.get("requested_joint_motion_remained_initial") is True, "negative_motion_initial", checks, errors)

    visual = manifest.get("visual_qa")
    _require(isinstance(visual, dict), "visual_qa_present", checks, errors)
    if isinstance(visual, dict):
        _require(
            float(visual.get("sample_rate_fps", 0.0))
            >= float(acceptance.get("visual_review_fps", 5.0)),
            "visual_dense_sampling",
            checks,
            errors,
        )
        for key in (
            "no_visible_interpenetration",
            "physical_contact_visible",
            "requested_motion_visible",
            "no_rendering_artifacts",
        ):
            _require(visual.get(key) is True, f"visual_{key}", checks, errors)

def _requested_joint_motion(plan_path: Path) -> dict[str, list[float]]:
    """Read the plan's per-joint extrema through the harness's canonical parser.

    Importing the application's ``artimo_plan.py`` rather than re-implementing the traversal
    is what guarantees the acceptance gate and the measured manifest describe the
    same requested extrema.
    """
    module_path = APPLICATION_ROOT / "artimo_plan.py"
    module_spec = importlib.util.spec_from_file_location(
        "artimo_verifier_plan", module_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.requested_extrema(module.read_plan(plan_path))


def _validate_execution_ownership(
    grasp: dict[str, Any], plan_path: Path, repo_root: Path
) -> tuple[bool, str | None]:
    """Run the same schema and per-control owner contract used by the simulator."""
    application_root = repo_root / "applications" / "artimo_robot_contact"
    schema_path = application_root / "schemas" / "artimo_robot_execution.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema_errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(grasp),
        key=lambda item: list(item.path),
    )
    if schema_errors:
        first = schema_errors[0]
        location = "/".join(str(part) for part in first.path) or "<root>"
        return False, f"execution schema error at {location}: {first.message}"
    module_path = application_root / "run_artimo_physics.py"
    module_spec = importlib.util.spec_from_file_location("artimo_verifier_physics", module_path)
    if module_spec is None or module_spec.loader is None:
        return False, f"Could not load {module_path}"
    module = importlib.util.module_from_spec(module_spec)
    # Dataclasses resolve the defining module through sys.modules while the
    # module body executes.
    import sys

    application_path = str(module_path.parent)
    inserted_application_path = application_path not in sys.path
    if inserted_application_path:
        sys.path.insert(0, application_path)
    sys.modules[module_spec.name] = module
    try:
        module_spec.loader.exec_module(module)
        plan = module.artimo_plan.read_plan(plan_path)
        module._validate_execution_against_plan(plan, grasp)
    except Exception as exc:
        return False, str(exc)
    finally:
        sys.modules.pop(module_spec.name, None)
        if inserted_application_path:
            sys.path.remove(application_path)
    return True, None


def _current_lock_sha256(spec_path: Path, repo_root: Path) -> str:
    renderer_path = Path(__file__).with_name("render_task_prompt.py")
    module_spec = importlib.util.spec_from_file_location(
        "artimo_verifier_prompt_renderer", renderer_path
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load {renderer_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    _, lock, _ = module.build(spec_path, repo_root)
    return lock["lock_sha256"]


def verify(
    spec_path: Path,
    repo_root: Path = REPO_ROOT,
    expected_release_lock_sha256: str | None = None,
    expected_handoff_lock_sha256: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    spec_path = _repo_path(str(spec_path), repo_root)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output_dir = _repo_path(spec["output_dir"], repo_root)
    checks: dict[str, bool] = {}
    errors: list[str] = []

    files = {path.name for path in output_dir.iterdir()} if output_dir.is_dir() else set()
    _require(files == FINAL_FILES, "exact_three_file_delivery", checks, errors)
    if files != FINAL_FILES:
        return {"passed": False, "checks": checks, "errors": errors, "files": sorted(files)}

    result_path = output_dir / "result.json"
    grasp_path = output_dir / "grasp.json"
    video_path = output_dir / "video.mp4"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    grasp = json.loads(grasp_path.read_text(encoding="utf-8"))
    _require(result.get("passed") is True, "result_passed", checks, errors)
    execution_physics = grasp.get("physics_urdf")
    if execution_physics is not None:
        proxy_path = _repo_path(str(execution_physics), repo_root)
        expected_proxy_root = (repo_root / ".artimo-runs" / str(spec["task_id"])).resolve()
        try:
            proxy_path.relative_to(expected_proxy_root)
            proxy_scoped = True
        except ValueError:
            proxy_scoped = False
        _require(proxy_scoped, "execution_proxy_scoped_to_task_debug", checks, errors)
        _require(proxy_path.is_file(), "execution_proxy_exists", checks, errors)
        if proxy_path.is_file():
            proxy_hash = _sha256(proxy_path)
            rollout_inputs = result.get("native_rollout", {}).get("inputs", {})
            _require(
                rollout_inputs.get("collision_proxy_used") is True,
                "execution_proxy_used_in_rollout",
                checks,
                errors,
            )
            _require(
                rollout_inputs.get("simulated_urdf_sha256") == proxy_hash,
                "execution_proxy_hash_matches_rollout",
                checks,
                errors,
            )
    stages = grasp.get("stages")
    _require(isinstance(stages, list) and bool(stages), "contact_stages_present", checks, errors)
    if isinstance(stages, list):
        for index, stage in enumerate(stages):
            _require(isinstance(stage, dict), f"stage_{index}_object", checks, errors)
            if not isinstance(stage, dict):
                continue
            for key in (
                "id",
                "source_phase",
                "source_control_index",
                "interaction",
                "driver_joint",
                "contact_link",
                "contact_pose_link",
            ):
                _require(key in stage, f"stage_{index}_{key}_present", checks, errors)

    plan_path = _repo_path(spec["inputs"]["plan"], repo_root)
    ownership_ok, ownership_error = _validate_execution_ownership(grasp, plan_path, repo_root)
    _require(ownership_ok, "control_execution_ownership_valid", checks, errors)
    if ownership_error is not None:
        errors.append(f"control_execution_ownership_error: {ownership_error}")
    physical_contacts = result.get("evidence", {}).get("physical", {}).get("contacts", [])
    measured_stage_ids = {
        str(contact.get("stage_id"))
        for contact in physical_contacts
        if isinstance(contact, dict) and int(contact.get("target_contact_observations", 0)) > 0
    }
    declared_robot_stage_ids = {
        str(row.get("stage_id"))
        for row in grasp.get("control_execution", [])
        if isinstance(row, dict) and row.get("motion_owner") == "robot_contact"
    }
    _require(
        bool(declared_robot_stage_ids) and declared_robot_stage_ids == measured_stage_ids,
        "every_robot_owned_stage_has_contact_evidence",
        checks,
        errors,
    )

    decode_ok, metadata, decode_error = _decode_video(video_path)
    _require(decode_ok, "video_full_decode", checks, errors)
    streams = metadata.get("streams", []) if isinstance(metadata, dict) else []
    _require(bool(streams) and streams[0].get("codec_name") == "h264", "video_h264", checks, errors)
    _require(video_path.stat().st_size > 0, "video_nonempty", checks, errors)

    manifest = result.get("evidence")
    _require(isinstance(manifest, dict), "standard_manifest_present", checks, errors)
    if isinstance(manifest, dict):
        if expected_release_lock_sha256 is None:
            expected_release_lock_sha256 = _current_lock_sha256(spec_path, repo_root)
        _validate_standard_manifest(
            manifest,
            spec,
            _sha256(spec_path),
            _sha256(grasp_path),
            _requested_joint_motion(plan_path),
            expected_release_lock_sha256,
            expected_handoff_lock_sha256,
            checks,
            errors,
        )

    passed = not errors
    return {
        "schema_version": 1,
        "passed": passed,
        "task_id": spec["task_id"],
        "output_dir": str(output_dir.relative_to(repo_root)),
        "checks": checks,
        "errors": errors,
        "artifacts": {
            "video": {"sha256": _sha256(video_path), "bytes": video_path.stat().st_size, "probe": metadata},
            "grasp": {"sha256": _sha256(grasp_path), "bytes": grasp_path.stat().st_size},
            "result": {"sha256": _sha256(result_path), "bytes": result_path.stat().st_size},
        },
        "decode_error": decode_error or None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify(args.spec.expanduser(), args.repo_root.expanduser().resolve())
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.expanduser().write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
