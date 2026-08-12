#!/usr/bin/env python3
"""Render a deterministic agent-neutral prompt and immutable input lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any

import artimo_video


APPLICATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APPLICATION_ROOT.parents[1]
DOCS_ROOT = APPLICATION_ROOT / "docs"
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _inside_repo(path: Path, repo_root: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(repo_root))) == str(repo_root)
    except ValueError:
        return False


def _repo_file(value: str, repo_root: Path, label: str) -> Path:
    raw = Path(value).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    if not _inside_repo(resolved, repo_root):
        raise ValueError(f"{label} escapes repository root: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _repo_output(value: str, repo_root: Path) -> Path:
    raw = Path(value).expanduser()
    resolved = raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()
    if not _inside_repo(resolved, repo_root) or resolved == repo_root:
        raise ValueError(f"output_dir must be a repository subdirectory: {value}")
    return resolved


def _record(path: Path, repo_root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(repo_root)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _resolve_asset_reference(
    reference: str,
    referrer: Path,
    package_root: Path,
    repo_root: Path,
) -> Path:
    value = reference.strip()
    if value.startswith("package://"):
        candidate = package_root / value.removeprefix("package://")
    elif value.startswith("file://"):
        candidate = Path(value.removeprefix("file://"))
    elif re.match(r"^[A-Za-z]:[\\/]", value):
        # Some bundled Panda MTL files retain a build-machine Windows path.
        # Resolve its basename only when it is unique inside the URDF package.
        matches = sorted(package_root.rglob(Path(value.replace("\\", "/")).name))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"asset reference has {len(matches)} package matches: "
                f"{value!r} from {referrer}"
            )
        candidate = matches[0]
    else:
        candidate = referrer.parent / value
    resolved = candidate.resolve()
    if not _inside_repo(resolved, repo_root):
        raise ValueError(f"asset dependency escapes repository: {reference} from {referrer}")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing asset dependency: {reference} from {referrer}")
    return resolved


def _direct_asset_references(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".urdf":
        root = ET.parse(path).getroot()
        return [
            element.attrib["filename"]
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "mesh"
            and element.attrib.get("filename")
        ]
    if suffix == ".obj":
        references: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lstrip().lower().startswith("mtllib "):
                    references.extend(shlex.split(line.strip())[1:])
        return references
    if suffix == ".mtl":
        references = []
        texture_directives = {
            "map_ka", "map_kd", "map_ks", "map_ke", "map_ns", "map_d",
            "map_bump", "bump", "disp", "decal", "refl",
        }
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                fields = shlex.split(line.strip(), comments=True, posix=False)
                if fields and fields[0].lower() in texture_directives and len(fields) > 1:
                    references.append(fields[-1].strip("\"'"))
        return references
    if suffix == ".dae":
        root = ET.parse(path).getroot()
        return [
            (element.text or "").strip()
            for element in root.iter()
            if element.tag.rsplit("}", 1)[-1] == "init_from"
            and (element.text or "").strip()
        ]
    if suffix == ".gltf":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [
            item["uri"]
            for group in ("buffers", "images")
            for item in payload.get(group, [])
            if isinstance(item, dict)
            and isinstance(item.get("uri"), str)
            and not item["uri"].startswith("data:")
        ]
    return []


def _asset_dependency_records(
    urdfs: list[Path], repo_root: Path
) -> dict[str, Any]:
    """Hash all transitive file dependencies reachable from every URDF."""
    queue = deque((path, path.parent) for path in urdfs)
    visited = {path.resolve() for path in urdfs}
    dependencies: set[Path] = set()
    unresolved: list[dict[str, str]] = []
    while queue:
        referrer, package_root = queue.popleft()
        for reference in _direct_asset_references(referrer):
            try:
                dependency = _resolve_asset_reference(
                    reference, referrer, package_root, repo_root
                )
            except FileNotFoundError:
                # A missing URDF mesh changes collision/visual geometry and is
                # fatal. Bundled OBJ/MTL files sometimes name an optional,
                # absent material/texture; preserve that fact in the lock.
                if referrer.suffix.lower() == ".urdf":
                    raise
                unresolved.append(
                    {
                        "referrer": str(referrer.relative_to(repo_root)),
                        "reference": reference,
                    }
                )
                continue
            if dependency in visited:
                continue
            visited.add(dependency)
            dependencies.add(dependency)
            queue.append((dependency, package_root))
    return {
        "files": {
            str(path.relative_to(repo_root)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(dependencies)
        },
        "unresolved_references": sorted(
            unresolved, key=lambda item: (item["referrer"], item["reference"])
        ),
    }


def _first_line(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else None


def _toolchain_state() -> dict[str, Any]:
    video_encoder = artimo_video.select_encoder()
    return {
        "python": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "ffmpeg": _first_line([artimo_video.ffmpeg_executable(), "-version"]),
        "video_encoder": {
            "codec": video_encoder["codec"],
            "hardware_accelerated": video_encoder["hardware_accelerated"],
        },
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_sha256": (
            None
            if status is None
            else hashlib.sha256(status.encode("utf-8")).hexdigest()
        ),
    }


def _validate_spec(spec: dict[str, Any], repo_root: Path) -> dict[str, Path]:
    if spec.get("schema_version") != 2:
        raise ValueError("task spec schema_version must be 2")
    task_id = spec.get("task_id")
    if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
        raise ValueError("task_id must match [a-z0-9][a-z0-9._-]*")
    if not isinstance(spec.get("task_description"), str) or not spec[
        "task_description"
    ].strip():
        raise ValueError("task_description must be a non-empty string")
    inputs = spec.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("inputs must be an object")
    resolved: dict[str, Path] = {}
    for key in ("urdf", "plan"):
        value = inputs.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"inputs.{key} is required")
        resolved[key] = _repo_file(value, repo_root, f"inputs.{key}")
    trajectory_value = inputs.get("trajectory")
    if trajectory_value is not None:
        if not isinstance(trajectory_value, str) or not trajectory_value:
            raise ValueError("inputs.trajectory must be a path string")
        resolved["trajectory"] = _repo_file(
            trajectory_value, repo_root, "inputs.trajectory"
        )
    for key in ("physics_urdf",):
        value = inputs.get(key)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"inputs.{key} must be a path string")
            resolved[key] = _repo_file(value, repo_root, f"inputs.{key}")
    robot_value = inputs.get("robot_urdf")
    if not isinstance(robot_value, str) or not robot_value:
        raise ValueError("inputs.robot_urdf is required")
    resolved["robot_urdf"] = _repo_file(
        robot_value, repo_root, "inputs.robot_urdf"
    )
    supporting_files = inputs.get("supporting_files", [])
    if not isinstance(supporting_files, list) or not all(
        isinstance(value, str) and value for value in supporting_files
    ):
        raise ValueError("inputs.supporting_files must be an array of path strings")
    for index, value in enumerate(supporting_files):
        resolved[f"supporting_files[{index}]"] = _repo_file(
            value, repo_root, f"inputs.supporting_files[{index}]"
        )
    acceptance = spec.get("acceptance")
    if not isinstance(acceptance, dict):
        raise ValueError("acceptance must be an object")
    for key in (
        "require_zero_fixed_constraints",
        "minimum_joint_motion_ratio",
        "visual_review_fps",
    ):
        if key not in acceptance:
            raise ValueError(f"acceptance.{key} is required")
    output_dir = spec.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir is required")
    _repo_output(output_dir, repo_root)
    return resolved


def _workflow_files(repo_root: Path) -> list[Path]:
    candidates = [
        DOCS_ROOT / "workflow.md",
        DOCS_ROOT / "acceptance-contract.md",
        DOCS_ROOT / "failure-playbook.md",
        DOCS_ROOT / "obstacle-avoidance-workflow.md",
        repo_root / "AGENTS.md",
        repo_root / "README.md",
    ]
    candidates.extend(
        path
        for path in APPLICATION_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    )
    return sorted(set(path for path in candidates if path.is_file()))


def build(
    spec_path: Path, repo_root: Path = REPO_ROOT
) -> tuple[dict[str, Any], dict[str, Any], str]:
    repo_root = repo_root.resolve()
    spec_path = _repo_file(str(spec_path), repo_root, "task spec")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise ValueError("task spec root must be an object")
    resolved_inputs = _validate_spec(spec, repo_root)

    input_records = {
        key: _record(path, repo_root)
        for key, path in sorted(resolved_inputs.items())
    }
    urdfs = [
        path for key, path in resolved_inputs.items()
        if key in {"urdf", "physics_urdf", "robot_urdf"}
    ]
    dependency_records = _asset_dependency_records(urdfs, repo_root)
    workflow_records = {
        str(path.relative_to(repo_root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(_workflow_files(repo_root))
    }
    base_lock: dict[str, Any] = {
        "schema_version": 1,
        "task_id": spec["task_id"],
        "task_spec": {
            "path": str(spec_path.relative_to(repo_root)),
            "bytes": spec_path.stat().st_size,
            "sha256": _sha256(spec_path),
        },
        "inputs": input_records,
        "asset_dependencies": dependency_records,
        "workflow": workflow_records,
        "toolchain": _toolchain_state(),
        "output_dir": str(_repo_output(spec["output_dir"], repo_root).relative_to(repo_root)),
        "git": _git_state(repo_root),
    }
    lock_sha256 = hashlib.sha256(_canonical_bytes(base_lock)).hexdigest()
    prompt_lock = {**base_lock, "lock_sha256": lock_sha256}
    prompt = (
        "Before doing any task work, read these application documents completely:\n"
        "- `applications/artimo_robot_contact/docs/workflow.md`\n"
        "- `applications/artimo_robot_contact/docs/acceptance-contract.md`\n"
        "- `applications/artimo_robot_contact/docs/failure-playbook.md`\n"
        "- `applications/artimo_robot_contact/docs/obstacle-avoidance-workflow.md` "
        "when a later robot-contact stage must pass geometry moved earlier.\n\n"
        "These files are the robot-contact application instructions; do not substitute "
        "the repository-level `docs/` directory. Follow AGENTS.md. Do not "
        "read prior task outputs or edit generic code, schemas, application docs, source "
        "inputs, requirements, or AGENTS.md during the task run.\n\n"
        "`plan.json` is the sole authority for phase names, controls, targets, holds, "
        "returns, ordering, and release/contact-continuity boundaries. Do not open, "
        "read, or use any supporting file named `causal.json`; it may appear in the "
        "task specification and immutable lock only so its supplied bytes remain "
        "auditable. No supporting file or task description may create a stage, change "
        "a contact link, change a motion owner, or release a grasp absent an explicit "
        "boundary in `plan.json`. Consecutive robot-contact controls with no intervening "
        "plan `control_release` must use the same contact link and one uninterrupted "
        "`contact_sequence`, even when the controlled joint changes.\n\n"
        "This task has no elapsed-time, wall-clock, tool-window, or compute-time "
        "budget. Runtime may change only how a long command is launched and polled; "
        "it must never narrow, subsample, terminate, or reorder a declared candidate "
        "batch, skip the lateral-offset batch after all centered distances fail, or "
        "promote a failed row. If a command is interrupted, rerun or resume the "
        "byte-identical batch and wait for completion. A placement result with "
        "`execution: null` is diagnostic evidence only: never copy its `chosen` row "
        "into an execution, release solver, physics rollout, or video. The instruction "
        "to export a complete video when diagnostics fail applies only after the "
        "placement solver has emitted its feasible `execution.json`.\n\n"
        f"Task specification: `{spec_path.relative_to(repo_root)}`\n\n"
        "Immutable handoff lock:\n\n```json\n"
        + json.dumps(prompt_lock, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n```\n\n"
        "Create only task-specific execution data below this task's `.artimo-runs/` "
        "tree. Preserve every plan control and physical owner, use the generic "
        "orientation/placement/release/transit tools as applicable, run physical "
        "and byte-identical contact-disabled conditions once, perform video review, "
        "then publish exactly `video.mp4`, `grasp.json`, and "
        "`result.json`. Export the complete video even when diagnostics fail.\n"
    )
    lock = {
        **prompt_lock,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    }
    return spec, lock, prompt


def write_outputs(
    spec_path: Path,
    prompt_out: Path,
    lock_out: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    spec, lock, prompt = build(spec_path, repo_root)
    prompt_out.parent.mkdir(parents=True, exist_ok=True)
    lock_out.parent.mkdir(parents=True, exist_ok=True)
    prompt_out.write_text(prompt, encoding="utf-8")
    lock_out.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "task_id": spec["task_id"],
        "prompt": str(prompt_out),
        "input_lock": str(lock_out),
        "lock_sha256": lock["lock_sha256"],
        "prompt_sha256": lock["prompt_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--prompt-out", type=Path, required=True)
    parser.add_argument("--lock-out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    summary = write_outputs(
        args.spec.expanduser(),
        args.prompt_out.expanduser(),
        args.lock_out.expanduser(),
        args.repo_root.expanduser().resolve(),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
