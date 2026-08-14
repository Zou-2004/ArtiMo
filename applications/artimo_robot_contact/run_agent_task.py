#!/usr/bin/env python3
"""Prepare or launch one agent-neutral frozen ArtiMo robot-contact task."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import IO


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        candidate = root / f"attempt_{index:03d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(f"Too many attempts under {root}")


def _pump(source: IO[str], destination: IO[str], mirror: IO[str]) -> None:
    for line in iter(source.readline, ""):
        destination.write(line)
        destination.flush()
        mirror.write(line)
        mirror.flush()


def _agent_text_stream_options() -> dict[str, object]:
    """Return deterministic decoding for the Codex CLI text protocol.

    Windows otherwise uses the active ANSI code page (commonly GBK). Codex
    emits UTF-8, so one non-GBK byte can kill a pump thread, stop draining the
    pipe, and deadlock the agent once the OS pipe buffer fills.
    """
    return {
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }


def _source_guard_snapshot() -> dict[str, str]:
    """Hash executable/instruction sources without embedding legacy files in prompts."""
    roots = [APP_ROOT]
    paths = [REPO_ROOT / "AGENTS.md"]
    for root in roots:
        if root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    answer: dict[str, str] = {}
    for path in sorted(set(paths)):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        answer[str(path.relative_to(REPO_ROOT))] = digest
    return answer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Permit a repair run against an existing task output directory",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Freeze inputs and write the exact prompt without invoking an agent",
    )
    parser.add_argument(
        "--agent-command",
        help="Agent CLI command that reads the exact handoff prompt from stdin",
    )
    args = parser.parse_args()

    renderer = _load_module("artimo_prompt_renderer", APP_ROOT / "render_task_prompt.py")
    verifier = _load_module("artimo_delivery_verifier", APP_ROOT / "verify_delivery.py")
    spec_path = args.spec.expanduser()
    if not spec_path.is_absolute():
        spec_path = (REPO_ROOT / spec_path).resolve()
    task_spec, lock, prompt = renderer.build(spec_path, REPO_ROOT)
    source_guard = _source_guard_snapshot()
    run_root = (
        REPO_ROOT
        / ".artimo-runs"
        / task_spec["task_id"]
        / lock["lock_sha256"][:16]
    )
    attempt_dir = _next_attempt(run_root)
    prompt_path = attempt_dir / "prompt.md"
    lock_path = attempt_dir / "input-lock.json"
    prompt_path.write_text(prompt, encoding="utf-8")
    lock_path.write_text(
        json.dumps(lock, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.prepare_only or not args.agent_command:
        print(
            json.dumps(
                {
                    "task_id": task_spec["task_id"],
                    "prompt": str(prompt_path.relative_to(REPO_ROOT)),
                    "input_lock": str(lock_path.relative_to(REPO_ROOT)),
                    "lock_sha256": lock["lock_sha256"],
                    "prompt_sha256": lock["prompt_sha256"],
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    output_dir = renderer._repo_output(task_spec["output_dir"], REPO_ROOT)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output:
        raise RuntimeError(
            f"Refusing non-empty output {output_dir}; use a new path or "
            "--allow-existing-output for an explicit repair run"
        )
    command = shlex.split(args.agent_command)
    if not command:
        raise ValueError("--agent-command must contain an executable")
    (attempt_dir / "command.json").write_text(
        json.dumps(command, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (attempt_dir / "agent-stdout.log").open("w", encoding="utf-8") as stdout_log, (
        attempt_dir / "stderr.log"
    ).open("w", encoding="utf-8") as stderr_log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **_agent_text_stream_options(),
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_thread = threading.Thread(
            target=_pump, args=(process.stdout, stdout_log, sys.stdout), daemon=True
        )
        stderr_thread = threading.Thread(
            target=_pump, args=(process.stderr, stderr_log, sys.stderr), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()
        process.stdin.write(prompt)
        process.stdin.close()
        returncode = process.wait()
        stdout_thread.join()
        stderr_thread.join()
    if returncode != 0:
        print(f"Agent exited with status {returncode}; logs: {attempt_dir}", file=sys.stderr)
        return returncode

    if source_guard != _source_guard_snapshot():
        before = set(source_guard.items())
        after = set(_source_guard_snapshot().items())
        changed = sorted({path for path, _ in before.symmetric_difference(after)})
        raise RuntimeError(
            "Agent changed frozen application or skill files; per-task runs are "
            f"config-only. Changed paths: {changed[:20]}"
        )

    _release_spec, release_lock, release_prompt = renderer.build(spec_path, REPO_ROOT)
    for immutable_key in (
        "task_spec",
        "inputs",
        "asset_dependencies",
        "workflow",
        "toolchain",
        "output_dir",
    ):
        if lock[immutable_key] != release_lock[immutable_key]:
            raise RuntimeError(
                f"Agent changed immutable handoff field {immutable_key}; "
                f"evidence is in {attempt_dir}"
            )
    (attempt_dir / "release-prompt.md").write_text(release_prompt, encoding="utf-8")
    (attempt_dir / "release-lock.json").write_text(
        json.dumps(release_lock, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = verifier.verify(
        spec_path,
        REPO_ROOT,
        expected_release_lock_sha256=release_lock["lock_sha256"],
        expected_handoff_lock_sha256=lock["lock_sha256"],
    )
    report_path = attempt_dir / "verification.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "task_id": task_spec["task_id"],
                "passed": report["passed"],
                "output_dir": task_spec["output_dir"],
                "run_evidence": str(attempt_dir.relative_to(REPO_ROOT)),
                "verification": str(report_path.relative_to(REPO_ROOT)),
                "handoff_lock_sha256": lock["lock_sha256"],
                "handoff_prompt_sha256": lock["prompt_sha256"],
                "release_lock_sha256": release_lock["lock_sha256"],
                "release_prompt_sha256": release_lock["prompt_sha256"],
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    # Verification is retained as diagnostic evidence.  A completed agent run
    # with exported artifacts must not be turned into a stopped task merely
    # because a diagnostic boolean is false.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
