#!/usr/bin/env python3
import atexit
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image


def _env_int(name: str, default: int, *, aliases: list[str] | None = None) -> int:
    for key in [name] + list(aliases or []):
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return int(default)


# DEFAULT_DEVICE_TYPE = os.environ.get("CODEX_BLENDER_DEVICE_TYPE", "CUDA")
DEFAULT_DEVICE_TYPE = os.environ.get("CODEX_BLENDER_DEVICE_TYPE", "OPTIX")

DEFAULT_GPU_INDEX = int(os.environ.get("CODEX_BLENDER_GPU_INDEX", "0"))
# Primary Blender quality knob: Cycles sample count per render.
DEFAULT_SAMPLES = _env_int("CODEX_BLENDER_SAMPLES", 32, aliases=["CODEX_BLENDER_RENDER_SAMPLES"])
# Secondary speed/throughput knob for Cycles batch rendering.
DEFAULT_TILE_SIZE = _env_int("CODEX_BLENDER_TILE_SIZE", 256)
DEFAULT_DENOISE = os.environ.get("CODEX_BLENDER_DENOISE", "0") not in {"0", "false", "False"}
DEFAULT_CAMERA_AZIMUTH_OFFSET_DEG = float(os.environ.get("CODEX_BLENDER_CAMERA_AZIMUTH_OFFSET_DEG", "0.0"))
LOWMEM_RETRY_SAMPLES_L1 = _env_int("CODEX_BLENDER_LOWMEM_RETRY_SAMPLES_L1", 32)
LOWMEM_RETRY_TILE_SIZE_L1 = _env_int("CODEX_BLENDER_LOWMEM_RETRY_TILE_SIZE_L1", 128)
LOWMEM_RETRY_SAMPLES_L2 = _env_int("CODEX_BLENDER_LOWMEM_RETRY_SAMPLES_L2", 16)
LOWMEM_RETRY_TILE_SIZE_L2 = _env_int("CODEX_BLENDER_LOWMEM_RETRY_TILE_SIZE_L2", 64)
_SEEN_DEVICE_LOGS: set[str] = set()
PERSISTENT_WORKER_API_VERSION = 2
_ACTIVE_WORKER_SOCKETS: set[Path] = set()
_WORKER_CLEANUP_REGISTERED = False


def _shutdown_persistent_blender_worker(socket_path: Path) -> bool:
    try:
        if not socket_path.exists():
            return False
        _worker_request(socket_path, {"op": "shutdown"}, timeout_s=1.0)
        return True
    except Exception:
        return False


def shutdown_all_persistent_blender_workers() -> None:
    global _ACTIVE_WORKER_SOCKETS
    for socket_path in list(_ACTIVE_WORKER_SOCKETS):
        try:
            _shutdown_persistent_blender_worker(socket_path)
        except Exception:
            pass
    _ACTIVE_WORKER_SOCKETS.clear()


def _register_worker_cleanup_once() -> None:
    global _WORKER_CLEANUP_REGISTERED
    if _WORKER_CLEANUP_REGISTERED:
        return
    _WORKER_CLEANUP_REGISTERED = True
    atexit.register(shutdown_all_persistent_blender_workers)

    def _handler(signum, _frame):
        shutdown_all_persistent_blender_workers()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt()
        raise SystemExit(128 + int(signum))

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            continue


def _parse_basis_matrix() -> np.ndarray:
    raw = os.environ.get("CODEX_WORLD_TO_BLENDER_MATRIX")
    if raw:
        rows = []
        for row in raw.split(";"):
            vals = [float(x.strip()) for x in row.split(",") if x.strip()]
            if len(vals) != 3:
                raise ValueError(
                    "CODEX_WORLD_TO_BLENDER_MATRIX expects 3 rows of 3 comma-separated floats "
                    '(example: "1,0,0;0,0,-1;0,1,0").'
                )
            rows.append(vals)
        if len(rows) != 3:
            raise ValueError("CODEX_WORLD_TO_BLENDER_MATRIX must contain exactly 3 rows.")
        return np.asarray(rows, dtype=float)
    # Blender glTF importer applies axis conversion equivalent to Rx(+90deg):
    # (x_w, y_w, z_w) -> (x_b, -z_w, y_w)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )


def _camera_world_to_blender(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    return _parse_basis_matrix() @ v


def world_row_transform_to_blender(mat_rows) -> list[list[float]]:
    """Convert a row-vector world transform into Blender column-vector space."""
    tf = np.asarray(mat_rows, dtype=float).reshape(4, 4)
    linear_w = tf[:3, :3]
    trans_w = tf[:3, 3]
    basis = _parse_basis_matrix()
    basis_inv = np.linalg.inv(basis)
    out = np.eye(4, dtype=float)
    out[:3, :3] = basis @ linear_w @ basis_inv
    out[:3, 3] = basis @ trans_w
    return [[float(x) for x in row] for row in out.tolist()]


def _apply_camera_azimuth_offset(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if abs(DEFAULT_CAMERA_AZIMUTH_OFFSET_DEG) <= 1e-9:
        return eye, up
    theta = np.deg2rad(DEFAULT_CAMERA_AZIMUTH_OFFSET_DEG)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    yaw = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    eye_rel = eye - target
    eye_new = target + yaw @ eye_rel
    up_new = yaw @ up
    return eye_new, up_new


def _convert_view_for_blender(view: dict) -> dict:
    eye = _camera_world_to_blender(view["eye"])
    target = _camera_world_to_blender(view["target"])
    up = _camera_world_to_blender(view["up"])
    eye, up = _apply_camera_azimuth_offset(eye, target, up)
    out = dict(view)
    out["eye"] = [float(x) for x in eye.tolist()]
    out["target"] = [float(x) for x in target.tolist()]
    out["up"] = [float(x) for x in up.tolist()]
    return out


def find_blender_exe() -> str | None:
    env_candidates = [
        os.environ.get("BLENDER_BIN"),
        os.environ.get("BLENDER_PATH"),
    ]
    local_candidates = [
        Path(__file__).resolve().parent.parent / "blender-4.2.0-linux-x64" / "blender",
        Path(__file__).resolve().parent.parent / "blender" / "blender",
    ]
    for cand in env_candidates:
        if cand and Path(cand).exists():
            return str(Path(cand).resolve())
    for cand in local_candidates:
        if cand.exists():
            return str(cand.resolve())
    found = shutil.which("blender")
    return found


def _persistent_blender_enabled() -> bool:
    return _bool_env("CODEX_BLENDER_PERSISTENT", default=True)


def _worker_socket_path(gpu_index: int | None, device_type: str) -> Path:
    gi = "auto" if gpu_index is None else str(int(gpu_index))
    dev = str(device_type or DEFAULT_DEVICE_TYPE).strip().lower()
    return Path(tempfile.gettempdir()) / f"cbw_v{PERSISTENT_WORKER_API_VERSION}_{os.getuid()}_{gi}_{dev}.sock"


def _worker_log_path(gpu_index: int | None, device_type: str) -> Path:
    gi = "auto" if gpu_index is None else str(int(gpu_index))
    dev = str(device_type or DEFAULT_DEVICE_TYPE).strip().lower()
    return Path(tempfile.gettempdir()) / f"cbw_v{PERSISTENT_WORKER_API_VERSION}_{os.getuid()}_{gi}_{dev}.log"


def _worker_request(socket_path: Path, payload: dict, *, timeout_s: float = 10.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(float(timeout_s))
        sock.connect(str(socket_path))
        sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
            if b"\n" in data:
                break
    raw = b"".join(chunks).decode("utf-8", errors="ignore").strip()
    if not raw:
        raise RuntimeError("empty response from Blender worker")
    line = raw.splitlines()[0]
    return json.loads(line)


def _worker_is_ready(socket_path: Path) -> bool:
    if not socket_path.exists():
        return False
    try:
        resp = _worker_request(socket_path, {"op": "ping"}, timeout_s=0.5)
        return bool(resp.get("ok", False))
    except Exception:
        return False


def _launch_persistent_blender_worker(
    *,
    gpu_index: int | None,
    device_type: str,
) -> Path:
    _register_worker_cleanup_once()
    blender = find_blender_exe()
    if not blender:
        raise RuntimeError("Blender executable not found")
    socket_path = _worker_socket_path(gpu_index, device_type)
    log_path = _worker_log_path(gpu_index, device_type)
    script_path = Path(__file__).with_name("blender_render_worker.py")
    try:
        socket_path.unlink()
    except Exception:
        pass
    env = _build_blender_env(force_cpu=_default_force_cpu(), gpu_index=gpu_index)
    log_f = open(log_path, "ab", buffering=0)
    cmd = [
        blender,
        "-b",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        "--socket",
        str(socket_path),
        "--device_type",
        str(device_type),
        "--gpu_index",
        str(int(gpu_index if gpu_index is not None else 0)),
    ]
    subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    log_f.close()
    return socket_path


def _ensure_persistent_blender_worker(
    *,
    gpu_index: int | None,
    device_type: str,
) -> Path:
    _register_worker_cleanup_once()
    socket_path = _worker_socket_path(gpu_index, device_type)
    if _worker_is_ready(socket_path):
        _ACTIVE_WORKER_SOCKETS.add(socket_path)
        return socket_path
    socket_path = _launch_persistent_blender_worker(gpu_index=gpu_index, device_type=device_type)
    deadline = time.perf_counter() + 45.0
    while time.perf_counter() < deadline:
        if _worker_is_ready(socket_path):
            _ACTIVE_WORKER_SOCKETS.add(socket_path)
            print(
                "[INFO] Persistent Blender worker ready "
                f"(gpu_index={gpu_index if gpu_index is not None else 'auto'}, device_type={device_type})."
            )
            return socket_path
        time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for Blender worker at {socket_path}")


def _build_blender_env(force_cpu: bool, gpu_index: int | None) -> dict[str, str]:
    env = os.environ.copy()
    respect_visible = _bool_env("CODEX_BLENDER_RESPECT_VISIBLE_DEVICES", default=True)
    multi_gpu = _bool_env("CODEX_BLENDER_MULTI_GPU", default=False)
    if force_cpu:
        env["CODEX_BLENDER_FORCE_CPU"] = "1"
    else:
        env.pop("CODEX_BLENDER_FORCE_CPU", None)
        has_external_visible = bool(str(env.get("CUDA_VISIBLE_DEVICES", "")).strip()) or bool(
            str(env.get("OPTIX_VISIBLE_DEVICES", "")).strip()
        )
        if multi_gpu:
            env["CODEX_BLENDER_MULTI_GPU"] = "1"
            # Optional explicit GPU list like "0,1".
            gpu_list = str(env.get("CODEX_BLENDER_GPU_LIST", "")).strip()
            if gpu_list:
                env["CUDA_VISIBLE_DEVICES"] = gpu_list
                env["OPTIX_VISIBLE_DEVICES"] = gpu_list
        else:
            if gpu_index is not None and not (respect_visible and has_external_visible):
                env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
                env["OPTIX_VISIBLE_DEVICES"] = str(int(gpu_index))
    return env


def _default_force_cpu() -> bool:
    # GPU-first default: try GPU unless CPU is explicitly requested.
    # _run_blender() already falls back to CPU on common GPU failures.
    if _bool_env("CODEX_BLENDER_FORCE_CPU", default=False):
        return True
    if _bool_env("CODEX_BLENDER_USE_GPU", default=False):
        return False
    return _bool_env("CODEX_BLENDER_DEFAULT_CPU", default=False)


def _strict_gpu_required(force_cpu: bool) -> bool:
    if force_cpu:
        return False
    if _bool_env("CODEX_BLENDER_STRICT_GPU", default=False):
        return True
    if _bool_env("CODEX_BLENDER_USE_GPU", default=False) and not _bool_env("CODEX_BLENDER_ALLOW_CPU_FALLBACK", default=False):
        return True
    return False


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _decode_process_output(proc) -> tuple[str, str]:
    stdout_raw = getattr(proc, "stdout", None)
    stderr_raw = getattr(proc, "stderr", None)
    if isinstance(stdout_raw, bytes):
        stdout = stdout_raw.decode("utf-8", errors="ignore")
    else:
        stdout = str(stdout_raw or "")
    if isinstance(stderr_raw, bytes):
        stderr = stderr_raw.decode("utf-8", errors="ignore")
    else:
        stderr = str(stderr_raw or "")
    return stdout, stderr


def _extract_device_used(stdout: str, stderr: str = "") -> str | None:
    for text in (stdout, stderr):
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("CODEX_BLENDER_DEVICE_USED="):
                continue
            val = line.split("=", 1)[1].strip()
            if val:
                return val
    return None


def _log_device_used(device: str | None) -> None:
    if not device:
        return
    key = str(device).strip().upper()
    if not key or key in _SEEN_DEVICE_LOGS:
        return
    _SEEN_DEVICE_LOGS.add(key)
    print(f"[INFO] Blender Cycles device in use: {key}")


def _relay_blender_markers(text: str) -> None:
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("CODEX_BLENDER_DEVICE_USED="):
            val = line.split("=", 1)[1].strip()
            if val:
                print(f"[INFO] Blender child selected device: {val}")
        elif line.startswith("CODEX_BLENDER_TIMING="):
            val = line.split("=", 1)[1].strip()
            if val:
                print(f"[INFO] Blender timing {val}")


def _run_and_collect_blender_output(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        _relay_blender_markers(line)
    rc = proc.wait()
    return rc, "".join(lines)


def _is_gpu_oom(msg: str) -> bool:
    s = str(msg or "").lower()
    return any(
        key in s
        for key in [
            "out of gpu memory",
            "out of gpu and shared host memory",
            "failed to build optix acceleration structure",
            "launch failed in custreamsynchronize",
            "failed to retain cuda context",
        ]
    )


def _with_or_append_arg(cmd: list[str], name: str, value: str) -> list[str]:
    out = list(cmd)
    for i, tok in enumerate(out):
        if tok == name and i + 1 < len(out):
            out[i + 1] = str(value)
            return out
    out.extend([name, str(value)])
    return out


def _remove_flag(cmd: list[str], name: str) -> list[str]:
    out = []
    skip = False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok == name:
            continue
        out.append(tok)
    return out


def _lowmem_retry_cmd(cmd: list[str], level: int) -> list[str]:
    out = list(cmd)
    # Always disable denoiser under low-memory retry first.
    out = _remove_flag(out, "--denoise")
    if int(level) <= 1:
        out = _with_or_append_arg(out, "--samples", str(int(LOWMEM_RETRY_SAMPLES_L1)))
        out = _with_or_append_arg(out, "--tile_size", str(int(LOWMEM_RETRY_TILE_SIZE_L1)))
        return out
    # Level-2: more aggressive; switch OptiX -> CUDA and shrink settings further.
    out = _with_or_append_arg(out, "--device_type", "CUDA")
    out = _with_or_append_arg(out, "--samples", str(int(LOWMEM_RETRY_SAMPLES_L2)))
    out = _with_or_append_arg(out, "--tile_size", str(int(LOWMEM_RETRY_TILE_SIZE_L2)))
    return out


def _run_blender(cmd: list[str], *, gpu_index: int | None = DEFAULT_GPU_INDEX) -> None:
    force_cpu = _default_force_cpu()
    env = _build_blender_env(force_cpu=force_cpu, gpu_index=gpu_index)
    strict_gpu = _strict_gpu_required(force_cpu)
    t0 = time.perf_counter()
    try:
        print(
            "[INFO] Launching Blender render "
            f"(gpu_index={gpu_index if gpu_index is not None else 'auto'}, "
            f"force_cpu={int(force_cpu)}, strict_gpu={int(strict_gpu)})."
        )
        rc, combined = _run_and_collect_blender_output(cmd, env)
        stdout, stderr = combined, ""
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd, output=combined, stderr="")
        device = _extract_device_used(stdout, stderr)
        _log_device_used(device)
        if strict_gpu and (device is None or str(device).upper().startswith("CPU")):
            raise RuntimeError(
                "Strict GPU render is enabled but Blender did not use GPU "
                f"(detected device: {device or 'UNKNOWN'})."
            )
        print(f"[INFO] Blender render subprocess finished in {time.perf_counter() - t0:.2f}s.")
    except subprocess.CalledProcessError as exc:
        stdout, stderr = _decode_process_output(exc)
        msg_full = "\n".join(part for part in [stdout, stderr] if part)
        if _is_gpu_oom(msg_full):
            retry_errors = []
            for lvl in (1, 2):
                retry_cmd = _lowmem_retry_cmd(cmd, lvl)
                try:
                    proc_retry = subprocess.run(
                        retry_cmd,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        env=env,
                    )
                    r_stdout, r_stderr = _decode_process_output(proc_retry)
                    retry_device = _extract_device_used(r_stdout, r_stderr)
                    _log_device_used(retry_device)
                    if strict_gpu and (retry_device is None or str(retry_device).upper().startswith("CPU")):
                        raise RuntimeError(
                            "Strict GPU render is enabled but Blender low-memory retry did not use GPU "
                            f"(detected device: {retry_device or 'UNKNOWN'})."
                        )
                    print(
                        f"[INFO] Blender GPU low-memory retry succeeded at level {lvl} "
                        f"(samples/tile/device adjusted)."
                    )
                    return
                except subprocess.CalledProcessError as retry_exc:
                    r_stdout, r_stderr = _decode_process_output(retry_exc)
                    retry_errors.append(
                        "\n".join(
                            x
                            for x in [
                                f"[retry level {lvl}]",
                                r_stdout[-1200:] if r_stdout else "",
                                r_stderr[-2200:] if r_stderr else "",
                            ]
                            if x
                        )
                    )
            if strict_gpu:
                msg = "\n".join(
                    x
                    for x in [
                        "Blender GPU render failed after low-memory retries.",
                        stdout[-1200:] if stdout else "",
                        stderr[-2200:] if stderr else "",
                        "\n".join(retry_errors),
                    ]
                    if x
                )
                raise RuntimeError(msg) from exc
        gpu_failed = any(
            key in msg_full
            for key in [
                "Failed to retain CUDA context",
                "OPTIX",
                "CUDA error",
                "Failed to create CUDA context",
            ]
        )
        if strict_gpu:
            msg = "\n".join(
                part
                for part in [
                    "Blender render failed under strict GPU mode.",
                    stdout[-2000:] if stdout else "",
                    stderr[-4000:] if stderr else "",
                ]
                if part
            )
            raise RuntimeError(msg) from exc
        if gpu_failed and env.get("CODEX_BLENDER_FORCE_CPU") != "1":
            env_cpu = _build_blender_env(force_cpu=True, gpu_index=None)
            try:
                proc_cpu = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env_cpu)
                cpu_stdout, cpu_stderr = _decode_process_output(proc_cpu)
                _log_device_used(_extract_device_used(cpu_stdout, cpu_stderr))
                return
            except subprocess.CalledProcessError as exc_cpu:
                stdout, stderr = _decode_process_output(exc_cpu)
        msg = "\n".join(
            part for part in [
                "Blender render failed.",
                stdout[-2000:] if stdout else "",
                stderr[-4000:] if stderr else "",
            ] if part
        )
        raise RuntimeError(msg) from exc


def render_views_from_scene(
    scene,
    views: list[dict],
    resolution: tuple[int, int],
    *,
    fov_deg: float = 50.0,
    gpu_index: int | None = DEFAULT_GPU_INDEX,
    device_type: str = DEFAULT_DEVICE_TYPE,
    samples: int = DEFAULT_SAMPLES,
    tile_size: int = DEFAULT_TILE_SIZE,
    denoise: bool = DEFAULT_DENOISE,
) -> list[np.ndarray]:
    with tempfile.NamedTemporaryFile(prefix="blender_scene_", suffix=".glb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        scene.export(tmp_path)
        return render_views_from_glb(
            tmp_path,
            views,
            resolution,
            fov_deg=fov_deg,
            gpu_index=gpu_index,
            device_type=device_type,
            samples=samples,
            tile_size=tile_size,
            denoise=denoise,
        )
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


def render_views_from_glb(
    glb_path: str | Path,
    views: list[dict],
    resolution: tuple[int, int],
    *,
    fov_deg: float = 50.0,
    node_transforms: dict | None = None,
    global_transform: list[list[float]] | None = None,
    frame_idx: int | None = None,
    fps: int | None = None,
    keep_animation: bool = False,
    gpu_index: int | None = DEFAULT_GPU_INDEX,
    device_type: str = DEFAULT_DEVICE_TYPE,
    samples: int = DEFAULT_SAMPLES,
    tile_size: int = DEFAULT_TILE_SIZE,
    denoise: bool = DEFAULT_DENOISE,
) -> list[np.ndarray]:
    if not views:
        return []
    with tempfile.TemporaryDirectory(prefix="blender_views_") as tmpd:
        tmpd_path = Path(tmpd)
        payload_views = []
        out_paths = []
        for idx, view in enumerate(views):
            v = _convert_view_for_blender(view)
            out_path = tmpd_path / f"view_{idx:02d}.png"
            out_paths.append(out_path)
            frame_idx_local = v.get("frame_idx", frame_idx)
            payload_views.append(
                {
                    "id": str(v.get("id", f"V{idx+1}")),
                    "eye": [float(x) for x in np.asarray(v["eye"], dtype=float).tolist()],
                    "target": [float(x) for x in np.asarray(v["target"], dtype=float).tolist()],
                    "up": [float(x) for x in np.asarray(v["up"], dtype=float).tolist()],
                    "out_path": str(out_path),
                    "frame_idx": (None if frame_idx_local is None else int(frame_idx_local)),
                }
            )
        payload = {"views": payload_views}
        views_json = tmpd_path / "views.json"
        views_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tf_json = tmpd_path / "transforms.json"
        tf_json.write_text(
            json.dumps(
                {"node_transforms": node_transforms or {}, "global_transform": global_transform},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        worker_payload = {
            "op": "render_views",
            "glb_path": str(Path(glb_path).resolve()),
            "views": payload_views,
            "node_transforms": node_transforms or {},
            "global_transform": global_transform,
            "width": int(resolution[0]),
            "height": int(resolution[1]),
            "fov_deg": float(fov_deg),
            "device_type": str(device_type),
            "gpu_index": int(gpu_index if gpu_index is not None else 0),
            "samples": int(samples),
            "tile_size": int(tile_size),
            "denoise": bool(denoise),
            "keep_animation": bool(keep_animation),
            "frame_idx": (None if frame_idx is None else int(frame_idx)),
            "fps": (None if fps is None else int(fps)),
        }
        if _persistent_blender_enabled():
            try:
                socket_path = _ensure_persistent_blender_worker(gpu_index=gpu_index, device_type=device_type)
                resp = _worker_request(socket_path, worker_payload, timeout_s=600.0)
                if not bool(resp.get("ok", False)):
                    raise RuntimeError(str(resp.get("error") or "persistent Blender worker request failed"))
                device = str(resp.get("device_used") or "").strip()
                _log_device_used(device)
                if _strict_gpu_required(_default_force_cpu()) and (not device or device.upper().startswith("CPU")):
                    raise RuntimeError(
                        "Strict GPU render is enabled but persistent Blender worker did not use GPU "
                        f"(detected device: {device or 'UNKNOWN'})."
                    )
                images = []
                for out_path in out_paths:
                    if not out_path.exists():
                        raise RuntimeError(f"Missing persistent Blender render output: {out_path}")
                    images.append(np.array(Image.open(out_path).convert("RGB"), copy=True))
                return images
            except Exception as exc:
                print(f"[WARN] Persistent Blender worker failed ({exc}); falling back to one-shot Blender subprocess.")
        blender = find_blender_exe()
        if not blender:
            raise RuntimeError("Blender executable not found")
        script_path = Path(__file__).with_name("blender_render_views.py")
        cmd = [
            blender,
            "-b",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            "--glb",
            str(Path(glb_path).resolve()),
            "--views_json",
            str(views_json),
            "--transforms_json",
            str(tf_json),
            "--width",
            str(int(resolution[0])),
            "--height",
            str(int(resolution[1])),
            "--fov_deg",
            str(float(fov_deg)),
            "--device_type",
            str(device_type),
            "--gpu_index",
            str(int(gpu_index if gpu_index is not None else 0)),
            "--samples",
            str(int(samples)),
            "--tile_size",
            str(int(tile_size)),
        ]
        if denoise:
            cmd.append("--denoise")
        if bool(keep_animation):
            cmd.append("--keep_animation")
        if frame_idx is not None:
            cmd.extend(["--frame_idx", str(int(frame_idx))])
        if fps is not None:
            cmd.extend(["--fps", str(int(fps))])
        _run_blender(cmd, gpu_index=gpu_index)
        images = []
        for out_path in out_paths:
            if not out_path.exists():
                raise RuntimeError(f"Missing Blender render output: {out_path}")
            images.append(np.array(Image.open(out_path).convert("RGB"), copy=True))
        return images


def render_animation_sequence_from_glb(
    glb_path: str | Path,
    out_dir: str | Path,
    resolution: tuple[int, int],
    camera: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    frame_count: int,
    fps: int,
    fov_deg: float = 45.0,
    gpu_index: int | None = DEFAULT_GPU_INDEX,
    device_type: str = DEFAULT_DEVICE_TYPE,
    samples: int = DEFAULT_SAMPLES,
    tile_size: int = DEFAULT_TILE_SIZE,
    denoise: bool = DEFAULT_DENOISE,
) -> None:
    blender = find_blender_exe()
    if not blender:
        raise RuntimeError("Blender executable not found")
    if frame_count <= 0:
        raise RuntimeError("frame_count must be positive")
    script_path = Path(__file__).with_name("blender_render_animation.py")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    eye, target, up = camera
    eye = _camera_world_to_blender(np.asarray(eye, dtype=float))
    target = _camera_world_to_blender(np.asarray(target, dtype=float))
    up = _camera_world_to_blender(np.asarray(up, dtype=float))
    eye, up = _apply_camera_azimuth_offset(eye, target, up)
    with tempfile.TemporaryDirectory(prefix="blender_anim_") as tmpd:
        camera_json = Path(tmpd) / "camera.json"
        camera_json.write_text(
            json.dumps(
                {
                    "eye": [float(x) for x in np.asarray(eye, dtype=float).tolist()],
                    "target": [float(x) for x in np.asarray(target, dtype=float).tolist()],
                    "up": [float(x) for x in np.asarray(up, dtype=float).tolist()],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        cmd = [
            blender,
            "-b",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            "--glb",
            str(Path(glb_path).resolve()),
            "--out_dir",
            str(out_dir),
            "--camera_json",
            str(camera_json),
            "--width",
            str(int(resolution[0])),
            "--height",
            str(int(resolution[1])),
            "--frame_start",
            "0",
            "--frame_end",
            str(int(frame_count) - 1),
            "--frame_offset",
            "0",
            "--fps",
            str(int(fps)),
            "--fov_deg",
            str(float(fov_deg)),
            "--device_type",
            str(device_type),
            "--gpu_index",
            str(int(gpu_index if gpu_index is not None else 0)),
            "--samples",
            str(int(samples)),
            "--tile_size",
            str(int(tile_size)),
        ]
        if denoise:
            cmd.append("--denoise")
        _run_blender(cmd, gpu_index=gpu_index)
    for frame_idx in range(int(frame_count)):
        frame_path = out_dir / f"frame_{frame_idx:04d}.png"
        if not frame_path.exists():
            raise RuntimeError(f"Missing Blender animation frame: {frame_path}")
