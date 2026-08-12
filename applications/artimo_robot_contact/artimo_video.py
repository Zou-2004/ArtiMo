"""Hardware-accelerated H.264 video helpers with deterministic CPU fallback.

ArtiMo's physics and rendered pixels are independent of the video codec.  This
module therefore accelerates only the lossless hand-off points around those
pixels: H.264 encoding and decoding.  Availability is established by executing
the exact FFmpeg binary used by ImageIO, rather than by assuming that an
advertised encoder or CUDA device is usable.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import imageio.v2 as imageio
import imageio_ffmpeg


_ENCODER_ENV = "ARTIMO_VIDEO_ENCODER"
_DECODER_ENV = "ARTIMO_VIDEO_DECODER"


def _choice(name: str, allowed: set[str]) -> str:
    value = os.environ.get(name, "auto").strip().lower()
    aliases = {"gpu": "nvenc" if name == _ENCODER_ENV else "cuda"}
    value = aliases.get(value, value)
    if value not in allowed:
        raise ValueError(
            f"{name} must be one of {sorted(allowed)}, received {value!r}"
        )
    return value


def ffmpeg_executable() -> str:
    """Return the FFmpeg binary bundled with the pinned ImageIO dependency."""
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(
    command: list[str], *, timeout: float | None = 30
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 1, "", str(exc))


@functools.lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """Prove that FFmpeg can initialize and execute one NVENC frame."""
    completed = _run(
        [
            ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            # Current NVENC generations reject dimensions below their minimum
            # supported surface size; 256x256 is still a cheap one-frame probe.
            "color=c=black:s=256x256:r=1",
            "-frames:v",
            "1",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-f",
            "null",
            "-",
        ]
    )
    return completed.returncode == 0


def select_encoder() -> dict[str, Any]:
    """Select NVENC when usable, otherwise the existing libx264 backend."""
    requested = _choice(_ENCODER_ENV, {"auto", "nvenc", "cpu", "libx264"})
    use_nvenc = requested in {"auto", "nvenc"} and nvenc_available()
    if requested == "nvenc" and not use_nvenc:
        raise RuntimeError(
            "ARTIMO_VIDEO_ENCODER requested NVENC, but the pinned FFmpeg binary "
            "could not execute an h264_nvenc probe"
        )
    if use_nvenc:
        return {
            "codec": "h264_nvenc",
            "hardware_accelerated": True,
            "requested": requested,
            "ffmpeg": ffmpeg_executable(),
            "output_params": [
                "-preset",
                "p4",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "19",
                "-b:v",
                "0",
            ],
        }
    return {
        "codec": "libx264",
        "hardware_accelerated": False,
        "requested": requested,
        "ffmpeg": ffmpeg_executable(),
        "output_params": [],
    }


def open_h264_writer(
    path: Path, *, fps: float, macro_block_size: int
) -> tuple[Any, dict[str, Any]]:
    """Open an ImageIO writer using the selected, preflighted H.264 backend."""
    backend = select_encoder()
    kwargs: dict[str, Any] = {
        "fps": fps,
        "codec": backend["codec"],
        "macro_block_size": macro_block_size,
        "output_params": backend["output_params"],
    }
    if backend["hardware_accelerated"]:
        # ImageIO's generic quality flag maps to qscale, which NVENC ignores.
        # Constant quality is supplied explicitly through -cq above.
        kwargs["quality"] = None
    else:
        kwargs["quality"] = 8
    return imageio.get_writer(path, **kwargs), dict(backend)


def write_h264_video(
    path: Path,
    frames: Iterable[Any],
    *,
    fps: float,
    macro_block_size: int,
) -> dict[str, Any]:
    """Stream frames through the selected encoder and return backend evidence."""
    writer, backend = open_h264_writer(
        path, fps=fps, macro_block_size=macro_block_size
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return backend


def probe_video(video: Path) -> dict[str, Any]:
    """Return the ffprobe-shaped metadata used by review and verification.

    ``imageio-ffmpeg`` ships the exact FFmpeg binary needed by this application,
    while a separately installed ``ffprobe`` may be absent or an inaccessible
    Windows app-execution alias.  Reading only the generator's first metadata
    item avoids decoding frames and removes that unrelated system dependency.
    """
    frames = imageio_ffmpeg.read_frames(video)
    try:
        metadata = next(frames)
    finally:
        frames.close()
    width, height = metadata.get("source_size", metadata.get("size", (0, 0)))
    fps = float(metadata.get("fps") or 0.0)
    duration = float(metadata.get("duration") or 0.0)
    return {
        "streams": [
            {
                "codec_name": metadata.get("codec"),
                "width": int(width),
                "height": int(height),
                "nb_frames": None,
                "avg_frame_rate": f"{fps:g}/1" if fps > 0.0 else "0/0",
            }
        ],
        "format": {"duration": duration},
        "artimo_ffmpeg": {
            "executable": ffmpeg_executable(),
            "version": metadata.get("ffmpeg_version"),
        },
    }


def _cuda_decode_command(video: Path, *, frames: int | None) -> list[str]:
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "cuda",
        "-i",
        str(video),
    ]
    if frames is not None:
        command.extend(["-frames:v", str(int(frames))])
    command.extend(["-f", "null", "-"])
    return command


def select_decoder(video: Path) -> dict[str, Any]:
    """Preflight CUDA decoding for this video and select a safe reader mode."""
    requested = _choice(_DECODER_ENV, {"auto", "cuda", "cpu"})
    use_cuda = False
    if requested in {"auto", "cuda"}:
        use_cuda = _run(_cuda_decode_command(video, frames=1)).returncode == 0
    if requested == "cuda" and not use_cuda:
        raise RuntimeError(
            "ARTIMO_VIDEO_DECODER requested CUDA, but FFmpeg could not decode "
            f"the first frame of {video} with -hwaccel cuda"
        )
    return {
        "backend": "cuda" if use_cuda else "cpu",
        "hardware_accelerated": bool(use_cuda),
        "requested": requested,
        "ffmpeg": ffmpeg_executable(),
        "input_params": ["-hwaccel", "cuda"] if use_cuda else [],
    }


def decode_video_to_null(
    video: Path,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    """Fully decode a video, retrying on CPU only in automatic mode."""
    backend = select_decoder(video)
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        *backend["input_params"],
        "-i",
        str(video),
        "-f",
        "null",
        "-",
    ]
    completed = _run(command, timeout=None)
    if (
        completed.returncode != 0
        and backend["hardware_accelerated"]
        and backend["requested"] == "auto"
    ):
        backend = {
            **backend,
            "backend": "cpu",
            "hardware_accelerated": False,
            "input_params": [],
            "fallback_reason": completed.stderr.strip(),
        }
        completed = _run(
            [
                ffmpeg_executable(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-f",
                "null",
                "-",
            ],
            timeout=None,
        )
    return completed, backend
