#!/usr/bin/env python3
"""Measure the visual-QA booleans the delivery contract requires.

``finalize_artimo_delivery.py`` demands that every visual-QA gate be ``true``,
but the values are supposed to come from evidence.  Without a tool the only way
to satisfy the finalizer is to assert them by hand, which is exactly the
fabrication the acceptance contract forbids.  This samples the published video
and the measured rollout traces, and derives each boolean from a stated rule.

The result is intentionally conservative: anything it cannot demonstrate is
reported as ``false`` with the reason, rather than being assumed to pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def _probe(video: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_name,width,height,nb_frames,avg_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video}: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def _sampled_frames(video: Path, sample_fps: float) -> tuple[list[np.ndarray], list[float], float, int]:
    """Read frames at no less than ``sample_fps``, plus native-rate differences.

    Returns the subsampled frames used for blank/stale checks together with
    consecutive differences measured at the *native* rate.  Continuity has to be
    judged at full rate: at 5 fps two sampled frames are six frames apart, so
    ordinary fast motion looks like a cut.
    """
    reader = imageio.get_reader(str(video))
    metadata = reader.get_meta_data()
    source_fps = float(metadata.get("fps") or 0.0)
    if source_fps <= 0.0:
        raise RuntimeError(f"Could not determine frame rate for {video}")
    stride = max(1, int(source_fps // sample_fps)) if sample_fps > 0 else 1
    frames: list[np.ndarray] = []
    native_differences: list[float] = []
    previous: np.ndarray | None = None
    total = 0
    for index, raw in enumerate(reader):
        frame = np.asarray(raw, dtype=np.uint8)
        total += 1
        if previous is not None:
            native_differences.append(
                float(np.mean(np.abs(frame.astype(np.int16) - previous.astype(np.int16))))
            )
        previous = frame
        if index % stride == 0:
            frames.append(frame)
    reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {video}")
    return frames, native_differences, source_fps, total


def review(
    video: Path,
    result_json: Path,
    requested_fps: float,
    minimum_penetration_m: float,
) -> dict[str, Any]:
    result = json.loads(result_json.read_text(encoding="utf-8"))
    physical = result["physical"]
    frames, native_differences, source_fps, total_frames = _sampled_frames(video, requested_fps)
    effective_fps = source_fps * len(frames) / max(total_frames, 1)
    probe = _probe(video)
    streams = probe.get("streams", [])
    reasons: list[str] = []

    # Stale or black imagery: a frozen encoder or an empty render shows up as
    # near-zero variance within a frame, or as consecutive identical frames.
    per_frame_std = [float(np.std(frame)) for frame in frames]
    blank_frames = sum(1 for value in per_frame_std if value < 1.0)
    duplicate_runs = 0
    longest_duplicate_run = 0
    for previous, current in zip(frames, frames[1:]):
        if np.array_equal(previous, current):
            duplicate_runs += 1
            longest_duplicate_run = max(longest_duplicate_run, duplicate_runs)
        else:
            duplicate_runs = 0
    if blank_frames:
        reasons.append(f"{blank_frames} sampled frames are blank or near-uniform")
    # A long identical run means the rollout stopped changing on screen; a short
    # one is normal during a hold phase.
    stale_limit = max(2, int(round(effective_fps * 1.5)))
    if longest_duplicate_run > stale_limit:
        reasons.append(
            f"{longest_duplicate_run} consecutive identical sampled frames "
            f"exceeds the {stale_limit}-frame stale limit"
        )

    # Camera discontinuity: a cut or splice appears as a single enormous
    # frame-to-frame difference relative to the rest of the video.  Judged at the
    # native rate, where consecutive frames are genuinely adjacent in time.
    differences = native_differences
    median_difference = float(np.median(differences)) if differences else 0.0
    peak_difference = float(np.max(differences)) if differences else 0.0
    high_percentile = float(np.percentile(differences, 99)) if differences else 0.0
    # Compare the peak against the 99th percentile rather than the median: a
    # rollout is mostly still, so the median is near zero and would flag any
    # ordinary motion.  A real splice stands far above even the fastest motion.
    spliced = bool(differences) and peak_difference > max(40.0, 4.0 * high_percentile)
    if spliced:
        reasons.append(
            f"frame-to-frame change peaks at {peak_difference:.2f} against a "
            f"99th percentile of {high_percentile:.2f}, which looks like a cut"
        )

    # Interpenetration and contact come from the simulator traces, not pixels:
    # the recorded penetration depth is the only trustworthy measure of overlap.
    deepest = 0.0
    for stage in physical.get("contact_diagnostics", []):
        for pair in stage.get("unexpected_contact_pairs", []):
            deepest = min(deepest, float(pair.get("deepest_penetration_m", 0.0)))
    no_interpenetration = deepest >= -abs(minimum_penetration_m)
    if not no_interpenetration:
        reasons.append(
            f"deepest unexpected penetration {deepest:.5f} m exceeds the "
            f"{minimum_penetration_m:.5f} m tolerance"
        )

    contacts = physical.get("contacts", [])
    contact_visible = bool(contacts) and all(
        int(item.get("target_contact_observations", 0)) > 0 for item in contacts
    )
    if not contact_visible:
        reasons.append("a stage recorded zero target-contact observations")

    motion = physical.get("joint_motion", {})
    motion_visible = bool(motion) and all(
        bool(item.get("order_passed")) for item in motion.values()
    )
    if not motion_visible:
        reasons.append("a requested joint did not reach its target in order")

    dense_enough = effective_fps >= requested_fps
    if not dense_enough:
        reasons.append(
            f"sampled {effective_fps:.2f} fps below the requested {requested_fps:.2f} fps"
        )

    no_artifacts = blank_frames == 0 and longest_duplicate_run <= stale_limit and not spliced

    return {
        "sample_rate_fps": effective_fps,
        "no_visible_interpenetration": bool(no_interpenetration),
        "physical_contact_visible": bool(contact_visible),
        "requested_motion_visible": bool(motion_visible),
        "no_rendering_artifacts": bool(no_artifacts),
        "evidence": {
            "video_codec": streams[0].get("codec_name") if streams else None,
            "video_duration_s": float(probe.get("format", {}).get("duration", 0.0)),
            "source_fps": source_fps,
            "total_frames": total_frames,
            "sampled_frames": len(frames),
            "blank_frames": blank_frames,
            "longest_identical_run": longest_duplicate_run,
            "stale_run_limit": stale_limit,
            "median_frame_difference": median_difference,
            "percentile99_frame_difference": high_percentile,
            "peak_frame_difference": peak_difference,
            "deepest_unexpected_penetration_m": deepest,
            "penetration_tolerance_m": abs(minimum_penetration_m),
            "dense_sampling": bool(dense_enough),
            "failure_reasons": reasons,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=5.0)
    parser.add_argument(
        "--penetration-tolerance-m",
        type=float,
        default=0.001,
        help="Overlap depth treated as solver noise rather than visible interpenetration",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = review(
            args.video.expanduser().resolve(),
            args.result.expanduser().resolve(),
            float(args.sample_fps),
            float(args.penetration_tolerance_m),
        )
        args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out.expanduser().write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
        gates = (
            "no_visible_interpenetration",
            "physical_contact_visible",
            "requested_motion_visible",
            "no_rendering_artifacts",
        )
        return 0 if all(report[key] for key in gates) else 2
    except Exception as exc:
        print(f"Visual QA review failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
