#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

from build_textured_animated_glb import build_textured_animated_glb_from_urdf
from canonicalize_urdf_names import canonicalize_urdf_names


def _iter_asset_dirs(root: Path):
    for asset_dir in sorted(root.iterdir()):
        if asset_dir.is_dir() and (asset_dir / "mobility.urdf").exists():
            yield asset_dir


def _existing_build_is_usable(out_glb: Path, out_report: Path, requested_frames_per_joint: int) -> bool:
    if not (out_glb.exists() and out_report.exists()):
        return False
    if int(requested_frames_per_joint) <= 0:
        return True
    try:
        report = json.loads(out_report.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(report, dict):
        return False
    num_frames = int(report.get("num_frames") or 0)
    existing_frames_per_joint = int(report.get("frames_per_joint") or 0)
    movable = [x for x in (report.get("movable_joints") or []) if x]
    if movable and (num_frames <= 1 or existing_frames_per_joint <= 0):
        return False
    return True


def _process_one(
    asset_dir_str: str,
    overwrite: bool,
    fps: int,
    frames_per_joint: int,
    initial_pose_mode: str,
    max_preview_joints: int | None,
    canonicalize_urdf_names_flag: bool,
    keep_joint_names: bool,
) -> dict:
    asset_dir = Path(asset_dir_str).resolve()
    asset_id = asset_dir.name
    out_glb = asset_dir / f"animated_textured_{asset_id}.glb"
    out_report = out_glb.with_suffix(".report.json")
    started = time.time()
    if not overwrite and _existing_build_is_usable(out_glb, out_report, int(frames_per_joint)):
        return {
            "asset_id": asset_id,
            "asset_dir": str(asset_dir),
            "output_glb": str(out_glb),
            "report": str(out_report),
            "status": "skipped_exists",
            "duration_s": round(time.time() - started, 6),
        }
    rename_summary = None
    if canonicalize_urdf_names_flag:
        rename_summary = canonicalize_urdf_names(
            asset_dir / "mobility.urdf",
            rename_joints=not bool(keep_joint_names),
            write_map=True,
            backup=True,
            dry_run=False,
        )
    out_glb, report = build_textured_animated_glb_from_urdf(
        asset_dir,
        out_glb,
        fps=int(fps),
        frames_per_joint=int(frames_per_joint),
        initial_pose_mode=str(initial_pose_mode),
        max_preview_joints=None if max_preview_joints is None else int(max_preview_joints),
    )
    return {
        "asset_id": asset_id,
        "asset_dir": str(asset_dir),
        "output_glb": str(out_glb),
        "report": str(report),
        "status": "converted",
        "fps": int(fps),
        "frames_per_joint": int(frames_per_joint),
        "initial_pose_mode": str(initial_pose_mode),
        "max_preview_joints": None if max_preview_joints is None else int(max_preview_joints),
        "urdf_name_canonicalization": rename_summary,
        "duration_s": round(time.time() - started, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical animated_textured_<asset>.glb in-place for PartNet-Mobility assets.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2 or 1)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--assets", nargs="*", default=None, help="Optional explicit asset ids")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames_per_joint", type=int, default=12)
    parser.add_argument("--initial_pose_mode", choices=["zeros", "prismatic_lower", "lower"], default="zeros")
    parser.add_argument("--max_preview_joints", type=int, default=8)
    parser.add_argument("--canonicalize_urdf_names", action="store_true")
    parser.add_argument("--keep_joint_names", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    asset_dirs = list(_iter_asset_dirs(root))
    if args.assets:
        wanted = {str(x).strip() for x in args.assets if str(x).strip()}
        asset_dirs = [p for p in asset_dirs if p.name in wanted]
    if args.limit is not None:
        asset_dirs = asset_dirs[: max(0, int(args.limit))]
    if not asset_dirs:
        raise SystemExit("No matching asset directories found.")

    manifest_path = root / "animated_textured_build_manifest.json"
    summary_path = root / "animated_textured_build_summary.json"

    results: list[dict] = []
    converted = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as ex:
        future_map = {
            ex.submit(
                _process_one,
                str(asset_dir),
                bool(args.overwrite),
                int(args.fps),
                int(args.frames_per_joint),
                str(args.initial_pose_mode),
                None if args.max_preview_joints is None else int(args.max_preview_joints),
                bool(args.canonicalize_urdf_names),
                bool(args.keep_joint_names),
            ): asset_dir.name
            for asset_dir in asset_dirs
        }
        done = 0
        for fut in as_completed(future_map):
            asset_id = future_map[fut]
            done += 1
            try:
                row = fut.result()
            except Exception as exc:
                row = {
                    "asset_id": asset_id,
                    "asset_dir": str((root / asset_id).resolve()),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            status = str(row.get("status") or "")
            if status == "converted":
                converted += 1
            elif status == "skipped_exists":
                skipped += 1
            else:
                failed += 1
            results.append(row)
            if done % 25 == 0 or done == len(asset_dirs):
                print(
                    f"[INFO] done={done}/{len(asset_dirs)} converted={converted} skipped={skipped} failed={failed}",
                    flush=True,
                )

    results.sort(key=lambda x: str(x.get("asset_id") or ""))
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    final_with_glb = sum(
        1
        for asset_dir in asset_dirs
        if (Path(asset_dir) / f"animated_textured_{Path(asset_dir).name}.glb").exists()
    )
    summary = {
        "root": str(root),
        "total_requested": len(asset_dirs),
        "converted": converted,
        "skipped_exists": skipped,
        "failed": failed,
        "assets_with_canonical_glb_after_run": int(final_with_glb),
        "assets_missing_canonical_glb_after_run": int(len(asset_dirs) - final_with_glb),
        "jobs": int(args.jobs),
        "overwrite": bool(args.overwrite),
        "fps": int(args.fps),
        "frames_per_joint": int(args.frames_per_joint),
        "initial_pose_mode": str(args.initial_pose_mode),
        "max_preview_joints": None if args.max_preview_joints is None else int(args.max_preview_joints),
        "canonicalize_urdf_names": bool(args.canonicalize_urdf_names),
        "keep_joint_names": bool(args.keep_joint_names),
        "duration_s": round(time.time() - t0, 6),
        "manifest": str(manifest_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
