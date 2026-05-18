#!/usr/bin/env python3
"""
Convert Lightwheel_OpenSource articulated USD assets to this repo data format.

This reuses the ArtVIP conversion pipeline, but changes:
- asset discovery
- primary USD selection
- output asset naming

Output per asset:
  <out_root>/<asset_name>/
    mobility.urdf
    meshes/<link_name>.obj
    source_model.usd|usda|usdc
    animated_textured_<asset_name>.glb   (if --rebuild_canonical_glb)
    conversion_summary.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from pxr import Usd, UsdPhysics

import convert_artvip_to_data_format as artvip


def _iter_usd_files(root: Path):
    for ext in ("*.usd", "*.usda", "*.usdc"):
        for path in sorted(root.rglob(ext)):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.parts):
                continue
            yield path


def _choose_primary_usd(asset_dir: Path) -> Optional[Path]:
    candidates = []
    stem = asset_dir.name
    for ext in (".usd", ".usda", ".usdc"):
        p = asset_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for path in _iter_usd_files(asset_dir):
        if path.parent == asset_dir:
            candidates.append(path)
    return candidates[0] if candidates else None


def _stage_articulation_signature(usd_path: Path) -> tuple[int, int]:
    stage = Usd.Stage.Open(str(usd_path))
    if not stage:
        return 0, 0
    rigid = 0
    joints = 0
    for prim in stage.Traverse():
        try:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid += 1
        except Exception:
            pass
        t = str(prim.GetTypeName() or "")
        if t in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsFixedJoint"}:
            joints += 1
    return rigid, joints


def _is_object_asset_dir(asset_dir: Path) -> bool:
    usd_path = _choose_primary_usd(asset_dir)
    if usd_path is None:
        return False
    rigid, joints = _stage_articulation_signature(usd_path)
    return rigid > 0 and joints > 0


def _find_lightwheel_asset_dirs(lightwheel_root: Path):
    out = []
    seen = set()
    for usd_path in _iter_usd_files(lightwheel_root):
        asset_dir = usd_path.parent
        if asset_dir in seen:
            continue
        if _choose_primary_usd(asset_dir) != usd_path:
            continue
        seen.add(asset_dir)
        if _is_object_asset_dir(asset_dir):
            out.append(asset_dir)
    return sorted(out)


def _asset_name_from_path(asset_dir: Path, lightwheel_root: Path) -> str:
    rel = asset_dir.resolve().relative_to(lightwheel_root.resolve())
    parts = ["lightwheel"] + [artvip._safe_name(str(x)).lower() for x in rel.parts]
    return "__".join(parts)


def _monkeypatch_artvip_helpers():
    artvip._find_object_asset_dirs = _find_lightwheel_asset_dirs
    artvip._asset_name_from_path = _asset_name_from_path
    artvip._find_model_usd = _choose_primary_usd


def _has_path(edges: list[tuple[str, str]], start: str, goal: str) -> bool:
    if not start or not goal:
        return False
    stack = [str(start)]
    visited = set()
    adj = {}
    for a, b in edges:
        adj.setdefault(str(a), []).append(str(b))
    while stack:
        cur = stack.pop()
        if cur == goal:
            return True
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(adj.get(cur, []))
    return False


def _remove_cycle_creating_fixed_joints(urdf_path: Path, summary_path: Path | None = None) -> list[str]:
    if not urdf_path.exists():
        return []
    new_to_old_name = {}
    if summary_path is not None and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            mapping = summary.get("joint_name_map_old_to_new") or {}
            if isinstance(mapping, dict):
                new_to_old_name = {str(v): str(k) for k, v in mapping.items()}
        except Exception:
            new_to_old_name = {}
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    removed: list[str] = []
    while True:
        rows = []
        for joint_el in robot.findall("joint"):
            parent_el = joint_el.find("parent")
            child_el = joint_el.find("child")
            rows.append(
                {
                    "el": joint_el,
                    "name": str(joint_el.get("name") or ""),
                    "old_name": str(new_to_old_name.get(str(joint_el.get("name") or ""), str(joint_el.get("name") or ""))),
                    "type": str(joint_el.get("type") or ""),
                    "parent": str(parent_el.get("link") or "") if parent_el is not None else "",
                    "child": str(child_el.get("link") or "") if child_el is not None else "",
                }
            )
        candidates = []
        for row in rows:
            if row["type"] != "fixed":
                continue
            parent = str(row["parent"] or "")
            child = str(row["child"] or "")
            if not parent or not child:
                continue
            edges_wo = [(str(r["parent"]), str(r["child"])) for r in rows if r is not row]
            if _has_path(edges_wo, child, parent):
                old_name = str(row.get("old_name") or "")
                if "_to_" in old_name and not old_name.endswith("_child_frame"):
                    pri = 0
                elif old_name.endswith("_child_frame"):
                    pri = 2
                else:
                    pri = 1
                candidates.append((pri, old_name, row))
        victim = min(candidates, key=lambda x: (x[0], x[1]))[2] if candidates else None
        if victim is None:
            break
        robot.remove(victim["el"])
        removed.append(str(victim.get("old_name") or victim["name"] or ""))
    if removed:
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return removed


def _build_canonical_glb(
    out_asset: Path,
    asset_name: str,
    py_exec: str,
    *,
    canonical_frames_per_joint: int,
    canonical_fps: int,
) -> tuple[bool, str]:
    out_glb = out_asset / f"animated_textured_{asset_name}.glb"
    cmd = [
        py_exec,
        "tools/build_textured_animated_glb.py",
        "--asset_root",
        str(out_asset),
        "--build_mode",
        "urdf_preview",
        "--out_glb",
        str(out_glb),
        "--fps",
        str(int(canonical_fps)),
        "--frames_per_joint",
        str(int(canonical_frames_per_joint)),
        "--initial_pose_mode",
        "zeros",
    ]
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = str(proc.stderr or proc.stdout or "").strip()
        return False, stderr[:1200]
    return True, str(out_glb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Lightwheel OpenSource articulated assets to local data format")
    ap.add_argument("--lightwheel_root", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--asset_dir", type=Path, default=None, help="Convert only one asset directory")
    ap.add_argument("--max_assets", type=int, default=None, help="Optional cap for quick sampling")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rebuild_canonical_glb", action="store_true")
    ap.add_argument("--keep_existing", action="store_true")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--canonical_frames_per_joint", type=int, default=24)
    ap.add_argument("--canonical_fps", type=int, default=30)
    args = ap.parse_args()

    lightwheel_root = args.lightwheel_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    _monkeypatch_artvip_helpers()

    if args.asset_dir is not None:
        asset_dirs = [args.asset_dir.resolve()]
    else:
        asset_dirs = _find_lightwheel_asset_dirs(lightwheel_root)
    if args.max_assets is not None and int(args.max_assets) > 0:
        asset_dirs = asset_dirs[: int(args.max_assets)]

    print(f"Lightwheel root: {lightwheel_root}")
    print(f"Output root: {out_root}")
    print(f"Assets: {len(asset_dirs)}")
    print(f"Workers: {args.workers}")

    ok = 0
    fail = 0
    failures = []

    def run_one(d: Path):
        ok0, msg0 = artvip._convert_one(
            d,
            lightwheel_root,
            out_root,
            False,
            args.python,
            args.keep_existing,
            None,
            args.canonical_frames_per_joint,
            args.canonical_fps,
        )
        if not ok0:
            return ok0, msg0
        asset_name = _asset_name_from_path(d, lightwheel_root)
        out_asset = out_root / asset_name
        summary_path = out_asset / "conversion_summary.json"
        removed_fixed = _remove_cycle_creating_fixed_joints(out_asset / "mobility.urdf", summary_path)
        if removed_fixed:
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = {}
            summary["removed_cycle_creating_fixed_joints"] = removed_fixed
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.rebuild_canonical_glb:
            ok1, msg1 = _build_canonical_glb(
                out_asset,
                asset_name,
                args.python,
                canonical_frames_per_joint=args.canonical_frames_per_joint,
                canonical_fps=args.canonical_fps,
            )
            if not ok1:
                return False, f"{asset_name}: built urdf but glb build failed: {msg1}"
        return True, msg0

    if args.workers <= 1:
        for i, d in enumerate(asset_dirs, start=1):
            good, msg = run_one(d)
            print(f"[{i}/{len(asset_dirs)}] {'OK ' if good else 'ERR'} {msg}")
            if good:
                ok += 1
            else:
                fail += 1
                failures.append(msg)
    else:
        futs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for i, d in enumerate(asset_dirs, start=1):
                futs[ex.submit(run_one, d)] = (i, d)
            for fut in concurrent.futures.as_completed(futs):
                i, d = futs[fut]
                try:
                    good, msg = fut.result()
                except Exception as exc:
                    good, msg = False, f"{d}: exception {type(exc).__name__}: {exc}"
                print(f"[{i}/{len(asset_dirs)}] {'OK ' if good else 'ERR'} {msg}")
                if good:
                    ok += 1
                else:
                    fail += 1
                    failures.append(msg)

    report = {
        "lightwheel_root": str(lightwheel_root),
        "out_root": str(out_root),
        "num_assets": len(asset_dirs),
        "ok": ok,
        "failed": fail,
        "failures": failures,
    }
    rp = out_root / "_lightwheel_conversion_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. ok={ok} failed={fail}")
    print(f"Report: {rp}")


if __name__ == "__main__":
    main()
