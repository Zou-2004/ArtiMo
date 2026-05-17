#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

EVALUATION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EVALUATION_ROOT.parent
TOOLS_ROOT = REPO_ROOT / "tools"
for _path in (EVALUATION_ROOT, TOOLS_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_ablation_3d as ev  # noqa: E402
import evaluate_external_4d as ext  # noqa: E402


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for k, v in list(flat.items()):
                if isinstance(v, (list, dict)):
                    flat[k] = json.dumps(v, ensure_ascii=False)
            writer.writerow(flat)


def _asset_stem(asset_name: str) -> str:
    return str(asset_name).replace("__", "_")


def _prefix_for_asset_root(asset_root: str) -> str:
    root = str(asset_root)
    if "/data/causal_data/" in root:
        return "causal_data"
    if "/data/not_causal_data/" in root:
        return "not_causal_data"
    if "/hard_case/" in root:
        return "hard_case"
    return Path(root).parent.name


def _action_aliases(asset_name: str, action_name: str, asset_root: str) -> list[str]:
    aliases = [str(action_name)]
    root = str(asset_root)
    if "/data/not_causal_data/" in root:
        if action_name == "open_door":
            if asset_name.startswith("dishwasher_door_"):
                aliases.append("fully_open_dishwasher_door")
            if asset_name.startswith("microwave_door_"):
                aliases.append("fully_open_microwave_door")
            if asset_name.startswith("oven_door_"):
                aliases.append("fully_open_oven_door")
        if action_name == "open_lid" and asset_name.startswith("laptop_lid_"):
            aliases.append("fully_open_laptop_lid")
        if action_name == "slide_door_open" and asset_name.startswith("sliding_cabinet_door_"):
            aliases.append("fully_slide_open")
        if action_name == "open_upper_cabinet" and asset_name.startswith("upper_cabinet_door_"):
            aliases.append("fully_open_upper_cabinet_door")
    if "/hard_case/" in root and asset_name == "8_drawers":
        aliases.append("open_all_drawers_2s")
    return list(dict.fromkeys(aliases))


def _candidate_stems(case: dict[str, Any]) -> list[str]:
    prefix = _prefix_for_asset_root(str(case.get("asset_root") or ""))
    asset = _asset_stem(str(case.get("asset_name") or ""))
    raw_asset = str(case.get("asset_name") or "")
    prefixes = [prefix]
    # Some hard-case benchmark annotations keep class/case_id semantics even
    # after the actual URDF assets were moved under data/causal_data.
    if str(case.get("class") or "") == "hard_case" or str(case.get("case_id") or "").startswith("causal_output:small_furniture__table__"):
        prefixes.append("hard_case")
    actions = _action_aliases(
        str(case.get("asset_name") or ""),
        str(case.get("action_name") or ""),
        str(case.get("asset_root") or ""),
    )
    stems = [
        f"{pref}_{asset}_{action}"
        for pref in dict.fromkeys(prefixes)
        for action in actions
    ]
    # Animate3D batch exports use asset__action stems without the dataset
    # prefix, with nested asset namespaces flattened by replacing "__" -> "_".
    stems.extend(f"{asset}__{action}" for action in actions)
    stems.extend(f"{raw_asset}__{action}" for action in actions)
    return list(dict.fromkeys(stems))


def _build_mapping(
    cases: list[dict[str, Any]],
    pred_dir: Path,
    prediction_variant: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    if prediction_variant:
        rows: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for case in cases:
            pred = (case.get("variant_glbs") or {}).get(prediction_variant)
            if pred and Path(pred).exists():
                rows.append({**case, "prediction_file": str(Path(pred).resolve()), "prediction_stem": prediction_variant})
            else:
                missing.append(
                    {
                        "case_id": case.get("case_id"),
                        "asset_name": case.get("asset_name"),
                        "action_name": case.get("action_name"),
                        "variant": prediction_variant,
                        "prediction_file": pred,
                    }
                )
        return rows, missing, []

    fbx_by_stem = {p.stem: p for p in pred_dir.glob("*.fbx")}
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for case in cases:
        candidates = _candidate_stems(case)
        found = [stem for stem in candidates if stem in fbx_by_stem]
        if case.get("asset_name") == "8_drawers":
            found = []
        if len(found) == 1:
            rows.append({**case, "prediction_file": str(fbx_by_stem[found[0]].resolve()), "prediction_stem": found[0]})
        else:
            missing.append(
                {
                    "case_id": case.get("case_id"),
                    "asset_name": case.get("asset_name"),
                    "action_name": case.get("action_name"),
                    "candidates": candidates,
                    "found": found,
                }
            )
    keep = {row["prediction_stem"] for row in rows}
    extras = [p for stem, p in sorted(fbx_by_stem.items()) if stem not in keep]
    return rows, missing, extras


def _probe_coordinates(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    asset = ev.load_asset_geometry(Path(case["asset_root"]), int(args.num_points_per_link), float(args.scale_floor_ratio))
    gt_glb = Path(case["gt_glb"])
    match_points_by_link, _target_components = ext._gt_glb_first_frame_targets(gt_glb, asset, int(args.num_points_per_link))
    gt_points = ext._whole_points(match_points_by_link, asset.visual_links)
    raw = ext._load_mesh_sequence(Path(case["prediction_file"]))
    mesh0 = trimesh.Trimesh(vertices=raw.vertices_by_frame[0], faces=raw.faces, process=False)
    try:
        pred_points = np.asarray(mesh0.sample(int(args.align_sample_points)), dtype=np.float32)
    except Exception:
        pred_points = ext._sample_rows(raw.vertices_by_frame[0], int(args.align_sample_points), seed=7)
    _tf_none, info_none = ext._alignment_matrix(pred_points, gt_points, "none", int(args.align_sample_points))
    _tf_sim, info_sim = ext._alignment_matrix(pred_points, gt_points, "similarity", int(args.align_sample_points))
    gt_diag = float(np.linalg.norm(np.max(gt_points, axis=0) - np.min(gt_points, axis=0)))
    pred_diag = float(np.linalg.norm(np.max(pred_points, axis=0) - np.min(pred_points, axis=0)))
    return {
        "case_id": case.get("case_id"),
        "class": case.get("class"),
        "asset_name": case.get("asset_name"),
        "action_name": case.get("action_name"),
        "prediction_file": case.get("prediction_file"),
        "prediction_frames": len(raw.vertices_by_frame),
        "gt_diag": gt_diag,
        "pred_diag": pred_diag,
        "diag_ratio_pred_over_gt": pred_diag / gt_diag if gt_diag > 0 else None,
        "none_chamfer": info_none.get("chamfer"),
        "none_chamfer_over_gt_diag": float(info_none.get("chamfer", math.inf)) / gt_diag if gt_diag > 0 else None,
        "similarity_chamfer": info_sim.get("chamfer"),
        "similarity_chamfer_over_gt_diag": float(info_sim.get("chamfer", math.inf)) / gt_diag if gt_diag > 0 else None,
        "similarity_perm": info_sim.get("perm"),
        "similarity_signs": info_sim.get("signs"),
        "similarity_scale": info_sim.get("scale"),
    }


def _evaluate_case(case: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    asset = ev.load_asset_geometry(Path(case["asset_root"]), int(args.num_points_per_link), float(args.scale_floor_ratio))
    row, annotation = ev.parse_annotation_case(Path(case["annotation_path"]))
    gt_glb = Path(case["gt_glb"])
    gt_seq = ev.load_glb_sequence(gt_glb, asset, int(args.num_points_per_link))
    match_points_by_link, target_components = ext._gt_glb_first_frame_targets(gt_glb, asset, int(args.num_points_per_link))
    gt_plan = json.loads(Path(case["gt_plan_json"]).read_text(encoding="utf-8"))
    raw = ext._load_mesh_sequence(Path(case["prediction_file"]))
    pred_seq, match_diagnose = ext._sequence_from_external(raw, asset, match_points_by_link, target_components, args)
    meta = {
        "case_id": case.get("case_id"),
        "class": case.get("class"),
        "asset_name": case.get("asset_name"),
        "action_name": case.get("action_name"),
    }
    static_rows = None
    static_manifest = getattr(args, "gt_phase_static_manifest", None)
    if static_manifest:
        static_rows = ev._phase_static_manifest_by_case(Path(static_manifest)).get(str(case.get("case_id") or ""))
        if static_rows:
            gt_seq = ev.load_phase_static_sequence(static_rows, asset, int(args.num_points_per_link))
        else:
            raise FileNotFoundError(f"No static phase rows for case_id={case.get('case_id')} in {static_manifest}")
    per_case, per_phase, per_link, matched, _ = ev.evaluate_variant(
        meta,
        row,
        annotation,
        asset,
        gt_seq,
        pred_seq,
        gt_plan,
        str(args.variant_name),
        args,
        static_rows,
    )
    per_case["prediction_file"] = str(case["prediction_file"])
    per_case["prediction_num_frames"] = len(pred_seq.frames)
    per_case["prediction_last_time_s"] = float(pred_seq.frames[-1].time_s) if pred_seq.frames else None
    per_case["alignment_chamfer"] = (match_diagnose.get("alignment") or {}).get("chamfer")
    per_case["num_components"] = match_diagnose.get("num_components")
    return per_case, per_phase, per_link, matched, match_diagnose


def _mean_rows(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"num_cases": len(rows)}
    for key in keys:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate AnimateAnyMesh FBX outputs against the 3D benchmark.")
    parser.add_argument("--pred_dir", type=Path, default=REPO_ROOT / "aam_worldcoord_fbx")
    parser.add_argument(
        "--prediction_variant",
        default="",
        help="Use a variant GLB from the manifest, e.g. full_agent, as the prediction source instead of matching files in --pred_dir.",
    )
    parser.add_argument("--cases_manifest", type=Path, default=REPO_ROOT / "ablation_eval_combined_latest" / "diagnose" / "resolved_manifest.json")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "aam_3d_eval_latest")
    parser.add_argument("--variant_name", default="animate_anymesh")
    parser.add_argument("--clean_extra", action="store_true")
    parser.add_argument("--coordinate_probe_only", action="store_true")
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker_case_json", type=Path)
    parser.add_argument("--worker_out_json", type=Path)
    parser.add_argument("--num_points_per_link", type=int, default=2048)
    parser.add_argument("--voxel_resolution", type=int, default=64)
    parser.add_argument("--scale_floor_ratio", type=float, default=0.05)
    parser.add_argument("--dynamic_weight", type=float, default=0.8)
    parser.add_argument("--static_weight", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--pc_backend", choices=["numpy", "torch", "pytorch3d", "auto"], default="numpy")
    parser.add_argument("--gpu_devices", default="")
    parser.add_argument("--torch_chunk", type=int, default=1024)
    parser.add_argument("--pytorch3d_chunk", type=int, default=4096)
    parser.add_argument("--pc_fallback_numpy", action="store_true", default=True)
    parser.add_argument("--allow_equal_frames", action="store_true")
    parser.add_argument("--disable_terminal_state_check", action="store_true")
    parser.add_argument("--terminal_score_policy", choices=["min", "average"], default="min")
    parser.add_argument("--alignment_mode", choices=["none", "axis_extent", "similarity", "scale_translate_3d"], default="none")
    parser.add_argument("--align_sample_points", type=int, default=3000)
    parser.add_argument("--max_component_points", type=int, default=256)
    parser.add_argument("--min_component_faces", type=int, default=1)
    parser.add_argument("--assignment_mode", choices=["vertex", "component"], default="vertex")
    parser.add_argument("--gt_phase_static_manifest", type=Path, default=None)
    parser.add_argument("--prediction_fps", type=float, default=24.0)
    parser.add_argument("--prediction_duration_s", type=float, default=0.625)
    parser.add_argument(
        "--motion_scale_mode",
        choices=["scale_motion", "preserve_center_trajectory"],
        default="scale_motion",
    )
    return parser.parse_args()


def _worker_args(args: argparse.Namespace, case_path: Path, out_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker_case_json",
        str(case_path),
        "--worker_out_json",
        str(out_path),
        "--out_dir",
        str(args.out_dir),
        "--variant_name",
        str(args.variant_name),
        "--num_points_per_link",
        str(args.num_points_per_link),
        "--voxel_resolution",
        str(args.voxel_resolution),
        "--scale_floor_ratio",
        str(args.scale_floor_ratio),
        "--dynamic_weight",
        str(args.dynamic_weight),
        "--static_weight",
        str(args.static_weight),
        "--tau",
        str(args.tau),
        "--pc_backend",
        str(args.pc_backend),
        "--gpu_devices",
        str(args.gpu_devices),
        "--torch_chunk",
        str(args.torch_chunk),
        "--pytorch3d_chunk",
        str(args.pytorch3d_chunk),
        "--terminal_score_policy",
        str(args.terminal_score_policy),
        "--alignment_mode",
        str(args.alignment_mode),
        "--align_sample_points",
        str(args.align_sample_points),
        "--max_component_points",
        str(args.max_component_points),
        "--min_component_faces",
        str(args.min_component_faces),
        "--assignment_mode",
        str(args.assignment_mode),
        "--prediction_fps",
        str(args.prediction_fps),
        "--prediction_duration_s",
        str(args.prediction_duration_s),
        "--motion_scale_mode",
        str(args.motion_scale_mode),
    ]
    if args.gt_phase_static_manifest:
        cmd.extend(["--gt_phase_static_manifest", str(args.gt_phase_static_manifest)])
    if args.coordinate_probe_only:
        cmd.append("--coordinate_probe_only")
    if args.pc_fallback_numpy:
        cmd.append("--pc_fallback_numpy")
    if args.allow_equal_frames:
        cmd.append("--allow_equal_frames")
    if args.disable_terminal_state_check:
        cmd.append("--disable_terminal_state_check")
    return cmd


def _run_worker(case: dict[str, Any], idx: int, args: argparse.Namespace, diagnose_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    work_dir = diagnose_dir / "workers"
    work_dir.mkdir(parents=True, exist_ok=True)
    case_path = work_dir / f"case_{idx:04d}.json"
    out_path = work_dir / f"result_{idx:04d}.json"
    _write_json(case_path, case)
    proc = subprocess.run(
        _worker_args(args, case_path, out_path),
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0 or not out_path.exists():
        return None, {
            "case_id": case.get("case_id"),
            "prediction_file": case.get("prediction_file"),
            "returncode": proc.returncode,
            "output": proc.stdout[-4000:],
        }
    return json.loads(out_path.read_text(encoding="utf-8")), None


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    diagnose_dir = out_dir / "diagnose"
    if args.worker_case_json is not None and args.worker_out_json is not None:
        gc.disable()
        case = json.loads(Path(args.worker_case_json).read_text(encoding="utf-8"))
        if args.coordinate_probe_only:
            result = {"probe": _probe_coordinates(case, args)}
        else:
            per_case, per_phase, per_link, matched, match_diagnose = _evaluate_case(case, args)
            result = {
                "per_case": per_case,
                "per_phase": per_phase,
                "per_link": per_link,
                "matched": matched,
                "match_diagnose": match_diagnose,
            }
        _write_json(Path(args.worker_out_json), result)
        # ufbx can segfault during interpreter teardown on some multi-mesh FBX
        # files after results have already been written. Exit immediately so the
        # parent does not misclassify a successful worker as failed.
        os._exit(0)

    manifest = json.loads(Path(args.cases_manifest).read_text(encoding="utf-8"))
    cases = manifest.get("cases", manifest) if isinstance(manifest, dict) else manifest
    mapped, missing, extras = _build_mapping(cases, Path(args.pred_dir), str(args.prediction_variant or "").strip() or None)
    if args.max_cases and args.max_cases > 0:
        mapped = mapped[: int(args.max_cases)]

    deleted: list[str] = []
    if args.clean_extra:
        for path in extras:
            path.unlink()
            deleted.append(str(path))

    mapping_report = {
        "manifest_cases": len(cases),
        "matched_cases": len(mapped),
        "missing_cases": missing,
        "extra_fbx": [str(p) for p in extras],
        "deleted_extra_fbx": deleted,
    }
    _write_json(diagnose_dir / "aam_mapping.json", mapping_report)
    print(f"[INFO] matched={len(mapped)} missing={len(missing)} extras={len(extras)} deleted={len(deleted)}")

    if args.coordinate_probe_only:
        probe_rows = []
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {
                pool.submit(_run_worker, case, idx, args, diagnose_dir): (idx, case)
                for idx, case in enumerate(mapped, start=1)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx, case = futures[fut]
                result, err = fut.result()
                if err is None and result is not None:
                    row = result["probe"]
                    probe_rows.append(row)
                    print(
                        f"[PROBE] {idx}/{len(mapped)} {case['asset_name']}/{case['action_name']} "
                        f"none_norm={row['none_chamfer_over_gt_diag']:.6f} sim_norm={row['similarity_chamfer_over_gt_diag']:.6f}",
                        flush=True,
                    )
                else:
                    errors.append(err or {"case_id": case.get("case_id"), "error": "unknown_worker_error"})
                    print(f"[ERROR] {idx}/{len(mapped)} {case.get('case_id')}", flush=True)
        _write_json(diagnose_dir / "coordinate_probe.json", {"rows": probe_rows, "errors": errors})
        _write_csv(
            out_dir / "coordinate_probe.csv",
            probe_rows,
            [
                "case_id",
                "class",
                "asset_name",
                "action_name",
                "prediction_frames",
                "gt_diag",
                "pred_diag",
                "diag_ratio_pred_over_gt",
                "none_chamfer",
                "none_chamfer_over_gt_diag",
                "similarity_chamfer",
                "similarity_chamfer_over_gt_diag",
                "similarity_perm",
                "similarity_signs",
                "similarity_scale",
                "prediction_file",
            ],
        )
        return 0 if not errors else 1

    per_case_rows: list[dict[str, Any]] = []
    per_phase_rows: list[dict[str, Any]] = []
    per_link_rows: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(_run_worker, case, idx, args, diagnose_dir): (idx, case)
            for idx, case in enumerate(mapped, start=1)
        }
        for fut in concurrent.futures.as_completed(futures):
            idx, case = futures[fut]
            result, err = fut.result()
            if err is None and result is not None:
                per_case = result["per_case"]
                per_phase = result["per_phase"]
                per_link = result["per_link"]
                matched = result["matched"]
                match_diagnose = result["match_diagnose"]
                per_case_rows.append(per_case)
                per_phase_rows.extend(per_phase)
                per_link_rows.extend(per_link)
                matched_rows.extend(matched)
                safe_id = f"{case['class']}__{case['asset_name']}__{case['action_name']}".replace("/", "__")
                _write_json(diagnose_dir / "matching" / f"{safe_id}.json", match_diagnose)
                print(
                    f"[OK] {idx}/{len(mapped)} {case['asset_name']}/{case['action_name']} "
                    f"PN_gIoU={per_case.get('PN_gIoU'):.6f} PN_PC={per_case.get('PN_PC'):.6f} PN_OC={per_case.get('PN_OC'):.6f}",
                    flush=True,
                )
            else:
                errors.append(err or {"case_id": case.get("case_id"), "prediction_file": case.get("prediction_file"), "error": "unknown_worker_error"})
                print(f"[ERROR] {idx}/{len(mapped)} {case.get('case_id')}", flush=True)

    metric_keys = [
        "PN_gIoU",
        "PN_PC",
        "PN_OC",
        "Dynamic_gIoU",
        "Dynamic_PC",
        "Dynamic_OC",
        "Static_gIoU",
        "Static_PC",
        "Static_OC",
    ]
    summary = {str(args.variant_name): _mean_rows(per_case_rows, metric_keys)}
    category_rows = []
    category_rows.append({"class": "overall", "variant": str(args.variant_name), **_mean_rows(per_case_rows, ["PN_gIoU", "PN_PC", "PN_OC"])})
    for cls in sorted({str(r.get("class")) for r in per_case_rows}):
        cls_rows = [r for r in per_case_rows if str(r.get("class")) == cls]
        category_rows.append({"class": cls, "variant": str(args.variant_name), **_mean_rows(cls_rows, ["PN_gIoU", "PN_PC", "PN_OC"])})

    _write_json(diagnose_dir / "errors.json", errors)
    _write_json(diagnose_dir / "matched_frames.json", matched_rows)
    _write_json(diagnose_dir / "summary.json", summary)
    _write_csv(
        out_dir / "aam_metrics.csv",
        per_case_rows,
        [
            "case_id",
            "class",
            "asset_name",
            "action_name",
            "variant",
            *metric_keys,
            "prediction_num_frames",
            "prediction_last_time_s",
            "alignment_chamfer",
            "num_components",
            "prediction_file",
        ],
    )
    _write_csv(out_dir / "aam_metrics_category_mean.csv", category_rows, ["class", "variant", "num_cases", "PN_gIoU", "PN_PC", "PN_OC"])
    _write_csv(
        diagnose_dir / "per_phase_metrics.csv",
        per_phase_rows,
        [
            "case_id",
            "class",
            "asset_name",
            "action_name",
            "variant",
            "phase_index",
            "phase_id",
            "PN_gIoU",
            "PN_PC",
            "PN_OC",
            "Dynamic_gIoU",
            "Dynamic_PC",
            "Dynamic_OC",
            "Static_gIoU",
            "Static_PC",
            "Static_OC",
        ],
    )
    _write_csv(
        diagnose_dir / "per_link_metrics.csv",
        per_link_rows,
        [
            "case_id",
            "class",
            "asset_name",
            "action_name",
            "variant",
            "phase_index",
            "phase_id",
            "link",
            "group",
            "PN_gIoU",
            "PN_PC",
            "PN_OC",
        ],
    )
    print(f"[INFO] wrote {out_dir / 'aam_metrics.csv'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(code))
