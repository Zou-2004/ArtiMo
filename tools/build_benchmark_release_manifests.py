#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
_PARTNET_DIR_RE = re.compile(r"^(\d+)(?:_(.+))?$")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _compiled_class_from_template_dir(name: str) -> str:
    for suffix in ("_constraint_templates", "_variant_templates", "_templates"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _compact_source_file(path: Any, dataset: str | None = None) -> str | None:
    text = str(path or "").strip()
    if not text:
        return None
    marker = str(dataset or "").strip()
    if marker and f"/{marker}/" in text:
        return f"{marker}/" + text.split(f"/{marker}/", 1)[1]
    for marker in ("ArtVIP", "PartNet-Mobility", "partnet-mobility", "Lightwheel", "ACD"):
        if f"/{marker}/" in text:
            return f"{marker}/" + text.split(f"/{marker}/", 1)[1]
    if text.startswith(str(REPO_ROOT)):
        try:
            return str(Path(text).resolve().relative_to(REPO_ROOT))
        except Exception:
            pass
    return text


def _partnet_raw_id_from_name(name: str) -> str | None:
    match = _PARTNET_DIR_RE.match(str(name or "").strip())
    return match.group(1) if match else None


def build_partnet_source_index(partnet_roots: list[Path]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for root in partnet_roots:
        if not root.exists():
            continue
        for asset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            raw_id = _partnet_raw_id_from_name(asset_dir.name)
            if raw_id is None:
                continue
            row = {
                "raw_id": raw_id,
                "dir_name": asset_dir.name,
                "category": asset_dir.name.split("_", 1)[1] if "_" in asset_dir.name else "",
            }
            index.setdefault(raw_id, row)
            meta_path = asset_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = _read_json(meta_path)
            except Exception:
                continue
            if isinstance(meta, dict):
                model_id = str(meta.get("model_id") or "").strip()
                anno_id = str(meta.get("anno_id") or "").strip()
                if model_id:
                    index.setdefault(model_id, row)
                if anno_id:
                    index.setdefault(anno_id, row)
    return index


def _partnet_release_source(source_asset: Any, partnet_index: dict[str, dict[str, str]]) -> tuple[str | None, str | None, str | None]:
    text = str(source_asset or "").strip()
    if not text:
        return None, None, None
    row = partnet_index.get(text)
    if row is None:
        raw_id = _partnet_raw_id_from_name(text)
        if raw_id is not None:
            dir_name = text if "_" in text else raw_id
            return raw_id, dir_name, f"PartNet-Mobility/{raw_id}"
        return text, None, f"PartNet-Mobility/{text}"
    raw_id = row.get("raw_id") or text
    return raw_id, row.get("dir_name"), f"PartNet-Mobility/{raw_id}"


def _release_source_fields(dataset: Any, source_file: Any, source_asset: Any, partnet_index: dict[str, dict[str, str]]) -> tuple[Any, Any, Any]:
    dataset_text = str(dataset or "")
    if dataset_text == "PartNet-Mobility":
        raw_id, source_dir, source_file_out = _partnet_release_source(source_asset, partnet_index)
        return raw_id or source_asset, source_file_out or source_file, source_dir
    return source_asset, source_file, None


def _action_prompt_from_source(source_annotation_root: Path | None, release_root: Path, ann_path: Path) -> str | None:
    if source_annotation_root is None or not source_annotation_root.exists():
        return None
    try:
        rel = ann_path.resolve().relative_to((release_root / "annotations").resolve())
    except Exception:
        return None
    src_path = source_annotation_root / rel
    if not src_path.exists():
        return None
    try:
        src = _read_json(src_path)
    except Exception:
        return None
    prompt = src.get("action_text") if isinstance(src, dict) else None
    if not prompt and isinstance(src, dict):
        ann = src.get("annotation")
        if isinstance(ann, dict):
            action = ann.get("action")
            if isinstance(action, dict):
                prompt = action.get("action_text")
    text = str(prompt or "").strip()
    return text or None


def _metadata_from_source_release(source_release_annotation_root: Path | None, release_root: Path, ann_path: Path) -> dict[str, Any]:
    if source_release_annotation_root is None or not source_release_annotation_root.exists():
        return {}
    try:
        rel = ann_path.resolve().relative_to((release_root / "annotations").resolve())
    except Exception:
        return {}
    src_path = source_release_annotation_root / rel
    if not src_path.exists():
        return {}
    try:
        src = _read_json(src_path)
    except Exception:
        return {}
    meta = src.get("metadata") if isinstance(src, dict) else None
    return meta if isinstance(meta, dict) else {}


def simplified_annotation_metadata(
    sidecar: dict[str, Any],
    ann_path: Path,
    *,
    release_root: Path,
    source_annotation_root: Path | None,
    source_release_annotation_root: Path | None,
    partnet_index: dict[str, dict[str, str]],
) -> dict[str, Any]:
    metadata = sidecar.get("metadata") if isinstance(sidecar.get("metadata"), dict) else {}
    source_metadata = _metadata_from_source_release(source_release_annotation_root, release_root, ann_path)
    raw_metadata = source_metadata or metadata
    detail = raw_metadata.get("source_dataset_detail") if isinstance(raw_metadata.get("source_dataset_detail"), dict) else {}
    paths = raw_metadata.get("paths") if isinstance(raw_metadata.get("paths"), dict) else {}
    cls = str(metadata.get("class") or raw_metadata.get("class") or _compiled_class_from_template_dir(ann_path.parents[1].name))
    asset = str(metadata.get("asset_name") or sidecar.get("asset_name") or "")
    action = str(metadata.get("action_name") or sidecar.get("action_name") or "")
    dataset = str(metadata.get("source_dataset") or raw_metadata.get("source_dataset") or "").strip() or None
    source_file = (
        metadata.get("source_file")
        or detail.get("source_model")
        or detail.get("source_file")
        or paths.get("asset_root_source")
        or paths.get("annotation_source")
    )
    source_asset = (
        metadata.get("source_asset")
        or detail.get("conversion_asset_name")
        or detail.get("model_id")
        or detail.get("anno_id")
        or detail.get("source_asset_dir")
    )
    source_asset, source_file, source_asset_dir = _release_source_fields(dataset, source_file, source_asset, partnet_index)
    out: dict[str, Any] = {
        "case_id": str(metadata.get("case_id") or raw_metadata.get("case_id") or sidecar.get("case_id") or f"{cls}:{asset}:{action}"),
        "class": cls,
        "asset_name": asset,
        "action_name": action,
    }
    if metadata.get("split") is not None or raw_metadata.get("benchmark_split") is not None:
        out["split"] = metadata.get("split") or raw_metadata.get("benchmark_split")
    if metadata.get("asset_collection") is not None or raw_metadata.get("asset_collection") is not None:
        out["asset_collection"] = metadata.get("asset_collection") or raw_metadata.get("asset_collection")
    action_prompt = metadata.get("action_prompt") or _action_prompt_from_source(source_annotation_root, release_root, ann_path)
    if action_prompt:
        out["action_prompt"] = str(action_prompt)
    if dataset is not None:
        out["source_dataset"] = dataset
    compact_file = _compact_source_file(source_file, dataset)
    if compact_file is not None:
        out["source_file"] = compact_file
    if source_asset:
        out["source_asset"] = _compact_source_file(source_asset, dataset)
    if source_asset_dir and str(source_asset_dir) != str(source_asset):
        out["source_asset_dir"] = str(source_asset_dir)
    return out


def _path_for_manifest(path: Path | None, cwd: Path) -> str | None:
    if path is None:
        return None
    try:
        if path.is_absolute():
            rel = path.resolve().relative_to(cwd.resolve())
            return str(rel)
    except Exception:
        pass
    return str(path)


def _asset_root_candidates(release_root: Path, collection: str, asset: str, data_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    candidates.append(release_root / "source_assets" / collection / asset)
    candidates.append(REPO_ROOT / "data" / collection / asset)
    for root in data_roots:
        candidates.append(root / collection / asset)
        candidates.append(root / asset)
    out: list[Path] = []
    seen: set[Path] = set()
    for cand in candidates:
        try:
            key = cand.resolve()
        except Exception:
            key = cand
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _choose_asset_root(release_root: Path, collection: str, asset: str, data_roots: list[Path]) -> tuple[Path, bool]:
    candidates = _asset_root_candidates(release_root, collection, asset, data_roots)
    for cand in candidates:
        if (cand / "mobility.urdf").exists():
            return cand.resolve(), False
    return candidates[0], True


def _prediction_dir_candidates(prediction_roots: list[Path], variant: str, cls: str, asset: str, action: str) -> list[Path]:
    out: list[Path] = []
    for root in prediction_roots:
        out.extend(
            [
                root / variant / cls / asset / action / asset,
                root / variant / cls / asset / action,
                root / variant / asset / action / asset,
                root / variant / asset / action,
                root / cls / asset / action / asset,
                root / cls / asset / action,
                root / asset / action / asset,
                root / asset / action,
            ]
        )
    return out


def _find_prediction_artifacts(
    prediction_roots: list[Path],
    variant: str,
    cls: str,
    asset: str,
    action: str,
) -> tuple[Path | None, Path | None, Path | None]:
    for cand in _prediction_dir_candidates(prediction_roots, variant, cls, asset, action):
        traj = cand / "trajectory.jsonl"
        plan = cand / "plan.json"
        glb = cand / "plan_animated.glb"
        if traj.exists() or glb.exists() or plan.exists():
            return (
                traj.resolve() if traj.exists() else None,
                plan.resolve() if plan.exists() else None,
                glb.resolve() if glb.exists() else None,
            )
    return None, None, None


def _iter_annotation_paths(release_root: Path) -> list[Path]:
    return sorted((release_root / "annotations").glob("*_templates/cases/*.json"))


def simplify_annotation_metadata_in_place(
    release_root: Path,
    source_annotation_root: Path | None,
    source_release_annotation_root: Path | None,
    partnet_index: dict[str, dict[str, str]],
) -> int:
    changed = 0
    for ann_path in _iter_annotation_paths(release_root):
        sidecar = _read_json(ann_path)
        if not isinstance(sidecar, dict):
            continue
        new_metadata = simplified_annotation_metadata(
            sidecar,
            ann_path,
            release_root=release_root,
            source_annotation_root=source_annotation_root,
            source_release_annotation_root=source_release_annotation_root,
            partnet_index=partnet_index,
        )
        if sidecar.get("metadata") == new_metadata:
            continue
        sidecar["metadata"] = new_metadata
        _write_json(ann_path, sidecar)
        changed += 1
    return changed


def build_case_manifest(
    release_root: Path,
    *,
    data_roots: list[Path],
    prediction_roots: list[Path],
    prediction_variant: str | None,
    partnet_index: dict[str, dict[str, str]],
    cwd: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    missing_assets: list[dict[str, str]] = []
    missing_gt: list[dict[str, str]] = []

    for ann_path in _iter_annotation_paths(release_root):
        sidecar = _read_json(ann_path)
        metadata = sidecar.get("metadata") if isinstance(sidecar, dict) and isinstance(sidecar.get("metadata"), dict) else {}
        cls = str(metadata.get("class") or _compiled_class_from_template_dir(ann_path.parents[1].name))
        asset = str(metadata.get("asset_name") or sidecar.get("asset_name") or "")
        action = str(metadata.get("action_name") or sidecar.get("action_name") or "")
        case_id = str(metadata.get("case_id") or sidecar.get("case_id") or f"{cls}:{asset}:{action}")
        collection = str(metadata.get("asset_collection") or ("not_causal_data" if cls == "non_causal" else "causal_data"))
        source_asset = metadata.get("source_asset")
        source_dataset = metadata.get("source_dataset")
        source_asset, source_file, source_asset_dir = _release_source_fields(
            source_dataset,
            metadata.get("source_file"),
            source_asset,
            partnet_index,
        )
        action_prompt = metadata.get("action_prompt")
        if not asset or not action:
            continue

        asset_root, asset_missing = _choose_asset_root(release_root, collection, asset, data_roots)
        if asset_missing:
            missing_assets.append({"case_id": case_id, "asset_root": str(asset_root)})

        gt_dir = release_root / "gt_animations" / cls / asset / action
        gt_plan = gt_dir / "plan.json"
        gt_traj = gt_dir / "animation" / "trajectory.jsonl"
        gt_glb = gt_dir / "animation" / "plan_animated.glb"
        if not gt_traj.exists() and not gt_glb.exists():
            missing_gt.append({"case_id": case_id, "gt_dir": str(gt_dir)})

        case = {
            "case_id": case_id,
            "class": cls,
            "asset_name": asset,
            "action_name": action,
            "action_prompt": action_prompt,
            "source_dataset": source_dataset,
            "source_file": source_file,
            "source_asset": source_asset,
            "source_asset_dir": source_asset_dir,
            "asset_collection": collection,
            "annotation_path": _path_for_manifest(ann_path, cwd),
            "gt_plan_json": _path_for_manifest(gt_plan if gt_plan.exists() else None, cwd),
            "gt_trajectory": _path_for_manifest(gt_traj if gt_traj.exists() else None, cwd),
            "gt_glb": _path_for_manifest(gt_glb if gt_glb.exists() else None, cwd),
        }
        if prediction_roots and prediction_variant:
            traj, plan, glb = _find_prediction_artifacts(prediction_roots, prediction_variant, cls, asset, action)
            if traj is not None or glb is not None:
                case["variants"] = {prediction_variant: _path_for_manifest(traj, cwd)}
                if plan is not None:
                    case["variant_plans"] = {prediction_variant: _path_for_manifest(plan, cwd)}
                if glb is not None:
                    case["variant_glbs"] = {prediction_variant: _path_for_manifest(glb, cwd)}
        cases.append({k: v for k, v in case.items() if v is not None})

    cases.sort(key=lambda r: (str(r.get("class")), str(r.get("asset_name")), str(r.get("action_name"))))
    summary = {
        "num_cases": len(cases),
        "missing_asset_roots": len(missing_assets),
        "missing_gt_sequences": len(missing_gt),
    }
    if missing_assets:
        summary["missing_assets"] = missing_assets[:50]
    if missing_gt:
        summary["missing_gt"] = missing_gt[:50]
    return cases, summary


def _phase_glb_for_row(case_dir: Path, row: dict[str, Any], fallback_glbs: list[Path]) -> Path | None:
    raw = row.get("static_glb") or row.get("output_glb")
    if raw:
        basename = Path(str(raw)).name
        cand = case_dir / basename
        if cand.exists():
            return cand.resolve()
    try:
        idx = int(row.get("phase_index", 0))
    except Exception:
        idx = 0
    if 0 <= idx < len(fallback_glbs):
        return fallback_glbs[idx].resolve()
    return None


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if not value:
        return []
    try:
        obj = json.loads(str(value))
    except Exception:
        return []
    return [str(x) for x in obj] if isinstance(obj, list) else []


def build_phase_static_manifest(release_root: Path, *, cwd: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted((release_root / "gt_phase_static_meshes").glob("*/*/*/phase_static_manifest.json")):
        case_dir = manifest_path.parent
        fallback_glbs = sorted([p for p in case_dir.glob("*.glb") if p.is_file()])
        data = _read_json(manifest_path)
        phases = data.get("phases") if isinstance(data, dict) else []
        for idx, src in enumerate(phases or []):
            if not isinstance(src, dict):
                continue
            row = dict(src)
            glb = _phase_glb_for_row(case_dir, row, fallback_glbs)
            cls = str(row.get("class") or case_dir.parents[1].name)
            asset = str(row.get("asset_name") or case_dir.parents[0].name)
            action = str(row.get("action_name") or case_dir.name)
            compact = {
                "case_id": row.get("case_id") or f"{cls}:{asset}:{action}",
                "class": cls,
                "asset_name": asset,
                "action_name": action,
                "phase_index": int(row.get("phase_index", idx)),
                "phase_id": row.get("phase_id"),
                "phase_endpoint_time_s": row.get("phase_endpoint_time_s"),
                "trajectory_time_s": row.get("trajectory_time_s"),
                "static_glb": _path_for_manifest(glb, cwd) if glb is not None else None,
            }
            dynamic_links = row.get("dynamic_links")
            if isinstance(dynamic_links, list):
                compact["dynamic_links"] = dynamic_links
                compact["dynamic_links_json"] = json.dumps(dynamic_links, ensure_ascii=False)
            rows.append({k: v for k, v in compact.items() if v is not None})
    rows.sort(
        key=lambda r: (
            str(r.get("class")),
            str(r.get("asset_name")),
            str(r.get("action_name")),
            int(r.get("phase_index", 0)),
        )
    )
    return rows


def build_asset_source_manifest(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_asset: dict[tuple[str, str], dict[str, Any]] = {}
    for case in cases:
        collection = str(case.get("asset_collection") or "")
        asset = str(case.get("asset_name") or "")
        if not asset:
            continue
        key = (collection, asset)
        row = by_asset.setdefault(
            key,
            {
                "asset_name": asset,
                "asset_collection": collection,
                "source_dataset": case.get("source_dataset"),
                "source_asset": case.get("source_asset"),
                "source_asset_dir": case.get("source_asset_dir"),
                "source_file": case.get("source_file"),
                "classes_json": [],
                "actions_json": [],
            },
        )
        classes = set(_json_list(row.get("classes_json")))
        actions = set(_json_list(row.get("actions_json")))
        if case.get("class"):
            classes.add(str(case.get("class")))
        if case.get("action_name"):
            actions.add(str(case.get("action_name")))
        row["classes_json"] = json.dumps(sorted(classes), ensure_ascii=False)
        row["actions_json"] = json.dumps(sorted(actions), ensure_ascii=False)
    return sorted(by_asset.values(), key=lambda r: (str(r.get("asset_collection")), str(r.get("asset_name"))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build release-local benchmark manifests for ArtiMo 2D/3D evaluation.")
    parser.add_argument("--release_root", type=Path, default=Path("benchmark_release"))
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument(
        "--data_root",
        type=Path,
        action="append",
        default=[],
        help="Optional root containing causal_data/not_causal_data asset folders. Can be passed multiple times.",
    )
    parser.add_argument(
        "--prediction_root",
        type=Path,
        action="append",
        default=[],
        help="Optional root containing one method's prediction outputs. If found, only that variant is added to matching cases.",
    )
    parser.add_argument("--prediction_variant", default="full_agent", help="Variant name to use with --prediction_root.")
    parser.add_argument(
        "--simplify_annotation_metadata",
        action="store_true",
        help="Rewrite release annotation metadata to compact source fields.",
    )
    parser.add_argument(
        "--source_annotation_root",
        type=Path,
        default=Path("/home/chunyu/causal_agent/benchmark_annotations"),
        help="Original benchmark annotation root used to recover action prompts when simplifying metadata.",
    )
    parser.add_argument(
        "--source_release_annotation_root",
        type=Path,
        default=Path("/home/chunyu/causal_agent/benchmark_release/annotations"),
        help="Original release annotation root used to recover compact source dataset/file metadata.",
    )
    parser.add_argument(
        "--partnet_root",
        type=Path,
        action="append",
        default=[],
        help="PartNet-Mobility source root used to map model_id/anno_id to raw numeric ids. Can be passed multiple times.",
    )
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    if not release_root.exists():
        raise SystemExit(f"Release root not found: {release_root}")
    source_annotation_root = args.source_annotation_root.resolve() if args.source_annotation_root and args.source_annotation_root.exists() else None
    source_release_annotation_root = (
        args.source_release_annotation_root.resolve()
        if args.source_release_annotation_root and args.source_release_annotation_root.exists()
        else None
    )
    out_dir = (args.out_dir or (release_root / "manifests")).resolve()
    cwd = REPO_ROOT.resolve()

    default_data_roots = [p for p in [REPO_ROOT / "data", Path("/home/chunyu/causal_agent/data")] if p.exists()]
    data_roots = [p.resolve() for p in [*default_data_roots, *(args.data_root or [])] if p.exists()]
    prediction_roots = [p.resolve() for p in (args.prediction_root or []) if p.exists()]
    default_partnet_roots = [
        p
        for p in [
            REPO_ROOT / "partnet-mobility",
            Path("/home/chunyu/causal_agent/partnet-mobility"),
        ]
        if p.exists()
    ]
    partnet_roots = [p.resolve() for p in [*default_partnet_roots, *(args.partnet_root or [])] if p.exists()]
    partnet_index = build_partnet_source_index(partnet_roots)
    simplified_annotations = (
        simplify_annotation_metadata_in_place(release_root, source_annotation_root, source_release_annotation_root, partnet_index)
        if args.simplify_annotation_metadata
        else 0
    )

    cases, case_summary = build_case_manifest(
        release_root,
        data_roots=data_roots,
        prediction_roots=prediction_roots,
        prediction_variant=str(args.prediction_variant or "full_agent"),
        partnet_index=partnet_index,
        cwd=cwd,
    )
    phase_rows = build_phase_static_manifest(release_root, cwd=cwd)
    asset_rows = build_asset_source_manifest(cases)

    eval_manifest_path = out_dir / "eval_manifest_225.json"
    asset_json_path = out_dir / "asset_source_manifest.json"
    asset_csv_path = out_dir / "asset_source_manifest.csv"
    phase_json_path = out_dir / "phase_static_manifest.json"
    phase_csv_path = out_dir / "phase_static_manifest.csv"
    summary_path = out_dir / "manifest_summary.json"

    _write_json(
        eval_manifest_path,
        {
            "cases": cases,
            "metadata": {
                "release_root": _path_for_manifest(release_root, cwd),
                "description": "Each case maps one benchmark annotation and one source asset to its ground-truth animation artifacts. Add a variants entry only when evaluating predictions.",
                **case_summary,
            },
        },
    )
    _write_json(asset_json_path, {"assets": asset_rows})
    _write_csv(
        asset_csv_path,
        asset_rows,
        [
            "asset_name",
            "asset_collection",
            "source_dataset",
            "source_asset",
            "source_asset_dir",
            "source_file",
            "classes_json",
            "actions_json",
        ],
    )
    _write_json(phase_json_path, {"phases": phase_rows})
    _write_csv(
        phase_csv_path,
        phase_rows,
        [
            "case_id",
            "class",
            "asset_name",
            "action_name",
            "phase_index",
            "phase_id",
            "phase_endpoint_time_s",
            "trajectory_time_s",
            "static_glb",
            "dynamic_links_json",
        ],
    )
    summary = {
        **case_summary,
        "num_assets": len(asset_rows),
        "num_static_phase_rows": len(phase_rows),
        "eval_manifest": _path_for_manifest(eval_manifest_path, cwd),
        "asset_source_manifest_json": _path_for_manifest(asset_json_path, cwd),
        "asset_source_manifest_csv": _path_for_manifest(asset_csv_path, cwd),
        "phase_static_manifest_json": _path_for_manifest(phase_json_path, cwd),
        "phase_static_manifest_csv": _path_for_manifest(phase_csv_path, cwd),
    }
    if simplified_annotations:
        summary["simplified_annotation_metadata"] = int(simplified_annotations)
    _write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
