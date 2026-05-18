#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _parse_source_roots(items: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for item in items:
        if "=" not in str(item):
            raise SystemExit(f"--source_root must be DATASET=PATH, got: {item}")
        dataset, raw_path = str(item).split("=", 1)
        dataset = dataset.strip()
        path = Path(raw_path).expanduser().resolve()
        if not dataset or not path.exists():
            raise SystemExit(f"Invalid source root: {item}")
        out.setdefault(dataset, []).append(path)
    return out


def _source_file_relative(row: dict[str, str]) -> Path | None:
    dataset = str(row.get("source_dataset") or "").strip()
    source_file = str(row.get("source_file") or "").strip()
    if not source_file:
        return None
    prefix = f"{dataset}/"
    if source_file.startswith(prefix):
        source_file = source_file[len(prefix) :]
    return Path(source_file)


def _converted_dir_from_source_file(row: dict[str, str]) -> str | None:
    dataset = str(row.get("source_dataset") or "").strip()
    rel = _source_file_relative(row)
    if rel is None:
        return None
    parts = list(rel.parts)
    if dataset == "ArtVIP":
        if "Articulated_objects" in parts:
            idx = parts.index("Articulated_objects")
            asset_parts = parts[idx + 1 : -1]
            return "__".join(asset_parts) if asset_parts else None
        return "__".join(parts[:-1]) if len(parts) > 1 else None
    if dataset == "Lightwheel":
        lowered = [p.lower() for p in parts]
        if "manipulation" in lowered:
            idx = lowered.index("manipulation")
            asset_parts = parts[idx : -1]
        else:
            asset_parts = parts[:-1]
        if asset_parts:
            return "lightwheel__" + "__".join(p.lower() for p in asset_parts)
    return None


def _candidate_source_dirs(row: dict[str, str], roots: list[Path]) -> list[Path]:
    source_asset = str(row.get("source_asset") or "").strip()
    source_asset_dir = str(row.get("source_asset_dir") or "").strip()
    asset_name = str(row.get("asset_name") or "").strip()
    rel_source_file = _source_file_relative(row)
    converted_dir = _converted_dir_from_source_file(row)
    candidates: list[Path] = []
    for root in roots:
        if source_asset:
            candidates.append(root / source_asset)
            candidates.extend(sorted(root.glob(f"{source_asset}_*")))
        if source_asset_dir:
            candidates.append(root / source_asset_dir)
        if converted_dir:
            candidates.append(root / converted_dir)
        if rel_source_file is not None:
            p = root / rel_source_file
            candidates.append(p if p.is_dir() else p.parent)
        if asset_name:
            candidates.append(root / asset_name)
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


def _find_source_dir(row: dict[str, str], source_roots: dict[str, list[Path]]) -> Path | None:
    dataset = str(row.get("source_dataset") or "").strip()
    roots = source_roots.get(dataset) or []
    for cand in _candidate_source_dirs(row, roots):
        if (cand / "mobility.urdf").exists():
            return cand.resolve()
    return None


def _link_or_copy(src: Path, dst: Path, copy_mode: str) -> str:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return "symlink"
    if copy_mode == "copy":
        shutil.copytree(src, dst)
        return "copy"
    try:
        shutil.copytree(src, dst, copy_function=os.link)
        return "hardlink"
    except OSError:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return "copy_fallback"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare benchmark asset folders from asset_source_manifest.csv by copying/linking source assets "
            "to <data_root>/<asset_collection>/<asset_name>. The causal_data/not_causal_data split is read "
            "from the manifest; users only provide dataset source roots."
        )
    )
    parser.add_argument("--asset_manifest", type=Path, default=Path("benchmark_release/manifests/asset_source_manifest.csv"))
    parser.add_argument("--out_data_root", type=Path, required=True)
    parser.add_argument(
        "--source_root",
        action="append",
        default=[],
        help=(
            "Dataset source root as DATASET=PATH, e.g. PartNet-Mobility=/path/to/partnet-mobility. "
            "Pass one entry per downloaded/converted dataset root. Can be passed multiple times."
        ),
    )
    parser.add_argument("--copy_mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--asset", action="append", default=[], help="Optional benchmark asset_name filter. Can be passed multiple times.")
    args = parser.parse_args()

    rows = _read_rows(args.asset_manifest.expanduser().resolve())
    wanted = {str(x).strip() for x in args.asset if str(x).strip()}
    if wanted:
        rows = [row for row in rows if str(row.get("asset_name") or "") in wanted]
    source_roots = _parse_source_roots(list(args.source_root or []))
    out_root = args.out_data_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    prepared = skipped = missing = failed = 0
    for row in rows:
        asset_name = str(row.get("asset_name") or "").strip()
        collection = str(row.get("asset_collection") or "causal_data").strip() or "causal_data"
        dst = out_root / collection / asset_name
        result = {
            "asset_name": asset_name,
            "asset_collection": collection,
            "source_dataset": row.get("source_dataset"),
            "source_asset": row.get("source_asset"),
            "source_asset_dir": row.get("source_asset_dir"),
            "target_asset_dir": str(dst),
        }
        if dst.exists() and not args.overwrite:
            result["status"] = "skipped_exists"
            skipped += 1
            results.append(result)
            continue
        src = _find_source_dir(row, source_roots)
        if src is None:
            result["status"] = "missing_source"
            missing += 1
            results.append(result)
            continue
        try:
            mode = _link_or_copy(src, dst, str(args.copy_mode))
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1
            results.append(result)
            continue
        result["status"] = "prepared"
        result["source_dir"] = str(src)
        result["copy_mode_used"] = mode
        prepared += 1
        results.append(result)

    manifest_path = out_root / "benchmark_asset_prepare_manifest.json"
    manifest_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "requested": len(rows),
        "prepared": prepared,
        "skipped_exists": skipped,
        "missing_source": missing,
        "failed": failed,
        "manifest": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
