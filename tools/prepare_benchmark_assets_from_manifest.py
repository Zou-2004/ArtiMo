#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import zipfile


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


def _source_asset_dir_relative(row: dict[str, str]) -> Path | None:
    rel = _source_file_relative(row)
    if rel is not None:
        if rel.suffix.lower() in {".usd", ".usda", ".usdc", ".urdf"}:
            return rel.parent
        return rel
    dataset = str(row.get("source_dataset") or "").strip()
    source_asset = str(row.get("source_asset") or "").strip()
    if dataset == "PartNet-Mobility" and source_asset:
        return Path("dataset") / source_asset
    return None


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


def _has_primary_usd(path: Path) -> bool:
    return any(path.glob("*.usd")) or any(path.glob("*.usda")) or any(path.glob("*.usdc"))


def _zip_cache_dir(zip_path: Path, cache_root: Path) -> Path:
    key = hashlib.sha1(str(zip_path.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_root / f"{zip_path.stem[:48]}_{key}"


def _zip_prefix_exists(names: set[str], prefix: Path) -> str | None:
    text = str(prefix).strip("/").replace("\\", "/")
    if not text:
        return None
    dir_prefix = text.rstrip("/") + "/"
    if any(name.startswith(dir_prefix) for name in names):
        return text.rstrip("/")
    return None


def _extract_zip_prefix(zip_path: Path, prefix: str, cache_root: Path) -> Path:
    out_root = _zip_cache_dir(zip_path, cache_root)
    done_marker = out_root / ".extract_done" / hashlib.sha1(prefix.encode("utf-8")).hexdigest()
    target = out_root / prefix
    if done_marker.exists() and target.exists():
        return target
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        prefix_slash = prefix.rstrip("/") + "/"
        members = [name for name in zf.namelist() if name.startswith(prefix_slash)]
        if not members:
            raise FileNotFoundError(f"{zip_path}: prefix not found: {prefix}")
        for member in members:
            zf.extract(member, out_root)
    done_marker.write_text("ok\n", encoding="utf-8")
    return target


def _zip_source_dir(row: dict[str, str], zip_path: Path, cache_root: Path) -> Path | None:
    rel = _source_asset_dir_relative(row)
    source_asset = str(row.get("source_asset") or "").strip()
    source_asset_dir = str(row.get("source_asset_dir") or "").strip()
    candidates: list[Path] = []
    if rel is not None:
        candidates.append(rel)
    if source_asset:
        candidates.append(Path(source_asset))
        candidates.append(Path("dataset") / source_asset)
    if source_asset_dir:
        candidates.append(Path(source_asset_dir))
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for cand in candidates:
            prefix = _zip_prefix_exists(names, cand)
            if prefix is not None:
                return _extract_zip_prefix(zip_path, prefix, cache_root).resolve()
    return None


def _candidate_source_dirs(row: dict[str, str], roots: list[Path]) -> list[Path]:
    source_asset = str(row.get("source_asset") or "").strip()
    source_asset_dir = str(row.get("source_asset_dir") or "").strip()
    asset_name = str(row.get("asset_name") or "").strip()
    rel_source_file = _source_file_relative(row)
    converted_dir = _converted_dir_from_source_file(row)
    candidates: list[Path] = []
    for root in roots:
        if root.is_file():
            continue
        if source_asset:
            candidates.append(root / source_asset)
            candidates.extend(sorted(root.glob(f"{source_asset}_*")))
            candidates.append(root / "dataset" / source_asset)
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


def _find_source_dir(row: dict[str, str], source_roots: dict[str, list[Path]], raw_extract_root: Path) -> Path | None:
    dataset = str(row.get("source_dataset") or "").strip()
    roots = source_roots.get(dataset) or []
    for cand in _candidate_source_dirs(row, roots):
        if (cand / "mobility.urdf").exists():
            return cand.resolve()
    for root in roots:
        if root.is_file() and zipfile.is_zipfile(root):
            cand = _zip_source_dir(row, root, raw_extract_root)
            if cand is not None and (cand / "mobility.urdf").exists():
                return cand.resolve()
    return None


def _find_raw_source_dir(row: dict[str, str], source_roots: dict[str, list[Path]], raw_extract_root: Path) -> Path | None:
    dataset = str(row.get("source_dataset") or "").strip()
    if dataset not in {"ArtVIP", "Lightwheel"}:
        return None
    roots = source_roots.get(dataset) or []
    for cand in _candidate_source_dirs(row, roots):
        if _has_primary_usd(cand):
            return cand.resolve()
    for root in roots:
        if root.is_file() and zipfile.is_zipfile(root):
            cand = _zip_source_dir(row, root, raw_extract_root)
            if cand is not None and _has_primary_usd(cand):
                return cand.resolve()
    return None


def _ancestor_named(path: Path, name: str) -> Path | None:
    for cand in [path, *path.parents]:
        if cand.name == name:
            return cand
    return None


def _raw_dataset_root(dataset: str, raw_dir: Path, source_roots: list[Path]) -> Path:
    if dataset == "ArtVIP":
        art_root = _ancestor_named(raw_dir, "Articulated_objects")
        if art_root is not None:
            return art_root.parent
    if dataset == "Lightwheel":
        lw_root = _ancestor_named(raw_dir, "Lightwheel_OpenSource")
        if lw_root is not None:
            return lw_root
    for root in source_roots:
        if root.is_dir():
            try:
                raw_dir.relative_to(root)
                return root
            except ValueError:
                pass
    return raw_dir.parent


def _convert_raw_usd_asset(
    row: dict[str, str],
    raw_dir: Path,
    source_roots: dict[str, list[Path]],
    converted_cache_root: Path,
    python_exe: str,
) -> Path | None:
    dataset = str(row.get("source_dataset") or "").strip()
    converted_dir = _converted_dir_from_source_file(row)
    if not converted_dir:
        return None
    dst = converted_cache_root / dataset / converted_dir
    if (dst / "mobility.urdf").exists():
        return dst.resolve()
    converted_cache_root.mkdir(parents=True, exist_ok=True)
    roots = source_roots.get(dataset) or []
    dataset_root = _raw_dataset_root(dataset, raw_dir, roots)
    script = Path(__file__).resolve().parent / (
        "convert_artvip_to_data_format.py" if dataset == "ArtVIP" else "convert_lightwheel_to_data_format.py"
    )
    if not script.exists():
        raise FileNotFoundError(f"Missing conversion script: {script}")
    cmd = [
        python_exe,
        str(script),
        "--out_root",
        str(converted_cache_root / dataset),
        "--asset_dir",
        str(raw_dir),
        "--keep_existing",
    ]
    if dataset == "ArtVIP":
        cmd[2:2] = ["--artvip_root", str(dataset_root)]
    else:
        cmd[2:2] = ["--lightwheel_root", str(dataset_root)]
    subprocess.run(cmd, check=True)
    return dst.resolve() if (dst / "mobility.urdf").exists() else None


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
            "Dataset source root as DATASET=PATH, e.g. PartNet-Mobility=/path/to/partnet-mobility-v0.zip. "
            "Pass one entry per downloaded raw dataset, source zip, or already-converted dataset root. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--copy_mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--asset", action="append", default=[], help="Optional benchmark asset_name filter. Can be passed multiple times.")
    parser.add_argument(
        "--raw_extract_root",
        type=Path,
        default=None,
        help="Cache directory for extracting selected assets from source zip files. Defaults to <out_data_root>/.raw_source_cache.",
    )
    parser.add_argument(
        "--converted_cache_root",
        type=Path,
        default=None,
        help="Cache directory for raw ArtVIP/Lightwheel USD to URDF conversion. Defaults to <out_data_root>/.converted_source_cache.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for raw ArtVIP/Lightwheel USD conversion.",
    )
    parser.add_argument(
        "--no_convert_raw_usd",
        action="store_true",
        help="Only accept already-converted folders with mobility.urdf; do not run ArtVIP/Lightwheel USD conversion.",
    )
    args = parser.parse_args()

    rows = _read_rows(args.asset_manifest.expanduser().resolve())
    wanted = {str(x).strip() for x in args.asset if str(x).strip()}
    if wanted:
        rows = [row for row in rows if str(row.get("asset_name") or "") in wanted]
    source_roots = _parse_source_roots(list(args.source_root or []))
    out_root = args.out_data_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    raw_extract_root = (args.raw_extract_root or (out_root / ".raw_source_cache")).expanduser().resolve()
    converted_cache_root = (args.converted_cache_root or (out_root / ".converted_source_cache")).expanduser().resolve()

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
        src = _find_source_dir(row, source_roots, raw_extract_root)
        if src is None and not args.no_convert_raw_usd:
            raw_src = _find_raw_source_dir(row, source_roots, raw_extract_root)
            if raw_src is not None:
                try:
                    src = _convert_raw_usd_asset(row, raw_src, source_roots, converted_cache_root, str(args.python))
                    result["raw_source_dir"] = str(raw_src)
                except Exception as exc:
                    result["status"] = "failed"
                    result["error"] = f"raw_conversion_{type(exc).__name__}: {exc}"
                    failed += 1
                    results.append(result)
                    continue
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
