#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Iterable


def _sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def _load_result_root(result_path: Path) -> dict:
    obj = json.loads(result_path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                return item
        return {}
    return obj if isinstance(obj, dict) else {}


def _semantic_slug_for_asset(asset_dir: Path) -> str:
    result_path = asset_dir / "result.json"
    if result_path.exists():
        root = _load_result_root(result_path)
        slug = _sanitize_slug(root.get("name") or root.get("text") or "")
        if slug != "unknown":
            return slug
    meta_path = asset_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if isinstance(meta, dict):
            slug = _sanitize_slug(meta.get("model_cat") or "")
            if slug != "unknown":
                return slug
    return "unknown"


def _iter_asset_dirs(src_root: Path) -> Iterable[Path]:
    for asset_dir in sorted(src_root.iterdir()):
        if asset_dir.is_dir() and (asset_dir / "mobility.urdf").exists():
            yield asset_dir


def _link_or_copy_file(src: Path, dst: Path, copy_mode: str) -> str:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_mode == "symlink":
        dst.symlink_to(src.resolve())
        return "symlink"
    if copy_mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy_fallback"
    shutil.copy2(src, dst)
    return "copy"


def _clone_asset_tree(src_asset: Path, dst_asset: Path, copy_mode: str) -> dict:
    file_count = 0
    mode_counts = {"copy": 0, "hardlink": 0, "symlink": 0, "copy_fallback": 0}
    for src_path in sorted(src_asset.rglob("*")):
        rel = src_path.relative_to(src_asset)
        dst_path = dst_asset / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        mode = _link_or_copy_file(src_path, dst_path, copy_mode)
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        file_count += 1
    return {"files_processed": file_count, "mode_counts": mode_counts}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename PartNet-Mobility assets to <id>_<semantic> and clone them into a new data root."
    )
    parser.add_argument("--src_root", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--copy_mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not src_root.exists():
        raise SystemExit(f"Source root not found: {src_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    total_files = 0
    converted = 0
    skipped = 0

    asset_dirs = list(_iter_asset_dirs(src_root))
    if args.limit is not None:
        asset_dirs = asset_dirs[: max(0, int(args.limit))]

    for src_asset in asset_dirs:
        asset_id = src_asset.name
        semantic_slug = _semantic_slug_for_asset(src_asset)
        out_name = f"{asset_id}_{semantic_slug}"
        dst_asset = out_root / out_name
        if dst_asset.exists():
            if not args.overwrite:
                skipped += 1
                manifest.append(
                    {
                        "asset_id": asset_id,
                        "semantic_slug": semantic_slug,
                        "source_asset": str(src_asset),
                        "output_asset": str(dst_asset),
                        "status": "skipped_exists",
                    }
                )
                continue
            shutil.rmtree(dst_asset)
        dst_asset.mkdir(parents=True, exist_ok=True)
        clone_stats = _clone_asset_tree(src_asset, dst_asset, args.copy_mode)
        total_files += int(clone_stats["files_processed"])
        converted += 1
        summary = {
            "asset_id": asset_id,
            "semantic_slug": semantic_slug,
            "source_asset": str(src_asset),
            "output_asset": str(dst_asset),
            "copy_mode_requested": args.copy_mode,
            "files_processed": int(clone_stats["files_processed"]),
            "mode_counts": clone_stats["mode_counts"],
            "status": "converted",
        }
        (dst_asset / "conversion_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest.append(summary)
        if converted % 100 == 0:
            print(f"[INFO] Converted {converted} assets...")

    manifest_path = out_root / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    index_path = out_root / "conversion_index.csv"
    with index_path.open("w", encoding="utf-8") as f:
        f.write("asset_id,semantic_slug,output_asset,status\n")
        for row in manifest:
            output_asset = Path(str(row.get("output_asset") or "")).name
            f.write(
                f"{row.get('asset_id','')},{row.get('semantic_slug','')},{output_asset},{row.get('status','')}\n"
            )

    print(f"Source root: {src_root}")
    print(f"Output root: {out_root}")
    print(f"Assets converted: {converted}")
    print(f"Assets skipped: {skipped}")
    print(f"Files processed: {total_files}")
    print(f"Manifest: {manifest_path}")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
