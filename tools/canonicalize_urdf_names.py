#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


TRAILING_INT_RE = re.compile(r"(\d+)$")


def _extract_trailing_int(name: str) -> int | None:
    m = TRAILING_INT_RE.search(str(name or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _children_by_tag(root: ET.Element, tag: str) -> list[ET.Element]:
    return [el for el in list(root) if _local_name(el.tag) == tag]


def _first_child(el: ET.Element, tag: str) -> ET.Element | None:
    for child in list(el):
        if _local_name(child.tag) == tag:
            return child
    return None


def _compact_link_mapping(link_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[int] = set()

    # Match the dataset converters: prefer numeric ids already present in
    # source body/link names, then assign deterministic free ids.
    for old in link_names:
        idx = _extract_trailing_int(old)
        if idx is None or idx in used:
            continue
        mapping[old] = f"link_{idx}"
        used.add(idx)

    next_id = 0
    for old in link_names:
        if old in mapping:
            continue
        while next_id in used:
            next_id += 1
        mapping[old] = f"link_{next_id}"
        used.add(next_id)
        next_id += 1
    return mapping


def _compact_joint_mapping(joint_names: list[str]) -> dict[str, str]:
    return {old: f"joint_{idx}" for idx, old in enumerate(joint_names)}


def canonicalize_urdf_names(
    urdf_path: Path,
    *,
    rename_joints: bool = True,
    write_map: bool = True,
    backup: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    link_elements = _children_by_tag(root, "link")
    joint_elements = _children_by_tag(root, "joint")
    link_names = [str(el.get("name") or "") for el in link_elements if str(el.get("name") or "")]
    joint_names = [str(el.get("name") or "") for el in joint_elements if str(el.get("name") or "")]

    link_map = _compact_link_mapping(link_names)
    joint_map = _compact_joint_mapping(joint_names) if rename_joints else {name: name for name in joint_names}

    changed = False
    for el in link_elements:
        old = str(el.get("name") or "")
        new = link_map.get(old)
        if new and new != old:
            el.set("name", new)
            changed = True

    for joint_el in joint_elements:
        old_joint = str(joint_el.get("name") or "")
        new_joint = joint_map.get(old_joint)
        if new_joint and new_joint != old_joint:
            joint_el.set("name", new_joint)
            changed = True

        parent_el = _first_child(joint_el, "parent")
        if parent_el is not None:
            old_parent = str(parent_el.get("link") or "")
            new_parent = link_map.get(old_parent)
            if new_parent and new_parent != old_parent:
                parent_el.set("link", new_parent)
                changed = True

        child_el = _first_child(joint_el, "child")
        if child_el is not None:
            old_child = str(child_el.get("link") or "")
            new_child = link_map.get(old_child)
            if new_child and new_child != old_child:
                child_el.set("link", new_child)
                changed = True

        mimic_el = _first_child(joint_el, "mimic")
        if mimic_el is not None and rename_joints:
            old_mimic = str(mimic_el.get("joint") or "")
            new_mimic = joint_map.get(old_mimic)
            if new_mimic and new_mimic != old_mimic:
                mimic_el.set("joint", new_mimic)
                changed = True

    for el in root.iter():
        if _local_name(el.tag) == "gazebo":
            old_ref = str(el.get("reference") or "")
            new_ref = link_map.get(old_ref) or (joint_map.get(old_ref) if rename_joints else None)
            if new_ref and new_ref != old_ref:
                el.set("reference", new_ref)
                changed = True
        if _local_name(el.tag) == "joint" and el not in joint_elements:
            old_name = str(el.get("name") or "")
            new_name = joint_map.get(old_name) if rename_joints else None
            if new_name and new_name != old_name:
                el.set("name", new_name)
                changed = True

    summary = {
        "urdf": str(urdf_path),
        "changed": bool(changed),
        "rename_joints": bool(rename_joints),
        "link_name_map_old_to_new": link_map,
        "link_name_map_new_to_old": {v: k for k, v in link_map.items()},
        "joint_name_map_old_to_new": joint_map,
        "joint_name_map_new_to_old": {v: k for k, v in joint_map.items()},
        "num_links": len(link_names),
        "num_joints": len(joint_names),
    }

    if dry_run:
        return summary

    if changed:
        if backup:
            backup_path = urdf_path.with_suffix(urdf_path.suffix + ".original_names.bak")
            if not backup_path.exists():
                shutil.copy2(urdf_path, backup_path)
            summary["backup"] = str(backup_path)
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    if write_map:
        map_path = urdf_path.parent / "urdf_name_map.json"
        map_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["map_path"] = str(map_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonicalize URDF link/joint names to link_<id> and joint_<id> before ArtiMo preprocessing."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--asset_root", type=Path, help="Asset directory containing mobility.urdf")
    group.add_argument("--urdf", type=Path, help="Explicit URDF path")
    parser.add_argument("--keep_joint_names", action="store_true", help="Only canonicalize link names")
    parser.add_argument("--no_backup", action="store_true", help="Do not write mobility.urdf.original_names.bak")
    parser.add_argument("--no_map", action="store_true", help="Do not write urdf_name_map.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    urdf = args.urdf if args.urdf is not None else Path(args.asset_root) / "mobility.urdf"
    summary = canonicalize_urdf_names(
        urdf,
        rename_joints=not bool(args.keep_joint_names),
        write_map=not bool(args.no_map),
        backup=not bool(args.no_backup),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
