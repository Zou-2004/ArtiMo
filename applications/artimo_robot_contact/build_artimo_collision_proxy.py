#!/usr/bin/env python3
"""Derive a physics URDF whose collision geometry matches declared proxies.

Source collision meshes are often loaded by PyBullet as a single convex hull per
``<collision>`` tag.  For a concave part -- a bin lid, a door frame, a handle
recess -- that hull fills the concavity and manufactures contacts that the real
geometry does not have, which makes an otherwise correct contact candidate look
like a collision.

This tool rewrites only collision geometry, from a declarative proxy spec.  It
never touches visual geometry, joint origins, axes, or limits, so the mechanism
the ArtiMo plan describes is preserved exactly.  It contains no asset registry:
every link name, mode, and bound comes from the spec file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p


APP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = APP_ROOT.parents[1]
DEBUG_ROOT = (REPO_ROOT / ".artimo-runs").resolve()
MODES = ("convex_decomposition", "box", "keep")


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _require_debug_output(output: Path, label: str) -> Path:
    """Keep generated, task-specific geometry out of reusable configuration."""
    resolved = output.expanduser().resolve()
    try:
        resolved.relative_to(DEBUG_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{label} must be below {DEBUG_ROOT}; generated asset geometry must "
            "not be stored in a reusable input directory"
        ) from exc
    return resolved


def _split_obj_groups(source: Path, destination: Path, stem: str) -> list[Path]:
    """Write each ``o``/``g`` group of an OBJ as its own file.

    V-HACD emits one OBJ containing many convex pieces.  PyBullet collapses a
    multi-group OBJ into a single hull, which would defeat the decomposition
    entirely, so each piece has to become its own ``<collision>`` tag.
    """
    vertices: list[str] = []
    groups: list[list[list[int]]] = []
    current: list[list[int]] | None = None
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            # Store the parsed coordinates, not the raw line.  V-HACD output can
            # omit the final newline, and re-emitting raw text would then fuse two
            # vertices into one malformed record.
            vertices.append(" ".join(fields[:4]))
        elif fields[0] in {"o", "g"}:
            current = []
            groups.append(current)
        elif fields[0] == "f":
            if current is None:
                current = []
                groups.append(current)
            current.append([int(field.split("/")[0]) for field in fields[1:]])
    groups = [group for group in groups if group]
    if not groups:
        raise ValueError(f"No faces found in decomposition output: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, faces in enumerate(groups):
        # OBJ face indices are global; re-index each piece so it stands alone.
        used = sorted({vertex for face in faces for vertex in face})
        remapped = {vertex: slot + 1 for slot, vertex in enumerate(used)}
        piece = destination / f"{stem}_{index:03d}.obj"
        with piece.open("w", encoding="utf-8") as handle:
            for vertex in used:
                handle.write(vertices[vertex - 1] + "\n")
            for face in faces:
                handle.write("f " + " ".join(str(remapped[v]) for v in face) + "\n")
        written.append(piece)
    return written


def _link_local_aabb(urdf: Path, link_name: str) -> tuple[list[float], list[float]]:
    """Measure a link's collision AABB in its own frame."""
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet measurement client")
    try:
        body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=client)
        index = None
        base = p.getBodyInfo(body, physicsClientId=client)[0].decode("utf-8")
        if base == link_name:
            index = -1
        for joint in range(p.getNumJoints(body, physicsClientId=client)):
            info = p.getJointInfo(body, joint, physicsClientId=client)
            if info[12].decode("utf-8") == link_name:
                index = joint
        if index is None:
            raise KeyError(f"URDF has no link named {link_name!r}")
        p.performCollisionDetection(physicsClientId=client)
        low, high = p.getAABB(body, index, physicsClientId=client)
        if index == -1:
            origin, rotation = p.getBasePositionAndOrientation(body, physicsClientId=client)
        else:
            state = p.getLinkState(body, index, computeForwardKinematics=True, physicsClientId=client)
            origin, rotation = state[4], state[5]
        inverse = p.invertTransform(origin, rotation)
        corners = [
            p.multiplyTransforms(inverse[0], inverse[1], corner, [0.0, 0.0, 0.0, 1.0])[0]
            for corner in (
                (low[0], low[1], low[2]), (high[0], low[1], low[2]),
                (low[0], high[1], low[2]), (high[0], high[1], low[2]),
                (low[0], low[1], high[2]), (high[0], low[1], high[2]),
                (low[0], high[1], high[2]), (high[0], high[1], high[2]),
            )
        ]
        array = np.asarray(corners, dtype=np.float64)
        return array.min(axis=0).tolist(), array.max(axis=0).tolist()
    finally:
        p.disconnect(client)


def _collision_tags_for_meshes(
    parent: ET.Element, pieces: list[tuple[str, ET.Element | None]]
) -> None:
    """Append one ``<collision>`` per proxy piece, each keeping its own origin.

    Every tag gets a fresh origin element; reusing one object would alias it
    across siblings and serialize invalid XML.  Origins are tracked per source
    mesh because a link's collision tags need not share a single transform.
    """
    for relative, origin in pieces:
        collision = ET.SubElement(parent, "collision")
        if origin is not None:
            ET.SubElement(collision, "origin", dict(origin.attrib))
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", {"filename": relative})


def _mesh_thinnest_extent(mesh_path: Path) -> float:
    """Return the smallest bounding-box side of an OBJ, in metres.

    A source file whose thinnest side is zero is an open boundary sheet, not a
    solid.  Convex decomposition cannot produce meaningful pieces from it and
    instead emits wide flat slabs; because a convex collision shape carries a few
    millimetres of margin, those slabs turn into a wall that seals an opening the
    real geometry leaves clear.
    """
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    for line in mesh_path.read_text(errors="replace").splitlines():
        if not line.startswith("v "):
            continue
        fields = line.split()
        for axis in range(3):
            value = float(fields[axis + 1])
            low[axis] = min(low[axis], value)
            high[axis] = max(high[axis], value)
    if any(v == float("inf") for v in low):
        return 0.0
    return min(high[axis] - low[axis] for axis in range(3))


def build(spec_path: Path, output_urdf: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec, dict) or spec.get("schema_version") != 1:
        raise ValueError("Proxy spec must be an object with schema_version 1")
    source_urdf = _resolve(spec["source_urdf"])
    if not source_urdf.is_file():
        raise FileNotFoundError(source_urdf)
    proxies = spec.get("links", {})
    if not isinstance(proxies, dict) or not proxies:
        raise ValueError("Proxy spec must declare at least one entry in links")

    output_urdf = _require_debug_output(output_urdf, "out-urdf")
    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    asset_root = output_urdf.parent / f"{output_urdf.stem}_collision"
    shutil.rmtree(asset_root, ignore_errors=True)

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    report: dict[str, Any] = {"source_urdf": str(source_urdf), "links": {}}

    # The derived URDF lives in a different directory from the source, so every
    # inherited mesh reference has to be re-pointed at the original file.  Visual
    # geometry is only re-pathed, never replaced: the rendered object must remain
    # the asset the ArtiMo plan describes.
    for element in root.iter("mesh"):
        reference = element.attrib.get("filename")
        if not reference or reference.startswith("package://"):
            continue
        resolved = (source_urdf.parent / reference).resolve()
        if resolved.is_file():
            element.attrib["filename"] = str(resolved)

    for link in root.iter("link"):
        name = link.attrib.get("name")
        if name not in proxies:
            continue
        entry = proxies[name]
        mode = entry.get("mode")
        if mode not in MODES:
            raise ValueError(f"link {name!r} has unsupported proxy mode {mode!r}")
        if mode == "keep":
            report["links"][name] = {"mode": mode, "collision_shapes": None}
            continue

        existing = list(link.findall("collision"))
        # Keep each source mesh paired with the origin declared alongside it, so
        # a rewritten link cannot silently shift its collision geometry.
        source_meshes = [
            (element.attrib["filename"], collision.find("origin"))
            for collision in existing
            for element in collision.iter("mesh")
            if element.attrib.get("filename")
        ]
        for collision in existing:
            link.remove(collision)

        if mode == "convex_decomposition":
            if not source_meshes:
                raise ValueError(f"link {name!r} has no source collision mesh to decompose")
            resolution = int(entry.get("resolution", 200000))
            maximum_hulls = int(entry.get("maximum_convex_hulls", 32))
            # Sheets thinner than this are treated as visual boundary faces and
            # dropped from collision, because decomposing them manufactures a wall.
            sheet_threshold = float(entry.get("drop_thinner_than_m", 0.001))
            pieces: list[tuple[str, ET.Element | None]] = []
            dropped: list[str] = []
            for mesh_index, (reference, origin) in enumerate(source_meshes):
                # References were rewritten to absolute paths above, but accept a
                # source-relative one too so the spec stays robust.
                mesh_path = Path(reference)
                if not mesh_path.is_absolute():
                    mesh_path = source_urdf.parent / reference
                mesh_path = mesh_path.resolve()
                if not mesh_path.is_file():
                    raise FileNotFoundError(f"link {name!r} references missing mesh {reference}")
                thinnest = _mesh_thinnest_extent(mesh_path)
                if thinnest < sheet_threshold:
                    dropped.append(f"{mesh_path.name} (thinnest {thinnest:.6f} m)")
                    continue
                stem = f"{name}_{mesh_index:02d}"
                decomposed = asset_root / f"{stem}_vhacd.obj"
                decomposed.parent.mkdir(parents=True, exist_ok=True)
                p.vhacd(
                    str(mesh_path),
                    str(decomposed),
                    str(asset_root / f"{stem}_vhacd.log"),
                    resolution=resolution,
                    maxNumVerticesPerCH=int(entry.get("maximum_vertices_per_hull", 64)),
                )
                written = _split_obj_groups(decomposed, asset_root / stem, stem)
                if len(written) > maximum_hulls:
                    raise ValueError(
                        f"link {name!r} decomposed into {len(written)} hulls, above the "
                        f"declared maximum_convex_hulls {maximum_hulls}"
                    )
                pieces.extend(
                    (str(piece.relative_to(output_urdf.parent)), origin) for piece in written
                )
            if not pieces:
                raise ValueError(
                    f"link {name!r} has no solid collision mesh left after dropping "
                    f"zero-thickness sheets: {dropped}"
                )
            _collision_tags_for_meshes(link, pieces)
            report["links"][name] = {
                "mode": mode,
                "collision_shapes": len(pieces),
                "dropped_zero_thickness_sheets": dropped,
            }
        else:
            low, high = _link_local_aabb(source_urdf, name)
            inflate = float(entry.get("inflate_m", 0.0))
            size = [max(high[axis] - low[axis], 1e-4) + 2.0 * inflate for axis in range(3)]
            center = [(high[axis] + low[axis]) * 0.5 for axis in range(3)]
            collision = ET.SubElement(link, "collision")
            ET.SubElement(
                collision,
                "origin",
                {"xyz": " ".join(f"{value:.9f}" for value in center), "rpy": "0 0 0"},
            )
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "box", {"size": " ".join(f"{value:.9f}" for value in size)})
            report["links"][name] = {
                "mode": mode,
                "collision_shapes": 1,
                "box_size_m": size,
                "box_center_link_m": center,
            }

    missing = sorted(set(proxies) - set(report["links"]))
    if missing:
        raise KeyError(f"Proxy spec names links absent from the URDF: {missing}")

    # Bullet's URDF reader truncates over-long lines, which silently corrupts the
    # last mesh filename on a link carrying dozens of proxy pieces.  Indenting
    # puts every <collision> on its own short line and avoids that limit.
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    # Loading the result is the only real proof that every rewritten tag parses
    # and that the joint tree survived untouched.
    client = p.connect(p.DIRECT)
    try:
        body = p.loadURDF(str(output_urdf), useFixedBase=True, physicsClientId=client)
        report["loaded_joint_count"] = p.getNumJoints(body, physicsClientId=client)
        report["collision_shape_counts"] = {
            "base": len(p.getCollisionShapeData(body, -1, physicsClientId=client)),
        }
        for joint in range(p.getNumJoints(body, physicsClientId=client)):
            link_name = p.getJointInfo(body, joint, physicsClientId=client)[12].decode("utf-8")
            report["collision_shape_counts"][link_name] = len(
                p.getCollisionShapeData(body, joint, physicsClientId=client)
            )
    finally:
        p.disconnect(client)
    report["output_urdf"] = str(output_urdf)
    _require_finite_geometry(output_urdf, report)
    return report


def _require_finite_geometry(urdf: Path, report: dict[str, Any]) -> None:
    """Reject a proxy whose collision geometry is degenerate or unbounded.

    A single malformed convex piece yields an astronomically large AABB, which
    then silently ruins grounding and makes every clearance measurement
    meaningless.  Catching it here keeps that failure from reaching a rollout.
    """
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("Could not connect PyBullet validation client")
    try:
        body = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=client)
        p.performCollisionDetection(physicsClientId=client)
        boxes = [(-1, p.getAABB(body, -1, physicsClientId=client))]
        boxes.extend(
            (index, p.getAABB(body, index, physicsClientId=client))
            for index in range(p.getNumJoints(body, physicsClientId=client))
        )
        extents: dict[str, list[float]] = {}
        for index, (low, high) in boxes:
            name = (
                p.getBodyInfo(body, physicsClientId=client)[0].decode("utf-8")
                if index == -1
                else p.getJointInfo(body, index, physicsClientId=client)[12].decode("utf-8")
            )
            extent = [float(high[axis]) - float(low[axis]) for axis in range(3)]
            if not all(np.isfinite(extent)) or max(extent) > 100.0:
                raise ValueError(
                    f"link {name!r} has a degenerate collision AABB (extent {extent}); "
                    "one decomposed convex piece is malformed"
                )
            extents[name] = extent
        report["collision_aabb_extent_m"] = extents
    finally:
        p.disconnect(client)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-urdf", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        report = build(args.spec.expanduser().resolve(), args.out_urdf)
        payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        if args.report is not None:
            report_path = _require_debug_output(args.report, "report")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(payload, encoding="utf-8")
        print(payload, end="")
        return 0
    except Exception as exc:
        print(f"Collision proxy build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
