#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_plan as rp


SUPPORTED_CONTROL_MODES = {
    "base_velocity",
    "base_velocity_decay",
    "joint_velocity",
    "joint_position",
    "hold_position",
    "spring_return",
    "mode_set",
}


def _sanitize_name(name: str | None) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or ""))
    return clean.strip("_") or "unnamed"


def _stable_indexed_name(raw_name: str | None, default_base: str, idx: int) -> str:
    base_name = _sanitize_name(raw_name or default_base)
    if re.search(r"_\d{2}$", base_name):
        return base_name
    return f"{base_name}_{idx:02d}"


def _rename_files_in_place(root_dir: Path, rename_map: dict[str, str]) -> None:
    staged: list[tuple[Path, Path]] = []
    for idx, (old_name, new_name) in enumerate(sorted(rename_map.items())):
        if old_name == new_name:
            continue
        src = root_dir / old_name
        dst = root_dir / new_name
        if not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(f"Cannot rename {src} -> {dst}: destination already exists")
        tmp = root_dir / f".codex_tmp_{idx:04d}_{src.name}"
        src.rename(tmp)
        staged.append((tmp, dst))
    for tmp, dst in staged:
        tmp.rename(dst)


def _sanitize_asset_in_place(asset_root: Path) -> dict[str, str]:
    urdf_src = next(asset_root.rglob("*.urdf"), None)
    if urdf_src is None:
        raise FileNotFoundError(f"No URDF found under {asset_root}")

    textured_dir = asset_root / "textured_objs"
    if not textured_dir.exists():
        raise FileNotFoundError(f"Missing textured_objs under {asset_root}")

    rename_map: dict[str, str] = {}
    for path in sorted(textured_dir.iterdir()):
        if path.suffix.lower() not in {".obj", ".mtl"}:
            continue
        new_name = _sanitize_name(path.name.replace(".", "_dot_")).replace("_dot_", ".")
        rename_map[path.name] = new_name
    _rename_files_in_place(textured_dir, rename_map)

    material_name_maps: dict[str, dict[str, str]] = {}
    for mtl_path in sorted(textured_dir.glob("*.mtl")):
        text = mtl_path.read_text(encoding="utf-8", errors="ignore")
        mat_map: dict[str, str] = {}
        new_lines: list[str] = []
        mat_counter = 0
        for line in text.splitlines():
            if line.startswith("newmtl "):
                raw_name = line.split(" ", 1)[1].strip()
                safe_name = _sanitize_name(raw_name) or f"material_{mat_counter}"
                if safe_name in mat_map.values():
                    safe_name = f"{safe_name}_{mat_counter}"
                mat_map[raw_name] = safe_name
                new_lines.append(f"newmtl {safe_name}")
                mat_counter += 1
            elif line.startswith("map_"):
                parts = line.split(None, 1)
                if len(parts) == 2:
                    tex_ref = parts[1].strip()
                    if tex_ref and not Path(tex_ref).is_absolute() and not tex_ref.startswith("package://"):
                        tex_path = (mtl_path.parent / tex_ref).resolve()
                        new_lines.append(f"{parts[0]} {tex_path}")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        for old_name, new_name in rename_map.items():
            new_lines = [ln.replace(old_name, new_name) for ln in new_lines]
        mtl_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        material_name_maps[mtl_path.name] = mat_map

    for obj_path in sorted(textured_dir.glob("*.obj")):
        text = obj_path.read_text(encoding="utf-8", errors="ignore")
        new_lines: list[str] = []
        obj_counter = 0
        group_counter = 0
        mat_map = material_name_maps.get(obj_path.with_suffix(".mtl").name, {})
        for line in text.splitlines():
            if line.startswith("mtllib "):
                old_name = line.split(" ", 1)[1].strip()
                new_lines.append(f"mtllib {rename_map.get(old_name, old_name)}")
            elif line.startswith("o "):
                new_lines.append(f"o object_{obj_counter}")
                obj_counter += 1
            elif line.startswith("g "):
                new_lines.append(f"g group_{group_counter}")
                group_counter += 1
            elif line.startswith("usemtl "):
                raw_name = line.split(" ", 1)[1].strip()
                new_lines.append(f"usemtl {mat_map.get(raw_name, _sanitize_name(raw_name))}")
            else:
                new_lines.append(line)
        obj_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    tree = ET.parse(urdf_src)
    root = tree.getroot()
    root.set("name", _sanitize_name(root.get("name")))
    for link in root.findall("link"):
        link_name = _sanitize_name(link.get("name"))
        link.set("name", link_name)
        for idx, visual in enumerate(link.findall("visual")):
            visual.set("name", _stable_indexed_name(visual.get("name"), f"{link_name}_visual", idx))
        for idx, collision in enumerate(link.findall("collision")):
            collision.set("name", _stable_indexed_name(collision.get("name"), f"{link_name}_collision", idx))
    for joint in root.findall("joint"):
        joint.set("name", _sanitize_name(joint.get("name")))
    for elem in root.iter():
        if elem.tag == "mesh":
            filename = elem.get("filename") or elem.get("file")
            if not filename:
                continue
            mesh_path = Path(filename)
            new_name = rename_map.get(mesh_path.name, mesh_path.name)
            elem.set("filename", str(mesh_path.with_name(new_name)).replace("\\", "/"))
    ET.indent(tree, space="  ")
    tree.write(urdf_src, encoding="utf-8", xml_declaration=True)
    return {
        "urdf_path": str(urdf_src.resolve()),
        "textured_objs_dir": str(textured_dir.resolve()),
    }


def _copy_import_asset(asset_root: Path, bundle_dir: Path) -> Path:
    asset_root = asset_root.resolve()
    import_asset_dir = (bundle_dir / "import_asset").resolve()
    if import_asset_dir.exists():
        shutil.rmtree(import_asset_dir)
    import_asset_dir.mkdir(parents=True, exist_ok=True)

    urdf_src = next(asset_root.rglob("*.urdf"), None)
    if urdf_src is None:
        raise FileNotFoundError(f"No URDF found under {asset_root}")
    shutil.copy2(urdf_src, import_asset_dir / urdf_src.name)

    textured_src = asset_root / "textured_objs"
    if not textured_src.exists():
        raise FileNotFoundError(f"Missing textured_objs under {asset_root}")
    shutil.copytree(textured_src, import_asset_dir / "textured_objs")

    images_src = asset_root / "images"
    if images_src.exists():
        shutil.copytree(images_src, import_asset_dir / "images")

    return import_asset_dir


def _summarize_link_visuals(urdf_path: Path) -> dict[str, object]:
    root = ET.parse(urdf_path).getroot()
    visual_counts: dict[str, int] = {}
    renderable_links: list[str] = []
    empty_links: list[str] = []
    for link in root.findall("link"):
        link_name = str(link.get("name") or "")
        if not link_name:
            continue
        visual_count = len(link.findall("visual"))
        visual_counts[link_name] = int(visual_count)
        if visual_count > 0:
            renderable_links.append(link_name)
        else:
            empty_links.append(link_name)
    return {
        "link_visual_counts": visual_counts,
        "renderable_links": renderable_links,
        "empty_links": empty_links,
    }


def _control_mode(ctrl: dict) -> str:
    mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip()
    if mode in {"position", "joint_position"}:
        return "joint_position"
    if mode in {"velocity", "joint_velocity"}:
        return "joint_velocity"
    if mode in {"base", "base_velocity"}:
        return "base_velocity"
    if mode in {"base_decay", "base_velocity_decay"}:
        return "base_velocity_decay"
    if mode == "hold":
        return "hold_position"
    return mode


def _load_joint_limits(urdf_path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    _links, joints = rp.parse_urdf(urdf_path)
    limits = {str(j.get("name")): (j.get("limit") or {}) for j in joints if j.get("name")}
    joint_meta = {
        str(j.get("name")): {
            "type": j.get("type"),
            "axis": j.get("axis") or [0.0, 0.0, 1.0],
            "parent": j.get("parent"),
            "child": j.get("child"),
            "limit": j.get("limit") or {},
        }
        for j in joints
        if j.get("name")
    }
    return limits, joint_meta


def _find_visual_glb(asset_root: Path) -> Path | None:
    canonical = asset_root / f"animated_textured_{asset_root.name}.glb"
    if canonical.exists():
        return canonical
    matches = sorted(asset_root.glob("animated_textured*.glb"))
    return matches[0] if matches else None


def _find_visual_report_path(asset_root: Path) -> Path | None:
    candidates: list[Path] = []
    canonical = asset_root / f"animated_textured_{asset_root.name}.report.json"
    if canonical.exists():
        candidates.append(canonical)
    candidates.extend(sorted(asset_root.glob(f"animated_textured_*{asset_root.name}*.report.json")))
    candidates.extend(sorted(asset_root.glob("animated_textured*.report.json")))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("parts"), list):
            return path
    return None


def _find_visual_report(asset_root: Path) -> dict | None:
    report_path = _find_visual_report_path(asset_root)
    if report_path is None:
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and isinstance(data.get("parts"), list):
        return data
    return None


def _prepare_visual_asset(asset_root: Path, bundle_dir: Path) -> dict | None:
    visual_glb = _find_visual_glb(asset_root)
    visual_report_path = _find_visual_report_path(asset_root)
    if visual_glb is None or visual_report_path is None:
        return None
    bundle_glb = (bundle_dir / "visual_asset.glb").resolve()
    bundle_report = (bundle_dir / "visual_asset.report.json").resolve()
    if visual_glb.resolve() != bundle_glb:
        shutil.copy2(visual_glb, bundle_glb)
    if visual_report_path.resolve() != bundle_report:
        shutil.copy2(visual_report_path, bundle_report)
    return {
        "glb_path": str(bundle_glb),
        "usd_path": str((bundle_dir / "visual_asset.usd").resolve()),
        "report_json_path": str(bundle_report),
    }


def _prepare_baked_render_visual(asset_root: Path, plan_json: Path, bundle_dir: Path) -> dict:
    """Bake the same textured animated GLB used by the Blender renderer.

    Isaac's URDF importer is still used for articulation execution and
    trajectory export, but OBJ/MTL conversion can lose or alter visual
    materials. The baked GLB preserves the source textured scene and animation
    exactly as `tools/run_plan.py` exports it for Blender.
    """
    visual_glb = _find_visual_glb(asset_root)
    if visual_glb is None:
        return {
            "mode": "disabled",
            "source": "imported_urdf",
            "reason": f"no animated_textured*.glb found under {asset_root}",
        }

    render_dir = (bundle_dir / "render_visual").resolve()
    if render_dir.exists():
        shutil.rmtree(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str((TOOLS_DIR / "run_plan.py").resolve()),
        "--asset_root",
        str(asset_root.resolve()),
        "--plan_json",
        str(plan_json.resolve()),
        "--out",
        str(render_dir),
        "--export_animated_glb",
        "--use_glb_scene",
        "auto",
        "--skip_frame_render",
    ]
    result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path = render_dir / "bake_render_visual.log"
    log_path.write_text(result.stdout or "", encoding="utf-8")
    glb_path = render_dir / "plan_animated.glb"
    if result.returncode != 0 or not glb_path.exists():
        raise RuntimeError(
            "Failed to bake textured GLB render visual for Isaac export. "
            f"See {log_path}."
        )
    return {
        "mode": "baked_glb",
        "source": "run_plan_plan_animated_glb",
        "glb_path": str(glb_path.resolve()),
        "usd_path": str((render_dir / "plan_animated.usd").resolve()),
        "log_path": str(log_path.resolve()),
        "hide_imported_urdf_visuals": True,
    }


def _estimate_base_displacement(plan: dict) -> np.ndarray:
    pos = np.zeros(3, dtype=np.float64)
    for seg in plan.get("timeline") or []:
        t0 = float(seg.get("t0", 0.0))
        t1 = float(seg.get("t1", t0))
        dt = max(0.0, t1 - t0)
        for ctrl in seg.get("controls") or []:
            mode = _control_mode(ctrl)
            if mode == "base_velocity":
                axis = np.asarray(ctrl.get("axis_world") or [1.0, 0.0, 0.0], dtype=np.float64)
                n = float(np.linalg.norm(axis))
                if n > 1.0e-8:
                    axis = axis / n
                pos += axis * float(ctrl.get("v_mps", 0.0)) * dt
            elif mode == "base_velocity_decay":
                axis = np.asarray(ctrl.get("axis_world") or [1.0, 0.0, 0.0], dtype=np.float64)
                n = float(np.linalg.norm(axis))
                if n > 1.0e-8:
                    axis = axis / n
                v0 = float(ctrl.get("v0_mps", ctrl.get("v_mps", 0.0)))
                tau = max(1.0e-6, float(ctrl.get("tau_s", 1.0)))
                pos += axis * (v0 * tau * (1.0 - math.exp(-dt / tau)))
    return pos.astype(np.float32)


def _load_selected_viewspec(plan_json: Path) -> dict | None:
    asset_out = plan_json.resolve().parent
    candidates = [
        asset_out / "loop" / "motion_viewspecs_selected.json",
        asset_out / "loop" / "coverage" / "iter00" / "coverage_vlm_selected_viewspecs.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("views"), list) and data["views"]:
            return data["views"][0]
    return None


def _compute_rest_scene_stats(urdf_path: Path) -> dict[str, list[float] | float]:
    links, joints = rp.parse_urdf(urdf_path)
    link_meshes = rp.load_link_meshes(links, urdf_path.parent, textured=False)
    link_tf_map = rp.compute_link_transforms(links, joints, {})
    bounds: list[np.ndarray] = []
    for link_name, meshes in link_meshes.items():
        link_tf = np.asarray(link_tf_map.get(link_name, np.eye(4)), dtype=np.float64)
        for mesh in meshes:
            if getattr(mesh, "vertices", None) is None or mesh.vertices.size == 0:
                continue
            mesh_world = mesh.copy()
            mesh_world.apply_transform(link_tf)
            bounds.append(np.asarray(mesh_world.bounds, dtype=np.float64))
    if not bounds:
        zero = np.zeros(3, dtype=np.float32)
        return {
            "min_corner": [0.0, 0.0, 0.0],
            "max_corner": [0.0, 0.0, 0.0],
            "center": [0.0, 0.0, 0.0],
            "radius": 1.0,
        }
    stacked = np.concatenate(bounds, axis=0)
    min_corner = stacked.min(axis=0).astype(np.float32)
    max_corner = stacked.max(axis=0).astype(np.float32)
    center = ((min_corner + max_corner) * 0.5).astype(np.float32)
    radius = float(max(0.1, np.linalg.norm(max_corner - min_corner) * 0.6))
    return {
        "min_corner": [float(x) for x in min_corner.tolist()],
        "max_corner": [float(x) for x in max_corner.tolist()],
        "center": [float(x) for x in center.tolist()],
        "radius": radius,
    }


def _compute_initial_root_translation(scene_min_corner: list[float] | np.ndarray, desired_ground_z: float = 0.01) -> np.ndarray:
    min_corner = np.asarray(scene_min_corner, dtype=np.float32).reshape(3)
    lift_z = max(0.0, float(desired_ground_z) - float(min_corner[2]))
    return np.asarray([0.0, 0.0, lift_z], dtype=np.float32)


def _compute_camera(
    scene_center: list[float] | np.ndarray,
    scene_radius: float,
    base_displacement: np.ndarray,
    selected_view: dict | None = None,
) -> dict[str, list[float] | list[int]]:
    center = np.asarray(scene_center, dtype=np.float32).reshape(3)
    radius = float(max(0.25, scene_radius))
    disp = np.asarray(base_displacement, dtype=np.float32)
    path_len = float(np.linalg.norm(disp))
    # Keep the camera static and frame the whole motion corridor. Isaac's
    # camera-follow rig is not reliable enough here, especially for wheeled
    # assets like trolley2 where it can jump into an invalid close-up.
    follow_base = False
    target = center + 0.5 * disp
    if isinstance(selected_view, dict):
        azim_deg = float(selected_view.get("azimuth_deg", 45.0))
        elev_deg = float(selected_view.get("elevation_deg", 25.0))
        distance_scale = float(selected_view.get("distance_scale", 1.0))
        azim = math.radians(azim_deg)
        elev = math.radians(elev_deg)
        motion_pad = 0.0 if follow_base else path_len
        dist = max(radius * 2.5 * max(0.5, distance_scale), radius * 2.0 + motion_pad * 1.6)
        eye = np.asarray(
            [
                target[0] + dist * math.cos(elev) * math.cos(azim),
                target[1] + dist * math.cos(elev) * math.sin(azim),
                target[2] + dist * math.sin(elev),
            ],
            dtype=np.float32,
        )
        return {
            "eye": [float(x) for x in eye.tolist()],
            "target": [float(x) for x in target.tolist()],
            "resolution": [1280, 720],
            "source": "coverage_loop_selected_view",
            "follow_base": follow_base,
            "view": {
                "id": str(selected_view.get("id") or ""),
                "azimuth_deg": azim_deg,
                "elevation_deg": elev_deg,
                "distance_scale": distance_scale,
                "fov_deg": float(selected_view.get("fov_deg", 35.0)),
            },
        }
    move_xy = disp[:2]
    if float(np.linalg.norm(move_xy)) > 1.0e-6:
        move_xy = move_xy / float(np.linalg.norm(move_xy))
        side_xy = np.asarray([-move_xy[1], move_xy[0]], dtype=np.float32)
        view_dir = np.asarray(
            [1.4 * move_xy[0] + 1.8 * side_xy[0], 1.4 * move_xy[1] + 1.8 * side_xy[1], 1.25],
            dtype=np.float32,
        )
    else:
        view_dir = np.asarray([1.6, -2.0, 1.25], dtype=np.float32)
    view_dir = view_dir / max(1.0e-6, float(np.linalg.norm(view_dir)))
    motion_pad = 0.0 if follow_base else path_len
    distance = max(radius * 3.0, radius * 2.0 + motion_pad * 0.9)
    eye = target + view_dir * distance
    return {
        "eye": [float(x) for x in eye.tolist()],
        "target": [float(x) for x in target.tolist()],
        "resolution": [1280, 720],
        "source": "auto_3q_view",
        "follow_base": follow_base,
    }


def _normalize_plan(plan: dict, joint_limits: dict[str, dict]) -> list[dict]:
    normalized: list[dict] = []
    for seg in list(plan.get("timeline") or []):
        seg_out = {
            "name": str(seg.get("name") or "segment"),
            "phase_type": str(seg.get("phase_type") or ""),
            "t0": float(seg.get("t0", 0.0)),
            "t1": float(seg.get("t1", seg.get("t0", 0.0))),
            "controls": [],
        }
        for ctrl in list(seg.get("controls") or []):
            mode = _control_mode(ctrl)
            if mode not in SUPPORTED_CONTROL_MODES:
                raise ValueError(f"Unsupported control mode in plan: {mode}")
            if mode == "mode_set":
                seg_out["controls"].append(
                    {
                        "mode": "mode_set",
                        "name": str(ctrl.get("name") or ctrl.get("mode") or ""),
                        "set": bool(ctrl.get("set", True)),
                    }
                )
                continue
            if mode == "base_velocity":
                seg_out["controls"].append(
                    {
                        "mode": "base_velocity",
                        "axis_world": list(ctrl.get("axis_world") or [1.0, 0.0, 0.0]),
                        "v_mps": float(ctrl.get("v_mps", ctrl.get("linear_velocity_mps", 0.0))),
                    }
                )
                continue
            if mode == "base_velocity_decay":
                seg_out["controls"].append(
                    {
                        "mode": "base_velocity_decay",
                        "axis_world": list(ctrl.get("axis_world") or [1.0, 0.0, 0.0]),
                        "v0_mps": float(ctrl.get("v0_mps", ctrl.get("v_mps", 0.0))),
                        "tau_s": float(ctrl.get("tau_s", 1.0)),
                    }
                )
                continue
            if mode == "joint_velocity":
                joints = list(ctrl.get("joints") or [])
                if ctrl.get("joint"):
                    joints = [str(ctrl["joint"])]
                seg_out["controls"].append(
                    {
                        "mode": "joint_velocity",
                        "joints": [str(j) for j in joints],
                        "omega_radps": float(ctrl.get("omega_radps", 0.0)),
                        "ramp_to_omega_radps": (
                            None if ctrl.get("ramp_to_omega_radps") is None else float(ctrl.get("ramp_to_omega_radps"))
                        ),
                        "decay": ctrl.get("decay"),
                    }
                )
                continue
            if mode == "joint_position":
                joint_name = str(ctrl.get("joint") or "")
                q_target = ctrl.get("q_target_rad")
                if q_target is None:
                    q_target = rp.parse_target_value(
                        ctrl.get("q_target_expr") or ctrl.get("target_expr"),
                        joint_limits.get(joint_name, {}),
                    )
                seg_out["controls"].append(
                    {
                        "mode": "joint_position",
                        "joint": joint_name,
                        "q_start_rad": None if ctrl.get("q_start_rad") is None else float(ctrl.get("q_start_rad")),
                        "q_target_rad": float(q_target),
                        "curve": str(ctrl.get("curve") or "linear"),
                    }
                )
                continue
            if mode == "hold_position":
                joints = list(ctrl.get("joints") or [])
                if ctrl.get("joint"):
                    joints = [str(ctrl["joint"])]
                seg_out["controls"].append(
                    {
                        "mode": "hold_position",
                        "joints": [str(j) for j in joints],
                    }
                )
                continue
            if mode == "spring_return":
                seg_out["controls"].append(
                    {
                        "mode": "spring_return",
                        "joint": str(ctrl.get("joint") or ""),
                        "spring_k": float(ctrl.get("spring_k", 4.0)),
                        "damping_c": float(ctrl.get("damping_c", 0.6)),
                        "rest_position": float(ctrl.get("rest_position", ctrl.get("target_rad", 0.0))),
                    }
                )
                continue
        normalized.append(seg_out)
    return normalized


def _write_runner(bundle_dir: Path) -> None:
    executor_src = TOOLS_DIR / "isaacsim_timeline_executor.py"
    executor_dst = bundle_dir / "isaacsim_timeline_executor.py"
    shutil.copy2(executor_src, executor_dst)

    runner_path = bundle_dir / "run_isaacsim_executor.py"
    runner_text = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from pathlib import Path
        import sys

        SCRIPT_DIR = Path(__file__).resolve().parent
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))

        from isaacsim_timeline_executor import main

        if __name__ == "__main__":
            default_spec = Path(__file__).resolve().with_name("executor_spec.json")
            if "--spec" not in sys.argv:
                sys.argv.extend(["--spec", str(default_spec)])
            main()
        """
    )
    runner_path.write_text(runner_text, encoding="utf-8")
    runner_path.chmod(0o755)

    shell_path = bundle_dir / "run_isaacsim_executor.sh"
    shell_text = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        ISAAC_PYTHON="${ISAAC_PYTHON:-isaacsim}"
        exec "${ISAAC_PYTHON}" "${SCRIPT_DIR}/run_isaacsim_executor.py" "$@"
        """
    )
    shell_path.write_text(shell_text, encoding="utf-8")
    shell_path.chmod(0o755)


def compile_executor(
    asset_root: Path,
    causal_json: Path,
    plan_json: Path,
    bundle_dir: Path,
    *,
    render_visual_mode: str = "imported_urdf",
) -> dict:
    asset_root = asset_root.resolve()
    causal_json = causal_json.resolve()
    plan_json = plan_json.resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    import_asset_root = _copy_import_asset(asset_root, bundle_dir)
    _sanitize_asset_in_place(import_asset_root)

    causal = json.loads(causal_json.read_text(encoding="utf-8"))
    plan = json.loads(plan_json.read_text(encoding="utf-8"))
    urdf_path = next(import_asset_root.rglob("*.urdf"), None)
    if urdf_path is None:
        raise FileNotFoundError(f"No URDF found under {import_asset_root}")
    joint_limits, joint_meta = _load_joint_limits(urdf_path)
    normalized_timeline = _normalize_plan(plan, joint_limits)
    has_base_motion = any(
        _control_mode(ctrl) in {"base_velocity", "base_velocity_decay"}
        for seg in normalized_timeline
        for ctrl in seg.get("controls", [])
    )
    base_disp = _estimate_base_displacement(plan)
    selected_view = _load_selected_viewspec(plan_json)
    rest_scene = _compute_rest_scene_stats(urdf_path)
    initial_root_translation = _compute_initial_root_translation(rest_scene["min_corner"])
    placed_center = (np.asarray(rest_scene["center"], dtype=np.float32) + initial_root_translation).astype(np.float32)
    camera = _compute_camera(
        placed_center,
        float(rest_scene["radius"]),
        base_disp,
        selected_view=selected_view,
    )
    visual_summary = _summarize_link_visuals(urdf_path)
    render_visual_mode = str(render_visual_mode or "imported_urdf").strip().lower()
    if render_visual_mode not in {"imported_urdf", "baked_glb"}:
        raise ValueError(f"Unsupported render_visual_mode: {render_visual_mode}")
    if render_visual_mode == "baked_glb":
        render_visual = _prepare_baked_render_visual(asset_root, plan_json, bundle_dir)
    else:
        render_visual = {
            "mode": "disabled",
            "source": "imported_urdf",
            "reason": "Isaac renders the imported URDF articulation directly from joint controls.",
        }
    spec = {
        "asset_name": asset_root.name,
        "asset_root": str(asset_root),
        "source_causal_json": str(causal_json),
        "source_plan_json": str(plan_json),
        "meta": {
            "fps": int(plan.get("meta", {}).get("fps", 30)),
            "duration_s": float(plan.get("meta", {}).get("duration_s", 4.0)),
        },
        "camera": camera,
        "causal": causal,
        "timeline": normalized_timeline,
        "joint_limits": joint_limits,
        "joint_meta": joint_meta,
        "has_base_motion": bool(has_base_motion),
        "supported_control_modes": sorted(SUPPORTED_CONTROL_MODES),
        "placement": {
            "initial_root_translation": [float(x) for x in initial_root_translation.tolist()],
            "desired_ground_z": 0.01,
            "rest_scene_min_corner": list(rest_scene["min_corner"]),
            "rest_scene_max_corner": list(rest_scene["max_corner"]),
            "rest_scene_center": list(rest_scene["center"]),
            "rest_scene_radius": float(rest_scene["radius"]),
        },
        "import": {
            "fix_base": not has_base_motion,
            "bundle_asset_root": str(import_asset_root),
            "urdf_path": str(urdf_path.resolve()),
            "textured_objs_dir": str((import_asset_root / "textured_objs").resolve()),
            "link_visual_counts": visual_summary["link_visual_counts"],
            "renderable_links": visual_summary["renderable_links"],
            "empty_links": visual_summary["empty_links"],
        },
        "outputs": {
            "out_dir": str((bundle_dir / "outputs").resolve()),
            "video_name": f"{asset_root.name}_isaac.mp4",
            "trajectory_npz_name": "trajectory.npz",
            "trajectory_jsonl_name": "trajectory.jsonl",
        },
        "custom_visuals": {
            "mode": "disabled",
            "source": "imported_urdf",
        },
        "render_visual": render_visual,
    }
    spec_path = bundle_dir / "executor_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_runner(bundle_dir)
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(
        textwrap.dedent(
            f"""\
            # Isaac Sim Executor Bundle

            Asset: `{asset_root.name}`

            Run:

            ```bash
            {bundle_dir / "run_isaacsim_executor.sh"}
            ```

            Generated code:
            - `run_isaacsim_executor.py`: bundle entry point.
            - `isaacsim_timeline_executor.py`: Isaac Sim runtime that applies the compiled joint timeline and records video/trajectory.

            This bundle was compiled from:
            - `{causal_json}`
            - `{plan_json}`
            """
        ),
        encoding="utf-8",
    )
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile causal.json + plan.json into an Isaac Sim executor bundle.")
    ap.add_argument("--asset_root", required=True)
    ap.add_argument("--causal_json", required=True)
    ap.add_argument("--plan_json", required=True)
    ap.add_argument("--bundle_dir", required=True)
    ap.add_argument(
        "--render_visual_mode",
        choices=["imported_urdf", "baked_glb"],
        default="imported_urdf",
        help=(
            "How Isaac should render the scene. imported_urdf keeps the visible animation tied to "
            "Isaac joint articulation; baked_glb overlays a pre-baked GLB render visual."
        ),
    )
    args = ap.parse_args()

    spec = compile_executor(
        asset_root=Path(args.asset_root),
        causal_json=Path(args.causal_json),
        plan_json=Path(args.plan_json),
        bundle_dir=Path(args.bundle_dir),
        render_visual_mode=str(args.render_visual_mode),
    )
    print(json.dumps({"bundle_dir": args.bundle_dir, "asset_name": spec["asset_name"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
