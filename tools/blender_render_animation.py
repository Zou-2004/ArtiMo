#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
from pathlib import Path


def _material_has_image_texture(mat) -> bool:
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return False
    for node in nt.nodes:
        if getattr(node, "type", "") != "TEX_IMAGE":
            continue
        if getattr(node, "image", None) is None:
            continue
        return True
    return False


def _material_principled_alpha(mat) -> float:
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return 1.0
    alpha = 1.0
    for node in nt.nodes:
        if getattr(node, "type", "") != "BSDF_PRINCIPLED":
            continue
        try:
            alpha = min(alpha, float(node.inputs["Alpha"].default_value))
        except Exception:
            continue
    return float(alpha)


def _object_world_bounds(obj):
    try:
        import mathutils  # type: ignore

        corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    except Exception:
        return None
    if not corners:
        return None
    xs = [float(v.x) for v in corners]
    ys = [float(v.y) for v in corners]
    zs = [float(v.z) for v in corners]
    return (
        (min(xs), min(ys), min(zs)),
        (max(xs), max(ys), max(zs)),
    )


def _scene_center(bpy):
    bounds = []
    for obj in bpy.data.objects:
        if getattr(obj, "type", "") != "MESH" or getattr(obj, "hide_render", False):
            continue
        b = _object_world_bounds(obj)
        if b is not None:
            bounds.append(b)
    if not bounds:
        return (0.0, 0.0, 0.0)
    mins = [min(b[0][i] for b in bounds) for i in range(3)]
    maxs = [max(b[1][i] for b in bounds) for i in range(3)]
    return tuple(0.5 * (mins[i] + maxs[i]) for i in range(3))


def _bounds_key(bounds, digits=5):
    (x0, y0, z0), (x1, y1, z1) = bounds
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    ex = x1 - x0
    ey = y1 - y0
    ez = z1 - z0
    return tuple(round(v, digits) for v in (cx, cy, cz, ex, ey, ez))


def _suppress_untextured_duplicate_meshes(bpy):
    groups = {}
    for obj in bpy.data.objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        bounds = _object_world_bounds(obj)
        if bounds is None:
            continue
        textured = False
        for slot in getattr(obj, "material_slots", []):
            mat = getattr(slot, "material", None)
            if mat is not None and _material_has_image_texture(mat):
                textured = True
                break
        groups.setdefault(_bounds_key(bounds), []).append((obj, textured))
    for items in groups.values():
        if len(items) < 2:
            continue
        has_textured = any(flag for _, flag in items)
        has_plain = any(not flag for _, flag in items)
        if not (has_textured and has_plain):
            continue
        for obj, textured in items:
            if textured:
                try:
                    bpy.ops.object.mode_set(mode="OBJECT")
                except Exception:
                    pass
                try:
                    bpy.ops.object.select_all(action="DESELECT")
                except Exception:
                    pass
                try:
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)
                    bpy.ops.object.mode_set(mode="EDIT")
                    bpy.ops.mesh.select_all(action="SELECT")
                    bpy.ops.mesh.flip_normals()
                finally:
                    try:
                        bpy.ops.object.mode_set(mode="OBJECT")
                    except Exception:
                        pass
                    try:
                        obj.select_set(False)
                    except Exception:
                        pass
                continue
            try:
                obj.hide_render = True
                obj.hide_set(True)
            except Exception:
                continue


def _attenuate_large_transparent_covers(bpy):
    mesh_objects = [obj for obj in bpy.data.objects if getattr(obj, "type", "") == "MESH" and not getattr(obj, "hide_render", False)]
    if not mesh_objects:
        return
    all_bounds = [_object_world_bounds(obj) for obj in mesh_objects]
    all_bounds = [b for b in all_bounds if b is not None]
    if not all_bounds:
        return
    mins = [min(b[0][i] for b in all_bounds) for i in range(3)]
    maxs = [max(b[1][i] for b in all_bounds) for i in range(3)]
    scene_extents = [maxs[i] - mins[i] for i in range(3)]
    scene_max_extent = max(1e-6, max(scene_extents))
    for obj in mesh_objects:
        bounds = _object_world_bounds(obj)
        if bounds is None:
            continue
        extents = [bounds[1][i] - bounds[0][i] for i in range(3)]
        thin_ratio = min(extents) / max(1e-6, max(extents))
        cover_ratio = max(extents) / scene_max_extent
        if thin_ratio > 0.12 or cover_ratio < 0.45:
            continue
        for slot in getattr(obj, "material_slots", []):
            mat = getattr(slot, "material", None)
            if mat is None:
                continue
            alpha = _material_principled_alpha(mat)
            blend = str(getattr(mat, "blend_method", "")).upper()
            if alpha >= 0.75 and blend not in {"BLEND", "HASHED"}:
                continue
            try:
                obj.hide_render = True
                obj.hide_set(True)
            except Exception:
                pass
            break


def _image_mean_luma(image) -> float | None:
    try:
        width = int(getattr(image, "size", [0, 0])[0])
        height = int(getattr(image, "size", [0, 0])[1])
        if width <= 0 or height <= 0:
            return None
        pixels = image.pixels[:]
        if not pixels:
            return None
        sample_px = min(width * height, 4096)
        stride_px = max(1, (width * height) // sample_px)
        total = 0.0
        count = 0
        for idx in range(0, width * height, stride_px):
            base = idx * 4
            r = float(pixels[base + 0])
            g = float(pixels[base + 1])
            b = float(pixels[base + 2])
            total += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
        if count <= 0:
            return None
        return total / float(count)
    except Exception:
        return None


def _darken_bright_textured_materials(bpy):
    brighten_threshold = float(os.environ.get("CODEX_BLENDER_BRIGHT_TEX_LUMA_MAX", "0.74"))
    darken_factor = float(os.environ.get("CODEX_BLENDER_BRIGHT_TEX_MULTIPLY", "0.78"))
    darken_factor = max(0.40, min(1.0, darken_factor))
    for mat in bpy.data.materials:
        nt = getattr(mat, "node_tree", None)
        if nt is None:
            continue
        principled = next((n for n in nt.nodes if getattr(n, "type", "") == "BSDF_PRINCIPLED"), None)
        if principled is None:
            continue
        base_links = [lk for lk in nt.links if lk.to_node == principled and getattr(lk.to_socket, "name", "") == "Base Color"]
        if not base_links:
            continue
        link = base_links[0]
        tex_node = link.from_node
        if getattr(tex_node, "type", "") != "TEX_IMAGE":
            continue
        image = getattr(tex_node, "image", None)
        if image is None:
            continue
        mean_luma = _image_mean_luma(image)
        if mean_luma is None or mean_luma < brighten_threshold:
            continue
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.label = "CodexDarkenBrightTexture"
        mix.inputs[0].default_value = 1.0
        mix.inputs[2].default_value = (darken_factor, darken_factor, darken_factor, 1.0)
        mix.location = (tex_node.location.x + 220.0, tex_node.location.y)
        nt.links.remove(link)
        nt.links.new(tex_node.outputs["Color"], mix.inputs[1])
        nt.links.new(mix.outputs["Color"], principled.inputs["Base Color"])


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    ap = argparse.ArgumentParser(description="Render an animated GLB sequence in Blender headless")
    ap.add_argument("--glb", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--camera_json", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frame_start", type=int, required=True)
    ap.add_argument("--frame_end", type=int, required=True)
    ap.add_argument("--frame_offset", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--fov_deg", type=float, default=45.0)
    ap.add_argument("--device_type", default="OPTIX")
    ap.add_argument("--gpu_index", type=int, default=0)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--denoise", action="store_true")
    return ap.parse_args(argv)


def _clear_scene(bpy):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block_coll in [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.actions,
    ]:
        for block in list(block_coll):
            if block.users == 0:
                block_coll.remove(block)


def _set_camera_lookat(mathutils, cam_obj, eye, target, up):
    e = mathutils.Vector(eye)
    t = mathutils.Vector(target)
    u = mathutils.Vector(up)
    f = (t - e).normalized()
    r = f.cross(u).normalized()
    tu = r.cross(f).normalized()
    mat = mathutils.Matrix(
        (
            (r.x, tu.x, -f.x, e.x),
            (r.y, tu.y, -f.y, e.y),
            (r.z, tu.z, -f.z, e.z),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    cam_obj.matrix_world = mat


def _point_object_toward(mathutils, obj, location, target):
    obj.location = tuple(float(v) for v in location)
    src = mathutils.Vector(location)
    dst = mathutils.Vector(target)
    direction = (dst - src)
    if direction.length <= 1.0e-8:
        return
    obj.rotation_euler = direction.normalized().to_track_quat("-Z", "Y").to_euler()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _bind_gpu_device_for_compute(prefs, compute_type: str, gpu_index: int) -> tuple[bool, str]:
    devices = list(getattr(prefs, "devices", []))
    gpu_devs = []
    for dev in devices:
        dev_type = str(getattr(dev, "type", "")).upper()
        if dev_type in {"CPU", ""}:
            continue
        if dev_type == str(compute_type).upper():
            gpu_devs.append(dev)
    if not gpu_devs:
        for dev in devices:
            try:
                dev.use = False
            except Exception:
                pass
        return False, "none"
    use_multi_gpu = _bool_env("CODEX_BLENDER_MULTI_GPU", default=False)
    if use_multi_gpu:
        selected = list(gpu_devs)
    else:
        sel_idx = int(gpu_index)
        if sel_idx < 0:
            sel_idx = 0
        if sel_idx >= len(gpu_devs):
            sel_idx = 0
        selected = [gpu_devs[sel_idx]]
    for dev in devices:
        try:
            dev.use = dev in selected
        except Exception:
            pass
    labels = []
    for dev in selected:
        sel_name = str(getattr(dev, "name", "")).strip() or "unnamed"
        sel_type = str(getattr(dev, "type", "")).upper()
        labels.append(f"{sel_type}:{sel_name}")
    if use_multi_gpu:
        return True, f"MULTI[{len(labels)}]:" + "|".join(labels)
    return True, labels[0]


def _configure_cycles(bpy, scene, device_type, gpu_index, samples, tile_size, denoise):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(samples)
    scene.cycles.use_denoising = bool(denoise)
    scene.cycles.max_bounces = 2
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 2
    if hasattr(scene.cycles, "tile_size"):
        scene.cycles.tile_size = int(tile_size)
    if denoise and hasattr(scene.cycles, "denoiser"):
        try:
            scene.cycles.denoiser = "OPTIX" if str(device_type).upper() == "OPTIX" else "OPENIMAGEDENOISE"
        except Exception:
            pass
    force_cpu = os.environ.get("CODEX_BLENDER_FORCE_CPU") == "1"
    try:
        if not force_cpu:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            compute_order = [str(device_type).upper()]
            if "OPTIX" not in compute_order:
                compute_order.append("OPTIX")
            if "CUDA" not in compute_order:
                compute_order.append("CUDA")
            for compute_type in compute_order:
                try:
                    prefs.compute_device_type = compute_type
                    prefs.get_devices()
                except Exception:
                    continue
                gpu_found, gpu_desc = _bind_gpu_device_for_compute(prefs, str(compute_type).upper(), int(gpu_index))
                if gpu_found:
                    scene.cycles.device = "GPU"
                    return f"GPU:{compute_type}:{gpu_desc}"
    except Exception:
        pass
    scene.cycles.device = "CPU"
    return "CPU"


def _setup_scene(bpy, width, height, fov_deg, fps, device_type, gpu_index, samples, tile_size, denoise):
    import mathutils  # type: ignore

    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.fps = int(fps)
    bg_strength = float(os.environ.get("CODEX_BLENDER_BG_STRENGTH", "0.65"))
    bg_gray = float(os.environ.get("CODEX_BLENDER_BG_GRAY", "0.92"))
    exposure = float(os.environ.get("CODEX_BLENDER_EXPOSURE", "-0.10"))
    sun_key_energy = float(os.environ.get("CODEX_BLENDER_SUN_KEY_ENERGY", "1.6"))
    sun_fill_energy = float(os.environ.get("CODEX_BLENDER_SUN_FILL_ENERGY", "0.6"))

    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.exposure = exposure
    except Exception:
        pass
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg is not None:
        bg_gray = max(0.0, min(1.0, bg_gray))
        bg.inputs[0].default_value = (bg_gray, bg_gray, bg_gray, 1.0)
        bg.inputs[1].default_value = max(0.0, bg_strength)
    selected_device = _configure_cycles(bpy, scene, device_type, gpu_index, samples, tile_size, denoise)

    cam_data = bpy.data.cameras.new("AnimCam")
    # Keep camera intrinsics consistent with project_points() in overlay pipeline:
    # use vertical FOV convention explicitly.
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = math.radians(float(fov_deg))
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    cam_obj = bpy.data.objects.new("AnimCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    target = _scene_center(bpy)

    sun1 = bpy.data.lights.new("SunKey", type="SUN")
    sun1.energy = max(0.0, sun_key_energy)
    sun1_obj = bpy.data.objects.new("SunKey", sun1)
    scene.collection.objects.link(sun1_obj)
    _point_object_toward(mathutils, sun1_obj, (2.0, -2.0, 3.0), target)

    sun2 = bpy.data.lights.new("SunFill", type="SUN")
    sun2.energy = max(0.0, sun_fill_energy)
    sun2_obj = bpy.data.objects.new("SunFill", sun2)
    scene.collection.objects.link(sun2_obj)
    _point_object_toward(mathutils, sun2_obj, (-2.0, 2.0, 2.5), target)

    return scene, cam_obj, selected_device


def main():
    args = _parse_args()

    import bpy  # type: ignore
    import mathutils  # type: ignore

    _clear_scene(bpy)
    scene = bpy.context.scene
    scene.render.fps = int(args.fps)
    bpy.ops.import_scene.gltf(filepath=str(Path(args.glb).absolute()))
    _suppress_untextured_duplicate_meshes(bpy)
    _attenuate_large_transparent_covers(bpy)
    _darken_bright_textured_materials(bpy)

    scene, cam_obj, selected_device = _setup_scene(
        bpy,
        args.width,
        args.height,
        args.fov_deg,
        args.fps,
        args.device_type,
        args.gpu_index,
        args.samples,
        args.tile_size,
        args.denoise,
    )
    print(f"CODEX_BLENDER_DEVICE_USED={selected_device}", flush=True)
    camera = json.loads(Path(args.camera_json).read_text(encoding="utf-8"))
    _set_camera_lookat(mathutils, cam_obj, camera["eye"], camera["target"], camera["up"])

    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene.frame_start = int(args.frame_start)
    scene.frame_end = int(args.frame_end)
    for frame_idx in range(int(args.frame_start), int(args.frame_end) + 1):
        scene.frame_set(frame_idx)
        out_idx = frame_idx - int(args.frame_start) + int(args.frame_offset)
        scene.render.filepath = str(out_dir / f"frame_{out_idx:04d}.png")
        bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
