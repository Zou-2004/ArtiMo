#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

_IMAGE_LUMA_CACHE: dict[tuple, float | None] = {}


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


def _iter_material_texture_nodes(mat):
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return []
    out = []
    for node in nt.nodes:
        if getattr(node, "type", "") != "TEX_IMAGE":
            continue
        if getattr(node, "image", None) is None:
            continue
        out.append(node)
    return out


def _image_mean_luma(image, max_samples: int = 2048) -> float | None:
    if image is None:
        return None
    width = int(getattr(image, "size", [0, 0])[0] or 0)
    height = int(getattr(image, "size", [0, 0])[1] or 0)
    if width <= 0 or height <= 0:
        return None

    filepath = str(getattr(image, "filepath_raw", None) or getattr(image, "filepath", "") or "").strip()
    if filepath:
        p = Path(filepath)
        if p.exists():
            try:
                st = p.stat()
                key = ("path", str(p.resolve()), int(st.st_mtime_ns), int(st.st_size))
                if key in _IMAGE_LUMA_CACHE:
                    return _IMAGE_LUMA_CACHE[key]
                try:
                    from PIL import Image as PILImage  # type: ignore

                    with PILImage.open(p) as img:
                        img = img.convert("RGB")
                        img.thumbnail((64, 64))
                        px = list(img.getdata())
                    if not px:
                        _IMAGE_LUMA_CACHE[key] = None
                        return None
                    accum = 0.0
                    for r, g, b in px:
                        accum += 0.2126 * (float(r) / 255.0) + 0.7152 * (float(g) / 255.0) + 0.0722 * (float(b) / 255.0)
                    out = float(accum / len(px))
                    _IMAGE_LUMA_CACHE[key] = out
                    return out
                except Exception:
                    pass
            except Exception:
                pass

    try:
        key = ("fallback", str(getattr(image, "name", "") or ""), width, height)
        if key in _IMAGE_LUMA_CACHE:
            return _IMAGE_LUMA_CACHE[key]
        pixels = image.pixels[:]
        total = int(len(pixels) // 4)
        if total <= 0:
            _IMAGE_LUMA_CACHE[key] = None
            return None
        step = max(1, total // max(1, int(max_samples)))
        accum = 0.0
        count = 0
        for idx in range(0, total, step):
            base = idx * 4
            r = float(pixels[base + 0])
            g = float(pixels[base + 1])
            b = float(pixels[base + 2])
            accum += 0.2126 * r + 0.7152 * g + 0.0722 * b
            count += 1
        if count <= 0:
            _IMAGE_LUMA_CACHE[key] = None
            return None
        out = float(accum / count)
        _IMAGE_LUMA_CACHE[key] = out
        return out
    except Exception:
        return None


def _find_bsdf_output_nodes(mat):
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return None, None
    bsdf = None
    output = None
    for node in nt.nodes:
        ntype = getattr(node, "type", "")
        if ntype == "BSDF_PRINCIPLED" and bsdf is None:
            bsdf = node
        elif ntype == "OUTPUT_MATERIAL" and output is None:
            output = node
    return bsdf, output


def _material_has_render_boost(mat) -> bool:
    nt = getattr(mat, "node_tree", None)
    if nt is None:
        return False
    for node in nt.nodes:
        if str(getattr(node, "label", "")).strip() == "CodexRenderBoost":
            return True
    return False


def _safe_socket(outputs, name: str):
    try:
        return outputs.get(name)
    except Exception:
        return None


def _link_shader_with_emission(nt, bsdf, emission, output):
    try:
        add = nt.nodes.new("ShaderNodeAddShader")
        add.label = "CodexRenderBoost"
    except Exception:
        return
    try:
        # Replace any existing surface links with BSDF + emission additive shading.
        for link in list(nt.links):
            if getattr(link, "to_node", None) == output and getattr(link, "to_socket", None) == output.inputs.get("Surface"):
                nt.links.remove(link)
    except Exception:
        pass
    try:
        nt.links.new(bsdf.outputs.get("BSDF"), add.inputs[0])
        nt.links.new(emission.outputs.get("Emission"), add.inputs[1])
        nt.links.new(add.outputs.get("Shader"), output.inputs.get("Surface"))
    except Exception:
        return


def _apply_render_only_material_boosts(bpy):
    dark_tex_strength = float(os.environ.get("CODEX_BLENDER_RENDER_DARK_TEX_EMISSIVE", "0.16"))
    bright_tex_strength = float(os.environ.get("CODEX_BLENDER_RENDER_BRIGHT_TEX_EMISSIVE", "0.28"))
    bright_solid_scale = float(os.environ.get("CODEX_BLENDER_RENDER_BRIGHT_SOLID_EMISSIVE", "0.32"))
    dark_luma_max = float(os.environ.get("CODEX_BLENDER_RENDER_DARK_TEX_LUMA_MAX", "0.35"))
    bright_luma_min = float(os.environ.get("CODEX_BLENDER_RENDER_BRIGHT_TEX_LUMA_MIN", "0.70"))
    bright_solid_luma_min = float(os.environ.get("CODEX_BLENDER_RENDER_BRIGHT_SOLID_LUMA_MIN", "0.55"))

    for mat in bpy.data.materials:
        if mat is None or not getattr(mat, "use_nodes", False) or _material_has_render_boost(mat):
            continue
        nt = getattr(mat, "node_tree", None)
        if nt is None:
            continue
        bsdf, output = _find_bsdf_output_nodes(mat)
        if bsdf is None or output is None:
            continue
        tex_nodes = _iter_material_texture_nodes(mat)
        if tex_nodes:
            tex_node = tex_nodes[0]
            luma = _image_mean_luma(getattr(tex_node, "image", None))
            strength = 0.0
            if luma is not None and luma <= dark_luma_max:
                strength = max(strength, dark_tex_strength)
            if luma is not None and luma >= bright_luma_min:
                strength = max(strength, bright_tex_strength)
            if strength <= 0.0:
                continue
            try:
                emission = nt.nodes.new("ShaderNodeEmission")
                emission.label = "CodexRenderBoost"
                if _safe_socket(tex_node.outputs, "Color") is not None:
                    nt.links.new(tex_node.outputs.get("Color"), emission.inputs.get("Color"))
                emission.inputs.get("Strength").default_value = float(strength)
            except Exception:
                continue
            _link_shader_with_emission(nt, bsdf, emission, output)
            continue

        try:
            base = [float(x) for x in bsdf.inputs.get("Base Color").default_value[:3]]
        except Exception:
            continue
        luma = 0.2126 * base[0] + 0.7152 * base[1] + 0.0722 * base[2]
        if luma < bright_solid_luma_min or bright_solid_scale <= 0.0:
            continue
        try:
            emission = nt.nodes.new("ShaderNodeEmission")
            emission.label = "CodexRenderBoost"
            emission.inputs.get("Color").default_value = (base[0], base[1], base[2], 1.0)
            emission.inputs.get("Strength").default_value = float(bright_solid_scale)
        except Exception:
            continue
        _link_shader_with_emission(nt, bsdf, emission, output)


def _set_principled_input(bsdf, names, value):
    for name in names:
        sock = bsdf.inputs.get(name)
        if sock is None:
            continue
        try:
            sock.default_value = value
            return True
        except Exception:
            continue
    return False


def _apply_reflective_mesh_finish(bpy):
    if str(os.environ.get("CODEX_BLENDER_REFLECTIVE_MESH", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    roughness = float(os.environ.get("CODEX_BLENDER_MESH_ROUGHNESS", "0.34"))
    specular = float(os.environ.get("CODEX_BLENDER_MESH_SPECULAR", "0.55"))
    coat = float(os.environ.get("CODEX_BLENDER_MESH_COAT", "0.18"))
    metallic = float(os.environ.get("CODEX_BLENDER_MESH_METALLIC", "0.0"))
    for mat in bpy.data.materials:
        if mat is None or not getattr(mat, "use_nodes", False):
            continue
        bsdf, _output = _find_bsdf_output_nodes(mat)
        if bsdf is None:
            continue
        _set_principled_input(bsdf, ["Roughness"], max(0.0, min(1.0, roughness)))
        _set_principled_input(bsdf, ["Specular IOR Level", "Specular"], max(0.0, min(1.0, specular)))
        _set_principled_input(bsdf, ["Coat Weight", "Clearcoat"], max(0.0, min(1.0, coat)))
        _set_principled_input(bsdf, ["Metallic"], max(0.0, min(1.0, metallic)))
        _set_principled_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.22)


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


def _visible_mesh_scene_bounds(bpy):
    mins = []
    maxs = []
    for obj in bpy.data.objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        if str(getattr(obj, "name", "")).startswith("CodexGroundPlane"):
            continue
        try:
            if bool(getattr(obj, "hide_render", False)):
                continue
        except Exception:
            pass
        bounds = _object_world_bounds(obj)
        if bounds is None:
            continue
        mins.append(bounds[0])
        maxs.append(bounds[1])
    if not mins:
        return None
    import numpy as np

    mn = np.min(np.asarray(mins, dtype=float), axis=0)
    mx = np.max(np.asarray(maxs, dtype=float), axis=0)
    return mn, mx


def _make_ground_material(bpy):
    mat = bpy.data.materials.new("CodexGlossyWhiteGround")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is not None:
        ground_color = _env_float_tuple("CODEX_BLENDER_GROUND_COLOR", (1.0, 1.0, 1.0))
        _set_principled_input(bsdf, ["Base Color"], (float(ground_color[0]), float(ground_color[1]), float(ground_color[2]), 1.0))
        _set_principled_input(bsdf, ["Roughness"], float(os.environ.get("CODEX_BLENDER_GROUND_ROUGHNESS", "0.22")))
        _set_principled_input(bsdf, ["Specular IOR Level", "Specular"], float(os.environ.get("CODEX_BLENDER_GROUND_SPECULAR", "0.75")))
        _set_principled_input(bsdf, ["Metallic"], float(os.environ.get("CODEX_BLENDER_GROUND_METALLIC", "0.0")))
        emission_strength = float(os.environ.get("CODEX_BLENDER_GROUND_EMISSION_STRENGTH", "0.0"))
        if emission_strength > 0.0:
            out = nt.nodes.get("Material Output")
            if out is not None:
                emission = nt.nodes.new(type="ShaderNodeEmission")
                emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
                emission.inputs["Strength"].default_value = emission_strength
                add = nt.nodes.new(type="ShaderNodeAddShader")
                for link in list(nt.links):
                    if link.to_node is out and link.to_socket == out.inputs["Surface"]:
                        nt.links.remove(link)
                nt.links.new(bsdf.outputs["BSDF"], add.inputs[0])
                nt.links.new(emission.outputs["Emission"], add.inputs[1])
                nt.links.new(add.outputs["Shader"], out.inputs["Surface"])
    return mat


def _world_up_ground_normal():
    import numpy as np

    raw = os.environ.get("CODEX_WORLD_TO_BLENDER_MATRIX")
    if raw:
        rows = []
        for row in raw.split(";"):
            vals = [float(x.strip()) for x in row.split(",") if x.strip()]
            if len(vals) == 3:
                rows.append(vals)
        if len(rows) == 3:
            basis = np.asarray(rows, dtype=float)
        else:
            basis = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)
    else:
        # Match the camera/object conversion used by blender_render.py:
        # (x_w, y_w, z_w) -> (x_b, -z_w, y_w).
        basis = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)
    normal = basis @ np.asarray([0.0, 0.0, 1.0], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=float)
    return normal / norm


def _add_shadow_reflection_ground(bpy):
    if str(os.environ.get("CODEX_BLENDER_SHADOW_GROUND", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    bounds = _visible_mesh_scene_bounds(bpy)
    if bounds is None:
        return
    mn, mx = bounds
    import itertools
    import numpy as np
    center = 0.5 * (mn + mx)
    extent = mx - mn
    size = max(1.0, float(max(extent[0], extent[1], extent[2])) * float(os.environ.get("CODEX_BLENDER_GROUND_SIZE_SCALE", "3.6")))
    normal = _world_up_ground_normal()
    corners = np.asarray(
        list(itertools.product([float(mn[0]), float(mx[0])], [float(mn[1]), float(mx[1])], [float(mn[2]), float(mx[2])])),
        dtype=float,
    )
    min_dot = float(np.min(corners @ normal))
    offset = max(0.002, 0.01 * float(max(extent[0], extent[1], extent[2], 1.0e-6)))
    plane_dot = min_dot - offset
    location = center + normal * (plane_dot - float(center @ normal))
    try:
        import mathutils  # type: ignore

        rot = mathutils.Vector((0.0, 0.0, 1.0)).rotation_difference(
            mathutils.Vector((float(normal[0]), float(normal[1]), float(normal[2])))
        )
        bpy.ops.mesh.primitive_plane_add(
            size=size,
            location=(float(location[0]), float(location[1]), float(location[2])),
            rotation=rot.to_euler(),
        )
        plane = bpy.context.object
        plane.name = "CodexGroundPlane"
        plane.data.materials.append(_make_ground_material(bpy))
        plane.visible_shadow = True
        if str(os.environ.get("CODEX_BLENDER_SHADOW_CATCHER_GROUND", "0")).strip().lower() in {"1", "true", "yes", "on"}:
            try:
                plane.is_shadow_catcher = True
            except Exception:
                pass
        plane.hide_select = True
    except Exception:
        return


def _env_float_tuple(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        vals = tuple(float(x.strip()) for x in raw.split(",") if x.strip())
        if len(vals) == len(default):
            return vals
    except Exception:
        pass
    return default


def _env_color(name: str, default: tuple[float, float, float]) -> tuple[float, float, float, float]:
    vals = _env_float_tuple(name, default)
    if len(vals) < 3:
        vals = default
    return (
        max(0.0, min(1.0, float(vals[0]))),
        max(0.0, min(1.0, float(vals[1]))),
        max(0.0, min(1.0, float(vals[2]))),
        1.0,
    )


def _composite_png_over_white(path: str):
    if str(os.environ.get("CODEX_BLENDER_COMPOSITE_WHITE_BG", "0")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        from PIL import Image as PILImage  # type: ignore

        p = Path(path)
        with PILImage.open(p) as img:
            rgba = img.convert("RGBA")
            white = PILImage.new("RGBA", rgba.size, (255, 255, 255, 255))
            white.alpha_composite(rgba)
            white.convert("RGB").save(p)
    except Exception as exc:
        print(f"[WARN] failed to composite white background for {path}: {exc}", flush=True)


def _bounds_key(bounds, digits=5):
    (x0, y0, z0), (x1, y1, z1) = bounds
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    ex = x1 - x0
    ey = y1 - y0
    ez = z1 - z0
    return tuple(round(v, digits) for v in (cx, cy, cz, ex, ey, ez))


def _object_is_textured_surface(obj) -> bool:
    try:
        if len(getattr(getattr(obj, "data", None), "uv_layers", [])) <= 0:
            return False
    except Exception:
        return False
    for slot in getattr(obj, "material_slots", []):
        mat = getattr(slot, "material", None)
        if mat is not None and _material_has_image_texture(mat):
            return True
    return False


def _flip_mesh_normals(bpy, obj):
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


def _normalize_textured_duplicate_meshes(bpy):
    groups = {}
    for obj in bpy.data.objects:
        if getattr(obj, "type", "") != "MESH":
            continue
        bounds = _object_world_bounds(obj)
        if bounds is None:
            continue
        textured = _object_is_textured_surface(obj)
        groups.setdefault(_bounds_key(bounds), []).append((obj, textured))
    for items in groups.values():
        if len(items) < 2:
            continue
        textured_items = [obj for obj, is_textured in items if is_textured]
        plain_items = [obj for obj, is_textured in items if not is_textured]
        if not textured_items or not plain_items:
            continue
        for obj in textured_items:
            _flip_mesh_normals(bpy, obj)
        for obj in plain_items:
            try:
                obj.hide_render = True
                obj.hide_set(True)
            except Exception:
                continue


def _parse_args():
    # Blender passes script args after `--`
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    ap = argparse.ArgumentParser(description="Render multi-view images from a GLB/FBX in Blender headless")
    ap.add_argument("--glb", required=True)
    ap.add_argument("--views_json", required=True)
    ap.add_argument("--transforms_json", required=False, default=None)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--fov_deg", type=float, default=50.0)
    ap.add_argument("--device_type", default="OPTIX")
    ap.add_argument("--gpu_index", type=int, default=0)
    ap.add_argument("--samples", type=int, default=128)
    ap.add_argument("--tile_size", type=int, default=256)
    ap.add_argument("--denoise", action="store_true")
    ap.add_argument("--keep_animation", action="store_true")
    ap.add_argument("--frame_idx", type=int, default=None)
    ap.add_argument("--fps", type=int, default=None)
    return ap.parse_args(argv)


def _clear_scene(bpy):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    # remove orphan data for repeatability
    for block_coll in [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
    ]:
        for b in list(block_coll):
            if b.users == 0:
                block_coll.remove(b)


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


def _clear_imported_object_animation(bpy):
    # Imported GLBs may carry transform actions on part nodes.
    # These actions can override externally provided node transforms at render time.
    for obj in bpy.data.objects:
        try:
            if obj.animation_data is not None:
                obj.animation_data_clear()
        except Exception:
            continue


def _scene_transform_roots(bpy):
    return [
        obj
        for obj in bpy.context.scene.objects
        if getattr(obj, "parent", None) is None
        and getattr(obj, "type", "") not in {"CAMERA", "LIGHT"}
        and str(getattr(obj, "name", "")) != "CodexGlobalTransform"
    ]


def _apply_global_transform_to_roots(bpy, mathutils, mat_rows):
    if mat_rows is None:
        return []
    try:
        global_mat = mathutils.Matrix(mat_rows)
    except Exception:
        return []
    saved = []
    for obj in _scene_transform_roots(bpy):
        try:
            old = obj.matrix_world.copy()
            obj.matrix_world = global_mat @ old
            saved.append((obj, old))
        except Exception:
            continue
    return saved


def _restore_object_matrices(saved):
    for obj, mat in saved:
        try:
            obj.matrix_world = mat
        except Exception:
            continue


def _setup_render_scene(bpy, width, height, fov_deg, device_type, gpu_index, samples, tile_size, denoise):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    try:
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = bool(denoise)
        scene.cycles.max_bounces = int(os.environ.get("CODEX_BLENDER_MAX_BOUNCES", "4"))
        scene.cycles.diffuse_bounces = 1
        scene.cycles.glossy_bounces = int(os.environ.get("CODEX_BLENDER_GLOSSY_BOUNCES", "3"))
        scene.cycles.transparent_max_bounces = 2
        if hasattr(scene.cycles, "tile_size"):
            scene.cycles.tile_size = int(tile_size)
        if denoise and hasattr(scene.cycles, "denoiser"):
            try:
                scene.cycles.denoiser = "OPTIX" if str(device_type).upper() == "OPTIX" else "OPENIMAGEDENOISE"
            except Exception:
                pass
    except Exception:
        pass
    force_cpu = os.environ.get("CODEX_BLENDER_FORCE_CPU") == "1"
    selected_device = "CPU"
    try:
        if force_cpu:
            raise RuntimeError("Forced CPU render")
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
            devices = list(getattr(prefs, "devices", []))
            gpu_found = False
            for dev_idx, dev in enumerate(devices):
                dev_type = str(getattr(dev, "type", "")).upper()
                enable = dev_idx == int(gpu_index) and dev_type in {compute_type, "CUDA", "OPTIX"}
                if enable:
                    gpu_found = True
                try:
                    dev.use = enable
                except Exception:
                    pass
            if gpu_found:
                scene.cycles.device = "GPU"
                selected_device = f"GPU:{compute_type}"
                break
        else:
            scene.cycles.device = "CPU"
    except Exception:
        scene.cycles.device = "CPU"
        selected_device = "CPU"
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = str(os.environ.get("CODEX_BLENDER_COMPOSITE_WHITE_BG", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    bg_strength = float(os.environ.get("CODEX_BLENDER_BG_STRENGTH", "0.9"))
    exposure = float(os.environ.get("CODEX_BLENDER_EXPOSURE", "0.0"))
    sun_key_energy = float(os.environ.get("CODEX_BLENDER_SUN_KEY_ENERGY", "2.5"))
    sun_fill_energy = float(os.environ.get("CODEX_BLENDER_SUN_FILL_ENERGY", "1.2"))
    area_key_energy = float(os.environ.get("CODEX_BLENDER_AREA_KEY_ENERGY", "450.0"))
    area_fill_energy = float(os.environ.get("CODEX_BLENDER_AREA_FILL_ENERGY", "70.0"))

    # Make output stable and avoid over-exposed white assets.
    try:
        scene.view_settings.view_transform = "Standard"
    except Exception:
        pass
    try:
        scene.view_settings.exposure = exposure
    except Exception:
        pass

    # White world background. Optionally decouple the camera background from the
    # lighting background so dim asset lighting does not turn the visible
    # backdrop gray.
    scene.world.use_nodes = True
    nt = scene.world.node_tree
    camera_bg_strength_raw = os.environ.get("CODEX_BLENDER_CAMERA_BG_STRENGTH")
    bg_color = _env_color("CODEX_BLENDER_BG_COLOR", (1.0, 1.0, 1.0))
    camera_bg_color = _env_color("CODEX_BLENDER_CAMERA_BG_COLOR", bg_color[:3])
    if camera_bg_strength_raw is not None:
        try:
            camera_bg_strength = max(0.0, float(camera_bg_strength_raw))
            nt.nodes.clear()
            out = nt.nodes.new(type="ShaderNodeOutputWorld")
            light_path = nt.nodes.new(type="ShaderNodeLightPath")
            bg_lighting = nt.nodes.new(type="ShaderNodeBackground")
            bg_camera = nt.nodes.new(type="ShaderNodeBackground")
            mix = nt.nodes.new(type="ShaderNodeMixShader")
            bg_lighting.inputs[0].default_value = bg_color
            bg_lighting.inputs[1].default_value = max(0.0, bg_strength)
            bg_camera.inputs[0].default_value = camera_bg_color
            bg_camera.inputs[1].default_value = camera_bg_strength
            nt.links.new(light_path.outputs["Is Camera Ray"], mix.inputs[0])
            nt.links.new(bg_lighting.outputs["Background"], mix.inputs[1])
            nt.links.new(bg_camera.outputs["Background"], mix.inputs[2])
            nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])
        except Exception:
            pass
    else:
        bg = nt.nodes.get("Background")
        if bg is not None:
            bg.inputs[0].default_value = bg_color
            bg.inputs[1].default_value = max(0.0, bg_strength)

    _apply_reflective_mesh_finish(bpy)
    _add_shadow_reflection_ground(bpy)

    cam_data = bpy.data.cameras.new("MotionCam")
    # Keep camera intrinsics consistent with project_points() in overlay pipeline:
    # use vertical FOV convention explicitly.
    cam_data.sensor_fit = "VERTICAL"
    cam_data.angle = math.radians(float(fov_deg))
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    cam_obj = bpy.data.objects.new("MotionCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    # Two sun lights for shape readability without texture washout.
    sun1 = bpy.data.lights.new("SunKey", type="SUN")
    sun1.energy = max(0.0, sun_key_energy)
    sun1_obj = bpy.data.objects.new("SunKey", sun1)
    scene.collection.objects.link(sun1_obj)
    sun1_obj.location = _env_float_tuple("CODEX_BLENDER_SUN_KEY_LOCATION", (2.0, -2.0, 3.0))
    sun1_obj.rotation_euler = _env_float_tuple("CODEX_BLENDER_SUN_KEY_ROTATION", (0.9, 0.0, 0.7))

    sun2 = bpy.data.lights.new("SunFill", type="SUN")
    sun2.energy = max(0.0, sun_fill_energy)
    sun2_obj = bpy.data.objects.new("SunFill", sun2)
    scene.collection.objects.link(sun2_obj)
    sun2_obj.location = _env_float_tuple("CODEX_BLENDER_SUN_FILL_LOCATION", (-2.0, 2.0, 2.5))
    sun2_obj.rotation_euler = _env_float_tuple("CODEX_BLENDER_SUN_FILL_ROTATION", (-0.4, 0.0, -2.3))

    area1 = bpy.data.lights.new("AreaKey", type="AREA")
    area1.energy = max(0.0, area_key_energy)
    area1.size = float(os.environ.get("CODEX_BLENDER_AREA_KEY_SIZE", "4.0"))
    area1_obj = bpy.data.objects.new("AreaKey", area1)
    scene.collection.objects.link(area1_obj)
    area1_obj.location = _env_float_tuple("CODEX_BLENDER_AREA_KEY_LOCATION", (2.8, -3.2, 4.2))
    area1_obj.rotation_euler = _env_float_tuple("CODEX_BLENDER_AREA_KEY_ROTATION", (0.9, 0.0, 0.65))

    area2 = bpy.data.lights.new("AreaFill", type="AREA")
    area2.energy = max(0.0, area_fill_energy)
    area2.size = float(os.environ.get("CODEX_BLENDER_AREA_FILL_SIZE", "5.0"))
    area2_obj = bpy.data.objects.new("AreaFill", area2)
    scene.collection.objects.link(area2_obj)
    area2_obj.location = _env_float_tuple("CODEX_BLENDER_AREA_FILL_LOCATION", (-3.5, 2.8, 3.0))
    area2_obj.rotation_euler = _env_float_tuple("CODEX_BLENDER_AREA_FILL_ROTATION", (-0.55, 0.0, -2.1))

    return scene, cam_obj, selected_device


def main():
    t_script = time.perf_counter()
    args = _parse_args()

    import bpy  # type: ignore
    import mathutils  # type: ignore

    _clear_scene(bpy)
    if args.fps is not None:
        try:
            bpy.context.scene.render.fps = int(args.fps)
        except Exception:
            pass

    glb_path = str(Path(args.glb).absolute())
    t0 = time.perf_counter()
    suffix = Path(glb_path).suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=glb_path)
        print(f"CODEX_BLENDER_TIMING=import_fbx_s={time.perf_counter() - t0:.2f}", flush=True)
    else:
        bpy.ops.import_scene.gltf(filepath=glb_path)
        print(f"CODEX_BLENDER_TIMING=import_gltf_s={time.perf_counter() - t0:.2f}", flush=True)
    t0 = time.perf_counter()
    _normalize_textured_duplicate_meshes(bpy)
    print(f"CODEX_BLENDER_TIMING=normalize_duplicate_meshes_s={time.perf_counter() - t0:.2f}", flush=True)
    t0 = time.perf_counter()
    _apply_render_only_material_boosts(bpy)
    print(f"CODEX_BLENDER_TIMING=render_material_boost_s={time.perf_counter() - t0:.2f}", flush=True)
    if not bool(args.keep_animation):
        t0 = time.perf_counter()
        _clear_imported_object_animation(bpy)
        print(f"CODEX_BLENDER_TIMING=clear_animation_s={time.perf_counter() - t0:.2f}", flush=True)

    t0 = time.perf_counter()
    scene, cam_obj, selected_device = _setup_render_scene(
        bpy,
        args.width,
        args.height,
        args.fov_deg,
        args.device_type,
        args.gpu_index,
        args.samples,
        args.tile_size,
        args.denoise,
    )
    print(f"CODEX_BLENDER_TIMING=setup_render_scene_s={time.perf_counter() - t0:.2f}", flush=True)
    if args.frame_idx is not None:
        try:
            scene.frame_set(int(args.frame_idx))
        except Exception:
            pass

    node_tfs = {}
    global_tf = None
    if args.transforms_json:
        transforms = json.loads(Path(args.transforms_json).read_text(encoding="utf-8"))
        node_tfs = transforms.get("node_transforms") or {}
        global_tf = transforms.get("global_transform")
    for name, mat_rows in node_tfs.items():
        obj = bpy.data.objects.get(str(name))
        if obj is None:
            continue
        try:
            obj.matrix_world = mathutils.Matrix(mat_rows)
        except Exception:
            continue

    print(f"CODEX_BLENDER_DEVICE_USED={selected_device}", flush=True)
    views = json.loads(Path(args.views_json).read_text(encoding="utf-8")).get("views") or []
    t_render_all = time.perf_counter()
    for view in views:
        frame_idx = view.get("frame_idx", args.frame_idx)
        if frame_idx is not None:
            try:
                scene.frame_set(int(frame_idx))
            except Exception:
                pass
        saved_global = _apply_global_transform_to_roots(bpy, mathutils, global_tf)
        eye = view["eye"]
        target = view["target"]
        up = view["up"]
        out_path = str(Path(view["out_path"]).absolute())
        _set_camera_lookat(mathutils, cam_obj, eye, target, up)
        scene.render.filepath = out_path
        t_view = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        _composite_png_over_white(out_path)
        _restore_object_matrices(saved_global)
        print(
            f"CODEX_BLENDER_TIMING=render_{str(view.get('id', 'view')).strip()}_s={time.perf_counter() - t_view:.2f}",
            flush=True,
        )
    print(f"CODEX_BLENDER_TIMING=render_all_views_s={time.perf_counter() - t_render_all:.2f}", flush=True)
    print(f"CODEX_BLENDER_TIMING=total_script_s={time.perf_counter() - t_script:.2f}", flush=True)


if __name__ == "__main__":
    main()
