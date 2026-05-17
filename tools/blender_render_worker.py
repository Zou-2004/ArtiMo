#!/usr/bin/env python3
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))


def _parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    ap = argparse.ArgumentParser(description="Persistent Blender render worker")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--device_type", default="CUDA")
    ap.add_argument("--gpu_index", type=int, default=0)
    return ap.parse_args(argv)


def _json_reply(conn, obj):
    conn.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def _handle_client(conn, state):
    raw = b""
    while not raw.endswith(b"\n"):
        chunk = conn.recv(65536)
        if not chunk:
            break
        raw += chunk
    if not raw:
        return
    req = json.loads(raw.decode("utf-8"))
    op = str(req.get("op") or "").strip().lower()
    if op == "ping":
        _json_reply(conn, {"ok": True, "status": "ready"})
        return
    if op == "shutdown":
        _json_reply(conn, {"ok": True, "status": "bye"})
        state["shutdown"] = True
        return
    if op != "render_views":
        _json_reply(conn, {"ok": False, "error": f"unsupported op: {op}"})
        return
    try:
        out = _render_views_request(req, state)
        _json_reply(conn, {"ok": True, **out})
    except Exception as exc:
        _json_reply(conn, {"ok": False, "error": str(exc)})


def _ensure_scene_loaded(bpy, views_mod, req, state):
    glb_path = str(Path(req["glb_path"]).resolve())
    keep_animation = bool(req.get("keep_animation", False))
    fps = req.get("fps")
    fps_sig = int(fps) if keep_animation and fps is not None else None
    try:
        st = Path(glb_path).stat()
        file_sig = (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        file_sig = (None, None)
    scene_sig = (glb_path, keep_animation, fps_sig, file_sig[0], file_sig[1])
    if state.get("scene_sig") == scene_sig and state.get("scene") is not None and state.get("cam_obj") is not None:
        return
    t0 = time.perf_counter()
    views_mod._clear_scene(bpy)
    if fps is not None:
        try:
            bpy.context.scene.render.fps = int(fps)
        except Exception:
            pass
    suffix = Path(glb_path).suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=glb_path)
        print(f"CODEX_BLENDER_TIMING=worker_import_fbx_s={time.perf_counter() - t0:.2f}", flush=True)
    else:
        bpy.ops.import_scene.gltf(filepath=glb_path)
        print(f"CODEX_BLENDER_TIMING=worker_import_gltf_s={time.perf_counter() - t0:.2f}", flush=True)
    t0 = time.perf_counter()
    views_mod._normalize_textured_duplicate_meshes(bpy)
    print(f"CODEX_BLENDER_TIMING=worker_normalize_duplicate_meshes_s={time.perf_counter() - t0:.2f}", flush=True)
    t0 = time.perf_counter()
    views_mod._apply_render_only_material_boosts(bpy)
    print(f"CODEX_BLENDER_TIMING=worker_render_material_boost_s={time.perf_counter() - t0:.2f}", flush=True)
    if not keep_animation:
        t0 = time.perf_counter()
        views_mod._clear_imported_object_animation(bpy)
        print(f"CODEX_BLENDER_TIMING=worker_clear_animation_s={time.perf_counter() - t0:.2f}", flush=True)
    baseline_object_matrices = {}
    for obj in bpy.data.objects:
        try:
            baseline_object_matrices[obj.name] = [list(row) for row in obj.matrix_world]
        except Exception:
            continue
    t0 = time.perf_counter()
    scene, cam_obj, selected_device = views_mod._setup_render_scene(
        bpy,
        int(req["width"]),
        int(req["height"]),
        float(req.get("fov_deg", 50.0)),
        str(req.get("device_type") or state["device_type"]),
        int(req.get("gpu_index", state["gpu_index"])),
        int(req.get("samples", 32)),
        int(req.get("tile_size", 256)),
        bool(req.get("denoise", False)),
    )
    print(f"CODEX_BLENDER_TIMING=worker_setup_render_scene_s={time.perf_counter() - t0:.2f}", flush=True)
    print(f"CODEX_BLENDER_DEVICE_USED={selected_device}", flush=True)
    state["scene_sig"] = scene_sig
    state["scene"] = scene
    state["cam_obj"] = cam_obj
    state["selected_device"] = selected_device
    state["baseline_object_matrices"] = baseline_object_matrices


def _apply_node_transforms(bpy, mathutils, node_tfs):
    for name, mat_rows in (node_tfs or {}).items():
        obj = bpy.data.objects.get(str(name))
        if obj is None:
            continue
        try:
            obj.matrix_world = mathutils.Matrix(mat_rows)
        except Exception:
            continue


def _apply_global_transform_to_roots(bpy, views_mod, mathutils, mat_rows):
    return views_mod._apply_global_transform_to_roots(bpy, mathutils, mat_rows)


def _restore_object_matrices(views_mod, saved):
    views_mod._restore_object_matrices(saved)


def _render_views_request(req, state):
    import bpy  # type: ignore
    import mathutils  # type: ignore
    import blender_render_views as views_mod  # type: ignore

    _ensure_scene_loaded(bpy, views_mod, req, state)
    scene = state["scene"]
    cam_obj = state["cam_obj"]

    scene.render.resolution_x = int(req["width"])
    scene.render.resolution_y = int(req["height"])
    try:
        cam_obj.data.sensor_fit = "VERTICAL"
        cam_obj.data.angle = views_mod.math.radians(float(req.get("fov_deg", 50.0)))
    except Exception:
        pass
    try:
        scene.cycles.samples = int(req.get("samples", 32))
        scene.cycles.use_denoising = bool(req.get("denoise", False))
        if hasattr(scene.cycles, "tile_size"):
            scene.cycles.tile_size = int(req.get("tile_size", 256))
    except Exception:
        pass
    if req.get("fps") is not None:
        try:
            scene.render.fps = int(req.get("fps"))
        except Exception:
            pass

    # Reset any per-request transforms back to the imported baseline.
    for name, mat_rows in (state.get("baseline_object_matrices") or {}).items():
        obj = bpy.data.objects.get(str(name))
        if obj is None:
            continue
        try:
            obj.matrix_world = mathutils.Matrix(mat_rows)
        except Exception:
            continue
    _apply_node_transforms(bpy, mathutils, req.get("node_transforms") or {})

    timings = {}
    t_all = time.perf_counter()
    for view in req.get("views") or []:
        frame_idx = view.get("frame_idx", req.get("frame_idx"))
        if frame_idx is not None:
            try:
                scene.frame_set(int(frame_idx))
            except Exception:
                pass
        saved_global = _apply_global_transform_to_roots(bpy, views_mod, mathutils, req.get("global_transform"))
        views_mod._set_camera_lookat(mathutils, cam_obj, view["eye"], view["target"], view["up"])
        out_path = str(Path(view["out_path"]).resolve())
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = out_path
        t0 = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        _restore_object_matrices(views_mod, saved_global)
        timings[str(view.get("id", Path(out_path).stem))] = time.perf_counter() - t0
    timings["all_views_s"] = time.perf_counter() - t_all
    return {
        "device_used": state.get("selected_device"),
        "timings": timings,
        "scene_sig": list(state.get("scene_sig") or []),
    }


def main():
    args = _parse_args()
    socket_path = str(Path(args.socket).resolve())
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    except Exception:
        pass
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(8)
    os.chmod(socket_path, 0o600)
    state = {
        "shutdown": False,
        "device_type": str(args.device_type),
        "gpu_index": int(args.gpu_index),
        "scene_sig": None,
        "scene": None,
        "cam_obj": None,
        "selected_device": None,
        "baseline_object_matrices": {},
    }
    print(f"CODEX_BLENDER_WORKER_READY={socket_path}", flush=True)
    try:
        while not state["shutdown"]:
            conn, _addr = server.accept()
            try:
                _handle_client(conn, state)
            finally:
                conn.close()
    finally:
        try:
            server.close()
        except Exception:
            pass
        try:
            os.unlink(socket_path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
