#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import loop_render as lr


_ROTATION_ARROW_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "arrow.png"
_ROTATION_ARROW_TEMPLATE_CACHE: dict[str, Image.Image | None] = {}


def wheel_child_links_from_plan(plan: dict | None, joints: list[dict], required_links: list[str] | None = None) -> list[str]:
    nl_parse = (plan or {}).get("nl_parse") or {}
    wheel_joints = [str(x).strip() for x in (nl_parse.get("wheel_joints") or []) if str(x).strip()]
    joint_to_child = {}
    for j in joints or []:
        jn = str(j.get("name") or "").strip()
        child = str(j.get("child") or "").strip()
        if jn and child:
            joint_to_child[jn] = child
    out = []
    for jn in wheel_joints:
        ln = joint_to_child.get(jn)
        if not ln:
            continue
        if required_links and ln not in required_links:
            continue
        if ln not in out:
            out.append(ln)
    return out


def semantics_wheel_links(asset_root: Path, required_links: list[str] | None = None) -> list[str]:
    path = Path(asset_root) / "semantics.txt"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = str(line or "").strip()
        if not s:
            continue
        parts = s.split()
        if not parts:
            continue
        link_name = str(parts[0]).strip()
        if "wheel" not in {str(x).strip().lower() for x in parts[1:]}:
            continue
        if required_links and link_name not in required_links:
            continue
        if link_name not in out:
            out.append(link_name)
    return out


def control_pattern_wheel_links(plan: dict | None, joints: list[dict], required_links: list[str] | None = None) -> list[str]:
    timeline = (plan or {}).get("timeline") or []
    joint_to_child = {}
    joint_type = {}
    for j in joints or []:
        jn = str(j.get("name") or "").strip()
        child = str(j.get("child") or "").strip()
        jt = str(j.get("type") or "").strip().lower()
        if jn and child:
            joint_to_child[jn] = child
            joint_type[jn] = jt
    out = []
    has_base_motion = False
    for seg in timeline:
        if not isinstance(seg, dict):
            continue
        for ctrl in seg.get("controls") or []:
            if not isinstance(ctrl, dict):
                continue
            mode = str(ctrl.get("mode") or ctrl.get("type") or "").strip().lower()
            if mode in {"base_velocity", "base_velocity_decay", "base", "base_decay"}:
                has_base_motion = True
            if mode not in {"joint_velocity", "joint_position"}:
                continue
            joint_name = str(ctrl.get("joint") or "").strip()
            child = joint_to_child.get(joint_name)
            if not child:
                continue
            if joint_type.get(joint_name) not in {"continuous", "revolute"}:
                continue
            if required_links and child not in required_links:
                continue
            if child not in out:
                out.append(child)
    return out if has_base_motion and len(out) >= 2 else []


def wheel_links_for_asset(plan: dict | None, asset_root: Path, joints: list[dict], required_links: list[str] | None = None) -> list[str]:
    for fn in (wheel_child_links_from_plan,):
        links = fn(plan, joints, required_links=required_links)
        if links:
            return links
    links = semantics_wheel_links(asset_root, required_links=required_links)
    if links:
        return links
    return control_pattern_wheel_links(plan, joints, required_links=required_links)


def segment_frame_window(sample_row: dict, traj_len: int, fps: int) -> tuple[int, int]:
    if int(traj_len) <= 0:
        return 0, 0
    try:
        t0 = float(sample_row.get("segment_t0", 0.0))
    except Exception:
        t0 = 0.0
    try:
        t1 = float(sample_row.get("segment_t1", t0))
    except Exception:
        t1 = t0
    i0 = int(round(t0 * float(max(1, fps))))
    i1 = int(round(t1 * float(max(1, fps)))) - 1
    i0 = max(0, min(int(traj_len) - 1, i0))
    i1 = max(0, min(int(traj_len) - 1, i1))
    if i1 < i0:
        i1 = i0
    return i0, i1


def _draw_arrow(draw: ImageDraw.ImageDraw, p0, p1, color, width=4):
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    draw.line((x0, y0, x1, y1), fill=color, width=max(1, int(width)))
    dx = x1 - x0
    dy = y1 - y0
    n = math.hypot(dx, dy)
    if n <= 1.0e-6:
        return
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    head_len = max(8.0, 2.6 * float(width))
    head_w = max(8.0, 2.2 * float(width))
    left = (x1 - ux * head_len + px * 0.5 * head_w, y1 - uy * head_len + py * 0.5 * head_w)
    right = (x1 - ux * head_len - px * 0.5 * head_w, y1 - uy * head_len - py * 0.5 * head_w)
    draw.polygon([(x1, y1), left, right], fill=color)


def _draw_trace_arrow(
    img_arr: np.ndarray,
    point: np.ndarray,
    direction: np.ndarray,
    color: np.ndarray,
    radius: int,
    thickness: int,
    offset_sign: float = 0.0,
):
    p = np.asarray(point, dtype=float).reshape(-1)
    d = np.asarray(direction, dtype=float).reshape(-1)
    if p.size < 2 or d.size < 2:
        return
    d = d[:2]
    dn = float(np.linalg.norm(d))
    if dn <= 1.0e-6:
        return
    u = d / dn
    perp = np.asarray([-u[1], u[0]], dtype=float)
    arrow_len = float(max(radius * 6.0, min(max(radius * 7.4, 20.0), 0.82 * dn)))
    head_len = float(max(radius * 2.2, min(max(radius * 2.8, 9.0), 0.36 * arrow_len)))
    head_w = float(max(radius * 5.2, min(max(radius * 6.2, 16.0), 0.48 * arrow_len)))
    arrow_t = int(max(2, min(int(max(3, thickness + 1)), int(max(3, round(radius * 1.05))))))
    arrow_t_bg = int(max(arrow_t + 1, 4))
    offset = perp * float(offset_sign) * max(0.75, radius * 0.25)
    center = p[:2] + offset
    tail = center - u * (0.38 * arrow_len)
    tip = center + u * (0.62 * arrow_len)
    shaft_end = tip - u * head_len
    left = tip - u * head_len + perp * (0.5 * head_w)
    right = tip - u * head_len - perp * (0.5 * head_w)
    inner_color = np.asarray(np.clip(np.round(np.asarray(color, dtype=float) * 0.62), 0, 255), dtype=np.uint8)
    outline_color = np.asarray([0, 0, 0], dtype=np.uint8)
    for col, thick in ((outline_color, arrow_t_bg), (inner_color, arrow_t)):
        lr.gop._draw_line(img_arr, tail[0], tail[1], shaft_end[0], shaft_end[1], col, thickness=thick)
        lr.gop._draw_line(img_arr, tip[0], tip[1], left[0], left[1], col, thickness=thick)
        lr.gop._draw_line(img_arr, tip[0], tip[1], right[0], right[1], col, thickness=thick)


def _draw_rotational_arrow_reference_style(
    image: np.ndarray,
    *,
    target_rect: tuple[float, float, float, float],
    color: np.ndarray,
    rot_dir: str,
    stroke_width: int = 7,
):
    def _load_template(which: str) -> Image.Image | None:
        def _largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
            h, w = mask.shape
            visited = np.zeros((h, w), dtype=bool)
            best_count = 0
            best_bbox = None
            ys, xs = np.where(mask)
            for sy, sx in zip(ys.tolist(), xs.tolist()):
                if visited[sy, sx]:
                    continue
                stack = [(sy, sx)]
                visited[sy, sx] = True
                count = 0
                min_x = max_x = int(sx)
                min_y = max_y = int(sy)
                while stack:
                    y, x = stack.pop()
                    count += 1
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                if count > best_count:
                    best_count = count
                    best_bbox = (min_x, min_y, max_x + 1, max_y + 1)
            return best_bbox

        key = str(which).strip().lower()
        if key in _ROTATION_ARROW_TEMPLATE_CACHE:
            return _ROTATION_ARROW_TEMPLATE_CACHE[key]
        if not _ROTATION_ARROW_TEMPLATE_PATH.exists():
            _ROTATION_ARROW_TEMPLATE_CACHE[key] = None
            return None
        try:
            src = Image.open(_ROTATION_ARROW_TEMPLATE_PATH).convert("L")
            if key == "ccw":
                crop_box = (20, 70, 650, 730)
            else:
                crop_box = (720, 70, 1350, 730)
            crop = src.crop(crop_box)
            arr = np.asarray(crop, dtype=np.uint8)
            alpha = np.clip((185.0 - arr.astype(np.float32)) * 3.8, 0.0, 255.0).astype(np.uint8)
            comp_bbox = _largest_component_bbox(alpha > 8)
            if comp_bbox is None:
                _ROTATION_ARROW_TEMPLATE_CACHE[key] = None
                return None
            x0, y0, x1, y1 = comp_bbox
            alpha_img = Image.fromarray(alpha[y0:y1, x0:x1], mode="L")
            pad = max(18, int(round(0.10 * max(alpha_img.size))))
            padded = Image.new("L", (alpha_img.size[0] + 2 * pad, alpha_img.size[1] + 2 * pad), 0)
            padded.paste(alpha_img, (pad, pad))
            alpha_img = padded
            _ROTATION_ARROW_TEMPLATE_CACHE[key] = alpha_img
            return alpha_img
        except Exception:
            _ROTATION_ARROW_TEMPLATE_CACHE[key] = None
            return None

    rgb = tuple(int(v) for v in np.asarray(color, dtype=np.uint8).reshape(-1)[:3])
    rx0, ry0, rx1, ry1 = [float(v) for v in target_rect]
    box_w = max(8.0, rx1 - rx0)
    box_h = max(8.0, ry1 - ry0)
    pil = Image.fromarray(image).convert("RGBA")

    templ_alpha = _load_template(rot_dir)
    if templ_alpha is not None:
        tw, th = templ_alpha.size
        fit_scale = 0.95
        scale = min(box_w / float(max(1, tw)), box_h / float(max(1, th))) * fit_scale
        out_w = max(8, int(round(float(tw) * scale)))
        out_h = max(8, int(round(float(th) * scale)))
        alpha_resized = templ_alpha.resize((out_w, out_h), Image.Resampling.LANCZOS)
        fill_alpha = alpha_resized.filter(ImageFilter.MaxFilter(size=max(3, 2 * (int(max(3, stroke_width - 2)) // 2) + 1)))
        outline_alpha = fill_alpha.filter(ImageFilter.MaxFilter(size=3))
        rgba_outline = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        rgba_outline.putalpha(outline_alpha)
        rgba_fill = Image.new("RGBA", (out_w, out_h), rgb + (0,))
        rgba_fill.putalpha(fill_alpha)
        px = int(round(rx0 + 0.5 * (box_w - out_w)))
        py = int(round(ry0 + 0.5 * (box_h - out_h)))
        pil.alpha_composite(rgba_outline, (px, py))
        pil.alpha_composite(rgba_fill, (px, py))
        image[:] = np.asarray(pil.convert("RGB"), dtype=np.uint8)
        return

    # Fallback vector draw if the reference template file is unavailable.
    cx = 0.5 * (rx0 + rx1)
    cy = 0.5 * (ry0 + ry1)
    radius = 0.5 * min(box_w, box_h) - 5.0
    draw = ImageDraw.Draw(pil)
    if str(rot_dir).strip().lower() == "cw":
        angles = np.linspace(np.deg2rad(248.0), np.deg2rad(-68.0), 84)
    else:
        angles = np.linspace(np.deg2rad(-68.0), np.deg2rad(248.0), 84)
    pts = np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    draw.line([tuple(map(float, p)) for p in pts], fill=rgb, width=int(max(3, stroke_width)), joint="curve")
    start = np.asarray(pts[0], dtype=float)
    cap_r = 0.52 * float(max(3, stroke_width))
    draw.ellipse((float(start[0] - cap_r), float(start[1] - cap_r), float(start[0] + cap_r), float(start[1] + cap_r)), fill=rgb)
    tip = np.asarray(pts[-1], dtype=float)
    tangent = np.asarray(pts[-1] - pts[-4], dtype=float)
    n = float(np.linalg.norm(tangent))
    if n > 1.0e-6:
        u = tangent / n
        perp = np.asarray([-u[1], u[0]], dtype=float)
        head_len = max(16.0, radius * 0.95)
        head_w = max(16.0, radius * 1.05)
        base_center = tip - u * head_len
        left = base_center + perp * (0.52 * head_w)
        right = base_center - perp * (0.52 * head_w)
        draw.polygon([tuple(map(float, tip)), tuple(map(float, left)), tuple(map(float, right))], fill=rgb)
    image[:] = np.asarray(pil.convert("RGB"), dtype=np.uint8)


def _draw_rotational_arrow_inner_axis_badge(
    image: np.ndarray,
    *,
    target_rect: tuple[float, float, float, float],
    axis_tag: str,
    toward_camera: bool,
    color: np.ndarray,
):
    tag = str(axis_tag or "").strip().upper()
    if not tag:
        return
    rgb = np.asarray(color, dtype=np.uint8).reshape(-1)[:3]
    rx0, ry0, rx1, ry1 = [float(v) for v in target_rect]
    box_w = max(8.0, rx1 - rx0)
    box_h = max(8.0, ry1 - ry0)
    cx = 0.5 * (rx0 + rx1)
    cy = 0.5 * (ry0 + ry1)
    clear_r = max(18.0, 0.31 * min(box_w, box_h))
    marker_r = max(7.0, 0.28 * clear_r)
    gap = max(4.0, 0.40 * marker_r)

    width, height = image.shape[1], image.shape[0]
    axis_scale = 4
    axis_text = tag
    _tx, _ty, axis_box = lr.gop._label_box(0.0, 0.0, axis_text, axis_scale, int(width), int(height))
    axis_w = float(axis_box[2] - axis_box[0])
    group_w = 2.0 * marker_r + gap + axis_w
    marker_cx = cx - 0.5 * group_w + marker_r
    axis_cx = marker_cx + marker_r + gap + 0.5 * axis_w

    pil = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    draw.ellipse(
        (marker_cx - marker_r, cy - marker_r, marker_cx + marker_r, cy + marker_r),
        fill=(255, 255, 255, 0),
        outline=tuple(int(v) for v in rgb) + (255,),
        width=2,
    )
    if toward_camera:
        dot_r = max(3.5, 0.42 * marker_r)
        draw.ellipse(
            (marker_cx - dot_r, cy - dot_r, marker_cx + dot_r, cy + dot_r),
            fill=tuple(int(v) for v in rgb) + (255,),
        )
    else:
        arm = max(3.0, 0.34 * marker_r)
        draw.line(
            (marker_cx - arm, cy - arm, marker_cx + arm, cy + arm),
            fill=(0, 0, 0, 255),
            width=4,
        )
        draw.line(
            (marker_cx - arm, cy + arm, marker_cx + arm, cy - arm),
            fill=(0, 0, 0, 255),
            width=4,
        )
        draw.line(
            (marker_cx - arm, cy - arm, marker_cx + arm, cy + arm),
            fill=tuple(int(v) for v in rgb) + (255,),
            width=2,
        )
        draw.line(
            (marker_cx - arm, cy + arm, marker_cx + arm, cy - arm),
            fill=tuple(int(v) for v in rgb) + (255,),
            width=2,
        )
    image[:] = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    lr.gop.draw_text(
        image,
        axis_cx,
        cy,
        axis_text,
        scale=axis_scale,
        color=tuple(int(v) for v in rgb),
        bg=None,
    )


def _draw_prismatic_arrow_reference_style(
    image: np.ndarray,
    *,
    target_rect: tuple[float, float, float, float],
    color: np.ndarray,
    direction_2d: np.ndarray | list[float] | tuple[float, float] | None = None,
    stroke_width: int = 6,
):
    rgb = np.asarray(color, dtype=np.uint8).reshape(-1)[:3]
    rx0, ry0, rx1, ry1 = [float(v) for v in target_rect]
    cx = 0.5 * (rx0 + rx1)
    cy = 0.5 * (ry0 + ry1)
    half_len = max(12.0, 0.28 * min(rx1 - rx0, ry1 - ry0) + 10.0)
    vec = np.asarray(direction_2d if direction_2d is not None else [0.0, -1.0], dtype=float).reshape(-1)
    if vec.size < 2:
        vec = np.asarray([0.0, -1.0], dtype=float)
    vec = vec[:2]
    n = float(np.linalg.norm(vec))
    if n <= 1.0e-6:
        vec = np.asarray([0.0, -1.0], dtype=float)
        n = 1.0
    vec = vec / n
    start = np.asarray([cx, cy], dtype=float) - vec * half_len
    end = np.asarray([cx, cy], dtype=float) + vec * half_len
    lr._draw_arrow_with_bg(image, start, end, np.asarray(rgb, dtype=np.uint8))


def _rect_intersection_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = [float(x) for x in a]
    bx0, by0, bx1, by1 = [float(x) for x in b]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return float(iw * ih)


def _expand_box(box, pad: int, resolution: tuple[int, int]) -> tuple[int, int, int, int]:
    if not (isinstance(box, (tuple, list)) and len(box) == 4):
        return tuple(int(v) for v in box)
    w, h = int(resolution[0]), int(resolution[1])
    x0, y0, x1, y1 = [int(v) for v in box]
    return (
        max(0, x0 - int(pad)),
        max(0, y0 - int(pad)),
        min(w - 1, x1 + int(pad)),
        min(h - 1, y1 + int(pad)),
    )


def _rgb_to_rgba01(rgb: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(rgb, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.asarray([0.0, 0.0, 0.0], dtype=float)
    arr = arr[:3]
    if float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    return np.asarray([arr[0], arr[1], arr[2], 1.0], dtype=float)


def _ensure_visible_rgb(rgb: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(rgb, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.asarray([35.0, 35.0, 35.0], dtype=float)
    arr = np.clip(arr[:3], 0.0, 255.0)
    luminance = 0.299 * float(arr[0]) + 0.587 * float(arr[1]) + 0.114 * float(arr[2])
    if luminance > 118.0:
        arr *= 118.0 / max(1.0, luminance)
    arr = np.clip(arr, 0.0, 215.0)
    return np.asarray(np.round(arr), dtype=np.uint8)


def _choose_label_position(
    box: tuple[int, int, int, int],
    text: str,
    resolution: tuple[int, int],
    scale: int,
    obstacles: list[tuple[float, float, float, float]] | None = None,
) -> tuple[int, int, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = [int(v) for v in box]
    cx = 0.5 * float(x0 + x1)
    cy = 0.5 * float(y0 + y1)
    pad = max(8, 4 * int(scale))
    candidates = [
        (x0 - pad, y0 - pad),
        (x0 - pad, cy),
        (cx, y0 - pad),
        (x1 + pad, y0 - pad),
        (x1 + pad, cy),
        (cx, y1 + pad),
    ]
    width, height = int(resolution[0]), int(resolution[1])
    best = None
    best_score = None
    obs = list(obstacles or [])
    for cand_x, cand_y in candidates:
        tx, ty, label_box = lr.gop._label_box(cand_x, cand_y, text, scale, width, height)
        overlap = 0.0
        for ob in obs:
            overlap += _rect_intersection_area(label_box, ob)
        center_penalty = abs(float(tx) - cx) + abs(float(ty) - cy)
        score = overlap * 1000.0 + center_penalty
        if best_score is None or score < best_score:
            best = (int(tx), int(ty), tuple(int(v) for v in label_box))
            best_score = score
            if overlap <= 1.0e-6:
                break
    if best is None:
        tx, ty, label_box = lr.gop._label_box(cx, cy, text, scale, width, height)
        return int(tx), int(ty), tuple(int(v) for v in label_box)
    return best


def _draw_motion_direction_tag(
    image: np.ndarray,
    *,
    anchor_rect: tuple[float, float, float, float],
    tag_text: str,
    base_rgba: np.ndarray,
    resolution: tuple[int, int],
    obstacles: list[tuple[float, float, float, float]] | None = None,
    projection_toward_camera: bool | None = None,
) -> tuple[int, int, tuple[int, int, int, int]] | None:
    _ = obstacles
    tag = str(tag_text or "").strip().upper()
    if not tag:
        return None
    scale = 3 if len(tag) <= 12 else 2
    ax0, _ay0, ax1, ay1 = [float(v) for v in anchor_rect]
    cx = 0.5 * (ax0 + ax1)
    width, height = int(resolution[0]), int(resolution[1])
    cand_y = min(float(height - 18), ay1 + 12.0)
    tx, ty, label_box = lr.gop._label_box(cx, cand_y, tag, scale, width, height)
    lr.gop.draw_label(image, tx, ty, tag, base_rgba, scale=scale)
    if projection_toward_camera is not None:
        rgb = np.asarray(base_rgba[:3], dtype=float).reshape(-1)
        if float(np.max(rgb)) <= 1.0:
            rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
        else:
            rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
        lbx0, lby0, lbx1, lby1 = [float(v) for v in label_box]
        label_h = max(1.0, lby1 - lby0)
        marker_size = max(36.0, min(52.0, label_h * 1.55))
        gap = 6.0
        mx1 = lbx0 - gap
        mx0 = mx1 - marker_size
        if mx0 < 4.0:
            mx0 = lbx1 + gap
            mx1 = mx0 + marker_size
        my0 = lby0 + 0.5 * (label_h - marker_size)
        my1 = my0 + marker_size
        _draw_axis_projection_marker(
            image,
            target_rect=(mx0, my0, mx1, my1),
            color=np.asarray(rgb, dtype=np.uint8),
            toward_camera=bool(projection_toward_camera),
            stroke_width=6,
        )
    return (int(tx), int(ty), tuple(int(v) for v in label_box))


def _axis_toward_camera(axis_world: np.ndarray, origin_world: np.ndarray, cam) -> bool:
    try:
        eye = np.asarray(cam[0], dtype=float).reshape(-1)[:3]
        origin = np.asarray(origin_world, dtype=float).reshape(-1)[:3]
        axis = np.asarray(axis_world, dtype=float).reshape(-1)[:3]
    except Exception:
        return True
    n_axis = float(np.linalg.norm(axis))
    if n_axis <= 1.0e-8:
        return True
    axis = axis / n_axis
    cam_vec = eye - origin
    n_cam = float(np.linalg.norm(cam_vec))
    if n_cam <= 1.0e-8:
        return True
    cam_vec = cam_vec / n_cam
    return bool(float(np.dot(axis, cam_vec)) >= 0.0)


def _axis_projection_note(axis_world: np.ndarray, origin_world: np.ndarray, cam) -> str:
    return "DOT OUT" if _axis_toward_camera(axis_world, origin_world, cam) else "CROSS IN"


def _draw_local_axis_marker_and_tag(
    image: np.ndarray,
    *,
    anchor_rect: tuple[float, float, float, float],
    axis_world: np.ndarray,
    axis_tag: str,
    origin_world: np.ndarray,
    cam,
    resolution: tuple[int, int],
    base_rgba: np.ndarray,
):
    rx0, ry0, rx1, ry1 = [float(v) for v in anchor_rect]
    marker_size = max(22.0, 0.24 * min(rx1 - rx0, ry1 - ry0))
    marker_rect = (
        float(rx0 + 6.0),
        float(ry0 + 6.0),
        float(rx0 + 6.0 + marker_size),
        float(ry0 + 6.0 + marker_size),
    )
    rgb = np.asarray(base_rgba[:3], dtype=float).reshape(-1)
    if float(np.max(rgb)) <= 1.0:
        rgb = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    _draw_axis_projection_marker(
        image,
        target_rect=marker_rect,
        color=np.asarray(rgb, dtype=np.uint8),
        toward_camera=_axis_toward_camera(np.asarray(axis_world, dtype=float), np.asarray(origin_world, dtype=float), cam),
        stroke_width=4,
    )


def _screen_dir_from_world_axis(rest_center, cam, resolution, axis_world, sign=1.0) -> np.ndarray:
    origin = np.asarray(rest_center, dtype=float)
    axis = np.asarray(axis_world, dtype=float).reshape(-1)
    if axis.size < 3:
        return np.asarray([1.0, 0.0], dtype=float)
    n = float(np.linalg.norm(axis[:3]))
    if n <= 1.0e-6:
        return np.asarray([1.0, 0.0], dtype=float)
    axis = axis[:3] / n
    pts3 = np.stack([origin, origin + float(sign) * axis], axis=0)
    proj = lr.gop.project_points(pts3, cam, resolution)
    if proj.shape[0] != 2 or np.any(proj[:, 2] <= 0):
        return np.asarray([1.0, 0.0], dtype=float)
    vec = np.asarray(proj[1, :2] - proj[0, :2], dtype=float)
    nv = float(np.linalg.norm(vec))
    return vec / max(1.0e-6, nv)


def _legend_axis_cardinal_dir(cam, axis_world: np.ndarray) -> np.ndarray:
    eye, target, up = cam
    pose = lr.gop.camera_pose_from_lookat(
        np.asarray(eye, dtype=float),
        np.asarray(target, dtype=float),
        np.asarray(up, dtype=float),
    )
    right_vec = np.asarray(pose[:3, 0], dtype=float)
    up_vec = np.asarray(pose[:3, 1], dtype=float)
    axis = np.asarray(axis_world, dtype=float).reshape(-1)
    if axis.size < 3:
        return np.asarray([0.0, -1.0], dtype=float)
    axis = axis[:3]
    n = float(np.linalg.norm(axis))
    if n <= 1.0e-8:
        return np.asarray([0.0, -1.0], dtype=float)
    axis = axis / n
    vx = float(np.dot(axis, right_vec))
    vy = -float(np.dot(axis, up_vec))
    if abs(vx) >= abs(vy):
        return np.asarray([1.0 if vx >= 0.0 else -1.0, 0.0], dtype=float)
    return np.asarray([0.0, 1.0 if vy >= 0.0 else -1.0], dtype=float)


def _project_axis_to_screen(
    origin_world: np.ndarray,
    cam,
    resolution,
    axis_world,
    sign: float = 1.0,
) -> tuple[np.ndarray | None, float, float]:
    origin = np.asarray(origin_world, dtype=float).reshape(-1)
    axis = np.asarray(axis_world, dtype=float).reshape(-1)
    if origin.size < 3 or axis.size < 3:
        return None, 0.0, 0.0
    axis = axis[:3]
    n = float(np.linalg.norm(axis))
    if n <= 1.0e-8:
        return None, 0.0, 0.0
    axis = axis / n
    signed_axis = axis * float(sign)
    pts3 = np.stack([origin[:3], origin[:3] + signed_axis], axis=0)
    proj = lr.gop.project_points(pts3, cam, resolution)
    if proj.shape[0] != 2 or np.any(proj[:, 2] <= 0):
        return None, 0.0, 0.0
    vec = np.asarray(proj[1, :2] - proj[0, :2], dtype=float)
    raw_norm = float(np.linalg.norm(vec))
    eye = np.asarray(cam[0], dtype=float).reshape(-1)
    cam_vec = eye[:3] - origin[:3]
    cam_norm = float(np.linalg.norm(cam_vec))
    facing = float(np.dot(signed_axis, cam_vec / max(1.0e-8, cam_norm))) if cam_norm > 1.0e-8 else 0.0
    if raw_norm <= 1.0e-8:
        return None, raw_norm, facing
    return vec / raw_norm, raw_norm, facing


def _rotation_direction_from_projected_motion(
    motion: dict | None,
    cam,
    resolution,
    axis_sign: float = 1.0,
    signed_axis_world: np.ndarray | None = None,
) -> str | None:
    if not isinstance(motion, dict):
        return None
    try:
        pts = np.stack(
            [
                np.asarray(motion["center_prev_world"], dtype=float),
                np.asarray(motion["center_curr_world"], dtype=float),
                np.asarray(motion["ref_prev_world"], dtype=float),
                np.asarray(motion["ref_curr_world"], dtype=float),
            ],
            axis=0,
        )
    except Exception:
        return None
    proj = lr.gop.project_points(pts, cam, resolution)
    if proj.shape[0] != 4 or np.any(proj[:, 2] <= 0):
        return None
    c0 = np.asarray(proj[0, :2], dtype=float)
    c1 = np.asarray(proj[1, :2], dtype=float)
    r0 = np.asarray(proj[2, :2], dtype=float)
    r1 = np.asarray(proj[3, :2], dtype=float)
    _ = axis_sign
    _ = signed_axis_world
    cross = float((r0 - c0)[0] * (r1 - c1)[1] - (r0 - c0)[1] * (r1 - c1)[0])
    if abs(cross) <= 1.0e-6:
        return None
    return "cw" if cross > 0.0 else "ccw"


def _draw_axis_projection_marker(
    image: np.ndarray,
    *,
    target_rect: tuple[float, float, float, float],
    color: np.ndarray,
    toward_camera: bool,
    stroke_width: int = 5,
):
    rgb = tuple(int(v) for v in np.asarray(color, dtype=np.uint8).reshape(-1)[:3])
    rx0, ry0, rx1, ry1 = [float(v) for v in target_rect]
    cx = 0.5 * (rx0 + rx1)
    cy = 0.5 * (ry0 + ry1)
    radius = max(14.0, 0.27 * min(rx1 - rx0, ry1 - ry0))
    pil = Image.fromarray(image).convert("RGBA")
    draw = ImageDraw.Draw(pil)
    outline_w = max(3, int(stroke_width))
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=(255, 255, 255, 0), outline=rgb + (255,), width=outline_w)
    if toward_camera:
        inner_r = max(2.0, 0.09 * radius)
        draw.ellipse((cx - inner_r - 0.7, cy - inner_r - 0.7, cx + inner_r + 0.7, cy + inner_r + 0.7), fill=(0, 0, 0, 255))
        draw.ellipse((cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r), fill=rgb + (255,))
    else:
        arm = max(2.2, min(radius - 8.5, 0.13 * radius))
        draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill=(0, 0, 0, 255), width=3)
        draw.line((cx - arm, cy + arm, cx + arm, cy - arm), fill=(0, 0, 0, 255), width=3)
        draw.line((cx - arm, cy - arm, cx + arm, cy + arm), fill=rgb + (255,), width=2)
        draw.line((cx - arm, cy + arm, cx + arm, cy - arm), fill=rgb + (255,), width=2)
    image[:] = np.asarray(pil.convert("RGB"), dtype=np.uint8)


def _signed_axis_label(axis_world: np.ndarray | list[float] | tuple[float, ...]) -> str:
    vec = np.asarray(axis_world, dtype=float).reshape(-1)
    if vec.size < 3:
        return "+X"
    vec = vec[:3]
    n = float(np.linalg.norm(vec))
    if n <= 1.0e-8:
        return "+X"
    vec = vec / n
    idx = int(np.argmax(np.abs(vec)))
    axes = ["X", "Y", "Z"]
    sign = "+" if float(vec[idx]) >= 0.0 else "-"
    return f"{sign}{axes[idx]}"


def _build_child_to_joint_map(joints: list[dict]) -> dict[str, str]:
    best: dict[str, tuple[int, str]] = {}
    for j in joints or []:
        child = str(j.get("child") or "").strip()
        jn = str(j.get("name") or "").strip()
        if not child or not jn:
            continue
        jt = str(j.get("type") or "").strip().lower()
        pri = 1 if jt == "fixed" else 0
        prev = best.get(child)
        if prev is None or pri < prev[0]:
            best[child] = (pri, jn)
    return {k: v[1] for k, v in best.items()}


def _xyz_overlay_box(width: int, height: int) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in lr._motion_axes_overlay_box(int(width), int(height)))


def _estimate_bg_color(img: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    strips = []
    pad = 6
    if y0 - pad > 0:
        strips.append(arr[max(0, y0 - pad) : y0, max(0, x0 - pad) : min(w, x1 + pad)].reshape(-1, 3))
    if y1 + pad < h:
        strips.append(arr[y1 : min(h, y1 + pad), max(0, x0 - pad) : min(w, x1 + pad)].reshape(-1, 3))
    if x0 - pad > 0:
        strips.append(arr[max(0, y0 - pad) : min(h, y1 + pad), max(0, x0 - pad) : x0].reshape(-1, 3))
    if x1 + pad < w:
        strips.append(arr[max(0, y0 - pad) : min(h, y1 + pad), x1 : min(w, x1 + pad)].reshape(-1, 3))
    if not strips:
        return (248, 248, 248)
    samples = np.concatenate([s for s in strips if s.size > 0], axis=0)
    if samples.size == 0:
        return (248, 248, 248)
    rgb = np.median(samples, axis=0)
    return tuple(int(v) for v in np.clip(np.round(rgb), 0, 255))


def _project_link_and_asset_boxes(
    *,
    asset_root: Path,
    traj_data: dict,
    frame_idx: int,
    viewspecs: dict,
    rest_center: np.ndarray,
    anchor_radius: float,
    link_name: str,
    resolution: tuple[int, int],
):
    single_asset_ctx = lr.load_asset_context(asset_root)
    _, joint_pos, base_tf = lr._frame_state_from_traj(traj_data, frame_idx)
    link_tf = lr.rp.compute_link_transforms(single_asset_ctx["links"], single_asset_ctx["joints"], joint_pos, base_tf=base_tf)
    world_link_meshes = lr.transform_link_meshes(single_asset_ctx["link_meshes"], link_tf)
    visual_links = [ln for ln, meshes in world_link_meshes.items() if meshes]
    primary_view = dict((viewspecs.get("views") or [{}])[0])
    cam = lr.compute_camera_for_viewspec(np.asarray(rest_center, dtype=float), float(max(0.05, anchor_radius)), primary_view)
    bbox_points_by_link = lr.gop.collect_link_bbox_points(world_link_meshes, visual_links)
    projected_boxes = lr.gop.project_link_boxes(bbox_points_by_link, cam, resolution)
    link_box = projected_boxes.get(link_name)
    if not (isinstance(link_box, (tuple, list)) and len(link_box) == 4):
        return None
    all_boxes = [tuple(int(v) for v in box) for box in projected_boxes.values() if isinstance(box, (tuple, list)) and len(box) == 4]
    asset_box = None
    if all_boxes:
        asset_box = (
            min(box[0] for box in all_boxes),
            min(box[1] for box in all_boxes),
            max(box[2] for box in all_boxes),
            max(box[3] for box in all_boxes),
        )
    return {
        "asset_ctx": single_asset_ctx,
        "cam": cam,
        "link_tf": link_tf,
        "world_link_meshes": world_link_meshes,
        "visual_links": visual_links,
        "projected_boxes": projected_boxes,
        "link_box": tuple(int(v) for v in link_box),
        "asset_box": (tuple(int(v) for v in asset_box) if isinstance(asset_box, (tuple, list)) and len(asset_box) == 4 else None),
    }


def _draw_wheel_context_boxes(
    image: np.ndarray,
    *,
    link_box: tuple[int, int, int, int],
    asset_box: tuple[int, int, int, int] | None,
    resolution: tuple[int, int],
    base_rgba: np.ndarray,
):
    link_box_vis = lr._scale_projected_box(
        link_box,
        resolution,
        scale=lr.TIMELINE_LINK_BBOX_SCALE,
        min_size_px=18,
    )
    if link_box_vis is None:
        link_box_vis = tuple(int(v) for v in link_box)
    lr.gop.draw_bbox_outline(image, link_box_vis, base_rgba, thickness=5)
    return link_box_vis, None


def postprocess_wheel_head_image(
    image_path: Path,
    *,
    asset_root: Path,
    traj_data: dict,
    viewspecs: dict,
    rest_center: np.ndarray,
    anchor_radius: float,
    row: dict,
    motion_label_legend: dict[str, str],
):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    link_name = str(row.get("link") or "").strip()
    if not link_name:
        img.save(image_path)
        return
    projected = _project_link_and_asset_boxes(
        asset_root=asset_root,
        traj_data=traj_data,
        frame_idx=int(row.get("frame_idx", 0)),
        viewspecs=viewspecs,
        rest_center=rest_center,
        anchor_radius=anchor_radius,
        link_name=link_name,
        resolution=(w, h),
    )
    if not isinstance(projected, dict):
        img.save(image_path)
        return
    color_map = lr.gop.build_distinct_link_color_map(projected["visual_links"])
    base_rgb = np.asarray([35, 35, 35], dtype=np.uint8)
    if link_name in color_map:
        base_rgb = np.asarray(np.clip(np.round(np.asarray(color_map[link_name][:3], dtype=float) * 255.0), 0, 255), dtype=np.uint8)
    base_rgb = _ensure_visible_rgb(base_rgb)
    base_rgba = _rgb_to_rgba01(base_rgb)
    arr_img = np.array(img, copy=True)
    link_box_vis, _asset_box_vis = _draw_wheel_context_boxes(
        arr_img,
        link_box=projected["link_box"],
        asset_box=projected["asset_box"],
        resolution=(w, h),
        base_rgba=base_rgba,
    )
    label_text = str(motion_label_legend.get(link_name, link_name))
    label_scale = max(2, int(lr.LABEL_SCALE_MOTION))
    label_lx, label_ly, _label_box = _choose_label_position(
        tuple(int(v) for v in link_box_vis),
        label_text,
        (w, h),
        label_scale,
        obstacles=[
            (8.0, 8.0, 152.0, 72.0),
            (float(w - 228), 10.0, float(w - 8), min(float(h - 8), 168.0)),
            tuple(float(v) for v in link_box_vis),
        ],
    )
    lr.gop.draw_label(arr_img, label_lx, label_ly, label_text, base_rgba, scale=label_scale)
    Image.fromarray(arr_img).save(image_path)


def _draw_custom_xyz_axes(img: Image.Image, viewspecs: dict, rest_center: np.ndarray, anchor_radius: float):
    primary_view = dict((viewspecs.get("views") or [{}])[0])
    cam = lr.compute_camera_for_viewspec(np.asarray(rest_center, dtype=float), float(max(0.05, anchor_radius)), primary_view)
    arr = np.array(img, copy=True)
    lr._draw_motion_corner_axes_box(arr, cam, img.size)
    img.paste(Image.fromarray(arr))


def postprocess_wheel_tail_image(
    image_path: Path,
    *,
    asset_root: Path,
    traj_npz_data,
    traj_data: dict,
    viewspecs: dict,
    rest_center: np.ndarray,
    anchor_radius: float,
    fps: int,
    row: dict,
    motion_label_legend: dict[str, str],
):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    link_name = str(row.get("link") or "").strip()
    if not link_name:
        img.save(image_path)
        return

    frame_idx = int(row.get("frame_idx", 0))
    resolution = (w, h)
    motion_window = segment_frame_window(row, int(np.asarray(traj_data["joint_angles"]).shape[0]), fps)
    camera_radius = max(0.05, float(anchor_radius))
    primary_view = dict((viewspecs.get("views") or [{}])[0])
    cam = lr.compute_camera_for_viewspec(np.asarray(rest_center, dtype=float), float(camera_radius), primary_view)

    projected = _project_link_and_asset_boxes(
        asset_root=asset_root,
        traj_data=traj_data,
        frame_idx=frame_idx,
        viewspecs=viewspecs,
        rest_center=rest_center,
        anchor_radius=anchor_radius,
        link_name=link_name,
        resolution=resolution,
    )
    if not isinstance(projected, dict):
        img.save(image_path)
        return
    single_asset_ctx = projected["asset_ctx"]
    cam = projected["cam"]
    link_tf = projected["link_tf"]
    world_link_meshes = projected["world_link_meshes"]
    visual_links = projected["visual_links"]
    color_map = lr.gop.build_distinct_link_color_map(visual_links)
    base_rgb = np.asarray([35, 35, 35], dtype=np.uint8)
    if link_name in color_map:
        base_rgb = np.asarray(np.clip(np.round(np.asarray(color_map[link_name][:3], dtype=float) * 255.0), 0, 255), dtype=np.uint8)
    base_rgb = _ensure_visible_rgb(base_rgb)
    base_rgba = _rgb_to_rgba01(base_rgb)
    overall_rgb = np.asarray([35, 35, 35], dtype=np.uint8)

    link_box_vis = _expand_box(projected["link_box"], 18, resolution)
    asset_box_vis = _expand_box(projected["asset_box"], 22, resolution) if isinstance(projected["asset_box"], (tuple, list)) and len(projected["asset_box"]) == 4 else None

    label_text = str(motion_label_legend.get(link_name, link_name))
    label_scale = max(2, int(lr.LABEL_SCALE_MOTION))
    label_lx, label_ly, label_box = _choose_label_position(
        tuple(int(v) for v in link_box_vis),
        label_text,
        resolution,
        label_scale,
        obstacles=[
            (8.0, 8.0, 152.0, 72.0),
            (float(w - 228), 10.0, float(w - 8), min(float(h - 8), 168.0)),
            tuple(float(v) for v in link_box_vis),
        ],
    )

    cues = row.get("motion_cues") if isinstance(row.get("motion_cues"), dict) else {}
    jt = None
    for item in cues.get("joint_trends") or []:
        if isinstance(item, dict) and str(item.get("link") or "").strip() == link_name:
            jt = item
            break
    joint_type = str((jt or {}).get("joint_type") or "").strip().lower()

    arr_img = np.array(img, copy=True)
    link_box_vis, asset_box_vis = _draw_wheel_context_boxes(
        arr_img,
        link_box=projected["link_box"],
        asset_box=projected["asset_box"],
        resolution=resolution,
        base_rgba=base_rgba,
    )
    lr.gop.draw_label(arr_img, label_lx, label_ly, label_text, base_rgba, scale=label_scale)
    _draw_local_wheel_arrow(
        arr_img,
        cam=cam,
        resolution=resolution,
        rest_center=rest_center,
        single_asset_ctx=single_asset_ctx,
        traj_data=traj_data,
        traj_npz_data=traj_npz_data,
        frame_idx=frame_idx,
        motion_window=motion_window,
        link_tf=link_tf,
        link_name=link_name,
        link_box=tuple(int(v) for v in link_box_vis),
        asset_box=(tuple(int(v) for v in asset_box_vis) if isinstance(asset_box_vis, (tuple, list)) and len(asset_box_vis) == 4 else None),
        label_box=(tuple(int(v) for v in label_box) if isinstance(label_box, (tuple, list)) and len(label_box) == 4 else None),
        base_rgb=base_rgb,
        cues=cues,
        joint_type=joint_type,
    )
    try:
        _draw_overall_motion_arrow(
            arr_img,
            asset_box=(tuple(int(v) for v in asset_box_vis) if isinstance(asset_box_vis, (tuple, list)) and len(asset_box_vis) == 4 else None),
            cam=cam,
            resolution=resolution,
            rest_center=np.asarray(rest_center, dtype=float),
            base_motion=(cues.get("base_motion") if isinstance(cues, dict) else None),
            overall_rgb=np.asarray(overall_rgb, dtype=np.uint8),
        )
    except Exception:
        pass
    Image.fromarray(arr_img).save(image_path)


def _draw_local_wheel_arrow(
    image: np.ndarray,
    *,
    cam,
    resolution,
    rest_center,
    single_asset_ctx,
    traj_data,
    traj_npz_data,
    frame_idx,
    motion_window,
    link_tf,
    link_name,
    link_box,
    asset_box,
    label_box,
    base_rgb,
    cues,
    joint_type,
):
    x0, y0, x1, y1 = [float(v) for v in link_box]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    w, h = resolution
    jt = None
    for item in (cues.get("joint_trends") or []):
        if isinstance(item, dict) and str(item.get("link") or "").strip() == link_name:
            jt = item
            break
    trend = str((jt or {}).get("trend") or "").strip().lower()
    xyz_box = _xyz_overlay_box(int(w), int(h))
    obstacles = [
        (8.0, 8.0, 152.0, 72.0),
        tuple(float(v) for v in xyz_box),
        (x0, y0, x1, y1),
    ]
    ax0 = ay0 = ax1 = ay1 = None
    if isinstance(asset_box, (tuple, list)) and len(asset_box) == 4:
        ax0, ay0, ax1, ay1 = [float(v) for v in asset_box]
        obstacles.append((ax0 - 2.0, ay0 - 2.0, ax1 + 2.0, ay1 + 2.0))
    if label_box is not None:
        obstacles.append(tuple(float(v) for v in label_box))

    def _rect_overlap_cost(rect):
        rx0, ry0, rx1, ry1 = rect
        if rx0 < 4 or ry0 < 4 or rx1 > w - 4 or ry1 > h - 4:
            return 1.0e9
        padded_rect = (rx0 - 10.0, ry0 - 10.0, rx1 + 10.0, ry1 + 10.0)
        overlap = 0.0
        for ox0, oy0, ox1, oy1 in obstacles:
            overlap += _rect_intersection_area(padded_rect, (ox0, oy0, ox1, oy1))
        return overlap

    def _score_rect(rect):
        overlap = _rect_overlap_cost(rect)
        if overlap >= 1.0e9:
            return overlap
        rx0, ry0, rx1, ry1 = rect
        dist = abs((0.5 * (rx0 + rx1)) - cx) + abs((0.5 * (ry0 + ry1)) - cy)
        return overlap * 1000.0 + dist

    def _clamp_candidate(cand_x: float, cand_y: float, cand_w: float, cand_h: float) -> tuple[float, float]:
        max_x = max(4.0, float(w) - 4.0 - cand_w)
        max_y = max(4.0, float(h) - 4.0 - cand_h)
        return (
            min(max(4.0, float(cand_x)), max_x),
            min(max(4.0, float(cand_y)), max_y),
        )

    motion_vectors = lr._compute_link_motion_vectors(
        single_asset_ctx,
        traj_data,
        frame_idx,
        link_tf,
        [link_name],
        motion_window=motion_window,
        trace_variant_index=0,
        use_best_trace_candidate=False,
        use_edge_variant_candidate=False,
    )
    motion = motion_vectors.get(link_name) if isinstance(motion_vectors, dict) else None
    tracks2d = lr._project_motion_tracks_2d(motion, cam, resolution) if isinstance(motion, dict) else []
    max_path_len_px = max((lr._track_path_len_px(track) for track in tracks2d), default=0.0)
    delta_q = float((jt or {}).get("delta_q") or 0.0)
    direction = str((jt or {}).get("direction") or "").strip().lower()
    # Local joint arrow should reflect the joint's own motion, not the link's
    # screen-space drift caused by whole-object base translation.
    if jt is not None and (trend == "static" or direction == "static" or abs(delta_q) <= 1.0e-12):
        return
    if trend == "static" and max_path_len_px <= 1.0e-6:
        return

    if joint_type in {"continuous", "revolute"}:
        rot_dir = "cw"
        child_joint = next((j for j in (single_asset_ctx.get("joints") or []) if str(j.get("child") or "").strip() == link_name), None)
        axis_local = np.asarray((child_joint or {}).get("axis") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
        center_world = np.asarray(np.asarray(link_tf.get(link_name, np.eye(4)), dtype=float)[:3, 3], dtype=float)
        visible_rot_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
        if axis_local.size >= 3 and link_name in link_tf:
            axis_local = axis_local[:3] / max(1.0e-8, float(np.linalg.norm(axis_local[:3])))
            tf = np.asarray(link_tf[link_name], dtype=float)
            world_axis = np.asarray(tf[:3, :3], dtype=float) @ axis_local
            world_axis = world_axis / max(1.0e-8, float(np.linalg.norm(world_axis)))
            if abs(delta_q) > 1.0e-12:
                rot_dir = "ccw" if delta_q > 0.0 else "cw"
                signed_joint_dir = 1.0 if delta_q > 0.0 else -1.0
            else:
                rot_dir = "ccw" if str((jt or {}).get("trend") or "").strip().lower() != "decrease" else "cw"
                signed_joint_dir = -1.0 if str((jt or {}).get("trend") or "").strip().lower() == "decrease" else 1.0
            visible_rot_axis = world_axis * float(signed_joint_dir)
            visible_rot_axis = visible_rot_axis / max(1.0e-8, float(np.linalg.norm(visible_rot_axis)))
        else:
            rot_dir = str((jt or {}).get("direction") or "").strip().lower() or "cw"
        projection_toward_camera = _axis_toward_camera(
            np.asarray(visible_rot_axis, dtype=float),
            np.asarray(center_world, dtype=float),
            cam,
        )
        box_w, box_h = 132.0, 108.0
        outer_gap = 14.0
        top_gap = outer_gap
        top_y = (ay0 if ay0 is not None else y0) - box_h - top_gap
        if label_box is not None:
            top_y = min(top_y, float(label_box[1]) - box_h - 8.0)
        half_w = 0.5 * box_w
        half_h = 0.5 * box_h
        side_gap = outer_gap
        below_gap = outer_gap
        left_anchor = ((ax0 if ax0 is not None else x0) - box_w - side_gap)
        right_anchor = ((ax1 if ax1 is not None else x1) + side_gap)
        below_y = ((ay1 if ay1 is not None else y1) + below_gap)
        candidates = [
            (cx - half_w, top_y),
            (left_anchor, cy - half_h),
            (right_anchor, cy - half_h),
            (cx - half_w, below_y),
        ]
        best_rect = None
        best_score = None
        for cand_x, cand_y in candidates:
            cand_x, cand_y = _clamp_candidate(cand_x, cand_y, box_w, box_h)
            rect = (cand_x, cand_y, cand_x + box_w, cand_y + box_h)
            score = _score_rect(rect)
            if best_score is None or score < best_score:
                best_score = score
                best_rect = rect
        if best_rect is None:
            return
        rx0, ry0, rx1, ry1 = [float(v) for v in best_rect]
        ccx = 0.5 * (rx0 + rx1)
        ccy = 0.5 * (ry0 + ry1)
        radius = min(0.5 * (rx1 - rx0), 0.5 * (ry1 - ry0)) - 8.0
        _draw_rotational_arrow_reference_style(
            image,
            target_rect=(rx0, ry0, rx1, ry1),
            color=base_rgb,
            rot_dir=rot_dir,
            stroke_width=5,
        )
        axis_tag = _signed_axis_label(visible_rot_axis)
        _proj_vec, proj_norm_px, _proj_facing = _project_axis_to_screen(
            np.asarray(center_world, dtype=float),
            cam,
            resolution,
            np.asarray(visible_rot_axis, dtype=float),
            sign=1.0,
        )
        _draw_rotational_arrow_inner_axis_badge(
            image,
            target_rect=(rx0, ry0, rx1, ry1),
            axis_tag=axis_tag,
            toward_camera=projection_toward_camera,
            color=np.asarray(base_rgb, dtype=np.uint8),
        )
        return

    tf = np.asarray(link_tf.get(link_name, np.eye(4)), dtype=float)
    center_world = np.asarray(tf[:3, 3], dtype=float)
    local_motion = cues.get("local_motion") if isinstance(cues.get("local_motion"), dict) else {}
    signed_axis_raw = None
    if isinstance(local_motion, dict):
        signed_axis_raw = local_motion.get("signed_axis_world")
    if not (isinstance(signed_axis_raw, (list, tuple, np.ndarray)) and len(signed_axis_raw) >= 3):
        signed_axis_raw = (jt or {}).get("signed_axis_world")
    if isinstance(signed_axis_raw, (list, tuple, np.ndarray)) and len(signed_axis_raw) >= 3:
        signed_axis = np.asarray(signed_axis_raw, dtype=float).reshape(-1)[:3]
        signed_axis = signed_axis / max(1.0e-8, float(np.linalg.norm(signed_axis)))
    else:
        child_joint = next((j for j in (single_asset_ctx.get("joints") or []) if str(j.get("child") or "").strip() == link_name), None)
        axis_local = np.asarray((child_joint or {}).get("axis") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
        if axis_local.size < 3 or link_name not in link_tf:
            return
        axis_local = axis_local[:3]
        axis_local = axis_local / max(1.0e-8, float(np.linalg.norm(axis_local)))
        axis_world = np.asarray(tf[:3, :3], dtype=float) @ axis_local
        axis_world = axis_world / max(1.0e-8, float(np.linalg.norm(axis_world)))
        trend_sign = 1.0 if str(trend or "").strip().lower() != "decrease" else -1.0
        signed_axis = np.asarray(axis_world * trend_sign, dtype=float)
        signed_axis = signed_axis / max(1.0e-8, float(np.linalg.norm(signed_axis)))
    # For prismatic helper arrows, use the same global-axis projection convention
    # as the top-right legend so the auxiliary cue stays visually consistent with
    # the displayed coordinate frame instead of varying with local perspective.
    prismatic_screen_dir = _legend_axis_cardinal_dir(cam, signed_axis)
    box_w, box_h = 122.0, 54.0
    outer_gap = 14.0
    top_y = (ay0 if ay0 is not None else y0) - box_h - outer_gap
    if label_box is not None:
        top_y = min(top_y, float(label_box[1]) - box_h - 10.0)
    candidates = [
        (cx - 61.0, top_y),
        (((ax0 if ax0 is not None else x0) - box_w - outer_gap), cy - 27.0),
        (((ax1 if ax1 is not None else x1) + outer_gap), cy - 27.0),
        (cx - 61.0, ((ay1 if ay1 is not None else y1) + outer_gap)),
    ]
    best_rect = None
    best_score = None
    for cand_x, cand_y in candidates:
        cand_x, cand_y = _clamp_candidate(cand_x, cand_y, box_w, box_h)
        rect = (cand_x, cand_y, cand_x + box_w, cand_y + box_h)
        score = _score_rect(rect)
        if best_score is None or score < best_score:
            best_score = score
            best_rect = rect
    if best_rect is None:
        return
    rx0, ry0, rx1, ry1 = [float(v) for v in best_rect]
    _draw_prismatic_arrow_reference_style(
        image,
        target_rect=(rx0, ry0, rx1, ry1),
        color=base_rgb,
        direction_2d=prismatic_screen_dir,
        stroke_width=6,
    )
    _draw_motion_direction_tag(
        image,
        anchor_rect=(rx0, ry0, rx1, ry1),
        tag_text=_signed_axis_label(signed_axis),
        base_rgba=_rgb_to_rgba01(base_rgb),
        resolution=(int(w), int(h)),
        obstacles=list(obstacles),
    )


def _overall_direction_from_base_motion(base_motion: dict | None, *, cam, resolution, rest_center) -> np.ndarray:
    bm = base_motion if isinstance(base_motion, dict) else {}
    axis = np.asarray(bm.get("axis_world") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
    if axis.size < 3:
        return np.asarray([1.0, 0.0], dtype=float)
    sign = -1.0 if str(bm.get("trend") or "").strip().lower() == "negative" else 1.0
    return _screen_dir_from_world_axis(rest_center, cam, resolution, axis, sign=sign)


def _draw_overall_motion_arrow(image: np.ndarray, *, asset_box, cam, resolution, rest_center, base_motion: dict | None, overall_rgb: np.ndarray):
    if not (isinstance(asset_box, (tuple, list)) and len(asset_box) == 4):
        return
    bm = base_motion if isinstance(base_motion, dict) else None
    if not isinstance(bm, dict):
        return
    trend = str(bm.get("trend") or "").strip().lower()
    if trend not in {"positive", "negative"}:
        return
    w, h = image.shape[1], image.shape[0]
    overall_dir = _overall_direction_from_base_motion(bm, cam=cam, resolution=resolution, rest_center=rest_center)
    ax0, ay0, ax1, ay1 = [float(v) for v in asset_box]
    candidates = [
        (0.5 * (ax0 + ax1), ay0 - 18.0),
        (ax1 + 14.0, 0.5 * (ay0 + ay1)),
        (ax0 - 14.0, 0.5 * (ay0 + ay1)),
        (0.5 * (ax0 + ax1), ay1 + 14.0),
    ]
    clipped = None
    for anchor in candidates:
        p0 = np.asarray(anchor, dtype=float)
        p1 = p0 + np.asarray(overall_dir, dtype=float) * 140.0
        clipped = _clip_segment_to_canvas_local(p0, p1, (w, h), margin=40.0)
        if clipped is not None:
            break
    if clipped is None:
        return
    start, end = clipped
    lr.gop._draw_line(image, float(start[0]), float(start[1]), float(end[0]), float(end[1]), overall_rgb, thickness=7)
    _draw_trace_arrow(image, end, np.asarray(end - start, dtype=float), overall_rgb, 7, 7, offset_sign=0.0)
    axis = np.asarray(bm.get("axis_world") or [1.0, 0.0, 0.0], dtype=float).reshape(-1)
    if axis.size >= 3:
        signed_axis = axis[:3] * (-1.0 if trend == "negative" else 1.0)
        tag = _signed_axis_label(signed_axis)
        mx = 0.5 * float(ax0 + ax1)
        above_y = float(ay0) - 34.0
        below_y = float(ay0) + 22.0
        tx, ty, box = lr.gop._label_box(mx, above_y, tag, 4, int(w), int(h))
        if box[1] <= 10:
            tx, ty, box = lr.gop._label_box(mx, below_y, tag, 4, int(w), int(h))
        lr.gop.draw_text(
            image,
            tx,
            ty,
            tag,
            scale=4,
            color=(20, 20, 20),
            bg=(245, 245, 245),
        )


def _clip_segment_to_canvas_local(p0, p1, resolution, margin=28.0):
    w, h = int(resolution[0]), int(resolution[1])
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    bounds = [
        (-dx, x0 - margin),
        (dx, float(w - margin) - x0),
        (-dy, y0 - margin),
        (dy, float(h - margin) - y0),
    ]
    for p, q in bounds:
        if abs(p) <= 1.0e-8:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return None
            if r < t1:
                t1 = r
    if t1 < t0:
        return None
    return (
        np.asarray([x0 + t0 * dx, y0 + t0 * dy], dtype=float),
        np.asarray([x0 + t1 * dx, y0 + t1 * dy], dtype=float),
    )
