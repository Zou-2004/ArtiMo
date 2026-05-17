#!/usr/bin/env python3
import math
import os

import numpy as np

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    torch = None  # type: ignore
    _HAS_TORCH = False


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_raster_device():
    if not _HAS_TORCH:
        return None
    if not _env_true("CODEX_TORCH_RASTER", True):
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _env_true("CODEX_TORCH_RASTER_ALLOW_CPU", False):
        return torch.device("cpu")
    return None


def get_linalg_device():
    if not _HAS_TORCH:
        return None
    if not _env_true("CODEX_TORCH_LINALG", True):
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _env_true("CODEX_TORCH_LINALG_ALLOW_CPU", False):
        return torch.device("cpu")
    return None


def _camera_pose_from_lookat(eye, target, up) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - eye
    fn = float(np.linalg.norm(forward))
    if fn <= 1.0e-12:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        forward = forward / fn
    right = np.cross(forward, up)
    rn = float(np.linalg.norm(right))
    if rn <= 1.0e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    true_up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _project_points_tensor(points, camera, resolution, device, fov_deg: float = 50.0):
    width = int(resolution[0])
    height = int(resolution[1])
    eye, target, up = camera
    pose = _camera_pose_from_lookat(eye, target, up)
    view = np.linalg.inv(pose)
    view_t = torch.as_tensor(view, dtype=torch.float32, device=device)
    pts = torch.as_tensor(np.asarray(points, dtype=np.float32), dtype=torch.float32, device=device)
    ones = torch.ones((pts.shape[0], 1), dtype=torch.float32, device=device)
    homog = torch.cat([pts, ones], dim=1)
    cam_pts = homog @ view_t.T
    z = -cam_pts[:, 2]
    z_safe = torch.where(torch.abs(z) > 1.0e-8, z, torch.full_like(z, 1.0e-8))
    fov = math.radians(float(fov_deg))
    aspect = float(width) / float(max(1, height))
    f = 1.0 / math.tan(fov * 0.5)
    x_ndc = (cam_pts[:, 0] * f / aspect) / z_safe
    y_ndc = (cam_pts[:, 1] * f) / z_safe
    x_screen = (x_ndc + 1.0) * 0.5 * float(max(0, width - 1))
    y_screen = (1.0 - (y_ndc + 1.0) * 0.5) * float(max(0, height - 1))
    return x_screen, y_screen, z


def rasterize_points_torch(
    points_by_link,
    colors_by_link,
    camera,
    resolution,
    point_size: int = 1,
    fov_deg: float = 50.0,
):
    device = get_raster_device()
    if device is None:
        return None

    width = int(resolution[0])
    height = int(resolution[1])
    link_names = list(points_by_link.keys())
    if width <= 0 or height <= 0:
        return {
            "image": np.zeros((0, 0, 3), dtype=np.uint8),
            "owner": np.zeros((0, 0), dtype=np.int32),
            "link_names": link_names,
            "visible_px": np.zeros((len(link_names),), dtype=np.int32),
            "visible_ratio": np.zeros((len(link_names),), dtype=np.float32),
            "projected_counts": {str(name): 0 for name in link_names},
        }

    point_size = max(1, int(point_size))
    offsets = None
    if point_size > 1:
        half = point_size // 2
        offs = torch.arange(-half, -half + point_size, dtype=torch.int32, device=device)
        oy, ox = torch.meshgrid(offs, offs, indexing="ij")
        offsets = torch.stack([ox.reshape(-1), oy.reshape(-1)], dim=1)

    all_pixels = []
    all_depths = []
    all_owner = []
    all_colors = []
    projected_counts = {}

    for link_idx, link_name in enumerate(link_names):
        points = np.asarray(points_by_link.get(link_name), dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            projected_counts[str(link_name)] = 0
            continue
        try:
            xs_f, ys_f, zs = _project_points_tensor(points, camera, resolution, device=device, fov_deg=fov_deg)
        except Exception:
            return None
        xs = torch.round(xs_f).to(torch.int32)
        ys = torch.round(ys_f).to(torch.int32)
        valid = (zs > 0.0) & torch.isfinite(zs) & (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
        projected_counts[str(link_name)] = int(torch.count_nonzero(valid).item())
        if not bool(torch.any(valid)):
            continue
        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]

        if offsets is not None:
            xs = xs[:, None] + offsets[:, 0].view(1, -1)
            ys = ys[:, None] + offsets[:, 1].view(1, -1)
            zs = zs[:, None].expand(-1, offsets.shape[0])
            keep = (xs >= 0) & (ys >= 0) & (xs < width) & (ys < height)
            if not bool(torch.any(keep)):
                continue
            xs = xs[keep]
            ys = ys[keep]
            zs = zs[keep]

        pixel_idx = ys.to(torch.int64) * int(width) + xs.to(torch.int64)
        owner = torch.full((pixel_idx.shape[0],), int(link_idx), dtype=torch.int32, device=device)
        rgba = np.clip(np.asarray((colors_by_link or {}).get(link_name, [0.0, 0.0, 0.0, 1.0]), dtype=np.float32), 0.0, 1.0)
        color = torch.as_tensor((rgba[:3] * 255.0).astype(np.uint8), dtype=torch.uint8, device=device).view(1, 3)
        color = color.expand(pixel_idx.shape[0], 3)

        all_pixels.append(pixel_idx)
        all_depths.append(zs.to(torch.float32))
        all_owner.append(owner)
        all_colors.append(color)

    if not all_pixels:
        return {
            "image": np.full((height, width, 3), 255, dtype=np.uint8),
            "owner": np.full((height, width), -1, dtype=np.int32),
            "link_names": link_names,
            "visible_px": np.zeros((len(link_names),), dtype=np.int32),
            "visible_ratio": np.zeros((len(link_names),), dtype=np.float32),
            "projected_counts": projected_counts,
        }

    pixel_idx = torch.cat(all_pixels, dim=0)
    depths = torch.cat(all_depths, dim=0)
    owner = torch.cat(all_owner, dim=0)
    colors = torch.cat(all_colors, dim=0)

    order = torch.argsort(depths, stable=True)
    pixel_idx = pixel_idx[order]
    owner = owner[order]
    colors = colors[order]

    order = torch.argsort(pixel_idx, stable=True)
    pixel_idx = pixel_idx[order]
    owner = owner[order]
    colors = colors[order]

    unique_pixels, counts = torch.unique_consecutive(pixel_idx, return_counts=True)
    first = torch.cumsum(counts, dim=0) - counts
    chosen_owner = owner[first]
    chosen_colors = colors[first]

    owner_flat = torch.full((height * width,), -1, dtype=torch.int32, device=device)
    owner_flat[unique_pixels] = chosen_owner
    image_flat = torch.full((height * width, 3), 255, dtype=torch.uint8, device=device)
    image_flat[unique_pixels] = chosen_colors

    visible_px_t = torch.bincount(chosen_owner.to(torch.long), minlength=len(link_names)).to(torch.int32)
    visible_px = visible_px_t.detach().cpu().numpy()
    visible_ratio = visible_px.astype(np.float32) / float(max(1, width * height))

    return {
        "image": image_flat.view(height, width, 3).detach().cpu().numpy(),
        "owner": owner_flat.view(height, width).detach().cpu().numpy(),
        "link_names": link_names,
        "visible_px": visible_px,
        "visible_ratio": visible_ratio,
        "projected_counts": projected_counts,
    }


def _umeyama_similarity_numpy(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected matching Nx3 arrays, got {src.shape} and {dst.shape}")
    n = src.shape[0]
    mu_x = src.mean(axis=0)
    mu_y = dst.mean(axis=0)
    x = src - mu_x
    y = dst - mu_y
    cov = (y.T @ x) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_x = (x * x).sum() / float(n)
    scale = float(np.trace(np.diag(D) @ S) / max(var_x, 1.0e-12))
    t = mu_y - scale * (R @ mu_x)
    T = np.eye(4)
    T[:3, :3] = scale * R
    T[:3, 3] = t
    return T, scale, R, t


def umeyama_similarity(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    device = get_linalg_device()
    if device is None:
        return _umeyama_similarity_numpy(src, dst)

    try:
        src_t = torch.as_tensor(src, dtype=torch.float64, device=device)
        dst_t = torch.as_tensor(dst, dtype=torch.float64, device=device)
        if src_t.shape != dst_t.shape or src_t.ndim != 2 or src_t.shape[1] != 3:
            raise ValueError(f"Expected matching Nx3 arrays, got {tuple(src_t.shape)} and {tuple(dst_t.shape)}")
        n = src_t.shape[0]
        mu_x = src_t.mean(dim=0)
        mu_y = dst_t.mean(dim=0)
        x = src_t - mu_x
        y = dst_t - mu_y
        cov = (y.transpose(0, 1) @ x) / float(n)
        U, D, Vh = torch.linalg.svd(cov, full_matrices=False)
        S = torch.eye(3, dtype=torch.float64, device=device)
        if float((torch.det(U) * torch.det(Vh)).item()) < 0.0:
            S[2, 2] = -1.0
        R = U @ S @ Vh
        var_x = (x * x).sum() / float(n)
        scale = torch.sum(D * torch.diagonal(S)) / torch.clamp(var_x, min=1.0e-12)
        t = mu_y - scale * (R @ mu_x)
        T = torch.eye(4, dtype=torch.float64, device=device)
        T[:3, :3] = scale * R
        T[:3, 3] = t
        return (
            T.detach().cpu().numpy(),
            float(scale.item()),
            R.detach().cpu().numpy(),
            t.detach().cpu().numpy(),
        )
    except Exception:
        return _umeyama_similarity_numpy(src, dst)
