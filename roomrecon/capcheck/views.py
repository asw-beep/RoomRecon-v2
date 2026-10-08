"""Which frames saw each surface point, from which directions, how sharply, and how much
texture the images show there.

Coverage follows COVER (Chen et al. 2026, arXiv 2604.05259): a surface is well covered when it
was seen from many different directions, here counted as occupied bins on a fixed sphere of
directions. COVER scores trained Gaussians; we score the phone's LiDAR surface before training.
A point counts as seen only where the frame's own LiDAR depth agrees with its distance, so
surfaces hidden behind furniture are not counted.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from roomrecon.capcheck.capture import Capture, Frame, load_depth_m, load_gray
from roomrecon.capcheck.surface import Surface


@dataclass
class ViewStats:
    n_views: np.ndarray
    dir_mask: np.ndarray
    n_dirs: np.ndarray
    best_footprint_mm: np.ndarray
    texture: np.ndarray
    mean_view_dir: np.ndarray


def fibonacci_sphere(n: int = 64) -> np.ndarray:
    i = np.arange(n) + 0.5
    polar = np.arccos(1 - 2 * i / n)
    azimuth = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar),
                     np.cos(polar)], axis=1)


def project(points: np.ndarray, K: np.ndarray, c2w: np.ndarray, w: int, h: int):
    """(idx, uv, z) of the points in front of the camera whose pixel lies in the image;
    z is the positive z-depth. OpenGL camera, as in surface.backproject."""
    cam = (points - c2w[:3, 3]) @ c2w[:3, :3]
    z = -cam[:, 2]
    front = z > 1e-6
    zf = np.where(front, z, 1.0)
    u = K[0, 2] + K[0, 0] * cam[:, 0] / zf
    v = K[1, 2] - K[1, 1] * cam[:, 1] / zf
    inside = front & (u >= -0.5) & (u < w - 0.5) & (v >= -0.5) & (v < h - 0.5)
    idx = np.flatnonzero(inside)
    return idx, np.stack([u[idx], v[idx]], axis=1), z[idx]


def visible(points: np.ndarray, frame: Frame, depth_m: np.ndarray, max_d=5.0, tol_abs=0.05,
            tol_rel=0.03):
    """(idx, uv) of the points this frame saw: in view, within max_d, and where the frame's
    measured depth agrees with the point's depth."""
    idx, uv, z = project(points, frame.K, frame.c2w, frame.width, frame.height)
    ui = np.clip(np.rint(uv[:, 0]).astype(int), 0, frame.width - 1)
    vi = np.clip(np.rint(uv[:, 1]).astype(int), 0, frame.height - 1)
    D = depth_m[vi, ui]
    ok = (D > 0) & (z <= max_d) & (np.abs(z - D) <= np.maximum(tol_abs, tol_rel * D))
    return idx[ok], uv[ok]


def texture_map(gray: np.ndarray) -> np.ndarray:
    g = np.hypot(ndimage.sobel(gray, axis=0), ndimage.sobel(gray, axis=1))
    return ndimage.uniform_filter(g, size=9)


def _median_by_group(idx: np.ndarray, vals: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, np.nan, np.float32)
    if len(idx) == 0:
        return out
    order = np.lexsort((vals, idx))
    idx, vals = idx[order], vals[order]
    counts = np.bincount(idx, minlength=n)
    starts = np.cumsum(counts) - counts
    has = counts > 0
    lo = starts[has] + (counts[has] - 1) // 2
    hi = starts[has] + counts[has] // 2
    out[has] = (vals[lo] + vals[hi]) / 2
    return out


def accumulate(surface: Surface, cap: Capture, frames_mask: np.ndarray | None = None) -> ViewStats:
    """Views of every surface point; `frames_mask` (one bool per frame) skips frames."""
    pts = surface.points
    n = len(pts)
    dirs64 = fibonacci_sphere(64)
    n_views = np.zeros(n, np.int32)
    dir_mask = np.zeros(n, np.uint64)
    footprint = np.full(n, np.inf, np.float32)
    dir_sum = np.zeros((n, 3))
    tex_idx, tex_val = [], []
    for k, f in enumerate(cap.frames):
        if frames_mask is not None and not frames_mask[k]:
            continue
        idx, uv = visible(pts, f, load_depth_m(f))
        if len(idx) == 0:
            continue
        vec = f.c2w[:3, 3] - pts[idx]
        dist = np.linalg.norm(vec, axis=1)
        d = vec / dist[:, None]
        bins = np.argmax(d @ dirs64.T, axis=1).astype(np.uint64)
        n_views[idx] += 1
        dir_mask[idx] |= np.left_shift(np.uint64(1), bins)
        footprint[idx] = np.minimum(footprint[idx], 1000.0 * dist / f.K[0, 0])
        dir_sum[idx] += d
        tex = texture_map(load_gray(f))
        ui = np.clip(np.rint(uv[:, 0]).astype(int), 0, f.width - 1)
        vi = np.clip(np.rint(uv[:, 1]).astype(int), 0, f.height - 1)
        tex_idx.append(idx)
        tex_val.append(tex[vi, ui].astype(np.float32))
    n_dirs = np.unpackbits(dir_mask.view(np.uint8)).reshape(-1, 64).sum(axis=1).astype(np.int32)
    norm = np.linalg.norm(dir_sum, axis=1)
    mean_dir = np.divide(dir_sum, norm[:, None], out=np.zeros_like(dir_sum), where=norm[:, None] > 0)
    texture = _median_by_group(np.concatenate(tex_idx) if tex_idx else np.zeros(0, int),
                               np.concatenate(tex_val) if tex_val else np.zeros(0, np.float32), n)
    return ViewStats(n_views=n_views, dir_mask=dir_mask, n_dirs=n_dirs,
                     best_footprint_mm=footprint, texture=texture, mean_view_dir=mean_dir)
