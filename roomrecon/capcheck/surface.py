"""Surface points of the room from the phone's LiDAR depth: every frame back-projected,
merged into a voxel grid, with normals facing the cameras that saw them."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from roomrecon.capcheck.capture import Capture, load_confidence, load_depth_m


@dataclass
class Surface:
    points: np.ndarray
    normals: np.ndarray
    voxel: float


def backproject(depth_m: np.ndarray, K: np.ndarray, c2w: np.ndarray, stride: int = 4,
                mask: np.ndarray | None = None, min_d=0.1, max_d=5.0) -> np.ndarray:
    """World points of every `stride`-th pixel with a usable depth, in row-major pixel order.
    OpenGL camera: x right, y up, looking down -z; depth is z-depth in metres."""
    vs, us = np.arange(0, depth_m.shape[0], stride), np.arange(0, depth_m.shape[1], stride)
    v, u = np.meshgrid(vs, us, indexing="ij")
    d = depth_m[np.ix_(vs, us)]
    ok = (d >= min_d) & (d <= max_d)
    if mask is not None:
        ok &= mask[np.ix_(vs, us)]
    u, v, d = u[ok], v[ok], d[ok]
    cam = np.stack([(u - K[0, 2]) * d / K[0, 0], -(v - K[1, 2]) * d / K[1, 1], -d], axis=1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def voxel_merge(points: np.ndarray, cams: np.ndarray, voxel: float):
    """(centroid, mean camera position) per occupied voxel. One int64 key per point and
    bincount sums: numpy 1.26's unique(axis=0) and add.at took minutes on 10M points."""
    k = np.floor(points / voxel).astype(np.int64)
    k -= k.min(axis=0)
    dims = k.max(axis=0) + 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    counts = np.bincount(inv)
    centres = np.stack([np.bincount(inv, points[:, i]) for i in range(3)], axis=1) / counts[:, None]
    cam_mean = np.stack([np.bincount(inv, cams[:, i]) for i in range(3)], axis=1) / counts[:, None]
    return centres, cam_mean


def _normals(points: np.ndarray, toward: np.ndarray, k: int = 16, chunk: int = 200_000) -> np.ndarray:
    """Smallest-variance direction of the k nearest neighbours, flipped to face `toward`."""
    k = min(k, len(points))
    _, idx = cKDTree(points).query(points, k=k, workers=-1)
    idx = idx.reshape(len(points), k)
    out = np.empty_like(points)
    for s in range(0, len(points), chunk):
        nb = points[idx[s:s + chunk]]
        nb = nb - nb.mean(axis=1, keepdims=True)
        _, vecs = np.linalg.eigh(np.matmul(nb.transpose(0, 2, 1), nb))
        out[s:s + chunk] = vecs[:, :, 0]
    flip = np.einsum("ni,ni->n", out, toward - points) < 0
    out[flip] *= -1
    return out


def build_surface(cap: Capture, voxel: float = 0.03, stride: int = 4) -> Surface:
    pts, cams = [], []
    for f in cap.frames:
        conf = load_confidence(f)
        p = backproject(load_depth_m(f), f.K, f.c2w, stride,
                        mask=None if conf is None else conf == 255)
        pts.append(p.astype(np.float32))
        cams.append(np.broadcast_to(f.c2w[:3, 3].astype(np.float32), p.shape))
    if not pts or sum(len(p) for p in pts) == 0:
        return Surface(np.zeros((0, 3)), np.zeros((0, 3)), voxel)
    p, c = np.concatenate(pts).astype(np.float64), np.concatenate(cams).astype(np.float64)
    centres, cam_mean = voxel_merge(p, c, voxel)
    return Surface(points=centres, normals=_normals(centres, cam_mean), voxel=voxel)
