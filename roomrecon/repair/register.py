"""Rigid alignment of a broken tracking piece onto the reference piece, from LiDAR clouds.

ARKit keeps every tracking session level with gravity, so two pieces of one capture differ by a
turn about the vertical plus a shift. We start from 36 turns about the vertical (each with the
shift that matches the clouds' centroids), refine each with point-to-plane ICP, and keep the start
that fits best. A solution that tilts away from the vertical is rejected: ARKit cannot produce it.
"""
import numpy as np
from scipy.spatial import cKDTree

SAMPLE = 5000            # source points used during the 36-start search (all points in the final refine)


def rot_about(axis: np.ndarray, deg: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = np.radians(deg)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def icp_point_to_plane(src, dst, dst_normals, T0, dists=(0.5, 0.2, 0.05), iters=30, tree=None) -> np.ndarray:
    tree = tree or cKDTree(dst)
    T = np.array(T0, float)
    for d in dists:
        for _ in range(iters):
            p = _apply(T, src)
            dd, j = tree.query(p, distance_upper_bound=d)
            m = np.isfinite(dd)
            if m.sum() < 6:
                break
            p, q, n = p[m], dst[j[m]], dst_normals[j[m]]
            A = np.c_[np.cross(p, n), n]
            b = np.einsum("ij,ij->i", q - p, n)
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            ang = np.linalg.norm(x[:3])
            dT = np.eye(4)
            if ang > 1e-12:
                dT[:3, :3] = rot_about(x[:3] / ang, np.degrees(ang))
            dT[:3, 3] = x[3:]
            T = dT @ T
            if np.linalg.norm(x) < 1e-6:
                break
    return T


def score(src, dst, T, inlier=0.05, tree=None) -> tuple[float, float]:
    """(share of source points within `inlier` of the destination, RMSE of those points)."""
    tree = tree or cKDTree(dst)
    dd, _ = tree.query(_apply(T, src))
    inl = dd < inlier
    rmse = float(np.sqrt(np.mean(dd[inl] ** 2))) if inl.any() else float("inf")
    return float(inl.mean()), rmse


def _yaw_deg(R: np.ndarray, up: np.ndarray) -> float:
    v = np.cross(up, [1.0, 0, 0] if abs(up[0]) < 0.9 else [0, 0, 1.0])
    v /= np.linalg.norm(v)
    w = R @ v
    w -= up * (w @ up)
    return float(np.degrees(np.arctan2(np.cross(v, w) @ up, v @ w)))


def register_piece(src_pts, dst_pts, dst_normals, up, n_starts=36, min_fitness=0.3, max_tilt_deg=3.0) -> dict:
    up = np.asarray(up, float) / np.linalg.norm(up)
    src, dst = np.asarray(src_pts, float), np.asarray(dst_pts, float)
    tree = cKDTree(dst)
    rng = np.random.default_rng(0)
    sample = src if len(src) <= SAMPLE else src[rng.choice(len(src), SAMPLE, replace=False)]
    cs, cd = sample.mean(axis=0), dst.mean(axis=0)
    best = None
    for k in range(n_starts):
        T0 = np.eye(4)
        T0[:3, :3] = rot_about(up, 360.0 * k / n_starts)
        T0[:3, 3] = cd - T0[:3, :3] @ cs
        T = icp_point_to_plane(sample, dst, dst_normals, T0, tree=tree)
        f, r = score(sample, dst, T, tree=tree)
        if best is None or f > best[1] + 1e-9 or (abs(f - best[1]) <= 1e-9 and r < best[2]):
            best = (T, f, r)
    T = icp_point_to_plane(src, dst, dst_normals, best[0], dists=(0.1, 0.05), tree=tree)
    fitness, rmse = score(src, dst, T, tree=tree)
    R = T[:3, :3]
    tilt = float(np.degrees(np.arccos(np.clip((R @ up) @ up, -1.0, 1.0))))
    reason = "low fitness" if fitness < min_fitness else ("tilt" if tilt >= max_tilt_deg else "")
    return {"T": T.tolist(), "fitness": fitness, "rmse": rmse, "yaw_deg": _yaw_deg(R, up),
            "tilt_deg": tilt, "accepted": reason == "", "reason": reason}
