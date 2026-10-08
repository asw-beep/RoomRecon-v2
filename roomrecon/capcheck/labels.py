"""Per-point labels (why a surface will be poor), weak regions worth recapturing, floor the
phone never saw, and the capture's verdict.

Under-covered thresholds are fixed in advance: fewer than 5 views (Pix4D's "5+ images" rule)
or views from fewer than 3 of 64 direction bins. The texture threshold is calibrated once on
vr_room and tested unchanged on sejal-cap2 (spec §4.4, §7).
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay

from roomrecon.capcheck.tracking import Tracking

OK, UNDER, TEXTURELESS, BROKEN = 0, 1, 2, 3
NAMES = {OK: "ok", UNDER: "under-covered", TEXTURELESS: "textureless", BROKEN: "broken tracking"}
MIN_VIEWS, MIN_DIRS = 5, 3
TAU_TEX = 0.02            # provisional; replaced by the vr_room calibration (Task 8)


def label_points(n_views: np.ndarray, n_dirs: np.ndarray, texture: np.ndarray,
                 broken_pts: np.ndarray, tau_tex: float = TAU_TEX) -> np.ndarray:
    lab = np.full(len(n_views), OK, np.uint8)
    lab[np.isfinite(texture) & (texture < tau_tex)] = TEXTURELESS
    lab[(n_views < MIN_VIEWS) | (n_dirs < MIN_DIRS)] = UNDER
    lab[np.asarray(broken_pts, bool)] = BROKEN
    return lab


def _sector(d: np.ndarray) -> str:
    """Compass side (in the aligned frame, y up) that a mean view direction points to."""
    if np.hypot(d[0], d[2]) < 1e-6:
        return "above" if d[1] > 0 else ("below" if d[1] < 0 else "nowhere")
    names = ["+x", "+x+z", "+z", "-x+z", "-x", "-x-z", "-z", "+x-z"]
    return names[int(np.round(np.degrees(np.arctan2(d[2], d[0])) / 45)) % 8]


def regions(points: np.ndarray, labels: np.ndarray, mean_view_dir: np.ndarray, floor_y: float,
            voxel=0.03, link=0.10, min_area=0.25) -> list[dict]:
    """Connected under-covered areas of at least `min_area` m², largest first. Positions in the
    frame the points are given in (aligned: y up, metres)."""
    sel = np.flatnonzero(labels == UNDER)
    if len(sel) == 0:
        return []
    p = points[sel]
    # `link`-sized cells, 26-connected: a pairwise graph of points within `link` had tens of
    # millions of edges in cluttered real rooms and took minutes with scipy 1.13
    cells = np.floor((p - p.min(axis=0)) / link).astype(int)
    grid = np.zeros(cells.max(axis=0) + 1, bool)
    grid[tuple(cells.T)] = True
    lab, _ = ndimage.label(grid, structure=np.ones((3, 3, 3), bool))
    comp = lab[tuple(cells.T)]
    big = np.flatnonzero(np.bincount(comp) * voxel ** 2 >= min_area)
    out = []
    for c in big[big > 0]:
        m = comp == c
        area = float(m.sum() * voxel ** 2)
        q = p[m]
        out.append({"n_points": int(m.sum()), "area_m2": area, "centre": q.mean(0).tolist(),
                    "y_range": [float(q[:, 1].min() - floor_y), float(q[:, 1].max() - floor_y)],
                    "size": np.ptp(q, axis=0).tolist(),
                    "seen_from": _sector(mean_view_dir[sel][m].mean(0))})
    return sorted(out, key=lambda r: -r["area_m2"])


def floor_not_seen(points_aligned: np.ndarray, floor_y: float, cell=0.10, band=0.30) -> dict:
    """Floor cells inside the room outline with no point within `band` of the floor."""
    floor = points_aligned[np.abs(points_aligned[:, 1] - floor_y) <= band][:, [0, 2]]
    if len(floor) < 3:
        return {"area_m2": 0.0, "cells": []}
    try:
        hull = Delaunay(floor[ConvexHull(floor).vertices])
    except Exception:                       # all floor points on a line
        return {"area_m2": 0.0, "cells": []}
    lo = floor.min(0)
    nx, nz = np.ceil((floor.max(0) - lo) / cell).astype(int) + 1
    seen = np.zeros((nx, nz), bool)
    ij = np.floor((floor - lo) / cell).astype(int)
    seen[ij[:, 0], ij[:, 1]] = True
    gx, gz = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    centres = lo + (np.stack([gx, gz], -1).reshape(-1, 2) + 0.5) * cell
    inside = (hull.find_simplex(centres) >= 0).reshape(nx, nz)
    holes = inside & ~seen
    return {"area_m2": float(holes.sum() * cell ** 2),
            "cells": (lo + (np.argwhere(holes) + 0.5) * cell).tolist()}


def verdict(tracking: Tracking, regs: list[dict]) -> str:
    if tracking.segment_changes or tracking.jumps:
        return "BROKEN TRACKING"
    return "RECAPTURE THESE AREAS" if regs else "OK"
