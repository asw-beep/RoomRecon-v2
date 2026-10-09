"""Top-down occupancy at body height and the collision accuracy measures (spec §3).

A cell is occupied when collision surface lies between 0.2 and 1.8 m above the floor in it: what
a walking person would bump into. Walkable = free cells 4-connected to the spawn.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


@dataclass
class Grid:
    lo: np.ndarray              # x, z of cell (0, 0)'s corner
    shape: tuple[int, int]
    cell: float


def make_grid(xz: np.ndarray, cell=0.05, margin=0.5) -> Grid:
    lo = xz.min(axis=0) - margin
    hi = xz.max(axis=0) + margin
    return Grid(lo=lo, shape=tuple(int(n) for n in np.ceil((hi - lo) / cell - 1e-9)), cell=cell)


def _cells(xz, grid):
    ij = np.floor((np.asarray(xz) - grid.lo) / grid.cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < grid.shape[0]) & (ij[:, 1] < grid.shape[1])
    return ij, ok


def occupancy(points, grid: Grid, floor_y: float, slab=(0.2, 1.8), scale=1.0) -> np.ndarray:
    occ = np.zeros(grid.shape, bool)
    p = np.asarray(points)
    if len(p) == 0:
        return occ
    y = p[:, 1] - floor_y
    p = p[(y >= slab[0] * scale) & (y <= slab[1] * scale)]
    ij, ok = _cells(p[:, [0, 2]], grid)
    occ[ij[ok, 0], ij[ok, 1]] = True
    return occ


def walkable(occ: np.ndarray, spawn_xz, grid: Grid) -> np.ndarray:
    """Free cells 4-connected to the spawn (or to the free cell nearest to it)."""
    free = ~occ
    if not free.any():
        return free
    ij, _ = _cells(np.atleast_2d(spawn_xz), grid)
    i, j = np.clip(ij[0], 0, np.array(grid.shape) - 1)
    if occ[i, j]:
        _, (ni, nj) = ndimage.distance_transform_edt(occ, return_indices=True)
        i, j = ni[i, j], nj[i, j]
    lab, _ = ndimage.label(free, structure=FOUR)
    return lab == lab[i, j]


def _dist_to(occ: np.ndarray, cell: float) -> np.ndarray:
    """Distance (same units as `cell`) from every cell to the nearest occupied one."""
    if not occ.any():
        return np.full(occ.shape, np.inf)
    return ndimage.distance_transform_edt(~occ) * cell


def compare(occ_src, occ_gt, occ_lidar, spawn_xz, grid: Grid, scale=1.0, tol=0.10, near=0.5) -> dict:
    a = grid.cell ** 2 / scale ** 2                       # m² per cell
    ws, wg = walkable(occ_src, spawn_xz, grid), walkable(occ_gt, spawn_xz, grid)
    union = (ws | wg).sum()
    d_gt = _dist_to(occ_gt, grid.cell)
    observed = occ_gt & (_dist_to(occ_lidar, grid.cell) <= tol * scale)
    missed = observed & (_dist_to(occ_src, grid.cell) > tol * scale)
    off = d_gt[occ_src & (d_gt <= near * scale)] / scale * 100
    return {"iou": float((ws & wg).sum() / union) if union else float("nan"),
            "phantom_m2": float((occ_src & wg & (d_gt > tol * scale)).sum() * a),
            "missed_pct": float(100 * missed.sum() / max(observed.sum(), 1)),
            "offset_median_cm": float(np.median(off)) if len(off) else float("nan"),
            "offset_p95_cm": float(np.percentile(off, 95)) if len(off) else float("nan"),
            "walkable_m2": float(ws.sum() * a)}
