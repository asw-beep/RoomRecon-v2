import numpy as np
import pytest

from roomrecon.collision.grid import Grid, compare, make_grid, occupancy, walkable

GRID = Grid(lo=np.array([-0.5, -0.5]), shape=(90, 70), cell=0.05)      # x -0.5..4.0, z -0.5..3.0
SPAWN = (1.75, 1.25)


def ring(lo=10, hi_i=80, hi_j=60):
    occ = np.zeros(GRID.shape, bool)
    occ[lo, lo:hi_j + 1] = occ[hi_i, lo:hi_j + 1] = True
    occ[lo:hi_i + 1, lo] = occ[lo:hi_i + 1, hi_j] = True
    return occ                                                          # 3.5 x 2.5 m room, 240 wall cells


def test_identical_source_is_perfect():
    gt = ring()
    m = compare(gt, gt, gt, SPAWN, GRID)
    assert m["iou"] == 1.0 and m["phantom_m2"] == 0 and m["missed_pct"] == 0 and m["offset_median_cm"] == 0


def test_phantom_block_area():
    gt = ring()
    src = gt.copy()
    src[30:40, 30:40] = True                                            # 0.5 x 0.5 m mid-room
    m = compare(src, gt, gt, SPAWN, GRID)
    assert m["phantom_m2"] == pytest.approx(0.25, abs=0.0025) and m["iou"] < 1


def test_missing_wall_segment():
    gt = ring()
    src = gt.copy()
    src[10, 20:40] = False                                              # 1 m gap in the x = 0 wall
    m = compare(src, gt, gt, SPAWN, GRID)
    # the 2 cells at each end of the gap are within the 10 cm tolerance of the remaining wall
    assert m["missed_pct"] == pytest.approx(100 * 16 / 240, abs=0.01)
    assert m["walkable_m2"] > compare(gt, gt, gt, SPAWN, GRID)["walkable_m2"]


def test_walls_shifted_inward():
    gt = ring()
    src = ring(lo=12, hi_i=78, hi_j=58)                                 # every wall 0.10 m inward
    m = compare(src, gt, gt, SPAWN, GRID)
    # 3 GT cells per corner are 11-14 cm (diagonal) from the shifted walls: 12 of 240 missed
    assert m["offset_median_cm"] == pytest.approx(10, abs=1) and m["missed_pct"] == pytest.approx(100 * 12 / 240)


def test_spawn_in_occupied_cell_uses_nearest_free():
    occ = ring()
    occ[30:40, 30:40] = True
    inside_block = walkable(occ, (-0.5 + 35 * 0.05, -0.5 + 35 * 0.05), GRID)
    elsewhere = walkable(occ, SPAWN if not occ[45, 35] else (0.75, 0.75), GRID)
    assert inside_block.sum() > 0 and np.array_equal(inside_block, elsewhere)


def test_far_points_ignored():
    pts = np.array([[1.0, 1.0, 1.0], [50.0, 1.0, 50.0]])
    near_only = occupancy(pts[:1], GRID, floor_y=0.0)
    assert np.array_equal(occupancy(pts, GRID, floor_y=0.0), near_only) and near_only.sum() == 1


def test_empty_slab_all_free():
    pts = np.array([[1.0, 3.0, 1.0], [2.0, 3.5, 2.0]])                 # above the 1.8 m body slab
    occ = occupancy(pts, GRID, floor_y=0.0)
    assert not occ.any()
    m = compare(occ, ring(), ring(), SPAWN, GRID)
    assert np.isfinite(m["iou"])


def test_make_grid_bounds():
    g = make_grid(np.array([[0.0, 0.0], [2.0, 1.0]]), cell=0.05, margin=0.5)
    assert np.allclose(g.lo, [-0.5, -0.5]) and g.shape == (60, 40)
