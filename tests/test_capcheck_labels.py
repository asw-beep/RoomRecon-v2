import numpy as np
import pytest

from roomrecon.capcheck.labels import (BROKEN, OK, TEXTURELESS, UNDER, floor_not_seen,
                                       label_points, regions, verdict)
from roomrecon.capcheck.tracking import Tracking


def track(changes=(), jumps=()):
    return Tracking(True, [], list(changes), list(jumps), [], [], [], frozenset(), None, None)


def test_label_priority():
    n_views = np.array([1, 1, 20, 20])
    n_dirs = np.array([1, 1, 6, 6])
    texture = np.array([0.0, 0.0, 0.0, 0.5])
    broken = np.array([True, False, False, False])
    assert list(label_points(n_views, n_dirs, texture, broken, tau_tex=0.02)) == [BROKEN, UNDER, TEXTURELESS, OK]


def test_narrow_directions_are_under_covered():
    lab = label_points(np.array([30]), np.array([2]), np.array([0.5]), np.array([False]), 0.02)
    assert lab[0] == UNDER


def test_unseen_point_is_under_not_textureless():
    lab = label_points(np.array([0]), np.array([0]), np.array([np.nan]), np.array([False]), 0.02)
    assert lab[0] == UNDER


def test_regions_drop_small_patches():
    g = np.arange(20) * 0.03
    patch = np.stack(np.meshgrid(g, [1.0], g), -1).reshape(-1, 3)            # 400 points, one plane
    rng = np.random.default_rng(0)
    lonely = np.c_[rng.uniform(3, 30, 50), np.full(50, 1.0), np.arange(50) * 1.0]  # >= 1 m apart
    pts = np.vstack([patch, lonely])
    labels = np.full(len(pts), UNDER, np.uint8)
    view = np.tile([1.0, 0.0, 0.0], (len(pts), 1))
    regs = regions(pts, labels, view, floor_y=0.0)
    assert len(regs) == 1
    r = regs[0]
    assert r["n_points"] == 400 and r["area_m2"] == pytest.approx(0.36, rel=1e-6)
    assert r["y_range"] == pytest.approx([1.0, 1.0]) and r["seen_from"] == "+x"


def test_ok_points_make_no_region():
    g = np.arange(20) * 0.03
    pts = np.stack(np.meshgrid(g, [1.0], g), -1).reshape(-1, 3)
    assert regions(pts, np.full(len(pts), OK, np.uint8), np.zeros_like(pts), floor_y=0.0) == []


def test_floor_not_seen_finds_a_hole():
    g = np.arange(0, 3.0, 0.03)
    xz = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    hole = (xz[:, 0] >= 1.0) & (xz[:, 0] < 2.0) & (xz[:, 1] >= 1.0) & (xz[:, 1] < 2.0)
    xz = xz[~hole]
    pts = np.c_[xz[:, 0], np.zeros(len(xz)), xz[:, 1]]
    walls = np.c_[np.zeros(100), np.linspace(0.5, 2.5, 100), np.linspace(0, 3, 100)]   # not floor
    out = floor_not_seen(np.vstack([pts, walls]), floor_y=0.0)
    assert out["area_m2"] == pytest.approx(1.0, abs=0.2)


def test_verdict_rules():
    reg = [{"area_m2": 0.5}]
    assert verdict(track(changes=["frame_00009"]), reg) == "BROKEN TRACKING"
    assert verdict(track(jumps=[{"frame": "frame_00002"}]), []) == "BROKEN TRACKING"
    assert verdict(track(), reg) == "RECAPTURE THESE AREAS"
    assert verdict(track(), []) == "OK"


def test_regions_far_apart_stay_separate():
    g = np.arange(20) * 0.03
    patch = np.stack(np.meshgrid(g, [1.0], g), -1).reshape(-1, 3)
    pts = np.vstack([patch, patch + [1.2, 0, 0]])                          # 0.6 m gap
    regs = regions(pts, np.full(len(pts), UNDER, np.uint8), np.zeros_like(pts), floor_y=0.0)
    assert [r["n_points"] for r in regs] == [400, 400]
