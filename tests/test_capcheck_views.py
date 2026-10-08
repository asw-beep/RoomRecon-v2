import numpy as np
import pytest
from PIL import Image

from roomrecon.capcheck.capture import load_capture
from roomrecon.capcheck.surface import backproject, build_surface
from roomrecon.capcheck.views import accumulate, fibonacci_sphere, project
from tests.synthroom import cam_rotation, look_at, make_room
from tests.test_capcheck_surface import ring_cams

HI = dict(w=128, h=96, f=100.0)          # same field of view as the 64x48 default, denser


def nearest(points, target):
    return int(np.argmin(np.linalg.norm(points - np.asarray(target), axis=1)))


def test_fibonacci_sphere_is_unit_and_spread():
    d = fibonacci_sphere(64)
    assert d.shape == (64, 3) and np.allclose(np.linalg.norm(d, axis=1), 1)
    assert np.abs(d.mean(axis=0)).max() < 0.05


def test_project_roundtrips_backproject():
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = cam_rotation(30, -20), [1.0, 1.4, 2.0]
    depth = np.linspace(1.0, 3.0, 48 * 64, dtype=np.float32).reshape(48, 64)
    pts = backproject(depth, K, c2w, stride=1)
    idx, uv, z = project(pts, K, c2w, 64, 48)
    assert len(idx) == len(pts)
    v, u = np.divmod(np.arange(48 * 64), 64)
    assert np.allclose(uv[:, 0], u[idx], atol=1e-6) and np.allclose(uv[:, 1], v[idx], atol=1e-6)
    assert np.allclose(z, depth.reshape(-1)[idx], atol=1e-5)


def test_occluded_points_are_not_seen(tmp_path):
    pillar = ((2.0, 0.0, 1.3), (2.4, 2.5, 1.7))
    a, b = (np.array([1.0, 1.25, 1.5]), -90, 0), (np.array([1.0, 1.25, 0.4]), -90, 0)
    root = make_room(tmp_path, [a, b], pillar=pillar, **HI)
    surface = build_surface(load_capture(root))
    vs = accumulate(surface, load_capture(root, exclude=frozenset({"frame_00002"})))
    p = surface.points
    wall = (p[:, 0] > 3.95) & (p[:, 1] > 0.5) & (p[:, 1] < 2.0)
    behind = wall & (p[:, 2] > 1.3) & (p[:, 2] < 1.7)
    beside = wall & (((p[:, 2] > 0.3) & (p[:, 2] < 0.6)) | ((p[:, 2] > 2.4) & (p[:, 2] < 2.7)))
    assert behind.sum() > 20 and beside.sum() > 20
    assert np.all(vs.n_views[behind] == 0)
    assert np.all(vs.n_views[beside] == 1)


def test_n_views_counts_cameras(tmp_path):
    target = (2.0, 0.0, 1.5)
    cams = [look_at((2.0 + 1.2 * np.cos(t), 1.4, 1.5 + 1.0 * np.sin(t)), target)
            for t in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
    cap = load_capture(make_room(tmp_path, cams, **HI))
    surface = build_surface(cap)
    vs = accumulate(surface, cap)
    assert vs.n_views[nearest(surface.points, target)] == 6


def test_n_dirs_grows_with_spread(tmp_path):
    target = (4.0, 1.25, 1.5)
    same = [look_at((2.5, 1.25, 1.5), (4.0, 1.25 + dy, 1.5 + dz))
            for dy, dz in [(0, 0), (0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2), (0.1, 0.1)]]
    spread = [look_at((4.0 - 1.5 * np.cos(t), 1.25, 1.5 + 1.5 * np.sin(t)), target)
              for t in np.radians(np.linspace(-60, 60, 6))]
    for i, (cams, check) in enumerate([(same, lambda n: n <= 2), (spread, lambda n: n >= 3)]):
        cap = load_capture(make_room(tmp_path / f"room{i}", cams, **HI))
        surface = build_surface(cap)
        vs = accumulate(surface, cap)
        k = nearest(surface.points, target)
        assert vs.n_views[k] == 6 and check(vs.n_dirs[k])


def test_texture_high_on_checker_low_on_flat(tmp_path):
    cap = load_capture(make_room(tmp_path, ring_cams(), checker_wall="x+", **HI))
    surface = build_surface(cap)
    vs = accumulate(surface, cap)
    p = surface.points
    inner = (p[:, 1] > 0.3) & (p[:, 1] < 2.2) & (p[:, 2] > 0.3) & (p[:, 2] < 2.7)
    checker = inner & (p[:, 0] > 3.95) & np.isfinite(vs.texture)
    flat = inner & (p[:, 0] < 0.05) & np.isfinite(vs.texture)
    assert checker.sum() > 20 and flat.sum() > 20
    assert np.median(vs.texture[checker]) > 10 * np.median(vs.texture[flat]) + 1e-6


def test_all_zero_depth_frame_sees_nothing(tmp_path):
    root = make_room(tmp_path, ring_cams(n=3), **HI)
    cap = load_capture(root)
    surface = build_surface(cap)
    Image.fromarray(np.zeros((96, 128), np.uint16)).save(cap.frames[1].depth_path)
    vs = accumulate(surface, cap)
    without = accumulate(surface, load_capture(root, exclude=frozenset({"frame_00002"})))
    assert not np.isnan(vs.n_views.astype(float)).any()
    assert np.array_equal(vs.n_views, without.n_views)
    seen = vs.n_views > 0
    assert np.all(np.isfinite(vs.best_footprint_mm[seen])) and np.all(np.isinf(vs.best_footprint_mm[~seen]))
    assert np.all(np.isnan(vs.texture[~seen]))
