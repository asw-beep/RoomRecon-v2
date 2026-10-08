import numpy as np
import pytest

from roomrecon.capcheck.capture import load_capture
from roomrecon.capcheck.surface import backproject, build_surface
from tests.synthroom import make_room

SIZE = (4.0, 2.5, 3.0)


def ring_cams(n=8, centre=(2.0, 1.4, 1.5), pitches=(-30, 0, 30)):
    return [(np.array(centre) + 0.3 * np.array([np.cos(a), 0, np.sin(a)]), np.degrees(a),
             pitches[i % len(pitches)])
            for i, a in enumerate(np.linspace(0, 2 * np.pi, n, endpoint=False))]


@pytest.fixture(scope="module")
def surface(tmp_path_factory):
    # 256x192 at f=200 (same field of view as the 64x48 default): at 64x48 the far walls are
    # sampled ~0.36 m apart, so 16-neighbour normals reach across walls; real captures are denser
    root = make_room(tmp_path_factory.mktemp("room"), ring_cams(), size=SIZE, w=256, h=192, f=200.0)
    return build_surface(load_capture(root))


def test_backproject_matches_the_projection_convention():
    K = np.array([[50.0, 0, 32], [0, 50.0, 24], [0, 0, 1]])
    pts = backproject(np.full((48, 64), 2.0, np.float32), K, np.eye(4), stride=1)
    assert pts.shape == (48 * 64, 3)
    assert pts[24 * 64 + 32] == pytest.approx([0, 0, -2])
    assert pts[24 * 64 + 42][0] == pytest.approx(0.4)
    assert pts[14 * 64 + 32][1] == pytest.approx(0.4)          # rows above the centre are +y


def test_surface_points_lie_on_the_box_walls(surface):
    p = surface.points
    d = np.min(np.abs(np.concatenate([p, p - np.array(SIZE)], axis=1)), axis=1)
    assert len(p) > 1000 and d.max() < 0.02


def test_normals_face_into_the_room(surface):
    p, n = surface.points, surface.normals
    wall = (p[:, 0] < 0.05) & (p[:, 1] > 0.2) & (p[:, 1] < 2.3) & (p[:, 2] > 0.2) & (p[:, 2] < 2.8)
    assert wall.sum() > 50
    assert np.all(n[wall, 0] > 0.9)
    assert np.allclose(np.linalg.norm(n, axis=1), 1, atol=1e-5)


def test_voxel_merge_one_point_per_voxel(surface):
    keys = np.floor(surface.points / surface.voxel).astype(np.int64)
    assert len(np.unique(keys, axis=0)) == len(surface.points)
