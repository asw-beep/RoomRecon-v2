import numpy as np

from roomrecon.collision.glb import read_glb, sample_triangles
from tests.glbwrite import write_glb

QUAD_V = np.array([[0, 0.5, 0], [1, 0.5, 0], [1, 0.5, 2], [0, 0.5, 2]], float)   # 1 x 2 m at y = 0.5
QUAD_F = np.array([[0, 1, 2], [0, 2, 3]])


def test_glb_round_trip_uint32(tmp_path):
    write_glb(tmp_path / "q.glb", QUAD_V, QUAD_F)
    V, F = read_glb(tmp_path / "q.glb")
    assert np.allclose(V, QUAD_V) and np.array_equal(F, QUAD_F)


def test_glb_uint16_and_node_translation(tmp_path):
    write_glb(tmp_path / "q.glb", QUAD_V, QUAD_F, index_type=5123, translation=(1, 2, 3))
    V, F = read_glb(tmp_path / "q.glb")
    assert np.allclose(V, QUAD_V + [1, 2, 3]) and np.array_equal(F, QUAD_F)


def test_sample_triangles_count_and_plane():
    p = sample_triangles(QUAD_V, QUAD_F, spacing=0.01)
    assert abs(len(p) - 20000) <= 1
    assert np.allclose(p[:, 1], 0.5)
    assert p[:, 0].min() >= 0 and p[:, 0].max() <= 1 and p[:, 2].min() >= 0 and p[:, 2].max() <= 2
