import numpy as np
import pytest

from roomrecon.repair.register import icp_point_to_plane, register_piece, rot_about

UP = np.array([0.0, 1.0, 0.0])


def _face(origin, u, v, normal, spacing):
    a = np.arange(0, np.linalg.norm(u) + 1e-9, spacing)
    b = np.arange(0, np.linalg.norm(v) + 1e-9, spacing)
    A, B = np.meshgrid(a, b)
    pts = origin + A.reshape(-1, 1) * (u / np.linalg.norm(u)) + B.reshape(-1, 1) * (v / np.linalg.norm(v))
    return pts, np.tile(normal, (len(pts), 1))


def _box(lo, size, inward, spacing):
    lo, (sx, sy, sz) = np.asarray(lo, float), size
    x, y, z = np.eye(3)
    s = 1.0 if inward else -1.0
    faces = [(lo, y * sy, z * sz, s * x), (lo + x * sx, y * sy, z * sz, -s * x),
             (lo, x * sx, z * sz, s * y), (lo + y * sy, x * sx, z * sz, -s * y),
             (lo, x * sx, y * sy, s * z), (lo + z * sz, x * sx, y * sy, -s * z)]
    parts = [_face(o, u, v, n, spacing) for o, u, v, n in faces]
    return np.vstack([p for p, _ in parts]), np.vstack([n for _, n in parts])


def box_cloud(spacing=0.05):
    """A 4 x 2.5 x 3 m room (inward normals) with an off-centre block, so no rotation of the
    room maps it onto itself."""
    rp, rn = _box((0, 0, 0), (4.0, 2.5, 3.0), True, spacing)
    bp, bn = _box((0.6, 0.0, 0.5), (0.6, 1.0, 0.4), False, spacing)
    return np.vstack([rp, bp]), np.vstack([rn, bn])


def moved(pts, R, t):
    return pts @ R.T + t


def test_rot_about_is_rotation():
    R = rot_about(np.array([0, 0, 1.0]), 90)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R @ [1, 0, 0], [0, 1, 0])


def test_icp_refines_small_offset():
    dst, n = box_cloud()
    R, t = rot_about(UP, 3), np.array([0.05, 0, -0.04])
    src = moved(dst, R, t)
    T = icp_point_to_plane(src, dst, n, np.eye(4))
    assert np.abs(moved(src, T[:3, :3], T[:3, 3]) - dst).max() < 0.005


def test_register_recovers_large_yaw_and_shift():
    dst, n = box_cloud()
    R, t = rot_about(UP, 70), np.array([0.8, 0, -0.5])
    src = moved(dst, R, t)
    r = register_piece(src, dst, n, UP)
    T = np.asarray(r["T"])
    assert r["accepted"] and r["reason"] == "" and r["fitness"] > 0.9
    assert np.abs(moved(src, T[:3, :3], T[:3, 3]) - dst).max() < 0.01
    assert abs(abs(r["yaw_deg"]) - 70) < 0.5 and r["tilt_deg"] < 0.5


def test_register_rejects_no_overlap():
    dst, n = box_cloud()
    blob = np.random.default_rng(0).uniform(0, 2, (3000, 3)) + [20, 0, 20]     # fits no surface
    r = register_piece(blob, dst, n, UP)
    assert r["accepted"] is False and r["reason"] == "low fitness"


def test_register_rejects_tilted_solution():
    dst, n = box_cloud()
    R = rot_about(UP, 40) @ rot_about(np.array([1.0, 0, 0]), 10)
    src = moved(dst, R, np.array([0.3, 0.1, 0.2]))
    r = register_piece(src, dst, n, UP)
    assert r["accepted"] is False and r["reason"] == "tilt"
    assert r["tilt_deg"] == pytest.approx(10, abs=1.0)
