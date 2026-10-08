import numpy as np
import pytest
from roomrecon import scene


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _cams(R_world, heights, n=20):
    """Cameras looking horizontally around a room whose true up is R_world @ +Y."""
    c2w = np.zeros((n, 4, 4))
    c2w[:, 3, 3] = 1
    for i in range(n):
        yaw = 2 * np.pi * i / n
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        c2w[i, :3, :3] = R_world @ Ry
        c2w[i, :3, 3] = R_world @ np.array([np.cos(yaw), heights[i % len(heights)], np.sin(yaw)])
    return c2w


def _room(R_world, floor=0.0, n=4000, floaters=0, seed=0):
    rng = np.random.default_rng(seed)
    floor_pts = np.c_[rng.uniform(-2, 2, n), np.full(n, floor), rng.uniform(-2, 2, n)]
    wall_pts = np.c_[np.full(n // 4, 2.0), rng.uniform(floor, floor + 2.5, n // 4),
                     rng.uniform(-2, 2, n // 4)]
    junk = np.c_[rng.uniform(-2, 2, floaters), rng.uniform(floor - 3, floor - 1, floaters),
                 rng.uniform(-2, 2, floaters)]
    return (R_world @ np.vstack([floor_pts, wall_pts, junk]).T).T


def test_up_recovered_under_tilted_world():
    R = _rot_x(70)
    up = scene.estimate_up(_cams(R, [1.4, 1.5]))
    assert np.allclose(up, R @ [0, 1, 0], atol=1e-6)


def test_up_from_portrait_frames_in_sensor_orientation():
    # phone held upright, frames saved sideways (iPhone sensor orientation): camera -X is up
    R = _rot_x(70)
    c2w = _cams(R, [1.4, 1.5])
    roll = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])   # Rz(-90): camera x -> world down
    c2w[:, :3, :3] = c2w[:, :3, :3] @ roll
    up = scene.estimate_up(c2w)
    assert np.allclose(up, R @ [0, 1, 0], atol=1e-6)


def test_up_rejects_inconsistent_cameras():
    c2w = _cams(np.eye(3), [1.4])
    c2w[::2, :3, 1] *= -1
    with pytest.raises(ValueError):
        scene.estimate_up(c2w)


def test_rotation_maps_up_to_y():
    up = np.array([0.3, 0.2, -0.9])
    up /= np.linalg.norm(up)
    R = scene.rotation_to_y_up(up)
    assert np.allclose(R @ up, [0, 1, 0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)


def test_floor_ignores_floaters_below():
    y = np.r_[np.zeros(3000), np.random.default_rng(1).uniform(-3, -1, 300), np.full(500, 2.5)]
    assert abs(scene.floor_height(y, camera_y=np.array([1.4, 1.5]))) < 0.05


# splat-transform v3.9.0, measured on the PC 2026-10-03: `-r 30,40,50` on a PLY with points at
# the unit axes, written back to PLY. Rows = where (1,0,0), (0,1,0), (0,0,1) went, PLY frame.
MEASURED_ROWS_30_40_50 = [[0.49240386486053467, 0.5868240594863892, 0.6427876353263855],
                          [-0.45682600140571594, 0.802872359752655, -0.3830222189426422],
                          [-0.7408430576324463, -0.10504046082496643, 0.663413941860199]]


def test_euler_matches_measured_splat_transform_convention():
    F = scene.PLAYCANVAS_FROM_PLY
    R_internal = F @ np.array(MEASURED_ROWS_30_40_50).T @ F
    assert np.allclose(scene.euler_xyz_deg(R_internal), [30, 40, 50], atol=1e-4)


def test_to_playcanvas_flips_x_and_y():
    c2w = np.eye(4)[None].copy()
    c2w[0, :3, 3] = [1, 2, 3]
    c2w_pc, verts_pc = scene.to_playcanvas(c2w, np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(verts_pc, [[-1, -2, 3]])
    assert np.allclose(c2w_pc[0, :3, 3], [-1, -2, 3])
    assert np.allclose(c2w_pc[0, :3, 1], [0, -1, 0])    # camera up axis flipped with the world


def test_scene_frame_normalises_scale_and_spawns_inside():
    R = _rot_x(-35)
    c2w = _cams(R, [0.70, 0.75])          # half-scale scene: cameras ~0.725 above floor
    verts = _room(R, floor=0.0, floaters=200)
    f = scene.scene_frame(c2w, verts, eye_height_m=1.45)
    assert f["scale"] == pytest.approx(1.45 / 0.725, rel=0.05)
    assert f["floor_y"] == pytest.approx(0.0, abs=0.05)
    sp = np.array(f["spawn_position"])
    assert sp[1] == pytest.approx(f["floor_y"] + 1.45, abs=0.05)
    assert abs(sp[0]) < 2.0 * f["scale"] and abs(sp[2]) < 2.0 * f["scale"]
    assert abs(f["spawn_forward"][1]) < 1e-9 and np.isclose(np.linalg.norm(f["spawn_forward"]), 1)
    assert f["scale_source"] == "median camera height normalised to 1.45 m"
