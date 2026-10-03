import json
import pytest
import numpy as np
from roomrecon.scene_cli import build_scene, load_c2w, viewer_settings


def test_viewer_settings_start_at_spawn_looking_forward():
    s = {"spawn_position": [1.0, 1.45, -2.0], "spawn_forward": [0.0, 0.0, 1.0]}
    v = viewer_settings(s)
    cam = v["cameras"][0]["initial"]
    assert v["version"] == 2 and v["startMode"] == "default"
    assert cam["position"] == [1.0, 1.45, -2.0]
    assert cam["target"] == [1.0, 1.45, -1.0]       # 1 m ahead, same eye height
    assert 30 <= cam["fov"] <= 120


def test_build_scene_minus_y_up_ply_needs_no_rotation_in_playcanvas():
    # Cameras upright with world up = -Y in the PLY frame (INRIA/COLMAP style); room floor at y=+1.
    n = 12
    c2w = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        a = 2 * np.pi * i / n
        Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        c2w[i, :3, :3] = np.diag([1.0, -1.0, -1.0]) @ Ry     # camera +Y points to world -Y
        c2w[i, :3, 3] = [np.cos(a), -0.45, np.sin(a)]        # 1.45 above the floor at y=+1
    rng = np.random.default_rng(0)
    verts = np.c_[rng.uniform(-2, 2, 3000), np.full(3000, 1.0), rng.uniform(-2, 2, 3000)]
    s = build_scene(c2w, verts, room="r", run_id="x")
    assert np.allclose(s["euler_xyz_deg"], [0, 0, 0], atol=1e-6)
    assert s["frame"] == "playcanvas (PLY rotated 180 deg about Z)"
    assert s["scale"] == pytest.approx(1.0, rel=0.02)
    assert s["floor_y"] == pytest.approx(-1.0, abs=0.05)


def test_load_c2w_list_of_3x4(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"file_path": "a.png", "transform": np.eye(4)[:3].tolist()}] * 2))
    c2w = load_c2w(p)
    assert c2w.shape == (2, 4, 4) and np.allclose(c2w[0], np.eye(4))


def test_load_c2w_frames_dict_4x4(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"frames": [{"transform_matrix": np.eye(4).tolist()}]}))
    assert load_c2w(p).shape == (1, 4, 4)
