import json

import numpy as np

from roomrecon.collision.__main__ import main
from roomrecon.collision.prep import points_to_splats
from roomrecon.scene import PLAYCANVAS_FROM_PLY
from tests.glbwrite import write_glb


def box_sides(lo, hi, top=False):
    """Triangles of the 4 vertical sides (and optionally the top) of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
                  [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], float)
    F = [[0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]]
    if top:
        F += [[4, 5, 6], [4, 6, 7]]
    return V, np.array(F)


def merge(*meshes):
    V, F, base = [], [], 0
    for v, f in meshes:
        V.append(v)
        F.append(f + base)
        base += len(v)
    return np.vstack(V), np.vstack(F)


def test_points_to_splats_header(tmp_path):
    p = np.random.default_rng(0).uniform(0, 1, (10, 3)).astype(np.float32)
    np.savez(tmp_path / "p.npz", points=p)
    points_to_splats(tmp_path / "p.npz", tmp_path / "p.ply")
    raw = (tmp_path / "p.ply").read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    assert b"element vertex 10" in raw[:end]
    assert np.allclose(np.frombuffer(raw[end:], "<f4").reshape(10, 14)[:, :3], p, atol=1e-6)


def test_cli_end_to_end(tmp_path):
    room = box_sides((0, 0, 0), (3.5, 2.5, 2.5))
    block = box_sides((1.5, 0, 1.0), (2.0, 1.0, 1.5), top=True)          # 0.5 x 0.5 m mid-room
    for name, mesh in {"GT": room, "M": room, "L": room, "S": merge(room, block)}.items():
        write_glb(tmp_path / f"{name}.glb", *mesh)
    g = np.linspace(0, 1, 60)
    floor = np.array([[x * 3.5, 0.0, z * 2.5] for x in g for z in g])
    walls = np.array([[x * 3.5, y * 2.5, 0.0] for x in g for y in g] + [[0.0, y * 2.5, z * 2.5] for z in g for y in g])
    np.savez(tmp_path / "gt.npz", points=np.vstack([floor, walls]))
    np.savez(tmp_path / "lidar.npz", points=np.vstack([floor, walls]))
    scene = {"rotation": PLAYCANVAS_FROM_PLY.tolist(), "scale": 1.0, "spawn_position": [0.75, 1.45, 0.75]}
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    out = tmp_path / "out"
    args = [a for n in ("S", "M", "L", "GT") for a in ("--glb", f"{n}={tmp_path / (n + '.glb')}")]
    assert main(args + ["--gt-points", str(tmp_path / "gt.npz"), "--lidar-points", str(tmp_path / "lidar.npz"),
                        "--scene", str(tmp_path / "scene.json"), "--out", str(out)]) == 0
    res = json.loads((out / "metrics.json").read_text())
    assert res["sources"]["M"]["phantom_m2"] == 0 and res["sources"]["S"]["phantom_m2"] > 0.2
    assert res["criteria"]["M_vs_S"]["P2"] is True and res["criteria"]["M_vs_S"]["P1"] is True
    assert set(res["criteria"]["L_vs_S"]) == {"P1", "P2", "P3", "P4"}
    assert (out / "topdown.png").stat().st_size > 0
