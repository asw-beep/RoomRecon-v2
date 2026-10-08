import json

import numpy as np
import pytest

from roomrecon.capcheck.__main__ import main
from roomrecon.capcheck.capture import load_capture
from roomrecon.capcheck.outputs import align, write_splat_ply
from roomrecon.capcheck.surface import build_surface
from roomrecon.scene_cli import build_scene
from tests.synthroom import make_room
from tests.test_capcheck_surface import ring_cams

HI = dict(w=128, h=96, f=100.0)
PROPS = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3"]


def test_splat_ply_header_and_count(tmp_path):
    frame = {"rotation": np.eye(3).tolist(), "scale": 1.0}
    p = tmp_path / "c.ply"
    write_splat_ply(p, np.random.default_rng(0).uniform(0, 1, (10, 3)), np.full((10, 3), 0.5), frame)
    raw = p.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode()
    assert "format binary_little_endian 1.0" in header and "element vertex 10" in header
    assert [l.split()[-1] for l in header.splitlines() if l.startswith("property")] == PROPS
    assert len(raw) == end + 10 * 14 * 4
    vals = np.frombuffer(raw[end:], "<f4").reshape(10, 14)
    assert np.allclose(vals[:, 3:6], 0) and np.allclose(vals[:, 10:14], [1, 0, 0, 0])
    assert np.allclose(vals[:, 7:10], np.log(0.015))


def test_align_puts_floor_at_floor_y(tmp_path):
    cap = load_capture(make_room(tmp_path, ring_cams(), **HI))
    s = build_surface(cap)
    frame = build_scene(np.stack([f.c2w for f in cap.frames]), s.points, room="t", run_id="t")
    a = align(s.points, frame)
    floor = s.points[:, 1] < 0.02
    assert np.percentile(a[floor, 1], 5) == pytest.approx(frame["floor_y"], abs=0.03)
    ceiling = s.points[:, 1] > 2.48
    assert np.median(a[ceiling, 1]) > np.median(a[floor, 1]) + 2.0        # upright after aligning


def test_cli_end_to_end_on_synthetic_room(tmp_path):
    room = make_room(tmp_path / "room", ring_cams(), source="polycam", **HI)
    out = tmp_path / "out"
    assert main([str(room), "--out", str(out), "--name", "synthetic"]) == 0
    for f in ["report.json", "report.txt", "coverage.ply", "topdown.png", "scene.json",
              "settings.json", "points.npz"]:
        assert (out / f).stat().st_size > 0, f
    rep = json.loads((out / "report.json").read_text())
    assert rep["verdict"] in ("OK", "RECAPTURE THESE AREAS", "BROKEN TRACKING")
    assert (out / "report.txt").read_text().startswith("Verdict: " + rep["verdict"])
    npz = np.load(out / "points.npz")
    assert {"points", "normals", "n_views", "n_dirs", "texture", "footprint", "labels",
            "broken_pts"} <= set(npz.files)
    assert len(npz["labels"]) == len(npz["points"]) == rep["surface_points"]
