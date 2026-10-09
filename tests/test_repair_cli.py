import json

import numpy as np
import pytest

from roomrecon.capcheck.capture import load_capture
from roomrecon.repair.__main__ import main
from roomrecon.repair.metrics import added_coverage, alignment_error
from roomrecon.repair.register import rot_about
from roomrecon.repair.variants import piece_cloud
from tests.synthroom import loop_cams, make_room

SEGMENTS = [1] * 24 + [2] * 2 + [3] * 14
HI = dict(w=256, h=192, f=200.0, pillar=((0.6, 0.0, 0.5), (1.2, 1.0, 0.9)))   # block breaks the room's symmetry


def names(a, b):
    return [f"frame_{i:05d}" for i in range(a, b + 1)]


def error_transform(roll=0.0):
    E = np.eye(4)
    E[:3, :3] = rot_about(np.array([0, 1.0, 0]), 70) @ rot_about(np.array([1.0, 0, 0]), roll)
    E[:3, 3] = [0.8, 0, -0.5]
    return E


def broken_room(dir, roll=0.0, segments=SEGMENTS, source="polycam"):
    """A full loop (frames 1-24), then a second loop round the same walls whose frames 27-40 were
    recorded in a wrong frame (pose = E @ true pose), like cap3's two overlapping sweeps."""
    meta = [{"tracking_segment": s, "timestamp": 1_000_000 * (i + 1)} for i, s in enumerate(segments)]
    cams = loop_cams(24) + loop_cams(16) if segments is SEGMENTS else loop_cams(len(segments))
    root = make_room(dir, cams, source=source, meta=meta, **HI)
    if segments is SEGMENTS:
        name = "transforms.json" if source == "polycam" else "transformations.json"
        doc = json.loads((root / name).read_text())
        E = error_transform(roll)
        for f in doc["frames"][26:]:
            f["transform_matrix"] = (E @ np.array(f["transform_matrix"])).tolist()
        (root / name).write_text(json.dumps(doc))
    return root


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    return broken_room(tmp_path_factory.mktemp("room"))


def test_alignment_error_drops_after_repair(room):
    cap = load_capture(room)
    ref = piece_cloud(cap, names(1, 24))
    assert alignment_error(cap, names(27, 40), ref) > 0.3
    # spec C1 bound; the 5 cm voxel reference alone limits this measure to ~1.3 cm (reference vs itself)
    assert alignment_error(cap, names(27, 40), ref, T=np.linalg.inv(error_transform())) < 0.03


def test_added_coverage_positive_when_piece_sees_more(tmp_path):
    cap = load_capture(make_room(tmp_path, loop_cams(40), **HI))      # true poses throughout
    assert added_coverage(cap, names(1, 24), names(27, 40)) > 0.02    # the loop's last part faces other walls
    again = load_capture(make_room(tmp_path / "again", loop_cams(40)[:24] + loop_cams(40)[4:16], **HI))
    assert added_coverage(again, names(1, 24), names(25, 36)) < 0.02  # a piece that repeats reference views


def test_cli_end_to_end(room, tmp_path):
    out = tmp_path / "out"
    assert main([str(room), "--out", str(out), "--min-frames", "10"]) == 0
    rep = json.loads((out / "repair.json").read_text())
    (piece,) = rep["pieces"]
    assert piece["accepted"] and piece["frames"] == ["frame_00027", "frame_00040", 14]
    assert piece["c1_after_m"] < 0.03 < 0.3 < piece["c1_before_m"] and rep["criteria"]["C1"] is True
    assert rep["reference"] == ["frame_00001", "frame_00024", 24] and rep["dropped"] == names(25, 26)
    assert rep["variants"] == {"raw": 40, "drop": 24, "repaired": 38}
    for v in ("raw", "drop", "repaired"):
        assert (out / v / "transforms.json").exists()


def test_cli_single_piece_nothing_to_repair(tmp_path, capsys):
    root = broken_room(tmp_path / "room", segments=[1] * 20)
    assert main([str(root), "--out", str(tmp_path / "out"), "--min-frames", "10"]) == 0
    assert "nothing to repair" in capsys.readouterr().out
    rep = json.loads((tmp_path / "out" / "repair.json").read_text())
    assert rep["pieces"] == [] and rep["variants"] == {"raw": 20, "drop": 20, "repaired": 20}


def test_cli_rejects_capture_without_tracking(tmp_path):
    root = make_room(tmp_path / "room", loop_cams(12))                  # MuSHRoom layout: no tracking data
    with pytest.raises(SystemExit) as e:
        main([str(root), "--out", str(tmp_path / "out")])
    assert "no tracking data" in str(e.value)


def test_cli_reports_rejected_piece(tmp_path):
    root = broken_room(tmp_path / "room", roll=10.0)
    assert main([str(root), "--out", str(tmp_path / "out"), "--min-frames", "10"]) == 0
    rep = json.loads((tmp_path / "out" / "repair.json").read_text())
    assert rep["pieces"][0]["accepted"] is False and rep["pieces"][0]["reason"] in ("tilt", "low fitness")
    assert rep["variants"]["repaired"] == 24 and rep["criteria"]["C1"] is False
