import json

import numpy as np
import pytest
from PIL import Image

from roomrecon.capcheck.capture import (load_capture, load_confidence, load_depth_m,
                                        read_exclude)
from tests.synthroom import make_room

CAMS = [(np.array([1.0, 1.4, 1.5]), 0, 0), (np.array([2.0, 1.4, 1.5]), 90, 0),
        (np.array([3.0, 1.4, 1.5]), 180, 0)]


def test_mushroom_reader_frames_and_intrinsics(tmp_path):
    cap = load_capture(make_room(tmp_path, CAMS))
    assert len(cap.frames) == 3 and cap.source == "mushroom" and cap.has_tracking is False
    f = cap.frames[0]
    assert f.K[0, 0] == 50 and f.c2w.shape == (4, 4) and (f.width, f.height) == (64, 48)
    assert f.meta == {} and f.conf_path is None


def test_polycam_reader_attaches_keyframe_meta(tmp_path):
    meta = [{"tracking_segment": 1}, {"tracking_segment": 1}, {"tracking_segment": 2}]
    cap = load_capture(make_room(tmp_path, CAMS, source="polycam", meta=meta))
    assert cap.source == "polycam" and cap.has_tracking is True
    assert cap.frames[2].meta["tracking_segment"] == 2
    assert cap.frames[0].meta["weakly_connected"] is False
    assert cap.frames[0].conf_path is not None


def test_exclude_drops_named_frames_after_matching(tmp_path):
    meta = [{"blur_score": 1.0}, {"blur_score": 2.0}, {"blur_score": 3.0}]
    cap = load_capture(make_room(tmp_path, CAMS, source="polycam", meta=meta),
                       exclude=frozenset({"frame_00002"}))
    assert [f.name for f in cap.frames] == ["frame_00001", "frame_00003"]
    assert cap.frames[1].meta["blur_score"] == 3.0


def test_frame_and_keyframe_count_mismatch_raises(tmp_path):
    root = make_room(tmp_path, CAMS, source="polycam")
    sorted((root / "keyframes" / "cameras").glob("*.json"))[1].unlink()
    with pytest.raises(ValueError) as e:
        load_capture(root)
    assert "3" in str(e.value) and "2" in str(e.value)


def test_depth_in_metres_and_missing_is_zero(tmp_path):
    root = make_room(tmp_path, CAMS)
    cap = load_capture(root)
    d = np.zeros((48, 64), np.uint16)
    d[0, 0] = 2500
    Image.fromarray(d).save(cap.frames[0].depth_path)
    m = load_depth_m(cap.frames[0])
    assert m.dtype == np.float32 and m.shape == (48, 64)
    assert m[0, 0] == pytest.approx(2.5) and m[1, 1] == 0


def test_confidence_resized_nearest(tmp_path):
    cap = load_capture(make_room(tmp_path, CAMS, source="polycam"))
    c = np.array([[0, 54, 255, 54], [255, 0, 54, 255], [54, 255, 0, 0]], np.uint8)
    Image.fromarray(c).save(cap.frames[0].conf_path)
    out = load_confidence(cap.frames[0])
    assert out.shape == (48, 64) and set(np.unique(out)) <= {0, 54, 255}
    assert load_confidence(load_capture(make_room(tmp_path / "m", CAMS)).frames[0]) is None


def test_read_exclude_ignores_blank_lines(tmp_path):
    p = tmp_path / "ex.txt"
    p.write_text("frame_00011\n\n  frame_00021  \n")
    assert read_exclude(p) == frozenset({"frame_00011", "frame_00021"})
