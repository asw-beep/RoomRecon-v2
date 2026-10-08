import json
import shutil

import numpy as np
import pytest

from roomrecon.capcheck.capture import load_capture
from roomrecon.capcheck.tracking import check_tracking
from tests.synthroom import make_room


def walk(n, jump_at=None):
    """Cameras stepping 0.05 m along x (0.2 s apart in synthroom); +2 m from frame `jump_at`."""
    return [(np.array([1.0 + 0.05 * i + (2.0 if jump_at and i + 1 >= jump_at else 0.0), 1.4, 1.5]), 0, 0)
            for i in range(n)]


def names(*idx):
    return {f"frame_{i:05d}" for i in idx}


def test_segment_change_and_shorter_segment_broken(tmp_path):
    meta = [{"tracking_segment": 1}] * 8 + [{"tracking_segment": 2}] * 3
    t = check_tracking(load_capture(make_room(tmp_path, walk(11), source="polycam", meta=meta)))
    assert t.available and t.segments == [1] * 8 + [2] * 3
    assert t.segment_changes == ["frame_00009"]
    assert t.broken == names(9, 10, 11) and t.jumps == []


def test_pose_jump_flagged(tmp_path):
    t = check_tracking(load_capture(make_room(tmp_path, walk(10, jump_at=8), source="polycam")))
    assert [j["frame"] for j in t.jumps] == ["frame_00008"]
    assert t.jumps[0]["speed_mps"] == pytest.approx(10.0, rel=0.05)
    assert t.broken == names(8, 9, 10)


def test_limited_and_weak_frames_listed(tmp_path):
    meta = [{}, {}, {"world_mapping_status": "limited"}, {}]
    root = make_room(tmp_path, walk(4), source="polycam", meta=meta)
    weak = sorted((root / "keyframes" / "corrected_cameras").glob("*.json"))[1]
    weak.write_text(json.dumps({"weakly_connected": True}))
    t = check_tracking(load_capture(root))
    assert t.limited == ["frame_00003"] and t.weakly_connected == ["frame_00002"]
    assert t.broken == frozenset() and len(t.blur_score) == 4 and len(t.angular_velocity) == 4


def test_mushroom_capture_has_no_tracking(tmp_path):
    t = check_tracking(load_capture(make_room(tmp_path, walk(3))))
    assert t.available is False and t.jumps == [] and t.broken == frozenset()
    assert t.blur_score is None


def test_missing_corrected_cameras_is_partial(tmp_path):
    root = make_room(tmp_path, walk(3), source="polycam")
    shutil.rmtree(root / "keyframes" / "corrected_cameras")
    t = check_tracking(load_capture(root))
    assert t.available is True and t.weakly_connected == []


def test_repeated_timestamp_is_flagged_not_divided(tmp_path):
    meta = [{"timestamp": 1_000_000}, {"timestamp": 1_200_000}, {"timestamp": 1_200_001},
            {"timestamp": 1_400_000}]
    root = make_room(tmp_path, walk(4), source="polycam", meta=meta)
    cams = sorted((root / "keyframes" / "cameras").glob("*.json"))
    m = json.loads(cams[2].read_text())
    m["timestamp"] = 1_200_000                         # same time as the frame before
    cams[2].write_text(json.dumps(m))
    t = check_tracking(load_capture(root))
    assert t.bad_timestamps == ["frame_00003"] and t.jumps == []
