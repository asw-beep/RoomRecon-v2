import json

import numpy as np
import pytest
from PIL import Image

from roomrecon.capcheck.capture import load_capture
from roomrecon.capcheck.tracking import check_tracking
from roomrecon.repair.register import rot_about
from roomrecon.repair.variants import held_out_names, split_pieces, write_variants
from tests.synthroom import loop_cams, make_room

SEGMENTS = [1] * 24 + [2] * 2 + [3] * 14


def names(a, b):
    return [f"frame_{i:05d}" for i in range(a, b + 1)]


@pytest.fixture(scope="module")
def room(tmp_path_factory):
    # 1 s apart: loop_cams alternates pitch by up to 40 deg, which at 0.2 s would read as a pose jump
    meta = [{"tracking_segment": s, "timestamp": 1_000_000 * (i + 1)} for i, s in enumerate(SEGMENTS)]
    return make_room(tmp_path_factory.mktemp("room"), loop_cams(40), source="polycam", meta=meta)


def test_split_pieces_reference_candidates_dropped(room):
    cap = load_capture(room)
    ref, cands, dropped = split_pieces(cap, check_tracking(cap), min_frames=10)
    assert ref == names(1, 24) and cands == [names(27, 40)] and dropped == names(25, 26)


def test_test_names_every_8th_of_reference():
    assert held_out_names(names(1, 24)) == ["frame_00001", "frame_00009", "frame_00017"]


def test_write_variants_share_test_split_and_correct_poses(room, tmp_path):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot_about(np.array([0, 1.0, 0]), 30), [0.2, 0, -0.1]
    out = write_variants(room, tmp_path / "v", names(1, 24), {n: T for n in names(27, 40)}, names(25, 26))
    docs = {k: json.loads((p / "transforms.json").read_text()) for k, p in out.items()}
    src = {f["file_path"]: np.array(f["transform_matrix"]) for f in json.loads((room / "transforms.json").read_text())["frames"]}
    assert {k: len(d["frames"]) for k, d in docs.items()} == {"raw": 40, "drop": 24, "repaired": 38}
    tests = docs["raw"]["test_filenames"]
    assert len(tests) == 3 and all(d["test_filenames"] == tests == d["val_filenames"] for d in docs.values())
    for d in docs.values():
        fps = [f["file_path"] for f in d["frames"]]
        assert sorted(d["train_filenames"] + d["test_filenames"]) == sorted(fps) and "ply_file_path" not in d
    rep = {f["file_path"]: np.array(f["transform_matrix"]) for f in docs["repaired"]["frames"]}
    moved = "./images/frame_00030.png"
    assert np.allclose(rep[moved], T @ src[moved], atol=1e-9)
    assert np.allclose(rep["./images/frame_00005.png"], src["./images/frame_00005.png"])
    assert np.allclose(np.array([f["transform_matrix"] for f in docs["raw"]["frames"]]),
                       np.array(list(src.values())))


def test_variant_images_resolve(room, tmp_path):
    out = write_variants(room, tmp_path / "v", names(1, 24), {n: np.eye(4) for n in names(27, 40)}, names(25, 26))
    doc = json.loads((out["repaired"] / "transforms.json").read_text())
    for f in doc["frames"]:
        Image.open(out["repaired"] / f["file_path"]).load()
        Image.open(out["repaired"] / f["depth_file_path"]).load()
