"""Pieces of a capture with broken tracking, and the three training variants.

A piece is a run of frames between tracking breaks (segment changes and pose jumps, as found by
the capture check). The largest piece is the reference. The variants share one held-out test
list (every 8th reference frame), so their renders are compared on the same frames:
raw (all frames, exported poses), drop (reference only), repaired (reference + accepted pieces
with corrected poses). DN-Splatter reads explicit split lists from transforms.json
(normal_nerfstudio.py:215-236); nerfstudio evaluates on "val" during training and "test" in
ns-eval, so both get the same list.
"""
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np

from roomrecon.capcheck.capture import Capture
from roomrecon.capcheck.surface import Surface, build_surface
from roomrecon.capcheck.tracking import Tracking

LINKED = ("images", "depth", "normals_from_pretrain")


def split_pieces(cap: Capture, tr: Tracking, min_frames=20):
    """(reference names, candidate pieces, dropped names)."""
    cuts = set(tr.segment_changes) | {j["frame"] for j in tr.jumps}
    pieces, cur = [], []
    for f in cap.frames:
        if f.name in cuts and cur:
            pieces.append(cur)
            cur = []
        cur.append(f.name)
    if cur:
        pieces.append(cur)
    ref = max(pieces, key=len)
    rest = [p for p in pieces if p is not ref]
    return ref, [p for p in rest if len(p) >= min_frames], [n for p in rest if len(p) < min_frames for n in p]


def subset(cap: Capture, names) -> Capture:
    keep = set(names)
    return Capture(root=cap.root, source=cap.source, frames=[f for f in cap.frames if f.name in keep],
                   has_tracking=cap.has_tracking)


def piece_cloud(cap: Capture, names) -> Surface:
    return build_surface(subset(cap, names), voxel=0.05, stride=8)


def held_out_names(reference: list[str]) -> list[str]:
    return list(reference[::8])


def _c2w(m) -> np.ndarray:
    m = np.asarray(m, float)
    return np.vstack([m, [0, 0, 0, 1]]) if m.shape == (3, 4) else m


def _link(src: Path, dst: Path):
    try:
        os.symlink(src.resolve(), dst, target_is_directory=True)
    except OSError:                          # e.g. Windows without symlink rights
        shutil.copytree(src, dst)


def write_variants(src_root: Path, out: Path, reference, accepted: dict, dropped) -> dict:
    src_root, out = Path(src_root), Path(out)
    doc = json.loads((src_root / "transforms.json").read_text())
    stem = lambda fr: Path(fr["file_path"]).stem
    by_name = {stem(fr): fr for fr in doc["frames"]}
    members = {"raw": [stem(fr) for fr in doc["frames"]], "drop": set(reference),
               "repaired": set(reference) | set(accepted)}
    tests = [by_name[n]["file_path"] for n in held_out_names(list(reference))]
    paths = {}
    for name in ("raw", "drop", "repaired"):
        keep = set(members[name])
        frames = []
        for fr in doc["frames"]:
            if stem(fr) not in keep:
                continue
            fr = copy.deepcopy(fr)
            if name == "repaired" and stem(fr) in accepted:
                fr["transform_matrix"] = (np.asarray(accepted[stem(fr)]) @ _c2w(fr["transform_matrix"])).tolist()
            frames.append(fr)
        d = {k: v for k, v in doc.items()
             if k not in ("frames", "ply_file_path", "train_filenames", "val_filenames", "test_filenames")}
        d["frames"] = frames
        d["test_filenames"] = list(tests)
        d["val_filenames"] = list(tests)
        d["train_filenames"] = [fr["file_path"] for fr in frames if fr["file_path"] not in set(tests)]
        vdir = out / name
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "transforms.json").write_text(json.dumps(d, indent=1))
        for sub in LINKED:
            if (src_root / sub).is_dir() and not (vdir / sub).exists():
                _link(src_root / sub, vdir / sub)
        paths[name] = vdir
    return paths
