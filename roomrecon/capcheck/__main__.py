"""Capture check: where a phone room capture will be poor after training, and why.
Usage: python -m roomrecon.capcheck CAPTURE --out DIR [--exclude FILE] [--tau-tex X] [--name NAME]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from roomrecon.capcheck.capture import load_capture, read_exclude
from roomrecon.capcheck.labels import (MIN_DIRS, MIN_VIEWS, NAMES, TAU_TEX, floor_not_seen,
                                       label_points, regions, verdict)
from roomrecon.capcheck.outputs import (align, label_colors, rotate_dirs, write_report,
                                        write_splat_ply, write_topdown)
from roomrecon.capcheck.surface import build_surface
from roomrecon.capcheck.tracking import check_tracking
from roomrecon.capcheck.views import accumulate
from roomrecon.scene_cli import build_scene, viewer_settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", help="file with frame names to leave out (e.g. held-out test frames)")
    ap.add_argument("--tau-tex", type=float, default=TAU_TEX)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exclude = read_exclude(Path(a.exclude)) if a.exclude else frozenset()
    cap = load_capture(Path(a.capture), exclude)
    name = a.name or cap.root.name

    surface = build_surface(cap)
    vs = accumulate(surface, cap)
    tr = check_tracking(cap)
    broken_pts = np.zeros(len(surface.points), bool)
    if tr.broken:
        keep = np.array([f.name not in tr.broken for f in cap.frames])
        broken_pts = (vs.n_views > 0) & (accumulate(surface, cap, frames_mask=keep).n_views == 0)
    labels = label_points(vs.n_views, vs.n_dirs, vs.texture, broken_pts, a.tau_tex)

    c2w = np.stack([f.c2w for f in cap.frames])
    frame = build_scene(c2w, surface.points, room=name, run_id="capcheck")
    pts_a = align(surface.points, frame)
    cams_a = align(c2w[:, :3, 3], frame)
    regs = regions(pts_a, labels, rotate_dirs(vs.mean_view_dir, frame), frame["floor_y"], surface.voxel)
    unseen = floor_not_seen(pts_a, frame["floor_y"])

    write_splat_ply(out / "coverage.ply", surface.points, label_colors(labels), frame)
    jump_idx = [i for i, f in enumerate(cap.frames) if f.name in {j["frame"] for j in tr.jumps}]
    write_topdown(out / "topdown.png", pts_a, labels, cams_a, tr.segments, jump_idx)
    (out / "scene.json").write_text(json.dumps(frame, indent=2))
    (out / "settings.json").write_text(json.dumps(viewer_settings(frame), indent=2))
    np.savez_compressed(out / "points.npz", points=surface.points, normals=surface.normals,
                        n_views=vs.n_views, n_dirs=vs.n_dirs, texture=vs.texture,
                        footprint=vs.best_footprint_mm, labels=labels, broken_pts=broken_pts)

    area = surface.voxel ** 2
    counts = {NAMES[k]: int((labels == k).sum()) for k in sorted(NAMES)}
    total = max(len(labels), 1)
    result = {
        "name": name, "capture": str(cap.root), "source": cap.source,
        "frames_used": len(cap.frames), "frames_excluded": len(exclude),
        "verdict": verdict(tr, regs), "surface_points": int(len(labels)),
        "areas_m2": {k: v * area for k, v in counts.items()},
        "percent": {k: 100.0 * v / total for k, v in counts.items()},
        "regions": regs, "floor_not_seen_m2": unseen["area_m2"],
        "tracking": {"available": tr.available, "segment_changes": tr.segment_changes,
                     "jumps": tr.jumps, "limited": tr.limited, "weakly_connected": tr.weakly_connected,
                     "bad_timestamps": tr.bad_timestamps, "n_broken_frames": len(tr.broken),
                     "broken_frames": sorted(tr.broken)},
        "scene": {"scale": frame["scale"], "floor_y": frame["floor_y"]},
        "settings": {"voxel_m": surface.voxel, "min_views": MIN_VIEWS, "min_dirs": MIN_DIRS,
                     "tau_tex": a.tau_tex},
        "run_time_s": time.perf_counter() - t0,
    }
    write_report(out, result)
    print(f"{name}: {result['verdict']} ({len(labels)} points, {result['run_time_s']:.0f} s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
