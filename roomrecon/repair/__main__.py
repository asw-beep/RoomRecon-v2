"""Repair a phone capture whose tracking broke, and write the raw / drop / repaired variants.
Usage: python -m roomrecon.repair CAPTURE --out DIR [--min-frames 20]
CAPTURE is an `ns-process-data polycam` folder with Polycam's keyframes/ (tracking data).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from roomrecon.capcheck.capture import load_capture
from roomrecon.capcheck.tracking import check_tracking
from roomrecon.repair.metrics import added_coverage, alignment_error
from roomrecon.repair.register import register_piece
from roomrecon.repair.variants import piece_cloud, split_pieces, subset, write_variants
from roomrecon.scene import estimate_up

C1_MAX_M, C3_MIN = 0.03, 0.05          # spec §4


def _span(names):
    return [names[0], names[-1], len(names)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-frames", type=int, default=20)
    a = ap.parse_args(argv)
    out = Path(a.out)
    cap = load_capture(Path(a.capture))
    if not cap.has_tracking:
        raise SystemExit(f"no tracking data: cannot find pieces in {cap.root}")
    ref, cands, dropped = split_pieces(cap, check_tracking(cap), a.min_frames)
    out.mkdir(parents=True, exist_ok=True)

    pieces, accepted, rejected = [], {}, []
    if cands:
        ref_cloud = piece_cloud(cap, ref)
        up = estimate_up(np.stack([f.c2w for f in subset(cap, ref).frames]))
        for p in cands:
            cloud = piece_cloud(cap, p)
            r = register_piece(cloud.points, ref_cloud.points, ref_cloud.normals, up)
            T = np.asarray(r["T"])
            r.update(frames=_span(p), c1_before_m=alignment_error(cap, p, ref_cloud),
                     c1_after_m=alignment_error(cap, p, ref_cloud, T))
            pieces.append(r)
            if r["accepted"]:
                accepted.update({n: T for n in p})
            else:
                rejected += p
    paths = write_variants(cap.root, out, ref, accepted, dropped + rejected)
    repaired_names = [n for n in accepted]
    c3 = added_coverage(load_capture(paths["repaired"]), ref, repaired_names) if repaired_names else 0.0
    good = [r for r in pieces if r["accepted"]]
    result = {
        "capture": str(cap.root), "reference": _span(ref), "pieces": pieces, "dropped": dropped,
        "c3_added_share": c3,
        "criteria": {"C1": bool(good) and all(r["c1_after_m"] <= C1_MAX_M for r in good), "C3": c3 >= C3_MIN},
        "variants": {k: len(json.loads((v / "transforms.json").read_text())["frames"]) for k, v in paths.items()},
    }
    (out / "repair.json").write_text(json.dumps(result, indent=2))
    if not cands:
        print(f"nothing to repair: one tracking piece ({len(ref)} frames); variants written to {out}")
    else:
        for r in pieces:
            print(f"piece {r['frames']}: {'accepted' if r['accepted'] else 'REJECTED (' + r['reason'] + ')'}, "
                  f"fitness {r['fitness']:.2f}, yaw {r['yaw_deg']:.1f} deg, tilt {r['tilt_deg']:.1f} deg, "
                  f"C1 {r['c1_before_m']:.3f} -> {r['c1_after_m']:.3f} m")
        print(f"C3 added surface {c3:.1%}; variants {result['variants']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
