"""Write a point set as opaque round splats, so splat-transform can build collision from it the
same way as from a trained splat. Positions stay in the PLY frame (identity alignment).
Usage: python -m roomrecon.collision.prep NPZ PLY [--key points] [--size 0.02]"""
import argparse

import numpy as np

from roomrecon.capcheck.outputs import write_splat_ply

IDENTITY = {"rotation": np.eye(3).tolist(), "scale": 1.0}


def points_to_splats(npz_path, ply_path, key="points", size=0.02):
    p = np.load(npz_path)[key].astype(float)
    write_splat_ply(ply_path, p, np.full((len(p), 3), 0.6), IDENTITY, size=size)
    return len(p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz")
    ap.add_argument("ply")
    ap.add_argument("--key", default="points")
    ap.add_argument("--size", type=float, default=0.02)
    a = ap.parse_args(argv)
    print(f"{points_to_splats(a.npz, a.ply, a.key, a.size)} splats -> {a.ply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
