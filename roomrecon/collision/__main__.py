"""Compare walk-collision meshes with the laser scan's (spec §3, §4).
Usage: python -m roomrecon.collision --glb S=... --glb M=... --glb L=... --glb GT=...
           --gt-points NPZ --lidar-points NPZ --scene SCENE_JSON --out DIR
GLBs are splat-transform outputs (aligned frame); the point sets are in the model's frame and are
aligned here with the scene's rotation and scale. Lengths are reported in metres (÷ scale)."""
import argparse
import json
from pathlib import Path

import numpy as np

from roomrecon.capcheck.outputs import align
from roomrecon.collision.glb import read_glb, sample_triangles
from roomrecon.collision.grid import compare, make_grid, occupancy, walkable
from roomrecon.scene import floor_height

CELL_M, MARGIN_M, SPACING_M = 0.05, 0.5, 0.01


def criteria(x: dict, s: dict) -> dict:
    return {"P1": x["iou"] >= s["iou"] + 0.02, "P2": x["phantom_m2"] <= s["phantom_m2"],
            "P3": x["missed_pct"] <= s["missed_pct"] + 2.0, "P4": x["offset_median_cm"] <= s["offset_median_cm"]}


def _figure(path, occ, gt, walk, grid, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5), squeeze=False)
    ext = [grid.lo[0], grid.lo[0] + grid.shape[0] * grid.cell, grid.lo[1], grid.lo[1] + grid.shape[1] * grid.cell]
    for ax, n in zip(axes[0], names):
        img = np.ones(grid.shape + (3,))
        img[walk[n]] = (0.85, 0.92, 1.0)                 # walkable from the spawn
        img[gt] = (0.6, 0.6, 0.6)                         # laser scan
        img[occ[n]] = (0.85, 0.15, 0.15)                  # this source
        img[occ[n] & gt] = (0.35, 0.1, 0.1)
        ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=ext, interpolation="nearest")
        ax.set_title({"S": "S: from the splat (default)", "M": "M: from DN-Splatter's mesh",
                      "L": "L: from the phone's LiDAR", "GT": "GT: laser scan"}.get(n, n))
        ax.set_aspect("equal")
    fig.suptitle("Collision at body height (red) over the laser scan (grey); blue = walkable from the spawn")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glb", action="append", required=True, help="NAME=PATH (S, M, L, GT)")
    ap.add_argument("--gt-points", required=True)
    ap.add_argument("--lidar-points", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    glbs = dict(g.split("=", 1) for g in a.glb)
    frame = json.loads(Path(a.scene).read_text())
    s = float(frame["scale"])
    spawn = np.asarray(frame["spawn_position"], float)
    gt_pts = align(np.load(a.gt_points)["points"].astype(float), frame)
    lidar = align(np.load(a.lidar_points)["points"].astype(float), frame)
    floor_y = floor_height(gt_pts[:, 1], np.array([spawn[1]]))
    grid = make_grid(gt_pts[:, [0, 2]], cell=CELL_M * s, margin=MARGIN_M * s)
    occ_l = occupancy(lidar, grid, floor_y, scale=s)
    occ = {n: occupancy(sample_triangles(*read_glb(p), spacing=SPACING_M * s), grid, floor_y, scale=s)
           for n, p in glbs.items()}
    res = {n: compare(occ[n], occ["GT"], occ_l, spawn[[0, 2]], grid, scale=s) for n in occ}
    crit = {f"{x}_vs_S": criteria(res[x], res["S"]) for x in ("M", "L") if x in res and "S" in res}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(
        {"floor_y": float(floor_y), "scale": s, "cell_m": CELL_M, "slab_m": [0.2, 1.8],
         "grid": {"lo": grid.lo.tolist(), "shape": list(grid.shape)}, "glbs": glbs,
         "sources": res, "criteria": {k: {p: bool(v) for p, v in c.items()} for k, c in crit.items()}}, indent=2))
    walk = {n: walkable(occ[n], spawn[[0, 2]], grid) for n in occ}
    _figure(out / "topdown.png", occ, occ["GT"], walk, grid, [n for n in ("S", "M", "L", "GT") if n in occ])
    for n, m in res.items():
        print(f"{n:3s} IoU {m['iou']:.3f}  phantom {m['phantom_m2']:.2f} m²  missed {m['missed_pct']:.1f} %  "
              f"offset {m['offset_median_cm']:.1f} / {m['offset_p95_cm']:.1f} cm  walkable {m['walkable_m2']:.1f} m²")
    print(json.dumps({k: {p: bool(v) for p, v in c.items()} for k, c in crit.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
