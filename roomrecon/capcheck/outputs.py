"""What the capture check writes: the coverage map as small splats the SuperSplat viewer
opens, a floor plan, and the report."""
import json
from pathlib import Path

import numpy as np

from roomrecon.capcheck.labels import BROKEN, NAMES, OK, TEXTURELESS, UNDER
from roomrecon.scene import PLAYCANVAS_FROM_PLY

COLORS = {OK: (0.2, 0.8, 0.2), UNDER: (0.9, 0.1, 0.1), TEXTURELESS: (0.2, 0.4, 0.95),
          BROKEN: (0.5, 0.5, 0.5)}
SEVERITY = {OK: 0, TEXTURELESS: 1, UNDER: 2, BROKEN: 3}       # worst label wins a floor cell
SH_C0 = 0.28209479


def align(points: np.ndarray, frame: dict) -> np.ndarray:
    """Capture frame -> the aligned PlayCanvas frame (y up, metric), as for the splats."""
    R, s = np.asarray(frame["rotation"]), float(frame["scale"])
    return s * (points @ PLAYCANVAS_FROM_PLY.T @ R.T)


def rotate_dirs(dirs: np.ndarray, frame: dict) -> np.ndarray:
    return dirs @ PLAYCANVAS_FROM_PLY.T @ np.asarray(frame["rotation"]).T


def write_splat_ply(path, points, colors, frame, size=0.015):
    """3DGS PLY of opaque round splats; positions stored so that the viewer's own 180° turn
    about Z (PLY -> PlayCanvas) shows them aligned."""
    pos = align(np.asarray(points, float), frame) @ PLAYCANVAS_FROM_PLY.T
    n = len(pos)
    data = np.zeros((n, 14), "<f4")
    data[:, 0:3] = pos
    data[:, 3:6] = (np.asarray(colors, float) - 0.5) / SH_C0
    data[:, 6] = 4.0
    data[:, 7:10] = np.log(size)
    data[:, 10] = 1.0
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1",
             "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
              + "".join(f"property float {p}\n" for p in props) + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def label_colors(labels: np.ndarray) -> np.ndarray:
    return np.array([COLORS[int(l)] for l in range(4)])[labels]


def write_topdown(path, points_aligned, labels, cam_aligned, segments, jump_idx, cell=0.10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    xz = points_aligned[:, [0, 2]]
    lo = xz.min(0) - cell
    ij = np.floor((xz - lo) / cell).astype(int)
    grid = np.full(ij.max(0) + 2, -1)
    sev = np.array([SEVERITY[l] for l in range(4)])[labels]
    np.maximum.at(grid, (ij[:, 0], ij[:, 1]), sev)
    order = [OK, TEXTURELESS, UNDER, BROKEN]                       # severity 0..3
    cmap = ListedColormap([COLORS[l] for l in order])
    shown = np.ma.masked_less(grid.T.astype(float), 0)
    ext = [lo[0], lo[0] + grid.shape[0] * cell, lo[1], lo[1] + grid.shape[1] * cell]
    fig, ax = plt.subplots(figsize=(8, 8 * (ext[3] - ext[2]) / max(ext[1] - ext[0], 1e-6) + 0.5))
    ax.imshow(shown, origin="lower", extent=ext, cmap=cmap, vmin=-0.5, vmax=3.5, interpolation="nearest")
    segs = np.asarray(segments) if len(segments) else np.zeros(len(cam_aligned), int)
    for s in np.unique(segs):
        m = segs == s
        ax.plot(cam_aligned[m, 0], cam_aligned[m, 2], ".-", lw=0.8, ms=2, label=f"camera path, segment {s}")
    if len(jump_idx):
        ax.plot(cam_aligned[jump_idx, 0], cam_aligned[jump_idx, 2], "kx", ms=10, mew=2, label="pose jump")
    handles = [Patch(color=COLORS[l], label=NAMES[l]) for l in order] + ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(np.floor(ext[0]), np.ceil(ext[1]) + 1))
    ax.set_yticks(np.arange(np.floor(ext[2]), np.ceil(ext[3]) + 1))
    ax.grid(True, lw=0.3, color="k", alpha=0.3)
    ax.set_title("Capture check: worst label per 10 cm floor-plan cell")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _region_line(i, r):
    c = r["centre"]
    return (f"  {i}. {r['area_m2']:.2f} m² at x={c[0]:+.2f}, z={c[2]:+.2f} m; "
            f"{r['y_range'][0]:.1f}-{r['y_range'][1]:.1f} m above the floor; "
            f"about {r['size'][0]:.1f} x {r['size'][1]:.1f} x {r['size'][2]:.1f} m; "
            f"mostly seen from the {r['seen_from']} side")


def write_report(dir, result: dict):
    d = Path(dir)
    (d / "report.json").write_text(json.dumps(result, indent=2))
    t = result["tracking"]
    lines = [f"Verdict: {result['verdict']}", "",
             f"Capture: {result['capture']} ({result['source']}, {result['frames_used']} frames used, "
             f"{result['frames_excluded']} excluded)",
             f"Surface: {result['surface_points']} points ({result['settings']['voxel_m'] * 100:.0f} cm voxels)", "",
             "Surface area by label:"]
    for name, a in result["areas_m2"].items():
        lines.append(f"  {name:16s} {a:7.2f} m²  {result['percent'][name]:5.1f} %")
    lines += ["", "Recapture these under-covered areas (largest first):"]
    lines += [_region_line(i + 1, r) for i, r in enumerate(result["regions"])] or ["  none"]
    lines += ["", f"Textureless (well covered, plain): {result['areas_m2']['textureless']:.2f} m². "
              "More views will not fix these: move closer or add light; expect blur there.",
              f"Floor never seen inside the room outline: {result['floor_not_seen_m2']:.2f} m²", "",
              "Tracking:" if t["available"] else "Tracking: not available for this capture"]
    if t["available"]:
        lines += [f"  segment changes: {t['segment_changes'] or 'none'}",
                  f"  pose jumps: {[j['frame'] for j in t['jumps']] or 'none'}",
                  f"  broken frames (outside the largest unbroken piece): {t['n_broken_frames']}",
                  f"  limited mapping: {len(t['limited'])} frames; weakly connected: {len(t['weakly_connected'])}; "
                  f"bad timestamps: {len(t['bad_timestamps'])}"]
    lines += ["", f"Settings: {json.dumps(result['settings'])}", f"Run time: {result['run_time_s']:.0f} s"]
    (d / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
