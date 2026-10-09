"""Make the report's data figures from the project's evidence (results/ and the PC scene folder).
Run from the repo root: python report/make_figures.py [--scenes E:\\roomrecon-scenes]
Only reads the evidence; writes into report/figures/."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
RES = REPO / "results"
OUT = REPO / "report" / "figures"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8})

# v1 (laptop, gsplat MCMC, full ARKitScenes bedroom walk): docs/lessons-learned-v1.md section 3
V1_STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
V1_MCMC = [17.4, 21.4, 22.1, 22.8, 22.6, 22.1, 22.1, 21.1, 19.4]
V1_ADC = [20.2, 21.0, 21.4, 21.5, 21.5, 21.4, 21.3, 21.1, 20.9]


def curve(csv_path, tag):
    rows = [r for r in csv.DictReader(open(csv_path)) if r["tag"] == tag]
    pts = sorted({int(r["step"]): float(r["value"]) for r in rows}.items())
    return np.array([p[0] for p in pts]) / 1000, np.array([p[1] for p in pts])


def fig_curves():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    a.plot(V1_STEPS, V1_MCMC, "o-", color="tab:red", label="MCMC densification")
    a.plot(V1_STEPS, V1_ADC, "s--", color="tab:orange", label="default densification")
    a.plot([15], [13.7], "o", color="tab:red", mfc="none")
    a.annotate("MCMC final (15k): 13.7 dB", xy=(15, 13.7), xytext=(5.0, 16.2), arrowprops={"arrowstyle": "->"})
    a.set(title="v1: whole-room 3DGS (laptop, ARKitScenes bedroom)", xlabel="step (thousands)",
          ylabel="held-out PSNR (dB)", ylim=(12, 28), xlim=(0, 16))
    a.legend(loc="upper left")
    t = "Eval Images Metrics Dict (all images)/"
    for f, tag, style, lab in [
        ("g1-vr_room-eval-curve.csv", t + "within_rgb_psnr", "-", "vr_room, DN-Splatter (held-out frames)"),
        ("g1-vr_room-eval-curve.csv", t + "with_rgb_psnr", "--", "vr_room, DN-Splatter (other capture)"),
        ("g3-plain-eval-curve.csv", t + "within_rgb_psnr", ":", "vr_room, plain splats (held-out frames)"),
        ("g4-sejal-cap2-eval-curve.csv", t + "rgb_psnr", "-.", "own room cap2, DN-Splatter (held-out)")]:
        x, y = curve(RES / f, tag)
        b.plot(x, y, style, marker=".", ms=3, label=lab)
    b.set(title="v2: DN-Splatter on phone poses (A100)", xlabel="step (thousands)", ylim=(12, 28))
    b.legend(loc="lower right")
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "curves.pdf")
    plt.close(fig)


def fig_renders():
    a = Image.open(RES / "g2-renders/long_capture_frame_00011.jpg")
    b = Image.open(RES / "g2-renders/short_capture_frame_00134.jpg")
    c = Image.open(RES / "g4-renders/test_frame_00129_gt-vs-render.jpg")
    fig = plt.figure(figsize=(7.0, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.337 * 2, 0.369 * 1.0])
    for ax, im, t in [(fig.add_subplot(gs[0, 0]), a, "(a) vr_room, held-out frame (render | photo)"),
                      (fig.add_subplot(gs[0, 1]), b, "(b) vr_room, other walk (render | photo)"),
                      (fig.add_subplot(gs[1, :]), c, "(c) own room, held-out frame (photo | render)")]:
        ax.imshow(im)
        ax.set_title(t, fontsize=8)
        ax.axis("off")
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.6)
    fig.savefig(OUT / "renders.jpg", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_teaser(scenes):
    rec = scenes / "recordings"
    frames = []
    for name, t in [("g2-vr_room-walk.mp4", 40), ("g4-sejal-cap2-walk.mp4", 5)]:
        p = OUT / f"_walk_{t}.jpg"
        import subprocess
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(rec / name),
                        "-frames:v", "1", str(p)], check=True)
        frames.append(Image.open(p).convert("RGB"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.75))
    for ax, im, title in zip(axes, frames, ["MuSHRoom vr_room (web data)", "Own room, captured on an iPhone Pro"]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "teaser.jpg", dpi=250)
    plt.close(fig)
    for p in OUT.glob("_walk_*.jpg"):
        p.unlink()


def fig_collision(scenes):
    from roomrecon.capcheck.outputs import align
    from roomrecon.collision.glb import read_glb, sample_triangles
    from roomrecon.collision.grid import make_grid, occupancy, walkable
    from roomrecon.scene import floor_height
    study = scenes / "collision-study"
    m = json.loads((study / "results/metrics.json").read_text())
    frame = json.loads((scenes / "vr_room/dns-30k-lite/scene.json").read_text())
    s = float(frame["scale"])
    spawn = np.asarray(frame["spawn_position"], float)
    gt_pts = align(np.load(study / "gt_points.npz")["points"].astype(float), frame)
    floor_y = floor_height(gt_pts[:, 1], np.array([spawn[1]]))
    assert abs(floor_y - m["floor_y"]) < 1e-9, "floor differs from the study's metrics.json"
    grid = make_grid(gt_pts[:, [0, 2]], cell=0.05 * s, margin=0.5 * s)
    assert list(grid.shape) == m["grid"]["shape"], "grid differs from the study's metrics.json"
    names = ["S", "M", "P", "L", "GT"]
    occ = {n: occupancy(sample_triangles(*read_glb(study / n / "room.collision.glb"), spacing=0.01 * s),
                        grid, floor_y, scale=s) for n in names}
    walk = {n: walkable(occ[n], spawn[[0, 2]], grid) for n in names}
    titles = {"S": "S: splat (default)", "M": "M: DN-Splatter mesh", "P": "P: Polycam mesh",
              "L": "L: raw phone LiDAR", "GT": "GT: laser scan"}
    ext = [grid.lo[0] / s, (grid.lo[0] + grid.shape[0] * grid.cell) / s,
           grid.lo[1] / s, (grid.lo[1] + grid.shape[1] * grid.cell) / s]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.25), sharey=True)
    for ax, n in zip(axes, names):
        img = np.ones(grid.shape + (3,))
        img[walk[n]] = (0.85, 0.92, 1.0)
        img[occ["GT"]] = (0.6, 0.6, 0.6)
        img[occ[n]] = (0.85, 0.15, 0.15)
        img[occ[n] & occ["GT"]] = (0.35, 0.1, 0.1)
        ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=ext, interpolation="nearest")
        r = m["sources"][n]
        sub = "" if n == "GT" else f"\nIoU {r['iou']:.2f}, missed {r['missed_pct']:.1f}%"
        ax.set_title(titles[n] + sub, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("x (m)", fontsize=6)
    axes[0].set_ylabel("z (m)", fontsize=6)
    fig.tight_layout(pad=0.4, w_pad=0.4, rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "collision.pdf")
    plt.close(fig)


def copy_capcheck(scenes):
    shutil.copy(scenes / "capcheck/cap3/topdown.png", OUT / "capcheck_cap3.png")
    err = Image.open(scenes / "capcheck/validation/error_by_label.png")
    err.crop((err.width // 2, 0, err.width, err.height)).save(OUT / "capcheck_error_cap2.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=r"E:\roomrecon-scenes")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sc = Path(a.scenes)
    fig_curves()
    fig_renders()
    fig_teaser(sc)
    fig_collision(sc)
    copy_capcheck(sc)
    print("\n".join(sorted(p.name for p in OUT.iterdir())))
