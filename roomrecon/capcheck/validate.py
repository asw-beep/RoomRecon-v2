"""Validation of the capture check against trained splats (spec §7): do the labels mark the
regions where the trained splat's held-out renders are wrong?

The check is run on the training frames only; each held-out test frame's own depth maps its
pixels to the check's surface points. Errors and labels are compared on 16x16-pixel patches.
The texture threshold is calibrated on one room and applied unchanged to the other.

Usage (GPU machine):
  python -m roomrecon.capcheck.validate --calib NAME CAPTURE NPZ RENDERS_RGB RENDERS_GT \n      --test NAME CAPTURE NPZ RENDERS_RGB RENDERS_GT --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from roomrecon.capcheck.capture import Capture, load_capture, load_depth_m
from roomrecon.capcheck.labels import BROKEN, NAMES, OK, TEXTURELESS, UNDER, label_points
from roomrecon.capcheck.surface import backproject
from roomrecon.capcheck.views import _median_by_group

BAD_SHARE = 0.20          # "bad" = the room's top 20 % patch error
AUC_MARGIN = 0.05         # texture must add this much AUC over coverage alone


def _find(d: Path, stem: str) -> Path:
    hits = [p for p in Path(d).rglob(f"{stem}.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not hits:
        raise FileNotFoundError(f"no image named {stem} under {d}")
    return hits[0]


def _rgb(path: Path, w: int, h: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if im.size != (w, h):
        im = im.resize((w, h), Image.BILINEAR)
    return np.asarray(im).astype(np.float32)


def frame_samples(cap: Capture, name: str, tree: cKDTree, radius=0.03, stride=4):
    """(v, u, matched point index or -1) for the frame's stride-4 depth samples."""
    f = next(fr for fr in cap.frames if fr.name == name)
    pts, v, u = backproject(load_depth_m(f), f.K, f.c2w, stride=stride, return_pixels=True)
    d, idx = tree.query(pts, distance_upper_bound=radius) if len(pts) else (np.zeros(0), np.zeros(0, int))
    return f, v, u, np.where(np.isfinite(d), idx, -1)


def patch_table(cap: Capture, test_names: list[str], renders_rgb: Path, renders_gt: Path, npz: dict,
                labels: np.ndarray, patch=16, radius=0.03, stride=4) -> dict[str, np.ndarray]:
    tree = cKDTree(npz["points"])
    cols = {k: [] for k in ("frame", "pv", "pu", "error", "label", "n_views", "n_dirs", "texture")}
    per = (patch // stride) ** 2
    for name in test_names:
        f, v, u, idx = frame_samples(cap, name, tree, radius, stride)
        err = np.abs(_rgb(_find(renders_rgb, name), f.width, f.height)
                     - _rgb(_find(renders_gt, name), f.width, f.height)).mean(axis=2) / 255.0
        npv, npu = f.height // patch, f.width // patch
        blocks = err[:npv * patch, :npu * patch].reshape(npv, patch, npu, patch).mean(axis=(1, 3))
        ok = (idx >= 0) & (v < npv * patch) & (u < npu * patch)
        pid = (v[ok] // patch) * npu + u[ok] // patch
        m = idx[ok]
        n = npv * npu
        matched = np.bincount(pid, minlength=n)
        keep = np.flatnonzero(matched >= per / 2)
        lab_counts = np.bincount(pid * 4 + labels[m], minlength=n * 4).reshape(n, 4)
        med = {k: _median_by_group(pid, np.asarray(npz[k], np.float32)[m], n) for k in ("n_views", "n_dirs", "texture")}
        cols["frame"].append(np.full(len(keep), name))
        cols["pv"].append(keep // npu)
        cols["pu"].append(keep % npu)
        cols["error"].append(blocks.reshape(-1)[keep])
        cols["label"].append(lab_counts[keep].argmax(axis=1).astype(np.uint8))
        for k in med:
            cols[k].append(med[k][keep])
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in cols.items()}


def bootstrap_mean_ci(values, groups, n=1000, seed=0) -> tuple[float, float, float]:
    """Mean, and its 95 % interval from resampling whole groups (test frames)."""
    values = np.asarray(values, float)
    _, g = np.unique(np.asarray(groups), return_inverse=True)
    sums, counts = np.bincount(g, values), np.bincount(g)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(sums), (n, len(sums)))
    boot = sums[pick].sum(axis=1) / np.maximum(counts[pick].sum(axis=1), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def auc(scores, positives) -> float:
    """Probability that a positive scores higher than a negative (ties count half)."""
    pos = np.asarray(positives, bool)
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_logistic(X, y, l2=1e-3) -> dict:
    """Logistic regression on standardised features; the standardisation travels with the model
    so it can be applied to another room."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]

    def loss(w):
        z = Z @ w
        p = 1 / (1 + np.exp(-z))
        reg = l2 * np.sum(w[1:] ** 2)
        return np.mean(np.logaddexp(0, z) - y * z) + reg, Z.T @ (p - y) / len(y) + 2 * l2 * np.r_[0, w[1:]]

    w = minimize(loss, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B").x
    return {"mu": mu, "sd": sd, "w": w}


def predict_logistic(model: dict, X) -> np.ndarray:
    Z = np.c_[np.ones(len(X)), (np.asarray(X, float) - model["mu"]) / model["sd"]]
    return 1 / (1 + np.exp(-(Z @ model["w"])))


def is_bad(table) -> np.ndarray:
    return table["error"] >= np.quantile(table["error"], 1 - BAD_SHARE)


def calibrate_tau(table) -> float:
    """Texture threshold that best separates bad patches among those not under-covered."""
    sel = (table["label"] != UNDER) & (table["label"] != BROKEN) & np.isfinite(table["texture"])
    bad = is_bad(table)[sel]
    tex = table["texture"][sel]
    if len(tex) == 0 or bad.all() or not bad.any():
        return 0.0                          # nothing to calibrate on: no point is textureless
    best, best_j = float(np.percentile(tex, 1)), -np.inf
    for tau in np.unique(np.percentile(tex, np.arange(1, 100))):
        pred = tex < tau
        j = pred[bad].mean() - pred[~bad].mean()
        if j > best_j:
            best, best_j = float(tau), j
    return best


FEATURES = {"coverage": lambda t: np.c_[np.log1p(t["n_views"]), t["n_dirs"]],
            "texture": lambda t: np.c_[np.log(np.nan_to_num(t["texture"], nan=0.0) + 1e-4)],
            "both": lambda t: np.c_[np.log1p(t["n_views"]), t["n_dirs"],
                                    np.log(np.nan_to_num(t["texture"], nan=0.0) + 1e-4)]}


def _room(spec: list[str]):
    name, capture, npz_path, rgb, gt = spec
    npz = dict(np.load(npz_path))
    cap = load_capture(Path(capture))
    tests = sorted({p.stem for p in Path(rgb).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")})
    return {"name": name, "cap": cap, "npz": npz, "rgb": Path(rgb), "gt": Path(gt), "tests": tests}


def _table(room, tau):
    npz = room["npz"]
    labels = label_points(npz["n_views"], npz["n_dirs"], npz["texture"], npz["broken_pts"], tau)
    return patch_table(room["cap"], room["tests"], room["rgb"], room["gt"], npz, labels), labels


def _label_stats(t):
    out = {}
    for k in (OK, UNDER, TEXTURELESS, BROKEN):
        m = t["label"] == k
        if m.sum() == 0:
            out[NAMES[k]] = {"patches": 0}
            continue
        mean, lo, hi = bootstrap_mean_ci(t["error"][m], t["frame"][m])
        out[NAMES[k]] = {"patches": int(m.sum()), "frames": int(len(set(t["frame"][m]))),
                         "mean_error": mean, "ci95": [lo, hi]}
    return out


def _figures(out: Path, rooms, tables, labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from roomrecon.capcheck.outputs import COLORS

    order = [OK, UNDER, TEXTURELESS, BROKEN]
    fig, axes = plt.subplots(1, len(rooms), figsize=(6 * len(rooms), 4.5), squeeze=False)
    for ax, room, t in zip(axes[0], rooms, tables):
        data = [t["error"][t["label"] == k] for k in order]
        ks = [k for k, d in zip(order, data) if len(d)]
        bp = ax.boxplot([t["error"][t["label"] == k] for k in ks], showfliers=False, patch_artist=True)
        for patch, k in zip(bp["boxes"], ks):
            patch.set_facecolor(COLORS[k])
        ax.set_xticks(range(1, len(ks) + 1), [f"{NAMES[k]}\n(n={int((t['label'] == k).sum())})" for k in ks])
        ax.set_ylabel("held-out patch error (mean |render - photo|, 0-1)")
        ax.set_title(room["name"])
    fig.tight_layout()
    fig.savefig(out / "error_by_label.png", dpi=120)
    plt.close(fig)

    room, t, lab = rooms[-1], tables[-1], labels[-1]          # the test room
    frames = sorted(set(t["frame"]), key=lambda f: -np.mean(t["label"][t["frame"] == f] != OK))
    picks = frames[:1] + sorted(set(t["frame"]), key=lambda f: -np.mean(t["label"][t["frame"] == f] == TEXTURELESS))[:1]
    tree = cKDTree(room["npz"]["points"])
    fig, axes = plt.subplots(len(picks), 4, figsize=(16, 3.4 * len(picks)), squeeze=False)
    for row, name in zip(axes, picks):
        f, v, u, idx = frame_samples(room["cap"], name, tree)
        gt = _rgb(_find(room["gt"], name), f.width, f.height)
        rd = _rgb(_find(room["rgb"], name), f.width, f.height)
        lab_img = np.ones((f.height, f.width, 3))
        cols = np.array([COLORS[k] for k in range(4)])
        m = idx >= 0
        for dv in range(4):
            for du in range(4):
                vv, uu = np.clip(v[m] + dv, 0, f.height - 1), np.clip(u[m] + du, 0, f.width - 1)
                lab_img[vv, uu] = cols[lab[idx[m]]]
        for ax, im, title in zip(row, [gt / 255, rd / 255, np.abs(rd - gt).mean(2) / 255, lab_img],
                                 ["photo (held out)", "trained splat", "error", "capture-check label"]):
            ax.imshow(im, cmap="magma" if im.ndim == 2 else None, vmin=0, vmax=1 if im.ndim == 3 else 0.5)
            ax.set_title(f"{name}: {title}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "examples.png", dpi=100)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    spec = ("NAME", "CAPTURE", "NPZ", "RENDERS_RGB", "RENDERS_GT")
    ap.add_argument("--calib", nargs=5, metavar=spec, required=True)
    ap.add_argument("--test", nargs=5, metavar=spec, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib, test = _room(a.calib), _room(a.test)

    tau = calibrate_tau(_table(calib, 0.0)[0])
    (tc, lc), (tt, lt) = _table(calib, tau), _table(test, tau)
    aucs = {}
    for fit, ev in ((calib, test), (test, calib)):
        tf, te = (tc, tt) if fit is calib else (tt, tc)
        aucs[f"fit {fit['name']} -> eval {ev['name']}"] = {
            k: auc(predict_logistic(fit_logistic(F(tf), is_bad(tf)), F(te)), is_bad(te)) for k, F in FEATURES.items()}
    stats = {calib["name"]: _label_stats(tc), test["name"]: _label_stats(tt)}

    st, key = stats[test["name"]], f"fit {calib['name']} -> eval {test['name']}"
    ok_hi = st["ok"].get("ci95", [np.nan, np.nan])[1]
    c1 = all(st[n].get("patches", 0) > 0 and st[n]["ci95"][0] > ok_hi for n in ("under-covered", "textureless"))
    gain = aucs[key]["both"] - aucs[key]["coverage"]
    result = {"tau_tex": tau, "bad_share": BAD_SHARE, "patch_px": 16, "match_radius_m": 0.03,
              "rooms": {r["name"]: {"test_frames": len(r["tests"]), "patches": int(len(t["error"]))}
                        for r, t in ((calib, tc), (test, tt))},
              "error_by_label": stats, "auc": aucs,
              "criteria": {"1_under_and_textureless_above_ok_in_test_room": bool(c1),
                           "2_texture_adds_auc": {"gain": gain, "margin": AUC_MARGIN, "pass": bool(gain >= AUC_MARGIN)}}}
    (out / "validation.json").write_text(json.dumps(result, indent=2))
    _figures(out, [calib, test], [tc, tt], [lc, lt])
    print(json.dumps(result["criteria"]), f"tau={tau:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
