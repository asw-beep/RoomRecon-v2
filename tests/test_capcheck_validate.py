import numpy as np
import pytest
from PIL import Image

from roomrecon.capcheck.capture import load_capture
from roomrecon.capcheck.labels import TEXTURELESS, label_points
from roomrecon.capcheck.surface import build_surface
from roomrecon.capcheck.validate import (auc, bootstrap_mean_ci, calibrate_tau, fit_logistic,
                                         patch_table, predict_logistic)
from roomrecon.capcheck.views import accumulate
from tests.synthroom import make_room
from tests.test_capcheck_surface import ring_cams


def test_auc_perfect_and_random():
    pos = np.array([0, 0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.3, 0.8, 0.9]), pos) == pytest.approx(1.0)
    assert auc(np.ones(5), pos) == pytest.approx(0.5)
    assert auc(-np.array([0.1, 0.2, 0.3, 0.8, 0.9]), pos) == pytest.approx(0.0)


def test_bootstrap_ci_contains_mean_and_narrows_with_more_groups():
    rng = np.random.default_rng(0)
    widths = []
    for n_groups in (8, 200):
        groups = np.repeat(np.arange(n_groups), 20)
        values = rng.normal(0.1, 0.05, n_groups)[groups] + rng.normal(0, 0.01, len(groups))
        mean, lo, hi = bootstrap_mean_ci(values, groups)
        assert mean == pytest.approx(values.mean()) and lo < mean < hi
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 2


def test_calibrate_tau_separates_synthetic_patches():
    n = 500
    bad = np.arange(n) < 100                                    # top 20 % error
    table = {"error": np.where(bad, 0.3, 0.05), "texture": np.where(bad, 0.01, 0.1),
             "label": np.zeros(n, np.uint8)}
    tau = calibrate_tau(table)
    assert 0.01 < tau <= 0.1



def test_calibrate_tau_without_covered_patches_is_zero():
    from roomrecon.capcheck.labels import UNDER
    table = {"error": np.linspace(0, 1, 50), "texture": np.full(50, 0.05), "label": np.full(50, UNDER, np.uint8)}
    assert calibrate_tau(table) == 0.0

def test_logistic_recovers_sign():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 1))
    model = fit_logistic(x, (x[:, 0] > 0).astype(float))
    p = predict_logistic(model, np.array([[-2.0], [2.0]]))
    assert p[1] > 0.9 > 0.1 > p[0]
    shifted = predict_logistic(model, x + 0.0)
    assert auc(shifted, x[:, 0] > 0) > 0.99


def test_patch_table_on_synthetic_room(tmp_path):
    hi = dict(w=128, h=96, f=100.0)
    cams = ring_cams() + [(np.array([2.0, 1.25, 1.5]), -90, 0)]          # last: faces the x=4 checker wall
    cap = load_capture(make_room(tmp_path / "room", cams, checker_wall="x+", **hi))
    surface = build_surface(cap)
    vs = accumulate(surface, cap)
    npz = {"points": surface.points, "n_views": vs.n_views, "n_dirs": vs.n_dirs,
           "texture": vs.texture, "broken_pts": np.zeros(len(surface.points), bool)}
    labels = label_points(vs.n_views, vs.n_dirs, vs.texture, npz["broken_pts"], 0.02)
    test = cap.frames[-1]
    rgb, gt = tmp_path / "rgb", tmp_path / "gt"
    rgb.mkdir(), gt.mkdir()
    img = np.asarray(Image.open(test.image_path).convert("RGB")).copy()
    Image.fromarray(img).save(gt / f"{test.name}.png")
    img[32:64, 48:80] = 255                                              # a bright square mid-wall
    Image.fromarray(img).save(rgb / f"{test.name}.png")
    t = patch_table(cap, [test.name], rgb, gt, npz, labels)
    assert len(t["error"]) > 10 and set(t) >= {"frame", "pv", "pu", "error", "label", "n_views", "n_dirs", "texture"}
    worst = np.argsort(t["error"])[-4:]
    assert np.all((t["pv"][worst] >= 2) & (t["pv"][worst] <= 3) & (t["pu"][worst] >= 3) & (t["pu"][worst] <= 4))
    assert t["error"][worst].min() > 0.2 and np.median(t["error"]) < 0.01
    assert not np.any(t["label"][worst] == TEXTURELESS)


def _room_with_renders(base, seed):
    import json
    from roomrecon.capcheck.__main__ import main as capcheck
    hi = dict(w=128, h=96, f=100.0)
    cams = ring_cams(n=12) + [(np.array([2.0, 1.25, 1.5]), -90, 0), (np.array([2.0, 1.25, 1.5]), 90, 0)]
    room = make_room(base / "room", cams, checker_wall="x+", **hi)
    tests = ["frame_00013", "frame_00014"]
    (base / "exclude.txt").write_text("\n".join(tests))
    assert capcheck([str(room), "--out", str(base / "cc"), "--exclude", str(base / "exclude.txt")]) == 0
    rng = np.random.default_rng(seed)
    for sub in ("rgb", "gt"):
        (base / sub).mkdir()
    for t in tests:
        img = np.asarray(Image.open(room / "images" / f"{t}.png").convert("RGB")).astype(float)
        Image.fromarray(img.astype(np.uint8)).save(base / "gt" / f"{t}.png")
        noisy = np.clip(img + rng.normal(0, 25, img.shape), 0, 255).astype(np.uint8)
        Image.fromarray(noisy).save(base / "rgb" / f"{t}.png")
    return [base.name, str(room), str(base / "cc" / "points.npz"), str(base / "rgb"), str(base / "gt")]


def test_validate_cli_end_to_end(tmp_path):
    import json
    from roomrecon.capcheck.validate import main
    calib = _room_with_renders(tmp_path / "calib", 0)
    test = _room_with_renders(tmp_path / "test", 1)
    out = tmp_path / "val"
    assert main(["--calib", *calib, "--test", *test, "--out", str(out)]) == 0
    res = json.loads((out / "validation.json").read_text())
    assert set(res["rooms"]) == {"calib", "test"} and res["tau_tex"] >= 0
    assert set(res["criteria"]) == {"1_under_and_textureless_above_ok_in_test_room", "2_texture_adds_auc"}
    assert (out / "error_by_label.png").stat().st_size > 0 and (out / "examples.png").stat().st_size > 0
