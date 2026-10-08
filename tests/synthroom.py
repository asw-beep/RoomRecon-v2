"""Synthetic room captures for the capcheck tests: a box room (y up), cameras inside it, exact
depth by ray casting, grey images with an optional checker wall. Written in the MuSHRoom layout
or the Polycam processed layout, so the tests go through the real readers."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def cam_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """OpenGL camera (looks down -z) turned by yaw about +y, then pitched about its x axis."""
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return ry @ rx


def _slab(o, d, lo, hi):
    """Entry distance of rays o + t d into the box [lo, hi] (inf where missed)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (np.asarray(lo) - o) / d
        t2 = (np.asarray(hi) - o) / d
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 1e-9)
    return np.where(hit, tmin, np.inf)


def render(pos, R, size, pillar, checker_wall, w, h, f):
    """(depth in metres, grey image 0-1) seen by one camera."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dirs_cam = np.stack([(u - w / 2) / f, -(v - h / 2) / f, -np.ones_like(u, float)], -1)
    d = dirs_cam @ R.T
    o = np.asarray(pos, float)
    size = np.asarray(size, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_axis = np.where(d > 0, (size - o) / d, np.where(d < 0, (0 - o) / d, np.inf))
    t_wall = t_axis.min(-1)
    axis = t_axis.argmin(-1)
    t_pil = _slab(o, d, *pillar) if pillar is not None else np.full(t_wall.shape, np.inf)
    t = np.minimum(t_wall, t_pil)
    p = o + t[..., None] * d
    img = np.full(t.shape, 0.5)
    if checker_wall:
        ax = "xyz".index(checker_wall[0])
        on = (t_pil > t_wall) & (axis == ax) & ((d[..., ax] > 0) == (checker_wall[1] == "+"))
        a, b = [p[..., i] for i in range(3) if i != ax]
        cell = (np.floor(a / 0.1) + np.floor(b / 0.1)) % 2
        img = np.where(on, cell, img)
    return t, img          # z-depth == t, because the camera-space ray has z = -1


def make_room(dir, cams, *, size=(4.0, 2.5, 3.0), pillar=None, checker_wall="x+",
              source="mushroom", meta=None, w=64, h=48, f=50.0, corrected=True) -> Path:
    root = Path(dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "depth").mkdir(exist_ok=True)
    frames = []
    for i, (pos, yaw, pitch) in enumerate(cams):
        name = f"frame_{i + 1:05d}"
        R = cam_rotation(yaw, pitch)
        depth, img = render(pos, R, size, pillar, checker_wall, w, h, f)
        Image.fromarray(np.round(depth * 1000).astype(np.uint16)).save(root / "depth" / f"{name}.png")
        Image.fromarray(np.round(img * 255).astype(np.uint8)).convert("RGB").save(root / "images" / f"{name}.png")
        c2w = np.eye(4)
        c2w[:3, :3], c2w[:3, 3] = R, pos
        frames.append({"fl_x": f, "fl_y": f, "cx": w / 2, "cy": h / 2, "w": w, "h": h,
                       "file_path": f"./images/{name}.png", "depth_file_path": f"./depth/{name}.png",
                       "transform_matrix": c2w.tolist()})
    doc = {"camera_model": "OPENCV", "frames": frames}
    if source == "mushroom":
        (root / "transformations.json").write_text(json.dumps(doc))
        return root
    (root / "transforms.json").write_text(json.dumps(doc))
    kf = root / "keyframes"
    for sub in ("cameras", "confidence") + (("corrected_cameras",) if corrected else ()):
        (kf / sub).mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        ts = 1_000_000 + i * 200_000                       # microseconds, 0.2 s apart
        m = {"timestamp": ts, "tracking_segment": 1, "world_mapping_status": "mapped",
             "blur_score": 150.0, "angular_velocity": 0.3, "feature_point_count": 200}
        m.update((meta or [{}] * len(frames))[i])
        ts = m["timestamp"]
        c2w = np.array(fr["transform_matrix"])
        pc = c2w[[1, 2, 0]]                                 # nerfstudio stores Polycam rows as (2, 0, 1)
        m.update({f"t_{r}{c}": float(pc[r, c]) for r in range(3) for c in range(4)})
        (kf / "cameras" / f"{ts}.json").write_text(json.dumps(m))
        Image.fromarray(np.full((h // 4, w // 4), 255, np.uint8)).save(kf / "confidence" / f"{ts}.png")
        if corrected:
            (kf / "corrected_cameras" / f"{ts}.json").write_text(json.dumps({"weakly_connected": False}))
    return root
