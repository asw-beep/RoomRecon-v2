"""Read a capture folder into one `Capture`: per-frame intrinsics, camera-to-world (OpenGL),
image, LiDAR depth and, for Polycam, the phone's per-frame tracking data.

Two layouts: `ns-process-data polycam` output (`transforms.json`, plus Polycam's raw
`keyframes/` when present) and MuSHRoom (`transformations.json`). Polycam keyframes are
matched to processed frames in order: frames by name, keyframes by timestamp (their file
names), which is how `ns-process-data` numbers them.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Frame:
    name: str
    K: np.ndarray
    width: int
    height: int
    c2w: np.ndarray
    image_path: Path
    depth_path: Path
    conf_path: Path | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Capture:
    root: Path
    source: str
    frames: list[Frame]
    has_tracking: bool


def _c2w(m) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    if m.shape == (3, 4):
        m = np.vstack([m, [0.0, 0.0, 0.0, 1.0]])
    if m.shape != (4, 4):
        raise ValueError(f"unexpected camera matrix shape {m.shape}")
    return m


def _frame(root: Path, f: dict) -> Frame:
    K = np.array([[f["fl_x"], 0, f["cx"]], [0, f["fl_y"], f["cy"]], [0, 0, 1]], dtype=float)
    return Frame(name=Path(f["file_path"]).stem, K=K, width=int(f["w"]), height=int(f["h"]),
                 c2w=_c2w(f["transform_matrix"]), image_path=root / f["file_path"],
                 depth_path=root / f["depth_file_path"])


def _keyframe_order(p: Path):
    return (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem)


def load_capture(root: Path, exclude: frozenset[str] = frozenset()) -> Capture:
    root = Path(root)
    if (root / "transforms.json").exists():
        source, doc = "polycam", json.loads((root / "transforms.json").read_text())
    elif (root / "transformations.json").exists():
        source, doc = "mushroom", json.loads((root / "transformations.json").read_text())
    else:
        raise FileNotFoundError(f"no transforms.json or transformations.json in {root}")
    frames = sorted((_frame(root, f) for f in doc["frames"]), key=lambda fr: fr.name)

    has_tracking = False
    cams_dir = root / "keyframes" / "cameras"
    if source == "polycam" and cams_dir.is_dir():
        cams = sorted(cams_dir.glob("*.json"), key=_keyframe_order)
        if len(cams) != len(frames):
            raise ValueError(f"{len(frames)} processed frames but {len(cams)} keyframes in "
                             f"{cams_dir}: cannot match them in order")
        for fr, cam in zip(frames, cams):
            fr.meta = json.loads(cam.read_text())
            corrected = root / "keyframes" / "corrected_cameras" / cam.name
            if corrected.exists():
                fr.meta["weakly_connected"] = json.loads(corrected.read_text()).get("weakly_connected")
            conf = root / "keyframes" / "confidence" / f"{cam.stem}.png"
            fr.conf_path = conf if conf.exists() else None
        has_tracking = True
    frames = [fr for fr in frames if fr.name not in exclude]
    return Capture(root=root, source=source, frames=frames, has_tracking=has_tracking)


def read_exclude(path: Path) -> frozenset[str]:
    return frozenset(s.strip() for s in Path(path).read_text().splitlines() if s.strip())


def _resize_nearest(a: np.ndarray, h: int, w: int) -> np.ndarray:
    if a.shape[:2] == (h, w):
        return a
    rows = (np.arange(h) * a.shape[0] / h).astype(int)
    cols = (np.arange(w) * a.shape[1] / w).astype(int)
    return a[rows][:, cols]


def load_depth_m(f: Frame) -> np.ndarray:
    """Depth in metres (float32, frame size); 0 where the phone measured nothing."""
    d = np.asarray(Image.open(f.depth_path)).astype(np.float32) / 1000.0
    return _resize_nearest(d, f.height, f.width)


def load_confidence(f: Frame) -> np.ndarray | None:
    if f.conf_path is None:
        return None
    return _resize_nearest(np.asarray(Image.open(f.conf_path).convert("L")), f.height, f.width)


def load_gray(f: Frame) -> np.ndarray:
    g = np.asarray(Image.open(f.image_path).convert("L")).astype(np.float32) / 255.0
    return _resize_nearest(g, f.height, f.width)
