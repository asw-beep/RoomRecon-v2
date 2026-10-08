"""Tracking health of a Polycam capture from the phone's own per-frame data: tracking
segments (ARKit restarted), pose jumps (relocalisation), limited mapping and frames Polycam's
pose graph marks as weakly connected. Polycam timestamps are microseconds.

Frames outside the largest unbroken piece of the capture (pieces split at every segment
change and every jump) are "broken": their poses cannot be trusted against the rest.
"""
from dataclasses import dataclass

import numpy as np

from roomrecon.capcheck.capture import Capture

MAX_SPEED_MPS = 1.5
MAX_ROT_DPS = 180.0


@dataclass
class Tracking:
    available: bool
    segments: list[int]
    segment_changes: list[str]
    jumps: list[dict]
    limited: list[str]
    weakly_connected: list[str]
    bad_timestamps: list[str]
    broken: frozenset[str]
    blur_score: np.ndarray | None
    angular_velocity: np.ndarray | None


def _rotation_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def check_tracking(cap: Capture) -> Tracking:
    if not cap.has_tracking:
        return Tracking(False, [], [], [], [], [], [], frozenset(), None, None)
    fr = cap.frames
    names = [f.name for f in fr]
    seg = [int(f.meta.get("tracking_segment", 0)) for f in fr]
    cuts, changes, jumps, bad = set(), [], [], []
    for i in range(1, len(fr)):
        if seg[i] != seg[i - 1]:
            changes.append(names[i])
            cuts.add(i)
        dt = (fr[i].meta.get("timestamp", 0) - fr[i - 1].meta.get("timestamp", 0)) / 1e6
        if dt <= 0:
            bad.append(names[i])
            continue
        speed = float(np.linalg.norm(fr[i].c2w[:3, 3] - fr[i - 1].c2w[:3, 3])) / dt
        rot = _rotation_deg(fr[i - 1].c2w[:3, :3], fr[i].c2w[:3, :3]) / dt
        if speed > MAX_SPEED_MPS or rot > MAX_ROT_DPS:
            jumps.append({"frame": names[i], "speed_mps": speed, "rot_dps": rot})
            cuts.add(i)
    bounds = [0, *sorted(cuts), len(fr)]
    pieces = [range(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    keep = set(max(pieces, key=len)) if pieces else set()
    broken = frozenset(names[i] for i in range(len(fr)) if i not in keep)

    def series(key):
        return np.array([float(f.meta.get(key, np.nan)) for f in fr])

    return Tracking(
        available=True, segments=seg, segment_changes=changes, jumps=jumps,
        limited=[f.name for f in fr if f.meta.get("world_mapping_status") == "limited"],
        weakly_connected=[f.name for f in fr if f.meta.get("weakly_connected") is True],
        bad_timestamps=bad, broken=broken,
        blur_score=series("blur_score"), angular_velocity=series("angular_velocity"))
