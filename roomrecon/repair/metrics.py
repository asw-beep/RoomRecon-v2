"""Measures of a tracking repair that need no training (spec §3.4).

C1, alignment: how far a piece's own LiDAR depth lands from the reference piece's surface.
C3, added coverage: how much of the repaired capture's surface only the repaired pieces see.
"""
import numpy as np
from scipy.spatial import cKDTree

from roomrecon.capcheck.capture import Capture, load_confidence, load_depth_m
from roomrecon.capcheck.surface import Surface, backproject, build_surface
from roomrecon.capcheck.views import accumulate
from roomrecon.repair.variants import subset


def alignment_error(cap: Capture, names: list[str], ref: Surface, T: np.ndarray | None = None,
                    every=4, stride=8) -> float:
    """Median over frames of the median distance from the frame's depth points to `ref`."""
    tree = cKDTree(ref.points)
    meds = []
    for f in subset(cap, names).frames[::every]:
        c2w = f.c2w if T is None else np.asarray(T) @ f.c2w
        conf = load_confidence(f)
        p = backproject(load_depth_m(f), f.K, c2w, stride=stride, mask=None if conf is None else conf == 255)
        if len(p):
            meds.append(float(np.median(tree.query(p)[0])))
    return float(np.median(meds)) if meds else float("nan")


def added_coverage(cap: Capture, reference: list[str], pieces: list[str]) -> float:
    """Share of the surface that `pieces` frames see and no `reference` frame sees."""
    sub = subset(cap, list(reference) + list(pieces))
    surf = build_surface(sub)
    if len(surf.points) == 0:
        return 0.0
    is_piece = np.array([f.name in set(pieces) for f in sub.frames])
    n_ref = accumulate(surf, sub, frames_mask=~is_piece).n_views
    n_piece = accumulate(surf, sub, frames_mask=is_piece).n_views
    return float(np.mean((n_piece > 0) & (n_ref == 0)))
