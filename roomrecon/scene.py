"""Where is up, where is the floor, how big is the room, where do you spawn.

Inputs are in the exported (nerfstudio) frame: camera-to-world matrices in the OpenGL
convention (camera +Y up, -Z forward) and mesh/splat vertices. People hold phones upright,
so the mean camera +Y is gravity's up. The scene is rotated so up is +Y (the viewer's
convention) and scaled so the median camera height above the floor is a typical handheld
eye height; the factor is recorded, and a factor near 1 means the poses were already metric.
"""
import numpy as np


def estimate_up(c2w: np.ndarray) -> np.ndarray:
    u = c2w[:, :3, 1].mean(axis=0)
    n = np.linalg.norm(u)
    if n < 0.5:
        raise ValueError(f"camera up vectors disagree (mean norm {n:.2f}); cannot estimate up")
    return u / n


def rotation_to_y_up(up: np.ndarray) -> np.ndarray:
    a = up / np.linalg.norm(up)
    b = np.array([0.0, 1.0, 0.0])
    v = np.cross(a, b)
    c = float(a @ b)
    if np.isclose(c, -1.0):                      # up points straight down: turn 180° about X
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def floor_height(vertex_y: np.ndarray, camera_y: np.ndarray, bins: int = 200) -> float:
    below = vertex_y[vertex_y < camera_y.min()]
    if below.size == 0:
        raise ValueError("no geometry below the cameras; cannot find the floor")
    hist, edges = np.histogram(below, bins=bins)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def scene_frame(c2w: np.ndarray, vertices: np.ndarray, eye_height_m: float = 1.45) -> dict:
    R = rotation_to_y_up(estimate_up(c2w))
    cam = (R @ c2w[:, :3, 3].T).T
    verts = (R @ vertices.T).T
    floor_raw = floor_height(verts[:, 1], cam[:, 1])
    h_raw = float(np.median(cam[:, 1] - floor_raw))
    if h_raw <= 0:
        raise ValueError("cameras are not above the floor estimate")
    s = eye_height_m / h_raw
    centre = np.median(cam, axis=0)
    k = int(np.argmin(np.linalg.norm(cam - centre, axis=1)))
    fwd = R @ (-c2w[k, :3, 2])
    fwd[1] = 0.0
    fwd = fwd / np.linalg.norm(fwd)
    spawn = cam[k] * s
    spawn[1] = floor_raw * s + eye_height_m
    return {"rotation": R.tolist(), "scale": s, "floor_y": floor_raw * s,
            "spawn_position": spawn.tolist(), "spawn_forward": fwd.tolist(),
            "scale_source": f"median camera height normalised to {eye_height_m} m",
            "median_camera_height_raw": h_raw}

