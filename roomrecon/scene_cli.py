"""Exported cameras + mesh -> scene.json. Runs on the GPU machine (needs open3d)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from roomrecon.scene import euler_xyz_deg, scene_frame, to_playcanvas


def build_scene(c2w_ply: np.ndarray, verts_ply: np.ndarray, *, room: str, run_id: str) -> dict:
    """scene.json content, in PlayCanvas' frame (what splat-transform and the viewer use)."""
    c2w, verts = to_playcanvas(c2w_ply, verts_ply)
    frame = scene_frame(c2w, verts)
    frame.update({"room": room, "run_id": run_id, "up_axis": "+Y",
                  "frame": "playcanvas (PLY rotated 180 deg about Z)",
                  "units": "metres (approximate)", "camera_count": int(len(c2w)),
                  "mesh_vertex_count": int(len(verts)),
                  "euler_xyz_deg": euler_xyz_deg(np.asarray(frame["rotation"]))})
    return frame


def viewer_settings(scene: dict, fov: float = 75.0) -> dict:
    """SuperSplat Viewer settings.json (v2) that starts the camera at the spawn point."""
    p = [float(v) for v in scene["spawn_position"]]
    f = scene["spawn_forward"]
    target = [p[i] + float(f[i]) for i in range(3)]
    return {"version": 2, "tonemapping": "linear", "highPrecisionRendering": False,
            "background": {"color": [0, 0, 0]}, "animTracks": [], "annotations": [],
            "cameras": [{"initial": {"position": p, "target": target, "fov": fov}}],
            "startMode": "default"}


def load_c2w(transforms_json: Path) -> np.ndarray:
    """(N,4,4) camera-to-world from `ns-export cameras` output (list or {"frames": [...]},
    key "transform" or "transform_matrix", 3x4 or 4x4)."""
    data = json.loads(Path(transforms_json).read_text())
    frames = data["frames"] if isinstance(data, dict) else data
    mats = []
    for f in frames:
        m = np.array(f.get("transform", f.get("transform_matrix")), dtype=float)
        if m.shape == (3, 4):
            m = np.vstack([m, [0.0, 0.0, 0.0, 1.0]])
        if m.shape != (4, 4):
            raise ValueError(f"unexpected camera matrix shape {m.shape}")
        mats.append(m)
    return np.stack(mats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--room", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    d = Path(a.export_dir)
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(d / "mesh.ply"))
    verts = np.asarray(mesh.vertices)
    frame = build_scene(load_c2w(d / "transforms_train.json"), verts,
                        room=a.room, run_id=a.run_id)
    (d / "scene.json").write_text(json.dumps(frame, indent=2))
    (d / "settings.json").write_text(json.dumps(viewer_settings(frame), indent=2))
    print(json.dumps({k: frame[k] for k in ("scale", "floor_y", "spawn_position")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
