"""Exported cameras + mesh -> scene.json. Runs on the GPU machine (needs open3d)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from roomrecon.scene import scene_frame


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
    c2w = load_c2w(d / "transforms_train.json")
    frame = scene_frame(c2w, verts)
    frame.update({"room": a.room, "run_id": a.run_id, "up_axis": "+Y",
                  "units": "metres (approximate)", "camera_count": int(len(c2w)),
                  "mesh_vertex_count": int(len(verts)), "euler_xyz_deg": None})
    (d / "scene.json").write_text(json.dumps(frame, indent=2))
    print(json.dumps({k: frame[k] for k in ("scale", "floor_y", "spawn_position")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
