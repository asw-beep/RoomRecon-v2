# Data — MuSHRoom vr_room (on the GPU machine)

Location: `~/roomrecon/datasets/mushroom/room_datasets/vr_room/` · prepared 2026-10-03.
Source: Zenodo 10230733 (iPhone) and 10222321 (reference mesh), CC BY 4.0. The iPhone
tarball was uploaded from the PC copy (`../Project/storage/projects/m1-mushroom-inspect/`);
its checksum matches the PC file.

```
3f7d577c832bc68a627b49c9cf50a6a9a2a1a4ecabf0d37d16249a393d2aea75  vr_room_iphone.tar.gz
3cf8eb5a6a9260a55b2777a906eae341b8b75fabed27f2c964ac5df99dc27639  vr_room_mesh_pd.tar.gz
```

| Sequence | images | depth | other |
|---|---|---|---|
| `iphone/long_capture` | 422 | 422 | `test.txt` (held-out frames: frame_00011, 00021, 00031, … every 10th), `transformations.json`, `transformations_colmap.json`, `polycam_mesh/`, `polycam_pointcloud.ply` |
| `iphone/short_capture` | 186 | 186 | `transformations.json`, `transformations_colmap.json` |

Reference mesh tarball (`vr_room_mesh_pd.tar.gz`) contents:

- `room_datasets/vr_room/gt_mesh.ply` — laser-scanned reference mesh (used for mesh metrics)
- `room_datasets/vr_room/gt_pd.ply` — reference point cloud
- `room_datasets/vr_room/icp_iphone.json` — alignment of the iPhone sequence to the reference
- `room_datasets/vr_room/icp_kinect.json` — same for the Kinect sequence (not used)

Normals (`normals_from_pretrain/`, Omnidata, `--resolution low`, batch 4 via `scripts/normals.py`): long_capture 844 files, short_capture 372 files = one `.png` + one `.npy` per image (422 and 186 images).
