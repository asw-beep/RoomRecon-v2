# G2 — Export of `vr_room` (splat, cameras, mesh, `scene.json`) and a first look at renders

Machine: GPU machine (A100 MIG, shared). Date: 2026-10-05. Run `dns-30k-lite`, final
checkpoint `step-000029999`. Command: `scripts/export_room.sh vr_room dns-30k-lite` (voxel 0.02).
Files on the PC (not in git, too large): `E:\roomrecon-scenes\vr_room\dns-30k-lite\`.

## Outputs

| File | Size | Content |
|---|---|---|
| `splat.ply` | 225,247,283 B | **908,249** Gaussians (`ns-export gaussian-splat`; 19 of 908,268 dropped for NaN/Inf scale) |
| `mesh.ply` | 27,036,373 B | 536,059 vertices, 966,347 faces (`gs-mesh o3dtsdf`, voxel 0.02, 15,927 connected clusters) |
| `transforms_train.json` / `_eval.json` | 264 kB / 160 kB | 376 train cameras (list of `{file_path, transform}`) |
| `scene.json` | | copied to `g2-vr_room-scene.json` |
| `settings.json` | | SuperSplat Viewer start camera at the spawn point |

`scene.json`: `scale` **1.038**, `median_camera_height_raw` **1.397** (inside 0.5–3, so ARKit
poses are close to metric, as expected), `floor_y` −1.383, `euler_xyz_deg` (−86.6, 0.3, 0.3)
(the dataset frame is Z-up), spawn at (3.71, 0.07, −0.87) facing (−0.85, 0, −0.52).

## Fixes needed on the way (in git)

| Problem | Fix |
|---|---|
| `gs-mesh` and `ns-export` crash on import: `GLIBCXX_3.4.29 not found` (they import open3d before torch, which loads the system's older libstdc++) | `export_room.sh` preloads the env's own `libstdc++.so.6` |
| `ns-export cameras`: `AssertionError camera.metadata` (DN-Splatter's MuSHRoom parser sets no camera metadata) | `setup_gpu.sh` patches nerfstudio to use the dataset pose when there is no metadata; exact, because the run's camera optimizer is `mode: off` (checked in `config.yml`) |

## Renders (render left, photo right; `g2-renders/`)

Rendered with `ns-render dataset --split test` (456 images: render + ground truth, both test
sets). Four inspected by eye:

- **Held-out frames of the training capture** (`long_capture_*`): close to the photo. Signs,
  pipes, the rug pattern, the TV and the cabinets are sharp; thin parts (coat-rack arms,
  keyboard) are slightly soft.
- **Frames from the other capture** (`short_capture_*`, viewpoints the training walk did not
  take): thin objects partly disappear (coat-rack arms, the tap), and surfaces seen close up
  are streaky (the wall beside the sofa, the blanket). This matches the lower `with` PSNR
  (23.4 vs 26.3 dB, see `g1-vr_room.md`).

Expectation for the walk test: views near the capture path look photoreal; walking close to
thin objects or into corners the capture did not cover shows floaters and streaks.
