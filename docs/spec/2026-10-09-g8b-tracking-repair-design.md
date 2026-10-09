# G8 (new): detect and repair broken phone tracking before training

Status: design approved in conversation 2026-10-09; this written spec awaits review.
Replaces the capture check as the G8 novelty attempt; the capture check
(`2026-10-08-g8-capture-check-design.md`, branch `g8-capture-check`) is the fallback.
Decision trail and prior work: `docs/report-notes.md`.

## 1. Claim

A phone capture whose tracking broke (ARKit restarted or relocalised mid-scan) can be
**detected and repaired before training**, using only the phone's own data, and the repaired
capture gives a measurably better walkable room than training on it as exported or throwing the
broken part away.

Our capture `sejal-cap3` (Polycam raw export, 491 frames) is such a capture: Polycam accepted
and processed it, but it holds two sweeps of the same room in different, inconsistent frames
(frames 1-117 and 125-491), with 7 transition frames of segment restarts and pose jumps
(0.2-4.3 m/s, 374-1146°/s between keyframes). The capture check's tracking part already detects
this (G8 capture check, verdict BROKEN TRACKING).

Prior work (searched 2026-10-09): Polycam Scenes merge captures only by hand-placing meshes and
cannot take splats; Scaniverse has no scan merging; ARKit has no API reporting pose jumps; LiDAR
scan alignment by ICP is standard in SLAM (KISS-ICP, GSFusion). No tool repairs an exported phone
capture's tracking before splat training.

## 2. Where it runs

Repair and measurements: GPU machine CPUs (data lives there), pixi env, nothing installed.
Tests: PC, synthetic rooms. Training: the A100 when IT restores GPU access in the container
(on 2026-10-09 `nvidia-smi` failed with NVML "Unknown Error" and torch saw no CUDA); otherwise
Kaggle T4 (`scripts/kaggle/own_room_kaggle.sh`). All three variants train on the same machine
with the same settings.

## 3. Method

### 3.1 Pieces
Reuse `roomrecon.capcheck.tracking.check_tracking`: split the frame sequence at every segment
change and every pose jump. The largest piece is the **reference** (cap3: frames 125-491). Other
pieces with at least 20 frames are **repaired**; shorter ones are dropped (cap3: the 7 transition
frames).

### 3.2 Registration (rigid, gravity-aware)
Per piece, a LiDAR point cloud: every frame's depth back-projected (stride 8, confidence 255),
merged in 5 cm voxels, normals from 16 neighbours (reusing `roomrecon.capcheck.surface`).
ARKit keeps each tracking session gravity-aligned, so a piece differs from the reference by a turn
about the vertical axis plus a shift. The vertical axis is `roomrecon.scene.estimate_up` of the
reference piece's cameras.

Search: 36 starting turns (every 10°) about the vertical, each with the shift that matches the two
clouds' centroids; each start refined by point-to-plane ICP (full 6-DoF, correspondence distance
0.5 → 0.2 → 0.05 m, up to 30 iterations each, our numpy/scipy implementation). The start with the
highest **fitness** (share of piece points within 5 cm of the reference after alignment) wins;
ties by lower RMSE.

Accept the transform only if fitness ≥ 0.3 and the solved tilt away from the vertical axis
< 3°; otherwise the piece is dropped and the failure reported. The accepted transform `T` is
applied to the piece's camera-to-world poses: `c2w' = T · c2w`.

### 3.3 Variants (three folders, same images, depth and normals)
- **raw**: all 491 frames, poses as exported.
- **drop**: reference piece only.
- **repaired**: reference piece plus repaired pieces with corrected poses (cap3: 484 frames).

All three `transforms.json` carry the same `test_filenames`: every 8th frame of the reference
piece, counting from its first frame (positions 0, 8, 16, ...; cap3: `frame_00125`, `frame_00133`, ...). DN-Splatter's `normal-nerfstudio` dataparser uses explicit split lists when present
(`normal_nerfstudio.py:215-232`), so the test frames are identical across variants; the
remaining frames of each variant are `train_filenames`. Images, depth and omnidata normals
(computed once for cap3) are shared by symlink; each variant gets its own seed point cloud from
its own poses (`pointcloud_utils.py --num-points 300000`).

### 3.4 Measurements
- **Alignment (C1)**: for every 4th frame of each repaired piece, back-project its depth (stride
  8) and take the median distance to the reference piece's surface points; report the median over
  frames, with the raw poses and with the repaired poses.
- **Coverage (C3)**: build the repaired variant's surface (capture check `build_surface`); the
  share of its points seen (depth-checked, `views.visible`) by repaired-piece frames and by no
  reference frame is the surface the repair adds.
- **Rendering (C2, C4)**: `ns-eval` PSNR / SSIM / LPIPS on the common test frames, final
  checkpoint, per variant.
- **Walk**: export (`export_room.sh`), collision (`collision_room.ps1`), SuperSplat walk test of
  raw and repaired.

## 4. Pass / fail (fixed before any run)

| | Criterion |
|---|---|
| C1 | Repaired-piece depth lands within **3 cm** (median) of the reference surface after repair, versus metres before |
| C2 | Repaired beats raw by **≥ 1.0 dB PSNR** on the common test frames, with SSIM higher and LPIPS lower |
| C3 | The repair adds **≥ 5 %** of the repaired variant's surface that only repaired-piece frames see |
| C4 | Repaired is no more than **0.3 dB PSNR** below drop on the common test frames |

The result is reported as measured, pass or fail. If the A100 and Kaggle are both unavailable,
C2 and C4 are reported as not run.

## 5. Training settings

`scripts/own_room.sh` (DN-Splatter, lite settings, as G4), the same number of steps for all three
variants: 30k on the A100 (stop-split 10k), or 15k on a Kaggle T4 (stop-split 5k) if that is
where they run. Machine and steps are recorded with the results.

## 6. Code

New package `roomrecon/repair/`: `register.py` (gravity-aware ICP, §3.2), `variants.py`
(pieces, corrected poses, the three folders with split lists, §3.1/3.3), `metrics.py` (C1, C3),
`__main__.py` (CLI: `python -m roomrecon.repair CAPTURE --out DIR` → `raw/`, `drop/`,
`repaired/`, `repair.json`). Shell wrapper `scripts/repair_room.sh` (GPU machine; also runs
`pointcloud_utils.py` per variant). Tests (PC, pytest) on synthetic rooms from
`tests/synthroom.py`: two pieces with a known turn and shift are recovered within 1 cm and 0.5°;
a piece with no overlap is rejected; variants share `test_filenames`; drop leaves out the
repaired piece; C1 shrinks after repair; C3 > 0 when the repaired piece sees extra surface.

## 7. Not doing

Pose-graph optimisation across many pieces, photometric pose refinement, scale correction
(ARKit is metric), other capture apps, changes to the `vr_room` demo.
