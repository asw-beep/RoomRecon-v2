# G3 — `vr_room` quality: DN-Splatter vs plain splats (GPU machine)

Machine: GPU machine, A100-PCIE-40GB MIG 7g.40gb, shared (another tenant held 33248 MiB).
Dates: training 2026-10-05, evaluation 2026-10-05/06. Room: MuSHRoom `vr_room`, iPhone mode.
Evidence: `g3-dns-ns_eval.json`, `g3-dns-mesh.json`, `g3-plain-ns_eval.json`, `g3-plain-mesh.json`,
`g3-plain-run.json`, `g3-plain-eval-curve.csv`, `g3-plain-attempts.log`; DN-Splatter's run
records are the G1 files (`g1-vr_room-run.json`, `g1-vr_room-eval-curve.csv`).

**DN-Splatter's depth and normal priors cost about 0.5 dB of image quality near the training
walk and buy 3.3× more accurate geometry: the mesh is 1.8 cm from the laser scan instead of
5.9 cm (F-score at 5 cm 0.93 vs 0.64).** For a walkable room, whose collision is built from
that geometry, DN-Splatter is the right choice.

## Results (final checkpoint, step 29999)

| Run | PSNR within | SSIM within | LPIPS within | PSNR with | SSIM with | LPIPS with | Depth abs. rel. within / with | Mesh acc (cm) | Mesh comp (cm) | Chamfer-L1 (cm) | Normal consistency | F-score @5 cm | Train time | GPU used max (MiB, whole card) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DN-Splatter `dns-30k-lite` | 26.32 | 0.862 | 0.143 | **23.50** | 0.812 | 0.192 | **0.016 / 0.022** | **1.76** | **2.29** | **2.03** | **0.896** | **0.926** | 5215 s | 38462 |
| Plain splats `plain-30k-lite2` | **26.84** | **0.875** | **0.124** | 23.24 | **0.818** | 0.192 | 0.056 / 0.083 | 5.85 | 5.86 | 5.86 | 0.809 | 0.638 | 4747 s | 38152 |

- **Image metrics** from `ns-eval` (DN-Splatter pipeline). *Within* = the 42 held-out frames of
  the long capture (`long_capture/test.txt`); *with* = the frames of MuSHRoom's separate short
  capture of the same room, viewpoints the training walk did not take.
- **Depth** is compared with the iPhone LiDAR depth of the test frames. DN-Splatter trains on that
  depth, so this measure favours it; the mesh metrics below do not.
- **Mesh** = `gs-mesh o3dtsdf`, voxel 0.02 m, from each run's final checkpoint, against the
  laser-scanned `gt_mesh.ply` with DN-Splatter's MuSHRoom protocol
  (`eval_mesh_mushroom_vis_cull.py --device iphone`: ICP alignment from `icp_iphone.json`, both
  meshes cut and culled to what the capture saw). Accuracy = our surface to the scan,
  completeness = the scan to our surface (mean distances); F-score threshold 5 cm.
  No frame correction was needed: training kept the dataset's frame
  (`orientation_method none`, `center_method none`, `auto_scale_poses false`).
- Final Gaussians: DN-Splatter 908,249; plain 881,612 (exported splats).
- Run time of the mesh evaluation: about 2 h 10 min (DN-Splatter) and 2 h 50 min (plain), single
  CPU core each, run in parallel for part of the time.

## Collapse check (Task 7 Step 2 rule)

| Run | Best within PSNR (step) | Last eval within PSNR (step) |
|---|---|---|
| DN-Splatter | 26.31 (27000) | 26.26 (29000) |
| Plain splats | 26.84 (29000) | 26.84 (29000) |

Neither run collapsed (v1's ARKitScenes run fell 22.8 → 13.7 dB).

## How the two runs differ, and what is not standard

- **Same for both:** data and splits, `train_room_lite.sh` low-memory settings (300k seed points
  instead of DN-Splatter's 1M default, splitting stops at step 10000, 30000 steps), spec §5.1
  checkpoint/eval settings, the same evaluation code.
- **Plain splats** = DN-Splatter's model with every prior off (depth, depth-smoothness, normal,
  normal-TV losses), 3D instead of flattened 2D Gaussians, and splatfacto's `max-gauss-ratio` 10
  (`PROFILE=plain`). It is not nerfstudio's `splatfacto` itself: with the MuSHRoom dataparser,
  splatfacto's pipeline would average the *within* and *with* test sets into one number
  (`mushroom_dataparser.py:323`). One splatfacto setting could not be kept: its downscaled
  warm-up (`num-downscales 2`) crashes DN-Splatter's normals-from-depth step
  (`dn_model.py:589`), so plain splats train at full resolution from the start, as DN-Splatter does.
- Both runs use fewer seed points than DN-Splatter's default; the effect of that on either
  method is not measured.

## Targets for later rooms (set from this one room)

For an own room captured with LiDAR and trained with DN-Splatter (`PROFILE=dn`):

- within PSNR ≥ 25 dB on held-out frames,
- depth abs. rel. ≤ 0.03 against the phone's LiDAR depth,
- the G2 walk test passes (start inside, walls stop at the visible surface, floor holds),
- where a reference scan exists (MuSHRoom rooms): mesh F-score at 5 cm ≥ 0.85.

Walk test for DN-Splatter: see `g2-walk-test.md`.
