# G4 — own room `sejal-cap2` trained, exported and walkable

Machine: GPU machine, A100-PCIE-40GB MIG 7g.40gb, shared (another tenant held ~33.7 GB at start).
Date: 2026-10-08. Run id `dns-30k-lite`, repo `4dc9f74` (training), DN-Splatter `97588b4`.
`scene.json` was made after two fixes, `e3fe046` and `c955a69` (below).
Evidence: `g4-sejal-cap2-run.json`, `g4-sejal-cap2-attempts.log`, `g4-sejal-cap2-eval-curve.csv`
(every TensorBoard scalar), `g4-sejal-cap2-ns_eval.json`, `g4-sejal-cap2-scene.json`,
`g4-renders/` (ground truth left, render right).

## Capture

- Sejal Satheesh, iPhone Pro (LiDAR), Polycam raw export (ARKit poses + LiDAR depth),
  2026-10-07. Three captures of the same room:
  - cap1: 32 frames, pipeline test only.
  - **cap2: 257 frames, one tracking segment, full coverage. Used.**
  - cap3: 491 frames, three tracking segments with ~2 m pose jumps (in Polycam's corrected
    poses too). Not used.
- `ns-process-data polycam --use-depth` (corrected images and cameras), omnidata normals,
  300k seed points from the LiDAR depth (`pointcloud_utils.py`).
- Phone poses kept as they are (spec §5.1): `normal-nerfstudio --orientation-method none
  --center-method none --auto-scale-poses False`. Every 8th frame held out: 224 train,
  33 test.

## Kaggle attempt (lost)

The A100 was full on the night of 2026-10-07, so cap1 and cap2 were first sent to a Kaggle T4
(`scripts/kaggle/`). Both sessions were still running at Kaggle's 12 h limit and were stopped
with no output saved. Most likely cause, not confirmed: the VRAM sampler outlived the training
script and held the notebook cell's output open. Fixed in `d4b1614`. The steps now live in
`scripts/own_room.sh`, which runs on either machine.

## Run (GPU machine)

`scripts/own_room.sh sejal-cap2 dns-30k-lite <data> 30000` (lite settings, stop-split 10k).

- Duration **5960 s (99 min)**, exit 0, **two attempts**. Attempt 1 crashed at step ~10,400
  with `CUDA error: an illegal memory access` in DN-Splatter's rasterisation
  (`dn_model.py:587`) while the card was nearly full. The script resumed from the step-10000
  checkpoint on its own and finished.
- Free memory at start 6755 MiB. Whole-card use 33692 → max **38948 MiB**, so our peak was about
  **5.3 GB**, assuming the other tenant stayed constant.
- Gaussians: 300k seed → peak **1.20M** at step 9000 → **746k** at the end.
- Checkpoints every 1k are kept on the GPU machine
  (`~/roomrecon/scenes/sejal-cap2/dns-30k-lite/train/`).

## Quality (DN-Splatter eval, all 33 held-out frames)

| Step | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 1000 | 18.78 | | |
| 10000 | 21.38 | 0.838 | 0.184 |
| 20000 | 22.21 | 0.852 | 0.170 |
| **30000 (final)** | **22.18** | **0.853** | **0.170** |

Final depth: abs_rel 0.017, RMSE 0.061 m, δ<1.25 0.989.

**No collapse.** PSNR plateaus after ~15k. The test frames are interleaved frames of the same
capture (like `vr_room`'s "within" set), but a different room, so the numbers are not directly
comparable with `vr_room`'s 26.3 dB.

**Renders** (`g4-renders/`): furniture, curtains, door and mattresses are sharp. The plain white
walls, ceiling and floor are smeary and streaky, worst close to the camera.

## Export and scene

- Splat: 746,156 Gaussians, 185 MB. Mesh: `gs-mesh o3dtsdf`, 2 cm voxels.
- Two `scene.json` bugs showed up on this room and are fixed:
  1. **Up direction** (`e3fe046`). Polycam saved every frame sideways (phone held upright,
     ceiling on the left edge of each frame). Camera +Y averaged to nothing (norm 0.17); camera
     −X is up (norm 0.87). `estimate_up` now tries +Y, then −X.
  2. **Floor** (`c955a69`). The densest level below the cameras was the bed tops, so the room was
     scaled **1.68×**. The floor is now the lowest dense level. Scale **1.038**, median camera
     height 1.40 m (ARKit poses are metric, so ~1 is expected).
     Checked on `vr_room` against MuSHRoom's laser scan (same frame as the G3 mesh eval): the
     old rule put the floor 13 cm too high, the new one is within ~1.5 cm. **The `vr_room`
     demo was deliberately left as it is** (walk-tested and released).
- Collision: `splat-transform` v3.10.0, 5 cm voxels, grid about 4.4 × 4.0 × 5.4 m.

## Walk test (PC, SuperSplat Viewer 1.37.0 with the RoomRecon patch)

Done by Aswin on 2026-10-08. Verdict: the output is good and satisfactory for us. The individual
G2 checks (spawn, walls, floor, alignment) and the FPS were not itemised or recorded.

## Backup demo

`E:\roomrecon-scenes\demo-sejal-cap2` (+ `.zip`, 168 MB), made with
`scripts/make_demo_bundle.ps1`. It is separate from `demo-vr_room`, which is unchanged.

## Not done

- No screen recording of the walk (same as G2).
- FPS in the viewer not recorded.
- Comparison with Polycam's own on-phone reconstruction of this room (G7).
