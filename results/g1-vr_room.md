# G1 — `vr_room` trained with DN-Splatter (30k steps, low-memory settings)

Machine: GPU machine, A100-PCIE-40GB MIG 7g.40gb, shared (another tenant held 33248 MiB).
Room: MuSHRoom `vr_room`, iPhone mode (ARKit poses + LiDAR depth, mono normals).
Date: 2026-10-05. Run id `dns-30k-lite`, repo `13184b1`, DN-Splatter `97588b4`.
Evidence files: `g1-vr_room-run.json`, `g1-vr_room-eval-curve.csv` (every TensorBoard scalar),
`g1-vr_room-attempts.log`.

## Why "lite"

The first full run (`dns-30k`, 2026-10-04, `scripts/train_room.sh`) ran out of GPU memory at
step ~1000 when the other tenant left only ~3 GB free. The cause: DN-Splatter's MuSHRoom
dataparser seeds **1,000,000** Gaussians from the iPhone point cloud by default
(`num_init_points`). The images are only 738×994, so lowering the resolution would not help.

`scripts/train_room_lite.sh` keeps every spec §5.1 setting (checkpoint every 1k, keep all,
eval all images every 1k, CPU image cache) and changes only:

| Setting | Default | Used |
|---|---|---|
| `mushroom --num-init-points` | 1,000,000 | **300,000** |
| `--pipeline.model.stop-split-at` | 15000 | **10000** |
| `PYTORCH_CUDA_ALLOC_CONF` | — | `expandable_segments:True` |
| Restart after a crash | manual | automatic, from the newest checkpoint |

**This is a deviation from DN-Splatter's defaults; its quality cost is not measured.** A
default run needs a GPU with more free memory (`train_room.sh`, estimated ≥ ~15 GB).

## Run

- Duration **5215 s (87 min)**, exit 0, **one attempt** (no crash, no restart).
- Free memory at start 7199 MiB. Whole-card use 33248 → max **38462 MiB**, so our peak was
  about **5.2 GB**, assuming the other tenant stayed constant.
- Gaussians: 300k seed → peak **1.47M** at step 9000 → **0.91M** at the end.
- Checkpoints 1000 … 29000 and 29999 are kept on the GPU machine
  (`~/roomrecon/scenes/vr_room/dns-30k-lite/train/vr_room/dn-splatter/dns-30k-lite/nerfstudio_models/`).

## Quality (DN-Splatter eval, all test images)

`within` = held-out frames of the training capture; `with` = frames from MuSHRoom's other
capture of the same room (the harder test).

| Step | Gaussians | within PSNR | within SSIM | within LPIPS | with PSNR | with SSIM | with LPIPS |
|---|---|---|---|---|---|---|---|
| 1000 | 439k | 20.62 | 0.731 | 0.416 | 19.82 | 0.720 | 0.432 |
| 10000 | 1.32M | 25.00 | 0.839 | 0.179 | | | |
| 20000 | 983k | 26.23 | 0.861 | 0.148 | | | |
| **27000 (best)** | 923k | **26.31** | **0.862** | **0.143** | 23.54 | 0.813 | 0.193 |
| 29000 (last eval) | 913k | 26.26 | 0.861 | 0.144 | 23.44 | 0.812 | 0.193 |

Depth at step 29000 (within): abs_rel 0.016, RMSE 0.096 m, δ<1.25 0.993.
The last evaluation ran at step 29000; the final step 29999 was not evaluated.

**No collapse.** PSNR rises steadily and plateaus after ~20k; the last eval is 0.05 dB below
the best (noise), unlike v1's ARKitScenes run (22.8 → 13.7 dB). Tasks 9–11 can use the final
checkpoint.

## Not yet checked

- Renders have not been looked at, only metrics (v1 lesson: look at actual renders).
- 0.91M Gaussians is about twice what v1 found comfortable in a browser (~400–500k). The
  SuperSplat walk test (Task 10) will measure whether it needs pruning.
