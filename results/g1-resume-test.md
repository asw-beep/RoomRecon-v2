# G1 — kill-and-resume test (smoke run, 3000 steps)

Machine: GPU machine, A100-PCIE-40GB MIG 7g.40gb, shared (another tenant held 34456 MiB).
Room: MuSHRoom `vr_room`, DN-Splatter (`scripts/train_room.sh`, settings of spec §5.1).
Date: 2026-10-03/04. Evidence files: `g1-smoke-run.json`, `g1-smoke-resumed-run.json`,
`g1-smoke-eval-curve.csv`.

## What happened

1. **Run `smoke`** trained from scratch, saved `step-000001000.ckpt`, ran the step-1000
   full evaluation, and was killed with `kill -9` at step ~1250 (as an OOM kill would).
   `run.json`: `status: failed`, `exit_code: 137`, `duration_s: 1250`, GPU used 34456 → max
   39268 MiB (whole card; the other tenant's share is included).
2. **Run `smoke-resumed2`** started with `RESUME_FROM=<smoke>/nerfstudio_models`. The log
   shows `Done loading Nerfstudio checkpoint`; the **first logged step is 1001**. It saved
   `step-000002000.ckpt` and `step-000003000.ckpt`. It was then stopped at step ~3890 with
   SIGTERM (`exit_code: 143`) because nerfstudio trains `--max-num-iterations` steps *on top
   of* the loaded step (fixed afterwards, see below). GPU used max 39552 MiB; duration 1932 s.

**Result: a killed run resumes from its last checkpoint and continues; nothing before the
last 1k-step checkpoint is lost.**

## Problems found and fixed on the way (all recorded in git)

| Problem | Fix |
|---|---|
| First eval-all-images crashed: `KeyError 'within_rgb_psnr'` (DN-Splatter bug, `dn_pipeline.py:516`) | `setup_gpu.sh` patches the lookup to the bare key |
| `--load-dir` resume crashed: `KeyError 'step'` (DN-Splatter's `normals` optimizer saves an empty Adam state) | `setup_gpu.sh` patches nerfstudio's loader to drop empty states |
| Resumed run overshot `--max-num-iterations` | `roomrecon/resume.py` + `train_room.sh` pass only the remaining steps |
| `uint8` image cache broke a DN-Splatter loss (`'Byte'` tensor) | back to the default float32 CPU cache |

## Evaluation curve works (from `g1-smoke-eval-curve.csv`)

| Step | within-walk held-out PSNR / SSIM / LPIPS | separate short walk PSNR / SSIM / LPIPS |
|---|---|---|
| 2000 | 22.06 / 0.782 / 0.287 | 20.61 / 0.758 / 0.319 |
| 3000 | 22.64 / 0.794 / 0.250 | 21.01 / 0.767 / 0.290 |

These are 3000-step smoke numbers, not results.

## Costs measured

- Training speed ≈ 120 ms/step (log).
- Caching/undistorting the 422 training images on the CPU: ≈ 7 min per run start
  (00:07:24 → 00:14:20 in the log). Eval images: ≈ 10 min at the first evaluation.
- The first run on a room builds its starting point cloud from iPhone depth (≈ 2 min, once).
