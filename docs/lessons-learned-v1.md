# Lessons Learned: M1 (to 2026-09-25)

M1 work is being restarted from a clean WSL environment. This file keeps what the first
pass taught us. Every number below was measured on **T1** (RTX 3050 Laptop 6 GB, driver
581.86, WSL2 Ubuntu-22.04) unless stated otherwise. The install and evidence log remains
in `docs/m1-downloads.csv`.

**Deleted in the reset:** all trained runs, datasets, the Python venv and the toolchain builds
(`~/roomrecon/` in WSL), plus the git-ignored outputs under `storage/projects/`. That includes
`storage/projects/m1-pose-validation/3dgs_pose_validation_report.md`, which ADR-006 cites.
Its findings are summarised in section 2 below. Nothing in the repo itself was removed.

---

## 1. Toolchain and environment

- **Reconstruction runs in WSL2, not native Windows.** ORB-SLAM3 has no practical Windows
  build. CUDA reaches WSL through the Windows driver, so no display driver goes inside WSL.
- **CUDA:** install `cuda-nvcc-12-1` + `cuda-cudart-dev-12-1` (~190 MB) from the
  `wsl-ubuntu` repo, not the full toolkit. The pip package `nvidia-cuda-nvcc-cu12` ships no
  `nvcc` binary and cannot build extensions.
- **Pin torch.** `pip install 'torchmetrics[image]'` silently replaced torch 2.5.1+cu121 with
  2.14.0+cu130. That grew the venv from 4.9 to 7.5 GB and broke extension builds. Always
  install with `-c constraints.txt` (torch/torchvision/numpy pinned).
- **numpy is 1.26.4, not 2.x.** gsplat's example requirements need `numpy<2`.
  `scripts/verify_env.py` caught this drift the first time it ran.
- `fused_ssim` needs `pip install --no-build-isolation`.
- **ORB-SLAM3 needs 3 local patches** (see `docs/m1-downloads.csv`):
  - C++17 for Pangolin 0.9.x
  - monocular-only targets
  - `LoopClosing.h` `bool mnFullBAIdx` changed to `int`, an upstream bug
- **gsplat 1.5.3 cannot resume training upstream.** Our patch
  `scripts/patches/gsplat-1.5.3-resume.patch` adds `--resume_every` / `--resume`. It passed
  a SIGKILL test: the resumed run matched uninterrupted runs within run-to-run spread.
- **COLMAP (apt 3.7) is CPU-only.** That is fine for sparse SfM. Exhaustive matching is
  O(n²) (~23 min for 128 images), so use the sequential matcher for video.
- **Dataset URLs move.** COLMAP sample datasets are now on GitHub releases (demuc.de is
  404). The Deep Blending mirror on Hugging Face ships COLMAP `sparse/` so matching can be
  skipped.
- **Shell traps (Windows host → WSL):**
  - `pkill -f` inside `wsl bash -c` matches its own shell.
  - PowerShell here-strings piped to `bash -s` add a BOM and CRLFs.
  - Git Bash rewrites `/mnt/...` paths.
  - Put multi-line WSL commands in a `.sh` file and run `wsl -- bash <file>`.

## 2. Camera poses: the main M1 finding

Scene: ARKitScenes 41254269, a plain-walled bedroom, 95 s iPad walk.

| Pose source | Walk covered | Error vs ARKit | 3DGS result |
|---|---|---|---|
| COLMAP SIFT (self-calibrated) | 24% | 1.2 cm | Sharp and coherent, but only a quarter of the room |
| SuperPoint + LightGlue (hloc) | 100% | 49 cm | **Total collapse**: every view renders as flat colour |
| ORB-SLAM3 mono | 7–93% of active map, 4–6 maps per run | 6–52 cm | not trained |
| ARKit device poses (as given) | 100% | reference | see section 3 |

A textured room (utility room) tracked cleanly: COLMAP covered 100% at 1.6 cm. **Image-only
poses fail specifically on low-texture rooms.** That finding led to ADR-006 (phone-tracked
poses preferred).

- **Accuracy matters more than coverage.** 100% coverage at ~0.5 m error gave no usable
  scene. 24% at ~1 cm gave a usable partial scene.
- **ATE and SSIM hid the failure.** 49 cm ATE suggested "warped", but the result was a
  total collapse. SSIM stayed 0.82 on renders that were empty. **PSNR plus actual renders
  are the honest signals.**
- **Bad poses surfaced as a resource problem.** The collapsed run allocated 7.67 GB on a
  6 GB card, spilled into shared memory and slowed from ~20 it/s to 1–3 s/step.
- **Warning signs before training:** reprojection error 1.55 px (good runs 0.74–1.2) and a
  focal length 2.4% off factory. These are candidate signals for an M2/M3 pre-training gate.
- **ORB-SLAM3 is nondeterministic.** ATE varied ~2× across runs on TUM fr3 (10–20 mm), so
  report its accuracy over repeated runs only.

### Importing device poses (`scripts/device_poses_to_colmap.py`)
- The pipeline takes poses as given, and COLMAP only triangulates with them held fixed. The
  script verifies no pose changed; the maximum change measured was 4e-16.
- ARKitScenes `.traj` rows are world→camera, axis-angle, OpenCV frame: straight onto
  COLMAP's qvec/tvec.
- **Capture-app `transforms.json` (NeRFCapture/instant-ngp) is camera→world in the OpenGL
  frame.** Flip y and z. We verified this rather than assuming it: with the flip COLMAP
  triangulated 4,475 points and filtered 10 observations; without it, 144 points and 3,200
  filtered.
- COLMAP triangulation is not bit-deterministic. The same input gave 11,034 vs 11,050
  points, with identical poses.

## 3. Training 3DGS on a whole room

Same settings throughout: gsplat MCMC, cap 1M Gaussians, 15k steps, 640×480 vga frames.
The held-out split is every 8th image.

| Run | Frames | gsplat scene scale | Held-out PSNR |
|---|---|---|---|
| ARKit poses, 22 s window (35–57 s) | 112 | 1.58 | **30.3** (SSIM 0.937, LPIPS 0.137) |
| ARKit poses, full 94 s walk | 472 | 2.98 | **13.7** at 15k: flat colour |

**Collapse onset (full walk), measured every 1k steps:**

| Step | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k |
|---|---|---|---|---|---|---|---|---|---|
| MCMC PSNR | 17.4 | 21.4 | 22.1 | **22.8** | 22.6 | 22.1 | 22.1 | 21.1 | 19.4 |
| MCMC Gaussians | 13k | 22k | 36k | 58k | 94k | 154k | 250k | 408k | 664k |
| `default` (ADC) PSNR | 20.2 | 21.0 | 21.4 | 21.5 | **21.5** | 21.4 | 21.3 | 21.1 | 20.9 |
| `default` Gaussians | 30k | 241k | 478k | 692k | 875k | 1.05M | 1.19M | 1.33M | 1.45M |

- **The whole room is learnable.** PSNR peaks at 22.8 by step 4k. Evaluating only at the end
  hid this: the final number alone said "total failure".
- **ARKit poses are not the cause.** The same poses give 30.3 on a 22 s window, and two
  independent pose sources both collapse on the full walk.
- **Ruled out:**
  - Far outlier init points: the window run had *more* of them (6.4% vs 0.08%).
  - Auto-exposure drift: luminance range is the same.
  - The hallway inflating the scene scale: the farthest cameras are at 73 s.
- **Switching densification to `default` does not fix it.** It also peaks early (~5k) and
  then declines, more slowly. Its LPIPS keeps improving to 9k while PSNR falls.
- **Open:** why held-out PSNR declines after ~4–5k on the full walk under both strategies.
  Candidates not yet tested:
  - a lower position learning rate (scene scale is 1.9× the window run's)
  - over-fitting to 416 training views while held-out views drift
  - a longer/lower LR schedule
  Save checkpoints and evaluate **every 1k steps** in any diagnostic run; only then is the
  peak visible.

### Capture density
A test phone capture with 32 photos over 79 s scored held-out PSNR 16.0. The median
view change between consecutive photos was 33°, max 132°. The geometry was placed
correctly, but sparse views produce streaks and floaters. **Capture guidance:** several
frames per second, a few hundred frames per room, move slowly.

## 4. Compute (T1)

- **6 GB VRAM is not the binding constraint for training.** 1M Gaussians peaked at ~3.9 GB
  total, ~2.2 GB for 3DGS itself. ~1 GB is taken by Windows at idle, so ~5 GB is usable.
- Measured throughput: 250k Gaussians trained at 16.4 it/s, 1M at 8.7 it/s. First use JIT-compiles gsplat's kernels,
  ~7 min one-off.
- Deep Blending drjohnson (indoor benchmark) scored PSNR 27.2 at 7k steps. It beat the
  outdoor south-building scene (22.3) by ~5 dB.
- gsplat's default `eval_steps` are [7000, 30000]. With `max_steps` 15000 the final model is
  never evaluated, so set `eval_steps` explicitly. Note that eval file names are 0-based
  (`val_step14999`).
- gsplat only auto-downscales `.jpg` images. For `.png` with `data_factor > 1`, create
  `images_<factor>/` yourself.

## 5. Browser viewer (GaussianSplats3D 0.4.7, Three.js)

- **The browser, not the GPU, is the limit.** At 250k splats the median frame is 2.85 ms
  and p95 is 31 ms (32 fps). At 1M, the median is 8.2 ms but p95 is 127 ms (8 fps). The likely
  product setting is ~400–500k, or LOD/compression in M5.
- Measure with `EXT_disjoint_timer_query_webgl2`. JS-clock timing reported nonsense
  (4000+ fps).
- **Scene conventions:**
  - Up axis is +Z (COLMAP), not -Y.
  - Frame the camera with 2–98th percentiles: outlier splats inflate the min/max box ~27×.
  - gsplat PLY is the standard INRIA layout, so no conversion is needed.
- **Load times:** 58k splats (14 MB) loads in a few seconds. 1M (236 MB) takes ~20–30 s.
- **A "hung" viewer was actually a JSON bug.** gsplat's `Parser` prints `Warning:
  image_path not found...` to **stdout**. Redirecting that output into `viewpoints.json`
  made the file invalid, and the page sat on "loading…" forever. Filter to the JSON line.
- The viewer drops splats with opacity below ~1/255. The browser view of one scene looked
  softer than gsplat's own render at the same pose; the cause is not yet determined (M5).
- ARKitScenes frames are stored in sensor orientation, so views appear rotated 90°. The
  training data itself is unaffected.

## 6. Process

- **Evaluate intermediate checkpoints.** A final-only metric turned "peaks at 22.8, then
  degrades" into "13.7, total failure".
- **One variable per ablation,** with the same data, schedule and seed-controlled config.
  Verify the input is identical before comparing: re-import and diff against the old run.
- **Check the data before training:** point and camera spread, outliers, exposure. The
  cheap checks cleared three suspects in minutes.
- **Keep evidence out of git-ignored folders** if it must survive a cleanup. The
  pose-validation report was lost this way; this file replaces it.
- **Disk:** the WSL `ext4.vhdx` does not shrink when files are deleted. It needs a
  compaction with WSL cleanly shut down. Never force-compact a disk that is still mounted:
  doing that once destroyed all Docker data.

## 7. T2 (Kaggle) run 1: 2026-09-25

The first Kaggle run trained everything and still ended with status ERROR and no results
file. Its numbers survive only in the Kaggle log, recorded in `m1-downloads.csv`.

- **What broke:** Kaggle's "T4" accelerator is **T4 ×2**. `nvidia-smi` prints one line
  per GPU, so the saved GPU memory baseline read `0\n0`, and an `int()` in the final
  packaging cell raised. Every VRAM figure had also been silently mixed across both GPUs.
- **Why it cost the run:** results sat in `/tmp` until that last cell, so one parse error
  lost the 30k PLY and every structured record. Failure handling was inverted:
  `train.py … || true` hid training crashes, while an optional telemetry parse was fatal.
- **Why it tested the wrong thing:** the hand-written notebook duplicated the committed
  path (`run_t2.sh` → `train.py`) and trained **INRIA** 3DGS. That is the code the
  ROADMAP proposes to drop. It did not exercise `setup_t2.sh` or our gsplat resume patch,
  so no T2 gate item closed.
- **Also:** `kaggle kernels output` on Windows wrote a 0-byte log while reporting success
  (codepage). The output listing is paginated (20 by default), another silent truncation.

What changed, and the rule behind each change:

- **One T2 entry point.** `push.sh` generates the notebook: clone at the exact commit, run
  `scripts/run_t2.sh`. All logic lives in the repo, where it is tested.
- **Hardware is data.** `scripts/gpu.py` lists every GPU. Each job is pinned to one GPU by
  **UUID**; nvidia-smi and CUDA order GPUs differently, so an index is ambiguous.
  `run.json` records all GPUs and the one used.
- **The exit code is the gate verdict.** Stages are recorded as required or best-effort,
  and `t2_verdict.py` always writes `verdict.json`, even after a failure. Kaggle's status
  now means "gate passed or not".
- **Write results where they persist, as they are produced,** and make reruns idempotent.
  Resume state lives in the session output. `push.sh push --continue` attaches it to the
  next session: finished runs are skipped and interrupted ones resume.
- **Measure noise before judging a difference:** at least 2 uninterrupted references for
  the resume check. The comparison is reported, not thresholded, until a tolerance is
  measured.
- **Nothing reaches a GPU session untested:** `push.sh` refuses to push a dirty tree, an
  unpushed commit, or failing unit tests.
- **A fetch that returns empty files is a failed fetch.**
