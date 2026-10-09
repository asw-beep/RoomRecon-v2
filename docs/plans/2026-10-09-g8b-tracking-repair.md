# G8 (new): Tracking Repair — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair cap3's broken tracking by gravity-aware ICP of its sweeps, write raw / drop / repaired training variants with one shared test split, measure alignment (C1) and added coverage (C3) on the CPU, train the three variants and measure rendering (C2, C4), and walk-test raw vs repaired.

**Architecture:** A new package `roomrecon/repair/` reusing the capture check: `capcheck.tracking` finds the pieces, `capcheck.surface`/`views` build clouds and coverage, `scene.estimate_up` gives the vertical. Registration is our own numpy/scipy point-to-plane ICP from 36 vertical-axis starts. Variants are folders of `transforms.json` + linked `images/`, `depth/`, `normals_from_pretrain/`, trained with the existing `scripts/own_room.sh` (A100) or `scripts/kaggle/own_room_kaggle.sh` (Kaggle T4).

**Tech Stack:** Python (PC 3.12 / numpy 2.4 for tests; GPU machine pixi env 3.10 / numpy 1.26, scipy 1.13), pytest, DN-Splatter (nerfstudio 1.1.3). No new dependencies.

**Spec:** `docs/spec/2026-10-09-g8b-tracking-repair-design.md`. Read it with this plan.

## Global Constraints

- Branch `g8-tracking-repair` (built on `g8-capture-check`, which is the fallback novelty: do not change its behaviour).
- Code must run on numpy 1.26 and 2.4; no open3d, no new packages.
- Heavy data and runs on the GPU machine; the PC runs pytest and views results. GPU-machine CPU runs set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (as `scripts/capcheck.sh`).
- Cameras: camera-to-world 4×4, OpenGL; repaired pose `c2w' = T · c2w`.
- Spec values: pieces split at segment changes and pose jumps (`capcheck.tracking`); keep pieces ≥ 20 frames; largest piece = reference. Piece clouds: depth stride 8, confidence 255, 5 cm voxels, 16-neighbour normals. Registration: 36 starts every 10° about the vertical (`scene.estimate_up` of the reference cameras), centroid-matching shift, point-to-plane ICP with correspondence distances 0.5 → 0.2 → 0.05 m, ≤ 30 iterations each; fitness = share of piece points within 0.05 m of the reference; best fitness wins, ties → lower RMSE; accept if fitness ≥ 0.3 and tilt < 3°.
- Variants: `raw` (all frames, exported poses), `drop` (reference only), `repaired` (reference + accepted pieces, corrected poses). Same `test_filenames` in all: reference positions 0, 8, 16, …; everything else of that variant is `train_filenames`. File paths in each variant's `transforms.json` are the source's own `file_path` / `depth_file_path` strings; `images/`, `depth/`, `normals_from_pretrain/` are directory symlinks to the source folder (copy if symlinks are not permitted, i.e. Windows tests).
- Criteria (spec §4): C1 alignment ≤ 0.03 m after repair; C2 repaired ≥ raw + 1.0 dB PSNR with SSIM higher and LPIPS lower; C3 added surface ≥ 5 %; C4 repaired ≥ drop − 0.3 dB PSNR. Reported as measured.
- Training: the same steps for all three variants: 30k (stop-split 10k) on the A100, or 15k (stop-split 5k) on a Kaggle T4.
- Never touch the `vr_room` demo outputs. Ask the user before pushing to `origin`; `git push gpu main` (fast-forward `main` to this branch first) is part of this approved work.
- Every decision and measured number also goes into `docs/report-notes.md`.

## Review Focus

1. A candidate piece that does not overlap the reference (fitness < 0.3) → rejected, left out of `repaired`, reported with its fitness, never silently merged (test in Task 1 and Task 3).
2. A registration that converges with a tilt (non-gravity-aligned solution) → rejected with reason "tilt" (test in Task 1).
3. A capture with no tracking break (one piece) → `repaired` equals `drop` equals `raw` minus nothing; the CLI says "nothing to repair" and still writes the three variants (test in Task 3).
4. Symlinks not permitted (Windows) → directories copied instead, variant still loads (test in Task 2 runs on the PC).
5. A capture without tracking data (MuSHRoom) → clear error "no tracking data: cannot find pieces" (test in Task 3).

---

### Task 1: Gravity-aware registration

**Files:**
- Create: `roomrecon/repair/__init__.py` (empty), `roomrecon/repair/register.py`, `tests/test_repair_register.py`

**Interfaces:**
- Produces:
  - `rot_about(axis: np.ndarray, deg: float) -> np.ndarray` (3×3, right-handed).
  - `icp_point_to_plane(src: np.ndarray, dst: np.ndarray, dst_normals: np.ndarray, T0: np.ndarray, dists=(0.5, 0.2, 0.05), iters=30) -> np.ndarray` (4×4). Each iteration: nearest `dst` point per `src` point (`cKDTree`, within the current distance), solve the linearised system `A x = b` with rows `A_i = [p_i × n_i, n_i]`, `b_i = (q_i − p_i)·n_i` for `x = [ω, t]` (least squares), compose `T ← [R(ω), t] · T` (`R(ω)` = `rot_about(ω/|ω|, |ω|)`); stop a stage when |x| < 1e-6.
  - `score(src, dst, T, inlier=0.05) -> tuple[float, float]` → (fitness, RMSE of inliers).
  - `register_piece(src_pts, dst_pts, dst_normals, up, n_starts=36, min_fitness=0.3, max_tilt_deg=3.0) -> dict` → `{"T": 4×4 list, "fitness", "rmse", "yaw_deg", "tilt_deg", "accepted": bool, "reason": "" | "low fitness" | "tilt"}`; tilt = angle between `R @ up` and `up`; yaw = rotation angle about `up`.

- [ ] **Step 1: Write the failing tests** (synthetic clouds, no images: helper `box_cloud(spacing=0.05)` in the test file = points + inward normals on the walls, floor and ceiling of a 4 × 2.5 × 3 m box, plus an off-centre 0.6 × 1.0 × 0.4 m block at (0.6, 0, 0.5) so the room has no rotational symmetry; y is up):
  - `test_rot_about_is_rotation`: orthonormal, det 1, rotates x to y about z by 90°.
  - `test_icp_refines_small_offset`: dst = box, src = box moved by 3° yaw + (0.05, 0, −0.04) → `icp_point_to_plane` from identity recovers the inverse within 0.005 m on points.
  - `test_register_recovers_large_yaw_and_shift`: src = box under yaw 70° + shift (0.8, 0, −0.5) → `accepted`, fitness > 0.9, applying `T` to src gives max point error < 0.01 m, |yaw| within 0.5° of the true correction, tilt < 0.5°.
  - `test_register_rejects_no_overlap`: src = a 2 × 2 m plane 20 m away → `accepted` False, reason "low fitness".
  - `test_register_rejects_tilted_solution`: src = box rolled 10° about x (plus a yaw) → `accepted` False, reason "tilt" (Review Focus 2).
- [ ] **Step 2: Run** `python -m pytest tests/test_repair_register.py -q` — expected: fails (module not found).
- [ ] **Step 3: Implement** `roomrecon/repair/register.py`.
- [ ] **Step 4: Run** — expected: 5 passed.
- [ ] **Step 5: Commit** `feat: gravity-aware point-to-plane ICP for tracking repair`.

---

### Task 2: Pieces and training variants

**Files:**
- Create: `roomrecon/repair/variants.py`, `tests/test_repair_variants.py`

**Interfaces:**
- Consumes: `Capture`, `Frame`, `load_capture` (capcheck.capture); `Tracking`, `check_tracking` (capcheck.tracking); `build_surface`, `Surface` (capcheck.surface); `tests/synthroom.make_room`.
- Produces:
  - `split_pieces(cap: Capture, tr: Tracking, min_frames=20) -> tuple[list[str], list[list[str]], list[str]]` → (reference frame names, candidate pieces (each a list of names, ≥ `min_frames`), dropped names (pieces shorter than `min_frames`)); pieces are cut before every name in `tr.segment_changes` and every `jump["frame"]`; reference = the largest piece (ties → earliest).
  - `subset(cap: Capture, names) -> Capture` — same root/source, only those frames, original order.
  - `piece_cloud(cap: Capture, names) -> Surface` — `build_surface(subset(cap, names), voxel=0.05, stride=8)`.
  - `test_names(reference: list[str]) -> list[str]` — positions 0, 8, 16, … of `reference`.
  - `write_variants(src_root: Path, out: Path, reference, accepted: dict[str, np.ndarray], dropped) -> dict[str, Path]` — `accepted` maps each frame name of an accepted piece to its 4×4 `T`; reads `src_root/transforms.json`; writes `out/raw/`, `out/drop/`, `out/repaired/`, each with `transforms.json` (the source document with `frames` filtered to the variant, repaired poses `T · c2w` for accepted frames, `ply_file_path` removed, `train_filenames`/`test_filenames` = the frames' `file_path` strings) and links `images`, `depth`, `normals_from_pretrain` (if present in the source) to the source directories (`os.symlink(..., target_is_directory=True)`; on `OSError` copy the directory). `raw` = all source frames; `drop` = reference; `repaired` = reference + accepted. Returns the three folder paths.

- [ ] **Step 1: Write the failing tests** (synthetic Polycam room: 30 cameras on a loop, meta `tracking_segment` 1 for frames 1–22, 2 for 23–24, 3 for 25–30?? — no: use 40 frames: segment 1 for frames 1–24, 2 for 25–26, 3 for 27–40, so pieces are 24 / 2 / 14; `min_frames=10` in the tests):
  - `test_split_pieces_reference_candidates_dropped`: reference = frames 1–24, one candidate = frames 27–40, dropped = frames 25–26.
  - `test_test_names_every_8th_of_reference`: `["frame_00001", "frame_00009", "frame_00017"]`.
  - `test_write_variants_share_test_split_and_correct_poses`: with a known `T` for the candidate piece: all three `transforms.json` have the same `test_filenames`; `drop` has 24 frames, `repaired` 38, `raw` 40; a repaired frame's pose equals `T · c2w` (atol 1e-9); a reference frame's pose is unchanged; `train_filenames` ∪ `test_filenames` = the variant's frames; `ply_file_path` absent.
  - `test_variant_images_resolve` (Review Focus 4, runs on the PC): every `file_path` and `depth_file_path` of `repaired` opens with Pillow from the variant folder.
- [ ] **Step 2: Run** `python -m pytest tests/test_repair_variants.py -q` — expected: fail.
- [ ] **Step 3: Implement** `roomrecon/repair/variants.py`.
- [ ] **Step 4: Run** — expected: 4 passed.
- [ ] **Step 5: Commit** `feat: tracking-repair pieces and raw/drop/repaired training variants`.

---

### Task 3: Metrics, CLI and wrapper

**Files:**
- Create: `roomrecon/repair/metrics.py`, `roomrecon/repair/__main__.py`, `scripts/repair_room.sh`, `tests/test_repair_cli.py`

**Interfaces:**
- Consumes: Tasks 1–2; `capcheck.surface.backproject`, `build_surface`; `capcheck.views.accumulate` (with `frames_mask`); `capcheck.capture.load_depth_m`; `scene.estimate_up`.
- Produces:
  - `alignment_error(cap: Capture, names: list[str], ref: Surface, T: np.ndarray | None = None, every=4, stride=8) -> float` — for every `every`-th frame in `names`: back-project its depth with `T · c2w` (or `c2w`), median nearest-neighbour distance to `ref.points`; returns the median over frames (C1).
  - `added_coverage(cap: Capture, reference: list[str], pieces: list[str]) -> float` — on `build_surface(subset(cap, reference + pieces))` (3 cm), the share of points with views from `pieces` frames and none from `reference` frames (two `accumulate` passes with `frames_mask`) (C3). `cap` here is the repaired variant loaded with `load_capture`.
  - CLI `python -m roomrecon.repair CAPTURE --out DIR [--min-frames 20]` → loads the capture (error "no tracking data: cannot find pieces" if `not cap.has_tracking`), `check_tracking`, `split_pieces`, piece clouds, `register_piece` per candidate (vertical = `estimate_up` of the reference cameras), `write_variants`, C1 before/after per accepted piece (reference surface = `piece_cloud` of the reference), C3 on the repaired variant, `repair.json` (`{"reference": [first, last, n], "pieces": [{"frames": [first, last, n], ...register_piece result, "c1_before_m", "c1_after_m"}], "dropped": [...], "c3_added_share", "criteria": {"C1": bool, "C3": bool}, "variants": {name: n_frames}}`), prints a one-line summary; with one piece only prints "nothing to repair" and still writes the variants.
  - `scripts/repair_room.sh CAPTURE OUT [args]` — runs the CLI in the pixi env (BLAS threads 1), then `python dn_splatter/data/mushroom_utils/pointcloud_utils.py --data-path OUT/<variant> --num-points 300000` for each variant (from the DN-Splatter dir).

- [ ] **Step 1: Write the failing tests** (synthetic Polycam room, 40 frames as in Task 2, with frames 27–40's recorded poses corrupted by a known `E` (yaw 70°, shift (0.8, 0, −0.5)) by rewriting their `transform_matrix` in `transforms.json`; depth and images stay true):
  - `test_alignment_error_drops_after_repair`: `alignment_error` with the corrupted poses > 0.3 m; with `T = E⁻¹` < 0.01 m.
  - `test_added_coverage_positive_when_piece_sees_more`: piece frames looking at a wall the reference never faces → > 0; a piece that duplicates reference views → < 0.02.
  - `test_cli_end_to_end`: `main([room, "--out", out, "--min-frames", "10"])` → 0; `repair.json` has one accepted piece, `c1_after_m < 0.02 < 0.3 < c1_before_m`, `criteria.C1` True; `out/{raw,drop,repaired}/transforms.json` exist.
  - `test_cli_single_piece_nothing_to_repair` (Review Focus 3) and `test_cli_rejects_capture_without_tracking` (Review Focus 5, MuSHRoom layout → non-zero exit / `SystemExit` with the message).
  - `test_cli_reports_rejected_piece` (Review Focus 1): corrupt with a 10° roll instead → piece `accepted` False, `repaired` has the reference frames only.
- [ ] **Step 2: Run** `python -m pytest tests/test_repair_cli.py -q` — expected: fail.
- [ ] **Step 3: Implement** `metrics.py`, `__main__.py`, `scripts/repair_room.sh` (`git update-index --chmod=+x`).
- [ ] **Step 4: Run** `python -m pytest tests -q` — expected: all pass.
- [ ] **Step 5: Commit** `feat: tracking-repair CLI with alignment and coverage measures`.

---

### Task 4: Repair cap3 and measure C1, C3 (GPU machine, CPU only)

- [ ] **Step 1:** Fast-forward `main` to this branch, `git push gpu main`.
- [ ] **Step 2:** On the GPU machine in tmux: `bash scripts/repair_room.sh ~/roomrecon/datasets/own/sejal-cap3/processed ~/roomrecon/datasets/own/sejal-cap3/variants`. Expected: reference frames 125–491 (367), one accepted piece frames 1–117, 7 dropped; variants raw 491 / drop 367 / repaired 484, each with `iphone_pointcloud. py`'s `iphone_pointcloud.ply`.
- [ ] **Step 3:** Copy `repair.json` to `results/g8b-repair.json`; record C1 (before → after) and C3 with PASS/FAIL in `docs/report-notes.md`. If the piece is rejected, stop and report to the user (the repair failed on real data; fallback discussion).
- [ ] **Step 4:** Visual check on the PC: the capture check's `coverage.ply` is not needed; instead write the two piece clouds (reference grey, piece before red / after green) as a small 3DGS PLY with `capcheck.outputs.write_splat_ply` into `E:\roomrecon-scenes\repair\cap3\` and look at them in the viewer.
- [ ] **Step 5:** Commit `exp: cap3 tracking repair - C1 and C3`.

---

### Task 5: Train the three variants and measure C2, C4

GPU source decided at run time: the A100 if `torch.cuda.is_available()` in the pixi env; else Kaggle (ask the user to confirm quota first).

- [ ] **Step 1 (A100):** normals once: `NORMALS_BATCH=4 python scripts/normals.py --data-dir <variants>/raw --resolution low` (in the DN-Splatter dir); link `raw/normals_from_pretrain` into `drop/` and `repaired/`. Then, one after another in tmux: `bash scripts/own_room.sh sejal-cap3-{raw,drop,repaired} dns-30k-lite <variant> 30000`.
- [ ] **Step 1 (Kaggle alternative):** change `own_room_kaggle.sh` push to copy with `cp -rL` (follow the variant's directory links); push each variant as room `sejal-cap3-{raw,drop,repaired}`, run `dns-15k-kaggle`, 15000 steps; fetch when complete.
- [ ] **Step 2:** For each variant read `eval/ns_eval.json` (PSNR, SSIM, LPIPS); check the eval curve for collapse (`eval_curve.csv`); look at 2 test renders per variant.
- [ ] **Step 3:** Write `results/g8b-tracking-repair.md`: machine, steps, per-variant metrics table, C1–C4 PASS/FAIL as measured, run times; copy the `ns_eval.json` files to `results/g8b-<variant>-ns_eval.json`; add to `docs/report-notes.md`; commit `exp: G8 tracking repair - three cap3 variants trained and compared`.

---

### Task 6: Walk demo and wrap-up

- [ ] **Step 1:** `scripts/export_room.sh` for `sejal-cap3-raw` and `sejal-cap3-repaired` (GPU machine), copy exports to `E:\roomrecon-scenes\sejal-cap3-{raw,repaired}\<run>\`, `scripts/collision_room.ps1` each, open both in the viewer for the user's walk test; record the user's verdict and FPS in `results/g8b-tracking-repair.md`.
- [ ] **Step 2:** Update `CLAUDE.md` status (G8 = tracking repair, results, fallback no longer needed or still needed) and `docs/report-notes.md` §6; commit `docs: G8 tracking repair results and demo`; ask the user before `git push origin`.
