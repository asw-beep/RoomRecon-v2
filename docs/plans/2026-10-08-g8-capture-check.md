# G8: Capture Check — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pre-training check (`python -m roomrecon.capcheck`) that labels every surface of a phone room capture as OK / under-covered / textureless / broken tracking, writes a report, a coloured 3D map and a floor plan, and a validation that measures the labels against the trained splats' held-out errors on `vr_room` and `sejal-cap2`.

**Architecture:** A new package `roomrecon/capcheck/` of small numpy/scipy modules: readers turn a capture folder into one `Capture`; LiDAR depth becomes voxelised surface points; every frame is projected onto the points (with a depth-based occlusion test) to count views, view directions and image texture; Polycam's per-frame data gives tracking health; labels, regions and a verdict follow; outputs reuse `roomrecon.scene`/`scene_cli` for alignment. Tests run on the PC on tiny synthetic rooms; real runs and validation run on the GPU machine.

**Tech Stack:** Python (PC 3.12 / numpy 2.4 for tests; GPU machine pixi env 3.10 / numpy 1.26, scipy 1.13, Pillow 10.3, matplotlib 3.8), pytest. No new dependencies, no open3d.

**Spec:** `docs/spec/2026-10-08-g8-capture-check-design.md`. Read it with this plan. Decisions and results also go to `docs/report-notes.md`.

## Global Constraints

- Heavy runs (real captures, validation) on the GPU machine; the PC runs pytest and views outputs only.
- Code must run on numpy 1.26 and 2.4: no `np.bitwise_count`, no numpy-2-only APIs. No open3d (avoids the libstdc++ `LD_PRELOAD` problem), no pandas, no sklearn.
- Cameras: camera-to-world 4×4, OpenGL (x right, y up, looking down −z). Pixel of a camera-space point (X, Y, Z), Z < 0: `u = cx + fx·X/(−Z)`, `v = cy − fy·Y/(−Z)`; back-projection of depth d (z-depth, metres) at (u, v): `X = (u−cx)·d/fx`, `Y = −(v−cy)·d/fy`, `Z = −d`.
- Depth PNGs are uint16 millimetres; 0 = missing. Use depth 0.1–5.0 m. Polycam confidence PNGs (256×192, values 0 / 54 / 255) are resized to image size with nearest neighbour; only 255 is trusted.
- Polycam `timestamp` is in microseconds; keyframe file names are the timestamps.
- Spec values: voxel 0.03 m; back-projection stride 4 px; occlusion tolerance max(0.05 m, 0.03·D); 64 Fibonacci view directions; under-covered = `n_views < 5` or `n_dirs < 3`; texture = mean Sobel magnitude of the grey image (0–1) in a 9×9 window, median over views; jumps = speed > 1.5 m/s or rotation > 180°/s; regions = 0.10 m neighbourhood, area ≥ 0.25 m² (area = points × 0.03²); floor cells 0.10 m, "seen" = a point within 0.30 m of the floor; splat size 0.015 m; colours OK green (0.2, 0.8, 0.2), under-covered red (0.9, 0.1, 0.1), textureless blue (0.2, 0.4, 0.95), broken grey (0.5, 0.5, 0.5); validation patches 16×16 px, match radius 0.03 m, "bad" = top 20 % error, bootstrap 95 % (1000 resamples of test frames, seed 0), AUC margin 0.05.
- Label priority: broken > under-covered > textureless > OK. Label codes: `OK=0, UNDER=1, TEXTURELESS=2, BROKEN=3`.
- GPU machine commands as in `CLAUDE.md` (`ssh -o BatchMode=yes gpu bash -s < script`); env python `~/roomrecon/tools/dn-splatter/.pixi/envs/default/bin/python`; run from `~/roomrecon/repo`. Outputs to `~/roomrecon/capcheck/<name>/`. Code reaches the GPU machine only via `git push gpu main`.
- Never touch the `vr_room` demo outputs (`~/roomrecon/scenes/vr_room/`, `E:\roomrecon-scenes\vr_room\`, `demo-vr_room`); read only.
- Ask the user before pushing to `origin`; `git push gpu main` is needed for every GPU-machine step and is part of this approved work.

## Review Focus

1. A frame whose depth is all zero or mostly missing (LiDAR dropout, window, mirror) → it adds no points and sees nothing; no NaN, no crash (test in Task 3).
2. A capture without `keyframes/`, or without `keyframes/corrected_cameras/` (cap1, MuSHRoom) → tracking is "not available" or partial, the rest runs (test in Task 4).
3. Excluding test frames leaves points seen by no remaining frame → `n_views = 0`, texture NaN, label UNDER, never TEXTURELESS (test in Task 5).
4. Processed frame count ≠ keyframe count → a clear error naming both counts, never a silent mismatch (test in Task 1).
5. Repeated or non-increasing timestamps → that step is flagged "bad timestamp", no division by zero (test in Task 4).

---

### Task 1: Capture readers

**Files:**
- Create: `roomrecon/capcheck/__init__.py` (empty), `roomrecon/capcheck/capture.py`
- Create: `tests/synthroom.py` (synthetic room writer, shared by later tasks), `tests/test_capcheck_capture.py`

**Interfaces:**
- Produces:
  - `Frame` dataclass: `name: str` (file stem, e.g. `"frame_00001"`), `K: np.ndarray` (3×3), `width: int`, `height: int`, `c2w: np.ndarray` (4×4), `image_path: Path`, `depth_path: Path`, `conf_path: Path | None`, `meta: dict` (Polycam camera fields + `"weakly_connected"` from `corrected_cameras` when present; `{}` otherwise).
  - `Capture` dataclass: `root: Path`, `source: str` (`"polycam"` | `"mushroom"`), `frames: list[Frame]`, `has_tracking: bool`.
  - `load_capture(root: Path, exclude: frozenset[str] = frozenset()) -> Capture` — `transforms.json` ⇒ polycam (keyframes optional), `transformations.json` ⇒ mushroom; frames sorted by name; excluded names dropped *after* keyframe matching.
  - `read_exclude(path: Path) -> frozenset[str]` — one name per line, stems, blank lines ignored.
  - `load_depth_m(f: Frame) -> np.ndarray` (H×W float32 metres, resized to `height×width` nearest if needed), `load_confidence(f: Frame) -> np.ndarray | None` (H×W uint8, nearest-resized), `load_gray(f: Frame) -> np.ndarray` (H×W float32 0–1).
  - `tests/synthroom.py`: `make_room(dir: Path, cams: list[tuple[np.ndarray, float, float]], *, size=(4.0, 2.5, 3.0), pillar=None, checker_wall="x+", source="mushroom", meta: list[dict] | None = None, w=64, h=48, f=50.0) -> Path`. Room = axis-aligned box from (0,0,0) to `size` (y up); each cam = (position, yaw°, pitch°); depth is ray-cast against the box interior and an optional `pillar` box `((x0,y0,z0),(x1,y1,z1))`; images are flat grey 0.5 except a 0.1 m black/white checker on `checker_wall`; writes the MuSHRoom layout, or the Polycam processed layout (`transforms.json`, `images/`, `depth/`, `keyframes/cameras/<ts>.json` from `meta`, `keyframes/confidence/<ts>.png` all 255) when `source="polycam"`.

- [ ] **Step 1: Write the failing tests** in `tests/test_capcheck_capture.py`:
  - `test_mushroom_reader_frames_and_intrinsics`: 3 cams → 3 frames, `K[0,0] == 50`, `c2w` shape (4,4), `source == "mushroom"`, `has_tracking is False`.
  - `test_polycam_reader_attaches_keyframe_meta`: `meta` with `tracking_segment` 1,1,2 → `frames[2].meta["tracking_segment"] == 2`, `has_tracking is True`.
  - `test_exclude_drops_named_frames_after_matching`: exclude `{"frame_00002"}` → names `["frame_00001", "frame_00003"]` and `frames[1].meta` is the third keyframe's.
  - `test_frame_and_keyframe_count_mismatch_raises`: delete one keyframe json → `ValueError` whose message contains both counts (Review Focus 4).
  - `test_depth_in_metres_and_missing_is_zero`: a depth PNG pixel of 2500 → 2.5; 0 stays 0.
  - `test_confidence_resized_nearest`: a 4×3 confidence PNG for a 64×48 frame → shape (48, 64), values only in {0, 54, 255}.
  - `test_read_exclude_ignores_blank_lines`.
- [ ] **Step 2: Run** `python -m pytest tests/test_capcheck_capture.py -q` — expected: fails (module not found).
- [ ] **Step 3: Implement** `tests/synthroom.py` and `roomrecon/capcheck/capture.py`; add `"roomrecon.capcheck"` to `packages` in `pyproject.toml`. Polycam matching: processed frames sorted by name ↔ `keyframes/cameras/*.json` sorted by name (= timestamp); counts must match or raise. `weakly_connected` comes from `keyframes/corrected_cameras/<same name>.json` if it exists. Intrinsics from the per-frame `fl_x, fl_y, cx, cy, w, h` (both formats carry them per frame).
- [ ] **Step 4: Run** the tests — expected: 7 passed.
- [ ] **Step 5: Commit** `feat: capcheck capture readers (Polycam processed, MuSHRoom)`.

---

### Task 2: Surface points

**Files:**
- Create: `roomrecon/capcheck/surface.py`, `tests/test_capcheck_surface.py`

**Interfaces:**
- Consumes: `Capture`, `load_depth_m`, `load_confidence` (Task 1); `tests/synthroom.make_room`.
- Produces:
  - `backproject(depth_m: np.ndarray, K: np.ndarray, c2w: np.ndarray, stride: int = 4, mask: np.ndarray | None = None, min_d=0.1, max_d=5.0) -> np.ndarray` (N×3 world points).
  - `Surface` dataclass: `points: np.ndarray` (N×3), `normals: np.ndarray` (N×3 unit), `voxel: float`.
  - `build_surface(cap: Capture, voxel: float = 0.03, stride: int = 4) -> Surface` — all frames' back-projected points (confidence 255 where available), merged per voxel (centroid, plus the mean position of the cameras that produced the voxel's points), normals = smallest-eigenvalue eigenvector of the 16 nearest neighbours' covariance (`scipy.spatial.cKDTree`), flipped so `dot(n, cam_mean − p) > 0`.

- [ ] **Step 1: Write the failing tests:**
  - `test_backproject_matches_the_projection_convention`: K with f=50, c=(32,24), identity c2w, depth 2 m everywhere → the centre pixel back-projects to (0, 0, −2); pixel (42, 24) to x = +0.4.
  - `test_surface_points_lie_on_the_box_walls`: 8 cams inside a 4×2.5×3 room → every point within 0.02 m of some wall plane.
  - `test_normals_face_into_the_room`: for points within 0.05 m of the x=0 wall, normal x-component > 0.9.
  - `test_voxel_merge_one_point_per_voxel`: no two points closer than 0.5·voxel along all axes in the same voxel (unique `floor(p/voxel)` keys).
- [ ] **Step 2: Run** `python -m pytest tests/test_capcheck_surface.py -q` — expected: fail.
- [ ] **Step 3: Implement** `roomrecon/capcheck/surface.py`.
- [ ] **Step 4: Run** — expected: 4 passed.
- [ ] **Step 5: Commit** `feat: capcheck surface points from LiDAR depth`.

---

### Task 3: Views, coverage and texture

**Files:**
- Create: `roomrecon/capcheck/views.py`, `tests/test_capcheck_views.py`

**Interfaces:**
- Consumes: Tasks 1–2.
- Produces:
  - `fibonacci_sphere(n: int = 64) -> np.ndarray` (n×3 unit).
  - `project(points: np.ndarray, K: np.ndarray, c2w: np.ndarray, w: int, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]` — returns `(idx, uv, z)`: indices of points in front of the camera and inside the image, their pixel coords (float), their z-depth (positive metres).
  - `visible(points, frame: Frame, depth_m: np.ndarray, max_d=5.0, tol_abs=0.05, tol_rel=0.03) -> tuple[np.ndarray, np.ndarray]` — `(idx, uv)` of points passing the depth-agreement test `|z − D(u,v)| ≤ max(tol_abs, tol_rel·D)` with `0 < D` and `z ≤ max_d`.
  - `texture_map(gray: np.ndarray) -> np.ndarray` — Sobel magnitude (`scipy.ndimage.sobel` on both axes, `hypot`), then `scipy.ndimage.uniform_filter(size=9)`.
  - `ViewStats` dataclass: `n_views` (N int32), `dir_mask` (N uint64), `n_dirs` (N int32), `best_footprint_mm` (N float32, inf if unseen), `texture` (N float32, NaN if unseen), `mean_view_dir` (N×3, unit, zeros if unseen; direction from point to camera).
  - `accumulate(surface: Surface, cap: Capture) -> ViewStats` — per frame: `visible`, add 1 to `n_views`, OR the bit of the nearest Fibonacci direction of (camera − point) into `dir_mask`, footprint = 1000·distance/fx, collect `(idx, texture_map(load_gray)[v,u])`; at the end median texture per point (sort by idx), `n_dirs` = popcount of `dir_mask` via `np.unpackbits(dir_mask.view(np.uint8)).reshape(-1, 64).sum(1)`.

- [ ] **Step 1: Write the failing tests:**
  - `test_project_roundtrips_backproject`: back-projected pixels project back to the same (u, v) within 1e-6.
  - `test_occluded_points_are_not_seen`: room with a pillar between camera A and the x=4 wall → wall points behind the pillar have `n_views == 0` for a capture of camera A only, while points beside it have 1.
  - `test_n_views_counts_cameras`: 6 cameras all seeing the floor centre → that point's `n_views == 6`.
  - `test_n_dirs_grows_with_spread`: 6 cameras at the same spot → `n_dirs` of a wall point ≤ 2; 6 cameras spread 120° around it → ≥ 3.
  - `test_texture_high_on_checker_low_on_flat`: median texture of points on the checker wall > 10× that of points on a flat wall.
  - `test_all_zero_depth_frame_sees_nothing` (Review Focus 1): one frame's depth file all zeros → `accumulate` runs, no NaN in `n_views`, that frame contributes 0 views.
- [ ] **Step 2: Run** `python -m pytest tests/test_capcheck_views.py -q` — expected: fail.
- [ ] **Step 3: Implement** `roomrecon/capcheck/views.py` (loop over frames; vectorised inside a frame).
- [ ] **Step 4: Run** — expected: 6 passed.
- [ ] **Step 5: Commit** `feat: capcheck views, COVER-style direction coverage and texture`.

---

### Task 4: Tracking health

**Files:**
- Create: `roomrecon/capcheck/tracking.py`, `tests/test_capcheck_tracking.py`

**Interfaces:**
- Consumes: `Capture`, `Frame.meta`, `Frame.c2w`.
- Produces:
  - `Tracking` dataclass: `available: bool`, `segments: list[int]` (per frame), `segment_changes: list[str]` (frame names where the segment changes), `jumps: list[dict]` (`{"frame", "speed_mps", "rot_dps"}`), `limited: list[str]`, `weakly_connected: list[str]`, `bad_timestamps: list[str]`, `broken: frozenset[str]` (split the frame sequence into pieces at every segment change and every jump; every frame outside the largest piece is broken, ties keep the earliest piece), `blur_score: np.ndarray | None`, `angular_velocity: np.ndarray | None`.
  - `check_tracking(cap: Capture) -> Tracking` — `available=False` and empty lists for MuSHRoom; speed = distance between consecutive camera centres / Δt (µs → s); rotation = angle of `R_prev.T @ R_cur` / Δt; Δt ≤ 0 ⇒ `bad_timestamps`, no speed computed.

- [ ] **Step 1: Write the failing tests:**
  - `test_segment_change_and_shorter_segment_broken`: segments 1×8 then 2×3 → `segment_changes == ["frame_00009"]`, the 3 frames of segment 2 in `broken`.
  - `test_pose_jump_flagged`: 10 frames, positions advance 0.05 m per 0.2 s, the step into frame 8 jumps 2 m → one jump at `frame_00008`, speed ≈ 10 m/s, `broken == {frame_00008, frame_00009, frame_00010}`.
  - `test_limited_and_weak_frames_listed`.
  - `test_mushroom_capture_has_no_tracking` (Review Focus 2) and `test_missing_corrected_cameras_is_partial`: Polycam without `corrected_cameras/` → `available` True, `weakly_connected == []`.
  - `test_repeated_timestamp_is_flagged_not_divided` (Review Focus 5).
- [ ] **Step 2: Run** `python -m pytest tests/test_capcheck_tracking.py -q` — expected: fail.
- [ ] **Step 3: Implement** `roomrecon/capcheck/tracking.py`.
- [ ] **Step 4: Run** — expected: 6 passed.
- [ ] **Step 5: Commit** `feat: capcheck tracking health from Polycam per-frame data`.

---

### Task 5: Labels, regions, verdict

**Files:**
- Create: `roomrecon/capcheck/labels.py`, `tests/test_capcheck_labels.py`

**Interfaces:**
- Consumes: `Surface`, `ViewStats`, `Tracking`, `accumulate`'s per-frame visibility (for "broken" points: points seen only by broken frames). Add to Task 3's `accumulate` an optional `frames_mask: np.ndarray | None` so it can be re-run on unbroken frames only; a point is BROKEN if `n_views_all > 0` and `n_views_unbroken == 0`.
- Produces:
  - Constants `OK, UNDER, TEXTURELESS, BROKEN = 0, 1, 2, 3`; `TAU_TEX: float` (provisional `0.02`; Task 8 replaces it with the calibrated value).
  - `label_points(n_views: np.ndarray, n_dirs: np.ndarray, texture: np.ndarray, broken_pts: np.ndarray, tau_tex: float = TAU_TEX) -> np.ndarray` (N uint8; `broken_pts` N bool) — priority broken > under (`n_views < 5` or `n_dirs < 3`) > textureless (`texture < tau_tex`, NaN never) > OK. Takes arrays, not `ViewStats`, so Task 8 can relabel from `points.npz`.
  - `regions(points, labels, voxel=0.03, link=0.10, min_area=0.25) -> list[dict]` — connected components of UNDER points (`cKDTree.query_pairs(link)` + `scipy.sparse.csgraph.connected_components`); each `{"n_points", "area_m2", "centre", "y_range", "size", "seen_from"}` (in the aligned frame, see Task 6; `seen_from` = compass sector of the mean view direction, or "nowhere").
  - `floor_not_seen(points_aligned: np.ndarray, floor_y: float, cell=0.10, band=0.30) -> dict` — `{"area_m2", "cells"}`: cells inside the convex hull (`scipy.spatial.ConvexHull` of x/z of points within `band` of the floor) with no such point.
  - `verdict(tracking: Tracking, regs: list[dict]) -> str` — `"BROKEN TRACKING"` if `segment_changes` or `jumps`, else `"RECAPTURE THESE AREAS"` if `regs`, else `"OK"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_label_priority`: a point that is broken, under and textureless → BROKEN; under + textureless → UNDER.
  - `test_unseen_point_is_under_not_textureless` (Review Focus 3): `n_views 0`, texture NaN → UNDER.
  - `test_regions_drop_small_patches`: 400 under points in a 0.6×0.6 m patch → one region with `area_m2 ≈ 0.36`; 50 isolated under points → none.
  - `test_floor_not_seen_finds_a_hole`: floor points on a 3×3 m square with a 1×1 m gap → `area_m2` within 0.2 of 1.0.
  - `test_verdict_rules`.
- [ ] **Step 2: Run** `python -m pytest tests/test_capcheck_labels.py -q` — expected: fail.
- [ ] **Step 3: Implement** `roomrecon/capcheck/labels.py` and the `frames_mask` option in `views.accumulate` (with a test there: `frames_mask` all False → `n_views` all 0).
- [ ] **Step 4: Run** `python -m pytest tests -q` — expected: all pass.
- [ ] **Step 5: Commit** `feat: capcheck labels, weak regions, unseen floor and verdict`.

---

### Task 6: Outputs and CLI

**Files:**
- Create: `roomrecon/capcheck/outputs.py`, `roomrecon/capcheck/__main__.py`, `scripts/capcheck.sh`, `tests/test_capcheck_outputs.py`

**Interfaces:**
- Consumes: everything above; `roomrecon.scene_cli.build_scene(c2w, verts, room=, run_id=)` and `viewer_settings(frame)`; `roomrecon.scene.PLAYCANVAS_FROM_PLY`.
- Produces:
  - `align(points: np.ndarray, frame: dict) -> np.ndarray` — capture frame → aligned PlayCanvas frame: `s · R · (F · p)` with `F = PLAYCANVAS_FROM_PLY`, `R = frame["rotation"]`, `s = frame["scale"]` (the same transform `collision_room.ps1` applies to splats).
  - `write_splat_ply(path, points_capture_frame, colors, frame, size=0.015)` — writes positions `F · align(p)` (so the viewer's own 180° turn shows them aligned), `f_dc_k = (rgb − 0.5)/0.28209479`, `opacity = 4.0` (logit), `scale_k = ln(size)`, `rot = (1, 0, 0, 0)`; binary little-endian 3DGS PLY with properties `x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3`.
  - `write_topdown(path, points_aligned, labels, cam_aligned, segments, jump_idx)` — matplotlib (Agg): 0.10 m cells coloured by the worst label in the column (BROKEN > UNDER > TEXTURELESS > OK), camera path coloured by segment, jumps as black ×, metre grid, legend.
  - `write_report(dir, result: dict)` — `report.json` (the dict) and `report.txt` (verdict first, then per-label area m² and %, regions in words, unseen floor, tracking lists, settings, inputs).
  - CLI `python -m roomrecon.capcheck CAPTURE --out DIR [--exclude FILE] [--tau-tex X] [--name NAME]` → runs Tasks 1–5 (a second `accumulate` pass with `frames_mask` = unbroken frames only when `tracking.broken` is non-empty; `broken_pts = (n_views > 0) & (n_views_unbroken == 0)`; `points.npz` also stores `broken_pts`), writes `report.*`, `coverage.ply`, `topdown.png`, `scene.json`, `settings.json`, `points.npz` (`points, normals, n_views, n_dirs, texture, footprint, labels`), prints the verdict and the run time.
  - `scripts/capcheck.sh CAPTURE OUT [args]` — `cd ~/roomrecon/repo && "$PIXI" run --manifest-path "$DNS/pixi.toml" python -m roomrecon.capcheck "$@"`.

- [ ] **Step 1: Write the failing tests:**
  - `test_splat_ply_header_and_count`: 10 points → header lists the 14 properties in order, `element vertex 10`, file size = header + 10·14·4 bytes.
  - `test_align_puts_floor_at_floor_y`: synthetic room → after `align`, the 5th-percentile y of floor points within 0.03 of `frame["floor_y"]`.
  - `test_cli_end_to_end_on_synthetic_room`: `main([room, "--out", tmp])` → returns 0, all six files exist, `report.json["verdict"]` in the three verdicts.
- [ ] **Step 2: Run** `python -m pytest tests/test_capcheck_outputs.py -q` — expected: fail.
- [ ] **Step 3: Implement** `outputs.py`, `__main__.py`, `scripts/capcheck.sh` (executable bit: `git update-index --chmod=+x`).
- [ ] **Step 4: Run** `python -m pytest tests -q` — expected: all pass.
- [ ] **Step 5: Commit** `feat: capcheck CLI, coverage splat, floor plan and report`.

---

### Task 7: First real runs (GPU machine)

No new code unless a run exposes a bug (then: failing test first, fix, commit). Push with `git push gpu main` first.

- [ ] **Step 1: cap3 onto the GPU machine.** `scp "E:\roomrecon-scenes\own\zips\10_7_2026 3.zip"` to `~/roomrecon/datasets/own/sejal-cap3/`, then in the pixi env `ns-process-data polycam --data <zip> --output-dir ~/roomrecon/datasets/own/sejal-cap3/processed --use-depth` (the same command used for cap2; check `ls processed/keyframes`). Expected: 491 keyframes.
- [ ] **Step 2: Pose-convention check on real data.** First, frame ↔ keyframe matching (cap2, cap3): each processed `c2w[:3, 3]` equals the matched `corrected_cameras` translation in nerfstudio's Polycam row order, `(t_23, t_03, t_13)`, within 1 mm (`polycam_utils.py:78-80`); a mismatch means the order assumption is wrong — stop and fix the reader. Then, for cap2 and `vr_room` long capture: back-project frame i (stride 8), run `visible` against frame i+2; expected: ≥ 50 % of frame i's points visible in frame i+2 and median `|z − D|` < 0.03 m. If not, the projection convention is wrong for that source: stop, find it, add a test, fix.
- [ ] **Step 3: Run the check** with `scripts/capcheck.sh` on cap1 (`sejal-2026-10-07`), cap2, cap3 and `vr_room/iphone/long_capture`, outputs `~/roomrecon/capcheck/{cap1,cap2,cap3,vr_room}`. Expected: each under 5 min (print time); verdicts recorded against the spec's expectations (cap3 BROKEN TRACKING, cap1 RECAPTURE, cap2 and `vr_room` not broken) — reported as found, not adjusted.
- [ ] **Step 4: blur_score direction.** Copy the 3 lowest and 3 highest `blur_score` frames of cap2 to the PC and look at them; write which end is sharp into `docs/report-notes.md`.
- [ ] **Step 5: Look at the outputs on the PC.** Fetch `report.txt`, `topdown.png`, `coverage.ply`, `settings.json` for cap2 and cap3 to `E:\roomrecon-scenes\capcheck\<name>\`; open cap2's `coverage.ply` in the local viewer (`http://localhost:3000/?content=...coverage.ply&settings=...settings.json`); check the map is upright and the red/blue areas are plausible (e.g. under the beds, plain walls). Show the user.
- [ ] **Step 6: Commit** any fixes; add the four verdicts and run times to `docs/report-notes.md` §3 and §2 (decisions such as the blur direction); commit `exp: capcheck first runs on cap1-3 and vr_room`.

---

### Task 8: Validation against the trained splats

**Files:**
- Create: `roomrecon/capcheck/validate.py`, `scripts/capcheck_validate.sh`, `tests/test_capcheck_validate.py`
- Modify: `roomrecon/capcheck/labels.py` (`TAU_TEX` = calibrated value)

**Interfaces:**
- Consumes: `load_capture`, `backproject`, `load_depth_m`, `points.npz` from Task 6.
- Produces:
  - `patch_table(cap: Capture, test_names: list[str], renders_rgb: Path, renders_gt: Path, npz: dict, labels: np.ndarray, patch=16, radius=0.03) -> dict[str, np.ndarray]` — per 16×16 patch with ≥ 50 % of its stride-4 samples matched: `frame`, `error` (mean |render − gt|/255 over RGB and pixels), `label` (majority of the matched points' `labels`), `n_views`, `n_dirs`, `texture` (medians). Render/gt files found by stem under each dir (`rglob`). The caller passes `labels = label_points(npz["n_views"], npz["n_dirs"], npz["texture"], npz["broken_pts"], tau)`; for calibration use `tau = 0` (no TEXTURELESS yet, UNDER/BROKEN unaffected).
  - `bootstrap_mean_ci(values, groups, n=1000, seed=0) -> tuple[float, float, float]` — mean and 95 % interval resampling groups (frames).
  - `auc(scores, positives) -> float` — Mann–Whitney via `scipy.stats.rankdata`.
  - `fit_logistic(X, y) -> np.ndarray` and `predict_logistic(w, X) -> np.ndarray` — with a bias column, `scipy.optimize.minimize` (L-BFGS-B, L2 1e-3), standardised features.
  - `calibrate_tau(table) -> float` — among patches with label ≠ UNDER, the texture threshold (candidates = the 1st…99th percentiles of texture) maximising Youden's J for `texture < τ` vs `bad` (top 20 % error).
  - CLI `python -m roomrecon.capcheck.validate --calib NAME:CAPTURE:NPZ:RGB:GT --test NAME:CAPTURE:NPZ:RGB:GT --out DIR` → `validation.json` (τ, per-label mean error + CI per room, AUCs coverage/texture/both in both directions, pass/fail of the two pre-agreed criteria), `error_by_label.png` (box plot per room), `examples.png` (2 test frames: ground truth | render | error | labels).

- [ ] **Step 1: Write the failing tests:**
  - `test_auc_perfect_and_random`: perfect scores → 1.0; constant scores → 0.5.
  - `test_bootstrap_ci_contains_mean_and_narrows_with_more_groups`.
  - `test_calibrate_tau_separates_synthetic_patches`: bad patches have texture 0.01, good 0.1 → τ between them.
  - `test_logistic_recovers_sign`: y = (x > 0) → positive weight on x.
  - `test_patch_table_on_synthetic_room`: render = gt except a bright square over the checker wall → the patches there have the highest error and label TEXTURELESS is absent on the checker wall.
- [ ] **Step 2: Run** `python -m pytest tests/test_capcheck_validate.py -q` — expected: fail.
- [ ] **Step 3: Implement** `validate.py` and `scripts/capcheck_validate.sh`.
- [ ] **Step 4: Run** `python -m pytest tests -q` — expected: all pass. Commit `feat: capcheck validation against held-out render errors`; `git push gpu main`.
- [ ] **Step 5: Training-frames-only runs (GPU machine).** Exclude files: `vr_room` = `long_capture/test.txt`; cap2 = stems of `~/roomrecon/scenes/sejal-cap2/dns-30k-lite/renders/test/rgb/*.jpg`. Run `scripts/capcheck.sh` with `--exclude` into `~/roomrecon/capcheck/{vr_room,cap2}-train`.
- [ ] **Step 6: Short-capture alignment check.** Back-project 10 `short_capture` frames of `vr_room`; median distance to the long-capture surface points (`cKDTree`) < 0.05 m ⇒ include the short capture as a third test set; otherwise drop it and note why.
- [ ] **Step 7: Validate.** `--calib vr_room` (renders `~/roomrecon/scenes/vr_room/dns-30k-lite/renders/test/{rgb,gt-rgb}/long_capture/images`) `--test cap2` (renders `.../sejal-cap2/dns-30k-lite/renders/test/{rgb,gt-rgb}`) into `~/roomrecon/capcheck/validation/`. Then set `TAU_TEX` in `labels.py` to the calibrated τ (rounded to 3 significant digits) and re-run cap2-train/cap1/cap3 checks so their reports use it.
- [ ] **Step 8: Record.** Copy `validation.json` and the two figures to `results/g8-validation.json`, `results/g8-error-by-label.png`, `results/g8-examples.png`; write `results/g8-capture-check.md` (setup, machine, τ, the table, both criteria with PASS/FAIL as measured, the four verdicts, run times, limitations); add decisions and numbers to `docs/report-notes.md`; commit `exp: G8 capture check validated on vr_room and sejal-cap2`.

---

### Task 9: Demo material and wrap-up

- [ ] **Step 1:** On the PC, `E:\roomrecon-scenes\capcheck\cap2\` and `\cap3\` hold `coverage.ply`, `settings.json`, `topdown.png`, `report.txt`; check cap2's map opens in the viewer and cap3's report shows the broken tracking. Copy the two `topdown.png` to `results/g8-topdown-cap2.png`, `results/g8-topdown-cap3.png`.
- [ ] **Step 2:** Update `CLAUDE.md` "Current status" (G8 done, where the check lives, how to run it) and `docs/report-notes.md` §6.
- [ ] **Step 3:** Commit `docs: G8 capture check results and demo material`; ask the user, then `git push origin main` (and `git push gpu main`).
- [ ] **Optional (only with time left, ask the user first):** train cap3 with its `broken` frames excluded (one A100 run with `scripts/own_room.sh`) and compare with the check's advice.
