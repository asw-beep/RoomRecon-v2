# G8 novelty: capture check that says where a room capture will fail, and why

Status: design, approved in conversation 2026-10-08; this written spec awaits review.
Goal G8 of `2026-10-03-roomrecon-v2-design.md` §9. Prior work and the decision trail are in
`docs/report-notes.md` §2 and §5.

## 1. Claim

Right after a phone capture of a room, and before any training, a check that uses only what the
phone recorded (poses, LiDAR depth, images, per-frame tracking data) can say **where** the
walkable splat will be poor **and why**, and so whether capturing more would help:

| Cause | Advice |
|---|---|
| Under-covered: too few views, or views from too narrow a range of directions | Recapture this area, from other directions |
| Textureless: plain surface, well covered | More views will not fix it: move closer, add light; expect blur here |
| Broken tracking: tracking segments, pose jumps, weakly connected frames | Drop or redo this part of the capture |

Existing capture feedback (Zhang et al. 2026, Yasunaga et al. 2025, LiMRSF 2024, Pix4D) answers
"where to capture more"; none tells you when more capture will not help, and none checks an
exported phone room capture before training. The coverage measure itself is reused from COVER
(Chen et al. 2026), moved before training onto LiDAR geometry.

**Measured claim:** on real room captures, the under-covered and textureless labels each mark
regions where the trained splat's held-out error is higher than elsewhere, and they mark
*different* regions (texture adds predictive power over coverage alone).

## 2. Where it runs

Everything on the GPU machine (CPU only, the existing DN-Splatter pixi env: numpy, scipy,
Pillow, matplotlib, open3d; nothing new installed). The laptop only views the small outputs.

Workflow after a capture: Polycam raw export → laptop → `scp` to the GPU machine →
`ns-process-data polycam --use-depth` (already part of our pipeline) → `scripts/capcheck.sh` →
copy `report.txt`, `topdown.png`, `coverage.ply` back to view. Campus network only.

## 3. Inputs

A nerfstudio-style capture folder, as both of our data sources already provide:

- **Own captures** (`ns-process-data polycam` output): `transforms.json` (per-frame
  intrinsics, camera-to-world, OpenGL convention), `images/` (994×738, Polycam's corrected
  images), `depth/` (uint16 millimetres, image size), and `keyframes/` (Polycam raw:
  `cameras/*.json` with `tracking_segment`, `world_mapping_status`, `blur_score`,
  `angular_velocity`, `feature_point_count`, `timestamp`; `corrected_cameras/*.json` with
  `weakly_connected`, `neighbors`; `confidence/*.png` with values 0 / 54 / 255).
  Frames match keyframes in timestamp order (`ns-process-data` numbers them that way; checked
  in implementation by comparing poses).
- **MuSHRoom** (`vr_room/iphone/long_capture`): `transformations.json`, `images/`, `depth/`
  (uint16 mm), `test.txt`. No tracking data and no confidence: tracking checks are skipped,
  all depth pixels with 0 < depth ≤ 5 m are used.

`--exclude <list>` drops frames (the held-out test frames, for validation; see §7).

## 4. Method

### 4.1 Surface points
Back-project every frame's depth (every 4th pixel, confidence 255 where available, depth
0.1–5 m; Polycam's 256×192 confidence maps resized to the image with nearest neighbour) to
world points; merge into a 3 cm voxel grid (centroid per voxel). Normals by PCA of
the 16 nearest neighbours (scipy `cKDTree`), flipped to face the cameras that produced them.

### 4.2 Which frames see each point
For each frame, project all points; a point is seen if it lands inside the image, in front of
the camera, within 5 m, and its depth agrees with the frame's own depth map at that pixel
(|z − D| ≤ max(5 cm, 3 % of D)). This occlusion test uses the measured LiDAR depth, so walls
hidden behind furniture do not count as seen. Vectorised per frame; frames in parallel
(multiprocessing) if needed. Target: under 5 min for 500 frames.

### 4.3 Coverage (reused from COVER)
Per point: number of seeing frames `n_views`; the set of view directions binned on a fixed
64-direction sphere (≈ 25–30° per bin), giving `n_dirs` occupied bins; and the best pixel
footprint (distance / focal length, mm per pixel), reported but not used for labels.
**Under-covered** if `n_views < 5` (Pix4D's "5+ images" rule) or `n_dirs < 3`. Fixed a priori,
not tuned.

### 4.4 Texture
Per point and seeing frame: mean Sobel gradient magnitude of the grey image (0–1) in a 9×9
window at the projected pixel; per point the median over its views (robust to one blurry view).
**Textureless** if below τ_tex. τ_tex is the only tuned value: on `vr_room` (the calibration
room), among patches that are not under-covered, the τ that best separates bad patches (top
20 % error) from the rest (maximum Youden's J); applied unchanged to `sejal-cap2` (the test
room); §7.

### 4.5 Tracking health (own captures only)
Per frame, from Polycam's data: tracking-segment changes; `world_mapping_status = limited`;
`weakly_connected = true`; pose jumps, where camera speed between consecutive keyframes exceeds
1.5 m/s or rotation exceeds 180°/s (timestamps give the time). `blur_score` and
`angular_velocity` are reported; whether high `blur_score` means sharp is checked by eye on the
extreme frames before it is used for anything.
Frames flagged as broken are listed with their segment, and their points are labelled
**broken tracking** if no unflagged frame also sees them.

### 4.6 Labels, regions and verdict
Point label, in priority order: broken tracking > under-covered > textureless > OK
(under-covered and textureless points are also counted as "both" for the analysis).
Weak regions = connected components (10 cm neighbourhood) of under-covered points with area
≥ 0.25 m² (points × voxel area); each is described by centre, height range above the floor,
size, and the directions it was seen from ("seen only from the doorway side").
**Never seen:** coverage can only label surfaces the phone saw. Floor cells (10 cm) inside the
room outline (convex hull of the floor points in the top-down view) with no point within 30 cm
of the floor are reported as "floor not seen" (e.g. under beds), with their area; report only.
**Verdict:** BROKEN TRACKING if any segment change or pose jump; else RECAPTURE THESE AREAS if
any weak region; else OK. Textureless regions are warnings, never a recapture request.

### 4.7 Alignment
Up, floor and scale from `roomrecon.scene.scene_frame` (cameras + surface points), so the map
is upright, metric and in the viewer's frame, like the splat exports.

## 5. Outputs (per capture, one folder)

- `report.txt` / `report.json`: verdict, counts, per-label surface area and %, weak regions in
  words, tracking problems with frame numbers, settings and input paths.
- `coverage.ply`: surface points as small splats (3DGS PLY: position, colour as SH DC, high
  opacity, 1.5 cm isotropic scale) coloured green OK, red under-covered, blue textureless,
  grey broken tracking; already aligned (§4.7); opens in our SuperSplat viewer.
- `topdown.png`: floor plan, 10 cm cells coloured by the worst label in the column, the camera
  path coloured by tracking segment, jumps marked; for the report and slides.
- `points.npz`: per-point arrays (position, normal, n_views, n_dirs, texture, label) for the
  validation.

## 6. Code

New package `roomrecon/capcheck/`, small single-purpose modules, pure numpy/scipy:
`capture.py` (readers → one `Capture` structure), `surface.py` (4.1), `views.py` (4.2–4.4),
`tracking.py` (4.5), `labels.py` (4.6), `outputs.py` (§5), `__main__.py` (CLI:
`python -m roomrecon.capcheck <capture> --out <dir> [--exclude file]`), `validate.py` (§7).
Shell wrappers `scripts/capcheck.sh`, `scripts/capcheck_validate.sh` run them in the pixi env
on the GPU machine. Tests (`tests/test_capcheck_*.py`, pytest on the PC) use small synthetic
rooms: a box with cameras where views, directions, occlusion and jumps are known exactly; plus a
one-frame Polycam fixture for the reader and a PLY round trip.

## 7. Validation

**Data (no new training):** `vr_room` `dns-30k-lite` (train = long-capture frames not in
`test.txt`; test = the 42 held-out long-capture frames, plus the short capture as harder views if its
poses are in the long capture's frame, which is checked first)
and `sejal-cap2` `dns-30k-lite` (train 224; test the 33 every-8th frames). Renders and ground
truth already exist on the GPU machine.

**Procedure:** run the check on each room's **training frames only** (`--exclude` the test
frames). For each test frame: error per pixel = mean |render − ground truth| over RGB (0–1);
back-project the test frame's own depth, match each pixel to the nearest surface point within
3 cm (unmatched pixels dropped); pool into 16×16-pixel patches (patch error = mean, patch
label = majority, patch coverage/texture = median) to soften small misalignments.

**Statistics:** mean patch error per label with a bootstrap 95 % interval over test frames;
AUC for finding "bad" patches (the room's top 20 % error) using coverage alone, texture alone,
and both (logistic regression on `log n_views`, `n_dirs`, texture; fitted on `vr_room`,
evaluated on `sejal-cap2`, and the reverse reported too).

**Agreed before running (pass / fail):**
1. In the test room, under-covered patches and textureless patches each have higher mean error
   than OK patches, with non-overlapping intervals.
2. Texture adds at least 0.05 AUC over coverage alone.
If either fails, that is reported as the result (a negative or partial finding), not tuned away.

**Tracking check:** run on cap1, cap2, cap3 and `vr_room`. Expected before running: cap3 BROKEN
TRACKING (three segments, ~2 m jumps), cap1 RECAPTURE (32 frames), cap2 and `vr_room` not
broken. Reported as found.

**Optional, only if time remains:** train cap3 with its broken frames dropped (one A100 run)
to show the advice works.

## 8. Not doing

Live on-phone feedback (future work); learning a predictor; changing training; anything on the
`vr_room` demo files; Scaniverse/other capture apps.

## 9. Plan of work (deadline 2026-10-11)

1. Readers, surface, views, labels, outputs with tests (PC), then the first real run on cap2 on
   the GPU machine; look at `coverage.ply` and `topdown.png`.
2. Tracking checks; run on cap1/cap2/cap3/`vr_room`.
3. Validation run and figures; results to `results/g8-capture-check.md`, decisions and numbers
   to `docs/report-notes.md`.
4. Demo: cap2 coverage map in the viewer, cap3 report.
