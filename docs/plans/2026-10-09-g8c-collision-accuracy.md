# G8 (third attempt): Collision Accuracy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure walk-collision accuracy against `vr_room`'s laser scan for collision built from the splat (S, the default), DN-Splatter's mesh (M), the phone's LiDAR (L) and the scan itself (GT), all through the demo's `splat-transform` pipeline, and decide P1–P4.

**Architecture:** New package `roomrecon/collision/`: `glb.py` reads `splat-transform`'s `.collision.glb` and samples its triangles; `grid.py` turns samples into a top-down body-height occupancy grid and computes IoU / phantom / missed / offset; `prep.py` writes point sets as splat PLYs; `__main__.py` runs the comparison. The laser scan and mesh are sampled on the GPU machine (open3d); `splat-transform` and the evaluation run on the PC via `scripts/collision_study.ps1`.

**Tech Stack:** Python (PC 3.12 numpy 2.4 scipy 1.17 matplotlib; GPU machine pixi env with open3d for sampling), node `@playcanvas/splat-transform` (PC), pytest.

**Spec:** `docs/spec/2026-10-09-g8c-collision-accuracy-design.md`.

## Global Constraints

- Branch `g8-collision`. The `vr_room` demo files (`E:\roomrecon-scenes\vr_room\`, `demo-vr_room`, release) are read only; all outputs go to `E:\roomrecon-scenes\collision-study\`.
- No new Python packages; no open3d on the PC.
- Frame: the demo's aligned frame from its `scene.json` (`capcheck.outputs.align`); lengths and areas divided by `scene["scale"]` (and its square) to report metres / m².
- Collision pipeline exactly as `scripts/collision_room.ps1`: `splat-transform -w SRC -r "<euler_xyz_deg>" -s "<scale>" aligned.ply`, then `splat-transform -w aligned.ply --seed-pos "<spawn_position>" --filter-cluster --voxel-size 0.05 --voxel-external-fill --voxel-carve --collision-mesh room.voxel.json` (writes `room.collision.glb`). Comma arguments quoted.
- Point sources as splats: opaque round splats of 0.02 m, written with `capcheck.outputs.write_splat_ply(path, points, colors, {"rotation": eye(3), "scale": 1.0}, size=0.02)` (positions stay in the PLY frame).
- Sampling: ~1 cm on surfaces (count = area / 1e-4, capped at 4,000,000) for the scan, the mesh and the collision triangles.
- Grid: 5 cm cells, body slab 0.2–1.8 m above the floor (×scale), floor = `scene.floor_height` of the aligned GT points with the spawn height as the camera height; spawn = `scene["spawn_position"]`; grid bounds = GT points' x/z extent ± 0.5 m (×scale).
- Metrics (spec §3): IoU of walkable areas (free cells 4-connected to the spawn); phantom = source-occupied cells inside GT's walkable area > 0.10 m from GT-occupied cells (m²); missed = observed GT-occupied cells (an L-occupied cell within 0.10 m) with no source-occupied cell within 0.10 m (%); offset = distances of source-occupied cells within 0.5 m of GT-occupied cells to the nearest GT-occupied cell (median, p95, cm). Distances via `scipy.ndimage.distance_transform_edt`.
- Criteria: P1 IoU(M) ≥ IoU(S) + 0.02; P2 phantom(M) ≤ phantom(S); P3 missed(M) ≤ missed(S) + 2 points; P4 median offset(M) ≤ median offset(S). Same for L vs S (secondary). Reported as measured.
- Ask the user before `git push origin`; record decisions and numbers in `docs/report-notes.md`.

## Review Focus

1. A `.glb` with 16-bit indices, several primitives, or a node transform → all triangles read in world positions (test in Task 1).
2. The spawn falls in an occupied cell → the flood fill starts from the nearest free cell (test in Task 2).
3. A source with points far outside the room (floaters) → ignored outside the grid bounds, no crash (test in Task 2).
4. A source with no points in the body slab → all-free grid, metrics still computed (test in Task 2).
5. The real `splat-transform` output differs from the assumed GLB layout → Task 4 Step 2 reads the real file first and stops if it cannot.

---

### Task 1: GLB reader and triangle sampling

**Files:** Create `roomrecon/collision/__init__.py` (empty), `roomrecon/collision/glb.py`, `tests/test_collision_glb.py`.

**Interfaces — Produces:**
- `read_glb(path) -> tuple[np.ndarray, np.ndarray]` — (vertices N×3 float64 in world coordinates, faces M×3 int64) from every mesh primitive of every node (TRIANGLES mode); applies node `matrix` or `translation`/`rotation`(quaternion xyzw)/`scale`; indices of component type 5121/5123/5125; non-indexed primitives → consecutive triples.
- `sample_triangles(V, F, spacing=0.01, seed=0, cap=4_000_000) -> np.ndarray` — area-weighted uniform samples, count = min(cap, max(1, round(total_area / spacing²))).
- Test helper `tests/glbwrite.py`: `write_glb(path, V, F, index_type=5125, translation=None)` (minimal glTF 2.0 binary: one mesh, one primitive, optional node translation).

- [ ] **Step 1: Failing tests:** `test_glb_round_trip_uint32` (two triangles of a 1 × 2 m quad → same V, F); `test_glb_uint16_and_node_translation` (indices 5123, translation (1, 2, 3) → vertices shifted); `test_sample_triangles_count_and_plane` (1 × 2 m quad at y = 0.5, spacing 0.01 → 20,000 ± 1 points, all y = 0.5, all within the quad).
- [ ] **Step 2: Run** `python -m pytest tests/test_collision_glb.py -q` — expected: fail (module not found).
- [ ] **Step 3: Implement** `roomrecon/collision/glb.py` and `tests/glbwrite.py`; add `"roomrecon.collision"` to `pyproject.toml` packages.
- [ ] **Step 4: Run** — expected: 3 passed.
- [ ] **Step 5: Commit** `feat: read splat-transform collision GLBs and sample their triangles`.

---

### Task 2: Occupancy grid and metrics

**Files:** Create `roomrecon/collision/grid.py`, `tests/test_collision_grid.py`.

**Interfaces — Produces:**
- `Grid` dataclass: `lo: np.ndarray` (x, z of cell (0, 0)), `shape: tuple[int, int]`, `cell: float`.
- `make_grid(xz: np.ndarray, cell=0.05, margin=0.5) -> Grid` — bounds = min/max of the given x/z points (the aligned GT points) ± `margin`.
- `occupancy(points, grid, floor_y, slab=(0.2, 1.8), scale=1.0) -> np.ndarray` (bool, `grid.shape`) — points with `floor_y + slab[0]·scale ≤ y ≤ floor_y + slab[1]·scale` inside the grid mark their cell.
- `walkable(occ, spawn_xz, grid) -> np.ndarray` — free cells 4-connected to the spawn cell (nearest free cell if the spawn cell is occupied).
- `compare(occ_src, occ_gt, occ_lidar, spawn_xz, grid, scale=1.0, tol=0.10, near=0.5) -> dict` → `{"iou", "phantom_m2", "missed_pct", "offset_median_cm", "offset_p95_cm", "walkable_m2"}` per the Global Constraints.

- [ ] **Step 1: Failing tests** (synthetic room: a 4 × 3 m wall ring one cell thick at 5 cm cells, spawn in the centre; `occ_lidar = occ_gt`):
  - `test_identical_source_is_perfect`: iou 1.0, phantom 0, missed 0, offset median 0.
  - `test_phantom_block_area`: source = GT + a 0.5 × 0.5 m block mid-room → phantom_m2 == 0.25 (±0.0025), iou < 1.
  - `test_missing_wall_segment`: source = GT minus 1 m of one wall → missed_pct ≈ 100 × 20 / (GT wall cells) (±1), walkable area of the source larger than GT's.
  - `test_walls_shifted_inward`: source walls moved 0.10 m inward → offset median 10 cm (±1), missed 0 (within tolerance).
  - `test_spawn_in_occupied_cell_uses_nearest_free` (Review Focus 2), `test_far_points_ignored` (Review Focus 3: a point 50 m away → same occupancy), `test_empty_slab_all_free` (Review Focus 4: all points at 3 m above the floor → occupancy all False, `compare` returns finite iou).
- [ ] **Step 2: Run** `python -m pytest tests/test_collision_grid.py -q` — expected: fail.
- [ ] **Step 3: Implement** `roomrecon/collision/grid.py`.
- [ ] **Step 4: Run** — expected: 7 passed.
- [ ] **Step 5: Commit** `feat: body-height occupancy grid and collision accuracy measures`.

---

### Task 3: Point splats, comparison CLI and the PC script

**Files:** Create `roomrecon/collision/prep.py`, `roomrecon/collision/__main__.py`, `scripts/collision_study.ps1`, `tests/test_collision_cli.py`.

**Interfaces:**
- Consumes: Tasks 1–2; `capcheck.outputs.write_splat_ply`, `align`; `scene.floor_height`.
- Produces:
  - `prep.points_to_splats(npz_path, ply_path, key="points", size=0.02)` — grey splats (Global Constraints); CLI `python -m roomrecon.collision.prep NPZ PLY [--key K]`.
  - CLI `python -m roomrecon.collision --glb NAME=PATH (repeat; names S, M, L, GT) --gt-points NPZ --lidar-points NPZ --scene SCENE_JSON --out DIR` → aligns the GT and LiDAR points (`align` with the scene), floor, grid, spawn; occupancy of every GLB's samples and of the LiDAR points; `compare` of every source against GT's GLB; criteria P1–P4 for M vs S and the same four for L vs S; writes `metrics.json` (`{"floor_y", "scale", "grid", "sources": {name: metrics}, "criteria": {"M_vs_S": {P1..P4: bool}, "L_vs_S": {...}}}`) and `topdown.png` (one panel per source: its occupied cells over GT's in grey, walkable area shaded).
  - `scripts/collision_study.ps1 -Study DIR -SceneDir DIR` — runs `prep` for `mesh_points.npz`, `lidar_points.npz`, `gt_points.npz` (in `-Study`); for S (`<SceneDir>\splat.ply`), M, L, GT runs the two `splat-transform` calls into `<Study>\<name>\`; then the CLI with `--out <Study>\results`.
- [ ] **Step 1: Failing tests:** `test_points_to_splats_header` (10 points → `element vertex 10`, positions unchanged); `test_cli_end_to_end` (four synthetic GLBs from `tests/glbwrite.write_glb`: GT = a box room, S = GT + a 0.5 m block, M = GT, L = GT; frame with `rotation = PLAYCANVAS_FROM_PLY`, scale 1, spawn mid-room → `metrics.json` exists, `sources.M.phantom_m2 == 0`, `sources.S.phantom_m2 > 0.2`, `criteria.M_vs_S.P2` True, `topdown.png` exists).
- [ ] **Step 2: Run** `python -m pytest tests/test_collision_cli.py -q` — expected: fail.
- [ ] **Step 3: Implement** `prep.py`, `__main__.py`, `scripts/collision_study.ps1`.
- [ ] **Step 4: Run** `python -m pytest tests -q` — expected: all pass.
- [ ] **Step 5: Commit** `feat: collision accuracy CLI and study script`.

---

### Task 4: Run the study on `vr_room`

- [ ] **Step 1 (GPU machine, CPU):** in the pixi env with open3d: read `~/roomrecon/datasets/mushroom/room_datasets/vr_room/gt_mesh.ply`, transform by `inv(icp_iphone.json["gt_transformation"])`, `sample_points_uniformly(min(4e6, area/1e-4))` → `~/roomrecon/collision-study/gt_points.npz` (`points`, float32); same for `~/roomrecon/scenes/vr_room/dns-30k-lite/export/mesh.ply` (no transform) → `mesh_points.npz`; copy `~/roomrecon/capcheck/vr_room/points.npz` (full-capture check) → `lidar_points.npz`. Print counts and areas. `scp` the three to `E:\roomrecon-scenes\collision-study\`.
- [ ] **Step 2:** Read the demo's `room.collision.glb` with `read_glb` and check it lines up with the GT points aligned by `scene.json` (median distance from collision samples to the nearest aligned GT point < 0.1 m); if the reader fails or the frames disagree, stop and fix (Review Focus 5).
- [ ] **Step 3:** `.\scripts\collision_study.ps1 -Study E:\roomrecon-scenes\collision-study -SceneDir E:\roomrecon-scenes\vr_room\dns-30k-lite`. Expected: four `room.collision.glb`, `results\metrics.json`, `results\topdown.png`.
- [ ] **Step 4:** Look at `topdown.png`; write `results/g8c-collision-accuracy.md` (sources, machine, the metric table, P1–P4 for M and L as measured, limitations: one room, 5 cm grid, GT through the same pipeline), copy `metrics.json` and `topdown.png` to `results/g8c-collision-*.{json,png}`, add decisions and numbers to `docs/report-notes.md`; commit `exp: G8 collision accuracy on vr_room against the laser scan`.
