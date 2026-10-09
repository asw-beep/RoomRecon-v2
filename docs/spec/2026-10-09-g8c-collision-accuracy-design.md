# G8 (third attempt): how accurate is walk collision built from a splat?

Status: design approved in conversation 2026-10-09; this written spec awaits review.
Earlier G8 attempts: the capture check (fallback, `2026-10-08-g8-capture-check-design.md`) and the
tracking repair (premise failed: cap3 was already aligned, `results/g8b-cap3-repair.json`).

## 1. Claim

Walk-mode collision for a captured room is normally built from the splat's Gaussians
(`splat-transform --collision-mesh`, used by SuperSplat and by our demo). Collision built from
DN-Splatter's depth-guided mesh instead (or from the phone's LiDAR directly) is **closer to the
real room**, measured against MuSHRoom's laser scan of `vr_room`. No published accuracy numbers
exist for splat-derived walk collision (splat-transform docs; Splat-Nav and NavGSim evaluate robot
maps, not walk collision against a laser scan).

## 2. Sources (one identical collision pipeline)

| | Geometry | Origin |
|---|---|---|
| S | the trained splat `splat.ply` as is (= the demo's collision source) | DN-Splatter `dns-30k-lite` |
| M | points sampled on DN-Splatter's mesh `mesh.ply` (`gs-mesh o3dtsdf`, 2 cm) | same model |
| L | the phone's LiDAR surface: capture check `build_surface` of the whole long capture (3 cm voxels) | no training |
| GT | points sampled on the laser scan `gt_mesh.ply`, moved into the model's frame with `inv(icp_iphone.json)` (as DN-Splatter's mesh evaluation, G3) | reference |

M, L and GT are written as opaque round splats (2 cm, `capcheck.outputs.write_splat_ply` with an
identity frame, so positions stay in the PLY frame). All four go through exactly the demo's
pipeline (`scripts/collision_room.ps1`): `splat-transform -r <euler> -s <scale>` with the demo's
`scene.json` (read only), then `--seed-pos <spawn> --filter-cluster --voxel-size 0.05
--voxel-external-fill --voxel-carve --collision-mesh`. The demo files are never modified; all
outputs go to `E:\roomrecon-scenes\collision-study\`.

Mesh and laser-scan sampling: GPU machine (open3d, the 328 MB scan lives there), ~1 cm spacing
(points = surface area / 1e-4, capped at 4 million). Everything else: the PC (`splat-transform`
needs node; inputs are a few tens of MB).

## 3. Evaluation

Frame: the demo's aligned frame (y up); lengths divided by the demo's scale (1.0378) to report
metres. Floor: the laser scan's floor (`scene.floor_height` on the GT points), not the demo's.

Each collision mesh (`.collision.glb`, read with a small GLB reader) is sampled at ~1 cm on its
triangles; points between 0.2 m and 1.8 m above the floor mark occupied cells of a top-down
5 cm grid (the body of a walking person). Spawn = the demo's spawn position.

- **Walkable area**: free cells reachable from the spawn (4-connected flood fill) inside the GT
  room's bounding box. **IoU** of a source's walkable area with GT's.
- **Phantom obstacles**: cells occupied in the source, inside GT's walkable area, more than
  10 cm from any GT-occupied cell; area in m².
- **Missed walls**: observed GT-occupied cells (an L point within 10 cm, i.e. the phone saw that
  wall) with no source-occupied cell within 10 cm; share in %.
- **Wall offset**: for source-occupied cells within 0.5 m of a GT-occupied cell, the distance to
  the nearest GT-occupied cell; median and 95th percentile in cm.

## 4. Pass / fail (fixed before any run)

Positive result = M beats S on all four:

| | Criterion |
|---|---|
| P1 | IoU(M) ≥ IoU(S) + 0.02 |
| P2 | phantom area(M) ≤ phantom area(S) |
| P3 | missed walls(M) ≤ missed walls(S) + 2 percentage points |
| P4 | median wall offset(M) ≤ median wall offset(S) |

L is compared with S the same way and reported as a secondary result. Reported as measured.

## 5. Code

`roomrecon/collision/`: `glb.py` (read POSITION + indices from a `.glb`), `grid.py` (sample
triangles, slab, occupancy grid, flood fill, the four metrics), `__main__.py` (CLI: given the
four `.collision.glb` files, the GT points, L points, scene.json → `metrics.json` + a top-down
figure of the four sources over GT). `scripts/collision_study.ps1` (PC: writes the M/L/GT splat
PLYs, runs the two splat-transform steps per source, then the CLI). A GPU-machine snippet in the
plan samples the mesh and the laser scan. Tests (PC): GLB round trip; synthetic occupancy grids
with known phantom / missed / offset / IoU values.

## 6. Not doing

Other rooms (no laser scans); changing splat-transform's parameters; using the result to change
the `vr_room` demo (a new collision for room 2 is optional afterwards, in new files).
