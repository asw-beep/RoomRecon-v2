# G8 (third attempt) — walk-collision accuracy against the laser scan (`vr_room`)

Machine: PC (collision builds with splat-transform v3.10.0, evaluation) and GPU machine (CPU:
sampling the laser scan and DN-Splatter's mesh with open3d). Date 2026-10-09. Spec:
`docs/spec/2026-10-09-g8c-collision-accuracy-design.md` (criteria fixed before the run).
Evidence: `g8c-collision-metrics.json`, `g8c-collision-topdown.png`.

## Setup

Four geometry sources, each through exactly the demo's collision pipeline (`collision_room.ps1`:
the demo's rotation and scale, then `--seed-pos <spawn> --filter-cluster --voxel-size 0.05
--voxel-external-fill --voxel-carve`):

- **S**: the trained splat `dns-30k-lite/splat.ply` (its collision is byte-identical in size to the
  walk-tested demo's, 274,532 B).
- **M**: 1.45 M points sampled on DN-Splatter's mesh (`gs-mesh o3dtsdf`, 2 cm; 144.7 m²), written as
  2 cm opaque splats.
- **L**: the phone's LiDAR surface from the capture check (981,622 points, 3 cm voxels), as splats.
- **GT**: 1.30 M points on MuSHRoom's laser scan (130.1 m²), moved into the model's frame with
  `inv(icp_iphone.json)`, as splats.

Frame checks before measuring: mesh → scan median 1.6 cm; raw LiDAR → scan 8.2 cm; scan at body
height → the demo's collision median 7.2 cm (p90 23 cm), so all share one frame. Floor = the scan's
(10 cm below the demo's old-rule floor).

Evaluation: top-down 5 cm cells, collision surface between 0.2 and 1.8 m above the floor; walkable =
free cells 4-connected to the demo's spawn; GT = the scan's collision through the same pipeline.

## Results

| Source | Walkable IoU vs GT | Phantom obstacles | Missed walls | Wall offset median / p95 | Walkable area |
|---|---|---|---|---|---|
| S — splat (default) | 0.577 | 0.00 m² | 17.8 % | 0 / 20 cm | 11.3 m² |
| **M — DN-Splatter mesh** | **0.809** | 0.04 m² | **1.7 %** | 0 / **10 cm** | 7.6 m² |
| L — phone LiDAR | 0.059 | 4.27 m² | 47.0 % | 5 / 41 cm | 11.4 m² |
| P — Polycam's own mesh (existing app) | 0.794 | 0.22 m² | 1.8 % | 0 / 10 cm | 6.9 m² |
| GT — laser scan | 1 | 0 | 0 | 0 / 0 | 6.5 m² |

Pre-agreed criteria, M vs S: **P1 PASS** (0.809 ≥ 0.577 + 0.02), **P2 FAIL** (phantom 0.04 m²
> 0.00 m²: 16 cells of 5 cm), **P3 PASS** (1.7 % ≤ 17.8 % + 2), **P4 PASS** (median 0 ≤ 0; a tie
at the 5 cm resolution — the p95 is 10 vs 20 cm). The pre-agreed "all four" rule is therefore
**not met**: M beats S clearly on overlap and missed walls, ties on median offset and adds a
little phantom obstacle area. L vs S: all four FAIL.

What the figure shows: S's collision has gaps (mostly along the left wall and at furniture at body
height), so the walkable area leaks out of the room (11.3 m² vs the scan's 6.5 m²); M follows the
scan closely with one small leak at the same wall; raw LiDAR, with its noise and stray points in
mid-room, makes collision unusable without the trained model's filtering.

## Limitations

One room (the only one with a laser scan); the scan's own collision through the same 5 cm voxel
pipeline is the reference; 5 cm cells make the median offset coarse; the walk test of the demo
(G2) felt right, so the leaks are in places the user did not walk.

## Walk test (user, PC, 2026-10-09)

Walking both versions freely, the user felt no difference between S and M
(`walk-recordings.md`). The measured gap is therefore not a noticeable everyday difference in a free
walk; it is where the collision is wrong at furniture and some wall stretches.

## Added 2026-10-09: Polycam's own mesh as a fifth source (P)

MuSHRoom ships Polycam's reconstruction of the same `vr_room` capture (`polycam_mesh/textured.obj`,
97k vertices) in Polycam's own frame. It was aligned to the laser scan (model frame) by ICP from all
24 axis orientations (best: 75.5 % of points within 5 cm, inlier RMSE 2.0 cm, median 2.0 cm;
`g8c-polycam-align.json`) — a best-case alignment for Polycam, since our sources use the capture's
own poses. Sampled at ~1 cm (149.5 m², 1.50 M points) and built with the same pipeline (splat-transform
v3.10.1 fetched by npx for this one source; v3.10.0 for the others). S, M and L numbers reproduced
exactly in the five-source rerun. Result: Polycam's mesh makes walk collision as good as DN-Splatter's
mesh (IoU 0.794 vs 0.809, missed walls 1.8 % vs 1.7 %) with more phantom obstacles (0.22 vs 0.04 m²);
both are far better than the splat-derived default.
