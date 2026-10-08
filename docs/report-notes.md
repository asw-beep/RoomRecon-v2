# Report notes: decisions and results (running log)

Everything the final report must mention, in one place, with pointers to the evidence. Added to
as the work goes on; newest decisions at the end of each table. Numbers are copied from
`results/`, which stays the source of truth (each number there states its machine).

## 1. What we built

A pipeline from a phone capture of a room to a photoreal 3D model you can walk through with
collision: iPhone capture (ARKit poses + LiDAR depth) → DN-Splatter (nerfstudio) Gaussian splat
and mesh on the GPU machine → automatic alignment (`scene.json`: up, floor, scale, spawn) →
collision mesh (`splat-transform`) → first-person walk in the SuperSplat Viewer on a laptop.
Shown on one web room (MuSHRoom `vr_room`) and one own room (`sejal-cap2`).

## 2. Decisions

| Date | Decision | Why | Alternatives / cost |
|---|---|---|---|
| 2026-10-03 | Restart as v2: use the phone's ARKit poses (+ LiDAR depth) instead of estimating poses from images | v1 lesson: image-only poses failed on a plain-walled bedroom (COLMAP registered 24% of frames; learned features 49 cm pose error, total collapse) | Pose estimation from images alone moved to future work |
| 2026-10-03 | Reuse existing tools: nerfstudio 1.1.3 + DN-Splatter, SuperSplat Viewer walk mode, `splat-transform`; write only glue | Project rule: never rebuild what a maintained tool does | Unity + UnityGaussianSplatting kept as viewer fallback (not needed) |
| 2026-10-03 | Data: MuSHRoom `vr_room` first (phone poses + LiDAR + laser-scanned reference mesh), then own captures | Same shape of data as our phones, and a reference mesh to measure against | ScanNet++ not used; ARKitScenes not reached |
| 2026-10-03 | Safe training settings (spec §5.1): checkpoint and evaluate every 1k steps, keep all, no pose recentring or rescaling for own data, images cached on the CPU | v1 lessons: final-only metrics hid collapse; a lost run took its report with it | — |
| 2026-10-04 | "Lite" training (`train_room_lite.sh`): 300k seed points instead of 1M, stop splitting at 10k, auto-resume after a crash | The shared A100 left only ~7 GB free; the default ran out of memory | A deviation from DN-Splatter's defaults; its quality cost is not measured. `train_room.sh` keeps the defaults for a freer GPU |
| 2026-10-06 | DN-Splatter (depth + normal priors) over plain splats for walkable rooms | G3: 3.3× more accurate geometry for ~0.5 dB less image quality; collision is built from geometry | — |
| 2026-10-05/08 | Backup demos as self-contained bundles in private GitHub releases (`demo-g2-vr_room`, `demo-g4-sejal-cap2`) | A demo that works offline if something fails on the day | The `vr_room` demo is kept unchanged on purpose (see §4) |
| 2026-10-07 | Own room: Sejal's iPhone 13 Pro, Polycam raw export; capture 2 of 3 used | cap1: 32 frames, too sparse; cap3: three tracking segments with ~2 m pose jumps (even in Polycam's corrected poses); cap2: 257 frames, one segment, full coverage | — |
| 2026-10-07 | Kaggle T4 as a fallback GPU when the A100 is full | The co-tenant held almost all 40 GB that night | Two Kaggle sessions were lost (12 h limit, no output); training went back to the A100 |
| 2026-10-08 | Novelty (G8): an after-capture check, run on the laptop before any training, that shows which regions of the room need more coverage and whether the phone's tracking broke (combining "capture doctor" and "trust map") | Grounded in our own failures (cap1 too sparse, cap3 broken tracking that Polycam accepted); needs no GPU; visible in the demo; measurable on our rooms. For the course evaluation | A live on-phone version needs our own iOS app: future work. Output: a walkable 3D coverage map in the same viewer + a top-down heatmap + a text report |

## 3. Results

| Goal | Result | Machine | Evidence |
|---|---|---|---|
| G1 `vr_room` trains | 30k steps in 87 min, one attempt, peak ~5.2 GB; within-PSNR 26.3 dB, no collapse | GPU machine (A100, shared) | `results/g1-vr_room.md` |
| G2 `vr_room` walkable | 908k Gaussians; spawn inside at eye height, walls stop you, collision matches; ~55 FPS | PC (RTX 3050 Laptop, browser) | `results/g2-export.md`, `g2-walk-test.md` |
| G3 quality | DN-Splatter vs plain splats: PSNR 26.32 vs 26.84 (within), 23.50 vs 23.24 (other capture); mesh accuracy 1.76 vs 5.85 cm against the laser scan; F-score @5 cm 0.926 vs 0.638 | GPU machine | `results/g3-vr_room-results.md` |
| G4 own room | `sejal-cap2`: 99 min, one crash resumed automatically; 22.18 dB / SSIM 0.853 / LPIPS 0.170 on 33 held-out frames; depth RMSE 6.1 cm; 746k Gaussians; walk-tested as good | GPU machine; walk test on the PC | `results/g4-sejal-cap2.md` |

Qualitative: furniture, curtains and doors render sharply; plain white walls, ceiling and floor
are smeary (`results/g4-renders/`).

## 4. Problems found and fixed (for "challenges" / "lessons")

| Problem | Fix | Commit / evidence |
|---|---|---|
| Out of memory on the shared A100 with DN-Splatter's 1M seed points | Lite training (300k seeds, auto-resume) | `results/g1-vr_room.md` |
| Viewer could not move (missing `postEffectSettings`) | Write the viewer's defaults into `settings.json` | `results/g2-walk-test.md` |
| UI too small to use | Local viewer patch `?uiscale=` | `viewer/` |
| Kaggle sessions never ended (12 h limit, nothing saved). Most likely cause: a leftover memory sampler held the notebook's output open (not confirmed) | Sampler now stopped properly | `d4b1614` |
| Training crashed once at step ~10.4k (`CUDA illegal memory access`, card nearly full) | Auto-resume from the 10k checkpoint; lost ~400 steps | `results/g4-sejal-cap2.md` |
| Own capture: Polycam saved frames sideways, so "up" was not found | Up from camera −X when +Y disagrees | `e3fe046` |
| Own capture: bed tops taken as the floor, room scaled 1.68× | Floor = lowest dense level; scale now 1.04 as expected for metric poses. Checked on `vr_room` against the laser scan: old rule 13 cm too high, new rule within ~1.5 cm. The released `vr_room` demo still uses the old rule (kept unchanged on purpose: eye height ~13 cm high) | `c955a69`, `results/g4-sejal-cap2.md` |

## 5. Novelty: prior work and what is ours

Searched 2026-10-08 (web + arXiv). The idea "show where more coverage is needed" exists in parts:

| Prior work | What it does | Difference from ours |
|---|---|---|
| Zhang et al., *An Object-Centered Data Acquisition Method for 3D Gaussian Splatting using Mobile Phones*, arXiv 2604.19216 (2026) | Live on-phone coverage of viewing directions on a sphere around one object, plus a stability gate; more coverage gives higher PSNR | Objects only; orientation only (no positions or geometry); live |
| Chen, Dai, Adang, Gao, Schwager (Stanford), *Coverage Optimization for Camera View Selection* (COVER), arXiv 2604.05259 (2026) | Shows a Fisher-information view score reduces to angular coverage per primitive; picks next best views; visualisable as an image of under-covered parts | Needs a trained splat; for robots / view selection, not a pre-training check for a person |
| Kopanas & Drettakis, *Improving NeRF Quality by Progressive Camera Placement for Unrestricted Navigation in Complex Environments*, arXiv 2309.00014 (VMV 2023) | Proposes next camera placements in rooms so rays cover space and angle uniformly | Planning with a camera budget, not checking a real phone capture |
| Pix4D quality report | Post-capture overlap map (images per pixel; red/yellow low, green ≥ 5) | Aerial photogrammetry; 2D |
| Polycam, Scaniverse (live meshing), Luma (capture rings for objects) | Live feedback on geometry holes / object paths | Geometry holes only; Polycam accepted cap3's broken tracking |
| Next-best-view methods for 3DGS (e.g. AGILE-GS, GO-PRE, SA-ResGS, 2026) | Uncertainty-driven view selection | Robots, after training |

No tool was found that checks an exported phone room capture for tracking breaks before
training (ARKit pose jumps are a known issue; Polycam re-optimises poses but exported cap3 with
2 m jumps).

**Ours:** the after-capture, pre-training check for room-scale phone captures, using only what
the phone recorded: coverage from ARKit poses + LiDAR depth (COVER's angular-coverage idea moved
before training, on LiDAR geometry instead of trained Gaussians; we reuse and cite it), tracking
health from the phone's per-frame tracking data, validated against the trained splat's actual
errors on held-out frames, and shown as a walkable 3D map. Modest but honest; the report must
cite the works above and say what is reused.

## 6. Still to do for the report

- Walk recordings (G2, G4) and room-2 FPS.
- Report format (open).
- G7: Polycam's own reconstruction of the same room, side by side.
- G8: design, build and measure the novelty (this log gets its decisions and results).
- Limitations to state: lite settings' quality cost unmeasured; smeary plain walls; `vr_room`
  demo floor 13 cm high; one own room only; Kaggle fallback unproven end to end.
