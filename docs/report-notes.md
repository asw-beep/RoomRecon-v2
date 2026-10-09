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
| 2026-10-08 | Sharpened G8 claim: the check says *why* a region will be poor (under-covered / textureless / broken tracking) and so whether more capture helps | Prior work only says "capture more here"; cap2's worst areas are plain walls that more views may not fix | Spec `docs/spec/2026-10-08-g8-capture-check-design.md`; pass/fail criteria fixed before any validation run |
| 2026-10-08 | Coverage measure reused from COVER (angular coverage), computed before training on the phone's LiDAR surface; under-covered = < 5 views (Pix4D's rule) or < 3 of 64 direction bins, fixed in advance; only the texture threshold is tuned, on `vr_room`, and tested unchanged on `sejal-cap2` | Avoids inventing a measure and avoids tuning to the result | — |
| 2026-10-09 | Heavy runs on the GPU machine only; the laptop runs tests and views results (user's rule) | — | Check runs on the GPU machine's CPUs in the existing env; nothing installed |
| 2026-10-09 | Polycam's `blur_score` is reported but not used for labels | Looked at cap2's extreme frames: the lowest scores (51-57) are sharp photos of plain walls, the highest (245-270) the patterned mattress, so the score mixes blur with plain surfaces | — |
| 2026-10-09 | G8 capture check put on hold; search for a new novelty with a likely positive result; fall back to the capture check if that fails | Its pre-agreed test failed: coverage did not predict held-out error (AUC 0.49-0.54), textureless areas had *lower* error; the tracking check worked (cap3 caught, others passed). Results kept below | Capture check stays on branch `g8-capture-check`, outputs in `E:\roomrecon-scenes\capcheck\` |
| 2026-10-09 | New G8 attempt: detect and repair broken phone tracking before training (cap3), measured against training on the raw export and on the main sweep only; pass/fail C1-C4 fixed in the spec before any run | The capture check's tracking part worked; repairing what it finds should give a clear, demoable gain. Prior work: no tool repairs an exported phone capture's tracking before splat training (Polycam Scenes: manual mesh placement; Scaniverse: no merge) | Spec `docs/spec/2026-10-09-g8b-tracking-repair-design.md`. Alternatives considered: collision accuracy vs the laser scan, exposure metadata in training |
| 2026-10-09 | Training for the new G8 on the A100 when IT restores GPU access, Kaggle T4 as backup | The container lost GPU access (NVML "Unknown Error", torch sees no CUDA) after the machine's outage | All three variants on the same machine and steps |
| 2026-10-09 | Tracking repair dropped (cap3 needed no repair); third G8 attempt: accuracy of walk collision built from the splat vs from DN-Splatter's mesh or the phone's LiDAR, against `vr_room`'s laser scan; criteria P1-P4 fixed in the spec before any run | No published accuracy for splat-derived walk collision; needs no GPU (the A100 is down); likely positive (DN-Splatter's mesh was 1.8 cm from the scan in G3) | Spec `docs/spec/2026-10-09-g8c-collision-accuracy-design.md`. Alternatives: back to the capture check (fallback), simulated tracking breaks |
| 2026-10-09 | Report: Elsevier `elsarticle` template (12 pt, double-spaced, line numbers) as given; template sections + one "what we explored and learned" section; ≥ 15 related works from 2024 onward plus credited foundations; collision study as the lead contribution, capture check and tracking repair as supporting/negative results; v1 in the journey; Polycam comparison added; highlights page if it comes naturally; ≤ 30 pages, not stretched; authors Aswin M (corresponding, 2023BCS0012) and Sejal Satheesh (2023BCS0048), IIIT Kottayam, as in the v1 paper | User's answers 2026-10-09 | — |
| 2026-10-09 | Report draft in `report/` (branch `report`): neutral title; no AI acknowledgement; 45 references, each checked against arXiv metadata or the venue page (`report/refcheck/`), venues not confirmed cited as arXiv; figures made from `results/` by `report/make_figures.py`; diagrams in TikZ; two future directions only (template requirement) | User's instructions; "never guess" | First full draft 46 pages, tightened to 38 (body ~30 + double-spaced references ~8); whether the 30-page limit includes references is open (asked) |

## 3. Results

| Goal | Result | Machine | Evidence |
|---|---|---|---|
| G1 `vr_room` trains | 30k steps in 87 min, one attempt, peak ~5.2 GB; within-PSNR 26.3 dB, no collapse | GPU machine (A100, shared) | `results/g1-vr_room.md` |
| G2 `vr_room` walkable | 908k Gaussians; spawn inside at eye height, walls stop you, collision matches; ~55 FPS | PC (RTX 3050 Laptop, browser) | `results/g2-export.md`, `g2-walk-test.md` |
| G3 quality | DN-Splatter vs plain splats: PSNR 26.32 vs 26.84 (within), 23.50 vs 23.24 (other capture); mesh accuracy 1.76 vs 5.85 cm against the laser scan; F-score @5 cm 0.926 vs 0.638 | GPU machine | `results/g3-vr_room-results.md` |
| G4 own room | `sejal-cap2`: 99 min, one crash resumed automatically; 22.18 dB / SSIM 0.853 / LPIPS 0.170 on 33 held-out frames; depth RMSE 6.1 cm; 746k Gaussians; walk-tested as good | GPU machine; walk test on the PC | `results/g4-sejal-cap2.md` |
| G8 capture check (on hold) | Verdicts as predicted: cap3 BROKEN TRACKING, cap1/cap2/`vr_room` RECAPTURE (not broken). Pre-agreed test on held-out frames (vr_room calibrates, cap2 tests; tau_tex 0.0046): mean patch error OK / under-covered / textureless = 0.0326 / 0.0358 / 0.0159 (vr_room), 0.056 / 0.063 / 0.034 (cap2), intervals overlap → criterion 1 FAIL; texture adds 0.046 AUC over coverage (< 0.05) → criterion 2 FAIL; AUC coverage 0.53-0.54. Exploratory (not pre-agreed) 1-SSIM: texture predicts bad patches with AUC 0.73-0.82 (more detail → more error), coverage 0.49-0.50. Run time 18-52 s per room on the PC; 14-33 min on the memory-starved GPU machine | PC (validation, check runs); GPU machine (first check runs) | ledger; `E:\roomrecon-scenes\capcheck\validation\` |
| G8 tracking repair on cap3 (premise failed) | Repair tool works (registration accepted the first sweep: fit 0.87, turn -0.9 deg, tilt 0.6 deg), but cap3's two sweeps were already aligned: first-sweep depth within 2.4 cm of the main sweep before repair with Polycam's corrected poses, and 2.7 cm with ARKit's original poses (main sweep vs itself 1.9 cm). ARKit relocalised; only the 7 transition frames (118-124) carry the 2 m jumps. C1 "metres before" premise false; C3 added surface 4.7 % (< 5 %) | GPU machine (CPU) | `results/g8b-cap3-repair.json`, ledger |
| G8 collision accuracy (`vr_room`, vs laser scan) | Walkable IoU / phantom / missed walls / offset p95: splat (default) 0.577 / 0.00 m² / 17.8 % / 20 cm; DN-Splatter mesh 0.809 / 0.04 m² / 1.7 % / 10 cm; phone LiDAR 0.059 / 4.27 m² / 47 % / 41 cm. Mesh vs splat: P1, P3, P4 pass, P2 fails by 0.04 m² → the pre-agreed all-four rule is not met. The default splat collision leaks out of the room (walkable 11.3 vs 6.5 m²) | PC + GPU machine (CPU) | `results/g8c-collision-accuracy.md` |
| Collision walk test (user, 2026-10-09) | Walking S and M freely: "personally I didn't feel anything different" — the measured gap is not felt in a free walk | PC | `results/walk-recordings.md` |
| G7-lite: Polycam (existing app) vs ours, `vr_room` | Polycam's own mesh (in MuSHRoom), ICP-aligned to the laser scan (best case, median 2.0 cm): as walk collision IoU 0.794 / phantom 0.22 m² / missed 1.8 % vs our DN-Splatter mesh 0.809 / 0.04 / 1.7 %; DN-Splatter's mesh-accuracy protocol on Polycam's mesh running (GPU machine CPU) | PC + GPU machine (CPU) | `results/g8c-collision-accuracy.md`, `g8c-polycam-align.json` |

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
| Capture check far too slow on the GPU machine at first (cap2: 1178 s for 257 frames) | Profiled: numpy 1.26's `unique(axis=0)`/`add.at` (4M points: 240 s vs 9 s on the PC's numpy 2.4), 96 BLAS threads on tiny 3×3 problems (normals 16 s → 0.2 s with one thread), a pairwise neighbour graph for regions; replaced with one-key voxel merge, single-thread BLAS, grid-cell regions | `d1b7e55` |
| Checking our own conventions on real data before trusting the check | Processed poses equal Polycam's keyframe poses (0.000 mm, all three captures); depth from one frame lands within 0.9-1.8 cm (median) of the next frames' own LiDAR depth (cap1-3, `vr_room`) | ledger, Task 7 |

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
| Yasunaga, Saito, Mori (Keio / Stuttgart), *User-in-the-Loop View Sampling with Error Peaking Visualization*, arXiv 2506.21009 (2025) | Live on an iPhone 15 Pro (AR, LiDAR depth): highlights in red where a quick single-view reconstruction disagrees with the live camera, so the user captures there; better 3DGS than uniform/random sampling with the same number of views | Live, small scenes (0.15 m capture area); predicted errors from a quick proxy model, not coverage; no tracking-health check |
| Chang, Choi, Yeum, *3D Reconstruction by Looking: Instantaneous Blind Spot Detector for Indoor SLAM through Mixed Reality* (LiMRSF), arXiv 2411.12514 (2024) | Indoor LiDAR SLAM: shows the point cloud as a hologram over the real room in an MR headset, so the user sees blind spots, drift and double walls while scanning | Live, needs an MR headset + LiDAR rig; human inspection, no coverage measure, not for splats |
| NeRF Director (arXiv 2406.08839, 2024) | Revisits view selection for NeRF training with coverage-based sampling | Choosing among already-captured views |
| IndoorGS (CVPR 2025), PlanarGS (2025), DN-Splatter (2024) | Textureless indoor surfaces (white walls) are a known failure of photometric 3DGS; fixed with geometric / planar / depth priors | Training-side fixes, not capture feedback |

Second search (2026-10-08, deeper): **the idea of capture-coverage feedback is established**,
including for indoor scanning (LiMRSF) and for 3DGS capture (Zhang et al., Yasunaga et al.). No
pre-training check for exported phone room captures was found; practitioners inspect data by
hand or rely on COLMAP failing.

No tool was found that checks an exported phone room capture for tracking breaks before
training (ARKit pose jumps are a known issue; Polycam re-optimises poses but exported cap3 with
2 m jumps).

**Ours:** the after-capture, pre-training check for room-scale phone captures, using only what
the phone recorded: coverage from ARKit poses + LiDAR depth (COVER's angular-coverage idea moved
before training, on LiDAR geometry instead of trained Gaussians; we reuse and cite it), tracking
health from the phone's per-frame tracking data, validated against the trained splat's actual
errors on held-out frames, and shown as a walkable 3D map. Modest but honest; the report must
cite the works above and say what is reused.

### Prior work for the collision-accuracy result (searched 2026-10-09)

| Prior work | What it does | Difference from ours |
|---|---|---|
| Chen, Wang et al., *The Neverwhere Visual Parkour Benchmark Suite*, arXiv 2609.16443 (2026); docs at neverwhere.readthedocs.io | 60+ 3DGS reconstructions of indoor/outdoor scenes as closed-loop legged-robot benchmarks (MuJoCo); docs cover Polycam scanning and aligning collision geometry to the splats | Pairs splats with separate collision geometry for robot simulation; no measurement found of collision accuracy against a laser scan |
| LucidSim (*Learning Visual Parkour from Generated Images*, arXiv 2411.00083) | Collision mesh from a Polycam scan as the terrain, the splat render only as the robot's vision | Same "mesh for collision, splat for looks" practice; not measured against ground truth |
| PlayCanvas splat-transform collision docs; SuperSplat walk mode | Collision built from the splat's Gaussians (voxels, exterior fill, carve) | The default we measure; no published accuracy numbers |
| Splat-Nav (arXiv 2403.02751), NavGSim (arXiv 2603.15186) | Robot navigation / simulation in Gaussian-splat maps | Robot planning maps, not walk collision vs a laser scan |

**Ours (claim narrowly):** a controlled comparison, inside the SuperSplat walk pipeline, of collision built from the splat vs from DN-Splatter's mesh vs from raw phone LiDAR, against a laser scan of the same room. Not new: building collision from a splat, or using a mesh for collision with splat visuals (robotics already does). Also from ChatGPT's critique (2026-10-09): state the confound that M/L/GT were turned into 2 cm splats while S uses the real Gaussians; one room only; photometric error measures appearance, not geometry (relevant to the capture check's texture result).

## 6. Still to do for the report

- ~~Walk recordings (G2, G4) and room-2 FPS~~ done 2026-10-09: `results/walk-recordings.md` (both rooms: frame 10.0 ms / 100 FPS at every sample, GPU 2.5-8.3 ms, PC). Correction 2026-10-09: room 2's first reading of 18.0 ms was a misread 10.0 (dotted zero); G2's 18.0 ms for `vr_room` may be the same misread and cannot be re-checked.
- Report format (open).
- G7: Polycam's own reconstruction of the same room, side by side.
- G8: design, build and measure the novelty (this log gets its decisions and results).
- Limitations to state: lite settings' quality cost unmeasured; smeary plain walls; `vr_room`
  demo floor 13 cm high; one own room only; Kaggle fallback unproven end to end.
