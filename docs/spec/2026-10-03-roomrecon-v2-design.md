# RoomRecon v2 — Design

- **Date:** 2026-10-03 · **Revision:** 2 (after critique review) · **Status:** draft for user review
- **Authors:** Aswin M, Sejal Satheesh (course project, IIIT Kottayam, Sem 7 Computer Vision)
- **Deadline:** final demo + report on **2026-10-11**. Work is planned as goals in priority
  order (§9), not dated milestones.
- **Replaces:** the v1 plan (RGB-only, 6 GB laptop, capture-gate paper), now history in
  `docs/` and `docs/archive/`.

## 1. Goal

Capture an indoor room with an iPhone, and get back a **clean 3D model you can walk around
and explore freely**: a photoreal Gaussian splat for looks, plus a collision mesh so you
can't walk through walls.

**Priority 1:** a clean model, and being able to interact with it. Everything else comes
after.

**Success (by the deadline):** at least one MuSHRoom room **and** at least one room we
captured ourselves, opened in a first-person walk mode. You spawn inside, move with
WASD + mouse, can't pass through walls or fall through the floor, and it looks like the
real room from any angle without obvious floaters or holes. Quality is reported with
measured numbers and the machine they came from.

## 2. Principles

1. **Reuse existing software first.** Install, configure and extend maintained tools. Write
   code only for glue, format converters, and our own added features.
2. **Phone tracking is an input, not a problem to solve.** ARKit poses (+ LiDAR depth when
   present) come from the capture app.
3. **No made-up numbers.** Targets are set from spike measurements; every number states the
   machine it ran on.
4. **One step at a time**, confirmed with the user before each.
5. **Goals in priority order:** each goal is finished (exit criteria met) before the next
   starts. Whatever is unfinished at the deadline is cut from the end of the list.

## 3. Scope

**In scope:** iPhone capture-app exports (frames + ARKit poses [+ LiDAR depth]);
DN-Splatter training; splat + collision mesh; first-person walk in SuperSplat; MuSHRoom +
a few ARKitScenes rooms; own captures; a final comparison with Polycam/Scaniverse.

**Optional (goals G4+, only after G1–G3):** one small novelty (chosen from results, G8);
Unity desktop viewer; more own rooms.

**Out of scope:** pose estimation from images alone (future work / extension); ScanNet++;
multi-room, dynamic scenes, object editing, semantics, centimetre-accurate measurement.

## 4. Pipeline

```
1. Capture   iPhone Pro: Polycam (raw data) or NeRFCapture with LiDAR depth
             iPhone 16:  NeRFCapture streaming mode (to be tested; see §7)
             (web data first: MuSHRoom, ARKitScenes — same shape of data)
2. Import    nerfstudio / DN-Splatter importers (MuSHRoom parser, Polycam converter);
             our converter for NeRFCapture and ARKitScenes
3. Check     sanity report: poses present + finite, pose spread and view-angle gaps,
             frame count, blur, depth present? → fails fast with a reason
4. Train     DN-Splatter on the GPU machine, in tmux, with OUR settings (§5.1)
5. Export    splat .ply  +  mesh (DN-Splatter `gs-mesh o3dtsdf`)  +  scene.json
6. Collide   collision mesh from the splat (SuperSplat's splat-transform), compared
             with the DN-Splatter mesh; floater cleanup in SuperSplat
7. Explore   SuperSplat walk mode on the PC (first-person, collision)
8. Compare   same rooms in Polycam / Scaniverse (Approach 3), at the end (§8)
```

**Alternatives considered.** Unity + aras-p plugin was the original viewer. SuperSplat
(PlayCanvas, MIT) shipped a first-person walk mode with automatic collision in March 2026,
so building it in Unity would duplicate existing software. Unity remains the fallback if
SuperSplat falls short in the spike. On-phone apps alone (Approach 3) would be a clone and
are kept as the comparison reference. Our v1 gsplat scripts are the **out-of-stack
fallback** if DN-Splatter/nerfstudio cannot be made to work.

### Where each piece runs

| Machine | Role | Software |
|---|---|---|
| GPU machine (campus A100 40 GB, shared; `ssh gpu`) | All data, training, export, collision mesh | DN-Splatter + nerfstudio 1.1.3 in a pixi/conda env; splat-transform (Node) |
| PC (Windows, RTX 3050 6 GB) | Editing code, viewing, the demo | VS Code (+ Remote-SSH), Git, SuperSplat, Blender. **No WSL use** |
| Kaggle (T4) | **Not a fallback yet.** `push.sh` is built for gsplat scripts; it would need porting | — |

## 5. Components

| # | Component | Job | Built from | Our code |
|---|---|---|---|---|
| C1 | Importer | Export/dataset → nerfstudio dataset | DN-Splatter MuSHRoom parser, `convert-poly-to-ns` | NeRFCapture + ARKitScenes converters (fix: NeRFCapture `file_path` has no extension; per-frame intrinsics vary) |
| C2 | Sanity report | Reject unusable input before GPU time | — | Script → `sanity.json` |
| C3 | Trainer | Train with safe settings, record the run | DN-Splatter + nerfstudio | Wrapper fixing the settings in §5.1; collapse check; `run.json` |
| C4 | Exporter | Splat + mesh + `scene.json` | `ns-export gaussian-splat`, `gs-mesh o3dtsdf` | `scene.json` writer; mesh → GLB/OBJ, decimated (pymeshlab) |
| C5 | Collision + viewer | Walkable scene | SuperSplat walk mode, `splat-transform --collision-mesh` | Spawn point + floor check (§6) |
| C6 | Evaluation | Metrics + fly-through per run | nerfstudio eval, `ns-render`, DN-Splatter `eval_mesh_mushroom_vis_cull.py` | Collects results into `run.json` |

### 5.1 Settings our wrapper always sets (defaults are unsafe)

| Setting | Why |
|---|---|
| Save a checkpoint every 1k steps, **keep all** (`--steps-per-save 1000 --save-only-latest-checkpoint False`) | DN-Splatter's config effectively saves only at the end; a co-tenant OOM would lose the whole run. Also lets us keep the best step (v1 collapse) |
| Evaluate every 1k steps (all or a fixed subset of test images) | DN-Splatter's config effectively disables periodic eval; v1 showed final-only metrics hide collapse |
| `--orientation-method none --center-method none --auto-scale-poses False` for own + ARKitScenes data | nerfstudio's defaults recentre and rescale; exports then are not metric and splat/mesh can misalign. MuSHRoom's parser already does this, so the spike alone would not reveal it |
| `cache-images cpu` (+ uint8) | nerfstudio puts all training images on the GPU as float32; likely more than the ~5 GB free |
| Mesh TSDF voxel size set for a room, not an object | DN-Splatter's README defaults are object-scale |
| Resume test: kill a run, continue with `--load-dir` | Proves the shared GPU can't cost us a run |

Disk: ~30 checkpoints per run; check the container's disk quota before the first long run.

### 5.2 Data on disk (GPU machine, `~/roomrecon/`)

```
datasets/<source>/<room>/     raw downloads / phone exports (re-downloadable)
scenes/<room>/<run_id>/
  data/  sanity.json  train/  export/{splat.ply, mesh.glb, collision.glb, scene.json}
  eval/{metrics.json, flythrough.mp4}  run.json
```

`scene.json`: units (metres, approximate, from ARKit), up axis and the axis conversion for
the viewer, floor height, spawn position/direction, splat count, source, run id. A pytest
checks the axis conversion.

### 5.3 Environment

DN-Splatter pins nerfstudio 1.1.3 and gsplat 1.0.0 and breaks with newer gsplat; its pixi
setup uses Python 3.10, CUDA 11.8 and torch 2.2. Install with **pixi or conda**, which bring
their own Python and CUDA without sudo; the system nvcc 12.1 is not used. Plain nerfstudio
`splatfacto` runs **in the same environment** so the comparison is fair. The environment
is frozen with a lockfile. Both nerfstudio (last release Nov 2024) and DN-Splatter (last
commit Jul 2025) are no longer actively developed: pin everything.

## 6. Walking: collisions and spawn

- **Collision mesh:** generated by `splat-transform --collision-mesh` (floater filtering,
  walkable carving), with the DN-Splatter TSDF mesh as the alternative. The spike compares
  both.
- **Spawn:** at a real camera position from the capture trajectory (free space by
  construction), dropped to floor height + eye height.
- **Floor:** fitted plane (RANSAC) with a safety floor under it, so holes in the mesh can't
  drop you out.
- **Outer wall:** a bounding volume around the room, so mesh holes at windows or doors
  can't let you walk out of the room.
- Known limits: glass, mirrors and open doorways leave holes; floaters can become invisible
  walls.

## 7. Data and capture

1. **MuSHRoom** (primary web data; CC BY 4.0): iPhone poses + LiDAR depth, reference
   meshes, a separate test walk per room. `koivu` and `vr_room` are already downloaded.
   Caveats from v1's log: the separate test walk lines up only with the dataset's
   **COLMAP** poses (the iPhone/Polycam poses of the two walks are in different frames),
   and koivu's "COLMAP" pose file is a copy of the Polycam poses, so check koivu before use.
   Report which poses each metric used. Several rooms (kokko, sauna, olohuone) use auto
   exposure, so expect lower PSNR there.
2. **ARKitScenes** (second): 1–2 rooms, including the v1 plain-walled bedroom (41254269).
   Record its licence terms.
3. **Own captures:**
   - **iPhone Pro (LiDAR):** Polycam raw data or NeRFCapture with depth. This is the main
     own-capture device.
   - **iPhone 16 (no LiDAR):** Polycam raw export and Record3D need LiDAR, and NeRFCapture's
     offline mode is one tap per photo (v1: 32 photos → PSNR 16). The spike tests
     NeRFCapture **streaming** mode. If it fails, the no-LiDAR case = Pro capture with
     depth dropped.
   - Guidance (v1): several frames per second, a few hundred frames, move slowly.

## 8. Testing and what "clean" means

| Check | How | Source |
|---|---|---|
| Splat quality, new viewpoints | PSNR / SSIM / LPIPS on held-out views. MuSHRoom: the separate test walk (COLMAP poses); phone-pose runs: within-walk test split | nerfstudio / DN-Splatter eval |
| No training collapse | Eval every 1k steps; flag a run whose final score is well below its peak (threshold set from the spike); keep the best checkpoint | C3 |
| Mesh accuracy | Accuracy, completeness, F-score vs MuSHRoom reference mesh | DN-Splatter mesh eval |
| Looks clean | Same fixed fly-through video every run + checklist: no floaters in walkable space, flat walls, no floor holes | `ns-render` |
| Walkable | Spawn inside; walls block; floor holds; % floor area with holes; collider triangle count; a set of blocked/open path tests; FPS on the RTX 3050 | viewer + our checks |

**Comparisons:** (1) `splatfacto` vs DN-Splatter (same env); (2) with vs without depth,
on the same capture; (3) ours vs Polycam/Scaniverse on own rooms: align models (ICP),
held-out photos from the capture, the walk test, and a blind side-by-side. MuSHRoom can't
go into those apps, so this comparison uses own captures only.

**Our code:** pytest for pose conversions (NeRFCapture `transforms.json` camera→world
OpenGL; ARKitScenes `.traj` world→camera OpenCV), the `scene.json` axis conversion, and the
sanity report rejecting broken input.

## 9. Goals (priority order)

Each goal has exit criteria; it is done only when they are met with evidence (files,
numbers, a walk-through recording). Goals are worked strictly in order. **G1–G3 are the
minimum demo. G4 onward are added only once G1–G3 are done.**

| Goal | What | Done when |
|---|---|---|
| **G0 Setup** | New repo `RoomRecon` (GitHub `asw-beep/RoomRecon-v2`, private) with this spec + `CLAUDE.md`; DN-Splatter env (pixi/conda) on the GPU machine; MuSHRoom `vr_room` copied there | `ns-train` and DN-Splatter import cleanly; env lockfile committed; `vr_room` on the GPU machine |
| **G1 First room trains** | `vr_room` through DN-Splatter with the §5.1 settings | Training finishes; checkpoints every 1k; a killed run resumes; eval curve recorded; free VRAM, peak VRAM and time recorded |
| **G2 First room walkable** | Export splat + mesh + `scene.json`; collision mesh; open in SuperSplat walk mode | You spawn inside `vr_room`, walls block, floor holds, splat and collision line up; fly-through video + screen recording of the walk saved |
| **G3 Quality measured** | Metrics on `vr_room`: held-out PSNR/SSIM/LPIPS, mesh vs reference, walkability checks; `splatfacto` vs DN-Splatter | Results table with machine and settings; targets for later rooms set from it |
| **G4 Own room walkable** | One own room captured on the Pro (LiDAR); through the same pipeline | Own room passes the G2 walk test; metrics on held-out frames |
| **G5 More web rooms** | 2–4 more MuSHRoom rooms + 1 ARKitScenes room | Results table across rooms |
| **G6 iPhone 16 / no-LiDAR** | NeRFCapture streaming test; else Pro capture with depth dropped | With vs without depth compared on the same room |
| **G7 Comparison** | Same own room in Polycam / Scaniverse | Side-by-side per §8 |
| **G8 Novelty** | One small addition chosen from G1–G7 results | Implemented and measured |
| **G9 Old-data cleanup** | Off-machine copy, then remove `~/roomrecon` from WSL and compact | Space freed; copy verified |
| Extension | Pose estimation from images alone | Future work |

The report and demo are prepared alongside, from each goal's evidence.

**Reality check for 2026-10-11:** G0–G3 is the realistic core. G4 is a stretch that
depends on the Pro phone capture and G1–G3 going smoothly. G5 onward are likely to be
future work in the report.

## 10. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Shared GPU (~35 of 40 GB used by another tenant) | Out of memory; runs killed | Images cached on CPU; checkpoints every 1k + resume; check free memory at start; ask IT |
| GPU machine has no backup / may be rebuilt / campus-only | Lost data or no access | Keep an off-machine copy of old data before deleting from the PC; results to git or PC as produced; jobs run in tmux |
| DN-Splatter env won't build | Pipeline blocked | pixi/conda env; fallback = v1 gsplat scripts (out of the nerfstudio stack) |
| Splat/mesh misalignment, non-metric export | Walls in the wrong place | §5.1 pose settings; `scene.json` + pytest; spike checks alignment |
| Collision mesh holes / floaters | Falling out / invisible walls | Safety floor, bounding wall, spawn at a camera position; walkability checks |
| iPhone 16 can't make a dense capture | No no-LiDAR own data | Streaming test; else depth dropped from Pro captures |
| Full-room training degrades (v1) | Bad scenes | Eval every 1k; best checkpoint kept |
| SuperSplat walk mode insufficient | Viewer gap | Unity + aras-p as fallback |
| Deadline 2026-10-11 | Unfinished | Goals in priority order; G1–G3 is the minimum demo; the rest cut from the end |

## 11. Open items

- Report format.
- IT: how much of the A100 this container may use; whether `/home` survives a rebuild; the
  disk quota; VPN access from off campus.
- New repository (agreed): folder `...\Computer_vision\RoomRecon`, GitHub
  `asw-beep/RoomRecon-v2`, private. This spec and `CLAUDE.md` move there in G0.
- The v1 paper draft (`Paper/`) claims RGB-only, a 6 GB envelope and browser delivery; the
  report must be rewritten around v2.
- Unverified (spike checks): DN-Splatter peak VRAM on a room; whether `gs-mesh` exports in
  the dataparser frame or the world frame; NeRFCapture streaming on iOS 18; SuperSplat walk
  mode and `splat-transform --collision-mesh` behaviour on our exports.
