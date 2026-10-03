# RoomRecon (v2, restarted 2026-10-03)

Phone capture of an indoor room → a clean 3D model you can **walk around and explore freely**.
Course project (IIIT Kottayam, Sem 7 Computer Vision). Team: Aswin M, Sejal Satheesh.

> **History.** v1 (Aug–Sep 2026: RGB-only, 6 GB laptop, capture-gate paper) lives in the old
> repo `..\Project` (GitHub `asw-beep/RoomRecon`), kept as an archive. Its lessons are in
> `docs/lessons-learned-v1.md`. Do not follow v1's ROADMAP/ADRs.

## Current status

**Building G0–G3 — PAUSED 2026-10-04 by the user.** G0 done; G1 resume test passed; the full `vr_room` run (Task 7) ran out of GPU memory because the other tenant now holds ~37 GB of the A100 (~3 GB free). Progress ledger: `.superpowers/sdd/2026-10-03-g0-g3-first-walkable-room/progress.md` (git-ignored, on the PC); evidence in `results/`. Resume at plan Task 7 after checking free memory. Options on the table: ask IT; shrink to fit (half-res, stop-split-at 7000, auto-resume loop); Kaggle T4; laptop. Spec (revision 2, after a critique review):
`docs/spec/2026-10-03-roomrecon-v2-design.md`, the source of truth for the
design. Next steps, in order:
1. Execute `docs/plans/2026-10-03-g0-g3-first-walkable-room.md` task by task.
2. Goals G0–G3 (spec §9): new repo + DN-Splatter env; MuSHRoom `vr_room` trains (resume
   test); walkable in SuperSplat; quality measured. That is the minimum demo.
3. Copy the old data off the PC (keep an off-machine copy first), then migrate.

**DN-Splatter/nerfstudio defaults are unsafe for us.** Always use the settings in spec §5.1:
checkpoints every 1k, keep all, eval every 1k, no pose recentring/rescaling for own data,
images cached on the CPU.

Work one step at a time with the user, and confirm before each step.

## What we are building (agreed 2026-10-03)

| Decision | Choice |
|---|---|
| Input | Capture-app export from an iPhone: RGB frames + **ARKit poses** (+ **LiDAR depth** when the phone has it). Apps: NeRFCapture (proven in M1), Polycam, Record3D |
| Output | **Both**: a Gaussian splat for photoreal viewing **and** a mesh extracted from it (export + collisions) |
| Interaction | First-person **walk and explore**, no walking through walls |
| Pipeline (Approach 1) | **nerfstudio 1.1.3 + DN-Splatter** (pixi/conda env) on the GPU machine → splat `.ply` + mesh + `scene.json` → collision mesh via `splat-transform` → **SuperSplat walk mode** on the PC |
| Viewer fallback | Unity + aras-p UnityGaussianSplatting, only if SuperSplat falls short in the spike |
| Helper tools | viser (nerfstudio's live viewer) for training checks; SuperSplat for cleanup; Blender for meshes |
| Deadline | **2026-10-11** (final demo + report). Work is **goal-based, not dated**: goals G0–G9 in priority order (spec §9); G0–G3 is the minimum demo; unfinished goals are cut from the end |
| Final comparison | **Approach 3**: Polycam / Scaniverse on-phone reconstruction of the same rooms, as the quality reference |
| Later extension | Pose estimation from images alone (plain video / photos from any app). Not before the main project works |
| Novelty | Modest is fine, but not an exact clone. **Decided from real results (goal G8), only after G1–G3.** Note: walk-with-collision and floater cleanup already exist (SuperSplat, 2026), and LiDAR-vs-no-LiDAR on MuSHRoom repeats DN-Splatter's papers. Strongest remaining candidate: predicting walkability / failure from the phone's tracking data before or early in training |

### Rules for how we build

- **Reuse existing software first.** Install, configure and extend existing tools; write
  code only for what does not exist (glue, converters, our added features). Never rebuild
  something a maintained tool already does.
- **Priority 1 is a clean 3D model and interacting with it.** Everything else waits.
- **No fabricated numbers.** Only report what we measured, with the machine it ran on.
- Phones: Aswin has an **iPhone 16** (ARKit poses, no LiDAR, NeRFCapture installed); the
  teammate has an **iPhone Pro** (poses + LiDAR depth). The Pro is the main own-capture
  device: Polycam raw export and Record3D need LiDAR, and NeRFCapture's offline mode is one
  tap per photo. The iPhone 16 depends on NeRFCapture streaming mode working.

## Data

1. **Web datasets first:**
   - **MuSHRoom** (primary): iPhone ARKit poses + LiDAR depth, reference meshes, CC BY 4.0.
     DN-Splatter has a loader for it. Drop its depth to simulate the no-LiDAR iPhone 16.
     `koivu` and `vr_room` are already downloaded (`../Project/storage/projects/m1-mushroom-inspect/`, old repo, on the PC).
     Start the spike on `vr_room`.
   - **ARKitScenes** (second): iPad Pro LiDAR + ARKit poses, for variety (includes the M1
     plain-walled bedroom, 41254269).
   - **ScanNet++ is not used** (decided 2026-10-03).
2. **Our own captures** with both phones after the pipeline works on web data.
   M1 lesson: several frames per second, a few hundred frames per room, move slowly. A
   32-photo capture gave streaks and floaters.

## Workspace setup — read before touching anything

| | PC (laptop) | GPU machine ("T3") |
|---|---|---|
| Role | Code editing, Unity viewing, looking at results | **All data, tools and heavy compute** |
| Hardware | Win 11, RTX 3050 Laptop 6 GB | NVIDIA **A100-PCIE-40GB** (email said H100; it is not), MIG 7g.40gb, 96 CPUs, 188 GB RAM, 1.2 TB free |
| OS | Windows (native tools only; **WSL is not used for RoomRecon v2**) | Ubuntu 20.04 **Docker container**, Python 3.8 system |
| Tools | `ssh`/`scp` (Windows OpenSSH), Git for Windows, VS Code (+ Remote-SSH), SuperSplat, Blender (Unity only as fallback) | everything Linux/CUDA: nerfstudio, DN-Splatter, splat-transform |
| Access | — | `ssh gpu` (key auth, alias in `~/.ssh/config` → `aswinm@10.10.1.49`). **Campus network only**; off-campus VPN untested. Fallback: web desktop at `http://compute.iiitkottayam.ac.in:8080/#/` |

**GPU machine facts and gotchas**

- **The GPU is shared.** 35 of 40 GB were in use by someone else on 2026-10-03 (~5 GB
  free). Check free memory before every job (`nvidia-smi`; the `--query-gpu` memory
  fields show "Insufficient Permissions" inside MIG, so read the MIG table). Not yet
  answered by IT.
- **No backup, no separate volume.** `/home/aswinm` lives in the container layer and may
  not survive a container rebuild. Anything we can't regenerate goes into git or an
  off-machine copy as soon as it's produced. Datasets are re-downloadable.
- **sudo needs the user's password.** Claude cannot run apt. Ask the user to run
  `ssh -t gpu "sudo ..."` themselves.
- Installed on 2026-10-03: git, tmux, rsync, build-essential, pip3, python3-venv,
  `cuda-nvcc-12-1` + `cuda-cudart-dev-12-1` (`/usr/local/cuda-12.1`, nvcc V12.1.105). The
  Google Chrome apt source had an expired key; it was fixed by re-importing Google's key.
- Internet works from the machine: GitHub, PyPI, download.pytorch.org, HuggingFace.
- Run long jobs inside **tmux** so they survive a dropped SSH session.
- IT: "don't shut the machine down; just close the browser."
- **Never write the HPC password anywhere** (repo, memory, scripts).
- **Never upload any SSH key to GitHub** (no deploy keys, not even public halves). The GPU
  machine has no GitHub access. Code reaches it only by pushing from the PC:
  `git push gpu main` (remote `gpu` = `gpu:roomrecon/repo`, which has
  `receive.denyCurrentBranch updateInstead`, so the working tree updates on push). Never
  `git pull` on the GPU machine. GitHub (`origin`) is the PC's backup remote only.
- Inside a script piped to `ssh gpu bash -s`, give any inner `ssh`/`git` command
  `</dev/null`, or it swallows the rest of the script.

**Running commands on the GPU machine from Claude (PowerShell)**

```powershell
"<bash script>" | ssh -o BatchMode=yes gpu "tr -d '\r' | sed 's/^\xEF\xBB\xBF//' | bash -s"
```
PowerShell adds a BOM and CRLF to piped text, which breaks bash; strip them as above. A
`known_hosts ... Permission denied` warning is Claude's sandbox and harmless. Claude's
shell cannot read `C:\Users\ASWIN\.ssh` directly (sandboxed), and interactive commands
(passwords, `ssh-keygen`) must be run by the user in their own PowerShell window.

**PC disk:** C: has ~19 GB free, E: ~42 GB. The WSL disk (`E:\WSL\Ubuntu-22.04\ext4.vhdx`,
27 GB) holds the old v1 toolchain (`~/roomrecon`, 6.4 GB). It is to be deleted after
migration; the laptop no longer trains. WSL itself stays installed (other coursework, Docker
Desktop's backend), but RoomRecon only uses it once more: to copy the old data to the GPU
machine. It also holds **other coursework** (`chicago`,
`spark`, `hive`, `scala`, labs): never touch those. The vhdx does not shrink on its own:
compact it only with WSL fully shut down (`../Project/scripts/compact-wsl-vhdx.txt`). Force-compacting
a mounted disk once destroyed all Docker data. Docker Desktop disks (~26 GB) are not part
of this project.

## Lessons from v1 (M1, Aug–Sep 2026) that still apply

Full detail: `docs/lessons-learned-v1.md`.

- **Image-only poses fail on plain-walled rooms** (COLMAP covered 24% of a bedroom;
  learned features 49 cm error → total collapse). This is why v2 uses phone poses.
- **Full-room 3DGS training degraded after ~4–5k steps** (PSNR 22.8 → 13.7 at 15k) on the
  ARKitScenes bedroom, cause unknown. Watch for it with DN-Splatter.
- **Evaluate every ~1k steps and look at actual renders.** Final-only PSNR and SSIM hid
  failures.
- **Check the data before training:** pose spread, outliers, exposure.
- **Coordinate conventions:** NeRFCapture/instant-ngp `transforms.json` is camera→world,
  OpenGL (flip y and z for OpenCV/COLMAP). ARKitScenes `.traj` is world→camera, axis-angle,
  OpenCV. COLMAP scenes are +Z up. Verify, don't assume.
- **The viewer is the bottleneck, not training memory.** Browser: ~8 fps p95 at 1M
  splats; ~400–500k is comfortable. Re-measure for Unity.
- 6 GB VRAM was never the binding limit (1M Gaussians ≈ 3.9 GB).
- **Kaggle** (T4 ×2, 16 GB, headless) is **not** a ready fallback for v2:
  `scripts/kaggle/push.sh` is built for the v1 gsplat scripts and would need porting.
  The v1 gsplat scripts are the out-of-stack fallback if DN-Splatter can't be made to work.
- Keep evidence out of git-ignored folders; a report was lost that way.
- `pip install` of extra packages can silently replace a pinned torch. Pin with a
  constraints file.

## Conventions

- Commits: `feat:` `fix:` `test:` `docs:` `exp:`, small and scoped. Branch first for
  non-trivial work.
- Every measured number states the machine it came from (PC / GPU machine / Kaggle).
- Ask the user before deleting anything, overwriting data, or pushing.
