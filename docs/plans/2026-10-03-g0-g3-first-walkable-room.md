# G0–G3: First Walkable Room — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MuSHRoom `vr_room` trained with DN-Splatter on the GPU machine, exported as splat + mesh + collision, walked through in the SuperSplat Viewer, and measured (goals G0–G3 of the spec).

**Architecture:** Existing tools do the work: DN-Splatter (nerfstudio plugin) trains and meshes on the GPU machine inside its own pixi environment; `splat-transform` builds the collision mesh and the SuperSplat Viewer provides walk mode on the PC. Our code is a small Python package `roomrecon/` (numpy only, unit-tested on the PC) plus shell wrappers that enforce safe training settings and write a `run.json` per run.

**Tech Stack:** pixi, Python 3.10, torch 2.2 + CUDA 11.8 (pixi-managed), nerfstudio @ a64026f, DN-Splatter (main, pinned at clone), Node 25 (PC), `@playcanvas/splat-transform`, `supersplat-viewer`, pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-10-03-roomrecon-v2-design.md` (revision 2). Read it with this plan.

## Global Constraints

- Deadline 2026-10-11. Goals are worked in order; G0–G3 is the minimum demo.
- Reuse existing software first; our code only for glue, settings, `scene.json`, checks.
- Every training run sets: `--steps-per-save 1000 --save-only-latest-checkpoint False --steps-per-eval-all-images 1000` and caches images on the CPU (spec §5.1).
- MuSHRoom runs use DN-Splatter's `mushroom` dataparser (it already uses no recentring/rescaling). Own/ARKitScenes data (later goals) must use `--orientation-method none --center-method none --auto-scale-poses False`.
- No fabricated numbers: every number in `results/` comes from a file produced by a run, with the machine named.
- GPU machine: `ssh gpu`; shared A100 (~5 GB was free on 2026-10-03); no sudo for Claude; no backup; jobs run in `tmux`.
- Claude runs remote commands with: `"<script>" | ssh -o BatchMode=yes gpu "tr -d '\r' | sed 's/^\xEF\xBB\xBF//' | bash -s"` (PowerShell adds BOM/CRLF).
- Never write the HPC password anywhere.
- Paths on the GPU machine: tools `~/roomrecon/tools/`, data `~/roomrecon/datasets/`, runs `~/roomrecon/scenes/`, repo clone `~/roomrecon/repo/`.
- Repo on the PC: `E:\IIT Jammu Internship\IIITK CLASS\SEM-7\Computer_vision\RoomRecon`, GitHub `asw-beep/RoomRecon-v2` (private).

## Review Focus

1. **Co-tenant takes the GPU memory mid-run** → the run dies, but resumes from the last 1k checkpoint with no lost progress beyond 1k steps (Task 6 kill/resume test).
2. **Too little free VRAM at start** → the wrapper refuses to start and says how much is free, instead of crashing later (Task 3 `--need-mib`, used in Task 6).
3. **Splat, collision mesh and spawn point in different frames** → you spawn outside or inside a wall. Expected: all three derived from the same exported frame; checked by the walk test and by the splat-transform convention test (Task 8, Task 10).
4. **Floaters under the floor or above the ceiling** → wrong floor height / spawn. Expected: floor estimate ignores sparse points and uses the dominant horizontal surface below the cameras (Task 8 test with injected floaters).
5. **Non-metric scene scale** → eye height and capsule size wrong in walk mode. Expected: scale normalised so median camera height above the floor is 1.45 m, with the applied factor recorded in `scene.json` (Task 8 test).

---

## File Structure (new repo)

```
RoomRecon/
  CLAUDE.md                       project + workspace instructions (moved from old repo)
  README.md                       what this is, how to run G0–G3
  .gitignore
  pyproject.toml                  package metadata + pytest config
  docs/
    spec/2026-10-03-roomrecon-v2-design.md
    plans/2026-10-03-g0-g3-first-walkable-room.md
    lessons-learned-v1.md         copied from the old repo
  env/
    dn-splatter.commit            pinned DN-Splatter commit hash
    pixi.lock                     lockfile copied back from the GPU machine
  roomrecon/
    __init__.py
    gpu.py                        free memory on the shared MIG GPU; sampler
    runrecord.py                  run.json start/finish
    scene.py                      up axis, floor, scale, spawn, Euler angles (numpy only)
    scene_cli.py                  reads exported cameras + mesh, writes scene.json (GPU machine)
  scripts/
    setup_gpu.sh                  pixi + DN-Splatter env (idempotent, no sudo)
    fetch_mushroom.sh             vr_room data + reference mesh + normals
    train_room.sh                 ns-train with safe settings + run.json + VRAM sampling
    tb_metrics.py                 eval-every-1k curve from tensorboard events → CSV
    export_room.sh                splat + cameras + mesh + scene.json
    make_test_ply.py              tiny 3-splat PLY for the Euler-convention check
    collision_room.ps1            PC: splat-transform → aligned splat + collision.glb
  tests/
    test_gpu.py  test_runrecord.py  test_scene.py  test_scene_cli.py
  results/                        small evidence only (run.json, metrics CSV/JSON, notes)
```

Large files (datasets, checkpoints, `.ply`, `.glb`, videos) never go into git.

---

### Task 1: G0 — Create the new repository

**Files:**
- Create: `RoomRecon/.gitignore`, `RoomRecon/README.md`, `RoomRecon/pyproject.toml`, `RoomRecon/roomrecon/__init__.py`, `RoomRecon/results/.gitkeep`
- Copy: `Project/CLAUDE.md` → `RoomRecon/CLAUDE.md`; spec → `docs/spec/`; this plan → `docs/plans/`; `Project/docs/lessons-learned.md` → `docs/lessons-learned-v1.md`

**Interfaces:**
- Produces: the repo every later task commits into; package name `roomrecon`; pytest run from repo root.

- [ ] **Step 1: Create the folder and copy the documents (PowerShell on the PC)**

```powershell
$old = "E:\IIT Jammu Internship\IIITK CLASS\SEM-7\Computer_vision\Project"
$new = "E:\IIT Jammu Internship\IIITK CLASS\SEM-7\Computer_vision\RoomRecon"
New-Item -ItemType Directory -Force "$new\docs\spec","$new\docs\plans","$new\roomrecon","$new\scripts","$new\tests","$new\results","$new\env" | Out-Null
Copy-Item "$old\CLAUDE.md" "$new\CLAUDE.md"
Copy-Item "$old\docs\superpowers\specs\2026-10-03-roomrecon-v2-design.md" "$new\docs\spec\"
Copy-Item "$old\docs\superpowers\plans\2026-10-03-g0-g3-first-walkable-room.md" "$new\docs\plans\"
Copy-Item "$old\docs\lessons-learned.md" "$new\docs\lessons-learned-v1.md"
```

- [ ] **Step 2: Edit `RoomRecon/CLAUDE.md` for its new home**

Replace the "Restart notice" block with:

```markdown
> **History.** v1 (Aug–Sep 2026: RGB-only, 6 GB laptop, capture-gate paper) lives in the old
> repo `..\Project` (GitHub `asw-beep/RoomRecon`), kept as an archive. Its lessons are in
> `docs/lessons-learned-v1.md`. Do not follow v1's ROADMAP/ADRs.
```

Then change the spec path `docs/superpowers/specs/2026-10-03-roomrecon-v2-design.md` to `docs/spec/2026-10-03-roomrecon-v2-design.md`, `docs/lessons-learned.md` to `docs/lessons-learned-v1.md`, and the "Current status" step 1 to: `1. Execute docs/plans/2026-10-03-g0-g3-first-walkable-room.md task by task.` Replace `storage/projects/m1-mushroom-inspect/` with `..\Project\storage\projects\m1-mushroom-inspect\ (old repo, on the PC)`. Replace `(scripts/compact-wsl-vhdx.txt)` with `(..\Project\scripts\compact-wsl-vhdx.txt)`.

- [ ] **Step 3: Write `.gitignore`**

```gitignore
# data and outputs never go into git (they live on the GPU machine)
*.ply
*.glb
*.obj
*.ckpt
*.mp4
*.tar.gz
*.zip
datasets/
scenes/
outputs/
__pycache__/
.pytest_cache/
*.egg-info/
.venv/
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[project]
name = "roomrecon"
version = "0.1.0"
description = "Phone capture of a room -> walkable 3D model (splat + collision)"
requires-python = ">=3.10"
dependencies = ["numpy"]

[tool.setuptools]
packages = ["roomrecon"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: Write `roomrecon/__init__.py` and `README.md`**

`roomrecon/__init__.py`:

```python
"""RoomRecon v2: glue around DN-Splatter and SuperSplat. See docs/spec/."""
```

`README.md`:

```markdown
# RoomRecon v2

Phone capture of an indoor room → a Gaussian splat you can walk around in, with collisions.
Training runs on the campus GPU machine (DN-Splatter); viewing runs on the PC (SuperSplat Viewer).

- Design: `docs/spec/2026-10-03-roomrecon-v2-design.md`
- Current plan: `docs/plans/2026-10-03-g0-g3-first-walkable-room.md`
- Unit tests (PC): `python -m pytest`
```

- [ ] **Step 6: Install test tools on the PC and check pytest runs**

Run: `python -m pip install --user numpy pytest` then `python -m pytest` (from `RoomRecon/`)
Expected: `no tests ran` (exit code 5). That is correct at this point.

- [ ] **Step 7: Initialise git, create the private GitHub repo, push**

```powershell
cd "E:\IIT Jammu Internship\IIITK CLASS\SEM-7\Computer_vision\RoomRecon"
git init -b main
git add .
git commit -m "docs: start RoomRecon v2 with spec, plan, CLAUDE.md and v1 lessons"
gh repo create asw-beep/RoomRecon-v2 --private --source . --push
```

Expected: `gh` prints the new repo URL; `git log --oneline` shows one commit.

- [ ] **Step 8: Give the GPU machine read-only access (deploy key)**

```powershell
"ssh-keygen -t ed25519 -N '' -f ~/.ssh/roomrecon_deploy -C roomrecon-gpu -q; cat ~/.ssh/roomrecon_deploy.pub" | ssh -o BatchMode=yes gpu "tr -d '\r' | sed 's/^\xEF\xBB\xBF//' | bash -s" > "$env:TEMP\roomrecon_deploy.pub"
gh repo deploy-key add "$env:TEMP\roomrecon_deploy.pub" --repo asw-beep/RoomRecon-v2 --title "gpu-machine (read-only)"
```

Then on the GPU machine (via the same ssh pipe):

```bash
mkdir -p ~/roomrecon
cat >> ~/.ssh/config <<'EOF'
Host github-roomrecon
    HostName github.com
    User git
    IdentityFile ~/.ssh/roomrecon_deploy
    IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
ssh -o StrictHostKeyChecking=accept-new -T github-roomrecon 2>&1 | head -1
git clone git@github-roomrecon:asw-beep/RoomRecon-v2.git ~/roomrecon/repo
git -C ~/roomrecon/repo log --oneline -1
```

Expected: the `ssh -T` line says `Hi asw-beep/RoomRecon-v2! You've successfully authenticated`; the clone shows the same commit as the PC.

---

### Task 2: G0 — DN-Splatter environment on the GPU machine

**Files:**
- Create: `scripts/setup_gpu.sh`, `env/dn-splatter.commit`, `env/pixi.lock` (copied back), `results/env-gpu.md`

**Interfaces:**
- Produces: `~/.pixi/bin/pixi`; DN-Splatter clone at `~/roomrecon/tools/dn-splatter` with a working pixi env; every later remote command runs tools as `~/.pixi/bin/pixi run --manifest-path ~/roomrecon/tools/dn-splatter/pixi.toml <cmd>`.

- [ ] **Step 1: Write `scripts/setup_gpu.sh`**

```bash
#!/bin/bash
# GPU machine environment for RoomRecon v2: pixi + DN-Splatter (no sudo needed).
# Idempotent: re-running skips what exists. Usage: bash scripts/setup_gpu.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$HOME/roomrecon/tools"
DNS="$TOOLS/dn-splatter"
PIXI="$HOME/.pixi/bin/pixi"
mkdir -p "$TOOLS" "$HOME/roomrecon/datasets" "$HOME/roomrecon/scenes"

[ -x "$PIXI" ] || curl -fsSL https://pixi.sh/install.sh | bash
"$PIXI" --version

if [ ! -d "$DNS/.git" ]; then
  git clone https://github.com/maturk/dn-splatter "$DNS"
fi
PIN="$(cat "$REPO/env/dn-splatter.commit" 2>/dev/null || true)"
if [ -n "$PIN" ]; then git -C "$DNS" checkout -q "$PIN"; fi
git -C "$DNS" rev-parse HEAD

cd "$DNS"
"$PIXI" install
"$PIXI" run download-omnidata
"$PIXI" run python -c "import torch, nerfstudio, gsplat; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('gsplat', gsplat.__version__)"
"$PIXI" run pip install -e "$REPO" pytest
echo "setup_gpu: OK"
```

- [ ] **Step 2: Commit and run it (first run, no pin yet)**

On the PC: `git add scripts/setup_gpu.sh; git commit -m "feat: GPU machine setup script (pixi + DN-Splatter)"; git push`.
On the GPU machine, in tmux (the install takes a while):

```bash
cd ~/roomrecon/repo
tmux new-session -d -s setup "bash scripts/setup_gpu.sh 2>&1 | tee ~/roomrecon/setup_gpu.log"
```

Wait for it to finish (check with `tail -5 ~/roomrecon/setup_gpu.log`).
Expected last lines: a `torch 2.2.x cuda 11.8 gpu True NVIDIA A100-PCIE-40GB ...` line, a `gsplat 1.0.0` line, `setup_gpu: OK`.
If `pixi install` fails on glibc: run `ldd --version | head -1` (pixi.toml needs glibc ≥ 2.30; Ubuntu 20.04 has 2.31) and report the exact error to the user before trying anything else.

- [ ] **Step 3: Verify the training flags this plan relies on exist**

```bash
cd ~/roomrecon/tools/dn-splatter
~/.pixi/bin/pixi run ns-train dn-splatter --help 2>&1 | grep -E -- "--steps-per-save|--save-only-latest-checkpoint|--steps-per-eval-all-images|--load-dir|--max-num-iterations|cache-images" 
~/.pixi/bin/pixi run ns-train dn-splatter mushroom --help 2>&1 | grep -E -- "--data|--mode"
which ns-export gs-mesh ns-eval 2>/dev/null; ~/.pixi/bin/pixi run which ns-export gs-mesh ns-eval
```

Expected: every flag is listed. If `cache-images` is missing, record the datamanager flags that *are* listed (`grep -i datamanager`) and tell the user; Task 6 then drops that flag and relies on the free-memory check instead.

- [ ] **Step 4: Pin and record**

```bash
git -C ~/roomrecon/tools/dn-splatter rev-parse HEAD
```

On the PC: write that hash into `env/dn-splatter.commit`; copy the lockfile: `scp gpu:roomrecon/tools/dn-splatter/pixi.lock env/pixi.lock`. Write `results/env-gpu.md` with the exact lines printed in Step 2 (torch/CUDA/GPU/gsplat), the commit hash, the date, and "machine: GPU machine (A100-PCIE-40GB MIG 7g.40gb)".

- [ ] **Step 5: Commit**

```powershell
git add env/ results/env-gpu.md
git commit -m "feat: pin DN-Splatter commit and pixi lockfile for the GPU machine"
git push
```

---

### Task 3: Free GPU memory on the shared MIG GPU (`roomrecon/gpu.py`)

**Files:**
- Create: `roomrecon/gpu.py`, `tests/test_gpu.py`

**Interfaces:**
- Produces: `parse_mig_memory(text: str) -> tuple[int, int]` (used_mib, total_mib); `free_mib() -> int`; CLI `python -m roomrecon.gpu --need-mib N` (exit 0 if free ≥ N else exit 1, prints `free_mib=<n>`); CLI `python -m roomrecon.gpu --sample <csv> --interval <s>` (appends `unix_time,used_mib,total_mib` rows forever).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_gpu.py
import pytest
from roomrecon.gpu import parse_mig_memory

# Real `nvidia-smi` excerpt from the GPU machine, 2026-10-03.
SMI = """\
+-----------------------------------------------------------------------------------------+
| MIG devices:                                                                            |
+------------------+----------------------------------+-----------+-----------------------+
| GPU  GI  CI  MIG |                     Memory-Usage |        Vol|      Shared           |
|      ID  ID  Dev |                       BAR1-Usage | SM     Unc| CE ENC DEC OFA JPG    |
|==================+==================================+===========+=======================|
|  0    0   0   0  |           35278MiB / 40447MiB    | 98      0 |  7   0    5    1    1 |
|                  |                13MiB / 65536MiB  |           |                       |
+------------------+----------------------------------+-----------+-----------------------+
"""

def test_parses_mig_row_not_bar1_row():
    assert parse_mig_memory(SMI) == (35278, 40447)

def test_no_mig_table_raises():
    with pytest.raises(ValueError, match="no MIG device row"):
        parse_mig_memory("| NVIDIA-SMI 555.42.06 |\n")
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_gpu.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'roomrecon.gpu'`

- [ ] **Step 3: Implement**

```python
# roomrecon/gpu.py
"""Free memory on the shared MIG A100.

Inside a MIG instance `nvidia-smi --query-gpu=memory.*` returns "Insufficient Permissions",
so we read the MIG device table of plain `nvidia-smi` instead. Another tenant uses the same
card, so free memory must be checked at job start, not assumed.
"""
import argparse
import re
import subprocess
import sys
import time

_MIG_ROW = re.compile(r"^\|\s+\d+\s+\d+\s+\d+\s+\d+\s+\|\s+(\d+)MiB\s*/\s*(\d+)MiB")


def parse_mig_memory(text: str) -> tuple[int, int]:
    """(used_mib, total_mib) of the first MIG device row in `nvidia-smi` output."""
    for line in text.splitlines():
        m = _MIG_ROW.match(line)
        if m:
            return int(m.group(1)), int(m.group(2))
    raise ValueError("no MIG device row found in nvidia-smi output")


def _smi() -> str:
    return subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout


def free_mib() -> int:
    used, total = parse_mig_memory(_smi())
    return total - used


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--need-mib", type=int, help="exit 1 if less than this is free")
    ap.add_argument("--sample", help="append used/total MiB rows to this CSV forever")
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args(argv)
    if a.sample:
        while True:
            used, total = parse_mig_memory(_smi())
            with open(a.sample, "a") as f:
                f.write(f"{int(time.time())},{used},{total}\n")
            time.sleep(a.interval)
    free = free_mib()
    print(f"free_mib={free}")
    if a.need_mib is not None and free < a.need_mib:
        print(f"refusing to start: {free} MiB free, {a.need_mib} MiB needed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gpu.py -v`
Expected: 2 passed

- [ ] **Step 5: Check it against the real machine**

Push, then on the GPU machine: `cd ~/roomrecon/repo && ~/.pixi/bin/pixi run --manifest-path ~/roomrecon/tools/dn-splatter/pixi.toml python -m roomrecon.gpu`
Expected: `free_mib=<number>`, consistent with the MIG row of `nvidia-smi` run right after.

- [ ] **Step 6: Commit**

```powershell
git add roomrecon/gpu.py tests/test_gpu.py
git commit -m "feat: read free memory from the MIG table on the shared GPU"
git push
```

---

### Task 4: Run records (`roomrecon/runrecord.py`)

**Files:**
- Create: `roomrecon/runrecord.py`, `tests/test_runrecord.py`

**Interfaces:**
- Produces: `start(path, *, run_id, room, method, command, repo_commit, dn_splatter_commit, free_mib, machine) -> dict` writes `run.json` with `status="running"` and `started_at` (UTC ISO); `finish(path, *, exit_code, vram_csv=None) -> dict` adds `ended_at`, `duration_s`, `status` (`"ok"` if exit 0 else `"failed"`) and, if the CSV exists, `gpu_used_mib_max` and `gpu_used_mib_at_start`. CLI: `python -m roomrecon.runrecord start <path> --run-id ... --room ... --method ... --command ... --repo-commit ... --dn-splatter-commit ... --free-mib ... --machine ...` and `python -m roomrecon.runrecord finish <path> --exit-code N [--vram-csv f]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_runrecord.py
import json
from roomrecon import runrecord

def _start(tmp_path):
    p = tmp_path / "run.json"
    runrecord.start(p, run_id="r1", room="vr_room", method="dn-splatter",
                    command="ns-train ...", repo_commit="abc", dn_splatter_commit="def",
                    free_mib=5169, machine="gpu-machine A100 MIG")
    return p

def test_start_writes_running_record(tmp_path):
    p = _start(tmp_path)
    rec = json.loads(p.read_text())
    assert rec["status"] == "running"
    assert rec["free_mib_at_start"] == 5169
    assert rec["started_at"].endswith("Z")

def test_finish_ok_with_vram_samples(tmp_path):
    p = _start(tmp_path)
    csv = tmp_path / "vram.csv"
    csv.write_text("100,35000,40447\n110,39000,40447\n120,37000,40447\n")
    rec = runrecord.finish(p, exit_code=0, vram_csv=csv)
    assert rec["status"] == "ok"
    assert rec["gpu_used_mib_at_start"] == 35000
    assert rec["gpu_used_mib_max"] == 39000
    assert rec["duration_s"] >= 0

def test_finish_failed_without_csv(tmp_path):
    p = _start(tmp_path)
    rec = runrecord.finish(p, exit_code=137, vram_csv=tmp_path / "missing.csv")
    assert rec["status"] == "failed" and rec["exit_code"] == 137
    assert "gpu_used_mib_max" not in rec
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_runrecord.py -v`
Expected: FAIL with `ImportError: cannot import name 'runrecord'`

- [ ] **Step 3: Implement**

```python
# roomrecon/runrecord.py
"""run.json: one record per training/export run, so every number is traceable.

GPU memory on the shared card is sampled for the whole machine; our own use is roughly
max - at_start, valid only if the other tenant's use stayed constant. Both raw values
are kept so the reader can judge.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def start(path, *, run_id, room, method, command, repo_commit, dn_splatter_commit,
          free_mib, machine) -> dict:
    rec = {"run_id": run_id, "room": room, "method": method, "command": command,
           "repo_commit": repo_commit, "dn_splatter_commit": dn_splatter_commit,
           "machine": machine, "free_mib_at_start": int(free_mib),
           "started_at": _now(), "status": "running"}
    Path(path).write_text(json.dumps(rec, indent=2))
    return rec


def finish(path, *, exit_code, vram_csv=None) -> dict:
    rec = json.loads(Path(path).read_text())
    rec["ended_at"] = _now()
    t0 = datetime.strptime(rec["started_at"], "%Y-%m-%dT%H:%M:%SZ")
    t1 = datetime.strptime(rec["ended_at"], "%Y-%m-%dT%H:%M:%SZ")
    rec["duration_s"] = int((t1 - t0).total_seconds())
    rec["exit_code"] = int(exit_code)
    rec["status"] = "ok" if exit_code == 0 else "failed"
    if vram_csv and Path(vram_csv).exists():
        used = [int(line.split(",")[1]) for line in Path(vram_csv).read_text().splitlines() if line]
        if used:
            rec["gpu_used_mib_at_start"] = used[0]
            rec["gpu_used_mib_max"] = max(used)
    Path(path).write_text(json.dumps(rec, indent=2))
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("path")
    for k in ("run-id", "room", "method", "command", "repo-commit",
              "dn-splatter-commit", "machine"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--free-mib", type=int, required=True)
    f = sub.add_parser("finish")
    f.add_argument("path")
    f.add_argument("--exit-code", type=int, required=True)
    f.add_argument("--vram-csv")
    a = ap.parse_args(argv)
    if a.cmd == "start":
        start(a.path, run_id=a.run_id, room=a.room, method=a.method, command=a.command,
              repo_commit=a.repo_commit, dn_splatter_commit=a.dn_splatter_commit,
              free_mib=a.free_mib, machine=a.machine)
    else:
        finish(a.path, exit_code=a.exit_code, vram_csv=a.vram_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -v`
Expected: 5 passed (2 gpu + 3 runrecord)

- [ ] **Step 5: Commit**

```powershell
git add roomrecon/runrecord.py tests/test_runrecord.py
git commit -m "feat: run.json start/finish records with shared-GPU memory samples"
git push
```

---

### Task 5: G0 — `vr_room` data, reference mesh and normals on the GPU machine

**Files:**
- Create: `scripts/fetch_mushroom.sh`, `results/data-vr_room.md`

**Interfaces:**
- Consumes: the PC copy `..\Project\storage\projects\m1-mushroom-inspect\vr_room_iphone.tar.gz`.
- Produces: `~/roomrecon/datasets/mushroom/room_datasets/vr_room/iphone/{long_capture,short_capture}/` with `normals_from_pretrain/` generated; reference mesh files extracted under `~/roomrecon/datasets/mushroom/`; `DATA=~/roomrecon/datasets/mushroom/room_datasets/vr_room` used by Task 6.

- [ ] **Step 1: Write `scripts/fetch_mushroom.sh`**

```bash
#!/bin/bash
# MuSHRoom room → ~/roomrecon/datasets/mushroom (iPhone sequence + reference mesh) + normals.
# Usage: bash scripts/fetch_mushroom.sh vr_room
# The iPhone tarball may already be uploaded (scp from the PC); otherwise it is downloaded.
set -euo pipefail
ROOM="${1:?room name, e.g. vr_room}"
D="$HOME/roomrecon/datasets/mushroom"
DNS="$HOME/roomrecon/tools/dn-splatter"
PIXI="$HOME/.pixi/bin/pixi"
mkdir -p "$D"; cd "$D"
[ -f "${ROOM}_iphone.tar.gz" ] || wget -q "https://zenodo.org/records/10230733/files/${ROOM}_iphone.tar.gz"
[ -f "${ROOM}_mesh_pd.tar.gz" ] || wget -q "https://zenodo.org/records/10222321/files/${ROOM}_mesh_pd.tar.gz"
sha256sum "${ROOM}_iphone.tar.gz" "${ROOM}_mesh_pd.tar.gz" | tee "${ROOM}.sha256"
[ -d "room_datasets/${ROOM}/iphone/long_capture" ] || tar -xzf "${ROOM}_iphone.tar.gz"
tar -tzf "${ROOM}_mesh_pd.tar.gz" | head -20
tar -xzf "${ROOM}_mesh_pd.tar.gz"
cd "$DNS"
for seq in long_capture short_capture; do
  P="$D/room_datasets/${ROOM}/iphone/${seq}"
  if [ ! -d "$P/normals_from_pretrain" ]; then
    "$PIXI" run python dn_splatter/scripts/normals_from_pretrain.py --data-dir "$P" --resolution low
  fi
  echo "$seq images=$(ls "$P/images" | wc -l) normals=$(ls "$P/normals_from_pretrain" | wc -l)"
done
```

- [ ] **Step 2: Upload the tarball from the PC (saves a 233 MB download) and run**

```powershell
ssh gpu "mkdir -p ~/roomrecon/datasets/mushroom"
scp "E:\IIT Jammu Internship\IIITK CLASS\SEM-7\Computer_vision\Project\storage\projects\m1-mushroom-inspect\vr_room_iphone.tar.gz" gpu:roomrecon/datasets/mushroom/
```

Commit + push the script, then on the GPU machine: `cd ~/roomrecon/repo && tmux new-session -d -s data "bash scripts/fetch_mushroom.sh vr_room 2>&1 | tee ~/roomrecon/fetch_vr_room.log"`.
Expected end of log: `long_capture images=N normals=N` and `short_capture images=M normals=M` with equal counts per line.
If `normals_from_pretrain` writes somewhere else (count 0), run `ls $P` to find the folder it created and tell the user before Task 6; the dataparser must find normals there.

- [ ] **Step 3: Record what the reference mesh tarball contains**

From the log, write `results/data-vr_room.md`: the sha256 lines, the image/normal counts, the file list printed by `tar -tzf ... | head -20`, and which file is the reference mesh and which is the ICP/alignment file (needed in Task 11).

- [ ] **Step 4: Commit**

```powershell
git add scripts/fetch_mushroom.sh results/data-vr_room.md
git commit -m "feat: fetch MuSHRoom room data, reference mesh and normals"
git push
```

---

### Task 6: G1 — Training wrapper, smoke run, kill-and-resume test

**Files:**
- Create: `scripts/train_room.sh`, `scripts/tb_metrics.py`, `results/g1-resume-test.md`

**Interfaces:**
- Consumes: `roomrecon.gpu` CLI (Task 3), `roomrecon.runrecord` CLI (Task 4), `DATA` from Task 5.
- Produces: `scripts/train_room.sh <room> <run_id> <method> <data_dir> <max_iters> [extra ns-train args...]`. Outputs to `~/roomrecon/scenes/<room>/<run_id>/`: `train/<room>/<method>/<run_id>/config.yml`, `.../nerfstudio_models/step-*.ckpt`, `run.json`, `vram.csv`, `train.log`. If `RESUME_FROM=<nerfstudio_models dir>` is set, passes `--load-dir`. Minimum free memory: `NEED_MIB` (default 4000).
- `scripts/tb_metrics.py <train_dir> <out.csv>` writes `step,tag,value` for every eval scalar.

- [ ] **Step 1: Write `scripts/train_room.sh`**

```bash
#!/bin/bash
# Train one room with the settings the spec requires (§5.1), recording run.json.
# Usage: scripts/train_room.sh <room> <run_id> <method> <data_dir> <max_iters> [extra ns-train args...]
#   method: dn-splatter | splatfacto
# Env: NEED_MIB (default 4000) minimum free GPU memory; RESUME_FROM=<.../nerfstudio_models>
set -uo pipefail
ROOM="$1"; RUN="$2"; METHOD="$3"; DATA="$4"; ITERS="$5"; shift 5
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"
PIXI="$HOME/.pixi/bin/pixi"
PY() { "$PIXI" run --manifest-path "$DNS/pixi.toml" python "$@"; }
OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"; mkdir -p "$OUT"
cd "$REPO"

PY -m roomrecon.gpu --need-mib "${NEED_MIB:-4000}" || exit 3
FREE=$(PY -m roomrecon.gpu | sed 's/free_mib=//')

MODEL_ARGS=()
if [ "$METHOD" = "dn-splatter" ]; then
  MODEL_ARGS=(--pipeline.model.use-depth-loss True --pipeline.model.depth-lambda 0.2
              --pipeline.model.use-depth-smooth-loss True --pipeline.model.use-normal-loss True
              --pipeline.model.normal-supervision mono)
fi
RESUME=(); [ -n "${RESUME_FROM:-}" ] && RESUME=(--load-dir "$RESUME_FROM")

CMD=(ns-train "$METHOD" --output-dir "$OUT/train" --experiment-name "$ROOM" --timestamp "$RUN"
     --max-num-iterations "$ITERS" --steps-per-save 1000 --save-only-latest-checkpoint False
     --steps-per-eval-all-images 1000 --vis tensorboard --viewer.quit-on-train-completion True
     --pipeline.datamanager.cache-images cpu
     "${MODEL_ARGS[@]}" "${RESUME[@]}" "$@" mushroom --data "$DATA" --mode iphone)

PY -m roomrecon.runrecord start "$OUT/run.json" --run-id "$RUN" --room "$ROOM" --method "$METHOD" \
   --command "${CMD[*]}" --repo-commit "$(git rev-parse HEAD)" \
   --dn-splatter-commit "$(git -C "$DNS" rev-parse HEAD)" --free-mib "$FREE" \
   --machine "GPU machine A100-PCIE-40GB MIG 7g.40gb (shared)"

PY -m roomrecon.gpu --sample "$OUT/vram.csv" --interval 10 & SAMPLER=$!
( cd "$DNS" && "$PIXI" run "${CMD[@]}" ) 2>&1 | tee -a "$OUT/train.log"
CODE=${PIPESTATUS[0]}
kill "$SAMPLER" 2>/dev/null
PY -m roomrecon.runrecord finish "$OUT/run.json" --exit-code "$CODE" --vram-csv "$OUT/vram.csv"
echo "train_room: exit $CODE -> $OUT"
exit "$CODE"
```

(If Task 2 Step 3 found no `cache-images` flag, delete that one line.)

- [ ] **Step 2: Write `scripts/tb_metrics.py`**

```python
#!/usr/bin/env python
"""Dump every eval scalar from nerfstudio's tensorboard events to CSV (step,tag,value).

The final metric alone hid a training collapse in v1; this gives the whole curve.
Usage: python scripts/tb_metrics.py <train_dir> <out.csv>
"""
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

train_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for ev in sorted(train_dir.rglob("events.out.tfevents.*")):
    acc = EventAccumulator(str(ev.parent)); acc.Reload()
    for tag in acc.Tags()["scalars"]:
        if "Eval" in tag:
            rows += [(e.step, tag, e.value) for e in acc.Scalars(tag)]
rows = sorted(set(rows))
out.write_text("step,tag,value\n" + "".join(f"{s},{t},{v}\n" for s, t, v in rows))
print(f"{len(rows)} rows, tags: {sorted({t for _, t, _ in rows})}")
```

- [ ] **Step 3: Commit, push, pull on the GPU machine**

```powershell
git add scripts/train_room.sh scripts/tb_metrics.py
git commit -m "feat: training wrapper with safe settings, run.json and memory sampling"
git push
```

- [ ] **Step 4: Smoke run (3000 steps) — then kill it on purpose after a checkpoint**

On the GPU machine:

```bash
cd ~/roomrecon/repo
DATA=~/roomrecon/datasets/mushroom/room_datasets/vr_room
tmux new-session -d -s smoke "bash scripts/train_room.sh vr_room smoke dn-splatter $DATA 3000"
```

Watch `ls ~/roomrecon/scenes/vr_room/smoke/train/vr_room/dn-splatter/smoke/nerfstudio_models/`. As soon as `step-000001000.ckpt` (or the first saved step) appears and training has gone past it, kill the training process the way an OOM would:

```bash
pkill -9 -f "ns-train dn-splatter" ; sleep 5
cat ~/roomrecon/scenes/vr_room/smoke/run.json
```

Expected: the checkpoint file exists; `run.json` has `"status": "failed"` and a non-zero `exit_code`.
If the run dies before step 1000 with CUDA OOM: record `free_mib_at_start` and the error line in `results/g1-resume-test.md`, and tell the user (the co-tenant question goes to IT); do not change settings silently.

- [ ] **Step 5: Resume and finish**

```bash
M=~/roomrecon/scenes/vr_room/smoke/train/vr_room/dn-splatter/smoke/nerfstudio_models
tmux new-session -d -s smoke2 "RESUME_FROM=$M bash scripts/train_room.sh vr_room smoke-resumed dn-splatter $DATA 3000"
```

Expected: `train.log` of `smoke-resumed` reports loading the checkpoint and starts from the saved step (not step 0); it ends with `exit 0`; `ls` of its `nerfstudio_models` shows `step-000002000.ckpt` and `step-000002999.ckpt` (or the final step), and the first step file is the resumed one.

- [ ] **Step 6: Extract the curve and record the result**

```bash
cd ~/roomrecon/tools/dn-splatter
~/.pixi/bin/pixi run python ~/roomrecon/repo/scripts/tb_metrics.py ~/roomrecon/scenes/vr_room/smoke-resumed/train ~/roomrecon/scenes/vr_room/smoke-resumed/eval_curve.csv
```

Expected: rows for steps 1000/2000 (at least one eval tag containing `psnr`).
On the PC: `scp gpu:roomrecon/scenes/vr_room/smoke/run.json results/g1-smoke-run.json`, same for `smoke-resumed/run.json` → `results/g1-smoke-resumed-run.json` and `eval_curve.csv` → `results/g1-smoke-eval-curve.csv`. Write `results/g1-resume-test.md`: killed at which step, resumed from which checkpoint, resumed run's first logged step, both `run.json` summaries (free memory at start, used max, duration). Numbers copied from those files only.

- [ ] **Step 7: Commit**

```powershell
git add results/
git commit -m "test: G1 smoke run killed and resumed from checkpoint on the GPU machine"
git push
```

---

### Task 7: G1 — Full `vr_room` training run

**Files:**
- Create: `results/g1-vr_room-run.json`, `results/g1-vr_room-eval-curve.csv`, `results/g1-vr_room.md`

**Interfaces:**
- Produces: the trained run `~/roomrecon/scenes/vr_room/dns-30k/` with `config.yml` at `.../train/vr_room/dn-splatter/dns-30k/config.yml` (`CONFIG` in Tasks 9 and 11).

- [ ] **Step 1: Start the full run (30k steps, DN-Splatter's default length)**

```bash
cd ~/roomrecon/repo
DATA=~/roomrecon/datasets/mushroom/room_datasets/vr_room
tmux new-session -d -s full "bash scripts/train_room.sh vr_room dns-30k dn-splatter $DATA 30000"
```

If it dies (co-tenant OOM), restart with `RESUME_FROM=<its nerfstudio_models dir>` and run id `dns-30k-r1`; note every restart in `results/g1-vr_room.md`.

- [ ] **Step 2: When it ends, check for collapse**

```bash
cd ~/roomrecon/tools/dn-splatter
~/.pixi/bin/pixi run python ~/roomrecon/repo/scripts/tb_metrics.py ~/roomrecon/scenes/vr_room/dns-30k/train ~/roomrecon/scenes/vr_room/dns-30k/eval_curve.csv
grep -i psnr ~/roomrecon/scenes/vr_room/dns-30k/eval_curve.csv | sort -t, -k3 -g | tail -1
grep -i psnr ~/roomrecon/scenes/vr_room/dns-30k/eval_curve.csv | tail -1
```

Expected: the best PSNR row and the last PSNR row. If the last is well below the best (the v1 collapse pattern), write both steps and values into `results/g1-vr_room.md` and tell the user. Tasks 9–11 then use the best step's checkpoint: its `nerfstudio_models/step-<best>.ckpt` is kept because all checkpoints are kept.

- [ ] **Step 3: Copy evidence to the PC and commit**

```powershell
scp gpu:roomrecon/scenes/vr_room/dns-30k/run.json results/g1-vr_room-run.json
scp gpu:roomrecon/scenes/vr_room/dns-30k/eval_curve.csv results/g1-vr_room-eval-curve.csv
```

Write `results/g1-vr_room.md`: duration, free memory at start, GPU used max, best and final PSNR with steps, any restarts. Machine: GPU machine.

```powershell
git add results/
git commit -m "exp: G1 vr_room trained with DN-Splatter (30k) on the GPU machine"
git push
```

---

### Task 8: `scene.json` math (`roomrecon/scene.py`)

**Files:**
- Create: `roomrecon/scene.py`, `tests/test_scene.py`

**Interfaces:**
- Produces (all numpy, camera matrices are nerfstudio camera-to-world, OpenGL convention: camera +Y up, −Z forward):
  - `estimate_up(c2w: np.ndarray) -> np.ndarray` — unit vector, mean of camera +Y axes; `ValueError` if the mean has norm < 0.5.
  - `rotation_to_y_up(up: np.ndarray) -> np.ndarray` — 3×3 R with `R @ up = [0,1,0]`.
  - `floor_height(vertex_y: np.ndarray, camera_y: np.ndarray, bins: int = 200) -> float` — y of the densest horizontal layer of vertices below the lowest camera.
  - `scene_frame(c2w, vertices, eye_height_m: float = 1.45) -> dict` with keys `rotation` (3×3 list), `scale` (float), `floor_y` (after rotate+scale), `spawn_position` (list of 3), `spawn_forward` (list of 3, horizontal), `scale_source` (str), `median_camera_height_raw` (float).
  - `apply(points, frame) -> np.ndarray` — rotate then scale points as the frame says.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scene.py
import numpy as np
import pytest
from roomrecon import scene

def _rot_x(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _cams(R_world, heights, n=20):
    """Cameras looking horizontally around a room whose true up is R_world @ +Y."""
    c2w = np.zeros((n, 4, 4)); c2w[:, 3, 3] = 1
    for i in range(n):
        yaw = 2 * np.pi * i / n
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        c2w[i, :3, :3] = R_world @ Ry
        c2w[i, :3, 3] = R_world @ np.array([np.cos(yaw), heights[i % len(heights)], np.sin(yaw)])
    return c2w

def _room(R_world, floor=0.0, n=4000, floaters=0, rng=np.random.default_rng(0)):
    floor_pts = np.c_[rng.uniform(-2, 2, n), np.full(n, floor), rng.uniform(-2, 2, n)]
    wall_pts = np.c_[np.full(n // 4, 2.0), rng.uniform(floor, floor + 2.5, n // 4), rng.uniform(-2, 2, n // 4)]
    junk = np.c_[rng.uniform(-2, 2, floaters), rng.uniform(floor - 3, floor - 1, floaters), rng.uniform(-2, 2, floaters)]
    return (R_world @ np.vstack([floor_pts, wall_pts, junk]).T).T

def test_up_recovered_under_tilted_world():
    R = _rot_x(70)
    up = scene.estimate_up(_cams(R, [1.4, 1.5]))
    assert np.allclose(up, R @ [0, 1, 0], atol=1e-6)

def test_up_rejects_inconsistent_cameras():
    c2w = _cams(np.eye(3), [1.4]); c2w[::2, :3, 1] *= -1
    with pytest.raises(ValueError):
        scene.estimate_up(c2w)

def test_rotation_maps_up_to_y():
    up = np.array([0.3, 0.2, -0.9]); up /= np.linalg.norm(up)
    R = scene.rotation_to_y_up(up)
    assert np.allclose(R @ up, [0, 1, 0], atol=1e-9)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1)

def test_floor_ignores_floaters_below():
    y = np.r_[np.zeros(3000), np.random.default_rng(1).uniform(-3, -1, 300), np.full(500, 2.5)]
    assert abs(scene.floor_height(y, camera_y=np.array([1.4, 1.5]))) < 0.05

def test_scene_frame_normalises_scale_and_spawns_inside():
    R = _rot_x(-35)
    c2w = _cams(R, [0.70, 0.75])          # half-scale scene: cameras ~0.725 above floor
    verts = _room(R, floor=0.0, floaters=200) * 1.0
    f = scene.scene_frame(c2w, verts, eye_height_m=1.45)
    assert f["scale"] == pytest.approx(1.45 / 0.725, rel=0.05)
    assert f["floor_y"] == pytest.approx(0.0, abs=0.05)
    sp = np.array(f["spawn_position"])
    assert sp[1] == pytest.approx(f["floor_y"] + 1.45, abs=0.05)
    assert abs(sp[0]) < 2.0 * f["scale"] and abs(sp[2]) < 2.0 * f["scale"]
    assert abs(f["spawn_forward"][1]) < 1e-9 and np.isclose(np.linalg.norm(f["spawn_forward"]), 1)
    assert f["scale_source"] == "median camera height normalised to 1.45 m"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_scene.py -v`
Expected: FAIL with `ImportError: cannot import name 'scene'`

- [ ] **Step 3: Implement**

```python
# roomrecon/scene.py
"""Where is up, where is the floor, how big is the room, where do you spawn.

Inputs are in the exported (nerfstudio) frame: camera-to-world matrices in the OpenGL
convention (camera +Y up, -Z forward) and mesh/splat vertices. People hold phones upright,
so the mean camera +Y is gravity's up. The scene is rotated so up is +Y (the viewer's
convention) and scaled so the median camera height above the floor is a typical handheld
eye height; the factor is recorded, and a factor near 1 means the poses were already metric.
"""
import numpy as np


def estimate_up(c2w: np.ndarray) -> np.ndarray:
    u = c2w[:, :3, 1].mean(axis=0)
    n = np.linalg.norm(u)
    if n < 0.5:
        raise ValueError(f"camera up vectors disagree (mean norm {n:.2f}); cannot estimate up")
    return u / n


def rotation_to_y_up(up: np.ndarray) -> np.ndarray:
    a = up / np.linalg.norm(up)
    b = np.array([0.0, 1.0, 0.0])
    v = np.cross(a, b)
    c = float(a @ b)
    if np.isclose(c, -1.0):                      # up points straight down: turn 180° about X
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def floor_height(vertex_y: np.ndarray, camera_y: np.ndarray, bins: int = 200) -> float:
    below = vertex_y[vertex_y < camera_y.min()]
    if below.size == 0:
        raise ValueError("no geometry below the cameras; cannot find the floor")
    hist, edges = np.histogram(below, bins=bins)
    i = int(np.argmax(hist))
    return float((edges[i] + edges[i + 1]) / 2)


def scene_frame(c2w: np.ndarray, vertices: np.ndarray, eye_height_m: float = 1.45) -> dict:
    R = rotation_to_y_up(estimate_up(c2w))
    cam = (R @ c2w[:, :3, 3].T).T
    verts = (R @ vertices.T).T
    floor_raw = floor_height(verts[:, 1], cam[:, 1])
    h_raw = float(np.median(cam[:, 1] - floor_raw))
    if h_raw <= 0:
        raise ValueError("cameras are not above the floor estimate")
    s = eye_height_m / h_raw
    centre = np.median(cam, axis=0)
    k = int(np.argmin(np.linalg.norm(cam - centre, axis=1)))
    fwd = R @ (-c2w[k, :3, 2])
    fwd[1] = 0.0
    fwd = fwd / np.linalg.norm(fwd)
    spawn = cam[k] * s
    spawn[1] = floor_raw * s + eye_height_m
    return {"rotation": R.tolist(), "scale": s, "floor_y": floor_raw * s,
            "spawn_position": spawn.tolist(), "spawn_forward": fwd.tolist(),
            "scale_source": f"median camera height normalised to {eye_height_m} m",
            "median_camera_height_raw": h_raw}


def apply(points: np.ndarray, frame: dict) -> np.ndarray:
    return (np.asarray(frame["rotation"]) @ points.T).T * frame["scale"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```powershell
git add roomrecon/scene.py tests/test_scene.py
git commit -m "feat: up axis, floor, scale and spawn point for scene.json"
git push
```

---

### Task 9: G2 — Export splat, cameras, mesh and `scene.json`

**Files:**
- Create: `roomrecon/scene_cli.py`, `scripts/export_room.sh`, `results/g2-export.md`

**Interfaces:**
- Consumes: `CONFIG` from Task 7; `roomrecon.scene.scene_frame`, `roomrecon.scene.apply` (Task 8).
- Produces: `~/roomrecon/scenes/<room>/<run_id>/export/` with `splat.ply` (from `ns-export gaussian-splat`), `transforms_train.json` (from `ns-export cameras`), `mesh.ply` (from `gs-mesh o3dtsdf`), `scene.json`; and `scene.json` also holds `euler_xyz_deg` filled in by Task 10.
- `python -m roomrecon.scene_cli <export_dir> --room <room> --run-id <run_id>` → writes `<export_dir>/scene.json`.

- [ ] **Step 1: Write `roomrecon/scene_cli.py`**

```python
# roomrecon/scene_cli.py
"""Exported cameras + mesh -> scene.json. Runs on the GPU machine (needs open3d)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from roomrecon.scene import scene_frame


def load_c2w(transforms_json: Path) -> np.ndarray:
    """(N,4,4) camera-to-world from `ns-export cameras` output (list or {"frames": [...]},
    key "transform" or "transform_matrix", 3x4 or 4x4)."""
    data = json.loads(Path(transforms_json).read_text())
    frames = data["frames"] if isinstance(data, dict) else data
    mats = []
    for f in frames:
        m = np.array(f.get("transform", f.get("transform_matrix")), dtype=float)
        if m.shape == (3, 4):
            m = np.vstack([m, [0.0, 0.0, 0.0, 1.0]])
        if m.shape != (4, 4):
            raise ValueError(f"unexpected camera matrix shape {m.shape}")
        mats.append(m)
    return np.stack(mats)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--room", required=True)
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    d = Path(a.export_dir)
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(d / "mesh.ply"))
    verts = np.asarray(mesh.vertices)
    c2w = load_c2w(d / "transforms_train.json")
    frame = scene_frame(c2w, verts)
    frame.update({"room": a.room, "run_id": a.run_id, "up_axis": "+Y",
                  "units": "metres (approximate)", "camera_count": int(len(c2w)),
                  "mesh_vertex_count": int(len(verts)), "euler_xyz_deg": None})
    (d / "scene.json").write_text(json.dumps(frame, indent=2))
    print(json.dumps({k: frame[k] for k in ("scale", "floor_y", "spawn_position")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Add a test for the loader (runs on the PC; `open3d` is only imported inside `main`):

```python
# tests/test_scene_cli.py
import json
import numpy as np
from roomrecon.scene_cli import load_c2w

def test_load_c2w_list_of_3x4(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"file_path": "a.png", "transform": np.eye(4)[:3].tolist()}] * 2))
    assert load_c2w(p).shape == (2, 4, 4) and np.allclose(load_c2w(p)[0], np.eye(4))

def test_load_c2w_frames_dict_4x4(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"frames": [{"transform_matrix": np.eye(4).tolist()}]}))
    assert load_c2w(p).shape == (1, 4, 4)
```

Run: `python -m pytest tests/test_scene_cli.py -v` → Expected: 2 passed.
Step 3 also prints the first frame of the real `transforms_train.json`; if its key is neither `transform` nor `transform_matrix`, add that key to `load_c2w`, add a test case for it, and rerun.

- [ ] **Step 2: Write `scripts/export_room.sh`**

```bash
#!/bin/bash
# Export a trained room: splat, cameras, mesh, scene.json.
# Usage: scripts/export_room.sh <room> <run_id> [voxel_size]
set -euo pipefail
ROOM="$1"; RUN="$2"; VOX="${3:-0.02}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"; EXP="$OUT/export"; mkdir -p "$EXP"
CONFIG=$(ls "$OUT"/train/"$ROOM"/*/"$RUN"/config.yml | head -1)
cd "$DNS"
"$PIXI" run ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$EXP"
"$PIXI" run ns-export cameras --load-config "$CONFIG" --output-dir "$EXP"
"$PIXI" run gs-mesh o3dtsdf --load-config "$CONFIG" --output-dir "$EXP/mesh_o3dtsdf" --voxel-size "$VOX"
ls -la "$EXP" "$EXP/mesh_o3dtsdf"
cp "$(ls "$EXP"/mesh_o3dtsdf/*.ply | head -1)" "$EXP/mesh.ply"
cd "$REPO"
"$PIXI" run --manifest-path "$DNS/pixi.toml" python -m roomrecon.scene_cli "$EXP" --room "$ROOM" --run-id "$RUN"
```

- [ ] **Step 3: Check the gs-mesh flag name, then run the export**

```bash
cd ~/roomrecon/tools/dn-splatter && ~/.pixi/bin/pixi run gs-mesh o3dtsdf --help 2>&1 | grep -i -E "voxel|output|load"
```

If the voxel flag is spelled differently, fix the `gs-mesh` line in `export_room.sh` to the printed name before running. `0.02` is a room-scale voxel in the exported frame's units; if `scene.json` later shows `scale` far from 1 (non-metric frame), rerun with `0.02 / scale`.

Commit + push both files, then on the GPU machine:

```bash
cd ~/roomrecon/repo
bash scripts/export_room.sh vr_room dns-30k 2>&1 | tee ~/roomrecon/scenes/vr_room/dns-30k/export.log
head -c 600 ~/roomrecon/scenes/vr_room/dns-30k/export/transforms_train.json
cat ~/roomrecon/scenes/vr_room/dns-30k/export/scene.json
```

Expected: `splat.ply` exists (the name `ns-export gaussian-splat` writes; if it is named differently, use that name everywhere below), `mesh.ply` exists, `scene.json` prints `scale`, `floor_y`, `spawn_position`.
Sanity rule: `median_camera_height_raw` between 0.5 and 3 means poses are close to metric; record the value either way.

- [ ] **Step 4: Copy to the PC and record**

```powershell
$room = "E:\roomrecon-scenes\vr_room\dns-30k"; New-Item -ItemType Directory -Force $room | Out-Null
scp gpu:roomrecon/scenes/vr_room/dns-30k/export/splat.ply gpu:roomrecon/scenes/vr_room/dns-30k/export/mesh.ply gpu:roomrecon/scenes/vr_room/dns-30k/export/scene.json "$room\"
Copy-Item "$room\scene.json" results\g2-vr_room-scene.json
```

Write `results/g2-export.md`: file sizes (from `ls -la` in `export.log`), splat count (header line `element vertex N` of `splat.ply`: `ssh gpu "head -c 400 roomrecon/scenes/vr_room/dns-30k/export/splat.ply | grep -a 'element vertex'"`), mesh voxel size used, `scale` and `median_camera_height_raw`.

- [ ] **Step 5: Commit**

```powershell
git add roomrecon/scene_cli.py tests/test_scene_cli.py scripts/export_room.sh results/
git commit -m "feat: export splat, cameras, mesh and scene.json for a trained room"
git push
```

---

### Task 10: G2 — Align, build collision, walk in the SuperSplat Viewer (PC)

**Files:**
- Create: `scripts/make_test_ply.py`, `scripts/collision_room.ps1`, `results/g2-walk-test.md`
- Modify: `results/g2-vr_room-scene.json` (adds `euler_xyz_deg`)

**Interfaces:**
- Consumes: `E:\roomrecon-scenes\vr_room\dns-30k\{splat.ply, scene.json}`.
- Produces: `aligned.ply` (Y-up, metric-ish), `room.voxel.json` + `room.voxel.bin`, `room.collision.glb` in the same folder; the viewer URL used for the demo.

- [ ] **Step 1: Write `scripts/make_test_ply.py` (convention check input)**

```python
#!/usr/bin/env python
"""Write a 3-splat PLY at (1,0,0), (0,1,0), (0,0,1) to learn splat-transform's Euler convention.
Usage: python scripts/make_test_ply.py out.ply"""
import sys
import numpy as np

pts = np.eye(3, dtype=np.float32)
props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
         "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
rows = np.zeros((3, len(props)), dtype=np.float32)
rows[:, :3] = pts; rows[:, 6] = 5.0; rows[:, 7:10] = -4.0; rows[:, 10] = 1.0
header = "ply\nformat binary_little_endian 1.0\nelement vertex 3\n" + \
         "".join(f"property float {p}\n" for p in props) + "end_header\n"
with open(sys.argv[1], "wb") as f:
    f.write(header.encode()); f.write(rows.tobytes())
```

- [ ] **Step 2: Learn the Euler convention (one-off) and add it to `scene.py` with a test**

On the PC:

```powershell
python scripts/make_test_ply.py $env:TEMP\t.ply
npx -y @playcanvas/splat-transform $env:TEMP\t.ply -r 30,40,50 $env:TEMP\t_rot.csv
Get-Content $env:TEMP\t_rot.csv | Select-Object -First 4
```

The CSV rows give where (1,0,0), (0,1,0), (0,0,1) went: those three points are the columns of splat-transform's rotation matrix M for `-r 30,40,50`. Compare M with the two candidates `Rz(50)@Ry(40)@Rx(30)` and `Rx(30)@Ry(40)@Rz(50)` (compute them with numpy). Exactly one should match to 1e-4. Then add to `roomrecon/scene.py`:

```python
def euler_xyz_deg(R: np.ndarray) -> list[float]:
    """Angles (x, y, z) in degrees such that splat-transform `-r x,y,z` applies R.
    Convention measured on 2026-10-.. (plan Task 10 Step 2): R = Rz(z) @ Ry(y) @ Rx(x)."""
    y = np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1)))
    x = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return [float(x), float(y), float(z)]
```

That body is for `R = Rz @ Ry @ Rx`. If the measurement showed `R = Rx @ Ry @ Rz` instead, use this body and change the docstring accordingly:

```python
    y = np.degrees(np.arcsin(np.clip(R[0, 2], -1, 1)))
    x = np.degrees(np.arctan2(-R[1, 2], R[2, 2]))
    z = np.degrees(np.arctan2(-R[0, 1], R[0, 0]))
    return [float(x), float(y), float(z)]
```

And this test to `tests/test_scene.py` (write `M` from the CSV output, rounded to 4 decimals, as the literal):

```python
def test_euler_matches_measured_splat_transform_convention():
    # Columns = where splat-transform -r 30,40,50 sent (1,0,0), (0,1,0), (0,0,1); measured on the PC.
    M = np.array(MEASURED_COLUMNS).T   # replace MEASURED_COLUMNS with the 3 CSV rows as a list
    assert np.allclose(scene.euler_xyz_deg(M), [30, 40, 50], atol=1e-3)
```

Run: `python -m pytest -v` → Expected: 13 passed. Commit: `git add scripts/make_test_ply.py roomrecon/scene.py tests/test_scene.py; git commit -m "feat: Euler angles in splat-transform's measured convention"; git push`.

- [ ] **Step 3: Write `scripts/collision_room.ps1`**

```powershell
# Align the splat to Y-up metric frame, keep the room cluster, build walk collision.
# Usage: .\scripts\collision_room.ps1 -Dir E:\roomrecon-scenes\vr_room\dns-30k
param([Parameter(Mandatory)][string]$Dir, [double]$VoxelSize = 0.05)
$ErrorActionPreference = "Stop"
$s = Get-Content "$Dir\scene.json" | ConvertFrom-Json
$py = "import json,sys,numpy as np; sys.path.insert(0,'.'); from roomrecon.scene import euler_xyz_deg; " +
      "s=json.load(open(r'$Dir\scene.json')); s['euler_xyz_deg']=euler_xyz_deg(np.array(s['rotation'])); " +
      "json.dump(s,open(r'$Dir\scene.json','w'),indent=2); print(','.join(f'{v:.6f}' for v in s['euler_xyz_deg']))"
$rot = python -c $py
$seed = ($s.spawn_position | ForEach-Object { "{0:F4}" -f $_ }) -join ","
Write-Host "rotate $rot  scale $($s.scale)  seed $seed"
npx -y @playcanvas/splat-transform "$Dir\splat.ply" -r $rot -s $s.scale "$Dir\aligned.ply"
npx -y @playcanvas/splat-transform "$Dir\aligned.ply" --filter-cluster --seed-pos $seed `
    --voxel-size $VoxelSize "$Dir\room.voxel.json" --voxel-external-fill --voxel-carve --collision-mesh
Get-ChildItem $Dir | Select-Object Name, Length
```

- [ ] **Step 4: Run it**

Run: `.\scripts\collision_room.ps1 -Dir E:\roomrecon-scenes\vr_room\dns-30k`
Expected: `aligned.ply`, `room.voxel.json`, `room.voxel.bin`, `room.collision.glb` listed. If splat-transform rejects a flag, run `npx -y @playcanvas/splat-transform --help`, use the printed spelling, and note the change in `results/g2-walk-test.md`.

- [ ] **Step 5: Run the SuperSplat Viewer locally and open the room**

```powershell
cd E:\roomrecon-scenes
git clone https://github.com/playcanvas/supersplat-viewer.git
cd supersplat-viewer; npm install
Start-Process powershell -ArgumentList "-NoExit","-Command","cd E:\roomrecon-scenes\supersplat-viewer; npm run develop"
Start-Process powershell -ArgumentList "-NoExit","-Command","npx -y http-server E:\roomrecon-scenes\vr_room\dns-30k -p 8081 --cors"
Start-Process "http://localhost:3000/?content=http://localhost:8081/aligned.ply&collision=http://localhost:8081/room.collision.glb"
```

Expected: the room renders; walk mode is available (the viewer's walk/first-person control). If `collision` with the `.glb` does not block walls, try `collision=http://localhost:8081/room.voxel.json` and record which one works.

- [ ] **Step 6: Walk test (the G2 exit criterion) — the user does this; Claude records it**

Checklist, each answered yes/no by the user in walk mode:
1. Starting position is inside the room, at roughly eye height.
2. Walking into each wall stops you.
3. Walking around the whole room, you never fall through the floor.
4. Splat surfaces and collision line up (you stop at the visible wall, not before or behind it).
5. No large floaters block the walkable area.
6. Movement feels smooth (note the FPS if the viewer shows it; otherwise "smooth"/"choppy").

Record the answers, the URL used, the voxel size, and any fixes in `results/g2-walk-test.md`. The user also saves a short screen recording of the walk (Win+Alt+R in Xbox Game Bar) to `E:\roomrecon-scenes\vr_room\dns-30k\walk.mp4` (not in git).
If check 1 or 4 fails, the frames disagree: re-check Step 2's convention and that `splat.ply` and `transforms_train.json` came from the same `CONFIG`, then tell the user before changing anything else.

- [ ] **Step 7: Commit**

```powershell
Copy-Item E:\roomrecon-scenes\vr_room\dns-30k\scene.json results\g2-vr_room-scene.json
git add scripts/collision_room.ps1 results/
git commit -m "exp: G2 vr_room walkable in the SuperSplat Viewer with collision"
git push
```

---

### Task 11: G3 — Measure quality and compare with plain splats

**Files:**
- Create: `results/g3-vr_room-results.md`, `results/g3-*.json`

**Interfaces:**
- Consumes: `CONFIG` of `dns-30k` (Task 7); reference mesh + ICP file from Task 5's `results/data-vr_room.md`; `scripts/train_room.sh`, `scripts/export_room.sh`.

- [ ] **Step 1: Rendering metrics (held-out views) for DN-Splatter**

```bash
cd ~/roomrecon/tools/dn-splatter
CONFIG=$(ls ~/roomrecon/scenes/vr_room/dns-30k/train/vr_room/dn-splatter/dns-30k/config.yml)
~/.pixi/bin/pixi run ns-eval --load-config "$CONFIG" --output-path ~/roomrecon/scenes/vr_room/dns-30k/eval/ns_eval.json
cat ~/roomrecon/scenes/vr_room/dns-30k/eval/ns_eval.json
```

Expected: JSON with `psnr`, `ssim`, `lpips` (and DN-Splatter depth metrics). Note which split it evaluated: check `results/data-vr_room.md` / the dataparser log line in `train.log` for whether eval used `test.txt` (within the long walk) or the short capture (separate walk). Write that down next to the numbers.

- [ ] **Step 2: Mesh accuracy against the laser-scanned reference**

```bash
cd ~/roomrecon/tools/dn-splatter
GT=<reference mesh path from results/data-vr_room.md>
~/.pixi/bin/pixi run python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py \
  --gt_mesh_path "$GT" --pred_mesh_path ~/roomrecon/scenes/vr_room/dns-30k/export/mesh.ply \
  --output ~/roomrecon/scenes/vr_room/dns-30k/eval/mesh_eval --device iphone 2>&1 | tee ~/roomrecon/scenes/vr_room/dns-30k/eval/mesh_eval.log
```

Expected: accuracy, completeness, F-score (and normal consistency) printed. If the script's `--help` shows other required arguments (for example an ICP/transform file), add the file from `results/data-vr_room.md`. If F-score is near 0, the predicted mesh is in a different frame from the reference: report it to the user rather than guessing a transform.

- [ ] **Step 3: Plain-splat baseline in the same environment (comparison 1)**

```bash
cd ~/roomrecon/repo
DATA=~/roomrecon/datasets/mushroom/room_datasets/vr_room
tmux new-session -d -s base "bash scripts/train_room.sh vr_room sf-30k splatfacto $DATA 30000"
```

When done: `bash scripts/export_room.sh vr_room sf-30k`, then repeat Steps 1–2 with `sf-30k` paths. If splatfacto rejects the `mushroom` dataparser, record the error in the results file and tell the user; do not switch dataparsers silently (the comparison must use the same data split).

- [ ] **Step 4: Collapse check on both curves**

Run `scripts/tb_metrics.py` for `sf-30k` as in Task 7 Step 2. For each method record best PSNR (step) and final PSNR (step).

- [ ] **Step 5: Write the results table**

Copy `ns_eval.json` and the mesh metrics output of both runs to `results/g3-dns-ns_eval.json`, `results/g3-sf-ns_eval.json`, `results/g3-dns-mesh.txt`, `results/g3-sf-mesh.txt`. Write `results/g3-vr_room-results.md`:

```markdown
# G3 — vr_room quality (GPU machine, A100-PCIE-40GB MIG, shared)

| Run | PSNR | SSIM | LPIPS | best PSNR (step) | mesh acc | mesh comp | F-score | train time | GPU used max (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| DN-Splatter dns-30k | from g3-dns-ns_eval.json | … | … | from eval curve | from g3-dns-mesh.txt | … | … | run.json duration_s | run.json |
| splatfacto sf-30k | from g3-sf-ns_eval.json | … | … | … | … | … | … | … | … |

Eval split: <test.txt within long walk | short capture>. Poses: MuSHRoom `mushroom --mode iphone` dataparser.
Walk test (DN-Splatter): see g2-walk-test.md.
Targets for later rooms (set from these numbers): …
```

Every cell must be filled from the named file; a cell whose source file doesn't exist stays empty with the reason written beside the table.

- [ ] **Step 6: Update `CLAUDE.md` status and commit**

In `CLAUDE.md` "Current status", replace the plan line with: `G0–G3 done (see results/). Next: G4 own room (Pro phone capture).` Then:

```powershell
git add results/ CLAUDE.md
git commit -m "exp: G3 vr_room quality — DN-Splatter vs splatfacto, mesh vs reference"
git push
```

---

## Self-review notes (done while writing)

- Spec coverage for G0–G3: repo (T1), env + pins (T2), §5.1 settings (T6 wrapper), free-memory check (T3), run record (T4), data + reference mesh (T5), resume test (T6), full run + collapse check (T7, T11), `scene.json` (T8, T9), collision + walk (T10), metrics + comparison 1 (T11). G4+ are later plans.
- Steps that depend on facts we could not verify offline are written as "check, then use the printed value, and tell the user if it differs": `cache-images` flag, normals output folder, `gs-mesh` voxel flag, `ns-export cameras` JSON key, splat-transform Euler convention and flag spellings, the viewer's collision format, the reference-mesh file names.
