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

_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _now() -> str:
    return datetime.now(timezone.utc).strftime(_FMT)


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
    t0 = datetime.strptime(rec["started_at"], _FMT)
    t1 = datetime.strptime(rec["ended_at"], _FMT)
    rec["duration_s"] = int((t1 - t0).total_seconds())
    rec["exit_code"] = int(exit_code)
    rec["status"] = "ok" if exit_code == 0 else "failed"
    if vram_csv and Path(vram_csv).exists():
        used = [int(line.split(",")[1])
                for line in Path(vram_csv).read_text().splitlines() if line]
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
