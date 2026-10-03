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
