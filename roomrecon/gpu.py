"""Free memory on the shared MIG A100 (or any plain GPU, e.g. a Kaggle T4).

Inside a MIG instance `nvidia-smi --query-gpu=memory.*` returns "Insufficient Permissions",
so we read the MIG device table of plain `nvidia-smi` instead; a GPU without MIG has no such
table and answers the query. Another tenant uses the A100, so free memory must be checked at
job start, not assumed.
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


def parse_query_memory(text: str) -> tuple[int, int]:
    """(used_mib, total_mib) of the first GPU in `--query-gpu=memory.used,memory.total
    --format=csv,noheader,nounits` output."""
    used, total = (int(v) for v in text.splitlines()[0].split(","))
    return used, total


def _smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


def used_total() -> tuple[int, int]:
    try:
        return parse_mig_memory(_smi())
    except ValueError:
        return parse_query_memory(
            _smi("--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"))


def free_mib() -> int:
    used, total = used_total()
    return total - used


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--need-mib", type=int, help="exit 1 if less than this is free")
    ap.add_argument("--sample", help="append used/total MiB rows to this CSV forever")
    ap.add_argument("--interval", type=float, default=10.0)
    a = ap.parse_args(argv)
    if a.sample:
        while True:
            used, total = used_total()
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
