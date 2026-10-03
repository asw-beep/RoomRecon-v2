"""Steps left after a resume.

nerfstudio trains `--max-num-iterations` steps *after* the loaded checkpoint, so a resumed
run must be given the remaining count, or it overshoots (seen in the G1 smoke test).
Usage: python -m roomrecon.resume <nerfstudio_models_dir> <max_iters>   -> prints the count
"""
import re
import sys
from pathlib import Path

_STEP = re.compile(r"step-(\d+)\.ckpt$")


def remaining_steps(models_dir, max_iters: int) -> int:
    steps = [int(m.group(1)) for p in Path(models_dir).iterdir() if (m := _STEP.search(p.name))]
    if not steps:
        raise ValueError(f"no checkpoint in {models_dir}")
    left = int(max_iters) - max(steps)
    if left <= 0:
        raise ValueError(f"already reached step {max(steps)} of {max_iters}")
    return left


if __name__ == "__main__":
    print(remaining_steps(sys.argv[1], int(sys.argv[2])))
