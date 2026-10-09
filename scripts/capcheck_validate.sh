#!/bin/bash
# Validate the capture check against trained splats' held-out renders (GPU machine, CPU only).
# Usage: scripts/capcheck_validate.sh --calib NAME CAPTURE NPZ RENDERS_RGB RENDERS_GT \
#                                     --test  NAME CAPTURE NPZ RENDERS_RGB RENDERS_GT --out DIR
#   NPZ = points.npz of a capcheck run on the TRAINING frames only (--exclude the test frames).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # see capcheck.sh
cd "$REPO"
exec "$PIXI" run --manifest-path "$DNS/pixi.toml" python -m roomrecon.capcheck.validate "$@"
