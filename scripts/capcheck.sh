#!/bin/bash
# Capture check on the GPU machine (CPU only, in the DN-Splatter pixi env; nothing to install).
# Usage: scripts/capcheck.sh CAPTURE --out DIR [--exclude FILE] [--tau-tex X] [--name NAME]
#   CAPTURE = an `ns-process-data polycam` output folder (with keyframes/) or a MuSHRoom capture.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
# one BLAS thread: on the 96-core GPU machine, numpy's batched 3x3 eigh/matmul ran 80x slower
# with the default thread count (16 s vs 0.2 s for 18k normals, 2026-10-09)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
exec "$PIXI" run --manifest-path "$DNS/pixi.toml" python -m roomrecon.capcheck "$@"
