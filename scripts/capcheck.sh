#!/bin/bash
# Capture check on the GPU machine (CPU only, in the DN-Splatter pixi env; nothing to install).
# Usage: scripts/capcheck.sh CAPTURE --out DIR [--exclude FILE] [--tau-tex X] [--name NAME]
#   CAPTURE = an `ns-process-data polycam` output folder (with keyframes/) or a MuSHRoom capture.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
cd "$REPO"
exec "$PIXI" run --manifest-path "$DNS/pixi.toml" python -m roomrecon.capcheck "$@"
