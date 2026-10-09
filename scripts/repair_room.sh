#!/bin/bash
# Repair a phone capture's broken tracking and prepare the three training variants
# (GPU machine CPUs, DN-Splatter pixi env; nothing to install).
# Usage: scripts/repair_room.sh CAPTURE OUT [--min-frames N]
#   CAPTURE = an `ns-process-data polycam` folder with keyframes/; OUT gets raw/ drop/ repaired/
#   (each a training folder for scripts/own_room.sh) and repair.json.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1      # see capcheck.sh
CAP="$(cd "$1" && pwd)"; OUT="$2"; shift 2
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
cd "$REPO"
"$PIXI" run --manifest-path "$DNS/pixi.toml" python -m roomrecon.repair "$CAP" --out "$OUT" "$@" </dev/null
# seed points per variant from its own poses (DN-Splatter's tool, as for G4)
cd "$DNS"
for v in raw drop repaired; do
  "$PIXI" run python dn_splatter/data/mushroom_utils/pointcloud_utils.py --data-path "$OUT/$v" --num-points 300000 </dev/null
done
ls -la "$OUT"/*/iphone_pointcloud.ply
