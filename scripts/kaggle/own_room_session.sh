#!/bin/bash
# Runs INSIDE a Kaggle GPU session (started by own_room_kaggle.sh): builds the pinned DN-Splatter
# env, then normals, training, final eval and export for one own room, and copies the results to
# /kaggle/working/out (the session output). The steps themselves are in scripts/own_room.sh.
# Checkpoints other than the last stay in the session.
# Usage: bash scripts/kaggle/own_room_session.sh <room> <run_id> <iters>
# Env: as scripts/own_room.sh (STOP_SPLIT_AT, ...)
set -uo pipefail
ROOM="$1"; RUN="$2"; ITERS="$3"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTW=/kaggle/working/out; mkdir -p "$OUTW"
exec > >(tee -a "$OUTW/session.log") 2>&1
nvidia-smi
# the data dataset is mounted read-only somewhere under /kaggle/input: find it by its transforms
echo "== /kaggle/input (top of the tree):"; find /kaggle/input -maxdepth 4 | grep -v -E '/(images|depth)/' | head -60
T="$(find /kaggle/input -name transforms.json -not -path '*/repo/*' | head -1)"
[ -n "$T" ] || { echo "no transforms.json under /kaggle/input (see the tree above)"; exit 2; }
DATA_IN="$(dirname "$T")"
DATA="$HOME/roomrecon/datasets/own/$ROOM/processed"
mkdir -p "$(dirname "$DATA")" && cp -r "$DATA_IN" "$DATA"
echo "data: $DATA_IN -> $DATA ($(ls "$DATA/images" | wc -l) images)"

set -e
bash "$REPO/scripts/setup_gpu.sh"
set +e

GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
MACHINE="Kaggle $GPU" bash "$REPO/scripts/own_room.sh" "$ROOM" "$RUN" "$DATA" "$ITERS"
TRAIN_EXIT=$?
OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"
CONFIG="$(ls "$OUT"/train/"$ROOM"/*/"$RUN"/config.yml | head -1)"

# results only: the export, the run records, the last checkpoint (for a later re-export)
cp -r "$OUT/export" "$OUT/eval" "$OUT/renders" "$OUTW/" 2>/dev/null
cp "$OUT"/run.json "$OUT"/attempts.log "$OUT"/eval_curve.csv "$OUT"/vram.csv "$CONFIG" "$OUTW/" 2>/dev/null
LAST="$(ls "$(dirname "$CONFIG")"/nerfstudio_models/step-*.ckpt 2>/dev/null | sort | tail -1)"
[ -n "$LAST" ] && cp "$LAST" "$OUTW/"
ls -la "$OUTW"
# anything still holding the output pipe keeps the notebook cell open until Kaggle's 12 h limit
pkill -f 'roomrecon.gpu --sample' 2>/dev/null
exit "$TRAIN_EXIT"
