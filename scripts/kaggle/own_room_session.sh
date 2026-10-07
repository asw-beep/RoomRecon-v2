#!/bin/bash
# Runs INSIDE a Kaggle GPU session (started by own_room_kaggle.sh): builds the pinned DN-Splatter
# env, then normals, training, final eval and export for one own room, and copies the results to
# /kaggle/working/out (the session output). Checkpoints other than the last stay in the session.
# Usage: bash scripts/kaggle/own_room_session.sh <room> <run_id> <iters>
# Env: STOP_SPLIT_AT (default iters/2)
set -uo pipefail
ROOM="$1"; RUN="$2"; ITERS="$3"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
OUTW=/kaggle/working/out; mkdir -p "$OUTW"
exec > >(tee -a "$OUTW/session.log") 2>&1
nvidia-smi
# the data dataset is mounted read-only somewhere under /kaggle/input: find it by its transforms
DATA_IN="$(dirname "$(find /kaggle/input -name transforms.json -path "*processed*" | head -1)")"
[ -d "$DATA_IN" ] || { echo "no processed data under /kaggle/input"; exit 2; }
DATA="$HOME/roomrecon/datasets/own/$ROOM/processed"
mkdir -p "$(dirname "$DATA")" && cp -r "$DATA_IN" "$DATA"
echo "data: $DATA_IN -> $DATA ($(ls "$DATA/images" | wc -l) images)"

set -e
bash "$REPO/scripts/setup_gpu.sh"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
( cd "$DNS" && NORMALS_BATCH=4 "$PIXI" run python "$REPO/scripts/normals.py" --data-dir "$DATA" --resolution low )
echo "normals: $(ls "$DATA/normals_from_pretrain" | wc -l) files"
set +e

GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
DATASET=own MACHINE="Kaggle $GPU" STOP_SPLIT_AT="${STOP_SPLIT_AT:-$((ITERS / 2))}" \
  bash "$REPO/scripts/train_room_lite.sh" "$ROOM" "$RUN" "$DATA" "$ITERS"
TRAIN_EXIT=$?
echo "TRAIN_EXIT=$TRAIN_EXIT"

OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"
CONFIG="$(ls "$OUT"/train/"$ROOM"/*/"$RUN"/config.yml | head -1)"
RUNENV() { "$PIXI" run --manifest-path "$DNS/pixi.toml" env LD_PRELOAD="$DNS/.pixi/envs/default/lib/libstdc++.so.6" "$@"; }
if [ "$TRAIN_EXIT" -eq 0 ]; then
  mkdir -p "$OUT/eval"
  ( cd "$DNS" && RUNENV ns-eval --load-config "$CONFIG" --output-path "$OUT/eval/ns_eval.json" )
  echo "EVAL_EXIT=$?"
  bash "$REPO/scripts/export_room.sh" "$ROOM" "$RUN"
  echo "EXPORT_EXIT=$?"
  ( cd "$DNS" && RUNENV ns-render dataset --load-config "$CONFIG" --split test \
      --rendered-output-names rgb gt-rgb --output-path "$OUT/renders" )
fi
( cd "$DNS" && "$PIXI" run python "$REPO/scripts/tb_metrics.py" "$OUT/train" "$OUT/eval_curve.csv" )

# results only: the export, the run records, the last checkpoint (for a later re-export)
cp -r "$OUT/export" "$OUT/eval" "$OUT/renders" "$OUTW/" 2>/dev/null
cp "$OUT"/run.json "$OUT"/attempts.log "$OUT"/eval_curve.csv "$OUT"/vram.csv "$CONFIG" "$OUTW/" 2>/dev/null
LAST="$(ls "$(dirname "$CONFIG")"/nerfstudio_models/step-*.ckpt 2>/dev/null | sort | tail -1)"
[ -n "$LAST" ] && cp "$LAST" "$OUTW/"
ls -la "$OUTW"
exit "$TRAIN_EXIT"
