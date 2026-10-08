#!/bin/bash
# One own room end to end on this machine (GPU machine or a Kaggle session): normals if missing,
# lite training, final eval, export (splat, cameras, mesh, scene.json) and test-view renders.
# Usage: scripts/own_room.sh <room> <run_id> <processed_dir> <iters>
# Env: STOP_SPLIT_AT (default iters/3, the 10k-of-30k ratio of the vr_room runs), and the rest of
#      train_room_lite.sh's (MACHINE, NEED_MIB, ...); NORMALS_BATCH (4)
# Exits with the training exit code; eval/export results are in the log (EVAL_EXIT, EXPORT_EXIT).
set -uo pipefail
ROOM="$1"; RUN="$2"; DATA="$3"; ITERS="$4"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"

if [ -z "$(ls "$DATA/normals_from_pretrain" 2>/dev/null)" ]; then
  ( cd "$DNS" && NORMALS_BATCH="${NORMALS_BATCH:-4}" "$PIXI" run python "$REPO/scripts/normals.py" \
      --data-dir "$DATA" --resolution low ) || { echo "normals failed" >&2; exit 5; }
fi
echo "normals: $(ls "$DATA/normals_from_pretrain" | wc -l) files"

DATASET=own STOP_SPLIT_AT="${STOP_SPLIT_AT:-$((ITERS / 3))}" \
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
exit "$TRAIN_EXIT"
