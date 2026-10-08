#!/bin/bash
# Train one room with the settings the spec requires (§5.1), recording run.json.
# Usage: scripts/train_room.sh <room> <run_id> <method> <data_dir> <max_iters> [extra ns-train args...]
#   method: dn-splatter | splatfacto
# Env: NEED_MIB (default 4000) minimum free GPU memory; RESUME_FROM=<.../nerfstudio_models>
set -uo pipefail
ROOM="$1"; RUN="$2"; METHOD="$3"; DATA="$4"; ITERS="$5"; shift 5
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"
PIXI="$HOME/.pixi/bin/pixi"
PY() { "$PIXI" run --manifest-path "$DNS/pixi.toml" python "$@"; }
OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"; mkdir -p "$OUT"
cd "$REPO"

PY -m roomrecon.gpu --need-mib "${NEED_MIB:-4000}" || exit 3
FREE=$(PY -m roomrecon.gpu | sed 's/free_mib=//')

MODEL_ARGS=()
if [ "$METHOD" = "dn-splatter" ]; then
  MODEL_ARGS=(--pipeline.model.use-depth-loss True --pipeline.model.depth-lambda 0.2
              --pipeline.model.use-depth-smooth-loss True --pipeline.model.use-normal-loss True
              --pipeline.model.normal-supervision mono)
fi
RESUME=()
if [ -n "${RESUME_FROM:-}" ]; then
  RESUME=(--load-dir "$RESUME_FROM")
  # nerfstudio counts --max-num-iterations from the loaded step: pass only what is left
  ITERS=$(PY -m roomrecon.resume "$RESUME_FROM" "$ITERS") || exit 4
fi

CMD=(ns-train "$METHOD" --output-dir "$OUT/train" --experiment-name "$ROOM" --timestamp "$RUN"
     --max-num-iterations "$ITERS" --steps-per-save 1000 --save-only-latest-checkpoint False
     --steps-per-eval-all-images 1000 --vis tensorboard --viewer.quit-on-train-completion True
     --pipeline.datamanager.cache-images cpu
     "${MODEL_ARGS[@]}" "${RESUME[@]}" "$@" mushroom --data "$DATA" --mode iphone)

PY -m roomrecon.runrecord start "$OUT/run.json" --run-id "$RUN" --room "$ROOM" --method "$METHOD" \
   --command "${CMD[*]}" --repo-commit "$(git rev-parse HEAD)" \
   --dn-splatter-commit "$(git -C "$DNS" rev-parse HEAD)" --free-mib "$FREE" \
   --machine "GPU machine A100-PCIE-40GB MIG 7g.40gb (shared)"

# the env's python directly, so $! is the sampler itself: killing a backgrounded `PY` (pixi run)
# left the python child sampling forever, and on Kaggle it held the notebook cell open (2026-10-08)
"$DNS/.pixi/envs/default/bin/python" -m roomrecon.gpu --sample "$OUT/vram.csv" --interval 10 \
  </dev/null >>"$OUT/sampler.log" 2>&1 & SAMPLER=$!
( cd "$DNS" && "$PIXI" run "${CMD[@]}" ) 2>&1 | tee -a "$OUT/train.log"
CODE=${PIPESTATUS[0]}
kill "$SAMPLER" 2>/dev/null
PY -m roomrecon.runrecord finish "$OUT/run.json" --exit-code "$CODE" --vram-csv "$OUT/vram.csv"
echo "train_room: exit $CODE -> $OUT"
exit "$CODE"
