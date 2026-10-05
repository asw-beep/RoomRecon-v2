#!/bin/bash
# Low-memory variant of train_room.sh for the shared GPU (~7 GB free), with auto-resume.
# Same spec §5.1 settings (ckpt every 1k, keep all, eval every 1k, CPU image cache), plus:
#   - fewer seed points: DN-Splatter's MuSHRoom parser seeds 1,000,000 Gaussians from the
#     iPhone point cloud by default, which is what ran out of memory (Task 7, 2026-10-04)
#   - densification stops earlier
#   - if training dies (co-tenant squeezes us -> OOM), wait for memory and resume from the
#     newest checkpoint; give up when an attempt makes no checkpoint progress
# Use train_room.sh instead when the GPU has room (>= ~15 GB free).
# Usage: scripts/train_room_lite.sh <room> <run_id> <data_dir> <max_iters> [extra ns-train args...]
# Env: NUM_INIT_POINTS (300000)  STOP_SPLIT_AT (10000)  NEED_MIB (4000)  MAX_ATTEMPTS (8)
#      PROFILE: dn (default) = depth + normal priors | plain = no priors, splatfacto's settings
#      (the plain-3DGS baseline, evaluated by the same DN-Splatter pipeline on the same splits;
#      plain splatfacto would average MuSHRoom's two test sets into one number)
set -uo pipefail
ROOM="$1"; RUN="$2"; DATA="$3"; ITERS="$4"; shift 4
METHOD=dn-splatter
PROFILE="${PROFILE:-dn}"
case "$PROFILE" in
  dn)    MODEL_ARGS=(--pipeline.model.use-depth-loss True --pipeline.model.depth-lambda 0.2
                     --pipeline.model.use-depth-smooth-loss True --pipeline.model.use-normal-loss True
                     --pipeline.model.normal-supervision mono) ;;
  # every DN-Splatter prior off; the remaining defaults set to splatfacto's (nerfstudio 1.1.3),
  # except num-downscales stays 0: DN-Splatter's normals-from-depth uses the full-size camera
  # and crashes on splatfacto's downscaled warm-up (dn_model.py:589, 2026-10-05)
  plain) MODEL_ARGS=(--pipeline.model.use-depth-loss False --pipeline.model.use-depth-smooth-loss False
                     --pipeline.model.use-normal-loss False --pipeline.model.use-normal-tv-loss False
                     --pipeline.model.two-d-gaussians False --pipeline.model.max-gauss-ratio 10.0) ;;
  *) echo "PROFILE must be dn or plain" >&2; exit 2 ;;
esac
NUM_INIT_POINTS="${NUM_INIT_POINTS:-300000}"
STOP_SPLIT_AT="${STOP_SPLIT_AT:-10000}"
NEED_MIB="${NEED_MIB:-4000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-8}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"
PIXI="$HOME/.pixi/bin/pixi"
PY() { "$PIXI" run --manifest-path "$DNS/pixi.toml" python "$@"; }
OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"; mkdir -p "$OUT"
MODELS="$OUT/train/$ROOM/$METHOD/$RUN/nerfstudio_models"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"

latest_step() {   # newest checkpoint step, 0 if none
  ls "$MODELS" 2>/dev/null | sed -n 's/^step-0*\([0-9][0-9]*\)\.ckpt$/\1/p' | sort -n | tail -1 | grep . || echo 0
}
wait_for_memory() {   # up to 2 h
  for _ in $(seq 120); do PY -m roomrecon.gpu --need-mib "$NEED_MIB" && return 0; sleep 60; done
  return 1
}

PY -m roomrecon.gpu --need-mib "$NEED_MIB" || exit 3
FREE=$(PY -m roomrecon.gpu | sed 's/free_mib=//')

BASE=(ns-train "$METHOD" --output-dir "$OUT/train" --experiment-name "$ROOM" --timestamp "$RUN"
      --steps-per-save 1000 --save-only-latest-checkpoint False
      --steps-per-eval-all-images 1000 --vis tensorboard --viewer.quit-on-train-completion True
      --pipeline.datamanager.cache-images cpu "${MODEL_ARGS[@]}"
      --pipeline.model.stop-split-at "$STOP_SPLIT_AT" "$@")
DP=(mushroom --data "$DATA" --mode iphone --num-init-points "$NUM_INIT_POINTS")

PY -m roomrecon.runrecord start "$OUT/run.json" --run-id "$RUN" --room "$ROOM" --method "$METHOD ($PROFILE)" \
   --command "${BASE[*]} --max-num-iterations $ITERS ${DP[*]} (lite, auto-resume)" \
   --repo-commit "$(git rev-parse HEAD)" \
   --dn-splatter-commit "$(git -C "$DNS" rev-parse HEAD)" --free-mib "$FREE" \
   --machine "GPU machine A100-PCIE-40GB MIG 7g.40gb (shared)"

PY -m roomrecon.gpu --sample "$OUT/vram.csv" --interval 10 & SAMPLER=$!
CODE=1
for ATTEMPT in $(seq "$MAX_ATTEMPTS"); do
  BEFORE=$(latest_step)
  if [ "$BEFORE" -ge "$ITERS" ]; then CODE=0; break; fi
  if [ "$BEFORE" -gt 0 ]; then
    # nerfstudio counts --max-num-iterations from the loaded step: pass only what is left
    LEFT=$(PY -m roomrecon.resume "$MODELS" "$ITERS") || { CODE=4; break; }
    STEPS=(--max-num-iterations "$LEFT" --load-dir "$MODELS")
  else
    STEPS=(--max-num-iterations "$ITERS")
  fi
  echo "train_room_lite: attempt $ATTEMPT from step $BEFORE ($(date -u +%FT%TZ))" | tee -a "$OUT/attempts.log"
  ( cd "$DNS" && "$PIXI" run "${BASE[@]}" "${STEPS[@]}" "${DP[@]}" ) 2>&1 | tee -a "$OUT/train.log"
  CODE=${PIPESTATUS[0]}
  AFTER=$(latest_step)
  echo "train_room_lite: attempt $ATTEMPT exit $CODE, checkpoints $BEFORE -> $AFTER" | tee -a "$OUT/attempts.log"
  [ "$CODE" -eq 0 ] && break
  if [ "$AFTER" -le "$BEFORE" ] && [ "$ATTEMPT" -gt 1 ]; then
    echo "train_room_lite: no progress since last attempt, giving up" | tee -a "$OUT/attempts.log"; break
  fi
  wait_for_memory || { echo "train_room_lite: memory never freed up" | tee -a "$OUT/attempts.log"; break; }
done
kill "$SAMPLER" 2>/dev/null
PY -m roomrecon.runrecord finish "$OUT/run.json" --exit-code "$CODE" --vram-csv "$OUT/vram.csv"
echo "train_room_lite: exit $CODE -> $OUT"
exit "$CODE"
