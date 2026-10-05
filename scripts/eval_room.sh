#!/bin/bash
# Evaluate a trained and exported MuSHRoom room: ns-eval on the final checkpoint (DN-Splatter's
# pipeline reports the "within" and "with" test sets separately) and the exported mesh against
# the laser-scanned reference (DN-Splatter's MuSHRoom mesh evaluation, iPhone ICP alignment).
# Usage: scripts/eval_room.sh <room> <run_id> <room_dataset_dir>   (after export_room.sh)
# Writes ~/roomrecon/scenes/<room>/<run_id>/eval/{ns_eval.json, mesh_metrics.json, mesh_cull.ply}
set -euo pipefail
ROOM="$1"; RUN="$2"; DATA="$3"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"; EV="$OUT/eval"; mkdir -p "$EV"
CONFIG=$(ls "$OUT"/train/"$ROOM"/*/"$RUN"/config.yml | head -1)
# same libstdc++ preload as export_room.sh (open3d imported before torch)
RUNENV() { "$PIXI" run --manifest-path "$DNS/pixi.toml" env LD_PRELOAD="$DNS/.pixi/envs/default/lib/libstdc++.so.6" "$@"; }
cd "$DNS"
RUNENV ns-eval --load-config "$CONFIG" --output-path "$EV/ns_eval.json"
# the mesh evaluation writes its outputs next to the mesh it is given
cp "$OUT/export/mesh.ply" "$EV/mesh.ply"
RUNENV python dn_splatter/eval/eval_mesh_mushroom_vis_cull.py --gt-mesh-path "$DATA" \
    --pred-mesh-path "$EV/mesh.ply" --device iphone
ls -la "$EV"
