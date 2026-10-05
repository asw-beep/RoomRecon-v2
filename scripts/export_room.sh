#!/bin/bash
# Export a trained room: splat, cameras, mesh, scene.json.
# Usage: scripts/export_room.sh <room> <run_id> [voxel_size]
set -euo pipefail
ROOM="$1"; RUN="$2"; VOX="${3:-0.02}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DNS="$HOME/roomrecon/tools/dn-splatter"; PIXI="$HOME/.pixi/bin/pixi"
OUT="$HOME/roomrecon/scenes/$ROOM/$RUN"; EXP="$OUT/export"; mkdir -p "$EXP"
CONFIG=$(ls "$OUT"/train/"$ROOM"/*/"$RUN"/config.yml | head -1)
cd "$DNS"
"$PIXI" run ns-export gaussian-splat --load-config "$CONFIG" --output-dir "$EXP"
"$PIXI" run ns-export cameras --load-config "$CONFIG" --output-dir "$EXP"
# export_mesh.py imports open3d before torch, which loads the system libstdc++ (too old for
# nerfstudio's fpsample: GLIBCXX_3.4.29 not found); preload the env's own copy
"$PIXI" run env LD_PRELOAD="$DNS/.pixi/envs/default/lib/libstdc++.so.6" gs-mesh o3dtsdf --load-config "$CONFIG" --output-dir "$EXP/mesh_o3dtsdf" --voxel-size "$VOX"
ls -la "$EXP" "$EXP/mesh_o3dtsdf"
cp "$(ls "$EXP"/mesh_o3dtsdf/*.ply | head -1)" "$EXP/mesh.ply"
cd "$REPO"
"$PIXI" run --manifest-path "$DNS/pixi.toml" python -m roomrecon.scene_cli "$EXP" --room "$ROOM" --run-id "$RUN"
