#!/bin/bash
# GPU machine environment for RoomRecon v2: pixi + DN-Splatter (no sudo needed).
# Idempotent: re-running skips what exists. Usage: bash scripts/setup_gpu.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$HOME/roomrecon/tools"
DNS="$TOOLS/dn-splatter"
PIXI="$HOME/.pixi/bin/pixi"
mkdir -p "$TOOLS" "$HOME/roomrecon/datasets" "$HOME/roomrecon/scenes"

[ -x "$PIXI" ] || curl -fsSL https://pixi.sh/install.sh | bash
"$PIXI" --version

if [ ! -d "$DNS/.git" ]; then
  git clone https://github.com/maturk/dn-splatter "$DNS" </dev/null
fi
PIN="$(cat "$REPO/env/dn-splatter.commit" 2>/dev/null || true)"
if [ -n "$PIN" ]; then git -C "$DNS" checkout -q "$PIN"; fi
git -C "$DNS" rev-parse HEAD

cd "$DNS"
"$PIXI" install
"$PIXI" run download-omnidata
"$PIXI" run python -c "import torch, nerfstudio, gsplat; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('gsplat', gsplat.__version__)"
"$PIXI" run pip install -e "$REPO" pytest
echo "setup_gpu: OK"
