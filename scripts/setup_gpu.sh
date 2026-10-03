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
# Upstream pixi.toml installs nerfstudio from git commit a64026f, but dn-splatter's pyproject
# pins nerfstudio==1.1.3, so pixi cannot solve. Use the PyPI release the pyproject asks for.
sed -i 's#^nerfstudio = { git = .*#nerfstudio = "==1.1.3"#' pixi.toml
grep -n '^nerfstudio' pixi.toml
# pymcubes==0.1.2 (a nerfstudio dependency) has no wheel and fails to compile against
# NumPy 2 (PyArray_DOUBLE removed). torch 2.2 is built for NumPy 1.x anyway: pin numpy<2 in
# the env and build pymcubes against it instead of an isolated build env.
grep -q '^numpy = ' pixi.toml || sed -i '/^\[dependencies\]/a numpy = ">=1.26,<2"\ncython = "<3.1"\nsetuptools = "*"' pixi.toml
grep -q '^\[pypi-options\]' pixi.toml || printf '\n[pypi-options]\nno-build-isolation = ["pymcubes"]\n' >> pixi.toml
sed -n '/^\[dependencies\]/,/^\[pypi-dependencies\]/p;/^\[pypi-options\]/,$p' pixi.toml
# DN-Splatter bug: with periodic eval on (get_std=False), the MuSHRoom "within" average reads
# each per-image dict with the prefixed key -> KeyError 'within_rgb_psnr' at the first
# eval-all-images step. The "with" branch already uses the bare key; make "within" match.
sed -i 's/^\(\s*\)metrics_dict\["within_" + key\]$/\1metrics_dict[key]/' dn_splatter/dn_pipeline.py
grep -c 'metrics_dict\["within_" + key\]$' dn_splatter/dn_pipeline.py || true
"$PIXI" install
"$PIXI" run download-omnidata
"$PIXI" run python -c "import torch, nerfstudio, gsplat; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('gsplat', gsplat.__version__)"
"$PIXI" run pip install -e "$REPO" pytest
echo "setup_gpu: OK"
