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
# Install exactly the environment the GPU machine was built and tested with (env/pixi.lock is a
# copy of its lock), before any `pixi run` below could solve a fresh, unpinned one.
LOCKED=""
if [ -f "$REPO/env/pixi.lock" ]; then cp "$REPO/env/pixi.lock" pixi.lock; LOCKED=1; fi
"$PIXI" install ${LOCKED:+--frozen}
# Resume bug: DN-Splatter's `normals` optimizer saves an empty per-parameter Adam state, and
# torch's Adam.load_state_dict raises KeyError 'step' on it. An empty state means "no moments
# yet", so dropping it before loading is exact. Patched in nerfstudio's optimizer loader.
NSOPT="$("$PIXI" run python -c 'import nerfstudio.engine.optimizers as m; print(m.__file__)' 2>/dev/null | tail -1)"
grep -q 'ROOMRECON: drop empty' "$NSOPT" || sed -i 's/^\(\s*\)self.optimizers\[k\].load_state_dict(v)$/\1v["state"] = {i: s for i, s in v["state"].items() if s}  # ROOMRECON: drop empty Adam states\n\1self.optimizers[k].load_state_dict(v)/' "$NSOPT"
grep -n -A1 'ROOMRECON: drop empty' "$NSOPT"
# Export bug: `ns-export cameras` asserts camera.metadata, which DN-Splatter's MuSHRoom parser
# never sets. DN-Splatter trains with the camera optimizer off, so the optimized pose IS the
# dataset pose: use it whenever there is no metadata. Patched in nerfstudio's exporter utils.
# (located without importing it: the import itself hits the libstdc++ problem, see export_room.sh)
NSEXP="$("$PIXI" run python -c 'import importlib.util as u; print(u.find_spec("nerfstudio").submodule_search_locations[0])' 2>/dev/null | tail -1)/exporter/exporter_utils.py"
grep -q 'ROOMRECON: no metadata' "$NSEXP" || sed -i 's/^\(\s*\)if camera_optimizer is None:$/\1if camera_optimizer is None or cameras.metadata is None:  # ROOMRECON: no metadata/' "$NSEXP"
grep -n 'ROOMRECON: no metadata' "$NSEXP"
"$PIXI" run download-omnidata
"$PIXI" run python -c "import torch, nerfstudio, gsplat; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('gsplat', gsplat.__version__)"
"$PIXI" run pip install -e "$REPO" pytest
echo "setup_gpu: OK"
