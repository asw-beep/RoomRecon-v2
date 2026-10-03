#!/bin/bash
# MuSHRoom room → ~/roomrecon/datasets/mushroom (iPhone sequence + reference mesh) + normals.
# Usage: bash scripts/fetch_mushroom.sh vr_room
# The iPhone tarball may already be uploaded (scp from the PC); otherwise it is downloaded.
set -euo pipefail
ROOM="${1:?room name, e.g. vr_room}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
D="$HOME/roomrecon/datasets/mushroom"
DNS="$HOME/roomrecon/tools/dn-splatter"
PIXI="$HOME/.pixi/bin/pixi"
mkdir -p "$D"; cd "$D"
[ -f "${ROOM}_iphone.tar.gz" ] || wget -q "https://zenodo.org/records/10230733/files/${ROOM}_iphone.tar.gz"
[ -f "${ROOM}_mesh_pd.tar.gz" ] || wget -q "https://zenodo.org/records/10222321/files/${ROOM}_mesh_pd.tar.gz"
sha256sum "${ROOM}_iphone.tar.gz" "${ROOM}_mesh_pd.tar.gz" | tee "${ROOM}.sha256"
[ -d "room_datasets/${ROOM}/iphone/long_capture" ] || tar -xzf "${ROOM}_iphone.tar.gz"
tar -tzf "${ROOM}_mesh_pd.tar.gz" | head -20
tar -xzf "${ROOM}_mesh_pd.tar.gz"
cd "$DNS"
for seq in long_capture short_capture; do
  P="$D/room_datasets/${ROOM}/iphone/${seq}"
  if [ ! -d "$P/normals_from_pretrain" ]; then
    # batch 4: DN-Splatter's default 15 runs out of the ~6 GB free on the shared GPU
    NORMALS_BATCH=4 "$PIXI" run python "$REPO/scripts/normals.py" --data-dir "$P" --resolution low
  fi
  echo "$seq images=$(ls "$P/images" | wc -l) normals=$(ls "$P/normals_from_pretrain" | wc -l)"
done
