#!/usr/bin/env bash
# Sync skrl train/play scripts from Isaac Lab releases and reapply local tweaks.
set -euo pipefail

BASE_URL="https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/scripts/reinforcement_learning/skrl"
FILES=(train.py play.py)

fetch() {
  local url="$1"
  local dest="$2"
  curl -fsSL "${url}" -o "${dest}"
}

for name in "${FILES[@]}"; do
  url="${BASE_URL}/${name}"
  echo "Fetching ${url}"
  fetch "${url}" "${name}"
done

echo "Done."
