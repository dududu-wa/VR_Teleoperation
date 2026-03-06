#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-vr_teleop}"
ASSET_ROOT_DEFAULT="/home/ubuntu/lzxworkspace/codespace/unitree_mujoco"
export VR_TELEOP_ASSET_ROOT="${VR_TELEOP_ASSET_ROOT:-$ASSET_ROOT_DEFAULT}"

if [[ ! -d "$VR_TELEOP_ASSET_ROOT" ]]; then
  echo "[ERROR] VR_TELEOP_ASSET_ROOT does not exist: $VR_TELEOP_ASSET_ROOT"
  exit 1
fi

if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] Could not find conda.sh in ~/anaconda3 or ~/miniconda3"
  exit 1
fi

conda activate "$CONDA_ENV_NAME"

DEFAULT_ARGS=(
  --num-envs 256
  --device cuda:0
  --sim-backend isaacgym
)

USER_ARGS=("$@")

has_resume=0
for ((i=0; i<${#USER_ARGS[@]}; i++)); do
  if [[ "${USER_ARGS[$i]}" == "--resume" ]]; then
    has_resume=1
    break
  fi
done

if [[ $has_resume -eq 0 ]]; then
  latest_ckpt="$(ls -1t logs/g1_multigait/model_*.pt 2>/dev/null | head -n 1 || true)"
  if [[ -n "$latest_ckpt" ]]; then
    USER_ARGS+=(--resume "$latest_ckpt")
    echo "[INFO] Auto resume from: $latest_ckpt"
  else
    echo "[INFO] No checkpoint found, starting from scratch."
  fi
fi

echo "[INFO] Conda env: $CONDA_ENV_NAME"
echo "[INFO] Asset root: $VR_TELEOP_ASSET_ROOT"
echo "[INFO] Running: python scripts/train.py ${DEFAULT_ARGS[*]} ${USER_ARGS[*]}"

python -u - <<'PY' "${DEFAULT_ARGS[@]}" "${USER_ARGS[@]}"
import sys
import runpy
import isaacgym

sys.argv = ['scripts/train.py'] + sys.argv[1:]
runpy.run_path('scripts/train.py', run_name='__main__')
PY
