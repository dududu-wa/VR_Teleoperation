#!/bin/bash
# Run 3-way comparison with DTW/key-body metrics
# Checkpoints:
#   sw002_best: logs/r2_amp/Jun04_14-23-35_sw002/model_best_task.pt
#   style0_best: logs/r2_amp/Jun04_14-23-39_style0/model_best_task.pt
#   rint_v2: logs/r2_interrupt/Apr25_23-26-30_r2v2_ppo_version6/model_30000.pt

set -e
cd /home/ubuntu/lzxworkspace/codespace/VR_Teleoperation

SCRIPT=legged_gym/scripts/evaluate.py
N=20
SEED=1
SECS=10

CONDA=/home/ubuntu/anaconda3/bin/conda

echo "=== [2/3] style0_best ==="
$CONDA run -n hugwbc python $SCRIPT \
  --task r2amp \
  --load_run Jun04_14-23-39_style0 \
  --checkpoint -2 \
  --num_episodes $N \
  --seed $SEED \
  --episode_seconds $SECS \
  --headless \
  --output_dir outputs/eval3/style0_best

echo "=== [3/3] rint_v2 (r2int baseline) ==="
$CONDA run -n hugwbc python $SCRIPT \
  --task r2int \
  --load_run Apr25_23-26-30_r2v2_ppo_version6 \
  --checkpoint 30000 \
  --num_episodes $N \
  --seed $SEED \
  --episode_seconds $SECS \
  --headless \
  --output_dir outputs/eval3/rint_v2

echo "=== All done. Consolidating... ==="
$CONDA run -n hugwbc python - <<'PYEOF'
import json, csv, pathlib

runs = ["sw002_best", "style0_best", "rint_v2"]
base = pathlib.Path("outputs/eval3")

all_rows = []
for run in runs:
    mfile = base / run / "metrics.json"
    if not mfile.exists():
        print(f"MISSING: {mfile}")
        continue
    rows = json.loads(mfile.read_text())
    for r in rows:
        r["_run"] = run
    all_rows.extend(rows)

out = base / "all_metrics.json"
out.write_text(json.dumps(all_rows, indent=2))
print(f"Wrote {len(all_rows)} rows to {out}")

# Print summary table
COLS = ["_run","preset_name","task_return_mean","fall_rate","lin_vel_rmse",
        "joint_pose_error_dtw_m","key_body_error_dtw_m","amp_style_reward_raw_mean","disc_gap_mean"]
header = "\t".join(COLS)
print("\n" + header)
for r in all_rows:
    print("\t".join(str(r.get(c,"")) for c in COLS))
PYEOF
