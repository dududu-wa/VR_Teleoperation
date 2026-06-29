param(
    # Keep this default aligned with the local logs/r2_amp subtree so the helper can run without path overrides.
    [string]$LoadRun = "Jun19/Jun19_16-09-11_scratch_command_hold",
    [string]$CfgOverrideJson = "configs/ablation/scratch_command_hold.json",
    [string]$Checkpoint = "8000",
    [int]$NumEpisodes = 64,
    [int]$NumEnvs = 64,
    [string]$OutputRoot = "outputs/eval/run_disturb_sweep_command_hold_8000"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$repoWsl = "/mnt/e/codebase/VR_Teleoperation"
$python = "/opt/miniconda3/envs/r2gym/bin/python"
$envPrefix = "PATH=/opt/miniconda3/envs/r2gym/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin PYTHONPATH=${repoWsl}:${repoWsl}/rsl_rl LD_LIBRARY_PATH=/opt/miniconda3/envs/r2gym/lib:/mnt/e/wsl/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64"
$ratios = @("0.0", "0.2", "0.4", "0.6", "0.8", "1.0")

foreach ($ratio in $ratios) {
    $label = $ratio.Replace(".", "p")
    $outDir = "$OutputRoot/ratio_$label"
    Write-Host "START run preset disturb ratio=$ratio -> $outDir"
    wsl.exe -d Ubuntu-22.04 --cd $repoWsl -- sh -lc "$envPrefix $python legged_gym/scripts/evaluate.py --task=r2amp --headless --sim_device=cpu --rl_device=cpu --num_envs=$NumEnvs --load_run $LoadRun --checkpoint=$Checkpoint --cfg_override_json $CfgOverrideJson --num_episodes=$NumEpisodes --episode_seconds=10 --preset run --eval_disturb_ratio $ratio --output_dir $outDir"
    if ($LASTEXITCODE -ne 0) {
        throw "run disturb sweep failed at ratio=$ratio"
    }
}

$summaryScript = @'
import csv
import pathlib

root = pathlib.Path("$OutputRoot")
rows = []
for path in sorted(root.glob("ratio_*/metrics.csv")):
    with path.open(encoding="utf-8") as f:
        metrics = list(csv.DictReader(f))
    if len(metrics) != 1:
        raise SystemExit(f"{path} should contain exactly one run preset row")
    row = metrics[0]
    ratio = path.parent.name.replace("ratio_", "").replace("p", ".")
    rows.append({
        "disturb_ratio": ratio,
        "fall_rate": row["fall_rate"],
        "survival_time_mean_s": row["survival_time_mean_s"],
        "lin_vel_rmse": row["lin_vel_rmse"],
        "yaw_vel_rmse": row["yaw_vel_rmse"],
        "task_return_mean": row["task_return_mean"],
    })

summary = root / "run_disturb_sweep_summary.csv"
with summary.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "disturb_ratio",
            "fall_rate",
            "survival_time_mean_s",
            "lin_vel_rmse",
            "yaw_vel_rmse",
            "task_return_mean",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

try:
    import matplotlib.pyplot as plt

    x = [float(row["disturb_ratio"]) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plots = [
        ("fall_rate", "Fall rate"),
        ("survival_time_mean_s", "Survival time (s)"),
        ("lin_vel_rmse", "Linear velocity RMSE"),
        ("yaw_vel_rmse", "Yaw velocity RMSE"),
    ]
    for ax, (key, title) in zip(axes.flat, plots):
        y = [float(row[key]) for row in rows]
        ax.plot(x, y, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Disturb ratio")
        ax.grid(True, alpha=0.3)
    fig.savefig(root / "run_disturb_sweep.png", dpi=160)
except Exception as exc:
    (root / "run_disturb_sweep_plot_error.txt").write_text(str(exc), encoding="utf-8")

print(f"Wrote {summary}")
'@

$tmpScript = Join-Path $env:TEMP "run_disturb_sweep_summary.py"
$summaryScript = $summaryScript.Replace('$OutputRoot', $OutputRoot.Replace('\', '/'))
Set-Content -LiteralPath $tmpScript -Value $summaryScript -Encoding UTF8
python $tmpScript
