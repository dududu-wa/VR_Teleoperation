"""Run the approved Jul08_12 jump/stand full-disturbance diagnostic."""

import argparse
import subprocess
from typing import Callable, Dict, List


REPO_WSL = "/mnt/e/codebase/VR_Teleoperation"
PYTHON = "/opt/miniconda3/envs/r2gym/bin/python"
LOAD_RUN = (
    "Jul08_12/"
    "Jul08_12-34-51_selective_walk_profile_teacher_retention_disturb100_probe"
)
CFG_OVERRIDE = (
    "configs/ablation/"
    "selective_walk_profile_teacher_retention_disturb100_probe.json"
)
OUTPUT_DIR = (
    "outputs/eval/"
    "July08_12_selective_walk_profile_teacher_retention_disturb100_probe_"
    "16000_jump_stand_disturb100_failure_diagnostics"
)


def build_diagnostic_command() -> List[str]:
    """Return an argv-safe WSL CPU evaluation command for the fixed checkpoint."""
    return [
        "wsl.exe",
        "-d",
        "Ubuntu-22.04",
        "--cd",
        REPO_WSL,
        "--",
        "env",
        "PATH=/opt/miniconda3/envs/r2gym/bin:/opt/miniconda3/bin:/usr/local/sbin:"
        "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONPATH=/mnt/e/codebase/VR_Teleoperation:"
        "/mnt/e/codebase/VR_Teleoperation/rsl_rl",
        "LD_LIBRARY_PATH=/opt/miniconda3/envs/r2gym/lib:"
        "/mnt/e/wsl/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64",
        "KMP_DUPLICATE_LIB_OK=TRUE",
        PYTHON,
        "legged_gym/scripts/evaluate.py",
        "--task=r2amp",
        "--headless",
        "--sim_device=cpu",
        "--rl_device=cpu",
        "--num_envs=64",
        "--load_run",
        LOAD_RUN,
        "--checkpoint=16000",
        "--cfg_override_json",
        CFG_OVERRIDE,
        "--num_episodes=64",
        "--episode_seconds=10",
        "--preset",
        "jump",
        "--preset",
        "stand",
        "--eval_disturb_ratio=1.0",
        "--record_termination_reasons",
        "--record_state_trace",
        "--state_trace_window_steps=50",
        "--output_dir",
        OUTPUT_DIR,
    ]


def run_diagnostic(
    *,
    execute: bool,
    command_runner: Callable = subprocess.run,
) -> Dict[str, object]:
    """Print by default and execute only after an explicit opt-in."""
    command = build_diagnostic_command()
    if not execute:
        return {"executed": False, "command": command}
    command_runner(command, check=True)
    return {"executed": True, "command": command}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the fixed Jul08_12 jump/stand disturb100 diagnostic."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the WSL evaluation instead of only printing it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_diagnostic(execute=args.execute)
    if not result["executed"]:
        print(subprocess.list2cmdline(result["command"]))


if __name__ == "__main__":
    main()
