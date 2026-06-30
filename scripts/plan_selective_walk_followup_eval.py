"""Generate the fixed follow-up evaluation commands for the selective-walk run.

The script is intentionally a planner, not a launcher: training can be long and
machine-specific, while the post-training evaluation protocol should be stable
and reviewable before any WSL CPU jobs are started.
"""

import argparse
import json
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence


DEFAULT_REPO_WSL = "/mnt/e/codebase/VR_Teleoperation"
DEFAULT_PYTHON = "/opt/miniconda3/envs/r2gym/bin/python"
DEFAULT_CFG_OVERRIDE = (
    "configs/ablation/selective_walk_eval_manifold_conservative_disturb_release.json"
)
DEFAULT_ENV_PREFIX = (
    "PATH=/opt/miniconda3/envs/r2gym/bin:/opt/miniconda3/bin:/usr/local/sbin:"
    "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
    "PYTHONPATH=/mnt/e/codebase/VR_Teleoperation:"
    "/mnt/e/codebase/VR_Teleoperation/rsl_rl "
    "LD_LIBRARY_PATH=/opt/miniconda3/envs/r2gym/lib:"
    "/mnt/e/wsl/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64"
)


class EvalCommand(NamedTuple):
    label: str
    output_dir: str
    command: str


def _ratio_label(ratio: float) -> str:
    # Keep labels aligned with the existing June30 output directory names.
    known = {
        0.75: "075",
        0.9: "090",
        0.925: "0925",
        0.95: "095",
        1.0: "100",
    }
    return known.get(float(ratio), f"{ratio:g}".replace(".", ""))


def _ratio_arg(ratio: float) -> str:
    known = {
        0.75: "0.75",
        0.9: "0.9",
        0.925: "0.925",
        0.95: "0.95",
        1.0: "1.0",
    }
    return known.get(float(ratio), f"{ratio:g}")


def _build_evaluate_command(
    *,
    load_run: str,
    checkpoint: str,
    output_dir: str,
    cfg_override_json: str,
    repo_wsl: str,
    python: str,
    env_prefix: str,
    num_envs: int,
    num_episodes: int,
    episode_seconds: int,
    eval_disturb_ratio: Optional[float] = None,
    presets: Optional[Sequence[str]] = None,
    record_diagnostics: bool = False,
) -> str:
    args = [
        f"{env_prefix}",
        f"{python}",
        "legged_gym/scripts/evaluate.py",
        "--task=r2amp",
        "--headless",
        "--sim_device=cpu",
        "--rl_device=cpu",
        f"--num_envs={num_envs}",
        f"--load_run {load_run}",
        f"--checkpoint={checkpoint}",
        f"--cfg_override_json {cfg_override_json}",
        f"--num_episodes={num_episodes}",
        f"--episode_seconds={episode_seconds}",
        f"--output_dir {output_dir}",
    ]
    if presets:
        for preset in presets:
            args.append(f"--preset {preset}")
    if eval_disturb_ratio is not None:
        args.append(f"--eval_disturb_ratio={_ratio_arg(eval_disturb_ratio)}")
    if record_diagnostics:
        # Termination reasons and short pre-reset traces are the two diagnostics
        # used in the existing June30 boundary analysis.
        args.extend(["--record_termination_reasons", "--record_state_trace"])
    inner = " ".join(args)
    return f'wsl.exe -d Ubuntu-22.04 --cd {repo_wsl} -- sh -lc "{inner}"'


def build_eval_plan(
    *,
    load_run: str,
    checkpoint: str,
    output_prefix: str,
    cfg_override_json: str = DEFAULT_CFG_OVERRIDE,
    repo_wsl: str = DEFAULT_REPO_WSL,
    python: str = DEFAULT_PYTHON,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    num_envs: int = 64,
    num_episodes: int = 64,
    episode_seconds: int = 10,
) -> List[EvalCommand]:
    """Return the full post-training eval plan for one checkpoint."""
    commands: List[EvalCommand] = []

    def add(
        label: str,
        *,
        ratio: Optional[float] = None,
        presets: Optional[Sequence[str]] = None,
        diagnostics: bool = False,
    ) -> None:
        output_dir = f"{output_prefix}_{label}"
        commands.append(
            EvalCommand(
                label=label,
                output_dir=output_dir,
                command=_build_evaluate_command(
                    load_run=load_run,
                    checkpoint=checkpoint,
                    output_dir=output_dir,
                    cfg_override_json=cfg_override_json,
                    repo_wsl=repo_wsl,
                    python=python,
                    env_prefix=env_prefix,
                    num_envs=num_envs,
                    num_episodes=num_episodes,
                    episode_seconds=episode_seconds,
                    eval_disturb_ratio=ratio,
                    presets=presets,
                    record_diagnostics=diagnostics,
                ),
            )
        )

    add("baseline_full7")
    for ratio in (0.75, 0.9, 0.925, 0.95, 1.0):
        add(f"full7_disturb{_ratio_label(ratio)}", ratio=ratio)
    add(
        "failure_diagnostics_disturb0925",
        ratio=0.925,
        presets=("stand", "run", "jump"),
        diagnostics=True,
    )
    add(
        "failure_diagnostics_disturb095",
        ratio=0.95,
        presets=("stand", "run", "jump", "strafe_right"),
        diagnostics=True,
    )
    add(
        "failure_diagnostics_disturb100",
        ratio=1.0,
        presets=("stand", "run", "jump"),
        diagnostics=True,
    )
    return commands


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Print the selective-walk conservative follow-up evaluation plan."
    )
    parser.add_argument("--load_run", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--cfg_override_json", default=DEFAULT_CFG_OVERRIDE)
    parser.add_argument("--repo_wsl", default=DEFAULT_REPO_WSL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_episodes", type=int, default=64)
    parser.add_argument("--episode_seconds", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    commands = build_eval_plan(
        load_run=args.load_run,
        checkpoint=args.checkpoint,
        output_prefix=args.output_prefix,
        cfg_override_json=args.cfg_override_json,
        repo_wsl=args.repo_wsl,
        python=args.python,
        num_envs=args.num_envs,
        num_episodes=args.num_episodes,
        episode_seconds=args.episode_seconds,
    )
    if args.json:
        print(json.dumps([command._asdict() for command in commands], indent=2))
        return
    for command in commands:
        print(f"# {command.label}")
        print(command.command)
        print()


if __name__ == "__main__":
    main()
