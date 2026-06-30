"""Generate the formal training command for the selective-walk follow-up.

The script is a planner only. It keeps the long-running training launch command
reviewable and prevents smoke run names from being reused as formal runs.
"""

import argparse


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
DEFAULT_LOAD_RUN = "Jun17/Jun17_14-46-44_expert_hard_gate_selective_walk"
DEFAULT_CHECKPOINT = "-2"
DEFAULT_RUN_NAME = "selective_walk_eval_manifold_conservative_disturb_release"
DEFAULT_RESUME_ITERATION = 4000
DEFAULT_TARGET_ITERATION = 8000


def additional_iterations(*, resume_iteration: int, target_iteration: int) -> int:
    """Return runner.learn iterations needed to reach an absolute checkpoint id."""
    remaining = int(target_iteration) - int(resume_iteration)
    if remaining <= 0:
        raise ValueError("target_iteration must be greater than resume_iteration")
    return remaining


def build_train_command(
    *,
    repo_wsl: str = DEFAULT_REPO_WSL,
    python: str = DEFAULT_PYTHON,
    env_prefix: str = DEFAULT_ENV_PREFIX,
    load_run: str = DEFAULT_LOAD_RUN,
    checkpoint: str = DEFAULT_CHECKPOINT,
    cfg_override_json: str = DEFAULT_CFG_OVERRIDE,
    run_name: str = DEFAULT_RUN_NAME,
    resume_iteration: int = DEFAULT_RESUME_ITERATION,
    target_iteration: int = DEFAULT_TARGET_ITERATION,
    sim_device: str = "cpu",
    rl_device: str = "cpu",
    num_envs: int = None,
) -> str:
    """Return the reviewed WSL command for the formal follow-up training run."""
    max_iterations = additional_iterations(
        resume_iteration=resume_iteration,
        target_iteration=target_iteration,
    )
    args = [
        f"{env_prefix}",
        f"{python}",
        "legged_gym/scripts/train.py",
        "--task=r2amp",
        "--headless",
        f"--sim_device={sim_device}",
        f"--rl_device={rl_device}",
        "--resume",
        f"--load_run {load_run}",
        f"--checkpoint={checkpoint}",
        f"--cfg_override_json {cfg_override_json}",
        f"--run_name {run_name}",
        f"--max_iterations={max_iterations}",
    ]
    if num_envs is not None:
        args.append(f"--num_envs={num_envs}")
    inner = " ".join(args)
    return f'wsl.exe -d Ubuntu-22.04 --cd {repo_wsl} -- sh -lc "{inner}"'


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Print the selective-walk conservative follow-up training command."
    )
    parser.add_argument("--repo_wsl", default=DEFAULT_REPO_WSL)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--load_run", default=DEFAULT_LOAD_RUN)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cfg_override_json", default=DEFAULT_CFG_OVERRIDE)
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--resume_iteration", type=int, default=DEFAULT_RESUME_ITERATION)
    parser.add_argument("--target_iteration", type=int, default=DEFAULT_TARGET_ITERATION)
    parser.add_argument("--sim_device", default="cpu")
    parser.add_argument("--rl_device", default="cpu")
    parser.add_argument("--num_envs", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        build_train_command(
            repo_wsl=args.repo_wsl,
            python=args.python,
            load_run=args.load_run,
            checkpoint=args.checkpoint,
            cfg_override_json=args.cfg_override_json,
            run_name=args.run_name,
            resume_iteration=args.resume_iteration,
            target_iteration=args.target_iteration,
            sim_device=args.sim_device,
            rl_device=args.rl_device,
            num_envs=args.num_envs,
        )
    )


if __name__ == "__main__":
    main()
