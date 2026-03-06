#!/usr/bin/env python3
"""
Training entry point for G1 multi-gait intervention-robust locomotion.

Usage:
    python scripts/train.py
    python scripts/train.py --num-envs 256 --device cuda:0 --sim-backend isaacgym
    python scripts/train.py --use-intervention --initial-phase 0
"""

import os
import sys
import argparse
import time
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


DEFAULT_ARGS = {
    "num_envs": 256,
    "sim_backend": "isaacgym",
    "max_iterations": 10000,
    "num_steps": 24,
    "save_interval": 500,
    "experiment_name": "g1_multigait",
    "initial_phase": 0,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train G1 multi-gait policy')
    parser.add_argument('--num-envs', type=int, default=DEFAULT_ARGS["num_envs"],
                        help='Number of parallel environments')
    parser.add_argument('--sim-backend', type=str, default=DEFAULT_ARGS["sim_backend"],
                        choices=['isaacgym'],
                        help='Simulation backend entrypoint')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device')
    parser.add_argument('--max-iterations', type=int, default=DEFAULT_ARGS["max_iterations"],
                        help='Maximum training iterations')
    parser.add_argument('--num-steps', type=int, default=DEFAULT_ARGS["num_steps"],
                        help='Rollout steps per env per iteration')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--save-interval', type=int, default=DEFAULT_ARGS["save_interval"],
                        help='Checkpoint save interval (iterations)')
    parser.add_argument('--experiment-name', type=str, default=DEFAULT_ARGS["experiment_name"],
                        help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-intervention', action='store_true',
                        help='Enable upper-body intervention during training')
    parser.add_argument('--initial-phase', type=int, default=DEFAULT_ARGS["initial_phase"],
                        help='Starting curriculum phase')
    parser.add_argument('--experiment-config', type=str, default=None,
                        help='Path to configs/experiment/*.yaml')
    return parser.parse_args()


def main():
    args = parse_args()
    from vr_teleop.robot.g1_config import G1Config
    from vr_teleop.envs.g1_multigait_env import G1MultigaitEnv
    from vr_teleop.envs.observation import ObsConfig
    from vr_teleop.envs.reward import RewardConfig
    from vr_teleop.envs.termination import TerminationConfig
    from vr_teleop.envs.domain_rand import DomainRandConfig
    from vr_teleop.agents.runner import OnPolicyRunner
    from vr_teleop.curriculum.phase_curriculum import PhaseCurriculum, PhaseConfig
    from vr_teleop.intervention.intervention_generator import InterventionGenerator, InterventionConfig
    from vr_teleop.intervention.feasibility_filter import FeasibilityFilter, FeasibilityConfig
    from vr_teleop.utils.config_utils import load_experiment_config

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # ---- Optional experiment config override ----
    experiment_cfg = {}
    if args.experiment_config:
        experiment_cfg = load_experiment_config(args.experiment_config)

    def _use_cfg(arg_name: str, cfg_value):
        if cfg_value is None:
            return
        if getattr(args, arg_name) == DEFAULT_ARGS[arg_name]:
            setattr(args, arg_name, cfg_value)

    _use_cfg("sim_backend", experiment_cfg.get("sim_backend"))
    _use_cfg("num_envs", experiment_cfg.get("num_envs"))

    exp_cfg = experiment_cfg.get("experiment", {}) if isinstance(experiment_cfg, dict) else {}
    _use_cfg("experiment_name", exp_cfg.get("name"))

    algo_cfg = experiment_cfg.get("algo", {}) if isinstance(experiment_cfg, dict) else {}
    _use_cfg("max_iterations", algo_cfg.get("max_iterations"))
    _use_cfg("save_interval", algo_cfg.get("save_interval"))
    _use_cfg("num_steps", algo_cfg.get("num_steps_per_env"))

    cur_cfg = experiment_cfg.get("curriculum", {}) if isinstance(experiment_cfg, dict) else {}
    _use_cfg("initial_phase", cur_cfg.get("initial_phase"))

    int_cfg = experiment_cfg.get("intervention", {}) if isinstance(experiment_cfg, dict) else {}
    if isinstance(int_cfg, dict) and "use_disturb" in int_cfg and not args.use_intervention:
        args.use_intervention = bool(int_cfg["use_disturb"])

    env_cfg_blob = experiment_cfg.get("env", {}) if isinstance(experiment_cfg, dict) else {}
    # Support both styles:
    # 1) env: {num_envs, command_ranges...}
    # 2) env: {env: {episode_length, commands...}}
    env_cfg = env_cfg_blob.get("env", env_cfg_blob) if isinstance(env_cfg_blob, dict) else {}

    command_ranges = env_cfg.get("command_ranges")
    if command_ranges is None and isinstance(env_cfg.get("commands"), dict):
        commands_cfg = env_cfg["commands"]
        command_ranges = {
            gait: commands_cfg[gait]
            for gait in ("stand", "walk", "run")
            if gait in commands_cfg
        } or None
    gait_probs = env_cfg.get("gait_probs")
    if gait_probs is None and isinstance(env_cfg.get("commands"), dict):
        gait_probs = env_cfg["commands"].get("gait_probs")

    resampling_time = env_cfg.get("resampling_time")
    if resampling_time is None and isinstance(env_cfg.get("commands"), dict):
        resampling_time = env_cfg["commands"].get("resampling_time")

    gait_cfg = env_cfg.get("gait", {}) if isinstance(env_cfg.get("gait"), dict) else {}
    walk_freq = gait_cfg.get("walk_freq", 2.0)
    run_freq = gait_cfg.get("run_freq", 3.0)
    phase_offset = gait_cfg.get("phase_offset", 0.5)

    sim_dt = env_cfg.get("sim_dt", 0.005)
    decimation = env_cfg.get("decimation", 4)
    max_episode_length = env_cfg.get("max_episode_length")
    if max_episode_length is None:
        max_episode_length = env_cfg.get("episode_length")
    if max_episode_length is None:
        ep_s = env_cfg.get("episode_length_s")
        if ep_s is not None:
            max_episode_length = int(float(ep_s) / (float(sim_dt) * int(decimation)))
    if max_episode_length is None:
        max_episode_length = 1000

    if args.sim_backend != "isaacgym":
        raise ValueError(
            "Training backend must be 'isaacgym'. MuJoCo is only for post-training visualization."
        )

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Training G1 multi-gait policy")
    print(f"  Envs: {args.num_envs}")
    print(f"  Backend: {args.sim_backend}")
    print(f"  Device: {device}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Intervention: {args.use_intervention}")
    print(f"  Initial phase: {args.initial_phase}")

    # ---- Create configs ----
    robot_cfg = G1Config()
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig()
    term_cfg = TerminationConfig(episode_length=max_episode_length)
    rand_cfg = DomainRandConfig()

    # ---- Create environment ----
    env = G1MultigaitEnv(
        num_envs=args.num_envs,
        robot_cfg=robot_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        term_cfg=term_cfg,
        rand_cfg=rand_cfg,
        sim_dt=sim_dt,
        decimation=decimation,
        max_episode_length=max_episode_length,
        device=device,
        sim_backend=args.sim_backend,
        command_ranges=command_ranges,
        gait_probs=gait_probs,
        resampling_time=resampling_time if resampling_time is not None else 10.0,
        walk_freq=walk_freq,
        run_freq=run_freq,
        phase_offset=phase_offset,
    )
    resolved_backend = getattr(env.vec_env, "backend_name", args.sim_backend)
    print(f"  Resolved backend: {resolved_backend}")

    # ---- Config dicts for runner ----
    actor_critic_cfg = {}  # use defaults from ActorCritic
    ppo_cfg = {
        'num_learning_epochs': int(algo_cfg.get('num_learning_epochs', 8)),
        'num_mini_batches': int(algo_cfg.get('num_mini_batches', 4)),
    }
    runner_cfg = {
        'num_steps_per_env': args.num_steps,
        'save_interval': args.save_interval,
    }

    log_dir = os.path.join(args.log_dir, args.experiment_name)

    # ---- Create runner (creates actor-critic + PPO internally) ----
    runner = OnPolicyRunner(
        env=env,
        actor_critic_cfg=actor_critic_cfg,
        ppo_cfg=ppo_cfg,
        runner_cfg=runner_cfg,
        log_dir=log_dir,
        device=device,
    )

    # ---- Create curriculum ----
    phase_cfg = PhaseConfig(initial_phase=args.initial_phase)
    curriculum = PhaseCurriculum(cfg=phase_cfg)

    # ---- Create intervention generator ----
    intervention = None
    feasibility_filter = None
    if args.use_intervention:
        int_cfg = InterventionConfig()
        intervention = InterventionGenerator(
            num_envs=args.num_envs,
            cfg=int_cfg,
            robot_cfg=robot_cfg,
            device=torch.device(device),
        )
        feasibility_filter = FeasibilityFilter(
            num_envs=args.num_envs,
            cfg=FeasibilityConfig(),
            robot_cfg=robot_cfg,
            device=torch.device(device),
        )
        intervention.set_phase(args.initial_phase)
        env.attach_intervention(
            generator=intervention,
            feasibility_filter=feasibility_filter,
            auto_mode=True,
        )

    # ---- Resume from checkpoint ----
    if args.resume:
        print(f"Resuming from {args.resume}")
        runner.load(args.resume)

    # ---- Training ----
    print(f"\nStarting training for {args.max_iterations} iterations...")

    # Use runner.learn() which handles rollout + update + logging
    # But we also need curriculum updates per iteration.
    # So we run learn() in chunks, updating curriculum between.

    start_time = time.time()
    iters_done = 0
    chunk_size = 10  # update curriculum every 10 iterations

    while iters_done < args.max_iterations:
        # Apply curriculum settings before each chunk
        cmd_ranges = curriculum.get_command_ranges()
        env.update_command_ranges(cmd_ranges)
        env.update_gait_probs(curriculum.get_gait_probs())

        if intervention is not None:
            intervention.set_phase(curriculum.phase)

        # How many iterations in this chunk
        remaining = args.max_iterations - iters_done
        this_chunk = min(chunk_size, remaining)

        # Run training chunk
        runner.learn(num_learning_iterations=this_chunk)
        iters_done += this_chunk

        # Update curriculum metrics from env
        extras = env.extras
        ep_info = extras.get('episode', {})
        tracking_lin = ep_info.get('reward_tracking_lin_vel', None)
        tracking_ang = ep_info.get('reward_tracking_ang_vel', None)
        if tracking_lin is not None and tracking_ang is not None:
            tracking_reward = 0.5 * (tracking_lin + tracking_ang)
        else:
            tracking_reward = ep_info.get('reward_mean', 0.0)
        fall_rate = ep_info.get('contact_fall_rate', 0.0)
        orientation_fall = ep_info.get('orientation_fall_rate', 0.0)
        total_fall_rate = fall_rate + orientation_fall
        transition_failure = ep_info.get('transition_failure_rate', 0.0)
        ep_len = ep_info.get('episode_length_mean', 0.0)

        curriculum.update_metrics(
            tracking_reward=tracking_reward,
            fall_rate=total_fall_rate,
            transition_failure=transition_failure,
            mean_episode_length=ep_len,
        )

        # Check for phase promotion
        promoted = curriculum.check_promotion()
        if promoted:
            print(f"\n*** Phase promoted to {curriculum.phase} "
                  f"at iteration {iters_done} ***\n")

        # Progress log
        elapsed = time.time() - start_time
        fps = (iters_done * args.num_steps * args.num_envs) / max(elapsed, 1)
        print(f"Iter {iters_done:5d}/{args.max_iterations} | "
              f"Phase {curriculum.phase} | "
              f"Reward {tracking_reward:6.3f} | "
              f"Fall {total_fall_rate:5.3f} | "
              f"TransFail {transition_failure:5.3f} | "
              f"FPS {fps:7.0f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Final phase: {curriculum.phase}")
    runner.close()


if __name__ == '__main__':
    main()

