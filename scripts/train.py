#!/usr/bin/env python3
"""
Training entry point for G1 multi-gait intervention-robust locomotion.

Usage:
    python scripts/train.py
    python scripts/train.py --num-envs 256 --device cpu
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

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.envs.g1_multigait_env import G1MultigaitEnv
from vr_teleop.envs.observation import ObsConfig
from vr_teleop.envs.reward import RewardConfig
from vr_teleop.envs.termination import TerminationConfig
from vr_teleop.envs.domain_rand import DomainRandConfig
from vr_teleop.agents.runner import OnPolicyRunner
from vr_teleop.curriculum.phase_curriculum import PhaseCurriculum, PhaseConfig
from vr_teleop.intervention.intervention_generator import InterventionGenerator, InterventionConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train G1 multi-gait policy')
    parser.add_argument('--num-envs', type=int, default=256,
                        help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device')
    parser.add_argument('--max-iterations', type=int, default=10000,
                        help='Maximum training iterations')
    parser.add_argument('--num-steps', type=int, default=24,
                        help='Rollout steps per env per iteration')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='TensorBoard log directory')
    parser.add_argument('--save-interval', type=int, default=500,
                        help='Checkpoint save interval (iterations)')
    parser.add_argument('--experiment-name', type=str, default='g1_multigait',
                        help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-intervention', action='store_true',
                        help='Enable upper-body intervention during training')
    parser.add_argument('--initial-phase', type=int, default=0,
                        help='Starting curriculum phase')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Training G1 multi-gait policy")
    print(f"  Envs: {args.num_envs}")
    print(f"  Device: {device}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Intervention: {args.use_intervention}")
    print(f"  Initial phase: {args.initial_phase}")

    # ---- Create configs ----
    robot_cfg = G1Config()
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig()
    term_cfg = TerminationConfig(episode_length=1000)
    rand_cfg = DomainRandConfig()

    # ---- Create environment ----
    env = G1MultigaitEnv(
        num_envs=args.num_envs,
        robot_cfg=robot_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        term_cfg=term_cfg,
        rand_cfg=rand_cfg,
        device=device,
    )

    # ---- Config dicts for runner ----
    actor_critic_cfg = {}  # use defaults from ActorCritic
    ppo_cfg = {
        'num_learning_epochs': 8,
        'num_mini_batches': 4,
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
    if args.use_intervention:
        int_cfg = InterventionConfig()
        intervention = InterventionGenerator(
            num_envs=args.num_envs,
            cfg=int_cfg,
            robot_cfg=robot_cfg,
            device=torch.device(device),
        )
        intervention.set_phase(args.initial_phase)

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
        tracking_reward = ep_info.get('reward_mean', 0.0)
        fall_rate = ep_info.get('contact_fall_rate', 0.0)
        orientation_fall = ep_info.get('orientation_fall_rate', 0.0)
        total_fall_rate = fall_rate + orientation_fall
        ep_len = ep_info.get('episode_length_mean', 0.0)

        curriculum.update_metrics(
            tracking_reward=tracking_reward,
            fall_rate=total_fall_rate,
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
              f"FPS {fps:7.0f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s")
    print(f"Final phase: {curriculum.phase}")


if __name__ == '__main__':
    main()
