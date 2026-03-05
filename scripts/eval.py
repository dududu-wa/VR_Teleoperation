#!/usr/bin/env python3
"""
Evaluation script for trained G1 multi-gait policy.

Loads a checkpoint, runs batched episodes, and reports metrics.

Usage:
    python scripts/eval.py --checkpoint logs/g1_multigait/checkpoint_final.pt
    python scripts/eval.py --checkpoint model.pt --num-episodes 200 --gait walk
    python scripts/eval.py --checkpoint model.pt --per-gait
"""

import os
import sys
import argparse
import json
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
from vr_teleop.agents.actor_critic import ActorCritic
from vr_teleop.eval.evaluator import Evaluator, EvalConfig, EvalResult


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate G1 multi-gait policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--num-envs', type=int, default=64,
                        help='Number of parallel environments')
    parser.add_argument('--num-episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device')
    parser.add_argument('--gait', type=str, default=None,
                        choices=['stand', 'walk', 'run'],
                        help='Force specific gait (default: env random)')
    parser.add_argument('--vx', type=float, default=None,
                        help='Fixed forward velocity command')
    parser.add_argument('--per-gait', action='store_true',
                        help='Evaluate each gait separately')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--no-domain-rand', action='store_true',
                        help='Disable domain randomization during eval')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    # ---- Create configs ----
    robot_cfg = G1Config()
    obs_cfg = ObsConfig()
    reward_cfg = RewardConfig()
    term_cfg = TerminationConfig(episode_length=1000)
    rand_cfg = DomainRandConfig()
    if args.no_domain_rand:
        rand_cfg.randomize_friction = False
        rand_cfg.randomize_mass = False
        rand_cfg.randomize_pd_gains = False
        rand_cfg.randomize_motor_strength = False
        rand_cfg.push_robots = False

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

    # ---- Create actor-critic and load checkpoint ----
    num_actor_obs = env.num_obs
    num_critic_obs = env.num_privileged_obs
    num_actions = env.num_actions

    actor_critic = ActorCritic(
        num_actor_obs=num_actor_obs,
        num_critic_obs=num_critic_obs,
        num_actions=num_actions,
    ).to(device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    actor_critic.load_state_dict(ckpt['model_state_dict'])
    actor_critic.eval()
    print(f"  Loaded from iteration {ckpt.get('iter', '?')}")

    # ---- Run evaluation ----
    eval_cfg = EvalConfig(
        num_episodes=args.num_episodes,
        max_episode_length=1000,
    )
    evaluator = Evaluator(
        env=env,
        actor_critic=actor_critic,
        cfg=eval_cfg,
        device=device,
    )

    if args.per_gait:
        results = evaluator.evaluate_per_gait(verbose=True)
        # Combine for JSON output
        output_data = {}
        for gait_name, result in results.items():
            output_data[gait_name] = {
                'mean_episode_length': result.mean_episode_length,
                'mean_reward': result.mean_reward,
                'fall_rate': result.fall_rate,
                'survival_rate': result.survival_rate,
                'mean_tracking_error': result.mean_tracking_error,
                'tracking_success_rate': result.tracking_success_rate,
                'mean_torque_cost': result.mean_torque_cost,
            }
    else:
        # Build fixed commands if specified
        commands = None
        if args.vx is not None:
            commands = torch.tensor([args.vx, 0.0, 0.0])

        result = evaluator.evaluate(
            gait=args.gait, commands=commands, verbose=True)
        output_data = {
            'mean_episode_length': result.mean_episode_length,
            'mean_reward': result.mean_reward,
            'fall_rate': result.fall_rate,
            'survival_rate': result.survival_rate,
            'mean_tracking_error': result.mean_tracking_error,
            'tracking_success_rate': result.tracking_success_rate,
            'mean_torque_cost': result.mean_torque_cost,
        }

    # ---- Save results ----
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
