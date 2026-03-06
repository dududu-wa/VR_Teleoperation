#!/usr/bin/env python3
"""
Sim2Sim validation entrypoint.

Usage:
    python scripts/sim2sim.py --checkpoint logs/g1_multigait/checkpoint_final.pt
    python scripts/sim2sim.py --checkpoint model.pt --suite friction
"""

import argparse
import json
import os
import sys
from typing import Dict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Run sim2sim validation sweeps")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--suite", type=str, default="full",
                        choices=["full", "friction", "mass", "pd"],
                        help="Validation sweep to run")
    parser.add_argument("--num-envs", type=int, default=32,
                        help="Parallel envs used during each sweep setting")
    parser.add_argument("--episodes-per-setting", type=int, default=50,
                        help="Evaluation episodes per sweep setting")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device")
    parser.add_argument("--gait", type=str, default="walk",
                        choices=["stand", "walk", "run"],
                        help="Forced gait for sim2sim runs")
    parser.add_argument("--vx", type=float, default=0.4,
                        help="Forward velocity command")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    return parser.parse_args()


def _result_to_dict(result) -> Dict[str, float]:
    return {
        "mean_episode_length": float(result.mean_episode_length),
        "mean_reward": float(result.mean_reward),
        "fall_rate": float(result.fall_rate),
        "survival_rate": float(result.survival_rate),
        "mean_tracking_error": float(result.mean_tracking_error),
        "tracking_success_rate": float(result.tracking_success_rate),
        "transition_failure_rate": float(result.transition_failure_rate),
        "mean_torque_cost": float(result.mean_torque_cost),
    }


def _serialize_sweep(result_map) -> Dict[str, Dict[str, float]]:
    return {str(k): _result_to_dict(v) for k, v in result_map.items()}


def main():
    args = parse_args()
    from vr_teleop.deploy.sim2sim_runner import Sim2SimRunner, Sim2SimConfig

    cfg = Sim2SimConfig(
        num_envs=args.num_envs,
        num_episodes_per_setting=args.episodes_per_setting,
        device=args.device,
        gait=args.gait,
        vx=args.vx,
    )
    runner = Sim2SimRunner.from_checkpoint(args.checkpoint, cfg=cfg)

    if args.suite == "full":
        raw = runner.run_full_validation()
        output_data = {key: _serialize_sweep(val) for key, val in raw.items()}
    elif args.suite == "friction":
        raw = runner.sweep_friction([0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
        output_data = {"friction": _serialize_sweep(raw)}
    elif args.suite == "mass":
        raw = runner.sweep_mass_offset([-2.0, -1.0, 0.0, 1.0, 2.0, 4.0])
        output_data = {"mass": _serialize_sweep(raw)}
    else:
        raw = runner.sweep_pd_gain_scale([0.6, 0.8, 1.0, 1.2, 1.5])
        output_data = {"pd_gain": _serialize_sweep(raw)}

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
