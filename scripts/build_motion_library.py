#!/usr/bin/env python3
"""
Build a motion library of feasible upper-body motion clips.

Generates synthetic motion clips (arm swing, reaching, waving) and
validates them through the feasibility filter. Saves the library
for use in Phase 4+ training.

Usage:
    python scripts/build_motion_library.py --output data/motion_library
    python scripts/build_motion_library.py --output data/motion_library --num-clips 200
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.intervention.motion_library import MotionLibrary, MotionClip
from vr_teleop.intervention.feasibility_filter import FeasibilityFilter, FeasibilityConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Build motion library')
    parser.add_argument('--output', type=str, default='data/motion_library',
                        help='Output directory for motion library')
    parser.add_argument('--num-clips', type=int, default=100,
                        help='Number of clips to generate per motion type')
    parser.add_argument('--dt', type=float, default=0.02,
                        help='Timestep for motion clips')
    parser.add_argument('--validate', action='store_true',
                        help='Validate clips through feasibility filter')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


def generate_varied_arm_swings(n: int, dt: float, rng: np.random.Generator
                                ) -> list:
    """Generate arm swing clips with varied parameters."""
    clips = []
    for _ in range(n):
        duration = rng.uniform(1.0, 3.0)
        amplitude = rng.uniform(0.1, 0.6)
        frequency = rng.uniform(0.8, 2.5)
        clip = MotionLibrary.generate_arm_swing(
            duration=duration, dt=dt,
            amplitude=amplitude, frequency=frequency)
        clips.append(clip)
    return clips


def generate_varied_reaches(n: int, dt: float, rng: np.random.Generator
                             ) -> list:
    """Generate reaching clips with varied parameters."""
    clips = []
    for _ in range(n):
        duration = rng.uniform(0.8, 2.5)
        shoulder_pitch = rng.uniform(0.3, 1.2)
        elbow = rng.uniform(0.2, 0.8)
        arm = rng.choice(['left', 'right'])
        clip = MotionLibrary.generate_reaching(
            duration=duration, dt=dt,
            target_shoulder_pitch=shoulder_pitch,
            target_elbow=elbow,
            arm=arm)
        clips.append(clip)
    return clips


def generate_varied_waves(n: int, dt: float, rng: np.random.Generator
                           ) -> list:
    """Generate waving clips with varied parameters."""
    clips = []
    for _ in range(n):
        duration = rng.uniform(1.0, 3.0)
        arm = rng.choice(['left', 'right'])
        clip = MotionLibrary.generate_wave(
            duration=duration, dt=dt, arm=arm)
        clips.append(clip)
    return clips


def generate_random_smooth(n: int, dt: float, rng: np.random.Generator,
                            robot_cfg: G1Config) -> list:
    """Generate random smooth trajectories within joint limits."""
    clips = []
    upper_lower = np.array(robot_cfg.dof_pos_lower[15:])
    upper_upper = np.array(robot_cfg.dof_pos_upper[15:])

    for _ in range(n):
        duration = rng.uniform(1.5, 3.0)
        T = int(duration / dt)

        # Generate waypoints
        num_waypoints = rng.integers(2, 5)
        waypoint_times = np.sort(rng.uniform(0, duration, num_waypoints))
        waypoint_times = np.concatenate([[0.0], waypoint_times, [duration]])

        waypoints = np.zeros((len(waypoint_times), 14))
        # First waypoint is near zero (default pose)
        for i in range(1, len(waypoint_times)):
            # Random position within 60% of joint limits
            scale = 0.6
            waypoints[i] = rng.uniform(
                upper_lower * scale, upper_upper * scale)

        # Interpolate smoothly between waypoints
        t = np.linspace(0, duration, T)
        angles = np.zeros((T, 14), dtype=np.float32)
        for j in range(14):
            angles[:, j] = np.interp(t, waypoint_times, waypoints[:, j])

        # Lowpass smooth
        from scipy.ndimage import uniform_filter1d
        for j in range(14):
            angles[:, j] = uniform_filter1d(angles[:, j], size=int(0.1 / dt))

        clip = MotionClip(
            joint_angles=angles, dt=dt, duration=duration,
            motion_type='random_smooth',
        )
        clips.append(clip)

    return clips


def validate_clips(clips: list, robot_cfg: G1Config, dt: float) -> list:
    """Validate clips through feasibility filter.

    Returns list of clips that pass validation.
    """
    filter_cfg = FeasibilityConfig()
    feas_filter = FeasibilityFilter(
        num_envs=1, cfg=filter_cfg, robot_cfg=robot_cfg, device=torch.device('cpu'))
    valid_clips = []
    torso_euler = np.zeros((1, 3))  # Assume upright

    for clip in clips:
        is_valid = True
        prev_pos = clip.joint_angles[0:1].copy()

        for t_idx in range(1, clip.joint_angles.shape[0]):
            raw = torch.tensor(
                clip.joint_angles[t_idx:t_idx+1], dtype=torch.float32)
            current = torch.tensor(prev_pos, dtype=torch.float32)
            torso_t = torch.tensor(torso_euler, dtype=torch.float32)
            mask = torch.ones(1, dtype=torch.bool)

            filtered, safety_mask = feas_filter.filter(
                raw_targets=raw,
                current_upper_pos=current,
                torso_euler=torso_t,
                dt=dt,
                mask=mask,
            )

            if not safety_mask[0].item():
                is_valid = False
                break

            prev_pos = filtered.numpy()

        if is_valid:
            valid_clips.append(clip)

    return valid_clips


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    robot_cfg = G1Config()

    print(f"Building motion library")
    print(f"  Output: {args.output}")
    print(f"  Clips per type: {args.num_clips}")
    print(f"  dt: {args.dt}")

    library = MotionLibrary()

    # Generate clips
    print("\nGenerating arm swing clips...")
    swings = generate_varied_arm_swings(args.num_clips, args.dt, rng)
    print(f"  Generated {len(swings)} arm swing clips")

    print("Generating reaching clips...")
    reaches = generate_varied_reaches(args.num_clips, args.dt, rng)
    print(f"  Generated {len(reaches)} reaching clips")

    print("Generating wave clips...")
    waves = generate_varied_waves(args.num_clips, args.dt, rng)
    print(f"  Generated {len(waves)} wave clips")

    print("Generating random smooth clips...")
    try:
        randoms = generate_random_smooth(
            args.num_clips // 2, args.dt, rng, robot_cfg)
        print(f"  Generated {len(randoms)} random smooth clips")
    except ImportError:
        print("  Skipped (scipy not available)")
        randoms = []

    all_clips = swings + reaches + waves + randoms
    print(f"\nTotal clips generated: {len(all_clips)}")

    # Validate if requested
    if args.validate:
        print("\nValidating through feasibility filter...")
        valid_clips = validate_clips(all_clips, robot_cfg, args.dt)
        print(f"  Valid: {len(valid_clips)}/{len(all_clips)} "
              f"({100*len(valid_clips)/len(all_clips):.1f}%)")
        all_clips = valid_clips

    # Add to library
    for clip in all_clips:
        library.add_clip(clip)

    # Save
    os.makedirs(args.output, exist_ok=True)
    library.save(args.output)

    print(f"\nLibrary saved to {args.output}")
    print(f"  Total clips: {library.num_clips}")
    print(f"  Motion types: {library.motion_types}")
    for mt in library.motion_types:
        count = len(library._type_indices.get(mt, []))
        print(f"    {mt}: {count}")


if __name__ == '__main__':
    main()

