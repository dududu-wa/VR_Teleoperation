#!/usr/bin/env python3
"""
Visualization / playback script for trained G1 multi-gait policy.

Loads a checkpoint and runs the policy in a single MuJoCo viewer
with real-time rendering.

Usage:
    python scripts/play.py --checkpoint logs/g1_multigait/checkpoint_final.pt
    python scripts/play.py --checkpoint model.pt --gait walk --vx 0.5
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import mujoco
import mujoco.viewer

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.envs.g1_base_env import G1BaseEnv
from vr_teleop.envs.observation import ObservationBuilder, ObsConfig
from vr_teleop.agents.actor_critic import ActorCritic
from vr_teleop.utils.math_utils import (
    mujoco_quat_to_isaac, compute_projected_gravity,
    quat_rotate_inverse, get_euler_xyz
)
from vr_teleop.utils.config_utils import get_asset_path


def parse_args():
    parser = argparse.ArgumentParser(description='Play trained G1 policy')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint .pt file')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device for inference')
    parser.add_argument('--gait', type=str, default='walk',
                        choices=['stand', 'walk', 'run'],
                        help='Gait mode to use')
    parser.add_argument('--vx', type=float, default=0.4,
                        help='Forward velocity command')
    parser.add_argument('--vy', type=float, default=0.0,
                        help='Lateral velocity command')
    parser.add_argument('--wz', type=float, default=0.0,
                        help='Yaw rate command')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Playback duration in seconds')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Run without viewer (headless)')
    parser.add_argument('--record', type=str, default=None,
                        help='Record video to this path (requires no-viewer)')
    return parser.parse_args()


GAIT_MAP = {'stand': 0, 'walk': 1, 'run': 2}


class PolicyPlayer:
    """Runs a trained policy on a single MuJoCo environment with viewer."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu',
                 robot_cfg: G1Config = None, obs_cfg: ObsConfig = None):
        self.device = torch.device(device)
        self.robot_cfg = robot_cfg or G1Config.from_falcon_yaml_if_available()
        self.obs_cfg = obs_cfg or ObsConfig()

        # Load model
        asset_root = get_asset_path()
        model_path = os.path.join(asset_root, self.robot_cfg.mujoco_scene_file)
        if not os.path.exists(model_path):
            model_path = os.path.join(asset_root, self.robot_cfg.mujoco_model_file)

        self.base_env = G1BaseEnv(
            robot_cfg=self.robot_cfg,
            sim_dt=0.005,
            decimation=4,
            model_path=model_path,
        )
        self.dt = 0.005 * 4  # policy timestep

        # Observation builder (single env)
        self.obs_builder = ObservationBuilder(
            obs_cfg=self.obs_cfg,
            num_envs=1,
            device=self.device,
            lower_body_dofs=self.robot_cfg.lower_body_dofs,
        )

        # Load actor-critic
        num_actor_obs = self.obs_cfg.single_step_dim + \
            self.obs_cfg.history_obs_dim * self.obs_cfg.include_history_steps
        num_critic_obs = self.obs_cfg.critic_obs_dim
        num_actions = self.robot_cfg.lower_body_dofs

        self.actor_critic = ActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
        ).to(self.device)

        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.actor_critic.load_state_dict(ckpt['model_state_dict'])
        self.actor_critic.eval()
        print(f"  Loaded from iteration {ckpt.get('iter', '?')}")

        # Default DOF positions
        self.default_dof_pos = self.robot_cfg.get_default_dof_pos().to(self.device).unsqueeze(0)

        # State buffers (single env, unsqueezed to batch dim 1)
        self.last_actions = torch.zeros(1, num_actions, device=self.device)

        # Gait state
        self.gait_phase = 0.0
        self.walk_freq = 2.0
        self.run_freq = 3.0

    def reset(self):
        """Reset the environment."""
        self.base_env.reset()
        self.last_actions.zero_()
        self.gait_phase = 0.0
        # Reset observation history
        init_obs = self._build_single_step_obs(
            gait_id=0, commands=torch.zeros(1, 3, device=self.device),
            intervention_flag=torch.zeros(1, device=self.device),
            clock=torch.zeros(1, 2, device=self.device),
        )
        env_ids = torch.tensor([0], device=self.device)
        self.obs_builder.reset_history(env_ids, init_obs)

    def _get_state_tensors(self):
        """Extract state from base_env as tensors on device."""
        env = self.base_env
        # Base quaternion: MuJoCo wxyz -> xyzw
        quat_wxyz = torch.tensor(env.base_quat_wxyz, dtype=torch.float32)
        quat_xyzw = torch.cat([quat_wxyz[1:4], quat_wxyz[0:1]]).unsqueeze(0).to(self.device)

        base_ang_vel = torch.tensor(env.base_ang_vel, dtype=torch.float32).unsqueeze(0).to(self.device)
        base_lin_vel = torch.tensor(env.base_lin_vel, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Body-frame velocities
        base_ang_vel_body = quat_rotate_inverse(quat_xyzw, base_ang_vel)
        projected_gravity = compute_projected_gravity(quat_xyzw)

        # DOF state
        dof_pos = torch.tensor(env.dof_pos, dtype=torch.float32).unsqueeze(0).to(self.device)
        dof_vel = torch.tensor(env.dof_vel, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Relative DOF positions
        dof_pos_rel = dof_pos - self.default_dof_pos

        return {
            'base_ang_vel_body': base_ang_vel_body,
            'projected_gravity': projected_gravity,
            'dof_pos_rel': dof_pos_rel,
            'dof_vel': dof_vel,
        }

    def _build_single_step_obs(self, gait_id: int, commands: torch.Tensor,
                                intervention_flag: torch.Tensor,
                                clock: torch.Tensor) -> torch.Tensor:
        """Build single-step actor obs (1, 58)."""
        state = self._get_state_tensors()
        lower_dof_pos = state['dof_pos_rel'][:, :self.robot_cfg.lower_body_dofs]
        lower_dof_vel = state['dof_vel'][:, :self.robot_cfg.lower_body_dofs]

        return self.obs_builder.build_actor_obs(
            base_ang_vel=state['base_ang_vel_body'],
            projected_gravity=state['projected_gravity'],
            dof_pos_lower=lower_dof_pos,
            dof_vel_lower=lower_dof_vel,
            last_actions=self.last_actions,
            commands=commands,
            gait_id=torch.tensor([float(gait_id)], device=self.device),
            intervention_flag=intervention_flag,
            clock=clock,
        )

    def step(self, gait_id: int, commands: torch.Tensor):
        """Run one policy step.

        Args:
            gait_id: 0=stand, 1=walk, 2=run
            commands: (1, 3) [vx, vy, wz]

        Returns:
            actions: (15,) numpy array of lower body actions
        """
        # Update gait clock
        freq = 0.0
        if gait_id == 1:
            freq = self.walk_freq
        elif gait_id == 2:
            freq = self.run_freq
        self.gait_phase = (self.gait_phase + freq * self.dt) % 1.0

        clock = torch.zeros(1, 2, device=self.device)
        if gait_id != 0:
            phase_rad = 2.0 * np.pi * self.gait_phase
            clock[0, 0] = np.sin(phase_rad)
            clock[0, 1] = np.cos(phase_rad)

        intervention_flag = torch.zeros(1, device=self.device)

        # Build observation
        actor_obs = self._build_single_step_obs(
            gait_id, commands, intervention_flag, clock)
        self.obs_builder.update_history(actor_obs)
        full_obs = self.obs_builder.get_actor_obs_with_history(actor_obs)

        # Run policy (deterministic)
        with torch.inference_mode():
            actions_mean, _ = self.actor_critic.act_inference(full_obs)

        actions = actions_mean.squeeze(0)
        self.last_actions[0] = actions

        # Step physics
        actions_np = torch.zeros(self.robot_cfg.num_dofs, dtype=torch.float32)
        actions_np[:self.robot_cfg.lower_body_dofs] = actions.cpu()
        self.base_env.step(actions_np.numpy())

        return actions.cpu().numpy()

    def run_with_viewer(self, gait: str, vx: float, vy: float, wz: float,
                        duration: float):
        """Run policy with MuJoCo passive viewer."""
        gait_id = GAIT_MAP[gait]
        commands = torch.tensor([[vx, vy, wz]], device=self.device)

        if gait == 'stand':
            commands.zero_()

        self.reset()

        print(f"\nRunning policy: gait={gait}, vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")
        print(f"Duration: {duration:.0f}s")
        print("Close the viewer window to stop.\n")

        model = self.base_env.mj_model
        data = self.base_env.mj_data

        step_count = 0
        max_steps = int(duration / self.dt)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            while viewer.is_running() and step_count < max_steps:
                step_start = time.time()

                # Policy step
                actions = self.step(gait_id, commands)
                step_count += 1

                # Check if robot fell
                height = self.base_env.base_pos[2]
                if height < 0.3:
                    print(f"Robot fell at step {step_count} (height={height:.3f})")
                    self.reset()
                    step_count = 0

                # Sync viewer
                viewer.sync()

                # Real-time pacing
                elapsed = time.time() - step_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Periodic status
                if step_count % 50 == 0:
                    total_elapsed = time.time() - start_time
                    print(f"  Step {step_count:5d} | "
                          f"Height {height:.3f} | "
                          f"Time {total_elapsed:.1f}s")

        print(f"\nPlayback finished. Total steps: {step_count}")

    def run_headless(self, gait: str, vx: float, vy: float, wz: float,
                     duration: float):
        """Run policy without viewer (headless mode)."""
        gait_id = GAIT_MAP[gait]
        commands = torch.tensor([[vx, vy, wz]], device=self.device)

        if gait == 'stand':
            commands.zero_()

        self.reset()

        max_steps = int(duration / self.dt)
        print(f"Running headless: {max_steps} steps")

        for step_count in range(1, max_steps + 1):
            self.step(gait_id, commands)

            height = self.base_env.base_pos[2]
            if height < 0.3:
                print(f"Robot fell at step {step_count}")
                break

            if step_count % 100 == 0:
                print(f"  Step {step_count:5d} | Height {height:.3f}")

        print(f"Headless run complete.")


def main():
    args = parse_args()

    player = PolicyPlayer(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    if args.no_viewer:
        player.run_headless(
            gait=args.gait, vx=args.vx, vy=args.vy, wz=args.wz,
            duration=args.duration)
    else:
        player.run_with_viewer(
            gait=args.gait, vx=args.vx, vy=args.vy, wz=args.wz,
            duration=args.duration)


if __name__ == '__main__':
    main()
