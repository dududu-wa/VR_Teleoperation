"""
Policy inference wrapper for deployment.

Wraps a trained ActorCritic checkpoint into a stateful inference module
that maintains observation history and gait clock internally.
Suitable for real-time deployment on real robot or sim2sim.

Interface:
    wrapper = PolicyWrapper.from_checkpoint('model.pt')
    wrapper.reset()
    actions = wrapper.get_action(obs_dict)  # returns (15,) numpy
"""

import torch
import numpy as np
from typing import Dict, Optional

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.envs.observation import ObservationBuilder, ObsConfig
from vr_teleop.agents.actor_critic import ActorCritic
from vr_teleop.utils.math_utils import (
    compute_projected_gravity, quat_rotate_inverse, get_euler_xyz
)


class PolicyWrapper:
    """Stateful inference wrapper for trained locomotion policy.

    Manages observation history, gait clock, and provides a simple
    dict-in / numpy-out interface for deployment.
    """

    def __init__(self, actor_critic: ActorCritic,
                 robot_cfg: G1Config = None,
                 obs_cfg: ObsConfig = None,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.robot_cfg = robot_cfg or G1Config()
        self.obs_cfg = obs_cfg or ObsConfig()

        self.actor_critic = actor_critic.to(self.device)
        self.actor_critic.eval()

        # Observation builder (single env)
        self.obs_builder = ObservationBuilder(
            obs_cfg=self.obs_cfg,
            num_envs=1,
            device=self.device,
            lower_body_dofs=self.robot_cfg.lower_body_dofs,
        )

        # Default DOF positions
        self.default_dof_pos = self.robot_cfg.get_default_dof_pos().to(
            self.device).unsqueeze(0)

        # State
        self.last_actions = torch.zeros(
            1, self.robot_cfg.lower_body_dofs, device=self.device)
        self.gait_phase = 0.0
        self.walk_freq = 2.0
        self.run_freq = 3.0
        self.dt = 0.02  # policy timestep (5ms sim * 4 decimation)

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str,
        robot_cfg: G1Config = None,
        obs_cfg: ObsConfig = None,
        device: str = 'cpu',
    ) -> 'PolicyWrapper':
        """Create wrapper from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            robot_cfg: Robot configuration
            obs_cfg: Observation configuration
            device: Torch device

        Returns:
            PolicyWrapper instance
        """
        robot_cfg = robot_cfg or G1Config()
        obs_cfg = obs_cfg or ObsConfig()

        num_actor_obs = obs_cfg.single_step_dim + \
            obs_cfg.history_obs_dim * obs_cfg.include_history_steps
        num_critic_obs = obs_cfg.critic_obs_dim
        num_actions = robot_cfg.lower_body_dofs

        actor_critic = ActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
        )

        ckpt = torch.load(checkpoint_path, map_location=device)
        actor_critic.load_state_dict(ckpt['model_state_dict'])

        return cls(actor_critic, robot_cfg, obs_cfg, device)

    def reset(self):
        """Reset internal state (call at start of each episode)."""
        self.last_actions.zero_()
        self.gait_phase = 0.0
        # Reset history with zeros
        zero_obs = torch.zeros(
            1, self.obs_cfg.single_step_dim, device=self.device)
        env_ids = torch.tensor([0], device=self.device)
        self.obs_builder.reset_history(env_ids, zero_obs)

    @torch.inference_mode()
    def get_action(self, obs_dict: Dict[str, np.ndarray],
                   gait_id: int = 1,
                   commands: np.ndarray = None) -> np.ndarray:
        """Get policy action from observation dict.

        Args:
            obs_dict: Dictionary with keys:
                - 'base_quat_xyzw': (4,) base orientation [x,y,z,w]
                - 'base_ang_vel': (3,) base angular velocity (world frame)
                - 'dof_pos': (29,) joint positions
                - 'dof_vel': (29,) joint velocities
            gait_id: 0=stand, 1=walk, 2=run
            commands: (3,) [vx, vy, wz] velocity commands

        Returns:
            (15,) lower body joint action targets (action-scale space)
        """
        if commands is None:
            commands = np.zeros(3, dtype=np.float32)

        # Convert to tensors
        quat_xyzw = torch.tensor(
            obs_dict['base_quat_xyzw'], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        base_ang_vel = torch.tensor(
            obs_dict['base_ang_vel'], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        dof_pos = torch.tensor(
            obs_dict['dof_pos'], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        dof_vel = torch.tensor(
            obs_dict['dof_vel'], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        cmd = torch.tensor(
            commands, dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        # Derived quantities
        base_ang_vel_body = quat_rotate_inverse(quat_xyzw, base_ang_vel)
        projected_gravity = compute_projected_gravity(quat_xyzw)

        # Relative DOF positions
        dof_pos_rel = dof_pos - self.default_dof_pos
        lower_dof_pos = dof_pos_rel[:, :self.robot_cfg.lower_body_dofs]
        lower_dof_vel = dof_vel[:, :self.robot_cfg.lower_body_dofs]

        # Gait clock
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
        actor_obs = self.obs_builder.build_actor_obs(
            base_ang_vel=base_ang_vel_body,
            projected_gravity=projected_gravity,
            dof_pos_lower=lower_dof_pos,
            dof_vel_lower=lower_dof_vel,
            last_actions=self.last_actions,
            commands=cmd,
            gait_id=torch.tensor([float(gait_id)], device=self.device),
            intervention_flag=intervention_flag,
            clock=clock,
        )
        self.obs_builder.update_history(actor_obs)
        full_obs = self.obs_builder.get_actor_obs_with_history(actor_obs)

        # Inference
        actions_mean, _ = self.actor_critic.act_inference(full_obs)

        actions = actions_mean.squeeze(0)
        self.last_actions[0] = actions

        return actions.cpu().numpy()

    def get_action_from_flat_obs(self, flat_obs: np.ndarray) -> np.ndarray:
        """Get action from pre-built flat observation tensor.

        Args:
            flat_obs: (num_actor_obs,) pre-built observation

        Returns:
            (15,) lower body joint actions
        """
        obs = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(
            self.device)
        actions_mean, _ = self.actor_critic.act_inference(obs)
        actions = actions_mean.squeeze(0)
        self.last_actions[0] = actions
        return actions.cpu().numpy()

