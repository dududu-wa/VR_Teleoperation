"""
Reward functions for G1 multi-gait training.

13 reward components per the config:
  - tracking_lin_vel, tracking_ang_vel, alive  (positive)
  - torso_orientation, ang_vel_xy, base_height, action_rate,
    action_rate_second_order, torques, dof_acc, foot_slip,
    feet_contact_forces, transition_stability, standing_still,
    termination  (penalties)

All reward functions accept batched tensors (num_envs,) and return (num_envs,).
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RewardConfig:
    """Reward weights and parameters loaded from g1_rewards.yaml."""
    weights: Dict[str, float] = field(default_factory=lambda: {
        'tracking_lin_vel': 2.0,
        'tracking_ang_vel': 3.0,
        'alive': 0.2,
        'torso_orientation': -20.0,
        'ang_vel_xy': -0.5,
        'base_height': -40.0,
        'action_rate': -0.01,
        'action_rate_second_order': -0.005,
        'torques': -5.0e-6,
        'dof_acc': -2.5e-7,
        'foot_slip': -0.2,
        'feet_contact_forces': -0.2,
        'transition_stability': -5.0,
        'standing_still': -10.0,
        'termination': -200.0,
    })

    # Parameters
    tracking_sigma: float = 0.25
    base_height_target_stand: float = 0.793
    base_height_target_walk: float = 0.75
    base_height_target_run: float = 0.72
    max_contact_force: float = 500.0
    transition_window: float = 0.5  # seconds around gait transition


class RewardComputer:
    """Computes all reward components for vectorized environments.

    All methods operate on batched tensors of shape (num_envs, ...).
    """

    def __init__(self, reward_cfg: RewardConfig, dt: float, device: torch.device):
        self.cfg = reward_cfg
        self.dt = dt
        self.device = device

        # Pre-compute height targets as tensor for indexing by gait_id
        # gait_id: 0=stand, 1=walk, 2=run
        self.height_targets = torch.tensor([
            reward_cfg.base_height_target_stand,
            reward_cfg.base_height_target_walk,
            reward_cfg.base_height_target_run,
        ], device=device, dtype=torch.float32)

    def compute_all(
        self,
        # Velocity tracking
        commands: torch.Tensor,           # (N, 3) [vx, vy, wz]
        base_lin_vel: torch.Tensor,       # (N, 3) body frame
        base_ang_vel: torch.Tensor,       # (N, 3) body frame
        # Orientation
        projected_gravity: torch.Tensor,  # (N, 3)
        # Height
        base_height: torch.Tensor,        # (N,)
        gait_id: torch.Tensor,            # (N,) int, 0/1/2
        # Actions
        actions: torch.Tensor,            # (N, num_actions)
        last_actions: torch.Tensor,       # (N, num_actions)
        last_last_actions: torch.Tensor,  # (N, num_actions)
        # Torques and DOF
        torques: torch.Tensor,            # (N, num_dofs)
        dof_vel: torch.Tensor,            # (N, num_dofs)
        last_dof_vel: torch.Tensor,       # (N, num_dofs)
        # Foot
        foot_contact_forces: torch.Tensor,  # (N, 2) left, right magnitudes
        foot_velocities: torch.Tensor,      # (N, 2, 3) left, right lin velocities
        # Episode state
        is_terminated: torch.Tensor,      # (N,) bool
        is_timed_out: torch.Tensor,       # (N,) bool
        # Gait transition
        transition_mask: Optional[torch.Tensor] = None,  # (N,) bool, within transition window
        # Intervention
        interrupt_mask: Optional[torch.Tensor] = None,   # (N,) bool, interrupted envs
    ) -> Dict[str, torch.Tensor]:
        """Compute all reward components.

        Returns:
            Dict mapping reward name -> (N,) tensor of individual rewards.
            Also includes 'total' key with weighted sum.
        """
        rewards = {}

        # --- Positive rewards ---
        rewards['tracking_lin_vel'] = self._tracking_lin_vel(
            commands[:, :2], base_lin_vel[:, :2])
        rewards['tracking_ang_vel'] = self._tracking_ang_vel(
            commands[:, 2], base_ang_vel[:, 2])
        rewards['alive'] = torch.ones(commands.shape[0], device=self.device)

        # --- Penalties ---
        rewards['torso_orientation'] = self._torso_orientation(projected_gravity)
        rewards['ang_vel_xy'] = self._ang_vel_xy(base_ang_vel)
        rewards['base_height'] = self._base_height(base_height, gait_id)
        rewards['action_rate'] = self._action_rate(actions, last_actions)
        rewards['action_rate_second_order'] = self._action_rate_second_order(
            actions, last_actions, last_last_actions)
        rewards['torques'] = self._torques(torques)
        rewards['dof_acc'] = self._dof_acc(dof_vel, last_dof_vel)
        rewards['foot_slip'] = self._foot_slip(
            foot_contact_forces, foot_velocities)
        rewards['feet_contact_forces'] = self._feet_contact_forces(
            foot_contact_forces)
        rewards['standing_still'] = self._standing_still(
            commands, dof_vel, actions)

        # Transition stability (only penalize near gait transitions)
        if transition_mask is not None:
            rewards['transition_stability'] = self._transition_stability(
                commands, base_lin_vel, transition_mask)
        else:
            rewards['transition_stability'] = torch.zeros(
                commands.shape[0], device=self.device)

        # Termination penalty (not applied to timeouts)
        rewards['termination'] = is_terminated.float() * (~is_timed_out).float()

        # --- Interrupt masking ---
        # Some penalties are relaxed for interrupted environments
        if interrupt_mask is not None:
            mask_float = (~interrupt_mask).float()
            rewards['action_rate'] = rewards['action_rate'] * mask_float
            rewards['action_rate_second_order'] = (
                rewards['action_rate_second_order'] * mask_float)

        # --- Weighted sum ---
        total = torch.zeros(commands.shape[0], device=self.device)
        for name, value in rewards.items():
            weight = self.cfg.weights.get(name, 0.0)
            total += weight * value

        rewards['total'] = total
        return rewards

    # ---- Individual reward functions ----

    def _tracking_lin_vel(self, cmd_vel_xy: torch.Tensor,
                          actual_vel_xy: torch.Tensor) -> torch.Tensor:
        """exp(-||cmd_xy - actual_xy||^2 / sigma^2)"""
        sigma = self.cfg.tracking_sigma
        error = torch.sum(torch.square(cmd_vel_xy - actual_vel_xy), dim=-1)
        return torch.exp(-error / (sigma ** 2))

    def _tracking_ang_vel(self, cmd_wz: torch.Tensor,
                          actual_wz: torch.Tensor) -> torch.Tensor:
        """exp(-(cmd_wz - actual_wz)^2 / sigma^2)"""
        sigma = self.cfg.tracking_sigma
        error = torch.square(cmd_wz - actual_wz)
        return torch.exp(-error / (sigma ** 2))

    def _torso_orientation(self, projected_gravity: torch.Tensor) -> torch.Tensor:
        """Penalize torso tilt: ||gravity_xy||^2"""
        return torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)

    def _ang_vel_xy(self, base_ang_vel: torch.Tensor) -> torch.Tensor:
        """Penalize roll/pitch angular velocity: ||omega_xy||^2"""
        return torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)

    def _base_height(self, base_height: torch.Tensor,
                     gait_id: torch.Tensor) -> torch.Tensor:
        """Penalize deviation from gait-specific target height."""
        gait_id_clamped = gait_id.clamp(0, 2).long()
        target = self.height_targets[gait_id_clamped]
        return torch.square(base_height - target)

    def _action_rate(self, actions: torch.Tensor,
                     last_actions: torch.Tensor) -> torch.Tensor:
        """Penalize change in actions: ||a_t - a_{t-1}||^2"""
        return torch.sum(torch.square(actions - last_actions), dim=-1)

    def _action_rate_second_order(self, actions: torch.Tensor,
                                  last_actions: torch.Tensor,
                                  last_last_actions: torch.Tensor) -> torch.Tensor:
        """Penalize action acceleration: ||a_t - 2*a_{t-1} + a_{t-2}||^2"""
        return torch.sum(
            torch.square(actions - 2 * last_actions + last_last_actions), dim=-1)

    def _torques(self, torques: torch.Tensor) -> torch.Tensor:
        """Penalize large torques: sum(tau^2)"""
        return torch.sum(torch.square(torques), dim=-1)

    def _dof_acc(self, dof_vel: torch.Tensor,
                 last_dof_vel: torch.Tensor) -> torch.Tensor:
        """Penalize joint accelerations: sum((dq_t - dq_{t-1})^2 / dt^2)"""
        return torch.sum(
            torch.square((dof_vel - last_dof_vel) / self.dt), dim=-1)

    def _foot_slip(self, foot_contact_forces: torch.Tensor,
                   foot_velocities: torch.Tensor) -> torch.Tensor:
        """Penalize foot sliding when in contact.

        Args:
            foot_contact_forces: (N, 2) force magnitudes
            foot_velocities: (N, 2, 3) linear velocities of foot bodies
        """
        # Foot in contact if force > threshold
        contact = (foot_contact_forces > 1.0).float()  # (N, 2)
        # Horizontal velocity magnitude for each foot
        vel_xy = torch.norm(foot_velocities[:, :, :2], dim=-1)  # (N, 2)
        return torch.sum(contact * vel_xy, dim=-1)

    def _feet_contact_forces(self, foot_contact_forces: torch.Tensor) -> torch.Tensor:
        """Penalize excessive contact forces: sum(max(F - F_max, 0)^2)"""
        excess = (foot_contact_forces - self.cfg.max_contact_force).clamp(min=0.0)
        return torch.sum(torch.square(excess), dim=-1)

    def _transition_stability(self, commands: torch.Tensor,
                              base_lin_vel: torch.Tensor,
                              transition_mask: torch.Tensor) -> torch.Tensor:
        """Extra penalty for tracking error during gait transitions."""
        error = torch.sum(
            torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=-1)
        return error * transition_mask.float()

    def _standing_still(self, commands: torch.Tensor,
                        dof_vel: torch.Tensor,
                        actions: torch.Tensor) -> torch.Tensor:
        """Penalize motion when standing (commands near zero).

        Only active when commanded velocity is near zero.
        """
        # Standing = all velocity commands near zero
        cmd_mag = torch.norm(commands, dim=-1)
        is_standing = (cmd_mag < 0.1).float()  # (N,)

        # Penalize joint velocities + actions when standing
        motion_penalty = (
            torch.sum(torch.abs(dof_vel), dim=-1) * 0.1 +
            torch.sum(torch.abs(actions), dim=-1) * 0.5
        )
        return motion_penalty * is_standing
