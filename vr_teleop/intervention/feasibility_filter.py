"""
Feasibility filter for upper-body intervention signals.

Ensures intervention targets are physically feasible before being applied
to the robot, preventing damage and improving training stability.

Filter pipeline:
  1. Joint position limit clipping
  2. Velocity rate limiting (max joint velocity per step)
  3. Exponential lowpass smoothing
  4. Torso orientation safety check (rollback if torso tilts too much)
"""

import torch
from dataclasses import dataclass
from typing import Tuple

from vr_teleop.robot.g1_config import G1Config


@dataclass
class FeasibilityConfig:
    """Configuration for feasibility filter."""
    enable: bool = True
    torso_roll_limit: float = 0.3    # rad
    torso_pitch_limit: float = 0.5   # rad
    max_rate: float = 5.0            # rad/s per joint
    max_jerk: float = 50.0           # rad/s^2 per joint (unused currently)
    lowpass_alpha: float = 0.3       # exponential smoothing factor (0=ignore new, 1=no smooth)


class FeasibilityFilter:
    """Filters intervention targets for physical feasibility.

    Applied as a post-processing step on intervention generator output
    before sending to the environment.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: FeasibilityConfig = None,
        robot_cfg: G1Config = None,
        device: torch.device = None,
    ):
        self.cfg = cfg or FeasibilityConfig()
        self.robot_cfg = robot_cfg or G1Config()
        self.num_envs = num_envs
        self.device = device or torch.device('cpu')
        self.upper_dim = self.robot_cfg.upper_body_dofs  # 14

        # Joint limits for upper body
        pos_lower, pos_upper = self.robot_cfg.get_pos_limits()
        self.upper_pos_lower = pos_lower[self.robot_cfg.upper_body_indices].to(self.device)
        self.upper_pos_upper = pos_upper[self.robot_cfg.upper_body_indices].to(self.device)
        self.upper_default = self.robot_cfg.get_default_dof_pos()[
            self.robot_cfg.upper_body_indices
        ].to(self.device)

        # Velocity limits for upper body
        vel_limits = torch.tensor(self.robot_cfg.dof_vel_limit, dtype=torch.float32)
        self.upper_vel_limit = vel_limits[self.robot_cfg.upper_body_indices].to(self.device)

        # Smoothed output buffer
        self.smoothed_targets = torch.zeros(
            num_envs, self.upper_dim, device=self.device)
        # Previous targets for rate limiting
        self.prev_targets = torch.zeros(
            num_envs, self.upper_dim, device=self.device)

    def reset(self, env_ids: torch.Tensor, current_pos: torch.Tensor = None):
        """Reset filter state for specified environments.

        Args:
            env_ids: (K,) environment indices
            current_pos: (K, 14) current upper body positions (absolute)
        """
        if len(env_ids) == 0:
            return
        if current_pos is not None:
            self.smoothed_targets[env_ids] = current_pos
            self.prev_targets[env_ids] = current_pos
        else:
            self.smoothed_targets[env_ids] = self.upper_default.unsqueeze(0)
            self.prev_targets[env_ids] = self.upper_default.unsqueeze(0)

    def filter(
        self,
        raw_targets: torch.Tensor,
        current_upper_pos: torch.Tensor,
        torso_euler: torch.Tensor,
        dt: float,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply feasibility filter pipeline.

        Args:
            raw_targets: (N, 14) raw intervention targets (absolute joint pos)
            current_upper_pos: (N, 14) current upper body positions
            torso_euler: (N, 3) torso roll/pitch/yaw in radians
            dt: Policy timestep
            mask: (N,) bool, which envs have active intervention

        Returns:
            filtered_targets: (N, 14) filtered absolute joint positions
            safety_mask: (N,) bool, updated mask (False if safety triggered)
        """
        if not self.cfg.enable:
            return raw_targets, mask if mask is not None else torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device)

        targets = raw_targets.clone()

        # 1. Joint limit clipping
        targets = self._clip_joint_limits(targets)

        # 2. Rate limiting
        targets = self._rate_limit(targets, dt)

        # 3. Lowpass smoothing
        targets = self._lowpass_smooth(targets)

        # 4. Torso safety check
        safety_mask = self._check_torso_safety(torso_euler)
        if mask is not None:
            safety_mask = safety_mask & mask

        # Apply safety: revert unsafe envs to current position
        unsafe = ~safety_mask
        if unsafe.any():
            targets[unsafe] = current_upper_pos[unsafe]

        # Update state
        self.prev_targets = self.smoothed_targets.clone()
        self.smoothed_targets = targets.clone()

        return targets, safety_mask

    def _clip_joint_limits(self, targets: torch.Tensor) -> torch.Tensor:
        """Clip targets to joint position limits with small margin."""
        margin = 0.02  # small safety margin (rad)
        lower = self.upper_pos_lower.unsqueeze(0) + margin
        upper = self.upper_pos_upper.unsqueeze(0) - margin
        return torch.clamp(targets, lower, upper)

    def _rate_limit(self, targets: torch.Tensor, dt: float) -> torch.Tensor:
        """Limit per-step change based on max joint velocity."""
        max_delta = self.cfg.max_rate * dt  # max change per step
        delta = targets - self.smoothed_targets
        delta = torch.clamp(delta, -max_delta, max_delta)
        return self.smoothed_targets + delta

    def _lowpass_smooth(self, targets: torch.Tensor) -> torch.Tensor:
        """Exponential moving average smoothing."""
        alpha = self.cfg.lowpass_alpha
        return alpha * targets + (1.0 - alpha) * self.smoothed_targets

    def _check_torso_safety(self, torso_euler: torch.Tensor) -> torch.Tensor:
        """Check if torso orientation is within safe limits.

        Args:
            torso_euler: (N, 3) [roll, pitch, yaw]

        Returns:
            (N,) bool mask - True if safe
        """
        roll_ok = torch.abs(torso_euler[:, 0]) < self.cfg.torso_roll_limit
        pitch_ok = torch.abs(torso_euler[:, 1]) < self.cfg.torso_pitch_limit
        return roll_ok & pitch_ok

    def action_scale_to_absolute(
        self, action_targets: torch.Tensor, action_scale: float
    ) -> torch.Tensor:
        """Convert action-scale targets to absolute joint positions.

        Args:
            action_targets: (N, 14) targets in action-scale (relative to default)
            action_scale: Action scaling factor

        Returns:
            (N, 14) absolute joint positions
        """
        return action_targets * action_scale + self.upper_default.unsqueeze(0)

    def absolute_to_action_scale(
        self, abs_targets: torch.Tensor, action_scale: float
    ) -> torch.Tensor:
        """Convert absolute joint positions to action-scale targets.

        Args:
            abs_targets: (N, 14) absolute joint positions
            action_scale: Action scaling factor

        Returns:
            (N, 14) action-scale targets
        """
        return (abs_targets - self.upper_default.unsqueeze(0)) / action_scale
