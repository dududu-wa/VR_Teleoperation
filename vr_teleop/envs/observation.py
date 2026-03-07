"""
Observation builder and history buffer for G1 multi-gait environment.

Actor obs (58-dim per step):
    base_ang_vel(3), projected_gravity(3), dof_pos_lower(15), dof_vel_lower(15),
    last_actions(15), commands_vel(2), command_yaw(1), gait_id(1),
    intervention_flag(1), clock_input(2)

History buffer: 5 steps of proprioception-only (51-dim = no commands/clock/flags)

Critic obs (99-dim):
    actor_single_step(58) + privileged_info(41)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ObsConfig:
    """Observation space configuration."""
    single_step_dim: int = 58
    history_obs_dim: int = 51       # proprioception only (no commands/clock/flags)
    include_history_steps: int = 5
    critic_extra_dim: int = 41
    critic_obs_dim: int = 99       # single_step_dim + critic_extra_dim
    clip_observations: float = 100.0

    # Observation scales
    scales: Dict[str, float] = field(default_factory=lambda: {
        'base_ang_vel': 0.25,
        'dof_pos': 1.0,
        'dof_vel': 0.05,
        'commands': 1.0,
        'projected_gravity': 1.0,
    })

    # Noise config
    add_noise: bool = True
    noise_level: float = 1.0
    noise_scales: Dict[str, float] = field(default_factory=lambda: {
        'base_ang_vel': 0.2,
        'dof_pos': 0.01,
        'dof_vel': 0.2,
        'projected_gravity': 0.05,
    })


class ObservationBuffer:
    """FIFO history buffer for stacking past observations.

    Stores `include_history_steps` frames of proprioceptive observations.
    On reset, the buffer is zero-padded with only the latest frame filled.
    """

    def __init__(self, num_envs: int, num_obs: int,
                 include_history_steps: int, device: torch.device):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps
        self.device = device

        # (num_envs, history_steps, obs_dim)
        self.obs_buf = torch.zeros(
            num_envs, include_history_steps, num_obs,
            device=device, dtype=torch.float)

    def reset(self, reset_idxs: torch.Tensor, new_obs: torch.Tensor):
        """Reset buffer for specific environments with zero-padding.

        Args:
            reset_idxs: (K,) indices of envs to reset
            new_obs: (K, obs_dim) current observation for reset envs
        """
        if len(reset_idxs) == 0:
            return
        self.obs_buf[reset_idxs] = 0.0
        self.obs_buf[reset_idxs, -1, :] = new_obs.clone()

    def insert(self, new_obs: torch.Tensor):
        """Shift buffer and insert new observation.

        Args:
            new_obs: (num_envs, obs_dim) new observation
        """
        # Shift left (oldest dropped)
        self.obs_buf[:, :-1, :] = self.obs_buf[:, 1:, :].clone()
        # Insert at the end (newest)
        self.obs_buf[:, -1, :] = new_obs.clone()

    def get_flattened(self) -> torch.Tensor:
        """Return flattened history: (num_envs, history_steps * obs_dim)."""
        return self.obs_buf.reshape(self.num_envs, -1)

    def get_3d(self) -> torch.Tensor:
        """Return 3D tensor: (num_envs, history_steps, obs_dim)."""
        return self.obs_buf.clone()


class ObservationBuilder:
    """Builds actor and critic observations from environment state.

    Works with batched tensors (num_envs, dim) for vectorized environments.
    """

    # Proprioception slice indices within the 58-dim actor obs
    # base_ang_vel(0:3), gravity(3:6), dof_pos(6:21), dof_vel(21:36), actions(36:51)
    # Non-proprioception: commands_vel(51:53), command_yaw(53:54), gait_id(54:55),
    #                     intervention_flag(55:56), clock(56:58)
    PROPRIO_END = 51  # first 51 dims are proprioceptive

    def __init__(self, obs_cfg: ObsConfig, num_envs: int,
                 device: torch.device, lower_body_dofs: int = 15):
        self.cfg = obs_cfg
        self.num_envs = num_envs
        self.device = device
        self.lower_body_dofs = lower_body_dofs

        # Pre-compute scale and noise tensors for actor obs (58-dim)
        self.obs_scales = self._build_obs_scales()
        self.noise_vec = self._build_noise_vec() if obs_cfg.add_noise else None

        # History buffer for proprioceptive observations
        self.history_buffer = ObservationBuffer(
            num_envs=num_envs,
            num_obs=obs_cfg.history_obs_dim,
            include_history_steps=obs_cfg.include_history_steps,
            device=device,
        )

    def _build_obs_scales(self) -> torch.Tensor:
        """Build per-element scale factors for 58-dim actor obs."""
        s = self.cfg.scales
        n = self.lower_body_dofs  # 15
        scales = torch.ones(self.cfg.single_step_dim, device=self.device)

        offset = 0
        # base_ang_vel: 3
        scales[offset:offset + 3] = s['base_ang_vel']
        offset += 3
        # projected_gravity: 3
        scales[offset:offset + 3] = s['projected_gravity']
        offset += 3
        # dof_pos_lower: 15
        scales[offset:offset + n] = s['dof_pos']
        offset += n
        # dof_vel_lower: 15
        scales[offset:offset + n] = s['dof_vel']
        offset += n
        # last_actions: 15
        scales[offset:offset + n] = 1.0
        offset += n
        # commands_vel: 2, command_yaw: 1
        scales[offset:offset + 3] = s['commands']
        offset += 3
        # gait_id: 1, intervention_flag: 1, clock: 2
        # leave as 1.0
        return scales

    def _build_noise_vec(self) -> torch.Tensor:
        """Build per-element noise std for 58-dim actor obs."""
        ns = self.cfg.noise_scales
        n = self.lower_body_dofs
        noise = torch.zeros(self.cfg.single_step_dim, device=self.device)

        offset = 0
        # base_ang_vel: 3
        noise[offset:offset + 3] = ns['base_ang_vel']
        offset += 3
        # projected_gravity: 3
        noise[offset:offset + 3] = ns['projected_gravity']
        offset += 3
        # dof_pos_lower: 15
        noise[offset:offset + n] = ns['dof_pos']
        offset += n
        # dof_vel_lower: 15
        noise[offset:offset + n] = ns['dof_vel']
        offset += n
        # remaining (actions, commands, etc): no noise
        return noise * self.cfg.noise_level

    def build_actor_obs(
        self,
        base_ang_vel: torch.Tensor,      # (N, 3) body frame
        projected_gravity: torch.Tensor,  # (N, 3)
        dof_pos_lower: torch.Tensor,      # (N, 15) relative to default
        dof_vel_lower: torch.Tensor,      # (N, 15)
        last_actions: torch.Tensor,       # (N, 15)
        commands: torch.Tensor,           # (N, 3) [vx, vy, wz]
        gait_id: torch.Tensor,            # (N, 1) or (N,)
        intervention_flag: torch.Tensor,  # (N, 1) or (N,)
        clock: torch.Tensor,             # (N, 2) [sin, cos]
    ) -> torch.Tensor:
        """Build single-step actor observation (N, 58).

        Returns scaled observation (noise added if enabled).
        """
        gait_id = gait_id.view(-1, 1) if gait_id.dim() == 1 else gait_id
        intervention_flag = intervention_flag.view(-1, 1) if intervention_flag.dim() == 1 else intervention_flag

        obs = torch.cat([
            base_ang_vel,       # 3
            projected_gravity,  # 3
            dof_pos_lower,      # 15
            dof_vel_lower,      # 15
            last_actions,       # 15
            commands,           # 3 (vx, vy, wz)
            gait_id,            # 1
            intervention_flag,  # 1
            clock,              # 2
        ], dim=-1)  # (N, 58)

        # Apply scales
        obs = obs * self.obs_scales.unsqueeze(0)

        # Add noise
        if self.noise_vec is not None and self.cfg.add_noise:
            obs = obs + torch.randn_like(obs) * self.noise_vec.unsqueeze(0)

        # Clip
        obs = torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)

        return obs

    def build_critic_obs(
        self,
        actor_obs: torch.Tensor,          # (N, 58) single step
        base_lin_vel: torch.Tensor,        # (N, 3) body frame
        base_height: torch.Tensor,         # (N, 1) or (N,)
        foot_contact_forces: torch.Tensor, # (N, 2) left, right magnitudes
        friction_coeff: torch.Tensor,      # (N, 1)
        mass_offset: torch.Tensor,         # (N, 1)
        motor_strength: torch.Tensor,      # (N, 1)
        pd_gain_mult: torch.Tensor,        # (N, 2) kp_mult, kd_mult
        upper_dof_pos: torch.Tensor,       # (N, 14)
        upper_dof_vel: torch.Tensor,       # (N, 14)
        intervention_amp: torch.Tensor,    # (N, 1)
        intervention_freq: torch.Tensor,   # (N, 1)
    ) -> torch.Tensor:
        """Build critic observation (N, 99) = actor_obs(58) + privileged(41)."""
        base_height = base_height.view(-1, 1) if base_height.dim() == 1 else base_height
        friction_coeff = friction_coeff.view(-1, 1) if friction_coeff.dim() == 1 else friction_coeff
        mass_offset = mass_offset.view(-1, 1) if mass_offset.dim() == 1 else mass_offset
        motor_strength = motor_strength.view(-1, 1) if motor_strength.dim() == 1 else motor_strength
        intervention_amp = intervention_amp.view(-1, 1) if intervention_amp.dim() == 1 else intervention_amp
        intervention_freq = intervention_freq.view(-1, 1) if intervention_freq.dim() == 1 else intervention_freq

        privileged = torch.cat([
            base_lin_vel,         # 3
            base_height,          # 1
            foot_contact_forces,  # 2
            friction_coeff,       # 1
            mass_offset,          # 1
            motor_strength,       # 1
            pd_gain_mult,         # 2
            upper_dof_pos,        # 14
            upper_dof_vel,        # 14
            intervention_amp,     # 1
            intervention_freq,    # 1
        ], dim=-1)  # (N, 41)

        critic_obs = torch.cat([actor_obs, privileged], dim=-1)  # (N, 99)
        critic_obs = torch.clamp(critic_obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        return critic_obs

    def extract_proprioception(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Extract proprioceptive components (first 51 dims) from actor obs."""
        return actor_obs[:, :self.PROPRIO_END]

    def update_history(self, actor_obs: torch.Tensor):
        """Insert current proprioception into history buffer."""
        proprio = self.extract_proprioception(actor_obs)
        self.history_buffer.insert(proprio)

    def reset_history(self, reset_idxs: torch.Tensor, actor_obs: torch.Tensor):
        """Reset history for specific envs."""
        proprio = self.extract_proprioception(actor_obs)
        self.history_buffer.reset(reset_idxs, proprio[reset_idxs])

    def get_actor_obs_with_history(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Get full actor input: current obs + flattened history.

        Returns:
            (N, 58 + 51 * 5) = (N, 313) tensor
        """
        history = self.history_buffer.get_flattened()
        return torch.cat([actor_obs, history], dim=-1)

    def get_history_3d(self) -> torch.Tensor:
        """Get 3D history for HistoryEncoder: (N, H, 51)."""
        return self.history_buffer.get_3d()
