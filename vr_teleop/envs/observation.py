"""
Observation builder and history buffer for G1 multi-gait environment.

Actor obs (67-dim per step):
    base_ang_vel(3), projected_gravity(3), dof_pos_loco(13), dof_vel_loco(13),
    last_actions(13), upper_body_pos(16), commands_vel(2), command_yaw(1),
    gait_id(1), clock_input(2)

History buffer: 5 steps of proprioception-only (61-dim = no commands/clock)

Critic obs (96-dim):
    actor_single_step(67) + privileged_info(29)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from vr_teleop.envs.dof_indices import LOCO_DOF_INDICES, VR_DOF_INDICES, NUM_LOCO_DOFS, NUM_VR_DOFS


@dataclass
class ObsConfig:
    """Observation space configuration."""
    single_step_dim: int = 67
    history_obs_dim: int = 61       # proprioception only (no commands/clock)
    include_history_steps: int = 5
    critic_extra_dim: int = 29
    critic_obs_dim: int = 96        # single_step_dim + critic_extra_dim
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

    # Proprioception slice indices within the 67-dim actor obs
    # base_ang_vel(0:3), gravity(3:6), dof_pos_loco(6:19), dof_vel_loco(19:32),
    # actions(32:45), upper_body_pos(45:61)
    # Non-proprioception: commands_vel(61:63), command_yaw(63:64), gait_id(64:65),
    #                     clock(65:67)
    PROPRIO_END = 61  # first 61 dims are proprioceptive

    def __init__(self, obs_cfg: ObsConfig, num_envs: int,
                 device: torch.device, num_loco_dofs: int = NUM_LOCO_DOFS,
                 num_vr_dofs: int = NUM_VR_DOFS):
        self.cfg = obs_cfg
        self.num_envs = num_envs
        self.device = device
        self.num_loco_dofs = num_loco_dofs
        self.num_vr_dofs = num_vr_dofs

        # Pre-compute scale and noise tensors for actor obs (67-dim)
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
        """Build per-element scale factors for 67-dim actor obs."""
        s = self.cfg.scales
        n_loco = self.num_loco_dofs   # 13
        n_vr = self.num_vr_dofs       # 16
        scales = torch.ones(self.cfg.single_step_dim, device=self.device)

        offset = 0
        # base_ang_vel: 3
        scales[offset:offset + 3] = s['base_ang_vel']
        offset += 3
        # projected_gravity: 3
        scales[offset:offset + 3] = s['projected_gravity']
        offset += 3
        # dof_pos_loco: 13
        scales[offset:offset + n_loco] = s['dof_pos']
        offset += n_loco
        # dof_vel_loco: 13
        scales[offset:offset + n_loco] = s['dof_vel']
        offset += n_loco
        # last_actions: 13
        scales[offset:offset + n_loco] = 1.0
        offset += n_loco
        # upper_body_pos: 16
        scales[offset:offset + n_vr] = s['dof_pos']
        offset += n_vr
        # commands_vel: 2, command_yaw: 1
        scales[offset:offset + 3] = s['commands']
        offset += 3
        # gait_id: 1, clock: 2
        # leave as 1.0
        return scales

    def _build_noise_vec(self) -> torch.Tensor:
        """Build per-element noise std for 67-dim actor obs."""
        ns = self.cfg.noise_scales
        n_loco = self.num_loco_dofs
        n_vr = self.num_vr_dofs
        noise = torch.zeros(self.cfg.single_step_dim, device=self.device)

        offset = 0
        # base_ang_vel: 3
        noise[offset:offset + 3] = ns['base_ang_vel']
        offset += 3
        # projected_gravity: 3
        noise[offset:offset + 3] = ns['projected_gravity']
        offset += 3
        # dof_pos_loco: 13
        noise[offset:offset + n_loco] = ns['dof_pos']
        offset += n_loco
        # dof_vel_loco: 13
        noise[offset:offset + n_loco] = ns['dof_vel']
        offset += n_loco
        # last_actions: 13 - no noise
        offset += n_loco
        # upper_body_pos: 16
        noise[offset:offset + n_vr] = ns['dof_pos']
        offset += n_vr
        # remaining (commands, etc): no noise
        return noise * self.cfg.noise_level

    def build_actor_obs(
        self,
        base_ang_vel: torch.Tensor,      # (N, 3) body frame
        projected_gravity: torch.Tensor,  # (N, 3)
        dof_pos_loco: torch.Tensor,      # (N, 13) relative to default
        dof_vel_loco: torch.Tensor,      # (N, 13)
        last_actions: torch.Tensor,       # (N, 13)
        upper_body_pos: torch.Tensor,    # (N, 16) waist_yaw/roll + arms
        commands: torch.Tensor,           # (N, 3) [vx, vy, wz]
        gait_id: torch.Tensor,            # (N, 1) or (N,)
        clock: torch.Tensor,             # (N, 2) [sin, cos]
    ) -> torch.Tensor:
        """Build single-step actor observation (N, 67).

        Returns scaled observation (noise added if enabled).
        """
        gait_id = gait_id.view(-1, 1) if gait_id.dim() == 1 else gait_id

        obs = torch.cat([
            base_ang_vel,       # 3
            projected_gravity,  # 3
            dof_pos_loco,       # 13
            dof_vel_loco,       # 13
            last_actions,       # 13
            upper_body_pos,     # 16
            commands,           # 3 (vx, vy, wz)
            gait_id,            # 1
            clock,              # 2
        ], dim=-1)  # (N, 67)

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
        actor_obs: torch.Tensor,          # (N, 67) single step
        base_lin_vel: torch.Tensor,        # (N, 3) body frame
        base_height: torch.Tensor,         # (N, 1) or (N,)
        foot_contact_forces: torch.Tensor, # (N, 2) left, right magnitudes
        friction_coeff: torch.Tensor,      # (N, 1)
        mass_offset: torch.Tensor,         # (N, 1)
        motor_strength: torch.Tensor,      # (N, 1)
        pd_gain_mult: torch.Tensor,        # (N, 2) kp_mult, kd_mult
        upper_dof_vel: torch.Tensor,       # (N, 16) waist_yaw/roll + arms velocity
        intervention_amp: torch.Tensor,    # (N, 1)
        intervention_freq: torch.Tensor,   # (N, 1)
    ) -> torch.Tensor:
        """Build critic observation (N, 96) = actor_obs(67) + privileged(29)."""
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
            upper_dof_vel,        # 16
            intervention_amp,     # 1
            intervention_freq,    # 1
        ], dim=-1)  # (N, 29)

        critic_obs = torch.cat([actor_obs, privileged], dim=-1)  # (N, 96)
        critic_obs = torch.clamp(critic_obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        return critic_obs

    def extract_proprioception(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Extract proprioceptive components (first 61 dims) from actor obs."""
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
            (N, 67 + 61 * 5) = (N, 372) tensor
        """
        history = self.history_buffer.get_flattened()
        return torch.cat([actor_obs, history], dim=-1)

    def get_history_3d(self) -> torch.Tensor:
        """Get 3D history for HistoryEncoder: (N, H, 61)."""
        return self.history_buffer.get_3d()
