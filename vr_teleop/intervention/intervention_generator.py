"""
Upper-body intervention signal generator for G1 multi-gait training.

Generates 14-DOF (left arm 7 + right arm 7) intervention targets that
perturb the robot's upper body during training, building robustness
for real VR teleoperation.

Modes:
  - gaussian: Gaussian noise around current joint positions
  - uniform: Uniform random within joint-limit-scaled range
  - sinusoidal: Time-varying sinusoidal sweeps
  - structured: Coordinated arm motions (arm swing, turning gesture)
  - replay: Replay from pre-recorded motion library (Phase 4+)

The generator manages:
  - Per-env duty cycle (on/off toggle with switch_prob)
  - Phase-specific amplitude/frequency ranges
  - Transition stress amplification during gait transitions
  - Curriculum blending (fusion / clip_mean / clip_mean_rad)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vr_teleop.robot.g1_config import G1Config


@dataclass
class InterventionConfig:
    """Configuration for intervention signal generation."""
    use_disturb: bool = True
    disturb_dim: int = 14  # upper body DOFs

    # Enabled modes
    gaussian: bool = True
    uniform: bool = True
    sinusoidal: bool = True
    structured: bool = True
    replay: bool = False

    # Per-step switch probability for duty cycle toggle
    switch_prob: float = 0.01

    # Noise resample interval (steps)
    noise_update_step: int = 10

    # Curriculum blending method: 0=fusion, 1=clip_mean, 2=clip_mean_rad
    curriculum_method: int = 2

    # Phase-specific params (amp_range, freq_range, duty_range, transition_stress, hold_T)
    phase_params: Dict[str, dict] = field(default_factory=lambda: {
        'phase_0': {'amp': [0.0, 0.0], 'freq': [0.0, 0.0], 'duty': [0.0, 0.0],
                    'transition_stress': 1.0, 'hold_T': [3.0, 5.0]},
        'phase_1': {'amp': [0.05, 0.2], 'freq': [0.3, 1.0], 'duty': [0.1, 0.3],
                    'transition_stress': 1.0, 'hold_T': [1.0, 3.0]},
        'phase_2': {'amp': [0.1, 0.5], 'freq': [0.5, 2.0], 'duty': [0.2, 0.5],
                    'transition_stress': 1.5, 'hold_T': [0.5, 2.0]},
        'phase_3': {'amp': [0.2, 1.0], 'freq': [0.5, 3.0], 'duty': [0.5, 0.8],
                    'transition_stress': 2.0, 'hold_T': [0.3, 1.5]},
        'phase_4': {'amp': [0.2, 1.0], 'freq': [0.5, 3.0], 'duty': [0.5, 0.8],
                    'transition_stress': 2.0, 'hold_T': [0.3, 1.5],
                    'replay_mixing_ratio': 0.5},
    })

    # Action scale (must match env)
    action_scale: float = 0.25

    # Max curriculum level
    max_curriculum: float = 1.0


class InterventionGenerator:
    """Generates batched upper-body intervention signals for training.

    Integrates with G1MultigaitEnv via set_intervention() interface.
    """

    def __init__(
        self,
        num_envs: int,
        cfg: InterventionConfig = None,
        robot_cfg: G1Config = None,
        device: torch.device = None,
    ):
        self.cfg = cfg or InterventionConfig()
        self.robot_cfg = robot_cfg or G1Config()
        self.num_envs = num_envs
        self.device = device or torch.device('cpu')
        self.disturb_dim = self.cfg.disturb_dim  # 14

        # Joint limits for upper body (indices 15-28 in full 29-DOF)
        pos_lower, pos_upper = self.robot_cfg.get_pos_limits()
        self.upper_pos_lower = pos_lower[self.robot_cfg.upper_body_indices].to(self.device)  # (14,)
        self.upper_pos_upper = pos_upper[self.robot_cfg.upper_body_indices].to(self.device)  # (14,)
        self.upper_default = self.robot_cfg.get_default_dof_pos()[
            self.robot_cfg.upper_body_indices
        ].to(self.device)  # (14,)

        # ---- State buffers ----
        # Intervention targets (relative to default, in action-scale units)
        self.disturb_actions = torch.zeros(
            num_envs, self.disturb_dim, device=self.device)
        # On/off mask per env
        self.disturb_masks = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device)
        # Interrupt mask (exposed to env for reward masking)
        self.interrupt_mask = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device)

        # Per-env mode selection: 0=gaussian, 1=uniform, 2=sinusoidal, 3=structured
        self.mode_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # Curriculum factor per env (0=no disturb, 1=full disturb)
        self.curriculum_factor = torch.zeros(num_envs, device=self.device)

        # Current phase parameters (set by set_phase)
        self.current_phase = 0
        self._amp_range = [0.0, 0.0]
        self._freq_range = [0.0, 0.0]
        self._duty_range = [0.0, 0.0]
        self._transition_stress = 1.0
        self._hold_T_range = [3.0, 5.0]
        self._replay_mixing = 0.0

        # Per-env sampled parameters
        self.env_amp = torch.zeros(num_envs, device=self.device)
        self.env_freq = torch.zeros(num_envs, device=self.device)
        self.env_duty = torch.zeros(num_envs, device=self.device)
        self.env_hold_timer = torch.zeros(num_envs, device=self.device)

        # Sinusoidal phase accumulators
        self.sin_phase = torch.zeros(num_envs, self.disturb_dim, device=self.device)
        self.sin_freq = torch.zeros(num_envs, self.disturb_dim, device=self.device)

        # Step counter for resample interval
        self.step_count = 0

        # Build available mode list
        self._available_modes = []
        if self.cfg.gaussian:
            self._available_modes.append(0)
        if self.cfg.uniform:
            self._available_modes.append(1)
        if self.cfg.sinusoidal:
            self._available_modes.append(2)
        if self.cfg.structured:
            self._available_modes.append(3)
        if not self._available_modes:
            self._available_modes = [0]  # fallback to gaussian
        self._mode_tensor = torch.tensor(
            self._available_modes, dtype=torch.long, device=self.device)

    def set_phase(self, phase: int):
        """Update intervention parameters for a curriculum phase."""
        self.current_phase = phase
        phase_key = f'phase_{phase}'
        params = self.cfg.phase_params.get(
            phase_key, self.cfg.phase_params['phase_0'])

        self._amp_range = params['amp']
        self._freq_range = params['freq']
        self._duty_range = params['duty']
        self._transition_stress = params.get('transition_stress', 1.0)
        self._hold_T_range = params.get('hold_T', [3.0, 5.0])
        self._replay_mixing = params.get('replay_mixing_ratio', 0.0)

    def reset(self, env_ids: torch.Tensor, current_upper_pos: torch.Tensor = None):
        """Reset intervention state for specified environments.

        Args:
            env_ids: (K,) environment indices to reset
            current_upper_pos: (K, 14) current upper body joint positions (absolute)
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)

        # Reset masks: 50% start with intervention active
        self.disturb_masks[env_ids] = torch.rand(n, device=self.device) < 0.5
        self.interrupt_mask[env_ids] = self.disturb_masks[env_ids]

        # Sample initial intervention targets (zero-centered)
        if current_upper_pos is not None:
            self.disturb_actions[env_ids] = (
                current_upper_pos - self.upper_default.unsqueeze(0)
            ) / self.cfg.action_scale
        else:
            self.disturb_actions[env_ids] = 0.0

        # Sample per-env mode
        mode_idx = torch.randint(0, len(self._available_modes), (n,), device=self.device)
        self.mode_ids[env_ids] = self._mode_tensor[mode_idx]

        # Sample per-env params
        self.env_amp[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self._amp_range)
        self.env_freq[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self._freq_range)
        self.env_duty[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self._duty_range)

        # Sample hold timers
        self.env_hold_timer[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self._hold_T_range)

        # Reset sinusoidal phase
        self.sin_phase[env_ids] = torch.rand(
            n, self.disturb_dim, device=self.device) * 2 * np.pi
        self.sin_freq[env_ids] = torch.empty(
            n, self.disturb_dim, device=self.device).uniform_(
            max(self._freq_range[0], 0.1), max(self._freq_range[1], 0.5))

        # Reset curriculum factor
        self.curriculum_factor[env_ids] = 0.0

    def step(
        self,
        dt: float,
        current_upper_pos: torch.Tensor,
        policy_lower_actions: torch.Tensor,
        transition_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate intervention signals for one time step.

        Args:
            dt: Policy timestep (seconds)
            current_upper_pos: (N, 14) current upper body positions (absolute)
            policy_lower_actions: (N, 15) policy output (for curriculum blending)
            transition_mask: (N,) bool, envs currently in gait transition

        Returns:
            upper_actions: (N, 14) upper body joint targets (action-scale)
            mask: (N,) bool, active intervention mask
            amp: (N,) per-env amplitude (for critic obs)
            freq: (N,) per-env frequency (for critic obs)
        """
        if not self.cfg.use_disturb or self._amp_range[1] <= 0:
            return (
                torch.zeros(self.num_envs, self.disturb_dim, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )

        # 1. Toggle duty cycle
        self._toggle_duty_cycle()

        # 2. Resample signals periodically
        self.step_count += 1
        if self.step_count % self.cfg.noise_update_step == 0:
            self._resample_signals(current_upper_pos)
            self.step_count = 0

        # 3. Advance sinusoidal phase
        self.sin_phase += self.sin_freq * dt * 2 * np.pi
        self.sin_phase = self.sin_phase % (2 * np.pi)

        # 4. Apply transition stress amplification
        effective_amp = self.env_amp.clone()
        if transition_mask is not None:
            effective_amp[transition_mask] *= self._transition_stress

        # 5. Blend with curriculum
        upper_actions = self._apply_curriculum(
            policy_lower_actions, effective_amp)

        # 6. Mask inactive envs
        inactive = ~self.interrupt_mask
        upper_actions[inactive] = 0.0

        return (
            upper_actions,
            self.interrupt_mask.clone(),
            effective_amp,
            self.env_freq.clone(),
        )

    def update_curriculum(self, env_ids: torch.Tensor, delta: float = 0.05):
        """Update curriculum factor for environments (called by curriculum system).

        Args:
            env_ids: (K,) environments to update
            delta: Increment per call (+/-)
        """
        if len(env_ids) == 0:
            return
        self.curriculum_factor[env_ids] = (
            self.curriculum_factor[env_ids] + delta
        ).clamp(0.0, self.cfg.max_curriculum)

    def set_curriculum_factor(self, env_ids: torch.Tensor, value: float):
        """Directly set curriculum factor for environments."""
        self.curriculum_factor[env_ids] = value

    # ---- Private methods ----

    def _toggle_duty_cycle(self):
        """Randomly toggle intervention on/off per env."""
        switch_rand = torch.rand(self.num_envs, device=self.device)
        switch = switch_rand < self.cfg.switch_prob
        self.disturb_masks = torch.where(switch, ~self.disturb_masks, self.disturb_masks)
        self.interrupt_mask[:] = self.disturb_masks

    def _resample_signals(self, current_upper_pos: torch.Tensor):
        """Resample intervention targets for active environments."""
        active = self.disturb_masks
        if not active.any():
            return

        active_ids = active.nonzero(as_tuple=False).squeeze(-1)
        n = len(active_ids)

        # Per-mode generation
        gaussian_mask = self.mode_ids[active_ids] == 0
        uniform_mask = self.mode_ids[active_ids] == 1
        sinusoidal_mask = self.mode_ids[active_ids] == 2
        structured_mask = self.mode_ids[active_ids] == 3

        # Gaussian mode
        if gaussian_mask.any():
            g_ids = active_ids[gaussian_mask]
            targets = self._generate_gaussian(g_ids, current_upper_pos[g_ids])
            self.disturb_actions[g_ids] = targets

        # Uniform mode
        if uniform_mask.any():
            u_ids = active_ids[uniform_mask]
            targets = self._generate_uniform(u_ids)
            self.disturb_actions[u_ids] = targets

        # Sinusoidal mode: updated continuously in _apply_curriculum
        # (targets derived from sin_phase in real-time)
        if sinusoidal_mask.any():
            s_ids = active_ids[sinusoidal_mask]
            targets = self._generate_sinusoidal(s_ids)
            self.disturb_actions[s_ids] = targets

        # Structured mode
        if structured_mask.any():
            st_ids = active_ids[structured_mask]
            targets = self._generate_structured(st_ids, current_upper_pos[st_ids])
            self.disturb_actions[st_ids] = targets

    def _generate_gaussian(
        self, env_ids: torch.Tensor, current_pos: torch.Tensor
    ) -> torch.Tensor:
        """Gaussian noise around current joint positions.

        Returns: (K, 14) action-scale targets
        """
        n = len(env_ids)
        amp = self.env_amp[env_ids].unsqueeze(-1)  # (K, 1)

        noise = torch.randn(n, self.disturb_dim, device=self.device) * amp
        targets_abs = current_pos + noise

        # Clip to joint limits
        targets_abs = torch.clamp(
            targets_abs,
            self.upper_pos_lower.unsqueeze(0),
            self.upper_pos_upper.unsqueeze(0),
        )

        # Convert to action-scale (relative to default)
        return (targets_abs - self.upper_default.unsqueeze(0)) / self.cfg.action_scale

    def _generate_uniform(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Uniform random within scaled joint range.

        Returns: (K, 14) action-scale targets
        """
        n = len(env_ids)
        amp = self.env_amp[env_ids].unsqueeze(-1)  # (K, 1)

        # Sample uniformly in [lower + margin, upper - margin]
        joint_range = self.upper_pos_upper - self.upper_pos_lower  # (14,)
        center = (self.upper_pos_upper + self.upper_pos_lower) / 2.0  # (14,)

        # Scale range by amplitude
        half_range = joint_range / 2.0 * amp  # (K, 14)
        targets_abs = center.unsqueeze(0) + (
            2.0 * torch.rand(n, self.disturb_dim, device=self.device) - 1.0
        ) * half_range

        # Clip to limits
        targets_abs = torch.clamp(
            targets_abs,
            self.upper_pos_lower.unsqueeze(0),
            self.upper_pos_upper.unsqueeze(0),
        )

        return (targets_abs - self.upper_default.unsqueeze(0)) / self.cfg.action_scale

    def _generate_sinusoidal(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Time-varying sinusoidal sweep.

        Returns: (K, 14) action-scale targets
        """
        amp = self.env_amp[env_ids].unsqueeze(-1)  # (K, 1)
        phase = self.sin_phase[env_ids]  # (K, 14)

        # Sinusoidal offset from default
        joint_range = self.upper_pos_upper - self.upper_pos_lower  # (14,)
        offset = torch.sin(phase) * joint_range.unsqueeze(0) / 2.0 * amp

        targets_abs = self.upper_default.unsqueeze(0) + offset

        # Clip
        targets_abs = torch.clamp(
            targets_abs,
            self.upper_pos_lower.unsqueeze(0),
            self.upper_pos_upper.unsqueeze(0),
        )

        return (targets_abs - self.upper_default.unsqueeze(0)) / self.cfg.action_scale

    def _generate_structured(
        self, env_ids: torch.Tensor, current_pos: torch.Tensor
    ) -> torch.Tensor:
        """Structured coordinated arm motions.

        Patterns:
        - Arm swing: opposing shoulder pitch oscillation (natural gait)
        - Turning gesture: asymmetric shoulder roll/yaw

        Returns: (K, 14) action-scale targets
        """
        n = len(env_ids)
        amp = self.env_amp[env_ids].unsqueeze(-1)

        targets = torch.zeros(n, self.disturb_dim, device=self.device)

        # Split into two sub-patterns
        arm_swing_mask = torch.rand(n, device=self.device) < 0.5

        # ---- Arm swing pattern ----
        if arm_swing_mask.any():
            swing_ids = arm_swing_mask.nonzero(as_tuple=False).squeeze(-1)
            phase = self.sin_phase[env_ids[swing_ids], 0]  # use first channel

            # Left shoulder pitch (index 0 in upper body = joint 15)
            targets[swing_ids, 0] = torch.sin(phase) * amp[swing_ids, 0] * 0.5
            # Right shoulder pitch (index 7 in upper body = joint 22)
            targets[swing_ids, 7] = -torch.sin(phase) * amp[swing_ids, 0] * 0.5

            # Left elbow (index 3) and right elbow (index 10) for natural motion
            targets[swing_ids, 3] = torch.cos(phase) * amp[swing_ids, 0] * 0.3
            targets[swing_ids, 10] = -torch.cos(phase) * amp[swing_ids, 0] * 0.3

        # ---- Turning gesture pattern ----
        turning_ids = (~arm_swing_mask).nonzero(as_tuple=False).squeeze(-1)
        if len(turning_ids) > 0:
            phase = self.sin_phase[env_ids[turning_ids], 1]

            # Asymmetric shoulder roll: left up, right down (or vice versa)
            targets[turning_ids, 1] = torch.sin(phase) * amp[turning_ids, 0] * 0.4
            targets[turning_ids, 8] = -torch.sin(phase) * amp[turning_ids, 0] * 0.4

            # Shoulder yaw for reaching gesture
            targets[turning_ids, 2] = torch.cos(phase) * amp[turning_ids, 0] * 0.3
            targets[turning_ids, 9] = torch.cos(phase) * amp[turning_ids, 0] * 0.3

        # Convert amplitude to absolute positions, then to action scale
        targets_offset = targets  # already in joint-angle offset from default
        targets_abs = self.upper_default.unsqueeze(0) + targets_offset

        targets_abs = torch.clamp(
            targets_abs,
            self.upper_pos_lower.unsqueeze(0),
            self.upper_pos_upper.unsqueeze(0),
        )

        return (targets_abs - self.upper_default.unsqueeze(0)) / self.cfg.action_scale

    def _apply_curriculum(
        self,
        policy_actions: torch.Tensor,
        effective_amp: torch.Tensor,
    ) -> torch.Tensor:
        """Apply curriculum blending to intervention targets.

        Blending methods:
          0 (fusion): linear interpolation between policy upper-body and disturb
          1 (clip_mean): curriculum shifts center toward current joint pos
          2 (clip_mean_rad): curriculum scales both center and radius

        Returns: (N, 14) blended upper-body actions
        """
        method = self.cfg.curriculum_method
        curriculum = self.curriculum_factor  # (N,)

        if method == 0:
            return self._curriculum_fusion(policy_actions, curriculum)
        elif method == 1:
            return self._curriculum_clip_mean(policy_actions, curriculum)
        else:
            return self._curriculum_clip_mean_rad(policy_actions, curriculum)

    def _curriculum_fusion(
        self, policy_actions: torch.Tensor, curriculum: torch.Tensor
    ) -> torch.Tensor:
        """Fusion: linear blend between policy output and disturb target."""
        # Policy doesn't produce upper body actions, so use zero as policy's upper output
        policy_upper = torch.zeros(
            self.num_envs, self.disturb_dim, device=self.device)

        # Clamp disturb actions
        disturb = self._clamp_disturb_to_limits()

        fused = (
            curriculum.unsqueeze(-1) * disturb
            + (1.0 - curriculum.unsqueeze(-1)) * policy_upper
        )
        return fused

    def _curriculum_clip_mean(
        self, policy_actions: torch.Tensor, curriculum: torch.Tensor
    ) -> torch.Tensor:
        """Clip mean: curriculum shifts the clipping center."""
        # Upper body default-relative position as noise center
        noise_mean = curriculum.unsqueeze(-1) * self.disturb_actions

        # Max deviation from mean (using amp)
        amp = self.env_amp.unsqueeze(-1)  # (N, 1)
        disturb = torch.clamp(
            self.disturb_actions,
            noise_mean - amp / self.cfg.action_scale,
            noise_mean + amp / self.cfg.action_scale,
        )
        return disturb

    def _curriculum_clip_mean_rad(
        self, policy_actions: torch.Tensor, curriculum: torch.Tensor
    ) -> torch.Tensor:
        """Clip mean + radius: curriculum scales both center and clipping radius."""
        c = curriculum.unsqueeze(-1)  # (N, 1)
        noise_mean = c * self.disturb_actions

        amp = self.env_amp.unsqueeze(-1)
        radius = amp * c / self.cfg.action_scale

        disturb = torch.clamp(
            self.disturb_actions,
            noise_mean - radius,
            noise_mean + radius,
        )
        return disturb

    def _clamp_disturb_to_limits(self) -> torch.Tensor:
        """Clamp disturb_actions to joint limits in action-scale space."""
        lower_action = (
            self.upper_pos_lower.unsqueeze(0) - self.upper_default.unsqueeze(0)
        ) / self.cfg.action_scale
        upper_action = (
            self.upper_pos_upper.unsqueeze(0) - self.upper_default.unsqueeze(0)
        ) / self.cfg.action_scale
        return torch.clamp(self.disturb_actions, lower_action, upper_action)
