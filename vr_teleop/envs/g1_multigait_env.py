"""
G1 Multi-Gait Training Environment.

Top-level training environment that wraps a vectorized simulator backend
(IsaacGym-only for train/eval; MuJoCo is only for post-training playback) and provides:
- Gait command sampling (stand/walk/run)
- Velocity command sampling per gait mode
- Gait phase clock (sin/cos for periodic locomotion)
- Observation building (actor + critic)
- Reward computation
- Termination checking
- Episode management with auto-reset
- Intervention injection interface (upper body)

Implements the VecEnv interface expected by PPO runner:
    step(actions) -> (obs, privileged_obs, rewards, resets, extras)
"""

import torch
import numpy as np
from typing import Tuple, Union, Optional, Dict

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.envs.dof_indices import LOCO_DOF_INDICES, VR_DOF_INDICES, NUM_LOCO_DOFS
from vr_teleop.envs.observation import ObservationBuilder, ObsConfig
from vr_teleop.envs.reward import RewardComputer, RewardConfig
from vr_teleop.envs.termination import TerminationChecker, TerminationConfig
from vr_teleop.envs.domain_rand import DomainRandConfig
from vr_teleop.utils.math_utils import wrap_to_pi


class G1MultigaitEnv:
    """Multi-gait training environment for G1 robot.

    Provides the VecEnv interface for PPO training.
    """

    # Gait IDs
    STAND = 0
    WALK = 1
    RUN = 2

    def __init__(
        self,
        num_envs: int,
        robot_cfg: G1Config = None,
        obs_cfg: ObsConfig = None,
        reward_cfg: RewardConfig = None,
        term_cfg: TerminationConfig = None,
        rand_cfg: DomainRandConfig = None,
        sim_dt: float = 0.005,
        decimation: int = 4,
        max_episode_length: int = 1000,
        device: str = 'cuda:0',
        model_path: str = None,
        sim_backend: str = "isaacgym",
        # Command config
        gait_probs: list = None,
        command_ranges: dict = None,
        resampling_time: float = 10.0,
        # Gait clock config
        walk_freq: float = 2.0,
        run_freq: float = 3.0,
        phase_offset: float = 0.5,
    ):
        self.robot_cfg = robot_cfg or G1Config()
        self.obs_cfg = obs_cfg or ObsConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.term_cfg = term_cfg or TerminationConfig(episode_length=max_episode_length)
        self.rand_cfg = rand_cfg or DomainRandConfig()

        self.num_envs = num_envs
        self.dt = sim_dt * decimation
        self.max_episode_length = max_episode_length
        self.device = torch.device(device)
        self.sim_backend = sim_backend.lower()

        # Action dimensions
        self.num_actions = NUM_LOCO_DOFS  # 13 (12 legs + waist_pitch)
        self.num_obs = self.obs_cfg.single_step_dim + self.obs_cfg.history_obs_dim * self.obs_cfg.include_history_steps  # 58 + 51*5 = 313
        self.num_privileged_obs = self.obs_cfg.critic_obs_dim  # 99

        # ---- Create underlying vectorized environment ----
        self.vec_env = self._build_vec_env(
            num_envs=num_envs,
            sim_dt=sim_dt,
            decimation=decimation,
            max_episode_length=max_episode_length,
            device=device,
            model_path=model_path,
        )

        # ---- Observation builder ----
        self.obs_builder = ObservationBuilder(
            obs_cfg=self.obs_cfg,
            num_envs=num_envs,
            device=self.device,
            num_loco_dofs=NUM_LOCO_DOFS,
        )

        # ---- Reward computer ----
        self.reward_computer = RewardComputer(
            reward_cfg=self.reward_cfg,
            dt=self.dt,
            device=self.device,
            num_envs=num_envs,
        )

        # ---- Termination checker ----
        self.term_checker = TerminationChecker(
            term_cfg=self.term_cfg,
            device=self.device,
        )

        # ---- Command state ----
        self.gait_probs = gait_probs or [0.15, 0.55, 0.30]
        self.resampling_time = resampling_time
        self.resampling_steps = int(resampling_time / self.dt)

        # Default command ranges per gait
        self.command_ranges = command_ranges or {
            'stand': {'vx': [0.0, 0.0], 'vy': [0.0, 0.0], 'wz': [0.0, 0.0]},
            'walk': {'vx': [-0.6, 0.6], 'vy': [-0.3, 0.3], 'wz': [-0.5, 0.5]},
            'run': {'vx': [0.6, 2.0], 'vy': [-0.3, 0.3], 'wz': [-0.5, 0.5]},
        }

        # Gait and command buffers
        self.gait_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.commands = torch.zeros(num_envs, 3, device=self.device)  # [vx, vy, wz]
        self.command_timer = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # ---- Gait phase clock ----
        self.walk_freq = walk_freq
        self.run_freq = run_freq
        self.phase_offset = phase_offset  # trot: left/right 0.5 phase offset
        self.gait_phase = torch.zeros(num_envs, device=self.device)  # [0, 1)
        self.clock_input = torch.zeros(num_envs, 2, device=self.device)  # [sin, cos]

        # ---- Gait transition tracking ----
        self.prev_gait_id = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.transition_timer = torch.zeros(num_envs, device=self.device)
        self.transition_window = self.reward_cfg.transition_window

        # ---- Intervention state ----
        self.intervention_generator = None
        self.feasibility_filter = None
        self.auto_intervention = False
        self.motion_retargeter = None
        self.interrupt_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.intervention_flag = torch.zeros(num_envs, device=self.device)
        self.intervention_amp = torch.zeros(num_envs, device=self.device)
        self.intervention_freq = torch.zeros(num_envs, device=self.device)
        self.upper_body_actions = torch.zeros(num_envs, len(VR_DOF_INDICES), device=self.device)

        # Pre-compute index tensors on device for fast slicing
        self._loco_indices = torch.tensor(LOCO_DOF_INDICES, dtype=torch.long, device=self.device)
        self._vr_indices = torch.tensor(VR_DOF_INDICES, dtype=torch.long, device=self.device)

        # ---- Output buffers ----
        self.obs_buf = torch.zeros(num_envs, self.num_obs, device=self.device)
        self.privileged_obs_buf = torch.zeros(num_envs, self.num_privileged_obs, device=self.device)
        self.rew_buf = torch.zeros(num_envs, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.extras = {}

        # ---- Episode statistics ----
        self.episode_reward_sums = torch.zeros(num_envs, device=self.device)
        self.reward_components = {}

        # Per-component episode sums for accurate curriculum metrics
        self.episode_component_sums = {}

        # ---- Previous action for second-order action rate ----
        self.last_last_actions = torch.zeros(num_envs, self.num_actions, device=self.device)

    def _build_vec_env(
        self,
        num_envs: int,
        sim_dt: float,
        decimation: int,
        max_episode_length: int,
        device: str,
        model_path: Optional[str],
    ):
        """Instantiate vectorized simulator backend."""
        common_kwargs = dict(
            num_envs=num_envs,
            robot_cfg=self.robot_cfg,
            domain_rand_cfg=self.rand_cfg,
            sim_dt=sim_dt,
            decimation=decimation,
            max_episode_length=max_episode_length,
            device=device,
            model_path=model_path,
        )

        if self.sim_backend == "isaacgym":
            from vr_teleop.envs.isaac_vec_env import IsaacVecEnv
            return IsaacVecEnv(**common_kwargs)

        raise ValueError(
            f"Unsupported sim_backend '{self.sim_backend}'. "
            "Expected: ['isaacgym']"
        )

    def reset_all(self):
        """Reset all environments and return initial observations."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(env_ids)
        self._compute_observations()
        return self.obs_buf

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one policy step.

        Args:
            actions: (num_envs, 13) locomotion actions from policy

        Returns:
            (obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras)
        """
        # Store for second-order action rate
        self.last_last_actions[:] = self.vec_env.last_actions.clone()

        # Auto-generate upper-body intervention targets if enabled.
        if self.auto_intervention and self.intervention_generator is not None:
            self._update_intervention(actions)

        # Set upper body actions (from intervention or zeros)
        self.vec_env.set_upper_body_actions(self.upper_body_actions)

        # Step physics
        self.vec_env.step(actions)

        # Update episode length
        self.episode_length_buf += 1

        # Update gait phase clock
        self._update_gait_clock()

        # Resample commands if needed
        self._maybe_resample_commands()

        # Compute observations
        self._compute_observations()

        # Check terminations first (rewards need termination info)
        self._check_terminations()

        # Compute rewards (uses termination result for penalty)
        self._compute_rewards()

        # Accumulate episode rewards
        self.episode_reward_sums += self.rew_buf

        # Accumulate per-component sums for episode-level metrics
        for name, vals in self.reward_components.items():
            if name != 'total':
                if name not in self.episode_component_sums:
                    self.episode_component_sums[name] = torch.zeros(
                        self.num_envs, device=self.device)
                self.episode_component_sums[name] += vals

        # Update transition timer
        self._update_transition_state()

        # Handle resets
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._log_episode_info(reset_ids)
            self._reset_envs(reset_ids)
            # Recompute buffers without pushing history a second time.
            self._compute_observations(update_history=False)

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _reset_envs(self, env_ids: torch.Tensor):
        """Reset specified environments."""
        if len(env_ids) == 0:
            return

        # Reset underlying physics
        self.vec_env.reset(env_ids)

        # Resample gait and commands
        self._resample_gait(env_ids)
        self._resample_commands(env_ids)

        # Reset gait phase
        self.gait_phase[env_ids] = 0.0
        self._update_clock_for_envs(env_ids)

        # Reset episode state
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.episode_reward_sums[env_ids] = 0.0
        for sums in self.episode_component_sums.values():
            sums[env_ids] = 0.0
        self.command_timer[env_ids] = 0

        # Reset transition state
        self.prev_gait_id[env_ids] = self.gait_id[env_ids]
        self.transition_timer[env_ids] = 0.0

        # Reset feet air time tracking in reward computer
        self.reward_computer.reset_air_time(env_ids)

        # Reset intervention
        self.interrupt_mask[env_ids] = False
        self.intervention_flag[env_ids] = 0.0
        self.intervention_amp[env_ids] = 0.0
        self.intervention_freq[env_ids] = 0.0
        self.upper_body_actions[env_ids] = 0.0

        # Reset intervention modules with current upper-body pose.
        current_upper_pos = self.vec_env.dof_pos[env_ids][:, self._vr_indices]
        if self.intervention_generator is not None:
            self.intervention_generator.reset(env_ids, current_upper_pos=current_upper_pos)
        if self.feasibility_filter is not None:
            self.feasibility_filter.reset(env_ids, current_pos=current_upper_pos)

        # Reset action buffer
        self.last_last_actions[env_ids] = 0.0

        # Reset observation history for these envs
        # Build initial actor obs and use it to reset history
        actor_obs = self._build_actor_obs_single_step()
        self.obs_builder.reset_history(env_ids, actor_obs)

    def _resample_gait(self, env_ids: torch.Tensor):
        """Sample gait modes for environments."""
        n = len(env_ids)
        probs = torch.tensor(self.gait_probs, device=self.device)
        gait_ids = torch.multinomial(probs.unsqueeze(0).expand(n, -1), 1).squeeze(-1)
        self.gait_id[env_ids] = gait_ids

    def _resample_commands(self, env_ids: torch.Tensor):
        """Sample velocity commands based on current gait mode."""
        n = len(env_ids)
        if n == 0:
            return

        for gait_name, gait_idx in [('stand', 0), ('walk', 1), ('run', 2)]:
            mask = self.gait_id[env_ids] == gait_idx
            gait_env_ids = env_ids[mask]
            if len(gait_env_ids) == 0:
                continue

            ranges = self.command_ranges[gait_name]
            k = len(gait_env_ids)
            self.commands[gait_env_ids, 0] = torch.empty(k, device=self.device).uniform_(*ranges['vx'])
            self.commands[gait_env_ids, 1] = torch.empty(k, device=self.device).uniform_(*ranges['vy'])
            self.commands[gait_env_ids, 2] = torch.empty(k, device=self.device).uniform_(*ranges['wz'])

        # Zero out small commands
        cmd_mag = torch.norm(self.commands[env_ids, :2], dim=-1)
        small_cmd = cmd_mag < 0.1
        self.commands[env_ids[small_cmd], :2] = 0.0
        small_yaw = torch.abs(self.commands[env_ids, 2]) < 0.05
        self.commands[env_ids[small_yaw], 2] = 0.0

        # Reset command timer
        self.command_timer[env_ids] = 0

    def _maybe_resample_commands(self):
        """Resample commands when timer expires."""
        self.command_timer += 1
        resample_mask = self.command_timer >= self.resampling_steps
        resample_ids = resample_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            # Possibly change gait too
            self.prev_gait_id[resample_ids] = self.gait_id[resample_ids]
            self._resample_gait(resample_ids)
            self._resample_commands(resample_ids)
            # Mark as in gait transition
            gait_changed = self.gait_id[resample_ids] != self.prev_gait_id[resample_ids]
            self.transition_timer[resample_ids[gait_changed]] = self.transition_window

    def _update_gait_clock(self):
        """Advance gait phase clock based on current gait mode."""
        # Frequency per gait: stand=0, walk=walk_freq, run=run_freq
        freq = torch.zeros(self.num_envs, device=self.device)
        freq[self.gait_id == self.WALK] = self.walk_freq
        freq[self.gait_id == self.RUN] = self.run_freq

        # Advance phase
        self.gait_phase += freq * self.dt
        self.gait_phase = self.gait_phase % 1.0  # wrap to [0, 1)

        # Compute clock inputs (sin/cos of 2*pi*phase)
        phase_rad = 2.0 * np.pi * self.gait_phase
        self.clock_input[:, 0] = torch.sin(phase_rad)
        self.clock_input[:, 1] = torch.cos(phase_rad)

        # For standing: clock is zero
        stand_mask = self.gait_id == self.STAND
        self.clock_input[stand_mask] = 0.0

    def _update_clock_for_envs(self, env_ids: torch.Tensor):
        """Update clock input for specific environments."""
        phase_rad = 2.0 * np.pi * self.gait_phase[env_ids]
        self.clock_input[env_ids, 0] = torch.sin(phase_rad)
        self.clock_input[env_ids, 1] = torch.cos(phase_rad)
        stand_mask = self.gait_id[env_ids] == self.STAND
        self.clock_input[env_ids[stand_mask]] = 0.0

    def _update_transition_state(self):
        """Update gait transition timer."""
        self.transition_timer = (self.transition_timer - self.dt).clamp(min=0.0)

    def _build_actor_obs_single_step(self) -> torch.Tensor:
        """Build single-step actor observation (N, 67)."""
        # Locomotion DOF positions and velocities (13 each)
        dof_pos_rel = self.vec_env.get_dof_pos_relative()
        lower_dof_pos = dof_pos_rel[:, self._loco_indices]
        lower_dof_vel = self.vec_env.dof_vel[:, self._loco_indices]
        upper_body_pos = dof_pos_rel[:, self._vr_indices]

        return self.obs_builder.build_actor_obs(
            base_ang_vel=self.vec_env.base_ang_vel_body,
            projected_gravity=self.vec_env.projected_gravity,
            dof_pos_loco=lower_dof_pos,
            dof_vel_loco=lower_dof_vel,
            last_actions=self.vec_env.last_actions,
            upper_body_pos=upper_body_pos,
            commands=self.commands,
            gait_id=self.gait_id.float(),
            clock=self.clock_input,
        )

    def _compute_observations(self, update_history: bool = True):
        """Compute actor and critic observations."""
        if hasattr(self.vec_env, "set_command_context"):
            self.vec_env.set_command_context(
                commands=self.commands,
                gait_id=self.gait_id.float(),
                intervention_flag=self.intervention_flag,
                clock=self.clock_input,
                intervention_amp=self.intervention_amp,
                intervention_freq=self.intervention_freq,
            )

        # Single-step actor obs
        actor_obs = self._build_actor_obs_single_step()

        # Update history buffer
        if update_history:
            self.obs_builder.update_history(actor_obs)

        # Full actor obs = current + history
        self.obs_buf = self.obs_builder.get_actor_obs_with_history(actor_obs)

        # Critic obs (privileged)
        self.privileged_obs_buf = self.obs_builder.build_critic_obs(
            actor_obs=actor_obs,
            base_lin_vel=self.vec_env.base_lin_vel_body,
            base_height=self.vec_env.get_base_height(),
            foot_contact_forces=self.vec_env.foot_contact_forces,
            friction_coeff=self.vec_env.friction_coeffs,
            mass_offset=self.vec_env.mass_offsets,
            motor_strength=self.vec_env.motor_strengths,
            pd_gain_mult=torch.stack([
                self.vec_env.kp_multipliers,
                self.vec_env.kd_multipliers
            ], dim=-1),
            upper_dof_vel=self.vec_env.dof_vel[:, self._vr_indices],
            intervention_amp=self.intervention_amp,
            intervention_freq=self.intervention_freq,
        )

    def _update_intervention(self, policy_lower_actions: torch.Tensor):
        """Generate and apply upper-body intervention for current step."""
        transition_mask = self.transition_timer > 0.0
        current_upper_pos = self.vec_env.dof_pos[:, self._vr_indices]

        upper_actions, mask, amp, freq = self.intervention_generator.step(
            dt=self.dt,
            current_upper_pos=current_upper_pos,
            policy_lower_actions=policy_lower_actions,
            transition_mask=transition_mask,
        )

        if self.feasibility_filter is not None:
            raw_abs = self.feasibility_filter.action_scale_to_absolute(
                upper_actions, self.robot_cfg.action_scale)
            filtered_abs, safety_mask = self.feasibility_filter.filter(
                raw_targets=raw_abs,
                current_upper_pos=current_upper_pos,
                torso_euler=self.vec_env.base_euler,
                dt=self.dt,
                mask=mask,
            )
            upper_actions = self.feasibility_filter.absolute_to_action_scale(
                filtered_abs, self.robot_cfg.action_scale)
            mask = safety_mask

        self.upper_body_actions = upper_actions
        self.interrupt_mask = mask
        self.intervention_flag = mask.float()
        self.intervention_amp = amp * mask.float()
        self.intervention_freq = freq * mask.float()

    def _compute_rewards(self):
        """Compute all reward components and total reward."""
        is_timeout = self.episode_length_buf >= self.max_episode_length
        transition_mask = self.transition_timer > 0.0

        # Use masked termination signals so penalty matches actual reset logic
        # (contacts during intervention are masked and should not be penalized)
        term = self.extras.get('termination', {})
        is_terminated = (
            term.get('contact', torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            | term.get('orientation', torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
            | term.get('height', torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        )

        self.reward_components = self.reward_computer.compute_all(
            commands=self.commands,
            base_lin_vel=self.vec_env.base_lin_vel_body,
            base_ang_vel=self.vec_env.base_ang_vel_body,
            projected_gravity=self.vec_env.projected_gravity,
            base_height=self.vec_env.get_base_height(),
            gait_id=self.gait_id,
            actions=self.vec_env.actions,
            last_actions=self.vec_env.last_actions,
            last_last_actions=self.last_last_actions,
            torques=self.vec_env.torques[:, self._loco_indices],
            dof_vel=self.vec_env.dof_vel[:, self._loco_indices],
            last_dof_vel=self.vec_env.last_dof_vel[:, self._loco_indices],
            foot_contact_forces=self.vec_env.foot_contact_forces,
            foot_contact_forces_3d=self.vec_env.foot_contact_forces_3d,
            foot_velocities=self.vec_env.foot_velocities,
            is_terminated=is_terminated,
            is_timed_out=is_timeout,
            transition_mask=transition_mask,
            interrupt_mask=self.interrupt_mask,
        )

        self.rew_buf = self.reward_components['total']

    def _check_terminations(self):
        """Check termination conditions."""
        term_result = self.term_checker.check(
            base_euler=self.vec_env.base_euler,
            base_height=self.vec_env.get_base_height(),
            has_contact_termination=self.vec_env.has_contact_termination,
            episode_length_buf=self.episode_length_buf,
            interrupt_mask=self.interrupt_mask,
        )
        term_result['in_transition'] = self.transition_timer > 0.0
        self.reset_buf = term_result['reset']
        self.extras['termination'] = term_result
        # PPO expects 'time_outs' key for timeout bootstrapping
        self.extras['time_outs'] = term_result['timeout']

    def _log_episode_info(self, reset_ids: torch.Tensor):
        """Log episode statistics for environments being reset."""
        ep_lengths = self.episode_length_buf[reset_ids].float().clamp(min=1)
        self.extras['episode'] = {
            'reward_mean': self.episode_reward_sums[reset_ids].mean().item(),
            'episode_length_mean': ep_lengths.mean().item(),
        }
        # Log per-component episode averages (not single-step snapshots)
        for name, sums in self.episode_component_sums.items():
            avg = sums[reset_ids] / ep_lengths
            self.extras['episode'][f'reward_{name}'] = avg.mean().item()

        # Log termination types
        if 'termination' in self.extras:
            term = self.extras['termination']
            height = term.get('height', torch.zeros_like(term['contact']))
            fell = term['contact'] | term['orientation'] | height
            self.extras['episode']['timeout_rate'] = term['timeout'][reset_ids].float().mean().item()
            self.extras['episode']['contact_fall_rate'] = term['contact'][reset_ids].float().mean().item()
            self.extras['episode']['orientation_fall_rate'] = term['orientation'][reset_ids].float().mean().item()
            self.extras['episode']['height_fall_rate'] = height[reset_ids].float().mean().item()
            self.extras['episode']['fall_rate'] = fell[reset_ids].float().mean().item()
            in_transition = term.get('in_transition', torch.zeros_like(fell))
            transition_failure = fell & in_transition
            self.extras['episode']['transition_failure_rate'] = (
                transition_failure[reset_ids].float().mean().item()
            )

    # ---- Intervention interface ----
    def attach_intervention(self, generator=None, feasibility_filter=None,
                            auto_mode: bool = True):
        """Attach intervention modules and optionally enable auto updates."""
        self.intervention_generator = generator
        self.feasibility_filter = feasibility_filter
        self.auto_intervention = bool(auto_mode and generator is not None)

        if generator is not None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            current_upper_pos = self.vec_env.dof_pos[:, self._vr_indices]
            generator.reset(env_ids, current_upper_pos=current_upper_pos)
            self.interrupt_mask = generator.interrupt_mask.clone()
            self.intervention_flag = self.interrupt_mask.float()
        if feasibility_filter is not None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            current_upper_pos = self.vec_env.dof_pos[:, self._vr_indices]
            feasibility_filter.reset(env_ids, current_pos=current_upper_pos)

    def set_intervention(self, upper_actions: torch.Tensor,
                         mask: torch.Tensor,
                         amp: torch.Tensor = None,
                         freq: torch.Tensor = None):
        """Set upper body intervention signals.

        Args:
            upper_actions: (N, 16) upper body joint targets (VR DOFs)
            mask: (N,) bool, which envs have active intervention
            amp: (N,) intervention amplitude (for critic obs)
            freq: (N,) intervention frequency (for critic obs)
        """
        self.upper_body_actions = upper_actions
        self.interrupt_mask = mask
        self.intervention_flag = mask.float()
        if amp is not None:
            self.intervention_amp = amp
        else:
            self.intervention_amp = torch.zeros_like(self.intervention_amp)
        if freq is not None:
            self.intervention_freq = freq
        else:
            self.intervention_freq = torch.zeros_like(self.intervention_freq)

    def set_hand_intervention_targets(
        self,
        left_hand_pos: torch.Tensor,
        left_hand_quat_wxyz: torch.Tensor,
        right_hand_pos: torch.Tensor,
        right_hand_quat_wxyz: torch.Tensor,
        mask: torch.Tensor,
        amp: torch.Tensor = None,
        freq: torch.Tensor = None,
    ):
        """Set upper-body intervention via hand pose targets.

        This is the preferred interface for VR teleoperation:
            hand target pose -> retarget + IK -> upper joint targets.
        """
        retargeter = self._get_motion_retargeter()

        base_pos = self.vec_env.base_pos.detach().cpu().numpy()
        base_quat_xyzw = self.vec_env.base_quat_xyzw.detach().cpu().numpy()
        base_quat_wxyz = np.concatenate(
            [base_quat_xyzw[:, 3:4], base_quat_xyzw[:, 0:3]], axis=-1)

        upper_abs = retargeter.retarget(
            left_hand_pos=left_hand_pos.detach().cpu().numpy(),
            left_hand_quat=left_hand_quat_wxyz.detach().cpu().numpy(),
            right_hand_pos=right_hand_pos.detach().cpu().numpy(),
            right_hand_quat=right_hand_quat_wxyz.detach().cpu().numpy(),
            base_pos=base_pos,
            base_quat_wxyz=base_quat_wxyz,
            mask=mask.detach().cpu().numpy().astype(bool),
        )

        upper_abs_t = torch.from_numpy(upper_abs).float().to(self.device)

        # Optional safety filtering in joint space.
        if self.feasibility_filter is not None:
            current_upper_pos = self.vec_env.dof_pos[:, self._vr_indices]
            upper_abs_t, safety_mask = self.feasibility_filter.filter(
                raw_targets=upper_abs_t,
                current_upper_pos=current_upper_pos,
                torso_euler=self.vec_env.base_euler,
                dt=self.dt,
                mask=mask,
            )
            mask = safety_mask

        upper_actions = self.feasibility_filter.absolute_to_action_scale(
            upper_abs_t, self.robot_cfg.action_scale
        ) if self.feasibility_filter is not None else (
            (upper_abs_t - self.vec_env.default_dof_pos[:, self._vr_indices])
            / self.robot_cfg.action_scale
        )

        self.set_intervention(
            upper_actions=upper_actions,
            mask=mask,
            amp=amp,
            freq=freq,
        )

    def _get_motion_retargeter(self):
        """Create motion retargeter lazily."""
        if self.motion_retargeter is None:
            from vr_teleop.intervention.motion_retarget import MotionRetargeter
            if not hasattr(self.vec_env, "_envs"):
                raise RuntimeError(
                    "MotionRetargeter currently requires MuJoCo-compatible backend "
                    "with per-env `MjModel/MjData` handles."
                )
            mj_models = [e.mj_model for e in self.vec_env._envs]
            mj_datas = [e.mj_data for e in self.vec_env._envs]
            self.motion_retargeter = MotionRetargeter(
                mj_models=mj_models,
                mj_datas=mj_datas,
                robot_cfg=self.robot_cfg,
                device=self.device,
            )
        return self.motion_retargeter

    # ---- Curriculum interface ----

    def update_command_ranges(self, new_ranges: dict):
        """Update velocity command ranges (called by curriculum)."""
        self.command_ranges.update(new_ranges)

    def update_gait_probs(self, probs: list):
        """Update gait sampling probabilities (called by curriculum)."""
        self.gait_probs = probs

    # ---- VecEnv interface ----

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor:
        return self.privileged_obs_buf

