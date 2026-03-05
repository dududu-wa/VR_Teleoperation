"""
G1 Multi-Gait Training Environment.

Top-level training environment that wraps MujocoVecEnv and provides:
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
from vr_teleop.envs.mujoco_vec_env import MujocoVecEnv
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

        # Action dimensions
        self.num_actions = self.robot_cfg.lower_body_dofs  # 15
        self.num_obs = self.obs_cfg.single_step_dim + self.obs_cfg.history_obs_dim * self.obs_cfg.include_history_steps  # 58 + 51*5 = 313
        self.num_privileged_obs = self.obs_cfg.critic_obs_dim  # 99

        # ---- Create underlying vectorized environment ----
        self.vec_env = MujocoVecEnv(
            num_envs=num_envs,
            robot_cfg=self.robot_cfg,
            domain_rand_cfg=self.rand_cfg,
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
            lower_body_dofs=self.robot_cfg.lower_body_dofs,
        )

        # ---- Reward computer ----
        self.reward_computer = RewardComputer(
            reward_cfg=self.reward_cfg,
            dt=self.dt,
            device=self.device,
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

        # ---- Intervention state (placeholder - fully implemented in Wave 5) ----
        self.interrupt_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.intervention_flag = torch.zeros(num_envs, device=self.device)
        self.intervention_amp = torch.zeros(num_envs, device=self.device)
        self.intervention_freq = torch.zeros(num_envs, device=self.device)
        self.upper_body_actions = torch.zeros(num_envs, self.robot_cfg.upper_body_dofs, device=self.device)

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

        # ---- Previous action for second-order action rate ----
        self.last_last_actions = torch.zeros(num_envs, self.num_actions, device=self.device)

    def reset_all(self):
        """Reset all environments and return initial observations."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_envs(env_ids)
        self._compute_observations()
        return self.obs_buf

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one policy step.

        Args:
            actions: (num_envs, 15) lower body actions from policy

        Returns:
            (obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras)
        """
        # Store for second-order action rate
        self.last_last_actions[:] = self.vec_env.last_actions.clone()

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

        # Compute rewards
        self._compute_rewards()

        # Check terminations
        self._check_terminations()

        # Accumulate episode rewards
        self.episode_reward_sums += self.rew_buf

        # Update transition timer
        self._update_transition_state()

        # Handle resets
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._log_episode_info(reset_ids)
            self._reset_envs(reset_ids)
            # Recompute observations for reset envs
            self._compute_observations()

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
        self.command_timer[env_ids] = 0

        # Reset transition state
        self.prev_gait_id[env_ids] = self.gait_id[env_ids]
        self.transition_timer[env_ids] = 0.0

        # Reset intervention
        self.interrupt_mask[env_ids] = False
        self.intervention_flag[env_ids] = 0.0
        self.upper_body_actions[env_ids] = 0.0

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
        """Build single-step actor observation (N, 58)."""
        # Lower body relative DOF positions
        dof_pos_rel = self.vec_env.get_dof_pos_relative()
        lower_dof_pos = dof_pos_rel[:, :self.robot_cfg.lower_body_dofs]
        lower_dof_vel = self.vec_env.dof_vel[:, :self.robot_cfg.lower_body_dofs]

        return self.obs_builder.build_actor_obs(
            base_ang_vel=self.vec_env.base_ang_vel_body,
            projected_gravity=self.vec_env.projected_gravity,
            dof_pos_lower=lower_dof_pos,
            dof_vel_lower=lower_dof_vel,
            last_actions=self.vec_env.last_actions,
            commands=self.commands,
            gait_id=self.gait_id.float(),
            intervention_flag=self.intervention_flag,
            clock=self.clock_input,
        )

    def _compute_observations(self):
        """Compute actor and critic observations."""
        # Single-step actor obs
        actor_obs = self._build_actor_obs_single_step()

        # Update history buffer
        self.obs_builder.update_history(actor_obs)

        # Full actor obs = current + history
        self.obs_buf = self.obs_builder.get_actor_obs_with_history(actor_obs)

        # Critic obs (privileged)
        dof_pos_rel = self.vec_env.get_dof_pos_relative()
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
            upper_dof_pos=dof_pos_rel[:, self.robot_cfg.lower_body_dofs:],
            upper_dof_vel=self.vec_env.dof_vel[:, self.robot_cfg.lower_body_dofs:],
            intervention_amp=self.intervention_amp,
            intervention_freq=self.intervention_freq,
        )

    def _compute_rewards(self):
        """Compute all reward components and total reward."""
        is_timeout = self.episode_length_buf >= self.max_episode_length
        transition_mask = self.transition_timer > 0.0

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
            torques=self.vec_env.torques[:, :self.robot_cfg.lower_body_dofs],
            dof_vel=self.vec_env.dof_vel[:, :self.robot_cfg.lower_body_dofs],
            last_dof_vel=self.vec_env.last_dof_vel[:, :self.robot_cfg.lower_body_dofs],
            foot_contact_forces=self.vec_env.foot_contact_forces,
            foot_velocities=self.vec_env.foot_velocities,
            is_terminated=self.vec_env.has_contact_termination,
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
        self.reset_buf = term_result['reset']
        self.extras['termination'] = term_result

    def _log_episode_info(self, reset_ids: torch.Tensor):
        """Log episode statistics for environments being reset."""
        self.extras['episode'] = {
            'reward_mean': self.episode_reward_sums[reset_ids].mean().item(),
            'episode_length_mean': self.episode_length_buf[reset_ids].float().mean().item(),
        }
        # Log individual reward components
        for name, vals in self.reward_components.items():
            if name != 'total':
                self.extras['episode'][f'reward_{name}'] = vals[reset_ids].mean().item()

        # Log termination types
        if 'termination' in self.extras:
            term = self.extras['termination']
            self.extras['episode']['timeout_rate'] = term['timeout'][reset_ids].float().mean().item()
            self.extras['episode']['contact_fall_rate'] = term['contact'][reset_ids].float().mean().item()
            self.extras['episode']['orientation_fall_rate'] = term['orientation'][reset_ids].float().mean().item()

    # ---- Intervention interface (used by Wave 5) ----

    def set_intervention(self, upper_actions: torch.Tensor,
                         mask: torch.Tensor,
                         amp: torch.Tensor = None,
                         freq: torch.Tensor = None):
        """Set upper body intervention signals.

        Args:
            upper_actions: (N, 14) upper body joint targets
            mask: (N,) bool, which envs have active intervention
            amp: (N,) intervention amplitude (for critic obs)
            freq: (N,) intervention frequency (for critic obs)
        """
        self.upper_body_actions = upper_actions
        self.interrupt_mask = mask
        self.intervention_flag = mask.float()
        if amp is not None:
            self.intervention_amp = amp
        if freq is not None:
            self.intervention_freq = freq

    # ---- Curriculum interface (used by Wave 6) ----

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
