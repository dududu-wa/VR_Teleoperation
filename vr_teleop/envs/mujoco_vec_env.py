"""
Synchronous batched MuJoCo vectorized environment for G1 robot.

Manages N independent MjModel/MjData instances in a single process.
Physics runs on CPU, observations and rewards are computed as batched
torch tensors and transferred to GPU for training.

Implements the VecEnv interface expected by PPO / OnPolicyRunner.
"""

import os
import copy
import numpy as np
import mujoco
import torch
from typing import Tuple, Union, Optional, Dict

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.envs.g1_base_env import G1BaseEnv
from vr_teleop.envs.domain_rand import DomainRandomizer, DomainRandConfig
from vr_teleop.envs.observation import ObservationBuilder, ObsConfig
from vr_teleop.envs.dof_indices import LOCO_DOF_INDICES, VR_DOF_INDICES, NUM_LOCO_DOFS, NUM_VR_DOFS
from vr_teleop.utils.math_utils import (
    mujoco_quat_to_isaac, compute_projected_gravity,
    quat_rotate_inverse, get_euler_xyz
)
from vr_teleop.utils.config_utils import get_asset_path


class MujocoVecEnv:
    """Synchronous batched MuJoCo environment.

    VecEnv interface required by the PPO runner:
        - step(actions) -> (obs, privileged_obs, rewards, resets, extras)
        - reset(env_ids)
        - get_observations() -> obs_buf
        - get_privileged_observations() -> privileged_obs_buf
    """

    def __init__(
        self,
        num_envs: int,
        robot_cfg: G1Config = None,
        domain_rand_cfg: DomainRandConfig = None,
        sim_dt: float = 0.005,
        decimation: int = 4,
        max_episode_length: int = 1000,
        device: str = 'cuda:0',
        model_path: str = None,
    ):
        self.num_envs = num_envs
        self.cfg = robot_cfg or G1Config()
        self.rand_cfg = domain_rand_cfg or DomainRandConfig()
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.dt = sim_dt * decimation  # policy timestep
        self.max_episode_length = max_episode_length
        self.device = torch.device(device)
        self.num_dofs = self.cfg.num_dofs  # 29
        self.num_actions = NUM_LOCO_DOFS  # 13 (12 legs + waist_pitch)

        # Resolve model path
        if model_path is None:
            asset_root = get_asset_path()
            model_path = os.path.join(asset_root, self.cfg.mujoco_scene_file)
            if not os.path.exists(model_path):
                model_path = os.path.join(asset_root, self.cfg.mujoco_model_file)

        # Pre-compute index tensors for split control mapping
        self._loco_indices = torch.tensor(LOCO_DOF_INDICES, dtype=torch.long, device=self.device)
        self._vr_indices = torch.tensor(VR_DOF_INDICES, dtype=torch.long, device=self.device)

        # ---- Create N MuJoCo instances ----
        self._envs = []
        self._randomizers = []
        self._rngs = []
        for i in range(num_envs):
            env = G1BaseEnv(
                robot_cfg=self.cfg, sim_dt=sim_dt,
                decimation=decimation, model_path=model_path)
            randomizer = DomainRandomizer(
                cfg=self.rand_cfg, mj_model=env.mj_model,
                num_dofs=self.num_dofs, dt=self.dt)
            self._envs.append(env)
            self._randomizers.append(randomizer)
            self._rngs.append(np.random.default_rng(seed=i))

        # ---- State buffers (on GPU) ----
        # Base state
        self.base_quat_xyzw = torch.zeros(num_envs, 4, device=self.device)
        self.base_pos = torch.zeros(num_envs, 3, device=self.device)
        self.base_lin_vel = torch.zeros(num_envs, 3, device=self.device)  # world
        self.base_ang_vel = torch.zeros(num_envs, 3, device=self.device)  # world
        self.base_lin_vel_body = torch.zeros(num_envs, 3, device=self.device)
        self.base_ang_vel_body = torch.zeros(num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(num_envs, 3, device=self.device)
        self.base_euler = torch.zeros(num_envs, 3, device=self.device)  # roll, pitch, yaw

        # DOF state
        self.dof_pos = torch.zeros(num_envs, self.num_dofs, device=self.device)
        self.dof_vel = torch.zeros(num_envs, self.num_dofs, device=self.device)
        self.last_dof_vel = torch.zeros(num_envs, self.num_dofs, device=self.device)

        # Actions
        self.actions = torch.zeros(num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros(num_envs, self.num_actions, device=self.device)
        self.last_last_actions = torch.zeros(num_envs, self.num_actions, device=self.device)
        self.full_actions_29 = torch.zeros(num_envs, self.num_dofs, device=self.device)

        # Torques
        self.torques = torch.zeros(num_envs, self.num_dofs, device=self.device)

        # Foot state
        self.foot_contact_forces = torch.zeros(num_envs, 2, device=self.device)
        self.foot_contact_forces_3d = torch.zeros(num_envs, 2, 3, device=self.device)
        self.foot_velocities = torch.zeros(num_envs, 2, 3, device=self.device)

        # Contact termination
        self.has_contact_termination = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        # Default dof pos for observation (relative dof_pos)
        self.default_dof_pos = torch.tensor(
            self.cfg.get_default_dof_pos().numpy(),
            device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, 29)

        # Episode tracking
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.rew_buf = torch.zeros(num_envs, device=self.device)

        # Domain rand privileged info
        self.friction_coeffs = torch.ones(num_envs, device=self.device)
        self.mass_offsets = torch.zeros(num_envs, device=self.device)
        self.motor_strengths = torch.ones(num_envs, device=self.device)
        self.kp_multipliers = torch.ones(num_envs, device=self.device)
        self.kd_multipliers = torch.ones(num_envs, device=self.device)

        # Extras dict for logging
        self.extras = {}

        # Default observation interface so this class can be consumed directly.
        self.obs_cfg = ObsConfig()
        self.obs_builder = ObservationBuilder(
            obs_cfg=self.obs_cfg,
            num_envs=self.num_envs,
            device=self.device,
            num_loco_dofs=NUM_LOCO_DOFS,
        )
        self.obs_buf = torch.zeros(
            self.num_envs,
            self.obs_cfg.single_step_dim + self.obs_cfg.history_obs_dim * self.obs_cfg.include_history_steps,
            device=self.device,
            dtype=torch.float32,
        )
        self.privileged_obs_buf = torch.zeros(
            self.num_envs, self.obs_cfg.critic_obs_dim, device=self.device, dtype=torch.float32
        )
        self._command_context = {
            "commands": torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32),
            "gait_id": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "intervention_flag": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "clock": torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32),
            "intervention_amp": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
            "intervention_freq": torch.zeros(self.num_envs, device=self.device, dtype=torch.float32),
        }

    def reset_all(self):
        """Reset all environments."""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset(env_ids)

    def reset(self, env_ids: torch.Tensor):
        """Reset specific environments.

        Args:
            env_ids: (K,) tensor of environment indices to reset
        """
        if len(env_ids) == 0:
            return

        env_ids_cpu = env_ids.cpu().numpy().astype(int)

        for idx in env_ids_cpu:
            env = self._envs[idx]
            rand = self._randomizers[idx]
            rng = self._rngs[idx]

            # Reset physics
            env.reset()

            # Apply domain randomization
            rand.randomize_on_reset(env.mj_model, env.mj_data, rng)

            # Re-run forward after randomization changes
            mujoco.mj_forward(env.mj_model, env.mj_data)

        # Reset action buffers for these envs
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.full_actions_29[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False

        # Sync state from MuJoCo to tensors
        self._sync_state_from_mujoco(env_ids_cpu)
        self._refresh_default_observations(reset_env_ids=env_ids)

        # Update domain rand privileged info
        for idx in env_ids_cpu:
            info = self._randomizers[idx].get_privileged_info()
            self.friction_coeffs[idx] = info['friction_coeff']
            self.mass_offsets[idx] = info['mass_offset']
            self.motor_strengths[idx] = info['motor_strength']
            self.kp_multipliers[idx] = info['kp_multiplier']
            self.kd_multipliers[idx] = info['kd_multiplier']

    def step(self, actions: torch.Tensor):
        """Execute one policy step across all environments.

        Args:
            actions: (num_envs, num_actions) tensor of policy outputs (13-DOF locomotion)

        Returns:
            tuple: (obs, privileged_obs, rewards, resets, extras)
                   obs and rewards are on self.device (GPU)
        """
        # Store previous actions
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel

        # Clamp and store actions
        self.actions[:] = torch.clamp(actions, -self.cfg.action_clip_value, self.cfg.action_clip_value)

        # Build full 29-DOF action (lower body from policy, upper body zeros or from intervention)
        self.full_actions_29[:, self._loco_indices] = self.actions

        # Convert to numpy and step each environment
        actions_np = self.full_actions_29.cpu().numpy()

        for i in range(self.num_envs):
            env = self._envs[i]
            rand = self._randomizers[i]

            # Apply domain randomization to PD gains for this step
            if self.rand_cfg.randomize_pd_gains:
                env.kp = rand.get_randomized_kp(
                    self.cfg.get_kp().numpy())
                env.kd = rand.get_randomized_kd(
                    self.cfg.get_kd().numpy())

            if self.rand_cfg.randomize_motor_strength:
                env.torque_limits = rand.get_randomized_torque_limits(
                    self.cfg.get_torque_limits().numpy())

            # Step physics
            env.step(actions_np[i])

            # Maybe apply external push
            rand.maybe_push(env.mj_data, self._rngs[i])

        # Sync state from all environments
        all_ids = np.arange(self.num_envs)
        self._sync_state_from_mujoco(all_ids)

        # Increment episode length
        self.episode_length_buf += 1

        self._refresh_default_observations()

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def _sync_state_from_mujoco(self, env_ids: np.ndarray):
        """Sync MuJoCo state to GPU tensors for specified environments.

        Args:
            env_ids: numpy array of environment indices
        """
        # Collect numpy arrays
        n = len(env_ids)
        base_pos_np = np.zeros((n, 3))
        base_quat_wxyz_np = np.zeros((n, 4))
        base_lin_vel_np = np.zeros((n, 3))
        base_ang_vel_np = np.zeros((n, 3))
        dof_pos_np = np.zeros((n, self.num_dofs))
        dof_vel_np = np.zeros((n, self.num_dofs))
        torques_np = np.zeros((n, self.num_dofs))
        left_contact_np = np.zeros((n, 3))
        right_contact_np = np.zeros((n, 3))
        left_foot_vel_np = np.zeros((n, 3))
        right_foot_vel_np = np.zeros((n, 3))
        has_term_contact_np = np.zeros(n, dtype=bool)

        for j, idx in enumerate(env_ids):
            env = self._envs[idx]
            base_pos_np[j] = env.base_pos
            base_quat_wxyz_np[j] = env.base_quat_wxyz
            base_lin_vel_np[j] = env.base_lin_vel
            base_ang_vel_np[j] = env.base_ang_vel
            dof_pos_np[j] = env.dof_pos
            dof_vel_np[j] = env.dof_vel
            torques_np[j] = env.torques
            left_contact_np[j] = env.left_foot_contact_force
            right_contact_np[j] = env.right_foot_contact_force

            # Foot velocities
            left_v, right_v = env.get_foot_velocities()
            left_foot_vel_np[j] = left_v
            right_foot_vel_np[j] = right_v

            # Termination contact
            has_term_contact_np[j] = env.has_termination_contact()

        # Convert to torch and upload to GPU
        env_ids_torch = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        self.base_pos[env_ids_torch] = torch.from_numpy(base_pos_np).float().to(self.device)

        # Convert MuJoCo [w,x,y,z] to IsaacGym [x,y,z,w]
        quat_wxyz = torch.from_numpy(base_quat_wxyz_np).float().to(self.device)
        self.base_quat_xyzw[env_ids_torch] = torch.cat(
            [quat_wxyz[:, 1:4], quat_wxyz[:, 0:1]], dim=-1)

        self.base_lin_vel[env_ids_torch] = torch.from_numpy(base_lin_vel_np).float().to(self.device)
        self.base_ang_vel[env_ids_torch] = torch.from_numpy(base_ang_vel_np).float().to(self.device)
        self.dof_pos[env_ids_torch] = torch.from_numpy(dof_pos_np).float().to(self.device)
        self.dof_vel[env_ids_torch] = torch.from_numpy(dof_vel_np).float().to(self.device)
        self.torques[env_ids_torch] = torch.from_numpy(torques_np).float().to(self.device)

        # Foot contact force magnitudes
        left_mag = np.linalg.norm(left_contact_np, axis=-1)
        right_mag = np.linalg.norm(right_contact_np, axis=-1)
        self.foot_contact_forces[env_ids_torch, 0] = torch.from_numpy(left_mag).float().to(self.device)
        self.foot_contact_forces[env_ids_torch, 1] = torch.from_numpy(right_mag).float().to(self.device)

        # Foot contact force 3D vectors
        self.foot_contact_forces_3d[env_ids_torch, 0] = torch.from_numpy(left_contact_np).float().to(self.device)
        self.foot_contact_forces_3d[env_ids_torch, 1] = torch.from_numpy(right_contact_np).float().to(self.device)

        # Foot velocities
        self.foot_velocities[env_ids_torch, 0] = torch.from_numpy(left_foot_vel_np).float().to(self.device)
        self.foot_velocities[env_ids_torch, 1] = torch.from_numpy(right_foot_vel_np).float().to(self.device)

        # Contact termination
        self.has_contact_termination[env_ids_torch] = torch.from_numpy(has_term_contact_np).to(self.device)

        # Compute derived quantities (batched on GPU for the synced envs)
        quat = self.base_quat_xyzw[env_ids_torch]

        # Body-frame velocities
        self.base_lin_vel_body[env_ids_torch] = quat_rotate_inverse(
            quat, self.base_lin_vel[env_ids_torch])
        self.base_ang_vel_body[env_ids_torch] = quat_rotate_inverse(
            quat, self.base_ang_vel[env_ids_torch])

        # Projected gravity
        self.projected_gravity[env_ids_torch] = compute_projected_gravity(quat)

        # Euler angles
        roll, pitch, yaw = get_euler_xyz(quat)
        self.base_euler[env_ids_torch, 0] = roll
        self.base_euler[env_ids_torch, 1] = pitch
        self.base_euler[env_ids_torch, 2] = yaw

    def set_command_context(
        self,
        commands: Optional[torch.Tensor] = None,
        gait_id: Optional[torch.Tensor] = None,
        intervention_flag: Optional[torch.Tensor] = None,
        clock: Optional[torch.Tensor] = None,
        intervention_amp: Optional[torch.Tensor] = None,
        intervention_freq: Optional[torch.Tensor] = None,
    ):
        """Set command/intervention context used by default observation APIs."""
        if commands is not None:
            self._command_context["commands"] = commands.to(self.device)
        if gait_id is not None:
            self._command_context["gait_id"] = gait_id.to(self.device)
        if intervention_flag is not None:
            self._command_context["intervention_flag"] = intervention_flag.to(self.device)
        if clock is not None:
            self._command_context["clock"] = clock.to(self.device)
        if intervention_amp is not None:
            self._command_context["intervention_amp"] = intervention_amp.to(self.device)
        if intervention_freq is not None:
            self._command_context["intervention_freq"] = intervention_freq.to(self.device)

    def _refresh_default_observations(self, reset_env_ids: Optional[torch.Tensor] = None):
        """Build actor/critic observations for direct VecEnv compatibility."""
        dof_pos_rel = self.get_dof_pos_relative()
        lower_pos = dof_pos_rel[:, self._loco_indices]
        lower_vel = self.dof_vel[:, self._loco_indices]
        upper_pos = dof_pos_rel[:, self._vr_indices]

        actor_obs = self.obs_builder.build_actor_obs(
            base_ang_vel=self.base_ang_vel_body,
            projected_gravity=self.projected_gravity,
            dof_pos_loco=lower_pos,
            dof_vel_loco=lower_vel,
            last_actions=self.last_actions,
            upper_body_pos=upper_pos,
            commands=self._command_context["commands"],
            gait_id=self._command_context["gait_id"],
            clock=self._command_context["clock"],
        )
        if reset_env_ids is not None and len(reset_env_ids) > 0:
            self.obs_builder.reset_history(reset_env_ids, actor_obs)
        else:
            self.obs_builder.update_history(actor_obs)

        self.obs_buf = self.obs_builder.get_actor_obs_with_history(actor_obs)
        self.privileged_obs_buf = self.obs_builder.build_critic_obs(
            actor_obs=actor_obs,
            base_lin_vel=self.base_lin_vel_body,
            base_height=self.get_base_height(),
            foot_contact_forces=self.foot_contact_forces,
            friction_coeff=self.friction_coeffs,
            mass_offset=self.mass_offsets,
            motor_strength=self.motor_strengths,
            pd_gain_mult=torch.stack([self.kp_multipliers, self.kd_multipliers], dim=-1),
            upper_dof_vel=self.dof_vel[:, self._vr_indices],
            intervention_amp=self._command_context["intervention_amp"],
            intervention_freq=self._command_context["intervention_freq"],
        )

    def set_upper_body_actions(self, upper_actions: torch.Tensor, env_ids: torch.Tensor = None):
        """Set upper body DOF targets (from intervention or VR).

        Args:
            upper_actions: (N, 16) or (K, 16) upper body actions
            env_ids: optional subset of env indices
        """
        if env_ids is None:
            self.full_actions_29[:, self._vr_indices] = upper_actions
        else:
            subset = self.full_actions_29[env_ids]
            subset[:, self._vr_indices] = upper_actions
            self.full_actions_29[env_ids] = subset

    def get_dof_pos_relative(self) -> torch.Tensor:
        """Get joint positions relative to default stance (N, 29)."""
        return self.dof_pos - self.default_dof_pos

    def get_base_height(self) -> torch.Tensor:
        """Get base height above ground (N,)."""
        return self.base_pos[:, 2]

    def get_observations(self) -> torch.Tensor:
        """Return actor observations built from current internal state."""
        return self.obs_buf

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """Return critic observations built from current internal state."""
        return self.privileged_obs_buf

    @property
    def num_obs(self) -> int:
        """Number of actor observation dimensions."""
        return int(self.obs_buf.shape[-1])

    @property
    def num_privileged_obs(self) -> int:
        """Number of critic observation dimensions."""
        return int(self.privileged_obs_buf.shape[-1])

