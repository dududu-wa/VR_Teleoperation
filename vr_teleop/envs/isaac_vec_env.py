"""
Self-contained Isaac Gym vectorized environment backend.

This module intentionally does not import or depend on any sibling repository.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from vr_teleop.envs.observation import ObsConfig, ObservationBuilder
from vr_teleop.envs.dof_indices import LOCO_DOF_INDICES, VR_DOF_INDICES, NUM_LOCO_DOFS
from vr_teleop.utils.config_utils import get_asset_path
from vr_teleop.utils.math_utils import (
    compute_projected_gravity,
    get_euler_xyz,
    quat_rotate_inverse,
    wrap_to_pi,
)


def is_isaacgym_available() -> bool:
    """Return whether `isaacgym` can be imported."""
    try:
        import isaacgym  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _parse_sim_device(device: str) -> Tuple[str, int]:
    """Parse torch-like device string to Isaac Gym device type and index."""
    dev = str(device).strip().lower()
    if dev.startswith("cuda"):
        parts = dev.split(":", 1)
        index = int(parts[1]) if len(parts) == 2 and parts[1] else 0
        return "cuda", index
    return "cpu", 0


class IsaacVecEnv:
    """Native Isaac Gym backend without external project dependencies."""

    def __init__(
        self,
        num_envs: int,
        robot_cfg,
        domain_rand_cfg,
        sim_dt: float = 0.005,
        decimation: int = 4,
        max_episode_length: int = 1000,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
        headless: bool = True,
        **_: Any,
    ):
        if not is_isaacgym_available():
            raise RuntimeError(
                "Isaac Gym backend requested but `isaacgym` is unavailable. "
                "Install Isaac Gym and ensure its Python path is active."
            )

        from isaacgym import gymapi, gymtorch  # type: ignore

        self.gymapi = gymapi
        self.gymtorch = gymtorch
        self.gym = gymapi.acquire_gym()

        self.num_envs = int(num_envs)
        self.cfg = robot_cfg
        self.rand_cfg = domain_rand_cfg
        self.sim_dt = float(sim_dt)
        self.decimation = int(decimation)
        self.dt = self.sim_dt * self.decimation
        self.max_episode_length = int(max_episode_length)
        self.device = torch.device(device)
        self.headless = bool(headless)

        self.num_dofs = int(self.cfg.num_dofs)
        self.num_actions = NUM_LOCO_DOFS  # 13 (12 legs + waist_pitch)
        self.backend_name = "isaac_native_local"

        # Pre-compute index tensors for split control mapping
        self._loco_indices = torch.tensor(LOCO_DOF_INDICES, dtype=torch.long, device=self.device)
        self._vr_indices = torch.tensor(VR_DOF_INDICES, dtype=torch.long, device=self.device)
        self.using_fallback = False
        self.requested_backend = "isaacgym"

        self._sim_device_type, self._sim_device_id = _parse_sim_device(str(self.device))
        self._graphics_device_id = (
            -1 if self.headless else (self._sim_device_id if self._sim_device_type == "cuda" else 0)
        )

        self.sim = self._create_sim()
        self._add_ground_plane()

        self.model_path = self._resolve_model_path(model_path)
        self.asset = self._load_asset(self.model_path)

        self.num_asset_dofs = int(self.gym.get_asset_dof_count(self.asset))
        self.num_bodies = int(self.gym.get_asset_rigid_body_count(self.asset))
        if self.num_asset_dofs < self.num_dofs:
            raise RuntimeError(
                f"Loaded asset has {self.num_asset_dofs} dofs, but robot_cfg expects {self.num_dofs}."
            )

        self.envs: List[Any] = []
        self.actors: List[Any] = []
        self._actor_indices_list: List[int] = []
        self._create_envs()

        self.viewer = None
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, self.gymapi.CameraProperties())
            if self.viewer is None:
                raise RuntimeError("Failed to create Isaac Gym viewer.")
            self._setup_viewer_camera()

        self.gym.prepare_sim(self.sim)
        self._acquire_tensors()
        self._setup_body_indices()

        self._init_control_constants()
        self._allocate_buffers()
        self._init_observation_interface()

        self.reset_all()

    def _create_sim(self):
        sim_params = self.gymapi.SimParams()
        sim_params.dt = self.sim_dt
        sim_params.substeps = 1
        sim_params.up_axis = self.gymapi.UP_AXIS_Z
        sim_params.gravity = self.gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = self._sim_device_type == "cuda"

        sim_params.physx.use_gpu = self._sim_device_type == "cuda"
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.contact_offset = 0.02
        sim_params.physx.rest_offset = 0.0

        sim = self.gym.create_sim(
            self._sim_device_id,
            self._graphics_device_id,
            self.gymapi.SIM_PHYSX,
            sim_params,
        )
        if sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation.")
        return sim

    def _add_ground_plane(self):
        plane = self.gymapi.PlaneParams()
        plane.normal = self.gymapi.Vec3(0.0, 0.0, 1.0)
        plane.distance = 0.0
        plane.static_friction = 1.0
        plane.dynamic_friction = 1.0
        plane.restitution = 0.0
        self.gym.add_ground(self.sim, plane)

    def _resolve_model_path(self, model_path: Optional[str]) -> str:
        if model_path is not None:
            path = os.path.abspath(model_path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model path does not exist: {path}")
            return path

        asset_root = get_asset_path()
        # Try URDF first (preferred for Isaac Gym), then XML scene/model
        urdf_path = os.path.join(asset_root, getattr(self.cfg, 'urdf_file', ''))
        scene_path = os.path.join(asset_root, self.cfg.mujoco_scene_file)
        model_file_path = os.path.join(asset_root, self.cfg.mujoco_model_file)

        if urdf_path and os.path.exists(urdf_path):
            return urdf_path
        if os.path.exists(scene_path):
            return scene_path
        if os.path.exists(model_file_path):
            return model_file_path
        raise FileNotFoundError(
            "No G1 model file found. Checked:\n"
            f"  - {scene_path}\n"
            f"  - {model_file_path}"
        )

    def _load_asset(self, model_path: str):
        root = os.path.dirname(model_path)
        file = os.path.basename(model_path)
        options = self.gymapi.AssetOptions()
        options.default_dof_drive_mode = self.gymapi.DOF_MODE_EFFORT
        options.fix_base_link = False
        options.disable_gravity = False
        options.collapse_fixed_joints = False
        options.flip_visual_attachments = False
        options.replace_cylinder_with_capsule = False
        options.use_mesh_materials = True
        asset = self.gym.load_asset(self.sim, root, file, options)
        if asset is None:
            raise RuntimeError(f"Failed to load Isaac Gym asset: {model_path}")
        return asset

    def _create_envs(self):
        spacing = 2.0
        num_per_row = int(math.ceil(math.sqrt(self.num_envs)))
        env_lower = self.gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = self.gymapi.Vec3(spacing, spacing, spacing)

        dof_props = self.gym.get_asset_dof_properties(self.asset)
        dof_props["driveMode"][:] = self.gymapi.DOF_MODE_EFFORT
        dof_props["stiffness"][:] = 0.0
        dof_props["damping"][:] = 0.0

        default_pose = self.gymapi.Transform()
        default_pose.p = self.gymapi.Vec3(*self.cfg.init_pos)
        q = self.cfg.init_rot
        default_pose.r = self.gymapi.Quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            actor = self.gym.create_actor(env, self.asset, default_pose, "g1", i, 0, 0)
            self.gym.set_actor_dof_properties(env, actor, dof_props)

            actor_idx = self.gym.get_actor_index(env, actor, self.gymapi.DOMAIN_SIM)
            self.envs.append(env)
            self.actors.append(actor)
            self._actor_indices_list.append(int(actor_idx))

    def _setup_viewer_camera(self):
        if self.viewer is None or not self.envs:
            return
        cam_pos = self.gymapi.Vec3(3.0, 3.0, 2.0)
        cam_target = self.gymapi.Vec3(0.0, 0.0, 0.8)
        self.gym.viewer_camera_look_at(self.viewer, self.envs[0], cam_pos, cam_target)

    def _acquire_tensors(self):
        self._root_state_tensor = self.gymtorch.wrap_tensor(
            self.gym.acquire_actor_root_state_tensor(self.sim)
        )
        self._dof_state_tensor = self.gymtorch.wrap_tensor(
            self.gym.acquire_dof_state_tensor(self.sim)
        )
        self._rb_state_tensor = self.gymtorch.wrap_tensor(
            self.gym.acquire_rigid_body_state_tensor(self.sim)
        )
        self._net_contact_tensor = self.gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim)
        )

        self._sim_tensor_device = self._root_state_tensor.device
        self.actor_indices = torch.tensor(
            self._actor_indices_list, dtype=torch.int32, device=self._sim_tensor_device
        )

        self._root_state = self._root_state_tensor.view(self.num_envs, -1, 13)[:, 0, :]
        self._dof_state = self._dof_state_tensor.view(self.num_envs, self.num_asset_dofs, 2)
        self._rb_state = self._rb_state_tensor.view(self.num_envs, self.num_bodies, 13)
        self._net_contact_forces = self._net_contact_tensor.view(self.num_envs, self.num_bodies, 3)
        self._actuation_forces = torch.zeros(
            self.num_envs, self.num_asset_dofs, dtype=torch.float32, device=self._sim_tensor_device
        )

    def _setup_body_indices(self):
        body_names: Sequence[str] = self.gym.get_asset_rigid_body_names(self.asset)
        self._body_names = [str(n) for n in body_names]
        name_to_idx = {name: i for i, name in enumerate(self._body_names)}

        def _resolve_body(name: str, fallbacks: Sequence[str]) -> Optional[int]:
            if name in name_to_idx:
                return int(name_to_idx[name])
            lower_name = name.lower()
            for body, idx in name_to_idx.items():
                if lower_name in body.lower():
                    return int(idx)
            for token in fallbacks:
                token = token.lower()
                for body, idx in name_to_idx.items():
                    if token in body.lower():
                        return int(idx)
            return None

        self.left_foot_body_idx = _resolve_body(self.cfg.left_foot_name, ["left_ankle", "left_foot"])
        self.right_foot_body_idx = _resolve_body(self.cfg.right_foot_name, ["right_ankle", "right_foot"])

        term_indices = set()
        for idx, body_name in enumerate(self._body_names):
            lower = body_name.lower()
            for token in self.cfg.terminate_after_contacts_on:
                if str(token).lower() in lower:
                    term_indices.add(idx)
                    break
        self.termination_body_indices = sorted(term_indices)
        self.contact_force_threshold = 1.0

    def _init_control_constants(self):
        self.default_dof_pos = self.cfg.get_default_dof_pos().to(self.device).unsqueeze(0)
        self.kp = self.cfg.get_kp().to(self.device).unsqueeze(0)
        self.kd = self.cfg.get_kd().to(self.device).unsqueeze(0)
        self.torque_limits = self.cfg.get_torque_limits().to(self.device).unsqueeze(0)

        self._default_dof_pos_sim = torch.zeros(
            self.num_asset_dofs, dtype=torch.float32, device=self._sim_tensor_device
        )
        self._default_dof_pos_sim[: self.num_dofs] = self.default_dof_pos[0].to(self._sim_tensor_device)

    def _allocate_buffers(self):
        self.base_quat_xyzw = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_lin_vel_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_ang_vel_body = torch.zeros(self.num_envs, 3, device=self.device)
        self.projected_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_euler = torch.zeros(self.num_envs, 3, device=self.device)

        self.dof_pos = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dofs, device=self.device)

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.full_actions_29 = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.upper_body_actions = torch.zeros(
            self.num_envs, len(VR_DOF_INDICES), device=self.device
        )

        self.torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.foot_contact_forces = torch.zeros(self.num_envs, 2, device=self.device)
        self.foot_contact_forces_3d = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.foot_velocities = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.has_contact_termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.extras: Dict[str, Any] = {}

        self.friction_coeffs = torch.ones(self.num_envs, device=self.device)
        self.mass_offsets = torch.zeros(self.num_envs, device=self.device)
        self.motor_strengths = torch.ones(self.num_envs, device=self.device)
        self.kp_multipliers = torch.ones(self.num_envs, device=self.device)
        self.kd_multipliers = torch.ones(self.num_envs, device=self.device)

    def _init_observation_interface(self):
        self.obs_cfg = ObsConfig()
        self.obs_builder = ObservationBuilder(
            obs_cfg=self.obs_cfg,
            num_envs=self.num_envs,
            device=self.device,
            lower_body_dofs=NUM_LOCO_DOFS,
        )
        total_obs_dim = (
            self.obs_cfg.single_step_dim
            + self.obs_cfg.history_obs_dim * self.obs_cfg.include_history_steps
        )
        self.obs_buf = torch.zeros(self.num_envs, total_obs_dim, device=self.device, dtype=torch.float32)
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

    def _unwrap(self, tensor: torch.Tensor):
        return self.gymtorch.unwrap_tensor(tensor)

    def _refresh_sim_tensors(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _sync_from_sim(self):
        self.base_pos[:] = self._root_state[:, 0:3].to(self.device)
        self.base_quat_xyzw[:] = self._root_state[:, 3:7].to(self.device)
        self.base_lin_vel[:] = self._root_state[:, 7:10].to(self.device)
        self.base_ang_vel[:] = self._root_state[:, 10:13].to(self.device)

        self.dof_pos[:] = self._dof_state[:, : self.num_dofs, 0].to(self.device)
        self.dof_vel[:] = self._dof_state[:, : self.num_dofs, 1].to(self.device)

        self.base_lin_vel_body[:] = quat_rotate_inverse(self.base_quat_xyzw, self.base_lin_vel)
        self.base_ang_vel_body[:] = quat_rotate_inverse(self.base_quat_xyzw, self.base_ang_vel)
        self.projected_gravity[:] = compute_projected_gravity(self.base_quat_xyzw)

        roll, pitch, yaw = get_euler_xyz(self.base_quat_xyzw)
        self.base_euler[:, 0] = wrap_to_pi(roll)
        self.base_euler[:, 1] = wrap_to_pi(pitch)
        self.base_euler[:, 2] = wrap_to_pi(yaw)

        self.torques.zero_()
        self.torques[:, : self.num_dofs] = self._actuation_forces[:, : self.num_dofs].to(self.device)

        self.foot_contact_forces.zero_()
        self.foot_contact_forces_3d.zero_()
        if self.left_foot_body_idx is not None:
            left_force = self._net_contact_forces[:, self.left_foot_body_idx, :]
            self.foot_contact_forces[:, 0] = torch.linalg.norm(left_force, dim=-1).to(self.device)
            self.foot_contact_forces_3d[:, 0] = left_force.to(self.device)
        if self.right_foot_body_idx is not None:
            right_force = self._net_contact_forces[:, self.right_foot_body_idx, :]
            self.foot_contact_forces[:, 1] = torch.linalg.norm(right_force, dim=-1).to(self.device)
            self.foot_contact_forces_3d[:, 1] = right_force.to(self.device)

        self.foot_velocities.zero_()
        if self.left_foot_body_idx is not None:
            self.foot_velocities[:, 0, :] = self._rb_state[:, self.left_foot_body_idx, 7:10].to(self.device)
        if self.right_foot_body_idx is not None:
            self.foot_velocities[:, 1, :] = self._rb_state[:, self.right_foot_body_idx, 7:10].to(self.device)

        if self.termination_body_indices:
            term_forces = self._net_contact_forces[:, self.termination_body_indices, :]
            term_mag = torch.linalg.norm(term_forces, dim=-1)
            self.has_contact_termination[:] = (term_mag > self.contact_force_threshold).any(dim=1).to(
                self.device
            )
        else:
            self.has_contact_termination[:] = False

    def _build_torques(self) -> torch.Tensor:
        targets = self.default_dof_pos + self.cfg.action_scale * self.full_actions_29
        kp = self.kp * self.kp_multipliers.unsqueeze(-1)
        kd = self.kd * self.kd_multipliers.unsqueeze(-1)
        torques = kp * (targets - self.dof_pos) - kd * self.dof_vel
        torques = torques * self.motor_strengths.unsqueeze(-1)
        return torch.clamp(torques, -self.torque_limits, self.torque_limits)

    def _refresh_default_observations(self, reset_env_ids: Optional[torch.Tensor] = None):
        dof_pos_rel = self.get_dof_pos_relative()
        lower_pos = dof_pos_rel[:, self._loco_indices]
        lower_vel = self.dof_vel[:, self._loco_indices]

        actor_obs = self.obs_builder.build_actor_obs(
            base_ang_vel=self.base_ang_vel_body,
            projected_gravity=self.projected_gravity,
            dof_pos_lower=lower_pos,
            dof_vel_lower=lower_vel,
            last_actions=self.last_actions,
            commands=self._command_context["commands"],
            gait_id=self._command_context["gait_id"],
            intervention_flag=self._command_context["intervention_flag"],
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
            upper_dof_pos=dof_pos_rel[:, self._vr_indices],
            upper_dof_vel=self.dof_vel[:, self._vr_indices],
            intervention_amp=self._command_context["intervention_amp"],
            intervention_freq=self._command_context["intervention_freq"],
        )

    def _reset_sim_state(self, env_ids: torch.Tensor):
        env_ids_sim_long = env_ids.to(device=self._sim_tensor_device, dtype=torch.long)
        if env_ids_sim_long.numel() == 0:
            return

        init_pos = torch.tensor(self.cfg.init_pos, dtype=torch.float32, device=self._sim_tensor_device)
        init_rot = torch.tensor(self.cfg.init_rot, dtype=torch.float32, device=self._sim_tensor_device)

        self._root_state[env_ids_sim_long, 0:3] = init_pos
        self._root_state[env_ids_sim_long, 3:7] = init_rot
        self._root_state[env_ids_sim_long, 7:13] = 0.0

        self._dof_state[env_ids_sim_long, :, :] = 0.0
        self._dof_state[env_ids_sim_long, : self.num_dofs, 0] = self._default_dof_pos_sim[: self.num_dofs]
        self._actuation_forces[env_ids_sim_long, :] = 0.0

        actor_ids = self.actor_indices.index_select(0, env_ids_sim_long).contiguous()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            self._unwrap(self._root_state_tensor),
            self._unwrap(actor_ids),
            actor_ids.numel(),
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            self._unwrap(self._dof_state_tensor),
            self._unwrap(actor_ids),
            actor_ids.numel(),
        )
        self.gym.set_dof_actuation_force_tensor(self.sim, self._unwrap(self._actuation_forces.view(-1)))

    def set_command_context(
        self,
        commands: Optional[torch.Tensor] = None,
        gait_id: Optional[torch.Tensor] = None,
        intervention_flag: Optional[torch.Tensor] = None,
        clock: Optional[torch.Tensor] = None,
        intervention_amp: Optional[torch.Tensor] = None,
        intervention_freq: Optional[torch.Tensor] = None,
    ):
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

    def reset_all(self):
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.reset(env_ids)

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self.device, dtype=torch.long)

        self._reset_sim_state(env_ids)
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.full_actions_29[env_ids] = 0.0
        self.upper_body_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False

        self._refresh_sim_tensors()
        self._sync_from_sim()
        self._refresh_default_observations(reset_env_ids=env_ids)

    def step(self, actions: torch.Tensor):
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel

        actions = torch.clamp(actions, -self.cfg.action_clip_value, self.cfg.action_clip_value).to(self.device)
        self.actions[:] = actions
        self.full_actions_29[:, self._loco_indices] = actions
        self.full_actions_29[:, self._vr_indices] = self.upper_body_actions

        for _ in range(self.decimation):
            torques = self._build_torques()
            self._actuation_forces.zero_()
            self._actuation_forces[:, : self.num_dofs] = torques.to(self._sim_tensor_device)
            self.gym.set_dof_actuation_force_tensor(self.sim, self._unwrap(self._actuation_forces.view(-1)))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if self.viewer is not None:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            self._refresh_sim_tensors()

        self._sync_from_sim()
        self.episode_length_buf += 1
        self.rew_buf.zero_()
        self.reset_buf.zero_()
        self.extras = {}
        self._refresh_default_observations()

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def set_upper_body_actions(self, upper_actions: torch.Tensor, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            self.upper_body_actions[:] = upper_actions.to(self.device)
        else:
            env_ids = env_ids.to(self.device, dtype=torch.long)
            self.upper_body_actions[env_ids] = upper_actions.to(self.device)

    def get_dof_pos_relative(self) -> torch.Tensor:
        return self.dof_pos - self.default_dof_pos

    def get_base_height(self) -> torch.Tensor:
        return self.base_pos[:, 2]

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        return self.privileged_obs_buf

    @property
    def num_obs(self) -> int:
        return int(self.obs_buf.shape[-1])

    @property
    def num_privileged_obs(self) -> int:
        return int(self.privileged_obs_buf.shape[-1])

    def close(self):
        if self.viewer is not None:
            try:
                self.gym.destroy_viewer(self.viewer)
            except Exception:
                pass
            self.viewer = None
        if self.sim is not None:
            try:
                self.gym.destroy_sim(self.sim)
            except Exception:
                pass
            self.sim = None


IsaacGymVecEnv = IsaacVecEnv
