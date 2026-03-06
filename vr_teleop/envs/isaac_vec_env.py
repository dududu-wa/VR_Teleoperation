"""
Isaac Gym vectorized environment backend.

Implementation strategy:
1) Require Isaac Gym runtime.
2) Use external backend from sibling `unitree_rl_gym`.
3) No MuJoCo fallback in training backend path.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Any

import torch

from vr_teleop.envs.observation import ObsConfig, ObservationBuilder
from vr_teleop.utils.config_utils import get_unitree_rl_gym_root


def is_isaacgym_available() -> bool:
    """Return whether `isaacgym` can be imported."""
    try:
        import isaacgym  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


class _ExternalUnitreeIsaacVecEnv:
    """Adapter around `unitree_rl_gym` IsaacGym environment."""

    def __init__(
        self,
        num_envs: int,
        robot_cfg,
        domain_rand_cfg,
        sim_dt: float,
        decimation: int,
        max_episode_length: int,
        device: str,
        model_path: Optional[str] = None,
        external_task: str = "g1",
        external_headless: bool = True,
    ):
        self.num_envs = int(num_envs)
        self.cfg = robot_cfg
        self.rand_cfg = domain_rand_cfg
        self.sim_dt = float(sim_dt)
        self.decimation = int(decimation)
        self.dt = self.sim_dt * self.decimation
        self.max_episode_length = int(max_episode_length)
        self.device = torch.device(device)
        self.num_dofs = int(self.cfg.num_dofs)
        self.num_actions = int(self.cfg.lower_body_dofs)
        self.backend_name = "isaac_external_unitree_rl_gym"
        self.using_fallback = False

        self._external_env = self._create_external_env(
            task_name=external_task,
            num_envs=self.num_envs,
            sim_dt=self.sim_dt,
            decimation=self.decimation,
            device=device,
            headless=external_headless,
        )
        self._external_num_actions = int(getattr(self._external_env, "num_actions"))
        self._external_num_dofs = int(getattr(self._external_env, "num_dof"))
        self._ext_device = torch.device(getattr(self._external_env, "device", device))

        # Core buffers aligned with MujocoVecEnv API.
        self.base_quat_xyzw = torch.zeros(self.num_envs, 4, device=self.device)
        self.base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)  # world
        self.base_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)  # world
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
            self.num_envs, self.num_dofs - self.num_actions, device=self.device
        )

        self.torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device)
        self.foot_contact_forces = torch.zeros(self.num_envs, 2, device=self.device)
        self.foot_velocities = torch.zeros(self.num_envs, 2, 3, device=self.device)
        self.has_contact_termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.default_dof_pos = torch.tensor(
            self.cfg.get_default_dof_pos().numpy(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.extras = {}

        self.friction_coeffs = torch.ones(self.num_envs, device=self.device)
        self.mass_offsets = torch.zeros(self.num_envs, device=self.device)
        self.motor_strengths = torch.ones(self.num_envs, device=self.device)
        self.kp_multipliers = torch.ones(self.num_envs, device=self.device)
        self.kd_multipliers = torch.ones(self.num_envs, device=self.device)

        # Default observation interface (same pattern as MujocoVecEnv).
        self.obs_cfg = ObsConfig()
        self.obs_builder = ObservationBuilder(
            obs_cfg=self.obs_cfg,
            num_envs=self.num_envs,
            device=self.device,
            lower_body_dofs=self.cfg.lower_body_dofs,
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

        self.reset_all()

    def _create_external_env(
        self,
        task_name: str,
        num_envs: int,
        sim_dt: float,
        decimation: int,
        device: str,
        headless: bool,
    ):
        root = get_unitree_rl_gym_root()
        if root is None or not os.path.exists(root):
            raise RuntimeError("unitree_rl_gym repo not found")
        if root not in sys.path:
            sys.path.insert(0, root)

        import isaacgym  # type: ignore  # noqa: F401
        from isaacgym import gymapi  # type: ignore

        import legged_gym.envs  # noqa: F401  # trigger task registration
        from legged_gym.utils.task_registry import task_registry
        from legged_gym.utils.helpers import class_to_dict, parse_sim_params

        env_cfg, _ = task_registry.get_cfgs(task_name)
        env_cfg.env.num_envs = int(num_envs)
        if hasattr(env_cfg.sim, "dt"):
            env_cfg.sim.dt = float(sim_dt)
        if hasattr(env_cfg.control, "decimation"):
            env_cfg.control.decimation = int(decimation)

        # Build args namespace expected by parse_sim_params helper.
        class _Args:
            pass

        args = _Args()
        args.physics_engine = gymapi.SIM_PHYSX
        args.device = "cuda" if "cuda" in device else "cpu"
        args.use_gpu = bool("cuda" in device)
        args.subscenes = 0
        args.use_gpu_pipeline = bool("cuda" in device)
        args.num_threads = 0
        args.compute_device_id = 0
        args.sim_device_id = 0
        args.sim_device_type = "cuda" if "cuda" in device else "cpu"
        args.sim_device = device
        args.headless = bool(headless)
        args.rl_device = device
        args.num_envs = int(num_envs)
        args.seed = getattr(env_cfg, "seed", 1)
        args.max_iterations = None
        args.resume = False
        args.experiment_name = None
        args.run_name = None
        args.load_run = None
        args.checkpoint = None

        sim_params = parse_sim_params(args, {"sim": class_to_dict(env_cfg.sim)})
        task_cls = task_registry.get_task_class(task_name)
        env = task_cls(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=args.physics_engine,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env

    def _map_actions_to_external(self, actions: torch.Tensor) -> torch.Tensor:
        act = actions.to(self._ext_device)
        mapped = torch.zeros(self.num_envs, self._external_num_actions, device=self._ext_device)
        n = min(self._external_num_actions, act.shape[-1])
        mapped[:, :n] = act[:, :n]
        return mapped

    def _sync_from_external(self):
        env = self._external_env

        root_states = env.root_states.to(self.device)
        self.base_pos[:] = root_states[:, 0:3]
        self.base_quat_xyzw[:] = root_states[:, 3:7]
        self.base_lin_vel[:] = root_states[:, 7:10]
        self.base_ang_vel[:] = root_states[:, 10:13]

        if hasattr(env, "base_lin_vel"):
            self.base_lin_vel_body[:] = env.base_lin_vel.to(self.device)
        if hasattr(env, "base_ang_vel"):
            self.base_ang_vel_body[:] = env.base_ang_vel.to(self.device)
        if hasattr(env, "projected_gravity"):
            self.projected_gravity[:] = env.projected_gravity.to(self.device)
        if hasattr(env, "rpy"):
            self.base_euler[:] = env.rpy.to(self.device)

        self.dof_pos.zero_()
        self.dof_vel.zero_()
        self.torques.zero_()
        n = min(self._external_num_dofs, self.num_dofs)
        self.dof_pos[:, :n] = env.dof_pos[:, :n].to(self.device)
        self.dof_vel[:, :n] = env.dof_vel[:, :n].to(self.device)
        if hasattr(env, "torques"):
            self.torques[:, :n] = env.torques[:, :n].to(self.device)

        if hasattr(env, "contact_forces") and hasattr(env, "feet_indices"):
            cf = env.contact_forces[:, env.feet_indices, :].to(self.device)
            mags = torch.norm(cf, dim=-1)
            m = min(mags.shape[1], 2)
            self.foot_contact_forces.zero_()
            self.foot_contact_forces[:, :m] = mags[:, :m]

        if hasattr(env, "feet_vel"):
            fv = env.feet_vel.to(self.device)
            m = min(fv.shape[1], 2)
            self.foot_velocities.zero_()
            self.foot_velocities[:, :m] = fv[:, :m]

        self.rew_buf[:] = env.rew_buf.to(self.device)
        self.reset_buf[:] = env.reset_buf.to(self.device) > 0
        self.episode_length_buf[:] = env.episode_length_buf.to(self.device)
        self.extras = env.extras

        if hasattr(env, "time_out_buf"):
            self.has_contact_termination[:] = self.reset_buf & (~env.time_out_buf.to(self.device))
        else:
            self.has_contact_termination[:] = self.reset_buf

        if hasattr(env, "friction_coeffs"):
            fr = env.friction_coeffs.to(self.device).view(self.num_envs, -1)[:, 0]
            self.friction_coeffs[:] = fr

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

    def _refresh_default_observations(self, reset_env_ids: Optional[torch.Tensor] = None):
        dof_pos_rel = self.get_dof_pos_relative()
        lower_pos = dof_pos_rel[:, :self.cfg.lower_body_dofs]
        lower_vel = self.dof_vel[:, :self.cfg.lower_body_dofs]

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
            upper_dof_pos=dof_pos_rel[:, self.cfg.lower_body_dofs:],
            upper_dof_vel=self.dof_vel[:, self.cfg.lower_body_dofs:],
            intervention_amp=self._command_context["intervention_amp"],
            intervention_freq=self._command_context["intervention_freq"],
        )

    def reset_all(self):
        self._external_env.reset()
        self._sync_from_external()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._refresh_default_observations(reset_env_ids=env_ids)

    def reset(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(self._ext_device, dtype=torch.long)
        self._external_env.reset_idx(env_ids)
        self.actions[env_ids.to(self.device)] = 0.0
        self.last_actions[env_ids.to(self.device)] = 0.0
        self.last_last_actions[env_ids.to(self.device)] = 0.0
        self.full_actions_29[env_ids.to(self.device)] = 0.0
        self._sync_from_external()
        self._refresh_default_observations(reset_env_ids=env_ids.to(self.device))

    def step(self, actions: torch.Tensor):
        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel

        actions = torch.clamp(actions, -self.cfg.action_clip_value, self.cfg.action_clip_value).to(self.device)
        self.actions[:] = actions
        self.full_actions_29[:, : self.cfg.lower_body_dofs] = actions
        self.full_actions_29[:, self.cfg.lower_body_dofs :] = self.upper_body_actions

        ext_actions = self._map_actions_to_external(actions)
        self._external_env.step(ext_actions)

        self._sync_from_external()
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
        viewer = getattr(self._external_env, "viewer", None)
        if viewer is not None:
            try:
                self._external_env.gym.destroy_viewer(viewer)
            except Exception:
                pass


class IsaacVecEnv:
    """Primary Isaac backend (strict, no MuJoCo fallback)."""

    def __init__(
        self,
        *args,
        prefer_external: bool = True,
        external_task: str = "g1",
        external_headless: bool = True,
        **kwargs,
    ):
        self.requested_backend = "isaacgym"
        self.using_fallback = False
        self.backend_name = "isaac"
        self._delegate: Any = None

        if not is_isaacgym_available():
            raise RuntimeError(
                "Isaac Gym backend requested but `isaacgym` is unavailable. "
                "Install Isaac Gym and ensure its Python path is active."
            )
        if not prefer_external:
            raise RuntimeError(
                "Internal Isaac backend is not implemented; external backend is required. "
                "Set `prefer_external=True` and provide unitree_rl_gym."
            )

        try:
            self._delegate = _ExternalUnitreeIsaacVecEnv(
                *args,
                external_task=external_task,
                external_headless=external_headless,
                **kwargs,
            )
            self.backend_name = self._delegate.backend_name
            self.using_fallback = False
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize external Isaac backend from unitree_rl_gym. "
                f"Reason: {exc}"
            ) from exc

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)

    def reset_all(self):
        return self._delegate.reset_all()

    def reset(self, env_ids: torch.Tensor):
        return self._delegate.reset(env_ids)

    def step(self, actions: torch.Tensor):
        return self._delegate.step(actions)

    def close(self):
        close_fn = getattr(self._delegate, "close", None)
        if callable(close_fn):
            close_fn()


IsaacGymVecEnv = IsaacVecEnv
