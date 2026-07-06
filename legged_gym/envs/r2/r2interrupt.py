from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.r2.r2 import R2Robot
from legged_gym.envs.r2.r2interrupt_config import R2InterruptCfg
from copy import deepcopy

class R2InterruptRobot(R2Robot):
    def __init__(self, cfg: R2InterruptCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.use_disturb = cfg.disturb.use_disturb
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.initial_disturb(cfg)

        # Reconstruct Command Scale
        command_scale = [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel] 
        command_dims = 3
        if cfg.env.observe_gait_commands:
            command_scale += [self.obs_scales.gait_freq_cmd, self.obs_scales.gait_phase_cmd,
                              self.obs_scales.gait_phase_cmd, self.obs_scales.footswing_height_cmd,] 
            self.command_gait_freq_dim = 3
            self.command_gait_phase_dim = 4
            self.command_gait_duration_dim = 5
            self.command_swing_heights_dim = 6
            command_dims = 7

        if cfg.env.observe_body_height:
            command_scale.append(self.obs_scales.body_height_cmd) 
            self.command_body_height_dim = command_dims 
            command_dims += 1
        if cfg.env.observe_body_pitch:
            command_scale.append(self.obs_scales.body_pitch_cmd) 
            self.command_body_pitch_dim = command_dims
            command_dims += 1
        if cfg.env.observe_waist_roll:
            command_scale.append(self.obs_scales.waist_roll_cmd) 
            self.command_waist_roll_dim = command_dims
            command_dims += 1
        if self.interrupt_in_command:
            command_scale.append(1) 
            self.command_interrupt_flag_dim = command_dims
            command_dims += 1

        self.commands_scale = torch.tensor(command_scale, device=self.device, requires_grad=False)
        for name in self.curriculum_thresholds['disturb'].keys():
            self.command_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _create_envs(self):
        super()._create_envs()
        self.disturb_termination_contact_indices = torch.zeros(len(self.cfg.disturb.disturb_terminate_assets), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.disturb.disturb_terminate_assets)):
            self.disturb_termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.disturb.disturb_terminate_assets[i])
        self.noise_disturb_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.use_disturb:
            self.noise_env_nums = int(self.num_envs * self.cfg.disturb.noise_curriculum_ratio)
            self.high_track_mode[:self.noise_env_nums] = False
            self.noise_disturb_mode[:self.noise_env_nums] = True
        else:
            self.noise_env_nums = 0

    def initial_disturb(self, cfg: R2InterruptCfg):
        self.use_disturb = cfg.disturb.use_disturb
        self.disturb_dim = cfg.disturb.disturb_dim
        disturb_action_indices = getattr(cfg.disturb, "disturb_action_indices", None)
        if disturb_action_indices is None:
            raise ValueError("R2 interrupt requires explicit cfg.disturb.disturb_action_indices")
        if len(disturb_action_indices) != self.disturb_dim:
            raise ValueError("disturb_action_indices length must match cfg.disturb.disturb_dim")
        self.disturb_action_indices = torch.tensor(disturb_action_indices, dtype=torch.long, device=self.device, requires_grad=False)
        self.non_disturb_action_indices = torch.tensor(
            [i for i in range(self.num_dof) if i not in disturb_action_indices],
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        self.default_disturb_dof_pos = self.default_dof_pos.index_select(1, self.disturb_action_indices)
        self.disturb_dof_pos_limits = self.dof_pos_limits.index_select(0, self.disturb_action_indices)
        self.disturb_scale = cfg.disturb.disturb_scale
        self.disturb_switch_prob = cfg.disturb.switch_prob
        self.disturb_actions = torch.zeros(self.num_envs, self.disturb_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.executed_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturb_masks = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.disturb_noise_ratio = cfg.disturb.noise_ratio
        self.disturb_isnoise = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.interrupt_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.disturb_replace_action = cfg.disturb.replace_action
        self.disturb_rad = cfg.disturb.disturb_rad
        self.disturb_uniform = cfg.disturb.uniform_noise
        self.disturb_noise_update_step = cfg.disturb.noise_update_step
        self.disturb_noise_scale = torch.tensor(cfg.disturb.noise_scale).to(self.device).unsqueeze(0) 
        self.disturb_noise_lowerbound = torch.tensor(cfg.disturb.noise_lowerbound).to(self.device).unsqueeze(0) #+ 0.15
        self.disturb_uniform_scale = cfg.disturb.uniform_scale
        self.disturb_in_last_action = cfg.disturb.disturb_in_last_action
        self.obs_target_interrupt_in_privilege = cfg.disturb.obs_target_interrupt_in_privilege
        self.obs_executed_actions_in_privilege = cfg.disturb.obs_executed_actions_in_privilege
        self.start_disturb_by_curriculum = cfg.disturb.start_by_curriculum
        self.staged_disturb_release = getattr(cfg.disturb, "staged_release", False)
        self.staged_disturb_init_curriculum_to_level = bool(
            getattr(cfg.disturb, "stage_init_curriculum_to_level", False)
        )
        self.staged_disturb_levels = torch.tensor(
            getattr(cfg.disturb, "stage_levels", [0.0, 0.25, 0.5, 0.75, 1.0]),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        if self.staged_disturb_levels.numel() == 0:
            raise ValueError("cfg.disturb.stage_levels must not be empty")
        self.staged_disturb_stage_idx = 0
        self.staged_disturb_min_episodes = int(getattr(cfg.disturb, "stage_min_episodes", 512))
        self.staged_disturb_min_task_returns = self._expand_staged_disturb_gate_values(
            getattr(cfg.disturb, "stage_min_task_return", 20.0),
            "stage_min_task_return",
        )
        self.staged_disturb_max_fall_rates = self._expand_staged_disturb_gate_values(
            getattr(cfg.disturb, "stage_max_fall_rate", 0.10),
            "stage_max_fall_rate",
        )
        self.staged_disturb_min_task_return = self.staged_disturb_min_task_returns[0]
        self.staged_disturb_max_fall_rate = self.staged_disturb_max_fall_rates[0]
        self.staged_disturb_monitor_noise_only = bool(getattr(cfg.disturb, "stage_monitor_noise_only", True))
        self.staged_disturb_monitor_expert = getattr(cfg.disturb, "stage_monitor_expert", None)
        if self.staged_disturb_monitor_expert == "":
            self.staged_disturb_monitor_expert = None
        if self.staged_disturb_monitor_expert not in (None, "walk", "run", "jump"):
            raise ValueError("cfg.disturb.stage_monitor_expert must be one of None, walk, run, or jump")
        self.command_profile_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device, requires_grad=False
        )
        profile_mixture = getattr(cfg.commands, "profile_mixture", None)
        self.command_profile_names = (
            [str(profile.get("name", idx)) for idx, profile in enumerate(profile_mixture)]
            if profile_mixture
            else []
        )
        raw_monitor_profiles = getattr(cfg.disturb, "stage_monitor_profiles", None)
        if raw_monitor_profiles == "":
            raw_monitor_profiles = None
        if isinstance(raw_monitor_profiles, str):
            raw_monitor_profiles = [raw_monitor_profiles]
        self.staged_disturb_monitor_profiles = raw_monitor_profiles
        self.staged_disturb_monitor_profile_ids = None
        if raw_monitor_profiles is not None:
            profile_to_id = {name: idx for idx, name in enumerate(self.command_profile_names)}
            missing_profiles = [name for name in raw_monitor_profiles if name not in profile_to_id]
            if missing_profiles:
                raise ValueError(
                    "cfg.disturb.stage_monitor_profiles entries must match cfg.commands.profile_mixture names"
                )
            self.staged_disturb_monitor_profile_ids = torch.tensor(
                [profile_to_id[name] for name in raw_monitor_profiles],
                dtype=torch.long,
                device=self.device,
                requires_grad=False,
            )
        self.staged_disturb_regress_on_failure = bool(getattr(cfg.disturb, "stage_regress_on_failure", False))
        self.staged_disturb_regress_patience = max(1, int(getattr(cfg.disturb, "stage_regress_patience", 2)))
        self.staged_disturb_failure_windows = 0
        self.staged_disturb_episode_count = 0
        self.staged_disturb_return_sum = 0.0
        self.staged_disturb_fall_sum = 0.0
        if cfg.disturb.disturb_rad_curriculum:
            init_curriculum = (
                self._current_staged_disturb_level()
                if self.staged_disturb_release and self.staged_disturb_init_curriculum_to_level
                else 0.0
            )
            # Later-stage resume experiments can start at the current staged
            # cap instead of relearning easy disturbance levels; see Bengio
            # et al. 2009 and OpenAI et al. 2019 automatic domain randomization.
            self.disturb_rad_curriculum = torch.full(
                (self.num_envs,),
                init_curriculum,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        else:
            self.disturb_rad_curriculum = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self._cap_staged_disturb_curriculum()

        if hasattr(cfg.disturb, "disturb_terminate_assets"):
            self.disturb_terminate_assets = cfg.disturb.disturb_terminate_assets
        else:
            self.disturb_terminate_assets = []

        self.num_steps = 0
        self.interrupt_in_command = cfg.disturb.interrupt_in_cmd
        self.stand_interrupt_only = cfg.disturb.stand_interrupt_only

    def _expand_staged_disturb_gate_values(self, raw_values, field_name):
        """Expand scalar staged-release gates or validate per-stage gate lists."""
        num_stages = int(self.staged_disturb_levels.numel())
        if isinstance(raw_values, (list, tuple)):
            if len(raw_values) != num_stages:
                raise ValueError(f"cfg.disturb.{field_name} list length must match cfg.disturb.stage_levels")
            return [float(value) for value in raw_values]
        return [float(raw_values)] * num_stages

    def _current_staged_disturb_gate(self):
        gate_idx = min(self.staged_disturb_stage_idx, len(self.staged_disturb_min_task_returns) - 1)
        return (
            self.staged_disturb_min_task_returns[gate_idx],
            self.staged_disturb_max_fall_rates[gate_idx],
        )

    def _current_staged_disturb_level(self):
        if not self.staged_disturb_release:
            return float(self.cfg.disturb.max_curriculum)
        return float(self.staged_disturb_levels[self.staged_disturb_stage_idx].item())

    def _cap_staged_disturb_curriculum(self):
        if not self.staged_disturb_release or not hasattr(self, "disturb_rad_curriculum"):
            return
        stage_level = self._current_staged_disturb_level()
        self.disturb_rad_curriculum[:] = torch.clamp(self.disturb_rad_curriculum, max=stage_level)

    def _staged_disturb_expert_mask(self, env_ids):
        """Return staged-release monitor envs using the same command semantics as AMP routing."""
        if self.staged_disturb_monitor_expert is None:
            return torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        if len(env_ids) == 0:
            return torch.zeros(0, dtype=torch.bool, device=self.device)

        commands = self.commands[env_ids]
        amp_cfg = getattr(self.cfg, "amp", None)
        run_velocity_threshold = float(getattr(amp_cfg, "expert_run_velocity_threshold", 1.0))
        run_frequency_threshold = float(getattr(amp_cfg, "expert_run_frequency_threshold", 2.0))
        jump_swing_height_threshold = float(getattr(amp_cfg, "expert_jump_swing_height_threshold", 0.18))
        jump_body_height_threshold = float(getattr(amp_cfg, "expert_jump_body_height_threshold", 0.02))

        # Match R2Robot.get_amp_expert_ids(): jump wins first, then run, and
        # the remainder is walk. This keeps staged release aligned with the
        # discriminator/motion-prior route used by r2amp.
        is_jump = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        if commands.shape[1] > 7:
            is_jump = (
                (commands[:, 4] == 0)
                & (
                    (commands[:, 6] >= jump_swing_height_threshold)
                    | (commands[:, 7] > jump_body_height_threshold)
                )
            )

        is_run = torch.abs(commands[:, 0]) > run_velocity_threshold
        if commands.shape[1] > 3:
            is_run = is_run | (commands[:, 3] >= run_frequency_threshold)
        is_run = (~is_jump) & is_run

        if self.staged_disturb_monitor_expert == "jump":
            return is_jump
        if self.staged_disturb_monitor_expert == "run":
            return is_run
        return ~(is_jump | is_run)

    def _record_staged_disturb_episode_stats(self, env_ids):
        if (
            not self.staged_disturb_release
            or len(env_ids) == 0
            or not getattr(self, "init_done", False)
        ):
            return

        monitor_ids = env_ids
        if self.staged_disturb_monitor_noise_only and hasattr(self, "noise_disturb_mode"):
            monitor_ids = monitor_ids[self.noise_disturb_mode[monitor_ids]]
        if self.staged_disturb_monitor_expert is not None:
            monitor_ids = monitor_ids[self._staged_disturb_expert_mask(monitor_ids)]
        if self.staged_disturb_monitor_profile_ids is not None:
            profile_mask = torch.isin(
                self.command_profile_ids[monitor_ids],
                self.staged_disturb_monitor_profile_ids,
            )
            monitor_ids = monitor_ids[profile_mask]
        if len(monitor_ids) == 0:
            return

        task_returns = torch.zeros(len(monitor_ids), dtype=torch.float, device=self.device)
        for values in self.episode_sums.values():
            task_returns += values[monitor_ids]

        falls = (~self.time_out_buf[monitor_ids]).float()
        self.staged_disturb_episode_count += int(monitor_ids.numel())
        self.staged_disturb_return_sum += float(task_returns.sum().detach().cpu().item())
        self.staged_disturb_fall_sum += float(falls.sum().detach().cpu().item())

    def _maybe_advance_staged_disturb_release(self):
        if not self.staged_disturb_release:
            return
        if self.staged_disturb_episode_count < self.staged_disturb_min_episodes:
            self._cap_staged_disturb_curriculum()
            return

        avg_task_return = self.staged_disturb_return_sum / max(self.staged_disturb_episode_count, 1)
        fall_rate = self.staged_disturb_fall_sum / max(self.staged_disturb_episode_count, 1)
        min_task_return, max_fall_rate = self._current_staged_disturb_gate()
        can_advance = (
            avg_task_return >= min_task_return
            and fall_rate <= max_fall_rate
            and self.staged_disturb_stage_idx < int(self.staged_disturb_levels.numel()) - 1
        )
        if can_advance:
            self.staged_disturb_stage_idx += 1
            self.staged_disturb_failure_windows = 0
        elif avg_task_return >= min_task_return and fall_rate <= max_fall_rate:
            self.staged_disturb_failure_windows = 0
        elif self.staged_disturb_regress_on_failure and self.staged_disturb_stage_idx > 0:
            self.staged_disturb_failure_windows += 1
            if self.staged_disturb_failure_windows >= self.staged_disturb_regress_patience:
                # Adaptive curricula lower difficulty after repeated failure
                # windows; see Bengio et al. 2009 and OpenAI et al. 2019 ADR.
                self.staged_disturb_stage_idx -= 1
                self.staged_disturb_failure_windows = 0

        # Use non-overlapping windows so a bad phase cannot be hidden by older,
        # easier data after the stage cap changes.
        self.staged_disturb_episode_count = 0
        self.staged_disturb_return_sum = 0.0
        self.staged_disturb_fall_sum = 0.0
        self._cap_staged_disturb_curriculum()

    def _disturb_values(self, tensor):
        return tensor.index_select(1, self.disturb_action_indices)

    def _non_disturb_values(self, tensor):
        return tensor.index_select(1, self.non_disturb_action_indices)

    def _apply_command_profile_mixture(self, env_ids):
        """Optionally replace rectangular commands with weighted eval-like profiles."""
        profiles = getattr(self.cfg.commands, "profile_mixture", None)
        if not profiles:
            return False
        if not isinstance(profiles, (list, tuple)):
            raise ValueError("cfg.commands.profile_mixture must be a list of profile objects")

        weights = torch.tensor(
            [float(profile.get("weight", 1.0)) for profile in profiles],
            dtype=torch.float,
            device=self.device,
        )
        if torch.any(weights < 0) or float(weights.sum().item()) <= 0.0:
            raise ValueError("cfg.commands.profile_mixture weights must be non-negative and sum to > 0")

        choices = torch.multinomial(weights / weights.sum(), len(env_ids), replacement=True)
        self.standing_envs_mask[env_ids] = False
        self.command_profile_ids[env_ids] = choices.to(dtype=torch.long)
        for profile_idx, profile in enumerate(profiles):
            selected_env_ids = env_ids[choices == profile_idx]
            if len(selected_env_ids) == 0:
                continue

            command = torch.tensor(profile["command"], dtype=self.commands.dtype, device=self.device)
            jitter = torch.tensor(
                profile.get("jitter", [0.0] * int(command.numel())),
                dtype=self.commands.dtype,
                device=self.device,
            )
            if command.numel() != jitter.numel():
                raise ValueError("cfg.commands.profile_mixture command and jitter lengths must match")

            command_tensor = torch.zeros(self.commands.shape[1], dtype=self.commands.dtype, device=self.device)
            jitter_tensor = torch.zeros_like(command_tensor)
            command_dims = min(command_tensor.numel(), command.numel())
            command_tensor[:command_dims] = command[:command_dims]
            jitter_tensor[:command_dims] = jitter[:command_dims]

            noise = (2.0 * torch.rand(len(selected_env_ids), self.commands.shape[1], device=self.device) - 1.0)
            self.commands[selected_env_ids] = command_tensor + noise * jitter_tensor
            # Fixed-preset evaluation clears standing_envs_mask, so profile
            # mixture keeps that eval-like default unless a profile opts in.
            if bool(profile.get("standing", False)):
                self.standing_envs_mask[selected_env_ids] = True

        # Keep jittered profiles inside the declared command support so the
        # profile mixture remains compatible with the existing curriculum grid.
        self.commands[env_ids, 0].clip_(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1])
        self.commands[env_ids, 1].clip_(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1])
        self.commands[env_ids, 2].clip_(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1])
        if self.cfg.env.observe_gait_commands:
            self.commands[env_ids, self.command_gait_freq_dim].clip_(
                self.command_ranges["gait_frequency"][0],
                self.command_ranges["gait_frequency"][1],
            )
            self.commands[env_ids, self.command_swing_heights_dim].clip_(
                self.command_ranges["foot_swing_height"][0],
                self.command_ranges["foot_swing_height"][1],
            )
        if self.cfg.env.observe_body_height:
            self.commands[env_ids, self.command_body_height_dim].clip_(
                self.command_ranges["body_height"][0],
                self.command_ranges["body_height"][1],
            )
        if self.cfg.env.observe_body_pitch:
            self.commands[env_ids, self.command_body_pitch_dim].clip_(
                self.command_ranges["body_pitch"][0],
                self.command_ranges["body_pitch"][1],
            )
        if self.cfg.env.observe_waist_roll:
            self.commands[env_ids, self.command_waist_roll_dim].clip_(
                self.command_ranges["waist_roll"][0],
                self.command_ranges["waist_roll"][1],
            )

        self.velocity_level[env_ids] = torch.clip(
            torch.norm(self.commands[env_ids, :2], dim=-1) + 0.5 * torch.abs(self.commands[env_ids, 2]),
            min=1,
        )
        return True
        

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        if len(env_ids) == 0:
            return
        
        # update vel commands:
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # update high speed envs
        heading_mask = self.terrain_curriculum_mode[env_ids]
        self.heading_cmd[env_ids[heading_mask]] = torch_rand_float(
            self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids[heading_mask]), 1), device=self.device).squeeze(1)
        
        high_track_mask = self.high_track_mode[env_ids[~heading_mask]]
        high_track_envs = env_ids[~heading_mask][high_track_mask]
        if len(high_track_envs) > 0 and self.cfg.commands.curriculum:
            self.update_command_curriculum_grid(high_track_envs)
        
        if self.use_disturb:
            disturb_ready_mask = ~heading_mask
            if not self.start_disturb_by_curriculum:
                # Decouple interrupt ablations from terrain/command curriculum release.
                disturb_ready_mask = torch.ones_like(heading_mask)
            disturb_env_ids = env_ids[disturb_ready_mask]
            update_disturb_mask = self.noise_disturb_mode[disturb_env_ids]
            update_disturb_envs = disturb_env_ids[update_disturb_mask]
            noise_disturb_mask = self.disturb_masks[disturb_env_ids]
            noise_disturb_envs = disturb_env_ids[noise_disturb_mask]
            if len(update_disturb_envs) > 0 and self.cfg.disturb.disturb_rad_curriculum:
                self.update_disturb_curriculum_grid(update_disturb_envs, noise_disturb_envs)
        
        # sample envs as standing
        standing_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_standing = 1 / 10
        standing_env_ids = env_ids[torch.logical_and(0 <= standing_env_floats, standing_env_floats < probability_standing)] 
        none_standing_env_ids = env_ids[~torch.logical_and(0 <= standing_env_floats, standing_env_floats < probability_standing)]
        self.standing_envs_mask[standing_env_ids] = True 
        self.standing_envs_mask[none_standing_env_ids] = False 
        self.commands[standing_env_ids, :3] = 0 

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_vel).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_vel)

        # Velocity_level. 
        self.velocity_level[env_ids] = torch.clip(1.0*torch.norm(self.commands[env_ids, :2], dim=-1)+0.5*torch.abs(self.commands[env_ids, 2]), min=1)
        
        # clip commands for high speed envs
        # high_speed_env_mask = self.velocity_level[env_ids] > 1.8
        # self.commands[env_ids[high_speed_env_mask], 3] = self.commands[env_ids[high_speed_env_mask], 3].clip(min=2.0)  # Frequency

        if self.cfg.env.observe_gait_commands:
            # update gait commands
            self.commands[env_ids, self.command_gait_freq_dim] = torch_rand_float(self.command_ranges["gait_frequency"][0], self.command_ranges["gait_frequency"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # Frequency
            phases = torch.tensor([0, 0.5], device=self.device)
            random_indices = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
            self.commands[env_ids, self.command_gait_phase_dim] = phases[random_indices] # phases
            self.commands[env_ids, self.command_gait_duration_dim] = 0.5  # durations
            self.commands[env_ids, self.command_swing_heights_dim] = torch_rand_float(self.command_ranges["foot_swing_height"][0], self.command_ranges["foot_swing_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # swing_heights

            hopping_mask = self.commands[env_ids, 4] == 0
            walking_mask = self.commands[env_ids, 4] == 0.5
            hopping_env_ids = env_ids[hopping_mask]
            walking_env_ids = env_ids[walking_mask]

        if self.cfg.env.observe_body_height:
            self.commands[env_ids, self.command_body_height_dim] = torch_rand_float(self.command_ranges["body_height"][0], self.command_ranges['body_height'][1], (len(env_ids), 1), device=self.device).squeeze(1)

        if self.cfg.env.observe_body_pitch:
            self.commands[env_ids, self.command_body_pitch_dim] = torch_rand_float(self.command_ranges["body_pitch"][0], self.command_ranges['body_pitch'][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            # clip body_pitch for hopping
            if self.cfg.env.observe_gait_commands:
                self.commands[hopping_env_ids, self.command_body_pitch_dim] = self.commands[hopping_env_ids, self.command_body_pitch_dim].clip(max=0.3)

        if self.cfg.env.observe_waist_roll:
            self.commands[env_ids, self.command_waist_roll_dim] = torch_rand_float(self.command_ranges["waist_roll"][0], self.command_ranges['waist_roll'][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self._apply_command_profile_mixture(env_ids)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
        
        if self.interrupt_in_command:
            self.commands[env_ids, self.command_interrupt_flag_dim] = False
    
    def update_disturb_curriculum_grid(self, env_ids, noise_env_ids):
        if len(env_ids)==0: return
        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.max_episode_length, timesteps)

        # only for disturb masks
        curr_is_pass = torch.ones(len(noise_env_ids), dtype=bool, device=self.device)
        curr_is_down = torch.zeros(len(noise_env_ids), dtype=bool, device=self.device)

        for key, value in self.curriculum_thresholds['disturb'].items():
            all_rew = self.command_sums[key][noise_env_ids] / ep_len
            success_threshold = value * self.reward_scales[key]
            if key in self.curriculum_reward_list:
                success_threshold *= self.curriculum_scale

            curr_is_pass *= (all_rew > success_threshold)
            curr_is_down += (all_rew < success_threshold / 2)
        
        self.disturb_rad_curriculum[noise_env_ids] = torch.where(
            curr_is_down,
            (self.disturb_rad_curriculum[noise_env_ids] - 0.05).clip(min=0),
            torch.where(
                curr_is_pass,
                (self.disturb_rad_curriculum[noise_env_ids] + 0.05).clip(max=self.cfg.disturb.max_curriculum),
                self.disturb_rad_curriculum[noise_env_ids]
            )
        ) 
        self._cap_staged_disturb_curriculum()

        # resample all noise envs disturb
        self.disturb_masks[env_ids] = (torch.rand(len(env_ids))<=0.5).to(self.device) # Reset with half with disturb.
        is_noise = torch.rand(len(env_ids)) <= self.disturb_noise_ratio
        self.disturb_isnoise[env_ids] = is_noise.to(self.device)
        self.disturb_actions[env_ids] = self._disturb_values(self.dof_pos)[env_ids] - self.default_disturb_dof_pos
        if self.disturb_replace_action:
            self.interrupt_mask[env_ids] = self.disturb_masks[env_ids]
        else:
            self.interrupt_mask[env_ids] = self.disturb_masks[env_ids] * (~self.disturb_isnoise[env_ids])

    def _preprocess_obs(self):
        if self.interrupt_in_command:
            self.commands[:, self.command_interrupt_flag_dim] = self.interrupt_mask[:]
        super()._preprocess_obs()
        
    def add_other_privilege(self):
        if self.cfg.env.has_privileged_info and self.obs_target_interrupt_in_privilege:
            obs_target = self.disturb_actions * self.interrupt_mask.unsqueeze(-1)
            obs_target = torch.cat((obs_target, self.disturb_rad_curriculum.unsqueeze(-1)), dim=1)
            self.obs_buf = torch.cat((self.obs_buf, obs_target), dim=1)
        if self.cfg.env.has_privileged_info and self.obs_executed_actions_in_privilege:
            self.obs_buf = torch.cat((self.obs_buf, self.executed_actions), dim=1)

    def Gaussian_disturb_resample(self):
        '''
        Sample Gaussian Disturb Actions. 
        '''
        if self.disturb_dim == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        mean = torch.zeros(self.disturb_dim, device=self.device)
        std = torch.ones(self.disturb_dim, device=self.device) * self.disturb_scale

        return torch.clamp(
            torch.normal(mean, std) + self._disturb_values(self.dof_pos) - self.default_disturb_dof_pos,
            self.disturb_dof_pos_limits[:, 0].view(1,-1).repeat(self.num_envs, 1) - self.default_disturb_dof_pos,
            self.disturb_dof_pos_limits[:, 1].view(1,-1).repeat(self.num_envs, 1) - self.default_disturb_dof_pos
        )
    
    def Uniform_disturb_resample(self):
        '''Sample Noise from Uniform distribution'''
        if self.disturb_dim == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)

        scale = self.disturb_uniform_scale
        targets = scale * self.disturb_noise_scale * torch.rand((self.num_envs, self.disturb_dim), device=self.device) + self.disturb_noise_lowerbound + self.disturb_noise_scale * (1-scale)/2
        
        # Keep this legacy HugWBC 4+4 clipping branch out of R2's 10-slot full-arm contract.
        if self.disturb_dim == 8:
            left_env_mask = targets[:, 1] < 0.5
            targets[left_env_mask][:, [2, 3]] = 0
            right_env_maks =  targets[:, 5] > 0.5
            targets[right_env_maks][:, [6, 7]] = 0

        return torch.clamp(
            targets - self.default_disturb_dof_pos,
            self.disturb_dof_pos_limits[:, 0].view(1,-1).repeat(self.num_envs, 1) - self.default_disturb_dof_pos,
            self.disturb_dof_pos_limits[:, 1].view(1,-1).repeat(self.num_envs, 1) - self.default_disturb_dof_pos
        )

    def reset_idx(self, env_ids):         
        if hasattr(self, "staged_disturb_release"):
            self._record_staged_disturb_episode_stats(env_ids)
        super().reset_idx(env_ids)
        if self.use_disturb and self.cfg.disturb.disturb_rad_curriculum:
            self.extras['episode']['disturb_curriculum']= torch.mean(self.disturb_rad_curriculum[:self.noise_env_nums])
        if getattr(self, "staged_disturb_release", False):
            min_task_return, max_fall_rate = self._current_staged_disturb_gate()
            self.extras['episode']['staged_disturb_level'] = self._current_staged_disturb_level()
            self.extras['episode']['staged_disturb_stage'] = self.staged_disturb_stage_idx
            self.extras['episode']['staged_disturb_gate_min_task_return'] = min_task_return
            self.extras['episode']['staged_disturb_gate_max_fall_rate'] = max_fall_rate
            self.extras['episode']['staged_disturb_failure_windows'] = self.staged_disturb_failure_windows
            if self.staged_disturb_episode_count > 0:
                self.extras['episode']['staged_disturb_window_task_return'] = (
                    self.staged_disturb_return_sum / self.staged_disturb_episode_count
                )
                self.extras['episode']['staged_disturb_window_fall_rate'] = (
                    self.staged_disturb_fall_sum / self.staged_disturb_episode_count
                )
            
    def random_switch_disturb(self):
        switch_rand = torch.rand(self.num_envs, device=self.device)
        switch = switch_rand < self.disturb_switch_prob 
        self.disturb_masks = torch.where(switch, ~self.disturb_masks, self.disturb_masks)
        disturb_allowed_mask = ~self.terrain_curriculum_mode
        if not self.start_disturb_by_curriculum:
            # Keep noise-disturb partitioning but bypass the curriculum-mode gate.
            disturb_allowed_mask = torch.ones_like(self.terrain_curriculum_mode)
        self.disturb_masks[:] *= self.noise_disturb_mode[:] * disturb_allowed_mask
        if self.stand_interrupt_only:
            self.disturb_masks[:] *= self.standing_envs_mask[:]
        if self.disturb_replace_action:
            self.interrupt_mask[:] = self.disturb_masks[:]
        else:
            self.interrupt_mask[:] = self.disturb_masks[:] * (~self.disturb_isnoise[:])
            
    def resample_disturb_noise(self):
        self.disturb_actions = torch.where(
            self.disturb_isnoise.view(-1,1).repeat(1,self.disturb_dim),
            self.Uniform_disturb_resample()/self.cfg.control.action_scale if self.disturb_uniform else self.Gaussian_disturb_resample()/self.cfg.control.action_scale,
            self.disturb_actions
        )

    def post_physics_step(self):
        super().post_physics_step()
        self.num_steps += 1
        if self.use_disturb:
            if self.num_steps % self.disturb_noise_update_step == 0:
                self.resample_disturb_noise()
            self.num_steps %= self.disturb_noise_update_step          

    def training_curriculum(self):
        super().training_curriculum()
        self._maybe_advance_staged_disturb_release()
    
    def curriculum_disturb_fusion(self, actions):
        disturb_action = torch.clamp(
            self.disturb_actions,
            (- self.disturb_rad + self._disturb_values(self.dof_pos) - self.default_disturb_dof_pos) / self.cfg.control.action_scale,
            (self.disturb_rad + self._disturb_values(self.dof_pos) - self.default_disturb_dof_pos) / self.cfg.control.action_scale
        ) # Nosie or traj Target

        fused_disturb_action = self.disturb_rad_curriculum.unsqueeze(-1) * disturb_action +  \
                         (1 - self.disturb_rad_curriculum.unsqueeze(-1)) * self._disturb_values(actions)
        
        return fused_disturb_action

    def curriculum_disturb_clipping_mean(self, actions):
        # cliping mean with curriculum
        noise_mean = self.disturb_rad_curriculum.unsqueeze(-1) * (self._disturb_values(self.dof_pos) - self.default_disturb_dof_pos)+ \
                (1-self.disturb_rad_curriculum.unsqueeze(-1))  * (self._disturb_values(actions) * self.cfg.control.action_scale)

        disturb_actions = torch.clamp(
            self.disturb_actions,
            (- self.disturb_rad + noise_mean)/self.cfg.control.action_scale,
            (self.disturb_rad + noise_mean)/self.cfg.control.action_scale
        )
        return disturb_actions
    
    def curriculum_disturb_clipping_mean_rad(self, actions):
        # clipping mean with curriculum
        noise_mean = self.disturb_rad_curriculum.unsqueeze(-1) * (self._disturb_values(self.dof_pos) - self.default_disturb_dof_pos)+ \
                (1-self.disturb_rad_curriculum.unsqueeze(-1))  * (self._disturb_values(actions) * self.cfg.control.action_scale)
        
        # clipping action rate with curriculum by rad.
        disturb_actions = torch.clamp(
            self.disturb_actions,
            (- self.disturb_rad * self.disturb_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.control.action_scale,
            (self.disturb_rad * self.disturb_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.control.action_scale
        )
        return disturb_actions
        
    def calculate_action(self, actions):
        self.actions = actions.clone()
        clip_actions = self.cfg.normalization.clip_actions
        cliped_actions = torch.clip(actions.clone(), -clip_actions, clip_actions).to(self.device)
        if self.use_disturb:
            if self.cfg.disturb.disturb_curriculum_method == 0:
                disturb_action_clip = self.curriculum_disturb_fusion(cliped_actions)
            elif self.cfg.disturb.disturb_curriculum_method == 1:
                disturb_action_clip = self.curriculum_disturb_clipping_mean(cliped_actions)
            elif self.cfg.disturb.disturb_curriculum_method == 2:
                disturb_action_clip = self.curriculum_disturb_clipping_mean_rad(cliped_actions)

            if self.disturb_replace_action:
                cliped_actions[:, self.disturb_action_indices] = torch.where(
                    self.disturb_masks.view(-1, 1).repeat(1, self.disturb_dim),
                    disturb_action_clip,
                    self._disturb_values(cliped_actions)
                )
            else:
                # Apply additive interrupt only to the configured R2 arm joints.
                cliped_actions[:, self.disturb_action_indices] = torch.where(
                    self.disturb_masks.view(-1, 1).repeat(1, self.disturb_dim),
                    self._disturb_values(cliped_actions) + disturb_action_clip,
                    self._disturb_values(cliped_actions)
                )
            
            cliped_actions = torch.clip(cliped_actions, -clip_actions, clip_actions).to(self.device)
        if self.disturb_in_last_action:
            self.actions[:] = cliped_actions
        self.executed_actions[:] = cliped_actions
        return cliped_actions

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf[self.disturb_masks] = False
        self.large_ori_buf = torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        self.reset_buf |= self.large_ori_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
     
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        real_distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        des_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s
        
        # update level
        is_success_level = True
        for key, value in self.curriculum_thresholds['terrains_level'].items():
            task_reward = self.episode_sums[key][env_ids] / self.max_episode_length
            success_threshold = value * self.reward_scales[key] * torch.ones_like(task_reward)
            if key in self.curriculum_reward_list:
                success_threshold *= self.curriculum_scale
            is_success_level = is_success_level * (task_reward > success_threshold)
        
        self.terrain_levels[env_ids] -= 1 * (real_distance < des_distance * 0.5)
        self.terrain_levels[env_ids] += 1 * (real_distance > self.terrain.env_length / 2) * ~self.large_ori_buf[env_ids] * is_success_level
        
        self.max_reached_level[env_ids] = torch.where(self.terrain_levels[env_ids] > self.max_reached_level[env_ids],
                                                      self.terrain_levels[env_ids],
                                                      self.max_reached_level[env_ids])
        
        leave_max_level_envs = env_ids[self.terrain_levels[env_ids]>= self.max_terrain_level]
        
        self.terrain_levels[leave_max_level_envs] = torch.randint_like(leave_max_level_envs, 0, self.max_terrain_level)
        self.terrain_levels.clip_(min=0)

        if self.cfg.commands.curriculum:
            high_track_leave_envs = leave_max_level_envs[self.high_track_mode[leave_max_level_envs]]
            self.terrain_curriculum_mode[high_track_leave_envs] = False
            noise_disturb_leave_envs = leave_max_level_envs[self.noise_disturb_mode[leave_max_level_envs]]
            self.terrain_curriculum_mode[noise_disturb_leave_envs] = False

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],self.terrain_types[env_ids]]

    def _reward_shoulder_deviation(self):
        return super()._reward_shoulder_deviation() * (~self.interrupt_mask)
    
    def _reward_action_rate_upper(self): 
        if not self.use_disturb or self.disturb_dim <= 0:
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        diff_1 = torch.sum(torch.square(self._disturb_values(self.actions) - self._disturb_values(self.last_actions)), dim=1)
        diff_2 = torch.sum(torch.square(self._disturb_values(self.actions) - 2 * self._disturb_values(self.last_actions) + self._disturb_values(self.last_last_actions)), dim=1)
        return (diff_1 + diff_2) * (~self.interrupt_mask)
    
    def _reward_action_rate_lower(self): 
        if not self.use_disturb or self.disturb_dim <= 0:
            return super()._reward_action_rate()
        diff_1 = torch.sum(torch.square(self._non_disturb_values(self.actions) - self._non_disturb_values(self.last_actions)), dim=1)
        diff_2 = torch.sum(torch.square(self._non_disturb_values(self.actions) - 2 * self._non_disturb_values(self.last_actions) + self._non_disturb_values(self.last_last_actions)), dim=1)
        return diff_1 + diff_2

    def _reward_standing_joint_deviation(self):
        return super()._reward_standing_joint_deviation() * (~self.interrupt_mask)

    def _reward_collision(self):    
        # Penalize collisions on selected bodies For those caused by interruption , no penalty.
        return super()._reward_collision() * (~self.interrupt_mask)

    def _reward_feet_contact_forces(self):       
        # penalize high contact forces
        reward = torch.sum(
            torch.square(self.obs_scales.contact_force*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.)), 
            dim=1).clip(max=2.0)
        reward[self.standing_envs_mask] *= 0
        return reward

    def _reward_termination(self):
        # Terminal reward / penalty
        penaliez = self.reset_buf * ~self.time_out_buf
        return penaliez
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        if not self.use_disturb or self.disturb_dim <= 0:
            return super()._reward_dof_pos_limits()
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # Exempt only the explicitly interrupted arm joints, not the last N R2 joints.
        out_of_limits[:, self.disturb_action_indices] = 0
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        if not self.use_disturb or self.disturb_dim <= 0:
            return super()._reward_dof_acc()
        reward = torch.square((self.last_dof_vel - self.dof_vel) / self.dt)
        # Exempt only joints controlled by the external interrupt target.
        reward[:, self.disturb_action_indices] = 0
        return torch.sum(reward, dim=1)
    
    def _reward_dof_vel_limits(self):       
        # Penalize dof velocities too close to the limit
        if not self.use_disturb or self.disturb_dim <= 0:
            return super()._reward_dof_vel_limits()
        dof_vel_limits = torch.clip(10 * self.velocity_level.unsqueeze(-1).repeat(1,self.num_dof), min=10, max=20)
        error = torch.sum((torch.abs(self._disturb_values(self.dof_vel)) - self._disturb_values(dof_vel_limits)).clip(min=0., max=15.), dim=1)
        rew = 1 - torch.exp(-1 * error)
        return rew
