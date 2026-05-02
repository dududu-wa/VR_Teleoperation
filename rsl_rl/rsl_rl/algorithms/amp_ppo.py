from collections import defaultdict

import torch

from rsl_rl.algorithms.ppo import PPO


class AMPPPO(PPO):
    def __init__(
        self,
        actor_critic,
        discriminator,
        amp_replay_buffer,
        env,
        disc_learning_rate=5e-5,
        disc_grad_penalty=5.0,
        disc_logit_reg=0.05,
        disc_weight_decay=1e-4,
        disc_reward_scale=2.0,
        disc_batch_size=4096,
        stage2_cfg=None,
        **ppo_kwargs,
    ):
        super().__init__(actor_critic, **ppo_kwargs)
        self.discriminator = discriminator
        self.disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=disc_learning_rate,
            weight_decay=disc_weight_decay,
        )
        self.amp_replay_buffer = amp_replay_buffer
        self.env = env

        self.disc_grad_penalty = disc_grad_penalty
        self.disc_logit_reg = disc_logit_reg
        self.disc_reward_scale = disc_reward_scale
        self.disc_batch_size = disc_batch_size
        self.stage2_cfg = stage2_cfg or {}
        self.stage2_enabled = bool(self.stage2_cfg.get("enable", False))
        self.stage2_style_reward_weight = float(self.stage2_cfg.get("style_reward_weight", 0.0))
        self.stage2_gait_reward_weight = float(self.stage2_cfg.get("gait_reward_weight", 0.0))
        self.arm_recovery_reward_weight = float(self.stage2_cfg.get("arm_recovery_reward_weight", 0.0))
        self.stage2_gait_reward_terms = self.stage2_cfg.get("gait_reward_terms", {})
        self.residual_action_penalty_weight = float(self.stage2_cfg.get("residual_action_penalty_weight", 0.0))
        self.arm_limit_penalty_weight = float(self.stage2_cfg.get("arm_limit_penalty_weight", 0.0))
        self.stage2_reward_dt_scale = bool(self.stage2_cfg.get("stage2_reward_dt_scale", True))
        self.stage2_update_count = 0
        self.residual_warmup_iters = int(self.stage2_cfg.get("residual_warmup_iters", 0))

        self.task_reward_collector = []
        self.amp_obs_collector = []
        self.style_reward_collector = []
        self.gated_style_reward_collector = []
        self.gait_reward_collector = []
        self.stage2_reward_collector = []
        self.stage2_reward_rate_collector = []
        self.stage2_style_reward_rate_collector = []
        self.stage2_gait_reward_rate_collector = []
        self.stage2_arm_recovery_reward_rate_collector = []
        self.stage2_residual_penalty_rate_collector = []
        self.stage2_arm_limit_penalty_rate_collector = []
        self.stage2_safe_collector = []
        self.residual_penalty_collector = []
        self.arm_recovery_reward_collector = []
        self.arm_recovery_error_collector = []
        self.arm_penalty_collector = []
        self.arm_limit_violation_collector = []
        self.arm_residual_target_collector = []
        self.residual_action_abs_collector = []
        self.arm_residual_action_abs_collector = []
        self.stage2_breakdown_collectors = defaultdict(list)
        self._update_residual_scale()

    def process_env_step(self, rewards, dones, infos):
        if not isinstance(infos, dict):
            raise TypeError(f"Expected infos to be dict, got {type(infos)}")
        if "amp_obs" not in infos:
            raise KeyError("AMP enabled but infos['amp_obs'] is missing")

        amp_obs = infos["amp_obs"]
        if not torch.is_tensor(amp_obs):
            raise TypeError(f"infos['amp_obs'] must be a torch.Tensor, got {type(amp_obs)}")
        if amp_obs.shape[0] != rewards.shape[0]:
            raise ValueError(
                f"AMP obs batch size {amp_obs.shape[0]} does not match rewards batch size {rewards.shape[0]}"
            )
        rl_device = torch.device(self.device)
        if amp_obs.device != rl_device:
            amp_obs = amp_obs.to(self.device)

        self.amp_obs_collector.append(amp_obs.clone())

        task_reward = rewards
        if task_reward.dim() > 1:
            task_reward = task_reward.view(task_reward.shape[0], -1)
            if task_reward.shape[1] != 1:
                raise ValueError(
                    f"Expected one reward per env, got reward shape {rewards.shape}"
                )
            task_reward = task_reward.squeeze(-1)
        self.task_reward_collector.append(task_reward.detach())

        with torch.no_grad():
            disc_logit = self.discriminator(amp_obs)
            style_reward = torch.clamp(1.0 - 0.25 * (disc_logit - 1.0).pow(2), min=0.0)
            style_reward = (style_reward * self.disc_reward_scale).squeeze(-1)

        safe_mask = torch.ones_like(style_reward, dtype=torch.bool)
        arm_penalty = torch.ones_like(style_reward)
        arm_limit_violation = torch.zeros_like(style_reward)
        gated_style_reward = style_reward
        self.style_reward_collector.append(style_reward.detach())

        ppo_rewards = rewards
        if self.stage2_enabled:
            safe_mask = self._compute_stage2_safe_mask(style_reward)
            arm_penalty, arm_limit_violation = self._compute_arm_limit_terms(style_reward)
            gated_style_reward = style_reward * safe_mask.float() * arm_penalty
            gait_reward = self._compute_gait_reward(style_reward)
            gated_gait_reward = gait_reward * safe_mask.float()
            arm_recovery_reward, arm_recovery_error = self._compute_arm_recovery_terms(style_reward)
            gated_arm_recovery_reward = arm_recovery_reward * safe_mask.float()
            residual_penalty = self._compute_residual_action_penalty(style_reward)
            arm_residual_target = self._compute_arm_residual_target(style_reward)
            residual_action_abs, arm_residual_action_abs = self._compute_residual_action_abs_terms(style_reward)
            style_reward_rate = self.stage2_style_reward_weight * gated_style_reward
            gait_reward_rate = self.stage2_gait_reward_weight * gated_gait_reward
            arm_recovery_reward_rate = self.arm_recovery_reward_weight * gated_arm_recovery_reward
            residual_penalty_rate = self.residual_action_penalty_weight * residual_penalty
            arm_limit_penalty_rate = self.arm_limit_penalty_weight * arm_limit_violation
            stage2_reward_rate = (
                style_reward_rate
                + gait_reward_rate
                + arm_recovery_reward_rate
                - residual_penalty_rate
                - arm_limit_penalty_rate
            )
            stage2_bonus = stage2_reward_rate * self._get_stage2_reward_scale(style_reward)
            if rewards.dim() > 1:
                ppo_rewards = rewards + stage2_bonus.unsqueeze(-1)
            else:
                ppo_rewards = rewards + stage2_bonus

            self.gated_style_reward_collector.append(gated_style_reward.detach())
            self.gait_reward_collector.append(gait_reward.detach())
            self.stage2_reward_collector.append(stage2_bonus.detach())
            self.stage2_reward_rate_collector.append(stage2_reward_rate.detach())
            self.stage2_style_reward_rate_collector.append(style_reward_rate.detach())
            self.stage2_gait_reward_rate_collector.append(gait_reward_rate.detach())
            self.stage2_arm_recovery_reward_rate_collector.append(arm_recovery_reward_rate.detach())
            self.stage2_residual_penalty_rate_collector.append(residual_penalty_rate.detach())
            self.stage2_arm_limit_penalty_rate_collector.append(arm_limit_penalty_rate.detach())
            self.stage2_safe_collector.append(safe_mask.float().detach())
            self.residual_penalty_collector.append(residual_penalty.detach())
            self.arm_recovery_reward_collector.append(gated_arm_recovery_reward.detach())
            self.arm_recovery_error_collector.append(arm_recovery_error.detach())
            self.arm_penalty_collector.append(arm_penalty.detach())
            self.arm_limit_violation_collector.append(arm_limit_violation.detach())
            self.arm_residual_target_collector.append(arm_residual_target.detach())
            self.residual_action_abs_collector.append(residual_action_abs.detach())
            self.arm_residual_action_abs_collector.append(arm_residual_action_abs.detach())
            infos["amp_gait_reward"] = gait_reward.detach()
            infos["amp_arm_recovery_reward"] = gated_arm_recovery_reward.detach()
            infos["amp_arm_recovery_error"] = arm_recovery_error.detach()
            infos["amp_stage2_reward"] = stage2_bonus.detach()
            infos["amp_stage2_safe"] = safe_mask.detach()
            infos["amp_residual_penalty"] = residual_penalty.detach()
            infos["amp_arm_style_penalty"] = arm_penalty.detach()
            infos["amp_arm_limit_violation"] = arm_limit_violation.detach()
            infos["amp_arm_residual_target"] = arm_residual_target.detach()
            infos["amp_residual_action_abs"] = residual_action_abs.detach()
            infos["amp_arm_residual_action_abs"] = arm_residual_action_abs.detach()

        infos["amp_task_reward"] = task_reward.detach()
        infos["amp_style_reward_raw"] = style_reward.detach()
        infos["amp_style_reward_gated"] = gated_style_reward.detach()
        infos["amp_style_reward"] = gated_style_reward.detach() if self.stage2_enabled else style_reward.detach()

        super().process_env_step(ppo_rewards, dones, infos)

    def update(self):
        metrics = super().update()
        self.stage2_update_count += 1
        self._update_residual_scale()

        if self.amp_obs_collector:
            all_amp_obs = torch.cat(self.amp_obs_collector, dim=0)
            self.amp_replay_buffer.insert(all_amp_obs)
            self.amp_obs_collector.clear()

        if self.task_reward_collector:
            metrics["task_reward"] = torch.cat(self.task_reward_collector).mean().item()
            self.task_reward_collector.clear()

        if self.style_reward_collector:
            metrics["style_reward"] = torch.cat(self.style_reward_collector).mean().item()
            self.style_reward_collector.clear()

        if self.gated_style_reward_collector:
            metrics["gated_style_reward"] = torch.cat(self.gated_style_reward_collector).mean().item()
            self.gated_style_reward_collector.clear()

        if self.gait_reward_collector:
            metrics["gait_reward"] = torch.cat(self.gait_reward_collector).mean().item()
            self.gait_reward_collector.clear()

        if self.stage2_reward_collector:
            metrics["stage2_reward"] = torch.cat(self.stage2_reward_collector).mean().item()
            self.stage2_reward_collector.clear()

        if self.stage2_reward_rate_collector:
            metrics["stage2_reward_rate"] = torch.cat(self.stage2_reward_rate_collector).mean().item()
            self.stage2_reward_rate_collector.clear()

        if self.stage2_style_reward_rate_collector:
            metrics["stage2_style_reward_rate"] = torch.cat(self.stage2_style_reward_rate_collector).mean().item()
            self.stage2_style_reward_rate_collector.clear()

        if self.stage2_gait_reward_rate_collector:
            metrics["stage2_gait_reward_rate"] = torch.cat(self.stage2_gait_reward_rate_collector).mean().item()
            self.stage2_gait_reward_rate_collector.clear()

        if self.stage2_arm_recovery_reward_rate_collector:
            metrics["stage2_arm_recovery_reward_rate"] = torch.cat(
                self.stage2_arm_recovery_reward_rate_collector
            ).mean().item()
            self.stage2_arm_recovery_reward_rate_collector.clear()

        if self.stage2_residual_penalty_rate_collector:
            metrics["stage2_residual_penalty_rate"] = torch.cat(self.stage2_residual_penalty_rate_collector).mean().item()
            self.stage2_residual_penalty_rate_collector.clear()

        if self.stage2_arm_limit_penalty_rate_collector:
            metrics["stage2_arm_limit_penalty_rate"] = torch.cat(self.stage2_arm_limit_penalty_rate_collector).mean().item()
            self.stage2_arm_limit_penalty_rate_collector.clear()

        if self.stage2_safe_collector:
            metrics["stage2_safe_fraction"] = torch.cat(self.stage2_safe_collector).mean().item()
            self.stage2_safe_collector.clear()

        if self.residual_penalty_collector:
            metrics["residual_action_penalty"] = torch.cat(self.residual_penalty_collector).mean().item()
            self.residual_penalty_collector.clear()

        if self.arm_recovery_reward_collector:
            metrics["arm_recovery_reward"] = torch.cat(self.arm_recovery_reward_collector).mean().item()
            self.arm_recovery_reward_collector.clear()

        if self.arm_recovery_error_collector:
            metrics["arm_recovery_error_abs_rad"] = torch.cat(self.arm_recovery_error_collector).mean().item()
            self.arm_recovery_error_collector.clear()

        if self.arm_penalty_collector:
            metrics["arm_style_penalty"] = torch.cat(self.arm_penalty_collector).mean().item()
            self.arm_penalty_collector.clear()

        if self.arm_limit_violation_collector:
            metrics["arm_limit_violation"] = torch.cat(self.arm_limit_violation_collector).mean().item()
            self.arm_limit_violation_collector.clear()

        if self.arm_residual_target_collector:
            metrics["arm_residual_target_abs_rad"] = torch.cat(self.arm_residual_target_collector).mean().item()
            self.arm_residual_target_collector.clear()

        if self.residual_action_abs_collector:
            metrics["residual_action_abs"] = torch.cat(self.residual_action_abs_collector).mean().item()
            self.residual_action_abs_collector.clear()

        if self.arm_residual_action_abs_collector:
            metrics["arm_residual_action_abs"] = torch.cat(self.arm_residual_action_abs_collector).mean().item()
            self.arm_residual_action_abs_collector.clear()

        if self.stage2_breakdown_collectors:
            for k, values in self.stage2_breakdown_collectors.items():
                if values:
                    metrics[k] = max(values) if k.startswith("max_") else sum(values) / len(values)
            self.stage2_breakdown_collectors.clear()

        if self.amp_replay_buffer.count > 0:
            metrics.update(self._update_discriminator())

        return metrics

    def _update_residual_scale(self):
        actor = getattr(self.actor_critic, "actor", None)
        if not self.stage2_enabled or not hasattr(actor, "set_residual_scale"):
            return
        target_scale = float(getattr(actor, "target_residual_scale", self.stage2_cfg.get("residual_scale", 0.2)))
        if self.residual_warmup_iters <= 0:
            actor.set_residual_scale(target_scale)
            return
        progress = min(1.0, max(1, self.stage2_update_count) / self.residual_warmup_iters)
        actor.set_residual_scale(target_scale * progress)

    def _get_stage2_reward_scale(self, like):
        if not self.stage2_reward_dt_scale:
            return 1.0
        dt = getattr(self.env, "dt", 1.0)
        if torch.is_tensor(dt):
            return dt.to(device=like.device, dtype=like.dtype)
        return float(dt)

    def _record_stage2_breakdown(self, breakdown):
        for key, value in breakdown.items():
            self.stage2_breakdown_collectors[key].append(float(value))

    def _as_vector_like(self, values, like):
        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=like.dtype, device=like.device)
        else:
            values = values.to(device=like.device, dtype=like.dtype)
        if values.dim() > 1:
            values = values.view(values.shape[0], -1)
            if values.shape[1] != 1:
                raise ValueError(f"Expected one reward value per env, got shape {tuple(values.shape)}")
            values = values.squeeze(-1)
        return values

    def _compute_gait_reward(self, like):
        reward = torch.zeros_like(like)
        for name, weight in self.stage2_gait_reward_terms.items():
            reward_fn = getattr(self.env, f"_reward_{name}", None)
            if reward_fn is None:
                continue
            term = self._as_vector_like(reward_fn(), like)
            reward = reward + float(weight) * term
        return reward

    def _compute_residual_action_penalty(self, like):
        actor = getattr(self.actor_critic, "actor", None)
        residual_action = getattr(actor, "last_residual_action", None)
        if residual_action is None:
            return torch.zeros_like(like)
        residual_action = residual_action.to(device=like.device, dtype=like.dtype)
        return torch.mean(residual_action.pow(2), dim=-1)

    def _compute_residual_action_abs_terms(self, like):
        actor = getattr(self.actor_critic, "actor", None)
        residual_action = getattr(actor, "last_residual_action", None)
        if residual_action is None:
            return torch.zeros_like(like), torch.zeros_like(like)

        residual_action = residual_action.to(device=like.device, dtype=like.dtype)
        residual_action_abs = torch.mean(torch.abs(residual_action), dim=-1)

        arm_indices = self.stage2_cfg.get("arm_dof_indices", None)
        if not arm_indices:
            return residual_action_abs, torch.zeros_like(like)

        idx = torch.as_tensor(arm_indices, dtype=torch.long, device=like.device)
        if idx.numel() == 0:
            return residual_action_abs, torch.zeros_like(like)
        if bool(torch.any(idx < 0).item()) or bool(torch.any(idx >= residual_action.shape[1]).item()):
            raise ValueError(
                f"stage2.arm_dof_indices must be within [0, {residual_action.shape[1] - 1}]"
            )
        arm_residual_action_abs = torch.mean(torch.abs(residual_action[:, idx]), dim=-1)
        return residual_action_abs, arm_residual_action_abs

    def _compute_arm_residual_target(self, like):
        actor = getattr(self.actor_critic, "actor", None)
        residual_action = getattr(actor, "last_residual_action", None)
        residual_scale = getattr(actor, "residual_scale", None)
        arm_indices = self.stage2_cfg.get("arm_dof_indices", None)
        if residual_action is None or residual_scale is None or not arm_indices:
            return torch.zeros_like(like)

        residual_action = residual_action.to(device=like.device, dtype=like.dtype)
        idx = torch.as_tensor(arm_indices, dtype=torch.long, device=like.device)
        if idx.numel() == 0:
            return torch.zeros_like(like)
        if bool(torch.any(idx < 0).item()) or bool(torch.any(idx >= residual_action.shape[1]).item()):
            raise ValueError(
                f"stage2.arm_dof_indices must be within [0, {residual_action.shape[1] - 1}]"
            )

        action_scale = getattr(self.env, "action_scale", None)
        if action_scale is None:
            action_scale = getattr(getattr(getattr(self.env, "cfg", None), "control", None), "action_scale", 1.0)
        if not torch.is_tensor(action_scale):
            action_scale = torch.tensor(action_scale, dtype=like.dtype, device=like.device)
        else:
            action_scale = action_scale.to(device=like.device, dtype=like.dtype)
        if action_scale.numel() == residual_action.shape[1]:
            action_scale = action_scale.flatten()[idx]
        elif action_scale.numel() != 1:
            raise ValueError(
                f"Expected scalar action_scale or {residual_action.shape[1]} entries, got {action_scale.numel()}"
            )

        if not torch.is_tensor(residual_scale):
            residual_scale = torch.tensor(residual_scale, dtype=like.dtype, device=like.device)
        else:
            residual_scale = residual_scale.to(device=like.device, dtype=like.dtype)
        target_delta = residual_action[:, idx] * residual_scale * action_scale
        return torch.mean(torch.abs(target_delta), dim=1)

    def _compute_arm_recovery_terms(self, like):
        env = self.env
        arm_indices = self.stage2_cfg.get("arm_dof_indices", None)
        if arm_indices is None or not hasattr(env, "dof_pos"):
            return torch.zeros_like(like), torch.zeros_like(like)

        sigma = float(self.stage2_cfg.get("arm_recovery_sigma", 0.35))
        if sigma <= 0:
            raise ValueError("stage2.arm_recovery_sigma must be positive")

        idx = torch.as_tensor(arm_indices, dtype=torch.long, device=env.dof_pos.device)
        if idx.numel() == 0:
            return torch.zeros_like(like), torch.zeros_like(like)
        if bool(torch.any(idx < 0).item()) or bool(torch.any(idx >= env.dof_pos.shape[1]).item()):
            raise ValueError(
                f"stage2.arm_dof_indices must be within [0, {env.dof_pos.shape[1] - 1}]"
            )

        pos = env.dof_pos[:, idx]
        target_cfg = self.stage2_cfg.get("arm_recovery_target", None)
        if target_cfg is None:
            if not hasattr(env, "default_dof_pos"):
                return torch.zeros_like(like), torch.zeros_like(like)
            target = env.default_dof_pos
            if target.dim() == 2:
                target = target[:, idx]
            else:
                target = target[idx].unsqueeze(0)
        else:
            target = torch.as_tensor(target_cfg, dtype=pos.dtype, device=pos.device).flatten()
            if target.numel() == env.dof_pos.shape[1]:
                target = target[idx].unsqueeze(0)
            elif target.numel() == idx.numel():
                target = target.unsqueeze(0)
            else:
                raise ValueError(
                    "stage2.arm_recovery_target must have one value per arm DOF or one value per action"
                )

        error = pos - target
        mean_sq_error = torch.mean(error.pow(2), dim=1)
        mean_abs_error = torch.mean(torch.abs(error), dim=1)
        reward = torch.exp(-mean_sq_error / (sigma * sigma))
        return reward.to(device=like.device, dtype=like.dtype), mean_abs_error.to(device=like.device, dtype=like.dtype)

    def _compute_stage2_safe_mask(self, like):
        safe = torch.ones_like(like, dtype=torch.bool)
        env = self.env
        breakdown = {}

        min_base_height = self.stage2_cfg.get("safety_min_base_height", None)
        if min_base_height is not None and hasattr(env, "root_states"):
            if hasattr(env, "heights_below_base"):
                base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.heights_below_base, dim=-1)
            else:
                base_height = env.root_states[:, 2]
            height_ok = base_height >= float(min_base_height)
            safe &= height_ok.to(device=like.device)
            breakdown["safe_height"] = height_ok.float().mean().item()

        max_roll = self.stage2_cfg.get("safety_max_roll", None)
        max_pitch = self.stage2_cfg.get("safety_max_pitch", None)
        if hasattr(env, "rpy"):
            if max_roll is not None:
                roll_ok = torch.abs(env.rpy[:, 0]) <= float(max_roll)
                safe &= roll_ok.to(device=like.device)
                breakdown["safe_roll"] = roll_ok.float().mean().item()
            if max_pitch is not None:
                pitch_ok = torch.abs(env.rpy[:, 1]) <= float(max_pitch)
                safe &= pitch_ok.to(device=like.device)
                breakdown["safe_pitch"] = pitch_ok.float().mean().item()

        contact_force = self.stage2_cfg.get("safety_contact_force", None)
        if contact_force is not None and hasattr(env, "contact_forces") and hasattr(env, "penalised_contact_indices"):
            contact_norm = torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1)
            if contact_norm.numel() == 0:
                contact_ok = torch.ones_like(like, dtype=torch.bool)
                max_contact_force = 0.0
            else:
                contact_ok = ~torch.any(contact_norm > float(contact_force), dim=1)
                max_contact_force = contact_norm.max().item()
            safe &= contact_ok.to(device=like.device)
            breakdown["safe_contact"] = contact_ok.float().mean().item()
            breakdown["max_penalised_contact_force"] = max_contact_force

        dof_margin = float(self.stage2_cfg.get("safety_dof_limit_margin", 0.0))
        if dof_margin > 0 and hasattr(env, "dof_pos_limits") and hasattr(env, "dof_pos"):
            dof_check_indices = self.stage2_cfg.get("safety_dof_check_indices", None)
            if dof_check_indices is not None:
                idx = dof_check_indices
                lower = env.dof_pos_limits[idx, 0] + dof_margin
                upper = env.dof_pos_limits[idx, 1] - dof_margin
                near_limit = torch.any((env.dof_pos[:, idx] < lower) | (env.dof_pos[:, idx] > upper), dim=1)
            else:
                lower = env.dof_pos_limits[:, 0] + dof_margin
                upper = env.dof_pos_limits[:, 1] - dof_margin
                near_limit = torch.any((env.dof_pos < lower) | (env.dof_pos > upper), dim=1)
            dof_ok = ~near_limit
            safe &= dof_ok.to(device=like.device)
            breakdown["safe_dof"] = dof_ok.float().mean().item()

        self._record_stage2_breakdown(breakdown)
        return safe

    def _compute_arm_limit_terms(self, like):
        """Return style attenuation and a normalized arm-limit violation."""
        env = self.env
        arm_indices = self.stage2_cfg.get("arm_dof_indices", None)
        if arm_indices is None or not hasattr(env, "dof_pos") or not hasattr(env, "dof_pos_limits"):
            return torch.ones_like(like), torch.zeros_like(like)

        margin = float(self.stage2_cfg.get("arm_dof_limit_margin", 0.1))
        scale = float(self.stage2_cfg.get("arm_style_penalty_scale", 5.0))
        if margin < 0:
            raise ValueError("stage2.arm_dof_limit_margin must be non-negative")

        idx = torch.as_tensor(arm_indices, dtype=torch.long, device=env.dof_pos.device)
        if idx.numel() == 0:
            return torch.ones_like(like), torch.zeros_like(like)
        if bool(torch.any(idx < 0).item()) or bool(torch.any(idx >= env.dof_pos.shape[1]).item()):
            raise ValueError(
                f"stage2.arm_dof_indices must be within [0, {env.dof_pos.shape[1] - 1}]"
            )
        pos = env.dof_pos[:, idx]                       # (num_envs, num_arm_dofs)
        lower = env.dof_pos_limits[idx, 0] + margin     # safe lower bound
        upper = env.dof_pos_limits[idx, 1] - margin     # safe upper bound
        if bool(torch.any(upper < lower).item()):
            raise ValueError("stage2.arm_dof_limit_margin leaves no safe range for at least one arm DOF")

        # How far each joint exceeds the safe zone (0 if inside)
        over_lower = (lower - pos).clamp(min=0.0)       # positive when below safe lower
        over_upper = (pos - upper).clamp(min=0.0)       # positive when above safe upper
        violation = over_lower + over_upper              # (num_envs, num_arm_dofs)

        total_violation = violation.sum(dim=1)           # (num_envs,)
        penalty = torch.exp(-scale * total_violation)    # 1.0 when safe, approaches 0 near limits

        if margin > 0:
            normalized_violation = (violation / margin).sum(dim=1)
        else:
            normalized_violation = total_violation

        return (
            penalty.to(device=like.device, dtype=like.dtype),
            normalized_violation.to(device=like.device, dtype=like.dtype),
        )

    def _update_discriminator(self):
        metrics = defaultdict(float)
        half_batch = self.disc_batch_size // 2

        agent_amp_obs = self.amp_replay_buffer.sample(half_batch).to(self.device)
        ref_amp_obs_3d = self.env.collect_reference_motions(half_batch)
        ref_amp_obs = ref_amp_obs_3d.view(half_batch, -1)
        if ref_amp_obs.device != torch.device(self.device):
            ref_amp_obs = ref_amp_obs.to(self.device)

        agent_logit = self.discriminator(agent_amp_obs)
        ref_logit = self.discriminator(ref_amp_obs)
        disc_loss = 0.5 * (agent_logit ** 2).mean() + 0.5 * ((ref_logit - 1) ** 2).mean()

        grad_penalty = self.discriminator.compute_grad_penalty(torch.cat([agent_amp_obs, ref_amp_obs], dim=0))
        logit_reg = (agent_logit ** 2).mean() + (ref_logit ** 2).mean()

        total_loss = disc_loss + self.disc_grad_penalty * grad_penalty + self.disc_logit_reg * logit_reg

        self.disc_optimizer.zero_grad()
        total_loss.backward()
        self.disc_optimizer.step()

        metrics["disc_loss"] = disc_loss.item()
        metrics["disc_grad_penalty"] = grad_penalty.item()
        metrics["disc_agent_logit"] = agent_logit.mean().item()
        metrics["disc_ref_logit"] = ref_logit.mean().item()
        return metrics
