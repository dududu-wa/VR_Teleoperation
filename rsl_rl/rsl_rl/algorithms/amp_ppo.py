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
        self.stage2_gait_reward_terms = self.stage2_cfg.get("gait_reward_terms", {})
        self.residual_action_penalty_weight = float(self.stage2_cfg.get("residual_action_penalty_weight", 0.0))
        self.stage2_update_count = 0
        self.residual_warmup_iters = int(self.stage2_cfg.get("residual_warmup_iters", 0))

        self.task_reward_collector = []
        self.amp_obs_collector = []
        self.style_reward_collector = []
        self.gait_reward_collector = []
        self.stage2_reward_collector = []
        self.stage2_safe_collector = []
        self.residual_penalty_collector = []
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

        safe_mask = self._compute_stage2_safe_mask(style_reward)
        gated_style_reward = style_reward * safe_mask.float()
        self.style_reward_collector.append(style_reward.detach())

        ppo_rewards = rewards
        if self.stage2_enabled:
            gait_reward = self._compute_gait_reward(style_reward)
            gated_gait_reward = gait_reward * safe_mask.float()
            stage2_bonus = (
                self.stage2_style_reward_weight * gated_style_reward
                + self.stage2_gait_reward_weight * gated_gait_reward
            )
            residual_penalty = self._compute_residual_action_penalty(style_reward)
            stage2_bonus = stage2_bonus - self.residual_action_penalty_weight * residual_penalty
            if rewards.dim() > 1:
                ppo_rewards = rewards + stage2_bonus.unsqueeze(-1)
            else:
                ppo_rewards = rewards + stage2_bonus

            self.gait_reward_collector.append(gait_reward.detach())
            self.stage2_reward_collector.append(stage2_bonus.detach())
            self.stage2_safe_collector.append(safe_mask.float().detach())
            self.residual_penalty_collector.append(residual_penalty.detach())
            infos["amp_gait_reward"] = gait_reward.detach()
            infos["amp_stage2_reward"] = stage2_bonus.detach()
            infos["amp_stage2_safe"] = safe_mask.detach()
            infos["amp_residual_penalty"] = residual_penalty.detach()

        infos["amp_task_reward"] = task_reward.detach()
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

        if self.gait_reward_collector:
            metrics["gait_reward"] = torch.cat(self.gait_reward_collector).mean().item()
            self.gait_reward_collector.clear()

        if self.stage2_reward_collector:
            metrics["stage2_reward"] = torch.cat(self.stage2_reward_collector).mean().item()
            self.stage2_reward_collector.clear()

        if self.stage2_safe_collector:
            metrics["stage2_safe_fraction"] = torch.cat(self.stage2_safe_collector).mean().item()
            self.stage2_safe_collector.clear()

        if self.residual_penalty_collector:
            metrics["residual_action_penalty"] = torch.cat(self.residual_penalty_collector).mean().item()
            self.residual_penalty_collector.clear()

        if hasattr(self, "_last_safe_breakdown") and self._last_safe_breakdown:
            for k, v in self._last_safe_breakdown.items():
                metrics[k] = v

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
            safe &= height_ok
            breakdown["safe_height"] = height_ok.float().mean().item()

        max_roll = self.stage2_cfg.get("safety_max_roll", None)
        max_pitch = self.stage2_cfg.get("safety_max_pitch", None)
        if hasattr(env, "rpy"):
            if max_roll is not None:
                roll_ok = torch.abs(env.rpy[:, 0]) <= float(max_roll)
                safe &= roll_ok
                breakdown["safe_roll"] = roll_ok.float().mean().item()
            if max_pitch is not None:
                pitch_ok = torch.abs(env.rpy[:, 1]) <= float(max_pitch)
                safe &= pitch_ok
                breakdown["safe_pitch"] = pitch_ok.float().mean().item()

        contact_force = self.stage2_cfg.get("safety_contact_force", None)
        if contact_force is not None and hasattr(env, "contact_forces") and hasattr(env, "penalised_contact_indices"):
            contact_norm = torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1)
            contact_ok = ~torch.any(contact_norm > float(contact_force), dim=1)
            safe &= contact_ok
            breakdown["safe_contact"] = contact_ok.float().mean().item()
            breakdown["max_penalised_contact_force"] = contact_norm.max().item()

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
            safe &= dof_ok
            breakdown["safe_dof"] = dof_ok.float().mean().item()

        self._last_safe_breakdown = breakdown
        return safe

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
