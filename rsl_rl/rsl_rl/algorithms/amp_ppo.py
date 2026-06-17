from collections import defaultdict
from collections.abc import Mapping

import torch

from rsl_rl.algorithms.ppo import PPO


class AMPPPO(PPO):
    def __init__(
        self,
        actor_critic,
        *,
        discriminators,
        amp_replay_buffers,
        env,
        expert_style_enabled=None,
        disc_learning_rate=5e-5,
        disc_grad_penalty=5.0,
        disc_logit_reg=0.05,
        disc_weight_decay=1e-4,
        disc_reward_scale=5.0,
        style_reward_min=0.0,
        style_reward_max=5.0,
        normalize_style_reward=False,
        task_reward_weight=1.0,
        style_reward_weight=1.0,
        scale_style_reward_by_dt=False,
        style_reward_start_after=0,
        style_reward_warmup_iterations=0,
        style_reward_min_task_reward=None,
        style_reward_max_task_ratio=None,
        disc_batch_size=4096,
        **ppo_kwargs,
    ):
        super().__init__(actor_critic, **ppo_kwargs)
        if normalize_style_reward and style_reward_max <= style_reward_min:
            raise ValueError(
                "style_reward_max must be greater than style_reward_min "
                "when normalize_style_reward=True"
            )
        if task_reward_weight <= 0.0:
            raise ValueError("task_reward_weight must be positive")
        if style_reward_weight < 0.0:
            raise ValueError("style_reward_weight must be non-negative")
        if style_reward_start_after < 0:
            raise ValueError("style_reward_start_after must be non-negative")
        if style_reward_warmup_iterations < 0:
            raise ValueError("style_reward_warmup_iterations must be non-negative")
        if style_reward_max_task_ratio is not None and style_reward_max_task_ratio < 0.0:
            raise ValueError("style_reward_max_task_ratio must be non-negative")
        if scale_style_reward_by_dt and not hasattr(env, "dt"):
            raise ValueError("scale_style_reward_by_dt=True requires env.dt")
        if isinstance(discriminators, torch.nn.ModuleDict):
            self.discriminators = discriminators
        else:
            self.discriminators = torch.nn.ModuleDict({"default": discriminators})
        if isinstance(amp_replay_buffers, Mapping):
            self.amp_replay_buffers = dict(amp_replay_buffers)
        else:
            self.amp_replay_buffers = {"default": amp_replay_buffers}
        self.expert_names = list(self.discriminators.keys())
        if not self.expert_names:
            raise ValueError("AMPPPO requires at least one AMP discriminator")
        missing_buffers = [
            expert_name
            for expert_name in self.expert_names
            if expert_name not in self.amp_replay_buffers
        ]
        if missing_buffers:
            raise KeyError(f"Missing AMP replay buffers for experts: {missing_buffers}")
        # Keep single-expert aliases for legacy checkpoint and evaluation code paths.
        self.discriminator = self.discriminators[self.expert_names[0]]
        self.amp_replay_buffer = self.amp_replay_buffers[self.expert_names[0]]
        self.env = env
        self.expert_style_enabled = {expert_name: True for expert_name in self.expert_names}
        if expert_style_enabled is not None:
            self.expert_style_enabled.update(expert_style_enabled)
        self.disc_optimizers = {
            expert_name: torch.optim.AdamW(
                discriminator.parameters(),
                lr=disc_learning_rate,
                weight_decay=disc_weight_decay,
            )
            for expert_name, discriminator in self.discriminators.items()
        }
        self.disc_optimizer = self.disc_optimizers[self.expert_names[0]]

        self.disc_grad_penalty = disc_grad_penalty
        self.disc_logit_reg = disc_logit_reg
        self.disc_reward_scale = disc_reward_scale
        self.style_reward_min = style_reward_min
        self.style_reward_max = style_reward_max
        self.normalize_style_reward = normalize_style_reward
        self.task_reward_weight = task_reward_weight
        self.style_reward_weight = style_reward_weight
        self.style_reward_time_scale = float(env.dt) if scale_style_reward_by_dt else 1.0
        self.style_reward_start_after = int(style_reward_start_after)
        self.style_reward_warmup_iterations = int(style_reward_warmup_iterations)
        self.style_reward_min_task_reward = style_reward_min_task_reward
        self.style_reward_max_task_ratio = style_reward_max_task_ratio
        self.learning_iteration = 0
        self.disc_batch_size = disc_batch_size

        self.task_reward_collector = []
        self.task_reward_weighted_collector = []
        self.amp_obs_collector = []
        self.amp_expert_id_collector = []
        self.style_reward_raw_collector = []
        self.style_reward_normalized_collector = []
        self.style_reward_collector = []
        self.style_reward_schedule_collector = []
        self.style_reward_task_gate_collector = []
        self.mixed_reward_collector = []
        self.expert_fraction_collector = defaultdict(list)
        self.expert_style_reward_collector = defaultdict(list)

    def set_learning_iteration(self, it):
        self.learning_iteration = int(it)

    def _style_reward_schedule_weight(self):
        if self.learning_iteration < self.style_reward_start_after:
            return 0.0
        if self.style_reward_warmup_iterations > 0:
            warmup_step = self.learning_iteration - self.style_reward_start_after + 1
            return min(
                float(warmup_step) / float(self.style_reward_warmup_iterations),
                1.0,
            )
        return 1.0

    def _weight_style_reward(self, style_reward, task_reward, task_reward_weighted):
        schedule_weight = self._style_reward_schedule_weight()
        # AMP adds a style reward as an auxiliary RL reward (Peng et al. 2021,
        # arXiv:2104.02180); these gates keep task reward primary.
        style_reward_weighted = (
            self.style_reward_weight
            * self.style_reward_time_scale
            * schedule_weight
            * style_reward
        )

        task_gate = torch.ones_like(style_reward_weighted)
        if self.style_reward_min_task_reward is not None:
            task_gate = (task_reward >= self.style_reward_min_task_reward).to(
                style_reward_weighted.dtype
            )
            style_reward_weighted = style_reward_weighted * task_gate

        if self.style_reward_max_task_ratio is not None:
            max_style = task_reward_weighted.abs() * self.style_reward_max_task_ratio
            style_reward_weighted = torch.minimum(
                torch.clamp(style_reward_weighted, min=0.0),
                max_style,
            )

        return style_reward_weighted, task_gate

    def _resolve_expert_ids(self, infos, batch_size, device):
        # Single-expert AMP keeps the old env contract; multi-expert AMP must be
        # explicitly routed by the env-provided expert id for each sample.
        if "amp_expert_id" not in infos:
            if len(self.expert_names) > 1:
                raise KeyError("AMP experts require infos['amp_expert_id']")
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        expert_ids = infos["amp_expert_id"]
        if not torch.is_tensor(expert_ids):
            expert_ids = torch.as_tensor(expert_ids, dtype=torch.long).to(device)
        else:
            expert_ids = expert_ids.to(device=device, dtype=torch.long)
        expert_ids = expert_ids.view(-1)
        if expert_ids.shape[0] != batch_size:
            raise ValueError(
                f"AMP expert id batch size {expert_ids.shape[0]} does not match AMP obs batch size {batch_size}"
            )
        if expert_ids.numel() > 0:
            invalid = (expert_ids < 0) | (expert_ids >= len(self.expert_names))
            if torch.any(invalid):
                bad_ids = torch.unique(expert_ids[invalid]).detach().cpu().tolist()
                raise ValueError(f"Invalid AMP expert ids: {bad_ids}")
        return expert_ids

    def _routed_discriminator(self, amp_obs, expert_ids):
        # Each expert owns a discriminator; masked routing avoids mixing
        # expert-specific AMP scores before the canonical reward shaping step.
        if expert_ids.shape[0] != amp_obs.shape[0]:
            raise ValueError(
                f"AMP expert id batch size {expert_ids.shape[0]} does not match AMP obs batch size {amp_obs.shape[0]}"
            )
        if expert_ids.numel() > 0:
            invalid = (expert_ids < 0) | (expert_ids >= len(self.expert_names))
            if torch.any(invalid):
                bad_ids = torch.unique(expert_ids[invalid]).detach().cpu().tolist()
                raise ValueError(f"Invalid AMP expert ids: {bad_ids}")

        scores = amp_obs.new_empty((amp_obs.shape[0], 1))
        for expert_idx, expert_name in enumerate(self.expert_names):
            mask = expert_ids == expert_idx
            if torch.any(mask):
                scores[mask] = self.discriminators[expert_name](amp_obs[mask])
        return scores

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

        expert_ids = self._resolve_expert_ids(infos, amp_obs.shape[0], amp_obs.device)
        self.amp_obs_collector.append(amp_obs.clone())
        self.amp_expert_id_collector.append(expert_ids.clone())

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
            disc_score = self._routed_discriminator(amp_obs, expert_ids)
            # AMP style reward: max(0, 1 - 0.25 * (D - 1)^2).
            style_reward_base = torch.clamp(
                1.0 - 0.25 * torch.square(disc_score - 1.0),
                min=0.0,
                max=1.0,
            ).squeeze(-1)
            style_reward_raw = style_reward_base * self.disc_reward_scale
            style_reward_raw = torch.clamp(
                style_reward_raw,
                min=self.style_reward_min,
                max=self.style_reward_max,
            )

            style_reward = style_reward_raw
            if self.normalize_style_reward:
                style_reward = (
                    style_reward - self.style_reward_min
                ) / (self.style_reward_max - self.style_reward_min)
                style_reward = torch.clamp(style_reward, min=0.0, max=1.0)

        task_reward_weighted = self.task_reward_weight * task_reward
        style_reward_weighted, task_gate = self._weight_style_reward(
            style_reward,
            task_reward,
            task_reward_weighted,
        )
        for expert_idx, expert_name in enumerate(self.expert_names):
            mask = expert_ids == expert_idx
            fraction = mask.to(style_reward_weighted.dtype).mean()
            self.expert_fraction_collector[expert_name].append(fraction.detach())
            if not self.expert_style_enabled.get(expert_name, True):
                style_reward_weighted = style_reward_weighted.masked_fill(mask, 0.0)
        for expert_idx, expert_name in enumerate(self.expert_names):
            mask = expert_ids == expert_idx
            if torch.any(mask):
                expert_style_reward = style_reward_weighted[mask].mean()
            else:
                expert_style_reward = style_reward_weighted.new_tensor(0.0)
            self.expert_style_reward_collector[expert_name].append(
                expert_style_reward.detach()
            )
        mixed_reward = task_reward_weighted + style_reward_weighted

        self.task_reward_weighted_collector.append(task_reward_weighted.detach())
        self.style_reward_raw_collector.append(style_reward_raw.detach())
        self.style_reward_normalized_collector.append(style_reward.detach())
        self.style_reward_collector.append(style_reward_weighted.detach())
        schedule = torch.full_like(style_reward_weighted, self._style_reward_schedule_weight())
        self.style_reward_schedule_collector.append(schedule.detach())
        self.style_reward_task_gate_collector.append(task_gate.detach())
        self.mixed_reward_collector.append(mixed_reward.detach())

        infos["amp_task_reward"] = task_reward.detach()
        infos["amp_task_reward_contrib"] = task_reward_weighted.detach()
        infos["amp_task_reward_weighted"] = task_reward_weighted.detach()
        infos["amp_style_reward"] = style_reward_raw.detach()
        infos["amp_style_reward_raw"] = style_reward_raw.detach()
        infos["amp_style_reward_norm"] = style_reward.detach()
        infos["amp_style_reward_contrib"] = style_reward_weighted.detach()
        infos["amp_style_reward_schedule"] = schedule.detach()
        infos["amp_style_reward_task_gate"] = task_gate.detach()
        infos["amp_mixed_reward"] = mixed_reward.detach()

        super().process_env_step(mixed_reward, dones, infos)

    def update(self):
        metrics = super().update()
        task_reward_contrib_abs = None
        style_reward_contrib_abs = None

        if self.amp_obs_collector:
            all_amp_obs = torch.cat(self.amp_obs_collector, dim=0)
            if self.amp_expert_id_collector:
                all_expert_ids = torch.cat(self.amp_expert_id_collector, dim=0)
            else:
                all_expert_ids = torch.zeros(
                    all_amp_obs.shape[0],
                    dtype=torch.long,
                    device=all_amp_obs.device,
                )
            # Store policy AMP samples in the matching expert replay buffer so
            # discriminator updates compare policy and reference data per style.
            for expert_idx, expert_name in enumerate(self.expert_names):
                mask = all_expert_ids == expert_idx
                if torch.any(mask):
                    self.amp_replay_buffers[expert_name].insert(all_amp_obs[mask])
            self.amp_obs_collector.clear()
            self.amp_expert_id_collector.clear()

        if self.task_reward_collector:
            metrics["task_reward"] = torch.cat(self.task_reward_collector).mean().item()
            self.task_reward_collector.clear()

        if self.task_reward_weighted_collector:
            task_reward_contrib = torch.cat(self.task_reward_weighted_collector)
            metrics["task_reward_contrib"] = task_reward_contrib.mean().item()
            metrics["task_reward_weighted"] = metrics["task_reward_contrib"]
            task_reward_contrib_abs = task_reward_contrib.abs().mean().item()
            self.task_reward_weighted_collector.clear()

        if self.style_reward_raw_collector:
            metrics["style_reward_raw"] = torch.cat(self.style_reward_raw_collector).mean().item()
            self.style_reward_raw_collector.clear()

        if self.style_reward_normalized_collector:
            metrics["style_reward_normalized"] = torch.cat(self.style_reward_normalized_collector).mean().item()
            self.style_reward_normalized_collector.clear()

        if self.style_reward_collector:
            style_reward_contrib = torch.cat(self.style_reward_collector)
            metrics["style_reward_contrib"] = style_reward_contrib.mean().item()
            metrics["style_reward"] = metrics["style_reward_contrib"]
            style_reward_contrib_abs = style_reward_contrib.abs().mean().item()
            self.style_reward_collector.clear()

        for expert_name, values in self.expert_fraction_collector.items():
            if values:
                metrics[f"amp_expert_fraction/{expert_name}"] = torch.stack(values).mean().item()
        self.expert_fraction_collector.clear()

        for expert_name, values in self.expert_style_reward_collector.items():
            if values:
                metrics[f"style_reward_contrib/{expert_name}"] = torch.stack(values).mean().item()
        self.expert_style_reward_collector.clear()

        if self.style_reward_schedule_collector:
            metrics["style_reward_schedule"] = torch.cat(
                self.style_reward_schedule_collector
            ).mean().item()
            self.style_reward_schedule_collector.clear()

        if self.style_reward_task_gate_collector:
            metrics["style_reward_task_gate"] = torch.cat(
                self.style_reward_task_gate_collector
            ).mean().item()
            self.style_reward_task_gate_collector.clear()

        if task_reward_contrib_abs is not None and style_reward_contrib_abs is not None:
            metrics["style_to_task_abs_ratio"] = style_reward_contrib_abs / (task_reward_contrib_abs + 1e-8)

        if self.mixed_reward_collector:
            metrics["mixed_reward"] = torch.cat(self.mixed_reward_collector).mean().item()
            self.mixed_reward_collector.clear()

        if any(
            replay_buffer.count > 0
            for replay_buffer in self.amp_replay_buffers.values()
        ):
            metrics.update(self._update_discriminator())

        return metrics

    def _update_discriminator(self):
        metrics = {}
        aggregate_metrics = defaultdict(list)
        half_batch = self.disc_batch_size // 2

        for expert_idx, expert_name in enumerate(self.expert_names):
            replay_buffer = self.amp_replay_buffers[expert_name]
            if replay_buffer.count < half_batch:
                metrics[f"disc_update_skipped/{expert_name}"] = 1.0
                continue

            agent_amp_obs = replay_buffer.sample(half_batch).to(self.device)
            expert_ids = torch.full(
                (half_batch,),
                expert_idx,
                dtype=torch.long,
                device=self.device,
            )
            ref_amp_obs_3d = self.env.collect_reference_motions(
                half_batch,
                expert_ids=expert_ids,
            )
            ref_amp_obs = ref_amp_obs_3d.view(half_batch, -1)
            if ref_amp_obs.device != torch.device(self.device):
                ref_amp_obs = ref_amp_obs.to(self.device)

            discriminator = self.discriminators[expert_name]
            disc_optimizer = self.disc_optimizers[expert_name]
            agent_logit = discriminator(agent_amp_obs)
            ref_logit = discriminator(ref_amp_obs)
            # AMP uses an LSGAN discriminator objective (Mao et al. 2017,
            # ICCV) with -1 labels for policy samples and +1 labels for references.
            disc_loss = 0.5 * ((agent_logit + 1) ** 2).mean() + 0.5 * ((ref_logit - 1) ** 2).mean()

            grad_penalty = discriminator.compute_grad_penalty(torch.cat([agent_amp_obs, ref_amp_obs], dim=0))
            logit_reg = (agent_logit ** 2).mean() + (ref_logit ** 2).mean()

            total_loss = disc_loss + self.disc_grad_penalty * grad_penalty + self.disc_logit_reg * logit_reg

            disc_optimizer.zero_grad()
            total_loss.backward()
            disc_optimizer.step()

            per_expert_metrics = {
                "disc_loss": disc_loss.item(),
                "disc_grad_penalty": grad_penalty.item(),
                "disc_agent_logit": agent_logit.mean().item(),
                "disc_ref_logit": ref_logit.mean().item(),
            }
            for key, value in per_expert_metrics.items():
                metrics[f"{key}/{expert_name}"] = value
                aggregate_metrics[key].append(value)

        for key, values in aggregate_metrics.items():
            if values:
                metrics[key] = sum(values) / len(values)
        return metrics
