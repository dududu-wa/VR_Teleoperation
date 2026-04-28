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

        self.task_reward_collector = []
        self.amp_obs_collector = []
        self.style_reward_collector = []

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
            style_reward = -torch.log(1 - torch.sigmoid(disc_logit) + 1e-7)
            style_reward = (style_reward * self.disc_reward_scale).squeeze(-1)

        self.style_reward_collector.append(style_reward.detach())

        infos["amp_task_reward"] = task_reward.detach()
        infos["amp_style_reward"] = style_reward.detach()

        super().process_env_step(rewards, dones, infos)

    def update(self):
        metrics = super().update()

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

        if self.amp_replay_buffer.count > 0:
            metrics.update(self._update_discriminator())

        return metrics

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
