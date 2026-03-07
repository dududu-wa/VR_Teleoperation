"""
PPO algorithm for G1 multi-gait training.

Features:
- Clipped surrogate objective
- Clipped value loss
- Entropy bonus
- G1 symmetry loss (left-right permutation matrices)
- Privileged info reconstruction loss (MlpAdaptModel adaptation)
- Adaptive KL-based learning rate schedule
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

from vr_teleop.agents.actor_critic import ActorCritic
from vr_teleop.agents.rollout_storage import RolloutStorage
from vr_teleop.utils.symmetry import build_action_symmetry_matrix, build_obs_symmetry_matrix


class PPO:
    """Proximal Policy Optimization with symmetry and adaptation losses."""

    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic: ActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        # Symmetry loss
        use_symmetry_loss: bool = True,
        symmetry_loss_coef: float = 0.5,
        # Adaptation loss (privileged info reconstruction)
        sync_update: bool = True,
        adaptation_loss_coef: float = 1.0,
        # KL schedule
        schedule: str = 'adaptive',
        desired_kl: float = 0.01,
        # Observation dim for symmetry matrix
        single_step_obs_dim: int = 67,
        # Distillation
        distillation_loss=None,
        device: str = 'cpu',
    ):
        self.device = device

        # KL schedule
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # Symmetry
        self.use_symmetry_loss = use_symmetry_loss
        self.symmetry_loss_coef = symmetry_loss_coef

        # Adaptation
        self.sync_update = sync_update
        self.adaptation_loss_coef = adaptation_loss_coef

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later via init_storage()
        self.optimizer = optim.AdamW(
            self.actor_critic.parameters(), lr=learning_rate,
            weight_decay=weight_decay)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Pre-build symmetry matrices (moved to device on first use)
        if self.use_symmetry_loss:
            self._act_perm_mat = build_action_symmetry_matrix().to(self.device)
            self._obs_perm_mat = build_obs_symmetry_matrix(
                single_step_obs_dim).to(self.device)
            self._obs_sym_dim = single_step_obs_dim
            # Proprioception symmetry (first 61 dims of obs symmetry)
            # Used for mirroring the history buffer
            self._proprio_dim = 61  # from ObsConfig (PROPRIO_END)
            self._proprio_perm_mat = self._obs_perm_mat[:self._proprio_dim,
                                                         :self._proprio_dim].clone()

        # Distillation loss (optional, for teacher-student training)
        self.distillation_loss = distillation_loss

    def reset_noise_std(self):
        """Reset action noise std to init_noise_std for fresh exploration.

        Useful when loading a pretrained model to allow the policy to
        re-explore from the pretrained weights.
        """
        init_val = getattr(self.actor_critic, '_init_noise_std', 1.0)
        self.actor_critic.std.data.fill_(init_val)
        print(f"  Reset noise std to {init_val}")

    def init_storage(self, num_envs: int, num_transitions_per_env: int,
                     actor_obs_shape: list, critic_obs_shape: list,
                     action_shape: list):
        """Initialize rollout storage buffer."""
        teacher_actions_shape = (
            [self.distillation_loss.teacher_action_dim]
            if self.distillation_loss is not None else None
        )
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env,
            actor_obs_shape, critic_obs_shape, action_shape,
            self.device,
            teacher_actions_shape=teacher_actions_shape)

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Compute actions, values, and log probs for rollout collection.

        Args:
            obs: (N, H, obs_dim) or (N, H*obs_dim) actor observations
            critic_obs: (N, critic_dim) critic observations

        Returns:
            (N, num_actions) sampled actions
        """
        self.transition.actions = self.actor_critic.act(
            obs, privileged_obs=critic_obs).detach()
        self.transition.values = self.actor_critic.evaluate(
            critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # Record obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # Pre-compute teacher targets during rollout (avoids LSTM contamination
        # from shuffled mini-batches and removes teacher inference from inner loop)
        if self.distillation_loss is not None:
            self.transition.teacher_actions = self.distillation_loss.teacher.get_action(
                obs).detach()

        return self.transition.actions

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor,
                         infos: dict):
        """Process environment step results and store transition.

        Handles timeout bootstrapping: if episode ended due to time limit
        (not failure), bootstrap the value estimate.
        """
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrap on timeouts (not true terminations)
        if 'time_outs' in infos:
            self.transition.rewards += (
                self.gamma
                * torch.squeeze(
                    self.transition.values
                    * infos['time_outs'].unsqueeze(1).to(self.device), 1))

        self.storage.add_transitions(self.transition)
        self.transition.clear()

        # Reset teacher LSTM hidden state for terminated environments so that
        # the next episode's teacher targets are not contaminated by the previous
        # episode's recurrent state.
        if self.distillation_loss is not None:
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                self.distillation_loss.teacher.reset(done_ids)

    def compute_returns(self, last_critic_obs: torch.Tensor):
        """Compute GAE returns using last critic observation."""
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> dict:
        """Run PPO update over stored rollout data.

        Returns:
            dict of mean metrics over all mini-batch updates
        """
        metrics = defaultdict(float)

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs)

        for (obs_batch, critic_obs_batch, actions_batch,
             target_values_batch, advantages_batch, returns_batch,
             old_actions_log_prob_batch, old_mu_batch, old_sigma_batch,
             hid_states_batch, masks_batch,
             teacher_actions_batch) in generator:

            # Forward pass through actor
            self.actor_critic.act(
                obs_batch,
                privileged_obs=critic_obs_batch,
                sync_update=self.sync_update)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # ---- Adaptive KL learning rate schedule ----
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
                        + (torch.square(old_sigma_batch)
                           + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # ---- Surrogate loss ----
            ratio = torch.exp(
                actions_log_prob_batch
                - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = (
                -torch.squeeze(advantages_batch)
                * torch.clamp(ratio,
                              1.0 - self.clip_param,
                              1.0 + self.clip_param))
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---- Value loss ----
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(
                    value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # ---- Adaptation loss (privileged info reconstruction) ----
            adaptation_loss = torch.tensor(0.0, device=self.device)
            if self.sync_update:
                adaptation_loss = (
                    self.adaptation_loss_coef
                    * self.actor_critic.actor.get_adaptation_loss())

            # ---- Symmetry loss ----
            sym_loss = torch.tensor(0.0, device=self.device)
            if self.use_symmetry_loss:
                sym_loss = self._compute_symmetry_loss(
                    obs_batch, critic_obs_batch)

            # ---- Distillation loss (teacher-student) ----
            distill_loss = torch.tensor(0.0, device=self.device)
            if self.distillation_loss is not None and teacher_actions_batch is not None:
                distill_loss = self.distillation_loss.compute_precomputed(
                    mu_batch, teacher_actions_batch)

            # ---- Total loss ----
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + sym_loss
                + adaptation_loss
                + distill_loss
            )

            # ---- Gradient step ----
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # ---- Track metrics ----
            metrics['surrogate'] += surrogate_loss.item()
            metrics['value_function'] += value_loss.item()
            metrics['entropy'] += entropy_batch.mean().item()
            metrics['sym_loss'] += sym_loss.item()
            metrics['adaptation_loss'] += adaptation_loss.item()
            metrics['distillation_loss'] += distill_loss.item()
            metrics['learning_rate'] += self.learning_rate
            metrics['ratio'] += ratio.mean().item()

        # Average over all updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in metrics:
            metrics[k] /= num_updates

        self.storage.clear()

        # Decay distillation coefficient after each PPO update
        if self.distillation_loss is not None:
            metrics['distillation_coef'] = self.distillation_loss.coef
            self.distillation_loss.step()

        return dict(metrics)

    def _compute_symmetry_loss(
        self, obs_batch: torch.Tensor, critic_obs_batch: torch.Tensor
    ) -> torch.Tensor:
        """Compute symmetry loss using G1 left-right permutation matrices.

        The idea: if we mirror the observation (swap left/right),
        the action should also be mirrored. We enforce:
            ||act(obs) - S_a @ act(S_o @ obs)||^2

        For flat (B, 372) obs = [current_obs(67), history_proprio(5*61)]:
            - Mirror current_obs(67) with S_obs(67, 67)
            - Mirror each 61-dim proprioception step with S_proprio(61, 61)
        """
        # Get original actions (deterministic)
        origin_act, _ = self.actor_critic.act_inference(
            obs_batch, privileged_obs=critic_obs_batch)

        # Mirror the observation
        if obs_batch.dim() == 3:
            # (B, H, 58) -> apply obs_perm to each step
            mirror_obs = torch.matmul(obs_batch, self._obs_perm_mat)
        else:
            # (B, 372) flat = [current_obs(67), history_proprio(H*61)]
            B = obs_batch.shape[0]
            current_obs = obs_batch[:, :self._obs_sym_dim]  # (B, 67)
            history_flat = obs_batch[:, self._obs_sym_dim:]  # (B, 305)

            # Mirror current observation
            mirror_current = torch.matmul(current_obs, self._obs_perm_mat)

            # Mirror history (H steps of 51-dim proprioception)
            H = history_flat.shape[-1] // self._proprio_dim
            if H > 0 and history_flat.shape[-1] == H * self._proprio_dim:
                history_3d = history_flat.view(B, H, self._proprio_dim)
                mirror_history_3d = torch.matmul(
                    history_3d, self._proprio_perm_mat)
                mirror_history = mirror_history_3d.view(B, -1)
            else:
                valid_hist_dim = (history_flat.shape[-1] // self._proprio_dim) * self._proprio_dim
                if valid_hist_dim > 0:
                    valid_hist = history_flat[:, :valid_hist_dim].view(B, -1, self._proprio_dim)
                    mirror_valid = torch.matmul(valid_hist, self._proprio_perm_mat).view(B, -1)
                    tail = history_flat[:, valid_hist_dim:]
                    mirror_history = torch.cat([mirror_valid, tail], dim=-1)
                else:
                    mirror_history = history_flat

            mirror_obs = torch.cat([mirror_current, mirror_history], dim=-1)

        # Get mirrored actions
        mirror_act, _ = self.actor_critic.act_inference(
            mirror_obs, privileged_obs=critic_obs_batch)

        # Apply action permutation to mirrored actions to "recover"
        recovery_act = torch.matmul(mirror_act, self._act_perm_mat)

        # Loss: how different are original and recovered actions
        sym_loss = self.symmetry_loss_coef * (
            origin_act.detach() - recovery_act).pow(2).mean()

        return sym_loss
