"""
Rollout storage for PPO training.

Stores transitions collected during rollout, computes GAE advantages,
and generates mini-batches for PPO updates.
"""

import torch
import numpy as np
from typing import Tuple, Optional


class RolloutStorage:
    """Experience buffer for on-policy PPO training."""

    class Transition:
        """Single transition data container."""
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.teacher_actions = None  # pre-computed distillation targets

        def clear(self):
            self.__init__()

    def __init__(self, num_envs: int, num_transitions_per_env: int,
                 obs_shape: list, privileged_obs_shape: list,
                 actions_shape: list, device: str = 'cpu',
                 teacher_actions_shape: list = None):
        """Initialize rollout storage.

        Args:
            num_envs: Number of parallel environments
            num_transitions_per_env: Steps per rollout per env
            obs_shape: Actor observation shape (e.g., [5, 58] for 3D or [313] flat)
            privileged_obs_shape: Critic observation shape (e.g., [99])
            actions_shape: Action shape (e.g., [15])
            device: Torch device
        """
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core buffers: (T, N, ...)
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=device)

        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=device)
        else:
            self.privileged_observations = None

        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device)
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device, dtype=torch.uint8)

        # PPO buffers
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device)
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device)
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=device)

        # Optional teacher action buffer for knowledge distillation
        if teacher_actions_shape is not None:
            self.teacher_actions = torch.zeros(
                num_transitions_per_env, num_envs, *teacher_actions_shape, device=device)
        else:
            self.teacher_actions = None

        self.step = 0

    def add_transitions(self, transition: 'RolloutStorage.Transition'):
        """Add a transition to the buffer."""
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if self.teacher_actions is not None and transition.teacher_actions is not None:
            self.teacher_actions[self.step].copy_(transition.teacher_actions)
        self.step += 1

    def clear(self):
        """Reset step counter (buffers are overwritten)."""
        self.step = 0

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        """Compute GAE returns and advantages.

        Args:
            last_values: (N, 1) value estimates for the last state
            gamma: Discount factor
            lam: GAE lambda
        """
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (self.rewards[step]
                     + next_is_not_terminal * gamma * next_values
                     - self.values[step])
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Normalize advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        """Generate shuffled mini-batches for PPO updates.

        Yields:
            tuple of (obs, critic_obs, actions, values, advantages, returns,
                     old_log_probs, old_mu, old_sigma, hidden_states, masks)
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        # Flatten time and env dimensions
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        teacher_actions_flat = (
            self.teacher_actions.flatten(0, 1)
            if self.teacher_actions is not None else None
        )

        for epoch in range(num_epochs):
            indices = torch.randperm(
                num_mini_batches * mini_batch_size,
                requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    critic_observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),  # hidden states (unused for MLP)
                    None,          # masks (unused for MLP)
                    teacher_actions_flat[batch_idx] if teacher_actions_flat is not None else None,
                )
