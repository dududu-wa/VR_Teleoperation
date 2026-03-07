"""
Actor-Critic module for G1 multi-gait PPO training.

Actor: MlpAdaptModel (history encoder + state estimator + controller)
Critic: MLP (privileged obs → value)
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Optional

from vr_teleop.agents.networks import MlpAdaptModel, build_mlp


class ActorCritic(nn.Module):
    """Actor-Critic with MlpAdaptModel actor and MLP critic."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,          # history_steps * single_step_dim (e.g. 5*67=335)
        num_critic_obs: int,         # 96
        num_actions: int,            # 13
        # Actor config
        proprioception_dim: int = 61,
        cmd_dim: int = 6,
        history_length: int = 5,
        latent_dim: int = 32,
        privileged_recon_dim: int = 3,
        history_encoder_hidden: list = None,
        state_estimator_hidden: list = None,
        controller_hidden: list = None,
        # Critic config
        critic_hidden_dims: list = None,
        activation: str = 'elu',
        output_activation: str = None,
        # Noise config
        init_noise_std: float = 1.0,
        min_std: float = 0.1,
        max_std: float = 1.2,
        **kwargs,
    ):
        super().__init__()
        if history_encoder_hidden is None:
            history_encoder_hidden = [256, 128]
        if state_estimator_hidden is None:
            state_estimator_hidden = [64, 32]
        if controller_hidden is None:
            controller_hidden = [512, 256, 128]
        if critic_hidden_dims is None:
            critic_hidden_dims = [512, 256, 128]

        # ---- Actor (MlpAdaptModel) ----
        self.actor = MlpAdaptModel(
            act_dim=num_actions,
            proprioception_dim=proprioception_dim,
            cmd_dim=cmd_dim,
            privileged_recon_dim=privileged_recon_dim,
            latent_dim=latent_dim,
            history_length=history_length,
            history_encoder_hidden=history_encoder_hidden,
            state_estimator_hidden=state_estimator_hidden,
            controller_hidden=controller_hidden,
            activation=activation,
            output_activation=output_activation,
        )

        # ---- Critic (MLP: privileged obs → scalar value) ----
        self.critic = build_mlp(
            num_critic_obs, 1, critic_hidden_dims, activation)

        # ---- Action noise (learnable std) ----
        self._init_noise_std = init_noise_std
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.min_std = min_std
        self.max_std = max_std
        self.distribution = None
        self._warned_nonfinite_dist = False

        # Disable validation for speed
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        """Reset any internal state (unused for MLP, needed for interface)."""
        pass

    def forward(
        self,
        observations: torch.Tensor,
        privileged_obs: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Unified forward API.

        Returns deterministic action mean by default, matching inference usage.
        """
        if deterministic:
            actions_mean, _ = self.act_inference(
                observations, privileged_obs=privileged_obs, **kwargs)
            return actions_mean
        return self.act(
            observations, privileged_obs=privileged_obs, **kwargs)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor,
                            privileged_obs: torch.Tensor = None,
                            sync_update: bool = False, **kwargs):
        """Compute action distribution from observations."""
        mean = self.actor(observations, privileged_obs=privileged_obs,
                          sync_update=sync_update, **kwargs)
        if not torch.isfinite(mean).all():
            if not self._warned_nonfinite_dist:
                print("Warning: non-finite policy mean detected; sanitizing to keep training alive.")
                self._warned_nonfinite_dist = True
            mean = torch.nan_to_num(mean, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = torch.clamp(mean, -10.0, 10.0)

        std = torch.clamp(self.std, min=self.min_std, max=self.max_std)
        std = torch.nan_to_num(
            std,
            nan=self.min_std,
            posinf=self.max_std,
            neginf=self.min_std,
        )
        self.distribution = Normal(mean, mean * 0.0 + std)

    def act(self, observations: torch.Tensor,
            privileged_obs: torch.Tensor = None,
            sync_update: bool = False, **kwargs) -> torch.Tensor:
        """Sample actions from policy.

        Args:
            observations: (N, H, obs_dim) or (N, H*obs_dim) actor obs
            privileged_obs: (N, critic_dim) for sync_update

        Returns:
            (N, num_actions) sampled actions
        """
        self.update_distribution(observations, privileged_obs=privileged_obs,
                                sync_update=sync_update, **kwargs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor,
                      privileged_obs: torch.Tensor = None,
                      **kwargs):
        """Deterministic action for inference (no noise).

        Returns:
            (actions_mean, latent)
        """
        actions_mean = self.actor(observations, privileged_obs=privileged_obs,
                                  **kwargs)
        return actions_mean, self.actor.z

    def evaluate(self, critic_observations: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Compute value estimate from critic.

        Args:
            critic_observations: (N, critic_obs_dim)

        Returns:
            (N, 1) value estimates
        """
        return self.critic(critic_observations)
