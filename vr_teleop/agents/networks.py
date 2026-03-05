"""
Neural network architectures for G1 multi-gait policy.

MlpAdaptModel (from HugWBC):
    HistoryEncoder: (batch, H, proprio_dim) → flatten → MLP → latent
    StateEstimator: latent → privileged prediction (e.g., base_lin_vel)
    LowLevelController: cat(latent, pred, current_proprio, cmd) → actions

Standalone implementation, no isaacgym dependency.
"""

import torch
import torch.nn as nn
from typing import List, Optional


def get_activation(name: str) -> nn.Module:
    """Get activation module by name."""
    activations = {
        'elu': nn.ELU(),
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'selu': nn.SELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def build_mlp(input_dim: int, output_dim: int, hidden_dims: List[int],
              activation: str, output_activation: str = None) -> nn.Sequential:
    """Build a multi-layer perceptron.

    Architecture: input → hidden[0] → ... → hidden[-1] → output
    """
    act = get_activation(activation)
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dims[0]))
    layers.append(act)
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        layers.append(get_activation(activation))
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    if output_activation is not None:
        layers.append(get_activation(output_activation))
    return nn.Sequential(*layers)


class MlpAdaptModel(nn.Module):
    """Adaptive MLP actor model with history encoding and state estimation.

    Architecture:
        1. HistoryEncoder: flattened proprioception history → latent
        2. StateEstimator: latent → privileged info prediction
        3. LowLevelController: cat(latent, prediction, current_proprio, cmd) → actions

    Input x: (batch, history_steps, single_step_dim) where
        single_step_dim = proprioception_dim + cmd_dim
    """

    def __init__(
        self,
        act_dim: int,                    # 15 (lower body DOFs)
        proprioception_dim: int,         # 51 (from obs config)
        cmd_dim: int,                    # 7 (commands + flags + clock)
        privileged_recon_dim: int = 3,   # state estimator output dim
        latent_dim: int = 32,            # history encoder output dim
        history_length: int = 5,         # number of history steps
        history_encoder_hidden: List[int] = None,   # [256, 128]
        state_estimator_hidden: List[int] = None,   # [64, 32]
        controller_hidden: List[int] = None,         # [512, 256, 128]
        activation: str = 'elu',
        output_activation: str = None,
        **kwargs,
    ):
        super().__init__()
        self.is_recurrent = False
        self.act_dim = act_dim
        self.proprioception_dim = proprioception_dim
        self.cmd_dim = cmd_dim
        self.privileged_recon_dim = privileged_recon_dim
        self.latent_dim = latent_dim
        self.history_length = history_length

        if history_encoder_hidden is None:
            history_encoder_hidden = [256, 128]
        if state_estimator_hidden is None:
            state_estimator_hidden = [64, 32]
        if controller_hidden is None:
            controller_hidden = [512, 256, 128]

        # History encoder: flattened proprioception → latent
        history_input_dim = proprioception_dim * history_length
        self.history_encoder = build_mlp(
            history_input_dim, latent_dim, history_encoder_hidden, activation)

        # State estimator: latent → privileged prediction
        self.state_estimator = build_mlp(
            latent_dim, privileged_recon_dim, state_estimator_hidden, activation)

        # Low-level controller: cat(latent, pred, proprio, cmd) → actions
        controller_input_dim = latent_dim + privileged_recon_dim + proprioception_dim + cmd_dim
        self.controller = build_mlp(
            controller_input_dim, act_dim, controller_hidden, activation,
            output_activation)

        # Losses tracked during training
        self.privileged_recon_loss = torch.tensor(0.0)
        self.z = None  # latent for logging

    def forward(self, x: torch.Tensor, privileged_obs: torch.Tensor = None,
                sync_update: bool = False, **kwargs) -> torch.Tensor:
        """Forward pass.

        Supports two input formats:
          2D flat: (B, current_obs(58) + history_proprio(H*51))
                   = (B, 58 + 5*51) = (B, 313)
          3D:      (B, H, single_step_dim) where single_step_dim = 58
                   Each step has proprio(51) + cmd(7)

        Args:
            x: Actor observations in either format
            privileged_obs: (batch, critic_obs_dim) for sync_update
            sync_update: if True, compute adaptation loss

        Returns:
            (batch, act_dim) actions
        """
        single_dim = self.proprioception_dim + self.cmd_dim  # 58

        if x.dim() == 3:
            # 3D: (B, H, 58) - extract from structured history
            pro_obs_seq = x[..., :self.proprioception_dim]  # (B, H, 51)
            cmd = x[:, -1, self.proprioception_dim:single_dim]  # (B, 7)
            current_proprio = x[:, -1, :self.proprioception_dim]  # (B, 51)
            history_flat = pro_obs_seq.flatten(-2, -1)  # (B, H*51)
        else:
            # 2D flat: (B, 58 + H*51) = (B, 313)
            # Layout: [current_obs(58), flattened_history_proprio(H*51)]
            current_obs = x[:, :single_dim]           # (B, 58)
            history_flat = x[:, single_dim:]           # (B, H*51 = 255)
            current_proprio = current_obs[:, :self.proprioception_dim]  # (B, 51)
            cmd = current_obs[:, self.proprioception_dim:single_dim]    # (B, 7)

        # History encoding
        latent = self.history_encoder(history_flat)  # (B, latent)
        self.z = latent

        # State estimation (privileged prediction)
        privileged_pred = self.state_estimator(latent)  # (B, pred_dim)

        # Controller
        controller_input = torch.cat([
            latent, privileged_pred, current_proprio, cmd
        ], dim=-1)
        actions = self.controller(controller_input)

        # Adaptation loss (privileged info reconstruction)
        if sync_update and privileged_obs is not None:
            # Extract ground truth privileged info from critic obs
            # In critic_obs: actor_obs(58) + [base_lin_vel(3), ...]
            # base_lin_vel is at indices 58:61
            priv_start = single_dim
            priv_gt = privileged_obs[:, priv_start:priv_start + self.privileged_recon_dim]
            self.privileged_recon_loss = 2.0 * (privileged_pred - priv_gt.detach()).pow(2).mean()
        else:
            self.privileged_recon_loss = torch.tensor(0.0, device=x.device)

        return actions

    def get_adaptation_loss(self) -> torch.Tensor:
        """Return the privileged info reconstruction loss."""
        return self.privileged_recon_loss
