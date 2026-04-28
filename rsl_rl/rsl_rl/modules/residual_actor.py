import numpy as np
import torch
import torch.nn as nn

from rsl_rl.modules.net_model import MLP


class ResidualActor(nn.Module):
    """Frozen base actor plus a small trainable residual action head."""

    def __init__(
        self,
        base_actor,
        obs_shape,
        act_dim,
        hidden_dims=None,
        activation="elu",
        residual_scale=0.2,
        residual_min_scale=0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.base_actor = base_actor
        for param in self.base_actor.parameters():
            param.requires_grad_(False)

        obs_dim = int(np.prod(obs_shape))
        self.residual_net = nn.Sequential(*MLP(obs_dim, act_dim, hidden_dims, activation))
        self._zero_init_last_layer()

        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        self.target_residual_scale = float(residual_scale)
        self.residual_min_scale = float(residual_min_scale)
        self.z = 0
        self.last_base_action = None
        self.last_residual_action = None

    def _zero_init_last_layer(self):
        for layer in reversed(self.residual_net):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
                return

    def set_residual_scale(self, scale):
        scale = max(float(scale), self.residual_min_scale)
        scale = min(scale, self.target_residual_scale)
        self.residual_scale.fill_(scale)

    def forward(self, x, **kwargs):
        with torch.no_grad():
            base_action = self.base_actor(x, **kwargs)

        residual_input = x.flatten(start_dim=1)
        residual_action = self.residual_net(residual_input)
        self.z = getattr(self.base_actor, "z", 0)
        self.last_base_action = base_action.detach()
        self.last_residual_action = residual_action.detach()
        return base_action + self.residual_scale * residual_action

    def compute_adaptation_pred_loss(self, metrics):
        return self.residual_scale.new_zeros(())
