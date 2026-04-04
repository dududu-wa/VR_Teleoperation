import torch
import torch.nn as nn

from rsl_rl.modules.net_model import MLP


class AMPDiscriminator(nn.Module):
    def __init__(self, amp_obs_dim, hidden_dims=None, activation="relu"):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512]
        self.net = nn.Sequential(*MLP(amp_obs_dim, 1, hidden_dims, activation))

    def forward(self, amp_obs):
        return self.net(amp_obs)

    def compute_grad_penalty(self, amp_obs):
        amp_obs = amp_obs.clone().detach().requires_grad_(True)
        disc_out = self.forward(amp_obs)
        grad = torch.autograd.grad(disc_out.sum(), amp_obs, create_graph=True)[0]
        if torch.isnan(grad).any() or torch.isinf(grad).any():
            return torch.zeros((), device=amp_obs.device)
        return (grad.norm(2, dim=-1) ** 2).mean()
