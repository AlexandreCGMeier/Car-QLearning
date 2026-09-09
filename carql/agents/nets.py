"""Network definitions (PyTorch)."""
from __future__ import annotations

import torch
from torch import nn

_ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}


def mlp(sizes: list[int], activation: str = "elu", out_act: bool = False) -> nn.Sequential:
    act = _ACT[activation]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


class QNet(nn.Module):
    """Plain or dueling Q-network."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int], activation: str = "elu",
                 dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.body = mlp([obs_dim, *hidden], activation, out_act=True)
        h = hidden[-1]
        if dueling:
            self.value = mlp([h, h, 1], activation)
            self.advantage = mlp([h, h, n_actions], activation)
        else:
            self.head = mlp([h, n_actions], activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if self.dueling:
            v = self.value(z)
            a = self.advantage(z)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(z)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: list[int], activation: str = "elu"):
        super().__init__()
        self.pi = mlp([obs_dim, *hidden, n_actions], activation)
        self.v = mlp([obs_dim, *hidden, 1], activation)
        # small init on the policy output keeps early exploration broad
        last = [m for m in self.pi if isinstance(m, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pi(x), self.v(x).squeeze(-1)
