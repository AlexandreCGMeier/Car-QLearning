"""PPO (clipped objective, GAE) for the batched environment."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from ..config import AgentConfig
from .nets import ActorCritic


class PPOAgent:
    kind = "ppo"
    label = "PPO"

    def __init__(self, obs_dim: int, n_actions: int, cfg: AgentConfig, device: str = "cpu",
                 seed: int | None = None):
        self.cfg = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(device)
        self.net = ActorCritic(obs_dim, n_actions, cfg.hidden, cfg.activation).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0

    # ------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: float = 0.0) -> np.ndarray:
        """Greedy action (used by the viewer). ``eps`` is ignored."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        logits, _ = self.net(x)
        return logits.argmax(1).cpu().numpy()

    @torch.no_grad()
    def sample(self, obs: np.ndarray):
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        logits, v = self.net(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), v.cpu().numpy()

    @torch.no_grad()
    def scores(self, obs: np.ndarray) -> tuple[str, np.ndarray]:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        logits, _ = self.net(x)
        return "pi", torch.softmax(logits, 1).cpu().numpy()

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.net(x)[1].cpu().numpy()

    # ---------------------------------------------------------- learning
    def update(self, obs, actions, logp_old, returns, advantages) -> dict:
        """One PPO update over a flattened rollout (T*N samples)."""
        c = self.cfg
        dev = self.device
        obs = torch.as_tensor(obs, device=dev)
        actions = torch.as_tensor(actions, device=dev)
        logp_old = torch.as_tensor(logp_old, device=dev)
        returns = torch.as_tensor(returns, device=dev)
        adv = torch.as_tensor(advantages, device=dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(obs)
        mb = max(1, n // c.minibatches)
        stats = {"loss": 0.0, "entropy": 0.0, "clipfrac": 0.0}
        count = 0
        for _ in range(c.ppo_epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                i = perm[s:s + mb]
                logits, v = self.net(obs[i])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[i])
                ratio = torch.exp(logp - logp_old[i])
                pg = -torch.min(ratio * adv[i], torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv[i]).mean()
                vloss = F.mse_loss(v, returns[i])
                ent = dist.entropy().mean()
                loss = pg + c.value_coef * vloss - c.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max(c.grad_clip, 0.5))
                self.opt.step()
                stats["loss"] += float(loss.item())
                stats["entropy"] += float(ent.item())
                stats["clipfrac"] += float(((ratio - 1).abs() > c.clip).float().mean().item())
                count += 1
        self.n_updates += 1
        return {k: v / max(count, 1) for k, v in stats.items()}

    # -------------------------------------------------------- checkpoint
    def state(self) -> dict:
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict(), "n_updates": self.n_updates}

    def load_state(self, s: dict, strict: bool = True) -> None:
        self.net.load_state_dict(s["net"], strict=strict)
        if "opt" in s:
            try:
                self.opt.load_state_dict(s["opt"])
            except ValueError:
                pass
        self.n_updates = s.get("n_updates", 0)
