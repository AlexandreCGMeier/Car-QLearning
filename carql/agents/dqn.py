"""DQN family: vanilla / double / dueling, optional prioritised replay."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from ..config import AgentConfig
from .nets import QNet
from .replay import PrioritizedReplayBuffer, ReplayBuffer


class DQNAgent:
    kind = "dqn"

    def __init__(self, obs_dim: int, n_actions: int, cfg: AgentConfig, device: str = "cpu",
                 seed: int | None = None):
        self.cfg = cfg
        self.obs_dim, self.n_actions = obs_dim, n_actions
        self.device = torch.device(device)
        self.net = QNet(obs_dim, n_actions, cfg.hidden, cfg.activation, cfg.dueling).to(self.device)
        self.target = copy.deepcopy(self.net).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0
        self.buffer: ReplayBuffer | None = None
        self.gamma_eff = cfg.gamma ** max(1, getattr(cfg, 'n_step', 1))   # bootstrap discount for n-step targets

    # ------------------------------------------------------------ naming
    @property
    def label(self) -> str:
        parts = []
        if self.cfg.dueling:
            parts.append("Dueling")
        parts.append("DDQN" if self.cfg.double else "DQN")
        if self.cfg.per:
            parts.append("+PER")
        return " ".join(parts).replace(" +", "+")

    # ---------------------------------------------------------- buffers
    def make_buffer(self, seed: int | None = None) -> ReplayBuffer:
        c = self.cfg
        if c.per:
            self.buffer = PrioritizedReplayBuffer(c.buffer_size, self.obs_dim, c.per_alpha, c.per_beta0,
                                                  c.per_eps, seed, clip=c.per_clip)
        else:
            self.buffer = ReplayBuffer(c.buffer_size, self.obs_dim, seed)
        return self.buffer

    # ------------------------------------------------------------ acting
    @torch.no_grad()
    def q_values(self, obs: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        return self.net(x).cpu().numpy()

    def act(self, obs: np.ndarray, eps: float = 0.0) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        greedy = self.q_values(obs).argmax(1)
        if eps <= 0:
            return greedy
        explore = self.rng.random(len(obs)) < eps
        rand = self.rng.integers(0, self.n_actions, size=len(obs))
        return np.where(explore, rand, greedy)

    def scores(self, obs: np.ndarray) -> tuple[str, np.ndarray]:
        """(label, per-action values) for the inspection HUD."""
        return "Q", self.q_values(obs)

    # ---------------------------------------------------------- learning
    def update(self, beta: float | None = None) -> float:
        c, buf = self.cfg, self.buffer
        if beta is not None and isinstance(buf, PrioritizedReplayBuffer):
            buf.beta = beta
        idx, w, (obs, act, rew, nobs, done) = buf.sample(c.batch_size)
        dev = self.device
        obs = torch.as_tensor(obs, device=dev)
        act = torch.as_tensor(act, device=dev)
        rew = torch.as_tensor(rew, device=dev)
        nobs = torch.as_tensor(nobs, device=dev)
        done = torch.as_tensor(done, device=dev)
        w = torch.as_tensor(w, device=dev)

        q = self.net(obs).gather(1, act[:, None]).squeeze(1)
        with torch.no_grad():
            if c.double:
                a_star = self.net(nobs).argmax(1, keepdim=True)
                q_next = self.target(nobs).gather(1, a_star).squeeze(1)
            else:
                q_next = self.target(nobs).max(1).values
            target = rew + self.gamma_eff * (1.0 - done) * q_next
        td = target - q
        if c.huber:
            loss = (w * F.smooth_l1_loss(q, target, reduction="none")).mean()
        else:
            loss = (w * td.pow(2)).mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), c.grad_clip)
        self.opt.step()
        buf.update_priorities(idx, td.detach().cpu().numpy())

        self.n_updates += 1
        if c.target_update > 0:
            if self.n_updates % c.target_update == 0:
                self.target.load_state_dict(self.net.state_dict())
        else:
            with torch.no_grad():
                for p, tp in zip(self.net.parameters(), self.target.parameters()):
                    tp.mul_(1 - c.polyak).add_(c.polyak * p)
        return float(loss.item())

    # -------------------------------------------------------- checkpoint
    def state(self) -> dict:
        return {"net": self.net.state_dict(), "opt": self.opt.state_dict(), "n_updates": self.n_updates}

    def load_state(self, s: dict, strict: bool = True) -> None:
        self.net.load_state_dict(s["net"], strict=strict)
        self.target.load_state_dict(s["net"], strict=strict)
        if "opt" in s:
            try:
                self.opt.load_state_dict(s["opt"])
            except ValueError:
                pass
        self.n_updates = s.get("n_updates", 0)
