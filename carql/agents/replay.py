"""Replay buffers: uniform and prioritised (sum tree), both vectorised."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int | None = None):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add_batch(self, obs, actions, rewards, next_obs, dones) -> np.ndarray:
        k = len(obs)
        idx = (self.ptr + np.arange(k)) % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.dones[idx] = dones
        self.ptr = (self.ptr + k) % self.capacity
        self.size = min(self.size + k, self.capacity)
        return idx

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        w = np.ones(batch_size, dtype=np.float32)
        return idx, w, self._get(idx)

    def _get(self, idx):
        return (self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx])

    def update_priorities(self, idx, td_errors) -> None:   # no-op for uniform sampling
        pass


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritised experience replay (Schaul et al. 2016)."""

    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6, beta0: float = 0.4,
                 eps: float = 1e-3, seed: int | None = None, clip: float = 0.0):
        super().__init__(capacity, obs_dim, seed)
        self.alpha, self.beta, self.eps, self.clip = alpha, beta0, eps, clip
        self.depth = int(np.ceil(np.log2(max(capacity, 2))))
        self.n_leaves = 1 << self.depth
        self.tree = np.zeros(2 * self.n_leaves, dtype=np.float64)   # tree[1] = root
        self.max_priority = 1.0

    def _set(self, leaf_idx: np.ndarray, priority: np.ndarray) -> None:
        i = leaf_idx + self.n_leaves
        self.tree[i] = priority
        i = i // 2
        while np.any(i >= 1):
            i = np.unique(i[i >= 1])
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i = i // 2

    def add_batch(self, obs, actions, rewards, next_obs, dones) -> np.ndarray:
        idx = super().add_batch(obs, actions, rewards, next_obs, dones)
        self._set(idx, np.full(len(idx), self.max_priority ** self.alpha))
        return idx

    def sample(self, batch_size: int):
        total = self.tree[1]
        seg = total / batch_size
        targets = (np.arange(batch_size) + self.rng.random(batch_size)) * seg
        i = np.ones(batch_size, dtype=np.int64)
        for _ in range(self.depth):
            left = 2 * i
            go_right = targets > self.tree[left]
            targets = np.where(go_right, targets - self.tree[left], targets)
            i = np.where(go_right, left + 1, left)
        leaf = i - self.n_leaves
        leaf = np.clip(leaf, 0, self.size - 1)
        p = self.tree[leaf + self.n_leaves] / total
        w = (self.size * p) ** (-self.beta)
        w = (w / w.max()).astype(np.float32)
        return leaf, w, self._get(leaf)

    def update_priorities(self, idx, td_errors) -> None:
        err = np.abs(td_errors)
        if self.clip > 0:
            err = np.minimum(err, self.clip)
        p = (err + self.eps) ** self.alpha
        self.max_priority = max(self.max_priority, float(err.max() + self.eps))
        self._set(np.asarray(idx), p)
