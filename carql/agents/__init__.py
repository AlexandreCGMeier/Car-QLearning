"""Agent registry and checkpoint (de)serialisation.

Checkpoint = one ``.pt`` file containing the weights *and* everything needed to
rebuild the agent and its environment: the full config, the track name, the
episode/frame counters and the latest evaluation result.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..config import AgentConfig, Config
from .dqn import DQNAgent
from .ppo import PPOAgent

Agent = DQNAgent | PPOAgent


def make_agent(cfg: Config, obs_dim: int, n_actions: int, device: str = "cpu",
               seed: int | None = None) -> Agent:
    algo = cfg.agent.algo.lower()
    if algo == "dqn":
        return DQNAgent(obs_dim, n_actions, cfg.agent, device, seed)
    if algo == "ppo":
        return PPOAgent(obs_dim, n_actions, cfg.agent, device, seed)
    raise ValueError(f"unknown algo {cfg.agent.algo!r} (use 'dqn' or 'ppo')")


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    # tiny MLPs train fastest on CPU; GPUs only pay off with much bigger batches
    return "cpu"


def save_checkpoint(path: str | Path, agent: Agent, cfg: Config, *, episode: int, frames: int,
                    obs_dim: int, n_actions: int, eval: dict | None = None, extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 1,
        "algo": agent.kind,
        "label": agent.label,
        "config": cfg.to_dict(),
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "episode": int(episode),
        "frames": int(frames),
        "eval": eval or {},
        "agent": agent.state(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[Agent, dict]:
    """Rebuild the agent from a checkpoint. Returns (agent, metadata)."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = Config.from_dict(payload["config"])
    agent = make_agent(cfg, payload["obs_dim"], payload["n_actions"], device)
    agent.load_state(payload["agent"])
    if hasattr(agent, "net"):
        agent.net.eval()
    meta = {k: v for k, v in payload.items() if k != "agent"}
    meta["config"] = cfg
    meta["path"] = str(path)
    return agent, meta


__all__ = ["DQNAgent", "PPOAgent", "make_agent", "save_checkpoint", "load_checkpoint",
           "resolve_device", "AgentConfig"]
