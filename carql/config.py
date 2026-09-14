"""Configuration: plain dataclasses loaded from TOML, overridable from the CLI.

``configs/default.toml`` is the single place to look for every knob.  Any key
can be overridden on the command line with ``--set section.key=value``.
"""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"


@dataclass
class CarConfig:
    length: float = 25.0          # along the driving direction (world units)
    width: float = 15.0
    max_speed: float = 12.5       # world units per step
    min_speed: float = 0.78       # the car never stops (as in the original)
    acceleration: float = 0.156
    brake_factor: float = 2.0     # braking deceleration = brake_factor * acceleration
    friction: float = 0.98        # multiplicative per step
    turn_rate: float = 0.2        # radians per step at full steering authority
    steer_full_speed: float = 5.0 # below this speed steering authority scales down linearly
    drift_gain: float = 1.0       # 0 = no drift, 1 = original "Tokyo drift" amount
    drift_friction: float = 0.87
    drift_min_speed: float = 5.0


@dataclass
class EnvConfig:
    track: str = "classic"
    n_rays: int = 16
    fov: float = 360.0            # degrees covered by the rays
    ray_length: float = 600.0
    gate_lookahead: int = 2       # angles to the next N gates are part of the observation
    max_steps: int = 2000         # per episode
    stall_steps: int = 300        # end the episode if no gate is passed for this many steps
    reward_gate: float = 5.0
    reward_step: float = -0.01
    reward_progress: float = 0.01 # per world unit of forward progress toward the next gate
    reward_crash: float = -50.0
    reward_lap: float = 20.0      # bonus for completing a lap (closed tracks) / finishing (open)
    speed_bonus: float = 0.0      # optional: reward_progress-like bonus proportional to speed
    random_start: bool = False    # training aid: start at a random gate with random heading jitter
    start_jitter_deg: float = 0.0


@dataclass
class AgentConfig:
    algo: str = "dqn"             # "dqn" or "ppo"
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    activation: str = "elu"
    lr: float = 2.5e-4
    gamma: float = 0.99
    # --- dqn family
    double: bool = True
    dueling: bool = True
    per: bool = True
    per_alpha: float = 0.6
    per_beta0: float = 0.4
    per_eps: float = 1e-3
    buffer_size: int = 200_000
    batch_size: int = 128
    learning_starts: int = 5_000  # frames of random play before learning
    updates_per_step: int = 4     # gradient updates per batched env step
    n_step: int = 3               # multi-step returns
    target_update: int = 500      # gradient updates between hard target copies (0 = polyak)
    polyak: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frames: int = 500_000
    grad_clip: float = 10.0
    huber: bool = False
    reward_scale: float = 0.1     # rewards are multiplied by this before they enter the replay buffer
    per_clip: float = 5.0         # |TD error| is clipped to this for priorities (0 = no clip)
    # --- ppo
    rollout_steps: int = 128
    ppo_epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5


def agent_slug(a: "AgentConfig") -> str:
    """Short name of the agent variant, used for default run names (classic-dueling_per, ...)."""
    if a.algo.lower() == "ppo":
        return "ppo"
    base = "dueling" if a.dueling else ("ddqn" if a.double else "dqn")
    return base + ("_per" if a.per else "")


@dataclass
class TrainConfig:
    run: str = ""                 # run name; default derived from track + algo
    n_envs: int = 64              # cars simulated in parallel
    total_episodes: int = 3000
    checkpoint_every: int = 50    # episodes
    eval_every: int = 50          # episodes; greedy evaluation feeding the "best" checkpoint
    eval_envs: int = 8
    log_every: int = 50           # episodes
    seed: int = 0
    device: str = "auto"          # "auto", "cpu", "cuda", "mps"
    threads: int = 0              # torch CPU threads (0 = torch default)


@dataclass
class Config:
    car: CarConfig = field(default_factory=CarConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # --------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path | None = None, overrides: list[str] | None = None) -> "Config":
        cfg = cls()
        if path is not None:
            p = Path(path)
            if not p.exists():
                p = CONFIGS_DIR / (p.name if p.suffix else p.name + ".toml")
            with open(p, "rb") as f:
                data = tomllib.load(f)
            cfg = _merge(cfg, data)
        for ov in overrides or []:
            cfg.apply_override(ov)
        return cfg

    def apply_override(self, spec: str) -> None:
        if "=" not in spec:
            raise ValueError(f"override must look like section.key=value, got {spec!r}")
        key, value = spec.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"override key must be section.key, got {key!r}")
        section, name = parts
        sub = getattr(self, section)
        f = {x.name: x for x in fields(sub)}[name]
        setattr(sub, name, _coerce(value.strip(), f.type, getattr(sub, name)))

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_toml(self) -> str:
        lines = []
        for sec in fields(self):
            lines.append(f"[{sec.name}]")
            for f in fields(getattr(self, sec.name)):
                lines.append(f"{f.name} = {_toml_value(getattr(getattr(self, sec.name), f.name))}")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_toml())

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return _merge(cls(), d)


def _merge(cfg: Any, data: dict) -> Any:
    for k, v in data.items():
        if not hasattr(cfg, k):
            raise KeyError(f"unknown config key {k!r} in section {type(cfg).__name__}")
        cur = getattr(cfg, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(cfg, k, v)
    return cfg


def _coerce(text: str, typ: Any, current: Any) -> Any:
    t = str(typ)
    if isinstance(current, bool) or "bool" in t:
        return text.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(text))
    if isinstance(current, float):
        return float(text)
    if isinstance(current, list):
        return [int(x) for x in text.strip("[]").replace(" ", "").split(",") if x]
    return text


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return '"' + str(v).replace('"', '\\"') + '"'
