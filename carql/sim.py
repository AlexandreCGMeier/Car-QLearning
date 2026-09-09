"""Headless race simulation shared by the viewer and the HTML exporter.

A ``RaceSim`` puts one or more *drivers* (a policy plus a label/colour) on a
single batched ``CarEnv`` and steps them together.  Cars never collide with
each other.  Crashed cars respawn on the start line after a short delay (or
stay dead until everybody is done, if ``respawn=False``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .agents import load_checkpoint
from .config import Config
from .env import CarEnv, N_ACTIONS
from .runs import Run, resolve_run
from .track import Track

PALETTE = [
    (255, 99, 71), (65, 160, 255), (255, 205, 60), (90, 220, 130), (200, 120, 255),
    (255, 140, 30), (60, 220, 220), (255, 110, 180), (170, 200, 80), (200, 200, 200),
    (140, 90, 60), (120, 150, 255),
]


@dataclass
class Driver:
    label: str
    policy: object | None            # anything with .act(obs)->actions and .scores(obs); None = manual
    color: tuple[int, int, int]
    meta: dict = field(default_factory=dict)
    run: str = ""
    manual_action: int = N_ACTIONS - 1

    @property
    def episode(self) -> int | None:
        return self.meta.get("episode")

    @property
    def algo_label(self) -> str:
        return self.meta.get("label", "manual" if self.policy is None else "?")

    @property
    def eval_text(self) -> str:
        ev = self.meta.get("eval") or {}
        if not ev:
            return ""
        return f"eval {ev.get('score', 0):.1f} gates / {ev.get('laps', 0):.2f} laps"


def load_driver(path: str | Path, color=None, run_name: str = "", label: str | None = None,
                device: str = "cpu") -> Driver:
    agent, meta = load_checkpoint(path, device)
    ep = meta.get("episode", 0)
    lbl = label or (f"{run_name} ep {ep}" if run_name else f"ep {ep}")
    return Driver(lbl, agent, color or PALETTE[0], meta, run_name)


def drivers_from_specs(specs: list[str], default_spec: str = "spread:6", device: str = "cpu") -> list[Driver]:
    """``["runA", "runB:best", "runC:200,1500"]`` -> drivers (colours assigned round-robin)."""
    drivers: list[Driver] = []
    for spec in specs:
        name, _, ck = spec.partition(":")
        run = resolve_run(name)
        paths = run.select(ck or default_spec)
        if not paths:
            raise FileNotFoundError(f"run {run.name!r} has no checkpoints matching {ck or default_spec!r}")
        for p in paths:
            d = load_driver(p, PALETTE[len(drivers) % len(PALETTE)], run.name)
            drivers.append(d)
    return drivers


class RaceSim:
    def __init__(self, track: Track, drivers: list[Driver], cfg: Config | None = None, *,
                 respawn: bool = True, respawn_delay: int = 75, seed: int = 0,
                 manual_control: bool = False):
        self.track = track
        self.drivers = drivers
        if cfg is None:
            cfg = next((d.meta["config"] for d in drivers if "config" in d.meta), Config())
        self.cfg = cfg
        from dataclasses import replace
        env_cfg = replace(cfg.env, random_start=False, start_jitter_deg=0.0, max_steps=10 ** 9,
                          stall_steps=10 ** 9 if manual_control else max(cfg.env.stall_steps, 600))
        self.env = CarEnv(track, cfg, n=len(drivers), seed=seed, env=env_cfg)
        self.respawn = respawn
        self.respawn_delay = respawn_delay
        self.obs = self.env.reset()
        self.n = len(drivers)
        self.dead_timer = np.zeros(self.n, dtype=int)
        self.crashes = np.zeros(self.n, dtype=int)
        self.best_score = np.zeros(self.n, dtype=int)
        self.best_laps = np.zeros(self.n, dtype=int)
        self.total_reward = np.zeros(self.n)
        self.last_reward = np.zeros(self.n)
        self.last_actions = np.full(self.n, N_ACTIONS - 1)
        self.last_scores: list[tuple[str, np.ndarray]] = [("", np.zeros(N_ACTIONS))] * self.n
        self.step_count = 0
        self.events: list[tuple[int, int, str]] = []    # (step, car, "crash"/"lap"/...)

    # ---------------------------------------------------------- control
    def replace_driver(self, i: int, driver: Driver) -> None:
        self.drivers[i] = driver
        self.reset_car(i)

    def reset_car(self, i: int) -> None:
        self.obs = self.env.reset_done(self.obs, np.arange(self.n) == i)
        self.dead_timer[i] = 0
        self.total_reward[i] = 0.0

    def reset_all(self) -> None:
        self.obs = self.env.reset()
        self.dead_timer[:] = 0
        self.total_reward[:] = 0.0
        self.step_count = 0
        self.crashes[:] = 0
        self.best_score[:] = 0
        self.best_laps[:] = 0
        self.events.clear()

    # ------------------------------------------------------------- step
    def actions(self) -> np.ndarray:
        a = np.empty(self.n, dtype=int)
        for i, d in enumerate(self.drivers):
            if d.policy is None:
                a[i] = d.manual_action
                self.last_scores[i] = ("", np.zeros(N_ACTIONS))
            else:
                a[i] = int(d.policy.act(self.obs[i:i + 1])[0])
                self.last_scores[i] = d.policy.scores(self.obs[i:i + 1])
                self.last_scores[i] = (self.last_scores[i][0], self.last_scores[i][1][0])
        return a

    def step(self) -> dict:
        env = self.env
        active = env.alive & ~env.finished
        a = self.actions()
        a[~active] = N_ACTIONS - 1
        self.last_actions = a
        self.obs, r, done, info = env.step(a)
        self.last_reward = r
        self.total_reward += r
        self.step_count += 1
        self.best_score = np.maximum(self.best_score, info["score"])
        self.best_laps = np.maximum(self.best_laps, info["laps"])
        for i in np.flatnonzero(info["lap"]):
            self.events.append((self.step_count, i, "lap"))
        crashed = info["crash"] | info["stalled"]
        for i in np.flatnonzero(crashed):
            self.crashes[i] += 1
            self.events.append((self.step_count, i, "crash"))
        for i in np.flatnonzero(info["finished"] & done):
            self.events.append((self.step_count, i, "finish"))
        gone = ~(env.alive & ~env.finished)
        if self.respawn:
            self.dead_timer[gone] += 1
            to_reset = gone & (self.dead_timer > self.respawn_delay)
            if to_reset.any():
                self.obs = env.reset_done(self.obs, to_reset)
                self.dead_timer[to_reset] = 0
                self.total_reward[to_reset] = 0.0
        elif gone.all():
            self.dead_timer += 1
            if self.dead_timer.max() > self.respawn_delay * 2:
                self.obs = env.reset()
                self.dead_timer[:] = 0
                self.total_reward[:] = 0.0
        return info

    # ------------------------------------------------------------- view
    def leaderboard(self) -> list[int]:
        prog = self.env.lap_progress()
        key = prog + self.best_laps * 0.0   # current progress decides; crashes shown separately
        return list(np.argsort(-key, kind="stable"))

    def snapshot(self) -> dict:
        """Compact per-car state for recording."""
        env = self.env
        return {
            "x": env.pos[:, 0].copy(), "y": env.pos[:, 1].copy(), "h": env.heading_deg.copy(),
            "alive": (env.alive & ~env.finished).copy(), "score": env.score.copy(), "laps": env.laps.copy(),
            "gate": env.gate_idx.copy(), "speed": env.vel.copy(),
        }
