"""Batched car-racing environment (pure numpy).

``CarEnv`` simulates ``n`` cars at once on one track.  The physics are a
vectorised port of the original ``Game.py`` (velocity / steering authority /
drift momentum), the sensors are ray casts against the wall segments and the
reward is gate-based with a dense progress term.

API (gym-like, but batched):

    env = CarEnv(track, cfg, n=64)
    obs = env.reset()                      # (n, obs_dim)
    obs, reward, done, info = env.step(a)  # a: (n,) int actions
    obs = env.reset_done(obs, done)        # respawn finished cars

Cars never collide with each other, so the same env doubles as the arena for
race mode in the viewer.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from .config import CarConfig, Config, EnvConfig
from .track import Track

ACTIONS = ["left", "right", "accel", "brake", "accel+left", "accel+right",
           "brake+left", "brake+right", "coast"]
ACTION_SHORT = ["◀", "▶", "▲", "▼", "▲◀", "▲▶", "▼◀", "▼▶", "·"]
N_ACTIONS = len(ACTIONS)

_LEFT = np.array([a.endswith("left") for a in ACTIONS])
_RIGHT = np.array([a.endswith("right") for a in ACTIONS])
_ACCEL = np.array([a.startswith("accel") for a in ACTIONS])
_BRAKE = np.array([a.startswith("brake") for a in ACTIONS])


class CarEnv:
    def __init__(self, track: Track, cfg: Config | None = None, n: int = 1,
                 seed: int | None = None, car: CarConfig | None = None,
                 env: EnvConfig | None = None):
        cfg = cfg or Config()
        self.track = track
        self.car = car or cfg.car
        self.cfg = env or cfg.env
        self.n = n
        self.rng = np.random.default_rng(seed)

        self.walls = np.asarray(track.walls, dtype=float)
        self.gates = np.asarray(track.gates, dtype=float)
        self.gate_centers = track.gate_centers
        self.G = len(self.gates)
        if self.cfg.fov >= 360:
            self.ray_angles = np.radians(np.linspace(-180, 180, self.cfg.n_rays, endpoint=False))
        else:
            self.ray_angles = np.radians(np.linspace(-self.cfg.fov / 2, self.cfg.fov / 2, self.cfg.n_rays))
        self.obs_dim = self.cfg.n_rays + 3 + self.cfg.gate_lookahead
        self.n_actions = N_ACTIONS

        # state
        self.pos = np.zeros((n, 2))
        self.dir = np.zeros((n, 2))
        self.vel = np.zeros(n)
        self.drift = np.zeros(n)
        self.gate_idx = np.zeros(n, dtype=int)
        self.score = np.zeros(n, dtype=int)       # gates passed this episode
        self.laps = np.zeros(n, dtype=int)
        self.steps = np.zeros(n, dtype=int)
        self.since_gate = np.zeros(n, dtype=int)
        self.alive = np.ones(n, dtype=bool)
        self.ep_return = np.zeros(n)
        self.ray_dist = np.full((n, self.cfg.n_rays), self.cfg.ray_length)
        self.ray_hits = np.zeros((n, self.cfg.n_rays, 2))
        self.last_action = np.full(n, N_ACTIONS - 1)
        self.finished = np.zeros(n, dtype=bool)   # open tracks: crossed the finish line

    # ------------------------------------------------------------- reset
    def reset(self, idx=None, random_start: bool | None = None) -> np.ndarray:
        idx = np.arange(self.n) if idx is None else np.atleast_1d(np.asarray(idx))
        if len(idx) == 0:
            return self.observe()
        rs = self.cfg.random_start if random_start is None else random_start
        k = len(idx)
        if rs and self.G > 1:
            g = self.rng.integers(0, self.G, size=k)
            prev = (g - 1) % self.G
            start = self.gate_centers[prev].copy()
            if not self.track.closed:
                start = np.where((g == 0)[:, None], np.asarray(self.track.start_pos)[None], start)
            to_next = self.gate_centers[g] - start
            ang = geo.angle_of(to_next)
            self.gate_idx[idx] = g
        else:
            start = np.tile(np.asarray(self.track.start_pos, dtype=float), (k, 1))
            ang = np.full(k, self.track.start_angle)
            self.gate_idx[idx] = 0
        if self.cfg.start_jitter_deg > 0:
            ang = ang + self.rng.uniform(-self.cfg.start_jitter_deg, self.cfg.start_jitter_deg, size=k)
        self.pos[idx] = start
        self.dir[idx] = np.stack([np.cos(np.radians(ang)), np.sin(np.radians(ang))], axis=1)
        self.vel[idx] = self.car.min_speed
        self.drift[idx] = 0.0
        self.score[idx] = 0
        self.laps[idx] = 0
        self.steps[idx] = 0
        self.since_gate[idx] = 0
        self.alive[idx] = True
        self.finished[idx] = False
        self.ep_return[idx] = 0.0
        self.last_action[idx] = N_ACTIONS - 1
        self._cast_rays(idx)
        return self.observe()

    def reset_done(self, obs: np.ndarray, done: np.ndarray) -> np.ndarray:
        """Respawn cars whose episode ended and return the observation for the next step."""
        idx = np.flatnonzero(done)
        if len(idx) == 0:
            return obs
        new = self.reset(idx)
        out = obs.copy()
        out[idx] = new[idx]
        return out

    # -------------------------------------------------------------- step
    def step(self, actions) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        a = np.asarray(actions, dtype=int).reshape(self.n)
        car, cfg = self.car, self.cfg
        active = self.alive & ~self.finished
        self.last_action = a

        left, right = _LEFT[a], _RIGHT[a]
        accel, brake = _ACCEL[a], _BRAKE[a]

        # --- steering (authority scales with speed) and drift momentum
        mult = np.clip(self.vel / car.steer_full_speed, 0.0, 1.0)
        drift_amt = np.where(self.vel >= car.drift_min_speed,
                             self.vel * car.turn_rate * car.length / 72.0 * car.drift_gain, 0.0)
        turn = car.turn_rate * mult * (left.astype(float) - right.astype(float))
        self.dir = np.where(active[:, None], geo.rotate(self.dir, turn), self.dir)
        self.drift += np.where(active, drift_amt * (right.astype(float) - left.astype(float)), 0.0)

        # --- longitudinal
        acc = np.where(accel, car.acceleration, 0.0)
        acc = np.where(brake & (self.vel > 0), -car.brake_factor * car.acceleration, acc)
        vel = np.clip((self.vel + acc) * car.friction, car.min_speed, car.max_speed)
        self.vel = np.where(active, vel, self.vel)

        # --- movement: velocity along heading plus sideways drift, rescaled to speed
        perp = geo.rot90(self.dir)
        mv = self.vel[:, None] * self.dir + self.drift[:, None] * perp
        self.drift *= car.drift_friction
        norm = np.linalg.norm(mv, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        mv = mv / norm * self.vel[:, None]
        mv[~active] = 0.0

        to_gate = self.gate_centers[self.gate_idx] - self.pos
        d = np.linalg.norm(to_gate, axis=1, keepdims=True)
        d[d == 0] = 1.0
        progress = np.maximum(0.0, (mv * (to_gate / d)).sum(1))
        self.pos += mv
        self.steps += active
        self.since_gate += active

        # --- collisions
        edges = geo.rectangle_edges(self.pos, self.dir, car.length, car.width)   # (n,4,4)
        crash = geo.segments_intersect(edges[:, :, None, :], self.walls[None, None, :, :]).any(axis=(1, 2))
        crash &= active
        gate_seg = self.gates[self.gate_idx]                                       # (n,4)
        hit = geo.segments_intersect(edges, gate_seg[:, None, :]).any(axis=1) & active & ~crash

        # --- rewards
        reward = np.where(active, cfg.reward_step + cfg.reward_progress * progress, 0.0)
        if cfg.speed_bonus:
            reward += np.where(active, cfg.speed_bonus * self.vel / car.max_speed, 0.0)
        reward += np.where(hit, cfg.reward_gate, 0.0)
        reward += np.where(crash, cfg.reward_crash, 0.0)
        lap_done = np.zeros(self.n, dtype=bool)
        if hit.any():
            self.score[hit] += 1
            self.since_gate[hit] = 0
            self.gate_idx[hit] += 1
            wrapped = hit & (self.gate_idx >= self.G)
            if wrapped.any():
                lap_done |= wrapped
                reward += np.where(wrapped, cfg.reward_lap, 0.0)
                if self.track.closed:
                    self.gate_idx[wrapped] = 0
                    self.laps[wrapped] += 1
                else:
                    self.gate_idx[wrapped] = self.G - 1
                    self.finished[wrapped] = True
        self.alive &= ~crash
        timeout = active & (self.steps >= cfg.max_steps)
        stalled = active & (self.since_gate >= cfg.stall_steps)
        done = crash | timeout | stalled | (self.finished & active)
        self.ep_return += reward

        self._cast_rays()
        obs = self.observe()
        info = {
            "crash": crash, "timeout": timeout, "stalled": stalled, "gate": hit,
            "lap": lap_done, "finished": self.finished.copy(), "score": self.score.copy(),
            "laps": self.laps.copy(), "steps": self.steps.copy(), "return": self.ep_return.copy(),
            "progress": progress,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------- sensors
    def _cast_rays(self, idx=None) -> None:
        idx = np.arange(self.n) if idx is None else np.atleast_1d(idx)
        if len(idx) == 0:
            return
        dirs = geo.rotate(self.dir[idx][:, None, :], self.ray_angles[None, :])   # (k,R,2)
        origin = np.broadcast_to(self.pos[idx][:, None, :], dirs.shape)
        dist = geo.ray_segment_distance(origin, dirs, self.walls, self.cfg.ray_length)
        self.ray_dist[idx] = dist
        self.ray_hits[idx] = origin + dirs * dist[..., None]

    def observe(self) -> np.ndarray:
        cfg = self.cfg
        rays = 1.0 - np.clip(self.ray_dist, 1.0, cfg.ray_length) / cfg.ray_length
        speed = (self.vel - self.car.min_speed) / (self.car.max_speed - self.car.min_speed)
        pos_drift = np.clip(self.drift, 0, 8) / 8
        neg_drift = np.clip(-self.drift, 0, 8) / 8
        cols = [rays, speed[:, None], pos_drift[:, None], neg_drift[:, None]]
        for k in range(cfg.gate_lookahead):
            g = (self.gate_idx + k) % self.G if self.track.closed else np.minimum(self.gate_idx + k, self.G - 1)
            to_gate = self.gate_centers[g] - self.pos
            ang = geo.signed_angle_between(self.dir, to_gate) / 180.0
            cols.append(ang[:, None])
        return np.concatenate(cols, axis=1).astype(np.float32)

    # ------------------------------------------------------------- misc
    def lap_progress(self) -> np.ndarray:
        """Continuous race position: laps + fraction of gates passed (+ partial distance)."""
        to_gate = np.linalg.norm(self.gate_centers[self.gate_idx] - self.pos, axis=1)
        frac = np.clip(1.0 - to_gate / max(self.track.gate_spacing, 1.0), 0.0, 1.0)
        return self.laps + (self.gate_idx + frac) / self.G

    @property
    def heading_deg(self) -> np.ndarray:
        return geo.angle_of(self.dir)
