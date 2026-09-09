"""Training loops (DQN family and PPO) writing self-contained run folders."""
from __future__ import annotations

import csv
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .agents import DQNAgent, PPOAgent, load_checkpoint, make_agent, resolve_device, save_checkpoint
from .config import Config
from .env import CarEnv
from .runs import RUNS_DIR, Run
from .track import Track

METRIC_FIELDS = ["episode", "frames", "return", "score", "laps", "steps", "crashed", "eps", "loss", "time"]
EVAL_FIELDS = ["episode", "frames", "score", "laps", "steps", "crash_rate", "return", "time"]


class _Recorder:
    """Episode bookkeeping shared by both loops: metrics, eval, checkpoints, logging."""

    def __init__(self, cfg: Config, run_dir: Path, track: Track, agent, env: CarEnv,
                 start_episode: int = 0, start_frames: int = 0):
        self.cfg, self.run_dir, self.track, self.agent, self.env = cfg, run_dir, track, agent, env
        self.episode = start_episode
        self.frames = start_frames
        self.t0 = time.time()
        self.last_log_ep = start_episode
        self.last_ckpt_bucket = start_episode // cfg.train.checkpoint_every
        self.last_eval_bucket = start_episode // cfg.train.eval_every
        self.best_eval = -np.inf
        self.recent: list[dict] = []
        self.loss = float("nan")
        self.eps = 0.0
        self.last_eval: dict = {}
        eval_env_cfg = replace(cfg.env, random_start=False, start_jitter_deg=max(cfg.env.start_jitter_deg, 3.0))
        self.eval_env = CarEnv(track, cfg, n=cfg.train.eval_envs, seed=cfg.train.seed + 1000, env=eval_env_cfg)
        self.metrics_f = open(run_dir / "metrics.csv", "a", newline="")
        self.metrics_w = csv.DictWriter(self.metrics_f, fieldnames=METRIC_FIELDS)
        if self.metrics_f.tell() == 0:
            self.metrics_w.writeheader()
        self.eval_f = open(run_dir / "eval.csv", "a", newline="")
        self.eval_w = csv.DictWriter(self.eval_f, fieldnames=EVAL_FIELDS)
        if self.eval_f.tell() == 0:
            self.eval_w.writeheader()
        self.log_f = open(run_dir / "log.txt", "a")

    def close(self) -> None:
        for f in (self.metrics_f, self.eval_f, self.log_f):
            f.close()

    def log(self, msg: str) -> None:
        print(msg, flush=True)
        self.log_f.write(msg + "\n")
        self.log_f.flush()

    # ------------------------------------------------------------------
    def on_step(self, done: np.ndarray, info: dict, n: int) -> bool:
        """Call after every batched env step. Returns True when training should stop."""
        self.frames += n
        idx = np.flatnonzero(done)
        now = time.time() - self.t0
        for i in idx:
            self.episode += 1
            row = {"episode": self.episode, "frames": self.frames, "return": round(float(info["return"][i]), 3),
                   "score": int(info["score"][i]), "laps": int(info["laps"][i]), "steps": int(info["steps"][i]),
                   "crashed": int(info["crash"][i]), "eps": round(self.eps, 4),
                   "loss": round(self.loss, 5) if np.isfinite(self.loss) else "",
                   "time": round(now, 1)}
            self.metrics_w.writerow(row)
            self.recent.append(row)
        if len(self.recent) > 200:
            self.recent = self.recent[-200:]
        if idx.size:
            self.metrics_f.flush()
            self._maybe_eval()
            self._maybe_checkpoint()
            self._maybe_log()
        return self.episode >= self.cfg.train.total_episodes

    def _maybe_log(self) -> None:
        if self.episode - self.last_log_ep < self.cfg.train.log_every:
            return
        self.last_log_ep = self.episode
        r = self.recent[-100:]
        elapsed = time.time() - self.t0
        fps = self.frames / max(elapsed, 1e-9)
        ev = self.last_eval
        ev_txt = f" | eval {ev['score']:.1f} gates {ev['laps']:.2f} laps" if ev else ""
        self.log(f"ep {self.episode:6d} | {self.frames/1000:7.0f}k frames | ret {np.mean([x['return'] for x in r]):7.2f}"
                 f" | score {np.mean([x['score'] for x in r]):5.1f} | laps {np.mean([x['laps'] for x in r]):4.2f}"
                 f" | len {np.mean([x['steps'] for x in r]):5.0f} | crash {np.mean([x['crashed'] for x in r]):.2f}"
                 f" | eps {self.eps:.3f} | loss {self.loss:.4f} | {fps/1000:5.1f}k fps | {elapsed/60:5.1f} min{ev_txt}")

    def _maybe_eval(self) -> None:
        bucket = self.episode // self.cfg.train.eval_every
        if bucket <= self.last_eval_bucket:
            return
        self.last_eval_bucket = bucket
        self.last_eval = evaluate(self.agent, self.eval_env, self.cfg.env.max_steps)
        row = {"episode": self.episode, "frames": self.frames, "time": round(time.time() - self.t0, 1)}
        row.update({k: round(v, 4) for k, v in self.last_eval.items()})
        self.eval_w.writerow(row)
        self.eval_f.flush()
        metric = self.last_eval["laps"] * 1000 + self.last_eval["score"]
        if metric > self.best_eval:
            self.best_eval = metric
            self._save(self.run_dir / "best.pt")

    def _maybe_checkpoint(self, force: bool = False) -> None:
        bucket = self.episode // self.cfg.train.checkpoint_every
        if bucket <= self.last_ckpt_bucket and not force:
            return
        self.last_ckpt_bucket = bucket
        self._save(self.run_dir / "checkpoints" / f"ep_{self.episode:06d}.pt")
        self._save(self.run_dir / "latest.pt")

    def _save(self, path: Path) -> None:
        save_checkpoint(path, self.agent, self.cfg, episode=self.episode, frames=self.frames,
                        obs_dim=self.env.obs_dim, n_actions=self.env.n_actions, eval=self.last_eval,
                        extra={"track": self.track.name, "eps": self.eps, "elapsed": time.time() - self.t0})

    def finish(self) -> None:
        self.last_eval = evaluate(self.agent, self.eval_env, self.cfg.env.max_steps)
        self._maybe_checkpoint(force=True)
        self.log(f"done: {self.episode} episodes, {self.frames} frames, {(time.time()-self.t0)/60:.1f} min; "
                 f"final eval {self.last_eval['score']:.1f} gates / {self.last_eval['laps']:.2f} laps")
        self.close()


def evaluate(agent, env: CarEnv, max_steps: int) -> dict:
    """Greedy rollout from the start line; mean gates/laps/steps over env.n cars."""
    obs = env.reset()
    finished = np.zeros(env.n, dtype=bool)
    score = np.zeros(env.n); laps = np.zeros(env.n); steps = np.zeros(env.n); crashed = np.zeros(env.n)
    ret = np.zeros(env.n)
    for _ in range(max_steps + 1):
        a = agent.act(obs, 0.0)
        obs, r, done, info = env.step(a)
        newly = done & ~finished
        for k, arr in (("score", score), ("laps", laps), ("steps", steps), ("return", ret)):
            arr[newly] = info[k][newly]
        crashed[newly] = info["crash"][newly]
        finished |= done
        if finished.all():
            break
    live = ~finished
    score[live] = info["score"][live]; laps[live] = info["laps"][live]
    steps[live] = info["steps"][live]; ret[live] = info["return"][live]
    return {"score": float(score.mean()), "laps": float(laps.mean()), "steps": float(steps.mean()),
            "crash_rate": float(crashed.mean()), "return": float(ret.mean())}


# ---------------------------------------------------------------------------
def prepare_run(cfg: Config, run_name: str | None = None, resume: bool = False) -> tuple[Path, Track]:
    track = Track.load(cfg.env.track)
    name = run_name or cfg.train.run or f"{track.name}-{cfg.agent.algo}"
    cfg.train.run = name
    run_dir = RUNS_DIR / name
    if run_dir.exists() and not resume:
        if (run_dir / "checkpoints").exists() and any((run_dir / "checkpoints").iterdir()):
            raise FileExistsError(f"run {name!r} already exists; use --resume or pick another --run name")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    if not (run_dir / "config.toml").exists() or not resume:
        cfg.save(run_dir / "config.toml")
    if track.source and not (run_dir / "track.json").exists():
        shutil.copy(track.source, run_dir / "track.json")
    return run_dir, track


def train(cfg: Config, run_name: str | None = None, resume: bool = False) -> Path:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    if cfg.train.threads > 0:
        torch.set_num_threads(cfg.train.threads)
    device = resolve_device(cfg.train.device)
    run_dir, track = prepare_run(cfg, run_name, resume)
    env = CarEnv(track, cfg, n=cfg.train.n_envs, seed=cfg.train.seed)
    agent = make_agent(cfg, env.obs_dim, env.n_actions, device, seed=cfg.train.seed)
    start_ep = start_frames = 0
    if resume:
        latest = Run(run_dir).latest()
        if latest is not None:
            loaded, meta = load_checkpoint(latest, device)
            agent.load_state(loaded.state())
            start_ep, start_frames = meta["episode"], meta["frames"]
            print(f"resumed from {latest} (episode {start_ep})")
    rec = _Recorder(cfg, run_dir, track, agent, env, start_ep, start_frames)
    rec.log(f"run {cfg.train.run}: {agent.label} on track '{track.name}' ({track.describe()}), "
            f"{cfg.train.n_envs} cars in parallel, obs_dim {env.obs_dim}, device {device}")
    try:
        if isinstance(agent, DQNAgent):
            _train_dqn(cfg, env, agent, rec)
        else:
            _train_ppo(cfg, env, agent, rec)
    except KeyboardInterrupt:
        rec.log("interrupted - saving latest checkpoint")
    rec.finish()
    print(f"\ncheckpoints in {run_dir}/checkpoints  ->  carql show {cfg.train.run}")
    return run_dir


class NStepAccumulator:
    """Turn per-step transitions of N parallel envs into n-step transitions.

    Keeps the last ``n`` transitions of every env; once the window is full (or
    an episode ends) it emits ``(s_t, a_t, sum_k gamma^k r_{t+k}, s_{t+n}, done)``.
    """

    def __init__(self, n_envs: int, n: int, gamma: float):
        self.n_envs, self.n, self.gamma = n_envs, max(1, n), gamma
        self.queues: list[list[tuple]] = [[] for _ in range(n_envs)]

    def push(self, obs, act, rew, nobs, done):
        out = []
        for i in range(self.n_envs):
            q = self.queues[i]
            q.append((obs[i], act[i], float(rew[i]), nobs[i], bool(done[i])))
            if done[i]:
                while q:
                    out.append(self._collapse(q))
                    q.pop(0)
                q.clear()
            elif len(q) >= self.n:
                out.append(self._collapse(q))
                q.pop(0)
        return out

    def _collapse(self, q):
        ret = 0.0
        for k, (_, _, r, _, _) in enumerate(q):
            ret += (self.gamma ** k) * r
        return q[0][0], q[0][1], ret, q[-1][3], q[-1][4]


def _train_dqn(cfg: Config, env: CarEnv, agent: DQNAgent, rec: _Recorder) -> None:
    c = cfg.agent
    buf = agent.make_buffer(cfg.train.seed)
    agent.gamma_eff = c.gamma ** max(1, c.n_step)
    nstep = NStepAccumulator(env.n, c.n_step, c.gamma)
    obs = env.reset()
    n = env.n
    rng = np.random.default_rng(cfg.train.seed)
    while True:
        frames = rec.frames
        eps = max(c.eps_end, c.eps_start + (c.eps_end - c.eps_start) * frames / max(c.eps_decay_frames, 1))
        rec.eps = eps
        if len(buf) < c.learning_starts:
            a = rng.integers(0, env.n_actions, size=n)
        else:
            a = agent.act(obs, eps)
        nobs, r, done, info = env.step(a)
        trans = nstep.push(obs, a, r, nobs, done)
        if trans:
            o, aa, rr, no, dd = (np.asarray(x) for x in zip(*trans))
            buf.add_batch(o, aa, (rr * c.reward_scale).astype(np.float32), no, dd.astype(np.float32))
        obs = env.reset_done(nobs, done)
        if len(buf) >= c.learning_starts:
            beta = min(1.0, c.per_beta0 + (1 - c.per_beta0) * frames / max(2 * c.eps_decay_frames, 1))
            losses = [agent.update(beta) for _ in range(c.updates_per_step)]
            rec.loss = float(np.mean(losses))
        if rec.on_step(done, info, n):
            break


def _train_ppo(cfg: Config, env: CarEnv, agent: PPOAgent, rec: _Recorder) -> None:
    c = cfg.agent
    T, n = c.rollout_steps, env.n
    obs = env.reset()
    obs_buf = np.zeros((T, n, env.obs_dim), np.float32)
    act_buf = np.zeros((T, n), np.int64)
    logp_buf = np.zeros((T, n), np.float32)
    rew_buf = np.zeros((T, n), np.float32)
    done_buf = np.zeros((T, n), np.float32)
    val_buf = np.zeros((T + 1, n), np.float32)
    stop = False
    while not stop:
        for t in range(T):
            a, logp, v = agent.sample(obs)
            nobs, r, done, info = env.step(a)
            obs_buf[t], act_buf[t], logp_buf[t], rew_buf[t], done_buf[t], val_buf[t] = obs, a, logp, r, done, v
            obs = env.reset_done(nobs, done)
            if rec.on_step(done, info, n):
                stop = True
        val_buf[T] = agent.value(obs)
        adv = np.zeros((T, n), np.float32)
        last = np.zeros(n, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + c.gamma * val_buf[t + 1] * nonterm - val_buf[t]
            last = delta + c.gamma * c.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + val_buf[:T]
        stats = agent.update(obs_buf.reshape(T * n, -1), act_buf.reshape(-1), logp_buf.reshape(-1),
                             ret.reshape(-1), adv.reshape(-1))
        rec.loss = stats["loss"]
        rec.eps = stats["entropy"]
