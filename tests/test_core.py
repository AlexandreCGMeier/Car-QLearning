import json
from pathlib import Path

import numpy as np
import pytest

from carql import geometry as geo
from carql.agents import load_checkpoint, make_agent, save_checkpoint
from carql.agents.replay import PrioritizedReplayBuffer, ReplayBuffer
from carql.config import Config
from carql.env import ACTIONS, N_ACTIONS, CarEnv
from carql.track import Track, chain_walls, import_png, render_background

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ geometry
def test_segments_intersect_basic():
    a = np.array([0, 0, 10, 10.0])
    b = np.array([0, 10, 10, 0.0])
    c = np.array([20, 20, 30, 30.0])
    assert geo.segments_intersect(a, b)
    assert not geo.segments_intersect(a, c)
    # broadcasting: (2,4) vs (1,4)
    assert geo.segments_intersect(np.stack([a, c]), b[None]).tolist() == [True, False]


def test_ray_distance():
    origin = np.array([[0.0, 0.0]])
    direction = np.array([[1.0, 0.0]])
    walls = np.array([[5, -1, 5, 1.0], [3, 5, 3, 6.0]])   # second wall not on the ray
    d = geo.ray_segment_distance(origin, direction, walls, 100.0)
    assert d.shape == (1,)
    assert d[0] == pytest.approx(5.0)
    d = geo.ray_segment_distance(origin, -direction, walls, 100.0)
    assert d[0] == 100.0


def test_resample_and_simplify():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10.0]])
    r = geo.resample_polyline(sq, 2.0, closed=True)
    assert abs(len(r) - 20) <= 1
    zig = np.array([[0, 0], [1, 0.01], [2, 0], [3, 5], [4, 0.0]])
    s = geo.simplify_polyline(zig, 0.1)
    assert len(s) == 4 and s[0].tolist() == [0, 0] and s[-1].tolist() == [4, 0]


# --------------------------------------------------------------------- track
def test_classic_track_loads():
    t = Track.load("classic")
    assert t.walls.shape == (61, 4) and t.gates.shape == (57, 4)
    assert t.closed and len(chain_walls(t.walls)) == 2
    img = render_background(t, 0.5)
    assert img.size == (630, 350)


def test_generated_track_roundtrip(tmp_path):
    pts = np.array([[200, 150], [900, 150], [1000, 400], [700, 550], [200, 500.0]])
    t = Track.from_centerline(pts, 90, name="oval", size=(1200, 700), gate_spacing=80)
    assert len(t.walls) > 20 and len(t.gates) > 10
    # gates are ordered along the loop and roughly one width wide
    lens = np.linalg.norm(t.gates[:, :2] - t.gates[:, 2:], axis=1)
    assert np.allclose(lens, 90 * 1.15, atol=1e-6)
    p = tmp_path / "oval.json"
    t.save(p)
    t2 = Track.load(p)
    assert t2.name == "oval" and t2.centerline.shape == (5, 2)
    assert np.allclose(t2.walls, t.walls, atol=0.01)
    t2.width = 120
    t2.regenerate()
    assert not np.allclose(t2.gates, t.gates)


def test_png_import():
    t = import_png(ROOT / "assets" / "track_1.png", name="t1")
    assert t.closed and 40 < t.width < 60 and len(t.gates) > 30
    env = CarEnv(t, Config(), n=4, seed=0)
    env.reset()
    assert env.alive.all()


# ----------------------------------------------------------------------- env
def test_env_step_shapes_and_gates():
    t = Track.load("classic")
    cfg = Config()
    cfg.env.random_start = False
    env = CarEnv(t, cfg, n=8, seed=1)
    obs = env.reset()
    assert obs.shape == (8, env.obs_dim) and obs.dtype == np.float32
    assert np.all(obs[:, :cfg.env.n_rays] >= 0) and np.all(obs[:, :cfg.env.n_rays] <= 1)
    # accelerate straight: passes the first gate, then crashes into the bend
    scores, crashed = [], False
    for _ in range(300):
        obs, r, done, info = env.step(np.full(8, ACTIONS.index("accel")))
        if done.all():
            crashed = info["crash"].all()
            scores = info["score"]
            break
    assert crashed and scores.min() >= 1


def test_env_full_lap_with_gate_seeking_controller():
    """A trivial 'steer towards the next gate, brake in corners' controller must lap the track."""
    t = Track.load("classic")
    cfg = Config()
    cfg.env.random_start = False
    env = CarEnv(t, cfg, n=1, seed=0)
    obs = env.reset()
    for step in range(4000):
        ang = obs[0, cfg.env.n_rays + 3] * 180      # signed bearing to the next gate
        fast = obs[0, cfg.env.n_rays] > 0.45
        if ang > 8:
            a = ACTIONS.index("brake+left" if fast else "accel+left")
        elif ang < -8:
            a = ACTIONS.index("brake+right" if fast else "accel+right")
        else:
            a = ACTIONS.index("accel" if not fast else "coast")
        obs, r, done, info = env.step([a])
        assert not done[0], f"controller died at step {step} with {info['score'][0]} gates"
        if info["laps"][0] >= 1:
            return
    pytest.fail("no lap completed")


def test_env_random_start_and_reset_done():
    t = Track.load("classic")
    cfg = Config()
    cfg.env.random_start = True
    env = CarEnv(t, cfg, n=16, seed=3)
    obs = env.reset()
    assert len(set(env.gate_idx.tolist())) > 1
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, r, done, info = env.step(rng.integers(0, N_ACTIONS, 16))
        obs = env.reset_done(obs, done)
        assert env.alive.all()


# -------------------------------------------------------------------- replay
def test_prioritized_buffer_samples_high_priority_more():
    buf = PrioritizedReplayBuffer(64, 3, alpha=1.0, beta0=1.0, seed=0)
    for i in range(64):
        buf.add_batch(np.full((1, 3), i), [i % 9], [0.0], np.zeros((1, 3)), [0.0])
    idx = np.arange(64)
    td = np.zeros(64); td[7] = 100.0
    buf.update_priorities(idx, td)
    counts = np.zeros(64)
    for _ in range(50):
        i, w, batch = buf.sample(16)
        counts += np.bincount(i, minlength=64)
    assert counts[7] > counts.sum() * 0.5
    assert np.all(w <= 1.0 + 1e-6)


def test_uniform_buffer_wraps():
    buf = ReplayBuffer(10, 2, seed=0)
    buf.add_batch(np.ones((7, 2)), np.zeros(7), np.zeros(7), np.ones((7, 2)), np.zeros(7))
    buf.add_batch(np.ones((7, 2)) * 2, np.zeros(7), np.zeros(7), np.ones((7, 2)), np.zeros(7))
    assert len(buf) == 10 and buf.ptr == 4


# -------------------------------------------------------------------- agents
@pytest.mark.parametrize("algo", ["dqn", "ppo"])
def test_agent_checkpoint_roundtrip(tmp_path, algo):
    cfg = Config()
    cfg.agent.algo = algo
    cfg.agent.hidden = [32, 32]
    agent = make_agent(cfg, 21, N_ACTIONS, "cpu", seed=0)
    obs = np.random.default_rng(0).random((5, 21)).astype(np.float32)
    a = agent.act(obs)
    assert a.shape == (5,)
    label, vals = agent.scores(obs)
    assert vals.shape == (5, N_ACTIONS)
    p = save_checkpoint(tmp_path / "ck.pt", agent, cfg, episode=42, frames=1000, obs_dim=21, n_actions=N_ACTIONS,
                        eval={"score": 3.0})
    agent2, meta = load_checkpoint(p)
    assert meta["episode"] == 42 and meta["eval"]["score"] == 3.0
    assert np.array_equal(agent2.act(obs), a)


def test_dqn_update_reduces_loss():
    cfg = Config()
    cfg.agent.hidden = [32, 32]
    cfg.agent.batch_size = 32
    cfg.agent.per = True
    agent = make_agent(cfg, 4, N_ACTIONS, "cpu", seed=0)
    buf = agent.make_buffer(0)
    rng = np.random.default_rng(0)
    obs = rng.random((500, 4)).astype(np.float32)
    buf.add_batch(obs, rng.integers(0, N_ACTIONS, 500), np.ones(500), obs, np.ones(500))   # terminal: target = 1
    losses = [agent.update() for _ in range(200)]
    assert np.mean(losses[-20:]) < np.mean(losses[:20])


def test_train_smoke(tmp_path, monkeypatch):
    from carql import train as tr
    monkeypatch.setattr(tr, "RUNS_DIR", tmp_path)
    cfg = Config()
    cfg.agent.hidden = [32, 32]
    cfg.agent.learning_starts = 200
    cfg.train.n_envs = 8
    cfg.train.total_episodes = 12
    cfg.train.checkpoint_every = 5
    cfg.train.eval_every = 5
    cfg.train.eval_envs = 2
    cfg.env.max_steps = 50
    run_dir = tr.train(cfg, "smoke")
    cks = sorted((run_dir / "checkpoints").glob("ep_*.pt"))
    assert len(cks) >= 2 and (run_dir / "metrics.csv").exists() and (run_dir / "best.pt").exists()
    cfg.agent.algo = "ppo"
    cfg.agent.rollout_steps = 16
    run_dir = tr.train(cfg, "smoke-ppo")
    assert (run_dir / "latest.pt").exists()


# -------------------------------------------------------------------- config
def test_config_overrides_and_toml(tmp_path):
    cfg = Config.load(ROOT / "configs" / "default.toml", ["train.n_envs=16", "agent.per=false", "agent.hidden=[64,64]"])
    assert cfg.train.n_envs == 16 and cfg.agent.per is False and cfg.agent.hidden == [64, 64]
    cfg.save(tmp_path / "c.toml")
    cfg2 = Config.load(tmp_path / "c.toml")
    assert cfg2.to_dict() == cfg.to_dict()
    with pytest.raises(KeyError):
        Config.from_dict({"agent": {"nope": 1}})


# -------------------------------------------------------------------- replay html
def test_html_export(tmp_path):
    from carql.replay import write_html
    from carql.sim import Driver, RaceSim
    t = Track.load("classic")
    d = Driver("me", None, (255, 0, 0))
    sim = RaceSim(t, [d], Config(), manual_control=True)
    frames = []
    for _ in range(30):
        sim.step()
        frames.append(sim.snapshot())
    out = write_html(t, [d], frames, tmp_path / "r.html")
    txt = out.read_text()
    assert '"T":30' in txt and "data:image/png;base64" in txt
