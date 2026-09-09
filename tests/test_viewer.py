"""Viewer tests need an OpenGL display; they run under xvfb-run when available and skip otherwise."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HAS_DISPLAY = bool(os.environ.get("DISPLAY")) or sys.platform in ("darwin", "win32")
XVFB = shutil.which("xvfb-run")
pytestmark = pytest.mark.skipif(not (HAS_DISPLAY or XVFB), reason="no display / xvfb available")


def _run(code: str, env_extra: dict, timeout: int = 180) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-c", code]
    if not HAS_DISPLAY and XVFB:
        cmd = [XVFB, "-a", "-s", "-screen 0 1280x800x24", *cmd]
    env = {**os.environ, **env_extra}
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def test_builder_headless(tmp_path):
    shot = tmp_path / "build.png"
    out = tmp_path / "t.json"
    code = f"""
import pyglet
from pyglet.window import key, mouse
from carql.viewer.builder import BuilderWindow
w = BuilderWindow(None, size=(1000, 600), width=80, name="ptest")
w.path = r"{out}"
for wx, wy in [(150, 150), (800, 150), (850, 450), (200, 450)]:
    sx, sy = w.cam.to_screen(wx, wy)
    w.on_mouse_press(sx, sy, mouse.LEFT, 0); w.on_mouse_release(sx, sy, mouse.LEFT, 0)
assert w.track is not None and len(w.track.gates) > 5, "no track generated"
w.on_key_press(key.BRACKETRIGHT, 0)      # width +5
w.on_key_press(key.T, 0)                 # test drive
w.update(1/60)
w.on_key_press(key.T, 0)
w.on_key_press(key.S, 0)                 # save
w.on_draw()
pyglet.image.get_buffer_manager().get_color_buffer().save(r"{shot}")
print("OK", w.width, len(w.track.walls))
"""
    r = _run(code, {})
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OK 85" in r.stdout and out.exists() and shot.exists()


def test_show_and_race_headless(tmp_path):
    # tiny training run to have checkpoints, then open the viewer for a few frames
    runs = tmp_path / "runs"
    code = f"""
import numpy as np
from pathlib import Path
from carql import train as tr
tr.RUNS_DIR = Path(r"{runs}")
from carql.config import Config
cfg = Config(); cfg.agent.hidden=[16,16]; cfg.agent.learning_starts=100; cfg.train.n_envs=8
cfg.train.total_episodes=10; cfg.train.checkpoint_every=5; cfg.train.eval_every=5; cfg.train.eval_envs=2; cfg.env.max_steps=40
tr.train(cfg, "v")
import carql.runs as R; R.RUNS_DIR = tr.RUNS_DIR
from carql.viewer.show import ArenaWindow
from carql.sim import RaceSim, load_driver, PALETTE
from carql.track import Track
run = R.Run(tr.RUNS_DIR / "v")
drivers = [load_driver(c.path, PALETTE[i], "v") for i, c in enumerate(run.checkpoints())]
trk = run.track()
sim = RaceSim(trk, [drivers[-1]], drivers[-1].meta["config"])
w = ArenaWindow(trk, sim, "show", checkpoints=drivers, start_index=len(drivers)-1, run_name="v")
from pyglet.window import key
for _ in range(5): w.update(1/60)
w.on_key_press(key.LEFT, 0); w.on_key_press(key.V, 0); w.on_key_press(key.Q, 0); w.on_key_press(key.M, 0)
for _ in range(5): w.update(1/60)
w.on_draw()
w.win.close()
sim2 = RaceSim(trk, drivers)
w2 = ArenaWindow(trk, sim2, "race")
for _ in range(20): w2.update(1/60)
w2.on_key_press(key.TAB, 0); w2.on_draw()
w2.export_replay()
print("OK", len(drivers))
"""
    r = _run(code, {}, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "OK 2" in r.stdout
