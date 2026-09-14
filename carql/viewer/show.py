"""Show / race / play windows.

* ``show``: one car, a timeline of checkpoints along the bottom; <- / -> walks
  through training history, digits jump to deciles, click the timeline.
* ``race``: many drivers on the same track with a live leaderboard.
* ``play``: you drive (arrow keys) - handy to feel a new track.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import pyglet
from pyglet import shapes
from pyglet.math import Mat4
from pyglet.window import key

from ..config import ROOT, Config
from ..env import ACTIONS, N_ACTIONS
from ..runs import Run, latest_run, resolve_run
from ..sim import PALETTE, Driver, RaceSim, drivers_from_specs, load_driver
from ..track import Track
from .base import (BG, DIM, GATE_NEXT, TEXT, BarChart, Camera, CarGraphics, HelpOverlay, Panel,
                   TrackGraphics, make_window)

HELP_SHOW = """SHOW MODE

<- / ->        previous / next checkpoint     Home / End   first / last
1 .. 9         jump to 10% .. 90% of training   click timeline to jump
Space  pause   R  reset car   + / -  sim speed   . step once (paused)
V  rays   Q  Q-value bars   G  gates   W  walls   I  info panel
E  export the last 60 s as an HTML replay (replays/)
M  take the wheel (arrow keys), M again to hand back
H  toggle this help   Esc  quit"""

HELP_RACE = """RACE MODE

Tab            focus next car (rays / Q-values / next gate follow the focus)
Space  pause   R  restart race   + / -  sim speed   . step once (paused)
V  rays   Q  Q-value bars   G  gates   W  walls   I  info   L  leaderboard
E  export the last 60 s as an HTML replay (replays/)
H  toggle this help   Esc  quit"""

HELP_PLAY = """PLAY MODE

Arrow keys drive (up accelerate, down brake, left / right steer)
R  reset   V  rays   G  gates   W  walls   H  help   Esc  quit"""

KEY_ACTION = {  # (up, down, left, right) -> action index
    (True, False, False, False): 2, (True, False, True, False): 4, (True, False, False, True): 5,
    (False, True, False, False): 3, (False, True, True, False): 6, (False, True, False, True): 7,
    (False, False, True, False): 0, (False, False, False, True): 1,
}


class ArenaWindow:
    def __init__(self, track: Track, sim: RaceSim, mode: str, *, title: str = "carql",
                 checkpoints: list[Driver] | None = None, start_index: int = 0, speed: float = 1.0,
                 rays: bool = True, run_name: str = ""):
        self.track, self.sim, self.mode = track, sim, mode
        self.run_name = run_name
        self.checkpoints = checkpoints or []
        self.ck_index = start_index
        self.steps_per_frame = speed
        self.step_accum = 0.0
        self.paused = False
        self.single_step = False
        self.focus = 0
        self.show_rays = rays
        self.show_q = True
        self.show_info = True
        self.show_board = True
        self.manual = mode == "play"
        self.keys = key.KeyStateHandler()
        self.recording: deque = deque(maxlen=60 * 60)
        self.flash: list[tuple[float, str]] = []
        self.fps_t = time.time()
        self.fps_n = 0
        self.fps = 0.0

        self.win = make_window(1280, 800, title)
        self.win.push_handlers(on_draw=self.on_draw, on_resize=self.on_resize, on_key_press=self.on_key_press,
                               on_mouse_press=self.on_mouse_press, on_mouse_scroll=self.on_mouse_scroll)
        self.win.push_handlers(self.keys)      # on top of the stack: sees every press/release for manual driving
        self.cam = Camera(track.size, margin=14, top_reserved=0, bottom_reserved=70 if mode == "show" else 8)
        self.world_batch = pyglet.graphics.Batch()
        self.hud_batch = pyglet.graphics.Batch()
        self.track_gfx = TrackGraphics(track, self.world_batch, self.cam)
        n_rays = sim.env.cfg.n_rays
        car = sim.cfg.car
        self.cars = [CarGraphics(self.world_batch, self.cam, car.length, car.width, d.color, n_rays)
                     for d in sim.drivers]
        # HUD
        self.info = Panel(self.hud_batch, 12, 0, 470, 7)
        self.board = Panel(self.hud_batch, 0, 0, 400, min(len(sim.drivers), 14) + 1, font_size=11)
        self.qchart = BarChart(self.hud_batch, 0, 0, 470, 120)
        self.help = HelpOverlay(self.hud_batch, {"show": HELP_SHOW, "race": HELP_RACE, "play": HELP_PLAY}[mode])
        self.car_labels = [pyglet.text.Label(d.label, x=0, y=0, anchor_x="center", anchor_y="bottom", font_size=10,
                                             color=(*d.color, 255), batch=self.hud_batch,
                                             group=pyglet.graphics.Group(order=15))
                           for d in sim.drivers]
        self.flash_label = pyglet.text.Label("", x=0, y=0, anchor_x="center", anchor_y="center", font_size=26,
                                             color=(255, 255, 255, 255), batch=self.hud_batch,
                                             group=pyglet.graphics.Group(order=30), weight="bold")
        self.status = pyglet.text.Label("", x=0, y=0, anchor_x="right", anchor_y="top", font_size=10, color=DIM,
                                        batch=self.hud_batch, group=pyglet.graphics.Group(order=15))
        self.timeline = Timeline(self.hud_batch, self.checkpoints) if mode == "show" else None
        self.on_resize(self.win.width, self.win.height)
        self.apply_visibility()
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    # ----------------------------------------------------------- layout
    def on_resize(self, w: int, h: int) -> None:
        self.cam.resize(w, h)
        self.track_gfx.update_thickness()
        self.info.move(12, h - 12)
        self.board.move(w - 12 - self.board.width, h - 12)
        self.qchart.move(12, (82 if self.mode == "show" else 12))
        self.help.layout(w, h)
        if self.timeline:
            self.timeline.layout(12, 10, w - 24, 62)
        self.flash_label.position = (w / 2, h * 0.55, 0)
        self.status.position = (w - 14, (h - 12 - self.board.height - 8) if (self.mode == "race" and self.show_board) else h - 14, 0)

    def apply_visibility(self) -> None:
        self.info.set_visible(self.show_info)
        self.qchart.set_visible(self.show_q and self.sim.drivers[self.focus].policy is not None)
        self.board.set_visible(self.mode == "race" and self.show_board)
        for i, c in enumerate(self.cars):
            c.set_rays_visible(self.show_rays and (i == self.focus))
        for lab in self.car_labels:
            lab.visible = self.mode == "race"

    # ------------------------------------------------------------ input
    def on_key_press(self, symbol: int, modifiers: int) -> bool | None:
        sim = self.sim
        if symbol == key.ESCAPE:
            self.win.close()
            return True
        if symbol == key.H:
            self.help.set_visible(not self.help.visible)
        elif symbol == key.SPACE:
            self.paused = not self.paused
        elif symbol == key.PERIOD:
            self.single_step = True
        elif symbol == key.R:
            sim.reset_all() if self.mode != "show" else sim.reset_car(0)
            self.recording.clear()
        elif symbol in (key.PLUS, key.EQUAL, key.NUM_ADD):
            self.steps_per_frame = min(32.0, self.steps_per_frame * 2 if self.steps_per_frame >= 1 else 1.0)
            self.toast(f"speed x{self.steps_per_frame:g}")
        elif symbol in (key.MINUS, key.NUM_SUBTRACT):
            self.steps_per_frame = max(0.125, self.steps_per_frame / 2)
            self.toast(f"speed x{self.steps_per_frame:g}")
        elif symbol == key.V:
            self.show_rays = not self.show_rays
        elif symbol == key.Q:
            self.show_q = not self.show_q
        elif symbol == key.I:
            self.show_info = not self.show_info
        elif symbol == key.L:
            self.show_board = not self.show_board
            self.on_resize(self.win.width, self.win.height)
        elif symbol == key.G:
            self.track_gfx.show_gates = not self.track_gfx.show_gates
            self.track_gfx.update_thickness()
        elif symbol == key.W:
            self.track_gfx.show_walls = not self.track_gfx.show_walls
            self.track_gfx.update_thickness()
        elif symbol == key.TAB:
            self.focus = (self.focus + 1) % len(self.cars)
        elif symbol == key.E:
            self.export_replay()
        elif symbol == key.M and self.mode == "show":
            self.manual = not self.manual
            self.toast("manual control" if self.manual else "AI control")
        elif self.mode == "show":
            n = len(self.checkpoints)
            if symbol in (key.RIGHT, key.N) and not self.manual:
                self.select_checkpoint(self.ck_index + 1)
            elif symbol in (key.LEFT, key.P) and not self.manual:
                self.select_checkpoint(self.ck_index - 1)
            elif symbol == key.HOME:
                self.select_checkpoint(0)
            elif symbol == key.END:
                self.select_checkpoint(n - 1)
            elif key._1 <= symbol <= key._9 and n:
                frac = (symbol - key._0) / 10
                self.select_checkpoint(int(round(frac * (n - 1))))
            elif symbol == key._0 and n:
                self.select_checkpoint(0)
        self.apply_visibility()
        return None

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int) -> None:
        if self.timeline and self.timeline.hit(x, y):
            i = self.timeline.index_at(x)
            if i is not None:
                self.select_checkpoint(i)
                self.apply_visibility()

    def on_mouse_scroll(self, x, y, sx, sy) -> None:
        if self.mode == "show" and self.timeline and self.timeline.hit(x, y):
            self.select_checkpoint(self.ck_index + (1 if sy > 0 else -1))
            self.apply_visibility()

    def select_checkpoint(self, i: int) -> None:
        if not self.checkpoints:
            return
        i = int(np.clip(i, 0, len(self.checkpoints) - 1))
        self.ck_index = i
        d = self.checkpoints[i]
        self.sim.replace_driver(0, d)
        self.cars[0].body.color = d.color
        self.toast(f"checkpoint: episode {d.episode}")
        self.recording.clear()

    def toast(self, text: str, seconds: float = 1.6) -> None:
        self.flash = [(time.time() + seconds, text)]

    # ----------------------------------------------------------- update
    def manual_action(self) -> int:
        k = self.keys
        combo = (bool(k[key.UP]), bool(k[key.DOWN]), bool(k[key.LEFT]), bool(k[key.RIGHT]))
        return KEY_ACTION.get(combo, N_ACTIONS - 1)

    def update(self, dt: float) -> None:
        sim = self.sim
        if self.manual:
            sim.drivers[self.focus].manual_action = self.manual_action()
        if self.paused and not self.single_step:
            return
        n_steps = 1 if self.single_step else 0
        if not self.single_step:
            self.step_accum += self.steps_per_frame
            n_steps = int(self.step_accum)
            self.step_accum -= n_steps
        self.single_step = False
        for _ in range(n_steps):
            if self.manual:
                d = sim.drivers[self.focus]
                pol, d.policy = d.policy, None
                info = sim.step()
                d.policy = pol
            else:
                info = sim.step()
            self.recording.append(sim.snapshot())
            for i in np.flatnonzero(info["crash"]):
                if i == self.focus:
                    self.toast("CRASH" if self.mode != "race" else f"{sim.drivers[i].label} crashed", 0.8)
            for i in np.flatnonzero(info["lap"]):
                if self.mode == "show" or i == self.focus:
                    self.toast(f"LAP {int(info['laps'][i])}!", 1.2)
        self.fps_n += 1
        now = time.time()
        if now - self.fps_t > 0.5:
            self.fps = self.fps_n / (now - self.fps_t)
            self.fps_t, self.fps_n = now, 0
        _headless_hook(self)

    # ------------------------------------------------------------- draw
    def on_draw(self) -> None:
        win, sim, env = self.win, self.sim, self.sim.env
        pyglet.gl.glClearColor(BG[0] / 255, BG[1] / 255, BG[2] / 255, 1)
        win.clear()
        # world
        self.track_gfx.highlight_gate(int(env.gate_idx[self.focus]) if self.track_gfx.show_gates else None)
        for i, c in enumerate(self.cars):
            alive = env.alive[i] and not env.finished[i]
            c.set_alpha(255 if alive else 70)
            c.update(env.pos[i, 0], env.pos[i, 1], env.heading_deg[i],
                     env.ray_hits[i] if (self.show_rays and i == self.focus) else None)
        win.view = self.cam.matrix
        self.world_batch.draw()
        win.view = Mat4()
        # hud
        self._update_hud()
        self.hud_batch.draw()

    def _update_hud(self) -> None:
        sim, env = self.sim, self.sim.env
        f = self.focus
        d = sim.drivers[f]
        cfg = sim.cfg
        lines = []
        if self.mode == "show":
            n = len(self.checkpoints)
            head = f"{self.run_name}  |  {d.algo_label}" if self.run_name else d.algo_label
            lines.append((head, (255, 220, 120, 255)))
            lines.append(f"checkpoint {self.ck_index + 1}/{n}: episode {d.episode}   {d.eval_text}")
            lines.append(f"frames trained {d.meta.get('frames', 0):,}   eps at save {d.meta.get('eps', 0):.3f}")
        elif self.mode == "race":
            lines.append((f"focus: {d.label}  ({d.algo_label})", (*d.color, 255)))
            lines.append(d.eval_text or "")
        else:
            lines.append(("manual control - arrow keys", (255, 220, 120, 255)))
        state = "CRASHED" if not env.alive[f] else ("FINISHED" if env.finished[f] else "driving")
        lines.append(f"gates {env.score[f]:3d}/{env.G}   laps {env.laps[f]}   step {env.steps[f]:5d}   {state}")
        lines.append(f"speed {env.vel[f]:5.2f}/{cfg.car.max_speed:g}   drift {env.drift[f]:+6.2f}   "
                     f"action {ACTIONS[int(sim.last_actions[f])]}")
        lines.append(f"reward {sim.last_reward[f]:+7.3f}   episode return {sim.total_reward[f]:8.2f}   "
                     f"crashes {sim.crashes[f]}")
        lines.append(f"sim x{self.steps_per_frame:g}  {self.fps:4.0f} fps  {'PAUSED' if self.paused else ''}"
                     f"{'  MANUAL' if self.manual else ''}   H help")
        self.info.set(lines)
        if self.show_q and d.policy is not None:
            label, vals = sim.last_scores[f]
            self.qchart.set(label or "Q", vals, int(sim.last_actions[f]))
        if self.mode == "race" and self.show_board:
            rows = [("  #  driver                     laps gates  crashes", DIM)]
            order = sim.leaderboard()
            for rank, i in enumerate(order[:14]):
                dd = sim.drivers[i]
                mark = ">" if i == f else " "
                rows.append((f"{mark}{rank + 1:2d}  {dd.label[:24]:24s} {env.laps[i]:4d} {env.score[i]:5d}  {sim.crashes[i]:5d}",
                             (*dd.color, 255)))
            self.board.set(rows)
            for i, lab in enumerate(self.car_labels):
                sx, sy = self.cam.to_screen(env.pos[i, 0], env.pos[i, 1])
                lab.position = (sx, sy + 14, 0)
        if self.timeline:
            self.timeline.set_index(self.ck_index)
        # toast
        now = time.time()
        self.flash = [(t, s) for t, s in self.flash if t > now]
        self.flash_label.text = self.flash[-1][1] if self.flash else ""
        self.status.text = f"track {self.track.name} | {len(sim.drivers)} car(s) | Esc quit"

    # ----------------------------------------------------------- export
    def export_replay(self) -> None:
        from ..replay import write_html
        if not self.recording:
            self.toast("nothing recorded yet")
            return
        out_dir = ROOT / "replays"
        out_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = (self.run_name or self.mode).replace("/", "_")
        path = out_dir / f"{name}-{stamp}.html"
        write_html(self.track, self.sim.drivers, list(self.recording), path, title=f"{name} replay")
        self.toast(f"wrote {path.name}", 3)
        print(f"wrote {path}")


_headless_counter = {"n": 0}


def _headless_hook(win_obj) -> None:
    """Testing aid: CARQL_SCREENSHOT=path [CARQL_FRAMES=n] saves a frame and quits."""
    import os
    path = os.environ.get("CARQL_SCREENSHOT")
    if not path:
        return
    _headless_counter["n"] += 1
    if _headless_counter["n"] >= int(os.environ.get("CARQL_FRAMES", "120")):
        win_obj.on_draw()
        pyglet.image.get_buffer_manager().get_color_buffer().save(path)
        pyglet.app.exit()


class Timeline:
    """Checkpoint timeline: a dot per checkpoint, bar height = eval score, current one highlighted."""

    def __init__(self, batch, drivers: list[Driver], order: int = 20):
        self.drivers = drivers
        self.g0 = pyglet.graphics.Group(order=order)
        self.g1 = pyglet.graphics.Group(order=order + 1)
        self.bg = shapes.Rectangle(0, 0, 10, 10, color=(16, 16, 20, 175), batch=batch, group=self.g0)
        self.bars = [shapes.Rectangle(0, 0, 3, 1, color=(90, 120, 170), batch=batch, group=self.g1) for _ in drivers]
        self.dots = [shapes.Circle(0, 0, 3, color=(200, 200, 200), batch=batch, group=self.g1) for _ in drivers]
        self.marker = shapes.Rectangle(0, 0, 2, 10, color=GATE_NEXT, batch=batch, group=self.g1)
        self.label = pyglet.text.Label("", x=0, y=0, anchor_y="bottom", font_size=9, color=DIM, batch=batch, group=self.g1)
        self.eplabel = pyglet.text.Label("", x=0, y=0, anchor_x="center", anchor_y="bottom", font_size=9,
                                         color=(*GATE_NEXT, 255), batch=batch, group=self.g1)
        self.x = self.y = self.w = self.h = 0
        self.index = 0
        eps = [d.episode or 0 for d in drivers]
        self.ep_min, self.ep_max = (min(eps), max(eps)) if eps else (0, 1)
        self.scores = [((d.meta.get("eval") or {}).get("score", 0.0)) for d in drivers]
        self.smax = max(self.scores) if self.scores and max(self.scores) > 0 else 1.0

    def xpos(self, i: int) -> float:
        ep = self.drivers[i].episode or 0
        span = max(1, self.ep_max - self.ep_min)
        return self.x + 12 + (ep - self.ep_min) / span * (self.w - 24)

    def layout(self, x, y, w, h) -> None:
        self.x, self.y, self.w, self.h = x, y, w, h
        self.bg.position, self.bg.width, self.bg.height = (x, y), w, h
        base = y + 22
        for i, (bar, dot) in enumerate(zip(self.bars, self.dots)):
            px = self.xpos(i)
            bh = 2 + self.scores[i] / self.smax * (h - 30)
            bar.position, bar.height = (px - 1.5, base), bh
            dot.position = (px, base)
        self.label.position = (x + 8, y + 4, 0)
        self.label.text = (f"training timeline: episode {self.ep_min} .. {self.ep_max}   "
                           f"(bar = eval gates passed from the start line, max {self.smax:.0f})")
        self.set_index(self.index)

    def set_index(self, i: int) -> None:
        if not self.drivers:
            return
        self.index = i
        px = self.xpos(i)
        self.marker.position = (px - 1, self.y + 4)
        self.marker.height = self.h - 8
        self.eplabel.position = (px, self.y + self.h - 12, 0)
        self.eplabel.text = f"ep {self.drivers[i].episode}"
        for k, dot in enumerate(self.dots):
            dot.color = GATE_NEXT if k == i else (200, 200, 200)
            dot.radius = 4 if k == i else 3

    def hit(self, x, y) -> bool:
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h

    def index_at(self, x) -> int | None:
        if not self.drivers:
            return None
        return int(np.argmin([abs(self.xpos(i) - x) for i in range(len(self.drivers))]))


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def _resolve_track(run: Run | None, track: str | None) -> Track:
    if track:
        return Track.load(track)
    if run is not None:
        return run.track()
    return Track.load("classic")


def show(run_name: str | None, checkpoints: str = "all", track: str | None = None, start: str | None = None,
         speed: float = 1.0, rays: bool = True) -> None:
    run = resolve_run(run_name) if run_name else latest_run()
    if run is None:
        raise SystemExit("no runs found - train one first: carql train")
    paths = run.select(checkpoints)
    if not paths:
        raise SystemExit(f"run {run.name} has no checkpoints")
    print(f"loading {len(paths)} checkpoints from {run.name} ...")
    drivers = [load_driver(p, PALETTE[1], run.name, label=f"ep {i}") for i, p in enumerate(paths)]
    for d in drivers:
        d.label = f"{run.name} ep {d.episode}"
    trk = _resolve_track(run, track)
    idx = len(drivers) - 1
    if start == "first":
        idx = 0
    elif start and start != "latest":
        ep = int(start)
        idx = int(np.argmin([abs((d.episode or 0) - ep) for d in drivers]))
    sim = RaceSim(trk, [drivers[idx]], drivers[idx].meta["config"], respawn=True, respawn_delay=45)
    ArenaWindow(trk, sim, "show", title=f"carql show - {run.name}", checkpoints=drivers, start_index=idx,
                speed=speed, rays=rays, run_name=run.name)
    pyglet.app.run()


def race(run_specs: list[str], checkpoints: str = "spread:6", track: str | None = None, speed: float = 1.0,
         respawn: bool = True) -> None:
    drivers = drivers_from_specs(run_specs, checkpoints)
    first = resolve_run(run_specs[0].partition(":")[0])
    trk = _resolve_track(first, track)
    print(f"race: {len(drivers)} drivers on {trk.name}")
    for d in drivers:
        print(f"  {d.label:30s} {d.algo_label:20s} {d.eval_text}")
    sim = RaceSim(trk, drivers, respawn=respawn)
    ArenaWindow(trk, sim, "race", title=f"carql race - {trk.name}", speed=speed)
    pyglet.app.run()


def play(track: str = "classic") -> None:
    trk = Track.load(track)
    d = Driver("you", None, PALETTE[2])
    sim = RaceSim(trk, [d], Config(), respawn=True, respawn_delay=30, manual_control=True)
    ArenaWindow(trk, sim, "play", title=f"carql play - {trk.name}")
    pyglet.app.run()
