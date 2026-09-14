"""Track builder: click a centerline, pick a width, gates and walls are generated.

    carql build                    # new track
    carql build tracks/track_1.json  # edit a generated track

Left click adds a control point (or drags an existing one), right click
deletes one.  Everything else is on the H key.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyglet
from pyglet import shapes
from pyglet.math import Mat4
from pyglet.window import key, mouse

from ..config import Config
from ..sim import PALETTE, Driver, RaceSim
from ..track import TRACKS_DIR, Track, smooth_centerline
from .base import BG, DIM, GATE_NEXT, TEXT, Camera, CarGraphics, HelpOverlay, Panel, TrackGraphics, make_window

HELP = """TRACK BUILDER

left click        add a control point at the end / drag an existing point
shift + click     insert a point into the nearest segment
right click       delete the point under the cursor
Backspace         delete the last point        Ctrl+Z  undo
[ / ]             road width -/+ 5             , / .   gate spacing -/+ 10
C                 closed loop / open           D       reverse driving direction
1                 make the hovered point the start (closed loops)
T                 test drive with the arrow keys (T again to leave)
G  gates   W  walls   S  save   N  new (clear)   H  help   Esc  quit

Tracks save to tracks/<name>.json - train on one with  carql train -t <name>"""


class BuilderWindow:
    def __init__(self, track: Track | None, size=(1260.0, 700.0), width: float = 90.0, name: str | None = None):
        if track is not None and track.centerline is None:
            raise SystemExit(f"{track.name} has no centerline (hand-coded walls) and can't be edited; "
                             f"import a PNG or draw a new one instead")
        self.size = tuple(track.size) if track else tuple(size)
        self.points: list[list[float]] = [list(p) for p in track.centerline] if track else []
        self.width = float(track.width) if track else width
        self.closed = track.closed if track else True
        self.gate_spacing = track.gate_spacing if track else 60.0
        self.smooth = track.smooth if track else 2
        self.name = name or (track.name if track else f"custom-{time.strftime('%m%d-%H%M')}")
        self.path = Path(track.source) if (track and track.source) else TRACKS_DIR / f"{self.name}.json"
        self.history: list[list[list[float]]] = []
        self.drag_idx: int | None = None
        self.hover_idx: int | None = None
        self.dirty = True
        self.track: Track | None = None
        self.track_gfx: TrackGraphics | None = None
        self.sim: RaceSim | None = None
        self.car_gfx: CarGraphics | None = None
        self.msg = ""
        self.msg_t = 0.0
        self.keys = key.KeyStateHandler()

        self.win = make_window(1280, 800, f"carql build - {self.name}")
        self.win.push_handlers(on_draw=self.on_draw, on_resize=self.on_resize, on_key_press=self.on_key_press,
                               on_mouse_press=self.on_mouse_press, on_mouse_release=self.on_mouse_release,
                               on_mouse_drag=self.on_mouse_drag, on_mouse_motion=self.on_mouse_motion,
                               on_mouse_scroll=self.on_mouse_scroll)
        self.win.push_handlers(self.keys)      # on top: sees every press/release for the test drive
        self.cam = Camera(self.size, margin=14, bottom_reserved=8)
        self.world_batch = pyglet.graphics.Batch()
        self.overlay_batch = pyglet.graphics.Batch()
        self.hud_batch = pyglet.graphics.Batch()
        self.g_grid = pyglet.graphics.Group(order=-1)
        self.g_cl = pyglet.graphics.Group(order=10)
        self.grid_lines: list[shapes.Line] = []
        self.cl_lines: list[shapes.Line] = []
        self.ctrl_dots: list[shapes.Circle] = []
        self.start_marker = shapes.Triangle(0, 0, 0, 0, 0, 0, color=(80, 140, 255), batch=self.overlay_batch,
                                            group=self.g_cl)
        self.info = Panel(self.hud_batch, 12, 0, 560, 4)
        self.help = HelpOverlay(self.hud_batch, HELP)
        self._build_grid()
        self.on_resize(self.win.width, self.win.height)
        self.regenerate()
        pyglet.clock.schedule_interval(self.update, 1 / 60.0)

    # ------------------------------------------------------------ geometry
    def _build_grid(self) -> None:
        step = 100
        w, h = self.size
        col = (60, 60, 66)
        for x in np.arange(0, w + 1, step):
            self.grid_lines.append(shapes.Line(x, 0, x, h, thickness=1, color=col, batch=self.world_batch, group=self.g_grid))
        for y in np.arange(0, h + 1, step):
            self.grid_lines.append(shapes.Line(0, y, w, y, thickness=1, color=col, batch=self.world_batch, group=self.g_grid))
        self.grid_lines.append(shapes.Line(0, 0, w, 0, thickness=2, color=(90, 90, 96), batch=self.world_batch, group=self.g_grid))
        self.grid_lines.append(shapes.Line(0, 0, 0, h, thickness=2, color=(90, 90, 96), batch=self.world_batch, group=self.g_grid))
        self.grid_lines.append(shapes.Line(w, 0, w, h, thickness=2, color=(90, 90, 96), batch=self.world_batch, group=self.g_grid))
        self.grid_lines.append(shapes.Line(0, h, w, h, thickness=2, color=(90, 90, 96), batch=self.world_batch, group=self.g_grid))

    def regenerate(self) -> None:
        """Rebuild walls/gates from the control points and refresh the graphics."""
        self.dirty = False
        pts = np.asarray(self.points, dtype=float)
        if self.track_gfx is not None:
            self.track_gfx.sprite.delete()
            for s in self.track_gfx.walls + self.track_gfx.gates:
                s.delete()
            self.track_gfx = None
        self.track = None
        if len(pts) >= (3 if self.closed else 2):
            try:
                self.track = Track.from_centerline(pts, self.width, name=self.name, size=self.size,
                                                   closed=self.closed, gate_spacing=self.gate_spacing,
                                                   smooth=self.smooth)
                self.track_gfx = TrackGraphics(self.track, self.world_batch, self.cam)
            except Exception as e:      # degenerate geometry while drawing
                self.toast(f"cannot generate: {e}")
        self._refresh_overlay()
        if self.sim is not None and self.track is not None:
            self.start_test_drive()

    def _refresh_overlay(self) -> None:
        for s in self.cl_lines + self.ctrl_dots:
            s.delete()
        self.cl_lines, self.ctrl_dots = [], []
        pts = np.asarray(self.points, dtype=float)
        t = self.cam.px(1.5)
        if len(pts) >= 2:
            sm = smooth_centerline(pts, self.smooth, self.closed and len(pts) >= 3)
            loop = np.vstack([sm, sm[:1]]) if (self.closed and len(pts) >= 3) else sm
            for a, b in zip(loop[:-1], loop[1:]):
                self.cl_lines.append(shapes.Line(a[0], a[1], b[0], b[1], thickness=t, color=(255, 255, 255, 120),
                                                 batch=self.overlay_batch, group=self.g_cl))
        r = self.cam.px(5)
        for i, (x, y) in enumerate(pts):
            col = (255, 230, 80) if i == 0 else ((255, 120, 120) if i == self.hover_idx else (230, 230, 230))
            self.ctrl_dots.append(shapes.Circle(x, y, r, color=col, batch=self.overlay_batch, group=self.g_cl))
        if self.track is not None:
            sx, sy = self.track.start_pos
            a = np.radians(self.track.start_angle)
            d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
            L = self.cam.px(14); Wd = self.cam.px(8)
            p1 = (sx + d[0] * L, sy + d[1] * L); p2 = (sx + n[0] * Wd, sy + n[1] * Wd); p3 = (sx - n[0] * Wd, sy - n[1] * Wd)
            self.start_marker.x, self.start_marker.y = p1
            self.start_marker.x2, self.start_marker.y2 = p2
            self.start_marker.x3, self.start_marker.y3 = p3
            self.start_marker.visible = True
        else:
            self.start_marker.visible = False

    def push_history(self) -> None:
        self.history.append([list(p) for p in self.points])
        self.history = self.history[-100:]

    # ------------------------------------------------------------ helpers
    def nearest_point(self, wx: float, wy: float, radius_px: float = 12) -> int | None:
        if not self.points:
            return None
        pts = np.asarray(self.points)
        d = np.hypot(pts[:, 0] - wx, pts[:, 1] - wy)
        i = int(np.argmin(d))
        return i if d[i] <= self.cam.px(radius_px) else None

    def nearest_segment(self, wx: float, wy: float) -> int:
        pts = np.asarray(self.points)
        n = len(pts)
        best, best_d = n - 1, np.inf
        segs = n if self.closed else n - 1
        for i in range(segs):
            a, b = pts[i], pts[(i + 1) % n]
            ab = b - a
            tt = np.clip(np.dot([wx - a[0], wy - a[1]], ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
            p = a + tt * ab
            d = np.hypot(p[0] - wx, p[1] - wy)
            if d < best_d:
                best, best_d = i, d
        return best

    def toast(self, text: str, seconds: float = 2.0) -> None:
        self.msg, self.msg_t = text, time.time() + seconds

    # ------------------------------------------------------------- mouse
    def on_mouse_press(self, x, y, button, modifiers) -> None:
        if self.sim is not None:
            return
        wx, wy = self.cam.to_world(x, y)
        idx = self.nearest_point(wx, wy)
        if button == mouse.LEFT:
            if modifiers & key.MOD_SHIFT and len(self.points) >= 2:
                self.push_history()
                i = self.nearest_segment(wx, wy)
                self.points.insert(i + 1, [wx, wy])
                self.drag_idx = i + 1
            elif idx is not None:
                self.push_history()
                self.drag_idx = idx
            else:
                if not (0 <= wx <= self.size[0] and 0 <= wy <= self.size[1]):
                    return
                self.push_history()
                self.points.append([wx, wy])
                self.drag_idx = len(self.points) - 1
            self._refresh_overlay()
        elif button == mouse.RIGHT and idx is not None:
            self.push_history()
            del self.points[idx]
            self.hover_idx = None
            self.regenerate()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers) -> None:
        if self.drag_idx is None:
            return
        wx, wy = self.cam.to_world(x, y)
        self.points[self.drag_idx] = [float(np.clip(wx, 0, self.size[0])), float(np.clip(wy, 0, self.size[1]))]
        self._refresh_overlay()

    def on_mouse_release(self, x, y, button, modifiers) -> None:
        if self.drag_idx is not None:
            self.drag_idx = None
            self.regenerate()

    def on_mouse_motion(self, x, y, dx, dy) -> None:
        wx, wy = self.cam.to_world(x, y)
        h = self.nearest_point(wx, wy)
        if h != self.hover_idx:
            self.hover_idx = h
            self._refresh_overlay()

    def on_mouse_scroll(self, x, y, sx, sy) -> None:
        self.width = float(np.clip(self.width + 5 * np.sign(sy), 20, 400))
        self.regenerate()

    # -------------------------------------------------------------- keys
    def on_key_press(self, symbol, modifiers) -> bool | None:
        if symbol == key.ESCAPE:
            if self.sim is not None:
                self.stop_test_drive()
                return True
            self.win.close()
            return True
        if symbol == key.H:
            self.help.set_visible(not self.help.visible)
        elif symbol == key.T:
            self.stop_test_drive() if self.sim is not None else self.start_test_drive()
        elif self.sim is not None:
            if symbol == key.R:
                self.sim.reset_all()
            return None
        elif symbol == key.BRACKETLEFT:
            self.width = max(20.0, self.width - 5); self.regenerate()
        elif symbol == key.BRACKETRIGHT:
            self.width = min(400.0, self.width + 5); self.regenerate()
        elif symbol == key.COMMA:
            self.gate_spacing = max(20.0, self.gate_spacing - 10); self.regenerate()
        elif symbol == key.PERIOD:
            self.gate_spacing = min(300.0, self.gate_spacing + 10); self.regenerate()
        elif symbol == key.C:
            self.closed = not self.closed; self.regenerate()
        elif symbol == key.D:
            self.push_history()
            if self.closed and self.points:
                first = self.points[0]
                self.points = [first] + self.points[1:][::-1]
            else:
                self.points = self.points[::-1]
            self.regenerate()
        elif symbol == key._1 and self.hover_idx is not None and self.closed:
            self.push_history()
            self.points = self.points[self.hover_idx:] + self.points[:self.hover_idx]
            self.hover_idx = 0
            self.regenerate()
        elif symbol == key.BACKSPACE and self.points:
            self.push_history()
            self.points.pop()
            self.regenerate()
        elif symbol == key.Z and (modifiers & (key.MOD_CTRL | key.MOD_COMMAND)) and self.history:
            self.points = self.history.pop()
            self.regenerate()
        elif symbol == key.N:
            self.push_history()
            self.points = []
            self.regenerate()
        elif symbol == key.G and self.track_gfx:
            self.track_gfx.show_gates = not self.track_gfx.show_gates; self.track_gfx.update_thickness()
        elif symbol == key.W and self.track_gfx:
            self.track_gfx.show_walls = not self.track_gfx.show_walls; self.track_gfx.update_thickness()
        elif symbol == key.S:
            self.save()
        return None

    def save(self) -> None:
        if self.track is None:
            self.toast("nothing to save yet")
            return
        self.track.name = self.name
        p = self.track.save(self.path)
        self.toast(f"saved {p}", 3)
        print(f"saved {p}: {self.track.describe()}")

    # ------------------------------------------------------- test drive
    def start_test_drive(self) -> None:
        if self.track is None:
            self.toast("draw a track first")
            return
        if self.car_gfx is not None:
            self.car_gfx = None
        d = Driver("you", None, PALETTE[2])
        self.sim = RaceSim(self.track, [d], Config(), respawn=True, respawn_delay=30, manual_control=True)
        car = self.sim.cfg.car
        self.car_gfx = CarGraphics(self.overlay_batch, self.cam, car.length, car.width, d.color, self.sim.env.cfg.n_rays,
                                   group_order=12)
        self.car_gfx.set_rays_visible(True)
        self.toast("test drive: arrow keys, R reset, T back to editing")

    def stop_test_drive(self) -> None:
        if self.car_gfx is not None:
            for s in [self.car_gfx.body, self.car_gfx.nose, self.car_gfx.shadow, *self.car_gfx.rays, *self.car_gfx.dots]:
                s.delete()
        self.car_gfx = None
        self.sim = None

    # ------------------------------------------------------------ frame
    def on_resize(self, w, h) -> None:
        self.cam.resize(w, h)
        if self.track_gfx:
            self.track_gfx.update_thickness()
        self.info.move(12, h - 12)
        self.help.layout(w, h)
        self._refresh_overlay()

    def update(self, dt) -> None:
        if self.sim is not None:
            from .show import KEY_ACTION, N_ACTIONS
            k = self.keys
            combo = (bool(k[key.UP]), bool(k[key.DOWN]), bool(k[key.LEFT]), bool(k[key.RIGHT]))
            self.sim.drivers[0].manual_action = KEY_ACTION.get(combo, N_ACTIONS - 1)
            self.sim.step()
        import os
        if os.environ.get("CARQL_SCREENSHOT"):
            from .show import _headless_hook
            _headless_hook(self)

    def on_draw(self) -> None:
        win = self.win
        pyglet.gl.glClearColor(BG[0] / 255, BG[1] / 255, BG[2] / 255, 1)
        win.clear()
        if self.sim is not None and self.car_gfx is not None:
            env = self.sim.env
            self.car_gfx.set_alpha(255 if env.alive[0] else 70)
            self.car_gfx.update(env.pos[0, 0], env.pos[0, 1], env.heading_deg[0], env.ray_hits[0])
            if self.track_gfx:
                self.track_gfx.highlight_gate(int(env.gate_idx[0]))
        win.view = self.cam.matrix
        self.world_batch.draw()
        self.overlay_batch.draw()
        win.view = Mat4()
        t = self.track
        lines = [(f"{self.name}  ->  {self.path}", (255, 220, 120, 255))]
        if t is not None:
            lines.append(f"{len(self.points)} control points | width {self.width:.0f} | gate spacing {self.gate_spacing:.0f} "
                         f"| {'closed' if self.closed else 'open'} | {len(t.walls)} walls, {len(t.gates)} gates, lap ~{t.lap_length():.0f}")
        else:
            lines.append(f"{len(self.points)} control points | width {self.width:.0f} | click to add points "
                         f"({'need 3 for a loop' if self.closed else 'need 2'})")
        if self.sim is not None:
            env = self.sim.env
            lines.append(f"TEST DRIVE  gates {env.score[0]}  laps {env.laps[0]}  speed {env.vel[0]:.1f}  "
                         f"return {self.sim.total_reward[0]:.1f}  crashes {self.sim.crashes[0]}")
        else:
            lines.append("left click add/drag | right click delete | scroll width | S save | T test drive | H help")
        lines.append((self.msg if time.time() < self.msg_t else "", (255, 255, 255, 255)))
        self.info.set(lines)
        self.hud_batch.draw()


def build(track_path: str | None, size=(1260.0, 700.0), width: float = 90.0, name: str | None = None) -> None:
    track = Track.load(track_path) if track_path else None
    BuilderWindow(track, size=size, width=width, name=name)
    pyglet.app.run()
