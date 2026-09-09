"""pyglet 2 window with a world->screen view, anti-aliased track rendering and HUD helpers."""
from __future__ import annotations

import numpy as np
import pyglet
from pyglet import shapes
from pyglet.gl import GL_LINEAR, Config
from pyglet.math import Mat4, Vec3

from ..env import ACTIONS
from ..track import Track, render_background

BG = (40, 40, 44)
ROAD = (150, 150, 152)
GRASS = (72, 72, 76)
WALL = (235, 80, 80)
GATE = (90, 200, 110)
GATE_NEXT = (255, 230, 80)
TEXT = (235, 235, 235, 255)
DIM = (170, 170, 170, 255)
PANEL = (16, 16, 20, 175)

ACTION_KEYS = ["L", "R", "A", "B", "AL", "AR", "BL", "BR", "-"]


def make_window(width: int, height: int, title: str, **kw) -> pyglet.window.Window:
    """A resizable window with 4x MSAA when the driver offers it (falls back silently)."""
    for samples in (8, 4, 2, 0):
        try:
            if samples:
                cfg = Config(sample_buffers=1, samples=samples, double_buffer=True)
            else:
                cfg = Config(double_buffer=True)
            return pyglet.window.Window(width, height, title, resizable=True, config=cfg, **kw)
        except pyglet.window.NoSuchConfigException:
            continue
        except Exception:      # e.g. headless drivers raising other errors
            continue
    return pyglet.window.Window(width, height, title, resizable=True, **kw)


class Camera:
    """Fit a world rectangle into the window (letterboxed), y up."""

    def __init__(self, world_size, margin: int = 16, top_reserved: int = 0, bottom_reserved: int = 0):
        self.world = world_size
        self.margin = margin
        self.top = top_reserved
        self.bottom = bottom_reserved
        self.scale = 1.0
        self.ox = self.oy = 0.0
        self.win = (1, 1)

    def resize(self, w: int, h: int) -> None:
        self.win = (w, h)
        avail_w = max(1, w - 2 * self.margin)
        avail_h = max(1, h - 2 * self.margin - self.top - self.bottom)
        self.scale = min(avail_w / self.world[0], avail_h / self.world[1])
        self.ox = (w - self.world[0] * self.scale) / 2
        self.oy = self.bottom + (avail_h + 2 * self.margin - self.world[1] * self.scale) / 2

    @property
    def matrix(self) -> Mat4:
        return Mat4.from_translation(Vec3(self.ox, self.oy, 0)) @ Mat4.from_scale(Vec3(self.scale, self.scale, 1))

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return self.ox + x * self.scale, self.oy + y * self.scale

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        return (sx - self.ox) / self.scale, (sy - self.oy) / self.scale

    def px(self, n: float = 1.0) -> float:
        """World units per ``n`` screen pixels (for constant-width lines)."""
        return n / self.scale


def pil_to_texture(img):
    raw = img.tobytes()
    data = pyglet.image.ImageData(img.width, img.height, "RGB", raw, pitch=-img.width * 3)
    tex = data.get_texture()
    pyglet.gl.glBindTexture(tex.target, tex.id)
    pyglet.gl.glTexParameteri(tex.target, pyglet.gl.GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    pyglet.gl.glTexParameteri(tex.target, pyglet.gl.GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    return tex


class TrackGraphics:
    """Retained-mode drawing of a track: background texture, walls, gates."""

    def __init__(self, track: Track, batch: pyglet.graphics.Batch, cam: Camera, texture_scale: float = 2.0):
        self.track = track
        self.batch = batch
        self.cam = cam
        self.g_bg = pyglet.graphics.Group(order=0)
        self.g_lines = pyglet.graphics.Group(order=1)
        self.texture_scale = texture_scale
        img = render_background(track, texture_scale, road=ROAD, bg=GRASS)
        self.sprite = pyglet.sprite.Sprite(pil_to_texture(img), 0, 0, batch=batch, group=self.g_bg)
        self.sprite.scale = 1.0 / texture_scale
        self.walls = [shapes.Line(x1, y1, x2, y2, thickness=1, color=WALL, batch=batch, group=self.g_lines)
                      for x1, y1, x2, y2 in track.walls]
        self.gates = [shapes.Line(x1, y1, x2, y2, thickness=1, color=GATE, batch=batch, group=self.g_lines)
                      for x1, y1, x2, y2 in track.gates]
        self.show_walls = True
        self.show_gates = True
        self.update_thickness()

    def update_thickness(self) -> None:
        t = self.cam.px(2.0)
        for w in self.walls:
            w.thickness = t
            w.visible = self.show_walls
        for g in self.gates:
            g.thickness = self.cam.px(1.5)
            g.visible = self.show_gates

    def highlight_gate(self, idx: int | None, color=GATE_NEXT) -> None:
        for i, g in enumerate(self.gates):
            g.color = color if i == idx else GATE
            g.thickness = self.cam.px(3.0 if i == idx else 1.5)


class CarGraphics:
    """One car: body + nose + (optional) rays and a label drawn in screen space."""

    def __init__(self, batch, cam: Camera, length: float, width: float, color, n_rays: int = 0,
                 group_order: int = 5):
        self.cam = cam
        self.length, self.width = length, width
        self.color = color
        g = pyglet.graphics.Group(order=group_order)
        g_rays = pyglet.graphics.Group(order=group_order - 1)
        self.shadow = shapes.Rectangle(0, 0, length, width, color=(0, 0, 0, 90), batch=batch,
                                       group=pyglet.graphics.Group(order=group_order - 2))
        self.body = shapes.Rectangle(0, 0, length, width, color=color, batch=batch, group=g)
        self.nose = shapes.Rectangle(0, 0, length * 0.3, width * 0.7, color=_lighten(color), batch=batch, group=g)
        for s in (self.shadow, self.body):
            s.anchor_position = (length / 2, width / 2)
        self.nose.anchor_position = (-length * 0.15, width * 0.35)
        self.rays = [shapes.Line(0, 0, 0, 0, thickness=1, color=(255, 255, 255, 70), batch=batch, group=g_rays)
                     for _ in range(n_rays)]
        self.dots = [shapes.Circle(0, 0, 2.5, color=(255, 255, 255, 160), batch=batch, group=g_rays)
                     for _ in range(n_rays)]
        self.set_rays_visible(False)

    def set_rays_visible(self, v: bool) -> None:
        for r in self.rays:
            r.visible = v
        for d in self.dots:
            d.visible = v

    def set_alpha(self, a: int) -> None:
        self.body.opacity = a
        self.nose.opacity = a
        self.shadow.opacity = min(a, 90)

    def update(self, x: float, y: float, heading_deg: float, ray_hits=None) -> None:
        self.body.position = (x, y)
        self.nose.position = (x, y)
        self.shadow.position = (x + self.cam.px(2), y - self.cam.px(2))
        rot = -heading_deg          # pyglet shapes rotate clockwise
        self.body.rotation = rot
        self.nose.rotation = rot
        self.shadow.rotation = rot
        if ray_hits is not None and self.rays:
            r = self.cam.px(2.5)
            for line, dot, (hx, hy) in zip(self.rays, self.dots, ray_hits):
                line.x, line.y, line.x2, line.y2 = x, y, hx, hy
                line.thickness = self.cam.px(1.0)
                dot.position = (hx, hy)
                dot.radius = r


def _lighten(c, f: float = 0.45):
    return tuple(int(v + (255 - v) * f) for v in c[:3])


class Panel:
    """A translucent rectangle with a stack of labels (screen space)."""

    def __init__(self, batch, x: float, y_top: float, width: float, lines: int, font_size: int = 11,
                 order: int = 20, line_h: int | None = None, pad: int = 8):
        self.batch = batch
        self.x, self.y_top, self.width = x, y_top, width
        self.line_h = line_h or int(font_size * 1.6)
        self.pad = pad
        self.height = lines * self.line_h + 2 * pad
        g0 = pyglet.graphics.Group(order=order)
        g1 = pyglet.graphics.Group(order=order + 1)
        self.rect = shapes.Rectangle(x, y_top - self.height, width, self.height, color=PANEL, batch=batch, group=g0)
        self.labels = [pyglet.text.Label("", x=x + pad, y=y_top - pad - i * self.line_h, anchor_y="top",
                                         font_size=font_size, color=TEXT, batch=batch, group=g1,
                                         font_name=("Menlo", "Consolas", "DejaVu Sans Mono", "monospace"))
                       for i in range(lines)]

    def set(self, lines: list[str | tuple[str, tuple]]) -> None:
        for i, lab in enumerate(self.labels):
            if i < len(lines):
                item = lines[i]
                if isinstance(item, tuple):
                    lab.text, lab.color = item[0], item[1]
                else:
                    lab.text, lab.color = item, TEXT
            else:
                lab.text = ""

    def move(self, x: float, y_top: float) -> None:
        self.x, self.y_top = x, y_top
        self.rect.position = (x, y_top - self.height)
        for i, lab in enumerate(self.labels):
            lab.position = (x + self.pad, y_top - self.pad - i * self.line_h, 0)

    def set_visible(self, v: bool) -> None:
        self.rect.visible = v
        for lab in self.labels:
            lab.visible = v


class BarChart:
    """Per-action value bars (Q-values or policy probabilities)."""

    def __init__(self, batch, x: float, y: float, width: float, height: float, order: int = 20):
        self.x, self.y, self.width, self.height = x, y, width, height
        g0 = pyglet.graphics.Group(order=order)
        g1 = pyglet.graphics.Group(order=order + 1)
        n = len(ACTIONS)
        self.n = n
        self.bg = shapes.Rectangle(x, y, width, height, color=PANEL, batch=batch, group=g0)
        self.bw = (width - 16) / n
        self.bars = [shapes.Rectangle(x + 8 + i * self.bw + 2, y + 22, self.bw - 4, 1, color=(120, 160, 220),
                                      batch=batch, group=g1) for i in range(n)]
        self.keys = [pyglet.text.Label(ACTION_KEYS[i], x=x + 8 + i * self.bw + self.bw / 2, y=y + 6,
                                       anchor_x="center", font_size=9, color=DIM, batch=batch, group=g1)
                     for i in range(n)]
        self.title = pyglet.text.Label("", x=x + 8, y=y + height - 6, anchor_y="top", font_size=10,
                                       color=DIM, batch=batch, group=g1)

    def set(self, label: str, values: np.ndarray, chosen: int) -> None:
        v = np.asarray(values, dtype=float)
        lo, hi = float(v.min()), float(v.max())
        span = hi - lo if hi > lo else 1.0
        usable = self.height - 22 - 26
        for i, b in enumerate(self.bars):
            h = max(1.0, (v[i] - lo) / span * usable) if span else 1.0
            b.height = h
            b.color = (255, 200, 60) if i == chosen else (120, 160, 220)
        self.title.text = f"{label}  max {hi:+.2f}  min {lo:+.2f}  -> {ACTIONS[chosen]}"

    def move(self, x: float, y: float) -> None:
        self.x, self.y = x, y
        self.bg.position = (x, y)
        for i, b in enumerate(self.bars):
            b.position = (x + 8 + i * self.bw + 2, y + 22)
            self.keys[i].position = (x + 8 + i * self.bw + self.bw / 2, y + 6, 0)
        self.title.position = (x + 8, y + self.height - 6, 0)

    def set_visible(self, v: bool) -> None:
        self.bg.visible = v
        self.title.visible = v
        for b in self.bars:
            b.visible = v
        for k in self.keys:
            k.visible = v


class HelpOverlay:
    def __init__(self, batch, text: str, order: int = 40):
        g0 = pyglet.graphics.Group(order=order)
        g1 = pyglet.graphics.Group(order=order + 1)
        self.rect = shapes.Rectangle(0, 0, 10, 10, color=(10, 10, 14, 220), batch=batch, group=g0)
        self.label = pyglet.text.Label(text, x=0, y=0, anchor_x="center", anchor_y="center", multiline=True,
                                       width=520, font_size=11, color=TEXT, batch=batch, group=g1,
                                       font_name=("Menlo", "Consolas", "DejaVu Sans Mono", "monospace"))
        self.set_visible(False)

    def layout(self, w: int, h: int) -> None:
        self.rect.width, self.rect.height = w, h
        self.label.position = (w / 2, h / 2, 0)

    def set_visible(self, v: bool) -> None:
        self.rect.visible = v
        self.label.visible = v

    @property
    def visible(self) -> bool:
        return self.rect.visible
