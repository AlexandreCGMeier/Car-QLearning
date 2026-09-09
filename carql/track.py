"""Track representation, generation and (de)serialisation.

A track is a set of wall segments, an ordered list of reward gates and a start
pose, all in *world* coordinates (y up, arbitrary units; the classic track is
1260 x 700).  Tracks are stored as JSON in ``tracks/``.

Two ways to make a track:

* ``Track.from_centerline`` -- give it a polyline and a road width; the walls
  (as contours of the rasterised stroke) and evenly spaced gates are generated.
  This is what the builder (``carql build``) and the PNG importer use.
* explicit ``walls`` + ``gates`` -- what the converted legacy track uses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry as geo

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"
RASTER_SCALE = 2          # supersampling factor for wall extraction
GATE_OVERHANG = 1.15      # gates are a bit wider than the road so they always span it


@dataclass
class Track:
    name: str
    size: tuple[float, float]
    walls: np.ndarray                      # (W, 4)
    gates: np.ndarray                      # (G, 4), ordered
    start_pos: tuple[float, float]
    start_angle: float                     # degrees, CCW from +x
    closed: bool = True
    centerline: np.ndarray | None = None   # (N, 2) control points, if generated
    width: float | None = None             # road width, if generated
    gate_spacing: float = 60.0
    smooth: int = 2                        # Chaikin iterations applied to the centerline
    image: str | None = None               # optional background image (relative to repo root)
    source: str | None = None              # path this track was loaded from
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: str | Path) -> "Track":
        path = resolve_track_path(path)
        with open(path) as f:
            d = json.load(f)
        t = cls.from_dict(d)
        t.source = str(path)
        return t

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        self.source = str(path)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Track":
        cl = d.get("centerline")
        return cls(
            name=d["name"],
            size=tuple(d["size"]),
            walls=np.asarray(d["walls"], dtype=float).reshape(-1, 4),
            gates=np.asarray(d["gates"], dtype=float).reshape(-1, 4),
            start_pos=tuple(d["start"]["pos"]),
            start_angle=float(d["start"]["angle"]),
            closed=bool(d.get("closed", True)),
            centerline=None if cl is None else np.asarray(cl, dtype=float),
            width=d.get("width"),
            gate_spacing=float(d.get("gate_spacing", 60.0)),
            smooth=int(d.get("smooth", 2)),
            image=d.get("image"),
            meta=d.get("meta", {}),
        )

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "size": [float(self.size[0]), float(self.size[1])],
            "closed": self.closed,
            "start": {"pos": [round(float(self.start_pos[0]), 2), round(float(self.start_pos[1]), 2)],
                      "angle": round(float(self.start_angle), 2)},
            "gate_spacing": self.gate_spacing,
            "smooth": self.smooth,
        }
        if self.centerline is not None:
            d["centerline"] = np.round(self.centerline, 2).tolist()
            d["width"] = self.width
        if self.image:
            d["image"] = self.image
        if self.meta:
            d["meta"] = self.meta
        d["walls"] = np.round(self.walls, 2).tolist()
        d["gates"] = np.round(self.gates, 2).tolist()
        return d

    # ------------------------------------------------------------ helpers
    @property
    def n_gates(self) -> int:
        return len(self.gates)

    @property
    def gate_centers(self) -> np.ndarray:
        return (self.gates[:, :2] + self.gates[:, 2:]) / 2

    def smoothed_centerline(self) -> np.ndarray | None:
        if self.centerline is None:
            return None
        return smooth_centerline(self.centerline, self.smooth, self.closed)

    def lap_length(self) -> float:
        """Approximate lap length via gate centres (world units)."""
        c = self.gate_centers
        pts = np.vstack([[self.start_pos], c]) if not self.closed else c
        return float(geo.polyline_length(pts, closed=self.closed)[-1])

    def describe(self) -> str:
        kind = "generated" if self.centerline is not None else "explicit walls"
        return (f"{self.name}: {self.size[0]:.0f}x{self.size[1]:.0f}, {len(self.walls)} walls, "
                f"{len(self.gates)} gates, {'closed' if self.closed else 'open'}, {kind}"
                + (f", width {self.width:.0f}" if self.width else ""))

    # --------------------------------------------------------- generation
    @classmethod
    def from_centerline(cls, points, width: float, *, name: str = "untitled",
                        size=None, closed: bool = True, gate_spacing: float = 60.0,
                        smooth: int = 2, simplify_tol: float = 1.5,
                        start_offset: float = 0.0) -> "Track":
        """Build a full track (walls + gates + start) from a centerline polyline."""
        pts = np.asarray(points, dtype=float)
        if size is None:
            lo = pts.min(0) - width
            hi = pts.max(0) + width
            size = (float(hi[0] + lo[0]), float(hi[1] + lo[1]))
        walls, gates, start_pos, start_angle = generate_geometry(
            pts, width, size, closed, gate_spacing, smooth, simplify_tol, start_offset)
        return cls(name=name, size=size, walls=walls, gates=gates, start_pos=start_pos,
                   start_angle=start_angle, closed=closed, centerline=pts, width=width,
                   gate_spacing=gate_spacing, smooth=smooth)

    def regenerate(self) -> None:
        """Recompute walls/gates/start from the centerline (after editing)."""
        if self.centerline is None:
            raise ValueError("track has no centerline to regenerate from")
        self.walls, self.gates, self.start_pos, self.start_angle = generate_geometry(
            self.centerline, self.width, self.size, self.closed, self.gate_spacing, self.smooth)


# ---------------------------------------------------------------------------
# generation internals
# ---------------------------------------------------------------------------
def resolve_track_path(path: str | Path) -> Path:
    """Accept 'classic', 'classic.json', 'tracks/classic.json' or an absolute path."""
    p = Path(path)
    if p.exists():
        return p
    cands = [p.with_suffix(".json"), TRACKS_DIR / p.name, TRACKS_DIR / (p.name + ".json")]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"track not found: {path} (looked in {TRACKS_DIR})")


def smooth_centerline(points: np.ndarray, iterations: int, closed: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts.copy()
    return geo.chaikin(pts, iterations, closed)


def rasterize_road(points: np.ndarray, width: float, size, closed: bool, scale: float,
                   supersample: int = 1):
    """Draw the road stroke into a binary PIL image (L mode, road = 255).

    The image has ``size * scale * supersample`` pixels; y is flipped so that
    row 0 is the top of the world.
    """
    from PIL import Image, ImageDraw

    s = scale * supersample
    w, h = int(round(size[0] * s)), int(round(size[1] * s))
    img = Image.new("L", (max(w, 1), max(h, 1)), 0)
    draw = ImageDraw.Draw(img)
    pts = np.asarray(points, dtype=float)
    if closed and len(pts) > 1:
        pts = np.vstack([pts, pts[:1]])
    px = [(float(x * s), float((size[1] - y) * s)) for x, y in pts]
    r = width * s / 2
    if len(px) >= 2:
        draw.line(px, fill=255, width=int(round(width * s)), joint="curve")
    for x, y in px:      # round caps / joints
        draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
    if supersample > 1:
        img = img.resize((max(1, int(round(size[0] * scale))), max(1, int(round(size[1] * scale)))),
                         Image.LANCZOS)
    return img


def walls_from_mask(mask: np.ndarray, size, scale: float, simplify_tol: float) -> np.ndarray:
    """Extract wall segments as simplified contours of a road mask (rows = top)."""
    from skimage import measure

    padded = np.pad(mask.astype(float), 1)
    contours = measure.find_contours(padded, 0.5)
    walls = []
    for c in contours:
        c = c - 1.0                              # undo padding
        pts = np.stack([c[:, 1] / scale, size[1] - c[:, 0] / scale], axis=1)  # (x, y) world
        pts = geo.simplify_polyline(pts, simplify_tol)
        if len(pts) < 3:
            continue
        nxt = np.roll(pts, -1, axis=0)
        walls.append(np.concatenate([pts, nxt], axis=1))
    if not walls:
        return np.zeros((0, 4))
    return np.vstack(walls)


def generate_geometry(points, width, size, closed, gate_spacing, smooth,
                      simplify_tol: float = 1.5, start_offset: float = 0.0):
    pts = np.asarray(points, dtype=float)
    cl = smooth_centerline(pts, smooth, closed)
    img = rasterize_road(cl, width, size, closed, RASTER_SCALE)
    mask = np.asarray(img) > 127
    walls = walls_from_mask(mask, size, RASTER_SCALE, simplify_tol)

    # gates along the resampled centerline
    dense = geo.resample_polyline(cl, max(2.0, gate_spacing / 8), closed)
    cum = geo.polyline_length(dense, closed)
    total = cum[-1]
    half = width / 2 * GATE_OVERHANG
    if closed:
        s_vals = np.arange(gate_spacing, total - gate_spacing / 2, gate_spacing)
    else:
        s_vals = np.arange(gate_spacing, total - gate_spacing * 0.25, gate_spacing)
        s_vals = np.append(s_vals, total - 1.0)   # finish line at the end
    s_vals = (s_vals + start_offset) % total if closed else s_vals
    dense_closed = np.vstack([dense, dense[:1]]) if closed else dense
    tang = geo.tangents(dense, closed)
    gates = []
    for s in s_vals:
        i = int(np.searchsorted(cum, s, side="right") - 1)
        i = min(max(i, 0), len(dense_closed) - 2)
        seg_len = cum[i + 1] - cum[i]
        f = 0.0 if seg_len == 0 else (s - cum[i]) / seg_len
        p = dense_closed[i] * (1 - f) + dense_closed[i + 1] * f
        t = tang[i % len(tang)] * (1 - f) + tang[(i + 1) % len(tang)] * f
        t /= max(np.linalg.norm(t), 1e-9)
        n = geo.rot90(t)
        a, b = p - n * half, p + n * half
        gates.append([a[0], a[1], b[0], b[1]])
    gates = np.asarray(gates, dtype=float).reshape(-1, 4)

    # start pose
    s0 = start_offset % total if closed else 0.0
    i = int(np.searchsorted(cum, s0, side="right") - 1)
    i = min(max(i, 0), len(dense_closed) - 2)
    seg_len = cum[i + 1] - cum[i]
    f = 0.0 if seg_len == 0 else (s0 - cum[i]) / seg_len
    p0 = dense_closed[i] * (1 - f) + dense_closed[i + 1] * f
    t0 = tang[i % len(tang)]
    return walls, gates, (float(p0[0]), float(p0[1])), float(geo.angle_of(t0))


# ---------------------------------------------------------------------------
# PNG import: hand-drawn stroke -> centerline + width
# ---------------------------------------------------------------------------
def import_png(path: str | Path, *, name: str | None = None, height: float = 700.0,
               closed: bool = True, gate_spacing: float = 60.0, invert: bool = False,
               reverse: bool = False, control_spacing: float = 25.0) -> Track:
    """Turn a drawn road image (bright road on dark background) into a track.

    The stroke is skeletonised to recover the centerline and the distance
    transform gives the road width; the result is a regular generated track.
    """
    from PIL import Image
    from scipy import ndimage
    from skimage import measure, morphology

    path = Path(path)
    img = Image.open(path).convert("L")
    scale = height / img.height
    size = (img.width * scale, height)
    g = np.asarray(img).astype(float)
    thresh = (g.min() + g.max()) / 2
    mask = g < thresh if invert else g > thresh
    # keep the largest blob only
    lab = measure.label(mask)
    if lab.max() > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        mask = lab == counts.argmax()
    dist = ndimage.distance_transform_edt(mask)
    skel = morphology.skeletonize(mask)
    skel = _prune_skeleton(skel, closed)
    order = _trace_skeleton(skel)
    if len(order) < 10:
        raise ValueError("could not trace a centerline in the image")
    width_px = 2 * float(np.median(dist[order[:, 0], order[:, 1]]))
    pts = np.stack([order[:, 1] * scale, height - order[:, 0] * scale], axis=1)
    # clockwise by default (matches the classic track); reverse flips it
    area = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1]) / 2
    if (area > 0) != reverse:
        pts = pts[::-1]
    # start at the left-most point
    i0 = int(np.argmin(pts[:, 0]))
    pts = np.roll(pts, -i0, axis=0)
    ctrl = geo.resample_polyline(geo.simplify_polyline(pts, 1.0), control_spacing, closed)
    return Track.from_centerline(ctrl, width_px * scale, name=name or path.stem, size=size,
                                 closed=closed, gate_spacing=gate_spacing, smooth=2)


def _neighbors8(skel: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    k = np.ones((3, 3), int)
    k[1, 1] = 0
    return ndimage.convolve(skel.astype(int), k, mode="constant")


def _prune_skeleton(skel: np.ndarray, closed: bool, max_iter: int = 400) -> np.ndarray:
    skel = skel.copy()
    for _ in range(max_iter if closed else 25):
        nb = _neighbors8(skel)
        ends = skel & (nb <= 1)
        if not ends.any():
            break
        skel[ends] = False
    return skel


def _trace_skeleton(skel: np.ndarray) -> np.ndarray:
    """Walk along a (mostly) 1-pixel-wide skeleton and return ordered (row, col)."""
    pts = np.argwhere(skel)
    if len(pts) == 0:
        return pts
    alive = skel.copy()
    r, c = pts[0]
    order = [(r, c)]
    alive[r, c] = False
    offs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 4-nbrs first
    while True:
        found = False
        for dr, dc in offs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < alive.shape[0] and 0 <= cc < alive.shape[1] and alive[rr, cc]:
                alive[rr, cc] = False
                order.append((rr, cc))
                r, c = rr, cc
                found = True
                break
        if not found:
            break
    return np.asarray(order)


# ---------------------------------------------------------------------------
# rendering helpers shared by the viewer and the HTML export
# ---------------------------------------------------------------------------
def render_background(track: Track, scale: float, road=(158, 158, 158), bg=(96, 96, 96),
                      supersample: int = 2):
    """Anti-aliased RGB PIL image of the road at ``scale`` px per world unit.

    Generated tracks are drawn from their centerline; explicit-wall tracks use
    their background image if they have one, otherwise the wall rings are filled.
    """
    from PIL import Image, ImageDraw

    W = max(1, int(round(track.size[0] * scale)))
    H = max(1, int(round(track.size[1] * scale)))
    if track.centerline is not None:
        m = rasterize_road(track.smoothed_centerline(), track.width, track.size, track.closed,
                           scale, supersample)
        alpha = np.asarray(m).astype(float) / 255.0
        out = np.empty((H, W, 3), dtype=np.uint8)
        for i in range(3):
            out[..., i] = np.round(bg[i] + (road[i] - bg[i]) * alpha).astype(np.uint8)
        return Image.fromarray(out, "RGB")
    if track.image:
        p = Path(track.image)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent.parent / p
        if p.exists():
            return Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
    # explicit walls: chain segments into rings and fill outer minus inner
    s = scale * supersample
    img = Image.new("RGB", (int(W * supersample), int(H * supersample)), bg)
    draw = ImageDraw.Draw(img)
    rings = chain_walls(track.walls)
    rings.sort(key=lambda r: -abs(_ring_area(r)))
    for k, ring in enumerate(rings):
        col = road if k == 0 else bg
        draw.polygon([(x * s, (track.size[1] - y) * s) for x, y in ring], fill=col)
    if supersample > 1:
        img = img.resize((W, H), Image.LANCZOS)
    return img


def chain_walls(walls: np.ndarray, tol: float = 2.0) -> list[np.ndarray]:
    """Join wall segments sharing endpoints into ordered rings/chains."""
    segs = [np.asarray(w, dtype=float) for w in walls]
    used = [False] * len(segs)
    rings = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        chain = [segs[i][:2], segs[i][2:]]
        extended = True
        while extended:
            extended = False
            tail = chain[-1]
            for j in range(len(segs)):
                if used[j]:
                    continue
                a, b = segs[j][:2], segs[j][2:]
                if np.linalg.norm(a - tail) <= tol:
                    chain.append(b); used[j] = True; extended = True; break
                if np.linalg.norm(b - tail) <= tol:
                    chain.append(a); used[j] = True; extended = True; break
        rings.append(np.asarray(chain))
    return rings


def _ring_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) / 2)
