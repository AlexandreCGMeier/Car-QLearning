"""Vectorised 2-D segment geometry (numpy only).

All functions are batched: they take arrays of segments and return arrays of
results. Segments are stored as (..., 4) arrays of ``x1, y1, x2, y2``.
"""
from __future__ import annotations

import numpy as np


def rot90(v: np.ndarray) -> np.ndarray:
    """Rotate 2-D vectors by +90 degrees (counter-clockwise, y-up)."""
    return np.stack([-v[..., 1], v[..., 0]], axis=-1)


def rotate(v: np.ndarray, angle_rad: np.ndarray | float) -> np.ndarray:
    """Rotate 2-D vectors by ``angle_rad`` (counter-clockwise)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y = v[..., 0], v[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def angle_of(v: np.ndarray) -> np.ndarray:
    """Angle of 2-D vectors in degrees, measured counter-clockwise from +x."""
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def signed_angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle from ``a`` to ``b`` in degrees, in (-180, 180]."""
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = (a * b).sum(-1)
    return np.degrees(np.arctan2(cross, dot))


def segments_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean intersection test for broadcastable segment arrays ``a``, ``b``.

    ``a`` and ``b`` have shape (..., 4); the result has the broadcast shape.
    """
    x1, y1, x2, y2 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x3, y3, x4, y4 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den
    hit = (ua >= 0) & (ua <= 1) & (ub >= 0) & (ub <= 1) & (den != 0)
    return hit


def ray_segment_distance(origin: np.ndarray, direction: np.ndarray,
                         segments: np.ndarray, max_dist: float) -> np.ndarray:
    """Distance along each ray to the closest segment (``max_dist`` if none).

    origin:    (..., 2) ray start points
    direction: (..., 2) unit direction vectors (same leading shape as origin)
    segments:  (S, 4)   wall segments
    returns:   (...)    float distances in [0, max_dist]
    """
    ox = origin[..., 0][..., None]
    oy = origin[..., 1][..., None]
    dx = direction[..., 0][..., None]
    dy = direction[..., 1][..., None]
    x3, y3, x4, y4 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]
    ex, ey = x4 - x3, y4 - y3
    den = ey * dx - ex * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ex * (oy - y3) - ey * (ox - x3)) / den   # along the ray
        u = (dx * (oy - y3) - dy * (ox - x3)) / den   # along the segment
    valid = (den != 0) & (t >= 0) & (u >= 0) & (u <= 1)
    t = np.where(valid, t, max_dist)
    return np.minimum(t.min(axis=-1), max_dist)


def rectangle_edges(center: np.ndarray, direction: np.ndarray,
                    length: float, width: float) -> np.ndarray:
    """Edges of oriented rectangles.

    center:    (B, 2), direction: (B, 2) unit vectors along the long axis.
    returns:   (B, 4, 4) the four edge segments of each rectangle.
    """
    right = direction * (length / 2)
    up = rot90(direction) * (width / 2)
    c0 = center + right + up
    c1 = center + right - up
    c2 = center - right - up
    c3 = center - right + up
    corners = np.stack([c0, c1, c2, c3], axis=1)          # (B, 4, 2)
    nxt = np.roll(corners, -1, axis=1)
    return np.concatenate([corners, nxt], axis=-1)        # (B, 4, 4)


def polyline_length(points: np.ndarray, closed: bool) -> np.ndarray:
    """Cumulative arc length at each vertex (and the closing edge if closed)."""
    pts = np.asarray(points, dtype=float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample_polyline(points: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    """Resample a polyline at (approximately) uniform arc-length spacing."""
    pts = np.asarray(points, dtype=float)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    cum = polyline_length(pts, closed=False)
    total = cum[-1]
    if total <= 0:
        return pts[:1].copy()
    n = max(2, int(round(total / spacing)))
    if closed:
        targets = np.linspace(0, total, n, endpoint=False)
    else:
        targets = np.linspace(0, total, n)
    xs = np.interp(targets, cum, pts[:, 0])
    ys = np.interp(targets, cum, pts[:, 1])
    return np.stack([xs, ys], axis=1)


def chaikin(points: np.ndarray, iterations: int, closed: bool) -> np.ndarray:
    """Chaikin corner cutting for a smoother polyline."""
    pts = np.asarray(points, dtype=float)
    for _ in range(iterations):
        if closed:
            p0 = pts
            p1 = np.roll(pts, -1, axis=0)
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            pts = np.empty((len(pts) * 2, 2))
            pts[0::2] = q
            pts[1::2] = r
        else:
            p0, p1 = pts[:-1], pts[1:]
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            mid = np.empty((len(p0) * 2, 2))
            mid[0::2] = q
            mid[1::2] = r
            pts = np.vstack([pts[:1], mid, pts[-1:]])
    return pts


def tangents(points: np.ndarray, closed: bool) -> np.ndarray:
    """Unit tangent at each vertex (central differences)."""
    pts = np.asarray(points, dtype=float)
    if closed:
        t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    else:
        t = np.gradient(pts, axis=0)
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n == 0] = 1
    return t / n


def simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker simplification (iterative, keeps end points)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts.copy()
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[b] - pts[a]
        seg_len = np.linalg.norm(seg)
        rel = pts[a + 1:b] - pts[a]
        if seg_len == 0:
            d = np.linalg.norm(rel, axis=1)
        else:
            d = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / seg_len
        i = int(np.argmax(d))
        if d[i] > tolerance:
            idx = a + 1 + i
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return pts[keep]
