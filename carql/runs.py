"""Run folders: ``runs/<name>/`` with config, track copy, checkpoints and metrics."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT, Config
from .track import Track

RUNS_DIR = ROOT / "runs"
_CKPT_RE = re.compile(r"ep_(\d+)\.pt$")


@dataclass(order=True)
class Checkpoint:
    episode: int
    path: Path

    @property
    def name(self) -> str:
        return f"ep {self.episode}"


class Run:
    def __init__(self, path: str | Path):
        p = Path(path)
        if not p.exists():
            cand = RUNS_DIR / p.name
            if cand.exists():
                p = cand
        self.path = p
        self.name = p.name

    def __repr__(self) -> str:
        return f"Run({self.name})"

    @property
    def exists(self) -> bool:
        return self.path.exists()

    # --------------------------------------------------------- contents
    @property
    def ckpt_dir(self) -> Path:
        return self.path / "checkpoints"

    def checkpoints(self) -> list[Checkpoint]:
        out = []
        if self.ckpt_dir.exists():
            for f in self.ckpt_dir.iterdir():
                m = _CKPT_RE.search(f.name)
                if m:
                    out.append(Checkpoint(int(m.group(1)), f))
        return sorted(out)

    def latest(self) -> Path | None:
        p = self.path / "latest.pt"
        if p.exists():
            return p
        cks = self.checkpoints()
        return cks[-1].path if cks else None

    def best(self) -> Path | None:
        p = self.path / "best.pt"
        return p if p.exists() else self.latest()

    def config(self) -> Config:
        return Config.load(self.path / "config.toml")

    def track(self) -> Track:
        p = self.path / "track.json"
        if p.exists():
            return Track.load(p)
        return Track.load(self.config().env.track)

    def metrics(self) -> list[dict]:
        p = self.path / "metrics.csv"
        if not p.exists():
            return []
        with open(p) as f:
            return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]

    def evals(self) -> list[dict]:
        p = self.path / "eval.csv"
        if not p.exists():
            return []
        with open(p) as f:
            return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]

    # -------------------------------------------------------- selection
    def select(self, spec: str | None) -> list[Path]:
        """Resolve a checkpoint spec to paths.

        spec examples: ``latest`` (default), ``best``, ``all``, ``200``,
        ``200,1500,latest``, ``spread:6`` (six evenly spaced over training),
        ``every:1000``.
        """
        cks = self.checkpoints()
        if spec is None or spec == "" or spec == "latest":
            p = self.latest()
            return [p] if p else []
        if spec == "best":
            p = self.best()
            return [p] if p else []
        if spec == "all":
            return [c.path for c in cks]
        if spec.startswith("spread:"):
            k = int(spec.split(":")[1])
            if not cks:
                return []
            if k >= len(cks):
                return [c.path for c in cks]
            import numpy as np
            idx = np.unique(np.round(np.linspace(0, len(cks) - 1, k)).astype(int))
            return [cks[i].path for i in idx]
        if spec.startswith("every:"):
            step = int(spec.split(":")[1])
            return [c.path for c in cks if c.episode % step == 0] or [c.path for c in cks]
        out = []
        for part in spec.split(","):
            part = part.strip()
            if part in ("latest", "best"):
                out += self.select(part)
                continue
            ep = int(part)
            if not cks:
                raise FileNotFoundError(f"{self.name}: no checkpoints")
            nearest = min(cks, key=lambda c: abs(c.episode - ep))
            out.append(nearest.path)
        return out


def _num(v: str) -> float:
    """CSV cell -> float; empty cells (e.g. loss before learning starts) become NaN."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def list_runs(runs_dir: Path | None = None) -> list[Run]:
    runs_dir = runs_dir or RUNS_DIR
    if not runs_dir.exists():
        return []
    return sorted((Run(p) for p in runs_dir.iterdir() if (p / "config.toml").exists()),
                  key=lambda r: r.path.stat().st_mtime)


def resolve_run(name: str) -> Run:
    r = Run(name)
    if not r.exists:
        runs = list_runs()
        names = ", ".join(x.name for x in runs) or "(none)"
        raise FileNotFoundError(f"run {name!r} not found. Available runs: {names}")
    return r


def latest_run() -> Run | None:
    runs = list_runs()
    return runs[-1] if runs else None
