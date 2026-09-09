"""Learning curves from runs/<run>/metrics.csv and eval.csv (matplotlib)."""
from __future__ import annotations

import numpy as np

from .runs import list_runs, resolve_run


def _smooth(y, k: int):
    y = np.asarray(y, dtype=float)
    if len(y) < 2 or k <= 1:
        return y
    k = min(k, len(y))
    kern = np.ones(k) / k
    pad = np.concatenate([np.full(k - 1, y[0]), y])
    return np.convolve(pad, kern, mode="valid")


def plot_runs(names: list[str], out: str | None = None, window: int = 100) -> None:
    try:
        import matplotlib
        if out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is needed for plots:  pip install matplotlib") from e

    runs = [resolve_run(n) for n in names] if names else list_runs()
    if not runs:
        raise SystemExit("no runs to plot")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax_score, ax_eval, ax_len, ax_ret = axes.ravel()
    for r in runs:
        m = r.metrics()
        if not m:
            continue
        cfg = r.config()
        label = f"{r.name} ({cfg.agent.algo}{'/dueling' if cfg.agent.algo == 'dqn' and cfg.agent.dueling else ''})"
        ep = [x["episode"] for x in m]
        ax_score.plot(ep, _smooth([x["score"] for x in m], window), label=label)
        ax_len.plot(ep, _smooth([x["steps"] for x in m], window), label=label)
        ax_ret.plot(ep, _smooth([x["return"] for x in m], window), label=label)
        ev = r.evals()
        if ev:
            ax_eval.plot([x["episode"] for x in ev], [x["score"] for x in ev], marker="o", ms=3, label=label)
    ax_score.set_title(f"training score (gates / episode, {window}-episode mean)")
    ax_eval.set_title("greedy evaluation from the start line (gates)")
    ax_len.set_title("episode length (steps)")
    ax_ret.set_title("episode return")
    for ax in axes.ravel():
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    else:
        plt.show()
