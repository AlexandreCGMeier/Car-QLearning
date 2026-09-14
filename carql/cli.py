"""Command line interface: ``carql <command>`` (or ``python -m carql <command>``).

    carql train   [--config PRESET] [--track NAME] [--run NAME] [--episodes N] [--set k=v ...]
    carql show    [RUN] [--checkpoints SPEC] [--track NAME]
    carql race    RUN [RUN ...] [--checkpoints SPEC] [--track NAME]
    carql build   [TRACK.json] [--size WxH]
    carql track   import PNG | info TRACK | list
    carql runs
    carql export  RUN [RUN ...] [--checkpoints SPEC] [--steps N] --out replay.html
    carql plot    RUN [RUN ...] [--out curves.png]
    carql gui
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CONFIGS_DIR, Config


def _add_set(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="section.key=value",
                   help="override any config value (repeatable)")


def _load_cfg(args) -> Config:
    cfg = Config.load(CONFIGS_DIR / "default.toml")
    preset = getattr(args, "config", None)
    if preset:
        cfg = Config.load(CONFIGS_DIR / "default.toml")
        p = Path(preset)
        if not p.exists():
            p = CONFIGS_DIR / (preset if preset.endswith(".toml") else preset + ".toml")
        import tomllib
        with open(p, "rb") as f:
            from .config import _merge
            _merge(cfg, tomllib.load(f))
    for ov in args.overrides:
        cfg.apply_override(ov)
    if getattr(args, "track", None):
        cfg.env.track = args.track
    if getattr(args, "episodes", None):
        cfg.train.total_episodes = args.episodes
    if getattr(args, "run", None):
        cfg.train.run = args.run
    return cfg


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="carql", description="Deep-Q car racing: train, show, race, build tracks.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train an agent (writes runs/<run>/)")
    p.add_argument("--config", "-c", help="preset in configs/ (dqn, ddqn, dueling_per, ppo) or a .toml path")
    p.add_argument("--track", "-t", help="track name in tracks/ or a .json path")
    p.add_argument("--run", "-r", help="run name (default <track>-<algo>)")
    p.add_argument("--episodes", "-n", type=int, help="total episodes")
    p.add_argument("--resume", action="store_true", help="continue an existing run from latest.pt")
    _add_set(p)

    p = sub.add_parser("show", help="watch checkpoints of a run drive (default: newest run)")
    p.add_argument("run", nargs="?", help="run name or path (default: most recent run)")
    p.add_argument("--checkpoints", "-k", default="all",
                   help="which checkpoints to load: all | latest | best | spread:6 | every:500 | 200,1500,latest")
    p.add_argument("--track", "-t", help="drive on a different track than the one trained on")
    p.add_argument("--start", "-s", help="checkpoint to start on (episode number, 'latest' or 'first')")
    p.add_argument("--speed", type=float, default=1.0, help="simulation steps per frame")
    p.add_argument("--no-rays", action="store_true")

    p = sub.add_parser("race", help="several checkpoints / runs drive at the same time")
    p.add_argument("runs", nargs="+", help="run names; use RUN:SPEC to pick checkpoints per run")
    p.add_argument("--checkpoints", "-k", default="spread:6",
                   help="checkpoint spec applied to runs without their own (default spread:6; use 'best' for a tournament)")
    p.add_argument("--track", "-t")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--no-respawn", action="store_true", help="crashed cars stay dead until everyone is done")

    p = sub.add_parser("build", help="track builder (draw a centerline, gates are generated)")
    p.add_argument("track", nargs="?", help="existing track .json to edit")
    p.add_argument("--size", default="1260x700", help="world size for new tracks, e.g. 1600x900")
    p.add_argument("--width", type=float, default=90.0, help="initial road width")

    p = sub.add_parser("track", help="track utilities")
    ts = p.add_subparsers(dest="track_cmd", required=True)
    q = ts.add_parser("import", help="convert a drawn PNG (bright road on dark) into a track")
    q.add_argument("png")
    q.add_argument("--name")
    q.add_argument("--out", "-o")
    q.add_argument("--height", type=float, default=700.0, help="world height the image is scaled to")
    q.add_argument("--gate-spacing", type=float, default=60.0)
    q.add_argument("--open", action="store_true", help="the stroke is not a closed loop")
    q.add_argument("--invert", action="store_true", help="dark road on bright background")
    q.add_argument("--reverse", action="store_true", help="counter-clockwise driving direction")
    q = ts.add_parser("info", help="print track statistics")
    q.add_argument("track")
    ts.add_parser("list", help="list available tracks")
    q = ts.add_parser("preview", help="render a track to a PNG")
    q.add_argument("track")
    q.add_argument("--out", "-o")

    sub.add_parser("runs", help="list runs and their checkpoints")

    p = sub.add_parser("export", help="simulate headlessly and export a standalone HTML replay")
    p.add_argument("runs", nargs="+", help="run names (RUN:SPEC to pick checkpoints)")
    p.add_argument("--checkpoints", "-k", default="spread:6")
    p.add_argument("--track", "-t")
    p.add_argument("--steps", type=int, default=1800, help="simulation steps to record")
    p.add_argument("--out", "-o", default="replay.html")
    p.add_argument("--no-respawn", action="store_true")

    p = sub.add_parser("plot", help="learning curves for one or more runs")
    p.add_argument("runs", nargs="*", help="run names (default / 'all': every run)")
    p.add_argument("--out", "-o", default=None, help="write PNG instead of opening a window")

    p = sub.add_parser("play", help="drive a car yourself with the arrow keys")
    p.add_argument("--track", "-t", default="classic")

    sub.add_parser("gui", help="desktop launcher: train, show, race, build from one window")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.cmd == "train":
        from .train import train
        cfg = _load_cfg(args)
        train(cfg, args.run, resume=args.resume)
        return 0

    if args.cmd == "show":
        from .viewer.show import show
        show(args.run, checkpoints=args.checkpoints, track=args.track, start=args.start,
             speed=args.speed, rays=not args.no_rays)
        return 0

    if args.cmd == "race":
        from .viewer.show import race
        race(args.runs, checkpoints=args.checkpoints, track=args.track, speed=args.speed,
             respawn=not args.no_respawn)
        return 0

    if args.cmd == "build":
        from .viewer.builder import build
        w, h = (float(x) for x in args.size.lower().split("x"))
        build(args.track, size=(w, h), width=args.width)
        return 0

    if args.cmd == "play":
        from .viewer.show import play
        play(args.track)
        return 0

    if args.cmd == "gui":
        from .gui import main as gui_main
        gui_main()
        return 0

    if args.cmd == "track":
        return _track_cmd(args)

    if args.cmd == "runs":
        from .runs import list_runs
        runs = list_runs()
        if not runs:
            print("no runs yet - train one with: carql train")
        for r in runs:
            cks = r.checkpoints()
            ev = r.evals()
            best = max(ev, key=lambda e: (e["laps"], e["score"])) if ev else None
            cfg = r.config()
            tail = f", best eval {best['score']:.1f} gates / {best['laps']:.2f} laps @ ep {int(best['episode'])}" if best else ""
            print(f"{r.name:28s} {cfg.agent.algo:4s} track={cfg.env.track:12s} {len(cks):3d} checkpoints "
                  f"(ep {cks[0].episode}..{cks[-1].episode}){tail}" if cks else f"{r.name}: no checkpoints")
        return 0

    if args.cmd == "export":
        from .replay import export_runs
        out = export_runs(args.runs, checkpoints=args.checkpoints, track=args.track, steps=args.steps,
                          out=args.out, respawn=not args.no_respawn)
        print(f"wrote {out}")
        return 0

    if args.cmd == "plot":
        from .plot import plot_runs
        plot_runs(args.runs, out=args.out)
        return 0

    ap.print_help()
    return 1


def _track_cmd(args) -> int:
    from .track import TRACKS_DIR, Track, import_png, render_background

    if args.track_cmd == "list":
        for p in sorted(TRACKS_DIR.glob("*.json")):
            print(Track.load(p).describe())
        return 0
    if args.track_cmd == "info":
        t = Track.load(args.track)
        print(t.describe())
        print(f"start {t.start_pos} heading {t.start_angle:.1f} deg, lap ~{t.lap_length():.0f} units")
        return 0
    if args.track_cmd == "import":
        t = import_png(args.png, name=args.name, height=args.height, closed=not args.open,
                       gate_spacing=args.gate_spacing, invert=args.invert, reverse=args.reverse)
        out = Path(args.out) if args.out else TRACKS_DIR / f"{t.name}.json"
        t.save(out)
        print(f"wrote {out}: {t.describe()}")
        return 0
    if args.track_cmd == "preview":
        t = Track.load(args.track)
        img = render_background(t, scale=1.0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        H = t.size[1]
        for x1, y1, x2, y2 in t.walls:
            d.line([(x1, H - y1), (x2, H - y2)], fill=(220, 60, 60), width=2)
        for i, (x1, y1, x2, y2) in enumerate(t.gates):
            d.line([(x1, H - y1), (x2, H - y2)], fill=(60, 220, 60) if i else (255, 255, 0), width=2)
        sx, sy = t.start_pos
        d.ellipse([sx - 5, H - sy - 5, sx + 5, H - sy + 5], fill=(60, 120, 255))
        out = args.out or f"{t.name}_preview.png"
        img.save(out)
        print(f"wrote {out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
