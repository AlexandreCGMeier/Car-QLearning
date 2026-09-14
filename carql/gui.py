"""A small desktop launcher (tkinter, no extra dependencies): ``carql gui``.

Left: configure and start a training run, watch its log and learning curve live.
Right: pick runs and open the viewer (show / race), plot, export, build or play tracks.
Every button shows the exact ``carql`` command it runs in the status bar, so the
GUI doubles as a cheat sheet for the CLI.

The GUI deliberately imports nothing heavy: training and viewers run as separate
``carql`` processes, so the window stays responsive and a crash in a viewer
never takes the launcher down.
"""
from __future__ import annotations

import csv
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
TRACKS = ROOT / "tracks"
CONFIGS = ROOT / "configs"

_LOG_RE = re.compile(r"^ep\s+(\d+)\s*\|.*?\|\s*score\s+([\d.]+).*?(?:\|\s*eval\s+([\d.]+)\s+gates\s+([\d.]+)\s+laps)?\s*$")
MONO = ("Menlo", 11) if sys.platform == "darwin" else ("DejaVu Sans Mono", 10)


# ----------------------------------------------------------------------------- helpers
def carql_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "carql", *args]


def fmt_cmd(cmd: list[str]) -> str:
    return "carql " + " ".join(a if " " not in a else repr(a) for a in cmd[3:])


def list_tracks() -> list[str]:
    return sorted(p.stem for p in TRACKS.glob("*.json"))


def list_presets() -> list[str]:
    return [p.stem for p in sorted(CONFIGS.glob("*.toml")) if p.stem != "default"]


def agent_slug_from_toml(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            a = tomllib.load(f).get("agent", {})
    except Exception:
        return "?"
    if a.get("algo", "dqn") == "ppo":
        return "ppo"
    base = "dueling" if a.get("dueling", True) else ("ddqn" if a.get("double", True) else "dqn")
    return base + ("_per" if a.get("per", True) else "")


def preset_slug(preset: str) -> str:
    """Variant slug for a preset (layered on default.toml), for the default run name."""
    merged: dict = {}
    for p in (CONFIGS / "default.toml", CONFIGS / f"{preset}.toml"):
        if p.exists():
            with open(p, "rb") as f:
                merged.update(tomllib.load(f).get("agent", {}))
    if merged.get("algo") == "ppo":
        return "ppo"
    base = "dueling" if merged.get("dueling") else ("ddqn" if merged.get("double") else "dqn")
    return base + ("_per" if merged.get("per") else "")


def run_rows() -> list[dict]:
    rows = []
    if not RUNS.exists():
        return rows
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        cfg = d / "config.toml"
        if not cfg.exists():
            continue
        cks = sorted(int(m.group(1)) for p in (d / "checkpoints").glob("ep_*.pt")
                     if (m := re.search(r"ep_(\d+)\.pt$", p.name)))
        try:
            with open(cfg, "rb") as f:
                track = tomllib.load(f).get("env", {}).get("track", "?")
        except Exception:
            track = "?"
        best = ""
        ev = d / "eval.csv"
        if ev.exists():
            try:
                with open(ev) as f:
                    evs = [r for r in csv.DictReader(f)]
                if evs:
                    b = max(evs, key=lambda r: (float(r["laps"] or 0), float(r["score"] or 0)))
                    best = f"{float(b['score']):.0f} gates / {float(b['laps']):.2f} laps @ {int(float(b['episode']))}"
            except Exception:
                best = ""
        rows.append({"name": d.name, "algo": agent_slug_from_toml(cfg), "track": Path(track).stem,
                     "ckpts": f"{len(cks)} ({cks[0]}..{cks[-1]})" if cks else "0", "best": best,
                     "n_ckpts": len(cks)})
    return rows


# ----------------------------------------------------------------------------- chart
class Curve:
    """Tiny two-series line chart on a Canvas: training score and eval score vs episode."""

    def __init__(self, parent):
        self.c = tk.Canvas(parent, height=150, highlightthickness=0, bg="#24262b")
        self.c.bind("<Configure>", lambda e: self.draw())
        self.reset()

    def reset(self):
        self.ep: list[int] = []
        self.score: list[float] = []
        self.eval_ep: list[int] = []
        self.eval: list[float] = []
        self.draw()

    def add(self, ep: int, score: float, ev: float | None):
        self.ep.append(ep)
        self.score.append(score)
        if ev is not None and (not self.eval_ep or self.eval_ep[-1] != ep):
            self.eval_ep.append(ep)
            self.eval.append(ev)
        self.draw()

    def draw(self):
        c = self.c
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return
        pad_l, pad_r, pad_t, pad_b = 42, 10, 12, 20
        c.create_text(pad_l, 6, anchor="nw", fill="#9aa3ae", font=(MONO[0], 9),
                      text="gates per episode  ─ training (100-ep mean)   ─ greedy eval from start line")
        if not self.ep:
            c.create_text(w / 2, h / 2, fill="#6b7280", font=(MONO[0], 10), text="no training data yet")
            return
        xmax = max(self.ep[-1], 1)
        ymax = max(max(self.score), max(self.eval) if self.eval else 0, 1)
        ymax = float(int(ymax / 10 + 1) * 10)
        def X(e): return pad_l + (w - pad_l - pad_r) * e / xmax
        def Y(v): return h - pad_b - (h - pad_t - pad_b) * v / ymax
        for k in range(5):
            v = ymax * k / 4
            y = Y(v)
            c.create_line(pad_l, y, w - pad_r, y, fill="#33363d")
            c.create_text(pad_l - 4, y, anchor="e", fill="#9aa3ae", font=(MONO[0], 8), text=f"{v:.0f}")
        for k in range(5):
            e = xmax * k / 4
            c.create_text(X(e), h - pad_b + 3, anchor="n", fill="#9aa3ae", font=(MONO[0], 8), text=f"{e:.0f}")
        if len(self.ep) > 1:
            c.create_line(*[p for e, s in zip(self.ep, self.score) for p in (X(e), Y(s))], fill="#7fb3ff", width=1.5)
        if len(self.eval_ep) > 1:
            c.create_line(*[p for e, s in zip(self.eval_ep, self.eval) for p in (X(e), Y(s))], fill="#8dd9a6", width=2)
        elif self.eval_ep:
            c.create_oval(X(self.eval_ep[0]) - 2, Y(self.eval[0]) - 2, X(self.eval_ep[0]) + 2, Y(self.eval[0]) + 2, fill="#8dd9a6", outline="")


# ----------------------------------------------------------------------------- app
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("carql")
        root.geometry("1180x760")
        root.minsize(980, 620)
        self.proc: subprocess.Popen | None = None
        self.q: queue.Queue = queue.Queue()
        self.viewers: list[subprocess.Popen] = []

        style = ttk.Style()
        try:
            style.theme_use("aqua" if sys.platform == "darwin" else "clam")
        except tk.TclError:
            pass

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=5)
        outer.columnconfigure(1, weight=4)
        outer.rowconfigure(0, weight=1)
        self._build_train(outer)
        self._build_show(outer)
        self.status = ttk.Label(root, text="ready", anchor="w", font=MONO, padding=(10, 4))
        self.status.pack(fill="x", side="bottom")
        self.refresh_runs()
        self.root.after(100, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------- train pane
    def _build_train(self, outer):
        f = ttk.LabelFrame(outer, text="Train", padding=10)
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)
        r = 0
        ttk.Label(f, text="Track").grid(row=r, column=0, sticky="w")
        self.v_track = tk.StringVar(value="classic")
        self.cb_track = ttk.Combobox(f, textvariable=self.v_track, values=list_tracks(), state="readonly", width=18)
        self.cb_track.grid(row=r, column=1, sticky="ew", padx=(4, 12))
        ttk.Label(f, text="Agent preset").grid(row=r, column=2, sticky="w")
        self.v_preset = tk.StringVar(value="dueling_per")
        self.cb_preset = ttk.Combobox(f, textvariable=self.v_preset, values=list_presets(), state="readonly", width=14)
        self.cb_preset.grid(row=r, column=3, sticky="ew", padx=4)
        r += 1
        ttk.Label(f, text="Run name").grid(row=r, column=0, sticky="w", pady=(6, 0))
        self.v_run = tk.StringVar()
        ttk.Entry(f, textvariable=self.v_run).grid(row=r, column=1, sticky="ew", padx=(4, 12), pady=(6, 0))
        ttk.Label(f, text="Episodes").grid(row=r, column=2, sticky="w", pady=(6, 0))
        self.v_eps = tk.IntVar(value=3000)
        ttk.Spinbox(f, from_=50, to=100000, increment=250, textvariable=self.v_eps, width=8).grid(row=r, column=3, sticky="w", padx=4, pady=(6, 0))
        r += 1
        ttk.Label(f, text="Overrides").grid(row=r, column=0, sticky="w", pady=(6, 0))
        self.v_set = tk.StringVar()
        e = ttk.Entry(f, textvariable=self.v_set)
        e.grid(row=r, column=1, columnspan=3, sticky="ew", padx=4, pady=(6, 0))
        r += 1
        ttk.Label(f, text="e.g.  train.seed=1 env.reward_crash=-80 agent.hidden=[128,128]   (space separated)",
                  foreground="#7a8290").grid(row=r, column=1, columnspan=3, sticky="w", padx=4)
        r += 1
        row = ttk.Frame(f)
        row.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(10, 6))
        self.v_resume = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="resume existing run", variable=self.v_resume).pack(side="left")
        self.b_stop = ttk.Button(row, text="Stop (saves latest.pt)", command=self.stop_training, state="disabled")
        self.b_stop.pack(side="right")
        self.b_start = ttk.Button(row, text="Start training", command=self.start_training)
        self.b_start.pack(side="right", padx=(0, 6))
        r += 1
        self.curve = Curve(f)
        self.curve.c.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(4, 6))
        r += 1
        self.log = ScrolledText(f, height=14, font=MONO, wrap="none", bg="#1a1c20", fg="#e9ecef",
                                insertbackground="#e9ecef", relief="flat")
        self.log.grid(row=r, column=0, columnspan=4, sticky="nsew")
        f.rowconfigure(r, weight=1)
        self.v_track.trace_add("write", lambda *_: self._suggest_run())
        self.v_preset.trace_add("write", lambda *_: self._suggest_run())
        self._suggest_run()

    def _suggest_run(self):
        self.v_run.set(f"{self.v_track.get()}-{preset_slug(self.v_preset.get())}")

    def start_training(self):
        if self.proc is not None:
            messagebox.showinfo("carql", "A training run is already going. Stop it first.")
            return
        cmd = carql_cmd("train", "-c", self.v_preset.get(), "-t", self.v_track.get(),
                        "-r", self.v_run.get().strip() or f"{self.v_track.get()}-{preset_slug(self.v_preset.get())}",
                        "-n", str(self.v_eps.get()))
        if self.v_resume.get():
            cmd.append("--resume")
        for ov in self.v_set.get().split():
            cmd += ["--set", ov]
        run_dir = RUNS / cmd[cmd.index("-r") + 1]
        if (run_dir / "checkpoints").exists() and any((run_dir / "checkpoints").iterdir()) and not self.v_resume.get():
            messagebox.showwarning("carql", f"Run '{run_dir.name}' already has checkpoints.\n"
                                            f"Tick 'resume' to continue it, or choose another run name.")
            return
        self.log.delete("1.0", "end")
        self.curve.reset()
        self._say(fmt_cmd(cmd))
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, env=env, bufsize=1)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.b_start.config(state="disabled")
        self.b_stop.config(state="normal")

    def stop_training(self):
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)     # trainer catches KeyboardInterrupt and saves latest.pt
        except Exception:
            self.proc.terminate()
        self._say("stopping - waiting for the trainer to save latest.pt ...")

    def _reader(self, proc: subprocess.Popen):
        for line in proc.stdout:
            self.q.put(line.rstrip("\n"))
        proc.wait()
        self.q.put(None)

    def _pump(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item is None:
                    code = self.proc.returncode if self.proc else 0
                    self.proc = None
                    self.b_start.config(state="normal")
                    self.b_stop.config(state="disabled")
                    self._say(f"training finished (exit code {code})")
                    self.refresh_runs()
                    continue
                self.log.insert("end", item + "\n")
                self.log.see("end")
                m = _LOG_RE.match(item)
                if m:
                    ev = float(m.group(3)) if m.group(3) else None
                    self.curve.add(int(m.group(1)), float(m.group(2)), ev)
        except queue.Empty:
            pass
        self.root.after(150, self._pump)

    # -------------------------------------------------------------- show pane
    def _build_show(self, outer):
        f = ttk.LabelFrame(outer, text="Show, race, tracks", padding=10)
        f.grid(row=0, column=1, sticky="nsew")
        f.columnconfigure(0, weight=1)
        f.rowconfigure(1, weight=1)

        top = ttk.Frame(f)
        top.grid(row=0, column=0, sticky="ew")
        ttk.Label(top, text="Runs  (select one for show, several for race)").pack(side="left")
        ttk.Button(top, text="Refresh", command=self.refresh_runs).pack(side="right")

        cols = ("run", "algo", "track", "ckpts", "best")
        self.tree = ttk.Treeview(f, columns=cols, show="headings", selectmode="extended", height=9)
        for c, w in zip(cols, (150, 80, 70, 88, 170)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w", stretch=(c in ("run", "best")))
        self.tree.grid(row=1, column=0, sticky="nsew", pady=(4, 8))
        self.tree.bind("<Double-1>", lambda e: self.show())

        opts = ttk.Frame(f)
        opts.grid(row=2, column=0, sticky="ew")
        ttk.Label(opts, text="checkpoints").grid(row=0, column=0, sticky="w")
        self.v_spec = tk.StringVar(value="all")
        ttk.Combobox(opts, textvariable=self.v_spec, values=["all", "every:250", "every:500", "spread:6", "spread:10", "best", "latest"],
                     width=10).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(opts, text="start on").grid(row=0, column=2, sticky="w")
        self.v_start = tk.StringVar(value="first")
        ttk.Combobox(opts, textvariable=self.v_start, values=["first", "latest"], state="readonly", width=7).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(opts, text="speed").grid(row=0, column=4, sticky="w")
        self.v_speed = tk.StringVar(value="1")
        ttk.Combobox(opts, textvariable=self.v_speed, values=["0.5", "1", "2", "4"], width=4).grid(row=0, column=5, padx=4)
        ttk.Label(opts, text="on track").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.v_show_track = tk.StringVar(value="(run's own)")
        self.cb_show_track = ttk.Combobox(opts, textvariable=self.v_show_track, values=["(run's own)"] + list_tracks(), state="readonly", width=14)
        self.cb_show_track.grid(row=1, column=1, columnspan=3, sticky="w", padx=4, pady=(6, 0))

        btns = ttk.Frame(f)
        btns.grid(row=3, column=0, sticky="ew", pady=(10, 4))
        ttk.Button(btns, text="Show", command=self.show).pack(side="left")
        ttk.Button(btns, text="Race selected (spread:6)", command=self.race).pack(side="left", padx=6)
        ttk.Button(btns, text="Tournament (best of each)", command=lambda: self.race("best")).pack(side="left")
        btns2 = ttk.Frame(f)
        btns2.grid(row=4, column=0, sticky="ew", pady=(0, 4))
        ttk.Button(btns2, text="Plot curves", command=self.plot).pack(side="left")
        ttk.Button(btns2, text="Export HTML replay", command=self.export).pack(side="left", padx=6)

        ttk.Separator(f).grid(row=5, column=0, sticky="ew", pady=10)
        tr = ttk.Frame(f)
        tr.grid(row=6, column=0, sticky="ew")
        ttk.Label(tr, text="Tracks").pack(side="left")
        self.v_track2 = tk.StringVar(value="classic")
        self.cb_track2 = ttk.Combobox(tr, textvariable=self.v_track2, values=list_tracks(), state="readonly", width=16)
        self.cb_track2.pack(side="left", padx=6)
        ttk.Button(tr, text="Play", command=self.play).pack(side="left")
        ttk.Button(tr, text="Edit", command=self.edit_track).pack(side="left", padx=6)
        ttk.Button(tr, text="New track", command=self.new_track).pack(side="left")
        ttk.Label(f, text="Play = drive it with the arrow keys, Edit = open it in the builder. Viewer windows are separate processes; press H inside one for its keys, Esc to close it.",
                  foreground="#7a8290", wraplength=420, justify="left").grid(row=7, column=0, sticky="w", pady=(10, 0))

    def refresh_runs(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for r in run_rows():
            self.tree.insert("", "end", iid=r["name"], values=(r["name"], r["algo"], r["track"], r["ckpts"], r["best"]))
        tracks = list_tracks()
        self.cb_track["values"] = tracks
        self.cb_track2["values"] = tracks
        self.cb_show_track["values"] = ["(run's own)"] + tracks

    def _selected(self, need: int = 1) -> list[str]:
        sel = list(self.tree.selection())
        if len(sel) < need:
            messagebox.showinfo("carql", "Select a run in the list first." if need == 1 else "Select at least one run.")
        return sel

    def _track_arg(self) -> list[str]:
        t = self.v_show_track.get()
        return [] if t.startswith("(") else ["-t", t]

    def _spawn(self, cmd: list[str]):
        self._say(fmt_cmd(cmd))
        p = subprocess.Popen(cmd, cwd=ROOT)
        self.viewers.append(p)

    def show(self):
        sel = self._selected()
        if not sel:
            return
        self._spawn(carql_cmd("show", sel[0], "-k", self.v_spec.get(), "-s", self.v_start.get(),
                              "--speed", self.v_speed.get(), *self._track_arg()))

    def race(self, spec: str = "spread:6"):
        sel = self._selected()
        if not sel:
            return
        self._spawn(carql_cmd("race", *sel, "-k", spec, "--speed", self.v_speed.get(), *self._track_arg()))

    def plot(self):
        sel = list(self.tree.selection())
        self._spawn(carql_cmd("plot", *sel))

    def export(self):
        sel = self._selected()
        if not sel:
            return
        out = ROOT / "replays" / f"{'-'.join(sel)[:60]}-{time.strftime('%Y%m%d-%H%M%S')}.html"
        out.parent.mkdir(exist_ok=True)
        cmd = carql_cmd("export", *sel, "-k", "spread:6" if self.v_spec.get() == "all" else self.v_spec.get(),
                        "-o", str(out), *self._track_arg())
        self._say(fmt_cmd(cmd))
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            self._say(f"wrote {out}")
            messagebox.showinfo("carql", f"Replay written:\n{out}")
        else:
            messagebox.showerror("carql", r.stderr[-1500:] or "export failed")

    def play(self):
        self._spawn(carql_cmd("play", "-t", self.v_track2.get()))

    def edit_track(self):
        self._spawn(carql_cmd("build", self.v_track2.get()))

    def new_track(self):
        self._spawn(carql_cmd("build"))

    # --------------------------------------------------------------- misc
    def _say(self, text: str):
        self.status.config(text=text)

    def on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("carql", "Training is running. Stop it (latest.pt is saved) and quit?"):
                return
            self.stop_training()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
