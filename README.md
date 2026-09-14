# Car-QLearning

A car learns to drive around a track from 16 distance sensors, a speedometer and the bearing to the
next reward gates — first badly, then like a Tokyo drifter. This is the 2026 rewrite of the 2024
pyglet-1.5 / TensorFlow-1 project: same idea, same physics, modern stack.

* **Physics / environment** — pure numpy, *batched*: 64 cars simulate in parallel in one call
  (~20k car-steps/s on a laptop core). Gym-like `reset / step` API.
* **Agents** — PyTorch. DQN, Double DQN, Dueling DDQN, prioritised replay, n-step returns; and PPO.
* **Viewer** — pyglet 2 with multisampling (no aliasing), a checkpoint *timeline* so you can walk
  through training history live, a race mode with a leaderboard, Q-value / ray inspection.
* **Tracks** — JSON. Draw one with the mouse (`carql build`), import a hand-drawn PNG
  (`carql track import`), gates and walls are generated automatically.
* **Replays** — export any session as a self-contained HTML file.

Full command-line reference with every flag: [`docs/cli.html`](docs/cli.html) (open it in a browser).

## Install

Python 3.11+ (3.13 works). From the repo root:

```bash
uv pip install -e ".[plot]"              # or: pip install -e ".[plot]"
uv venv && source .venv/bin/activate     # or: python -m venv .venv
cd /Users/Alex/Documents/GitHub/Car-QLearning && source .venv/bin/activate

uv pip install -e ".[plot]"              # or: pip install -e ".[plot]"
carql --help
```

PyTorch and pyglet install from PyPI on macOS (Apple silicon), Linux and Windows.
Everything runs on the CPU; the networks are tiny, a GPU would only slow things down.

## The switch: train or show

```bash
carql train                       # trains Dueling DDQN+PER on tracks/classic.json -> runs/classic-dueling_per/
carql show                        # opens the most recent run in the viewer
carql gui                         # or do both from a small desktop launcher (tkinter)
```

That's it. `train` writes a self-contained run folder; `show` reads one:

```
runs/classic-dueling_per/
  config.toml        the exact configuration used (car physics, rewards, agent, schedule)
  track.json         a copy of the track
  checkpoints/       ep_000100.pt, ep_000200.pt, ...   (every train.checkpoint_every episodes)
  latest.pt          most recent weights          best.pt   best greedy evaluation so far
  metrics.csv        one row per training episode (return, gates, steps, epsilon, loss)
  eval.csv           greedy evaluation from the start line every train.eval_every episodes
  log.txt
```

Every checkpoint carries its full config, so `show` always rebuilds exactly the physics the agent
was trained with — and any checkpoint from any run can be dropped onto any track.

### Training options

```bash
carql train -c ppo                              # preset from configs/ (dqn, ddqn, dueling_per, ppo)
carql train -t track_1 -r mytrack-dqn -n 8000   # other track, run name (default <track>-<variant>), episode budget
carql train --set agent.hidden=[128,128] --set env.reward_crash=-80 --set train.n_envs=32
carql train -r classic-dqn --resume            # continue from latest.pt (uses the run's stored config)
carql runs                                     # list runs, checkpoint ranges and best evals
carql plot classic-dqn classic-ppo -o curves.png
```

All knobs are in `configs/default.toml` with comments; presets only change the `[agent]` section.
`--set section.key=value` overrides anything. Training prints a line every `train.log_every`
episodes and evaluates the greedy policy from the start line every `train.eval_every` episodes
(`eval N gates / M laps`) — that number is what the viewer's timeline shows.

Rough budget on a laptop: the dueling DDQN completes its first clean lap after 1–2k episodes
(a few minutes); PPO after ~1k episodes. The default 3000 episodes is plenty for both (15–30 min).
Cars spawn at random gates while training (`env.random_start`) so the whole track is seen early;
evaluation and the viewer always start on the start line.

## Show mode — the timeline

```bash
carql show classic-dqn                 # all checkpoints, starts on the newest
carql show classic-dqn --start first   # start with the very first, terrible one
carql show classic-dqn -k every:250    # only every 250th checkpoint
carql show classic-dqn -t track_1      # drive the classic policy on another track
```

The bottom bar is the training timeline: one dot per checkpoint, bar height = greedy evaluation
score. `←`/`→` step through it, `1`–`9` jump to 10 %–90 %, `Home`/`End`, or click. The car resets on
every switch, so the demo flow is: start on `Home`, let people laugh, walk right.

Keys: `Space` pause, `.` single step, `+`/`-` sim speed, `V` sensor rays, `Q` Q-value bars,
`G` gates, `W` walls, `I` info, `M` take the wheel with the arrow keys, `E` export the last 60 s as
an HTML replay to `replays/`, `H` help.

## Race mode

```bash
carql race classic-dqn                       # 6 checkpoints spread over training race each other
carql race classic-dqn -k 200,1000,3000,latest
carql race classic-dqn classic-ppo -k best   # tournament: best checkpoint of each run
carql race classic-dqn:spread:4 classic-ppo:best classic-ddqn:latest   # per-run specs
carql race classic-dqn --no-respawn          # crashed cars stay dead until everyone is done
```

Cars are ghosts to each other. The leaderboard sorts by lap progress and counts crashes;
`Tab` moves the inspection focus (rays, Q-values, next gate) between cars. `E` exports a replay.

### Architecture tournament

```bash
for c in dqn ddqn dueling_per ppo; do carql train -c $c -r classic-$c -n 4000; done
carql plot classic-dqn classic-ddqn classic-dueling_per classic-ppo -o tournament.png
carql race classic-dqn classic-ddqn classic-dueling_per classic-ppo -k best
```

## Tracks

```bash
carql track list
carql build                            # new track: click points, scroll = width, S = save
carql build track_1                    # edit a generated track
carql track import my_drawing.png      # bright road on dark background -> tracks/my_drawing.json
carql track preview classic -o classic.png
carql play -t track_1                  # drive it yourself
```

A generated track is a centerline (control points) plus a width. Walls are the contour of the
rasterised stroke, gates are cross-sections every `gate_spacing` units, the start is control point 0
(`1` in the builder moves it, `D` reverses direction). `tracks/classic.json` is the original
hand-coded track (walls and gates converted from `Game.py`); `tracks/track_1.json` was imported
from the original `track_1.png`.

World coordinates are arbitrary units (the classic track is 1260×700, the car is 25×15); the viewer
scales any track to the window.

## Replays

```bash
carql export classic-dqn -k spread:6 --steps 1800 -o replays/timeline.html
```

Simulates headlessly and writes one HTML file (canvas, anti-aliased, scrubber, leaderboard, trails)
you can send to anyone. `E` in the viewer does the same for the last minute you just watched.

## Layout

```
carql/
  env.py        batched car physics, sensors, rewards          geometry.py  vectorised segment maths
  track.py      track model, generation from centerline, PNG import, background rendering
  agents/       nets.py, replay.py (uniform + prioritised), dqn.py, ppo.py, checkpoint io
  train.py      training loops, run folders, evaluation        runs.py      run/checkpoint discovery
  sim.py        headless race simulation (viewer + export)     replay.py    HTML export
  viewer/       base.py (window, camera, track/car graphics, HUD), show.py, builder.py
  cli.py        carql train | show | race | build | track | runs | export | plot | play | gui
  gui.py        tkinter launcher: train with live log/curve, show, race, build, play
configs/        default.toml + presets        tracks/   *.json        tests/   pytest
```

Tests: `pytest` (viewer tests use `xvfb-run` on Linux and skip without a display).

## What changed from the 2024 version

* TF1 graph code → PyTorch; pyglet 1.5 + pygame vectors → pyglet 2 + numpy; Python 3.9 → 3.11+.
* One environment steps 64 cars at once instead of one; training that took hours takes minutes.
* Rewards: per-gate bonus, small step cost, dense progress term, big crash penalty
  (`env.reward_crash`) — the dense term makes a fast-and-crashy policy look attractive unless
  crashing is expensive, which was the single most important tuning knob.
* Checkpoints are saved on a fixed episode schedule with their config and evaluation score, so the
  early disasters are there for the demo, not just the final model.
* Tracks are data, not code: no more typing 57 gates in screen pixels.
