"""Standalone HTML replays (canvas, anti-aliased, no dependencies)."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np

from .runs import resolve_run
from .sim import Driver, RaceSim, drivers_from_specs
from .track import Track, render_background


def _bg_data_uri(track: Track, scale: float = 1.0) -> str:
    img = render_background(track, scale, road=(150, 150, 152), bg=(72, 72, 76))
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(track: Track, drivers: list[Driver], frames: list[dict], path: str | Path,
               title: str = "carql replay") -> Path:
    n = len(drivers)
    T = len(frames)
    x = np.array([f["x"] for f in frames]).round(1).tolist()
    y = np.array([f["y"] for f in frames]).round(1).tolist()
    h = np.array([f["h"] for f in frames]).round(1).tolist()
    alive = np.array([f["alive"] for f in frames]).astype(int).tolist()
    score = np.array([f["score"] for f in frames]).astype(int).tolist()
    laps = np.array([f["laps"] for f in frames]).astype(int).tolist()
    data = {
        "title": title,
        "size": list(track.size),
        "car": [25.0, 15.0],
        "walls": np.round(track.walls, 1).tolist(),
        "gates": np.round(track.gates, 1).tolist(),
        "drivers": [{"label": d.label, "color": list(d.color), "algo": d.algo_label,
                     "episode": d.episode} for d in drivers],
        "T": T, "n": n, "x": x, "y": y, "h": h, "alive": alive, "score": score, "laps": laps,
        "bg": _bg_data_uri(track),
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__TITLE__", title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
    return path


def export_runs(run_specs: list[str], checkpoints: str = "spread:6", track: str | None = None,
                steps: int = 1800, out: str = "replay.html", respawn: bool = True) -> Path:
    drivers = drivers_from_specs(run_specs, checkpoints)
    first = resolve_run(run_specs[0].partition(":")[0])
    trk = Track.load(track) if track else first.track()
    sim = RaceSim(trk, drivers, respawn=respawn)
    frames = []
    for _ in range(steps):
        sim.step()
        frames.append(sim.snapshot())
    names = ", ".join(sorted({d.run for d in drivers if d.run}))
    return write_html(trk, drivers, frames, out, title=f"{names} on {trk.name}")


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{margin:0;background:#1d1d21;color:#eee;font:13px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}
 #wrap{display:flex;gap:12px;padding:12px;box-sizing:border-box;height:100vh}
 #left{flex:1;display:flex;flex-direction:column;min-width:0}
 canvas{width:100%;height:auto;background:#28282c;border-radius:8px;flex:1;min-height:0;object-fit:contain}
 #bar{display:flex;align-items:center;gap:10px;padding:8px 4px}
 button{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:5px 12px;cursor:pointer}
 button:hover{background:#444} input[type=range]{flex:1}
 #board{width:300px;background:#141418;border-radius:8px;padding:10px;overflow:auto}
 #board h3{margin:0 0 8px;font-weight:600;font-size:14px;color:#ffd866}
 table{width:100%;border-collapse:collapse} td{padding:3px 4px;white-space:nowrap} td.n{text-align:right}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:middle}
 label{color:#aaa} .dim{color:#888}
</style></head><body>
<div id="wrap">
 <div id="left">
  <canvas id="c"></canvas>
  <div id="bar">
   <button id="play">Pause</button>
   <button id="restart">⟲</button>
   <label>speed <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
   <input type="range" id="scrub" min="0" value="0">
   <span id="time" class="dim"></span>
   <label><input type="checkbox" id="walls"> walls</label>
   <label><input type="checkbox" id="gates" checked> gates</label>
   <label><input type="checkbox" id="trails" checked> trails</label>
  </div>
 </div>
 <div id="board"><h3>__TITLE__</h3><table id="tbl"></table><p class="dim" id="note"></p></div>
</div>
<script>
const D = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const [W,H] = D.size; const SS = 2;           // supersample canvas for crisp lines
cv.width = W*SS; cv.height = H*SS;
const bg = new Image(); bg.src = D.bg;
let t = 0, playing = true, speed = 1, acc = 0;
const scrub = document.getElementById('scrub'); scrub.max = D.T-1;
const TRAIL = 90;
function w2c(x,y){ return [x*SS, (H-y)*SS]; }
function drawCar(i){
  const [L,Wd] = D.car; const x=D.x[t][i], y=D.y[t][i], h=D.h[t][i]*Math.PI/180;
  const alive = D.alive[t][i]; const col = D.drivers[i].color;
  ctx.save(); const [cx,cy]=w2c(x,y); ctx.translate(cx,cy); ctx.rotate(-h); ctx.scale(SS,SS);
  ctx.globalAlpha = alive?1:0.3;
  ctx.fillStyle='rgba(0,0,0,0.35)'; ctx.fillRect(-L/2+2,-Wd/2+2,L,Wd);
  ctx.fillStyle=`rgb(${col})`; ctx.fillRect(-L/2,-Wd/2,L,Wd);
  ctx.fillStyle='rgba(255,255,255,0.45)'; ctx.fillRect(L*0.2,-Wd*0.35,L*0.3,Wd*0.7);
  ctx.restore();
  ctx.save(); ctx.globalAlpha = alive?1:0.4; ctx.font = `${11*SS}px sans-serif`; ctx.textAlign='center';
  ctx.fillStyle=`rgb(${col})`; ctx.fillText(D.drivers[i].label, cx, cy-14*SS); ctx.restore();
}
function draw(){
  ctx.clearRect(0,0,cv.width,cv.height);
  if (bg.complete) ctx.drawImage(bg,0,0,cv.width,cv.height);
  ctx.lineWidth = 1.5*SS;
  if (document.getElementById('gates').checked){ ctx.strokeStyle='rgba(90,200,110,0.8)'; ctx.beginPath();
    for (const g of D.gates){ const a=w2c(g[0],g[1]), b=w2c(g[2],g[3]); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); } ctx.stroke(); }
  if (document.getElementById('walls').checked){ ctx.strokeStyle='rgba(235,80,80,0.9)'; ctx.lineWidth=2*SS; ctx.beginPath();
    for (const g of D.walls){ const a=w2c(g[0],g[1]), b=w2c(g[2],g[3]); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); } ctx.stroke(); }
  if (document.getElementById('trails').checked){
    for (let i=0;i<D.n;i++){ const col=D.drivers[i].color; ctx.lineWidth=2*SS; ctx.lineCap='round';
      for (let k=Math.max(1,t-TRAIL); k<=t; k++){ if(!D.alive[k][i]||!D.alive[k-1][i]) continue;
        const dx=D.x[k][i]-D.x[k-1][i], dy=D.y[k][i]-D.y[k-1][i]; if (dx*dx+dy*dy>400) continue;
        ctx.strokeStyle=`rgba(${col},${(k-(t-TRAIL))/TRAIL*0.6})`; ctx.beginPath();
        const a=w2c(D.x[k-1][i],D.y[k-1][i]), b=w2c(D.x[k][i],D.y[k][i]); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke(); } }
  }
  for (let i=0;i<D.n;i++) drawCar(i);
  // leaderboard
  const order=[...Array(D.n).keys()].sort((a,b)=> (D.laps[t][b]*1000+D.score[t][b])-(D.laps[t][a]*1000+D.score[t][a]));
  let rows='<tr class="dim"><td>#</td><td>driver</td><td class="n">laps</td><td class="n">gates</td></tr>';
  order.forEach((i,r)=>{ const d=D.drivers[i]; rows+=`<tr style="color:rgb(${d.color});opacity:${D.alive[t][i]?1:0.5}"><td>${r+1}</td><td><span class="sw" style="background:rgb(${d.color})"></span>${d.label}<div class="dim" style="font-size:11px">${d.algo}</div></td><td class="n">${D.laps[t][i]}</td><td class="n">${D.score[t][i]}</td></tr>`; });
  document.getElementById('tbl').innerHTML=rows;
  document.getElementById('time').textContent=`${(t/60).toFixed(1)} s / ${(D.T/60).toFixed(1)} s`;
  scrub.value=t;
}
function loop(){ if (playing){ acc+=speed; while(acc>=1){ acc-=1; t=(t+1)%D.T; } } draw(); requestAnimationFrame(loop); }
document.getElementById('play').onclick=e=>{ playing=!playing; e.target.textContent=playing?'Pause':'Play'; };
document.getElementById('restart').onclick=()=>{ t=0; };
document.getElementById('speed').onchange=e=>{ speed=parseFloat(e.target.value); };
scrub.oninput=e=>{ t=parseInt(e.target.value); };
document.getElementById('note').textContent=`${D.n} car(s), ${D.T} frames at 60 steps/s. Cars respawn on the start line after a crash.`;
bg.onload=draw; loop();
</script></body></html>
"""
