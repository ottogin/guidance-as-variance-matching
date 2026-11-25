#!/usr/bin/env python
"""App. C "Guidance reduces diversity": final samples of M(2) under two draws of a correlated error field.

    python reproduce/figures/fig_diversity_pair.py --simulate --data DIR/samples.npz --out diversity-field-pair.pdf

--simulate runs the sampler first (needs node): `node reproduce/toys/corr_flow_sim.js SEED 3 1 512 5000 OUT.bin` for
seeds 319 and 193 (l_x = 3, l_t = 1, 512 steps, 5,000 samples), saved with the mode centres in --data.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from reproduce.style import INK, GRID, COND, W_COL, rc  # noqa: E402
from vmg.runs import root  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(root() / "_toys" / "diversity_pair" / "samples.npz"))
ap.add_argument("--out", required=True)
ap.add_argument("--simulate", action="store_true", help="run corr_flow_sim.js for both seeds into --data first")
a = ap.parse_args()
SEEDS, LX, LT, STEPS, N = (319, 193), 3, 1, 512, 5000
if a.simulate:
    d = Path(a.data).parent
    d.mkdir(parents=True, exist_ok=True)
    js = Path(__file__).resolve().parents[1] / "toys" / "corr_flow_sim.js"
    out = {}
    for seed in SEEDS:
        b = d / f"seed{seed}.bin"
        subprocess.run(["node", str(js), str(seed), str(LX), str(LT), str(STEPS), str(N), str(b)], check=True)
        out[f"seed_{seed}"] = np.fromfile(b, dtype="<f4").reshape(-1, 2)
    th = 2 * np.pi * np.arange(10) / 10
    np.savez(a.data, mode_centers=np.stack([2 * np.cos(th), np.sin(th)], 1), **out)
    print("wrote", a.data)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
z = np.load(a.data)
plt = rc()
fig, axes = plt.subplots(1, 2, figsize=(W_COL, 1.33), sharex=True, sharey=True, constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02)
th = np.linspace(0, 2 * np.pi, 300)
for i, (ax, key) in enumerate(zip(axes, ("seed_319", "seed_193"))):
    ax.plot(2 * np.cos(th), np.sin(th), color=GRID, lw=0.6, zorder=0)
    ax.scatter(*z[key].T, s=0.5, color=COND, alpha=0.45, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(*z["mode_centers"].T, s=9, facecolors="none", edgecolors=INK, linewidths=0.5, zorder=3)
    ax.set(xlim=(-3.2, 3.2), ylim=(-2.3, 2.3), xticks=[], yticks=[], aspect="equal")
    ax.set_title(f"Field {i + 1}", pad=2)
    for s in ax.spines.values():
        s.set_color(GRID)
fig.savefig(a.out, dpi=400)
