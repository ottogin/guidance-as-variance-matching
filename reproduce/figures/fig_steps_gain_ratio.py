#!/usr/bin/env python
"""Guidance gain against sampler steps, as a difference and as a percent reduction.

    python reproduce/figures/fig_steps_gain_ratio.py --out results/steps-gain-ratio

SD 3.5 Medium on COCO, the w x steps sweep in results/gsweep/metrics (10,000 images per cell, Euler ODE, fp32
latent); the data of fig:steps-gain. At each step count N the best w
is picked from the grid every N shares, w in {1, 2, 3, 4, 4.5, 5, 6, 7, 8}, so the finer grids run at 20 and 40
steps cannot help those step counts. 2 steps is left out: diffusers' 2-step grid is one jump from noise.

Rows are FID and FD_DINOv2. Left column: a - b, with a = metric(w = 1) and b = metric(best w). Right column:
100 (a - b) / a, the percent reduction. Higher is a larger gain in both. Writes <out>.{pdf,png}.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reproduce.style import ACC, COND, GRID, INK, MUTED, W_COL, rc, style  # noqa: E402

GRIDW = (1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0)
STEPS = (4, 8, 20, 40, 80, 120, 200)
METRICS = (("fid", "FID", COND, "o"), ("fd_dinov2", r"FD$_{\rm DINOv2}$", ACC, "s"))


def load(metrics_dir):
    cells = {}
    for p in glob.glob(str(Path(metrics_dir) / "metrics-sd-coco-sd35-medium-*-s*-f32-pr.json")):
        m = re.search(r"medium-(unguided|const[\d.]+)-full-10k-o5705-s(\d+)-f32", p)
        if m:
            w = 1.0 if m.group(1) == "unguided" else float(m.group(1)[5:])
            cells[(int(m.group(2)), w)] = json.load(open(p))
    return cells


def best(cells, key):
    """{N: (best w, metric at w = 1, metric at best w)}."""
    out = {}
    for n in STEPS:
        ws = [w for w in GRIDW if (n, w) in cells]
        vals = [cells[(n, w)][key] for w in ws]
        i = int(np.argmin(vals))
        out[n] = (ws[i], cells[(n, 1.0)][key], vals[i])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--metrics", default=str(REPO / "results" / "gsweep" / "metrics"))
    ap.add_argument("--out", default=str(REPO / "results" / "steps-gain-ratio"))
    args = ap.parse_args()
    plt = rc(7.5, 6.2)
    from matplotlib.ticker import FixedLocator, NullLocator, PercentFormatter, ScalarFormatter

    cells = load(args.metrics)
    fig, axes = plt.subplots(2, 2, figsize=(W_COL, 2.9), sharex=True,
                             gridspec_kw={"wspace": 0.38, "hspace": 0.22})
    for row, (key, label, colour, marker) in zip(axes, METRICS):
        b = best(cells, key)
        gain = [b[n][1] - b[n][2] for n in STEPS]
        pct = [100 * (b[n][1] - b[n][2]) / b[n][1] for n in STEPS]
        print(f"{key:10s} " + "  ".join(f"{n}: -{g:.2f} -{p:.1f}% (w={b[n][0]:g})"
                                        for n, g, p in zip(STEPS, gain, pct)))
        for ax, y in zip(row, (gain, pct)):
            ax.plot(STEPS, y, color=colour, lw=1.6, marker=marker, ms=3.4, mec="white", mew=0.6)
            style(ax, grid=False)
            ax.grid(True, axis="y", color=GRID, lw=0.5)
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator(STEPS))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.set_xticklabels([str(n) if n != 120 else "" for n in STEPS])   # 80 / 120 / 200 crowd
        row[0].set_ylim(0, None)
        row[0].set_ylabel(label)
        row[1].set_ylim(0, 100)
        row[1].set_yticks([0, 25, 50, 75, 100])
        row[1].yaxis.set_major_formatter(PercentFormatter(decimals=0))
    axes[0, 0].set_title(r"improvement: $a-b$", fontsize=7, color=INK, pad=4)
    axes[0, 1].set_title(r"reduction: $(a-b)/a$", fontsize=7, color=INK, pad=4)
    fig.text(0.5, -0.04, r"$a$ = metric at $w=1$,  $b$ = metric at the best constant $w$",
             ha="center", va="top", fontsize=6.2, color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("sampler steps $N$")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
