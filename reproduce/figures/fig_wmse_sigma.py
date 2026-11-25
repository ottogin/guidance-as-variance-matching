#!/usr/bin/env python
"""Appendix figure fig:wmse-sigma: w*_MSE along the sampler for six guided pairs, overlaid in one panel.

    python reproduce/figures/fig_wmse_sigma.py --out paper-figures/wmse-sigma

Reads runs/<model>/<dataset>/wmse.json (compute_wmse.py) for the pairs below, all in
reproduce/rosters/wmse_pairs.py. The x-axis is progress through each pair's own sampler, from the first step
(highest noise) to the last, so that samplers with different noise schedules share one axis. The pure-noise
first step of the flow-matching models (sigma = inf, s = 1) is left out. No bootstrap bands: the figure shows
where each curve lies against w = 1, and the bands are narrow next to that distance.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.rosters.wmse_pairs import find  # noqa: E402
from vmg import runs  # noqa: E402

INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#c3c2b7"
EXTRA, EXTRA_INK = "#f6efe4", "#9a6a2a"        # the region w > 1 (extrapolation) and its label

# (roster pair, line label, colour), drawn in this order
PAIRS = [
    ("IN512 autog S<-XS (fid)", "EDM2 IN-512 (AG)", "#2a78d6"),
    ("IN64 autog S<-XS (fid)", "EDM2 IN-64 (AG)", "#7fb0ea"),
    ("FLUX.2-klein MJHQ", "FLUX.2 [klein]", "#1baf7a"),
    ("CogVideoX-2B (OpenVid)", "CogVideoX", "#e87ba4"),
    ("cross: SD1.5 <- CommonCanvas (COCO)", "SD 1.5 $\\leftarrow$ CC", "#eda100"),
    ("MDM KIT-ML (text-to-motion)", "MDM", "#4a3aa7"),
]
YLIM = (-0.5, 1.45)
LABEL_GAP = 0.14          # least vertical distance between two end labels, in units of w


def load(name):
    p = find(name)
    r = json.load(open(runs.root() / p.model / p.dataset / "wmse.json"))
    t = np.asarray(r["sigma"], float)
    w = np.asarray(r["w_mse"], float)
    keep = np.isfinite(t)                                 # drop the pure-noise step of flow models
    t, w = t[keep], w[keep]
    return w[np.argsort(-t)], p.pub                       # sampling order: high noise first


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                                "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})

    fig, ax = plt.subplots(figsize=(3.3, 2.25))           # one CVPR column
    ax.axhspan(1.0, YLIM[1], color=EXTRA, lw=0, zorder=0)
    ax.axhline(1.0, color=INK, lw=0.6, ls=(0, (4, 3)), zorder=1)
    ends, pubs = [], []
    for name, label, col in PAIRS:
        w, pub = load(name)
        ax.plot(np.linspace(0, 1, len(w)), w, color=col, lw=1.2, zorder=3)
        ends.append([w[-1], label, col])
        if "CommonCanvas" not in name:
            pubs.append(pub)
    ends.sort(key=lambda e: e[0])                        # end labels in order, spread apart where they crowd
    ys = [e[0] for e in ends]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + LABEL_GAP)
    for (y0, label, col), y in zip(ends, ys):
        ax.annotate(label, (1, y0), xytext=(1.03, y), textcoords=("axes fraction", "data"), fontsize=5.8,
                    color=col, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.4, shrinkA=0, shrinkB=1))

    ax.set_xlim(0, 1)
    ax.set_ylim(*YLIM)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["first step\n(high noise)", "", "last step\n(low noise)"])
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.2, colors=INK, length=2, width=0.5, color=BASE, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
        ax.spines[sp].set_linewidth(0.5)
    ax.set_ylabel("$w^\\star_{\\mathrm{MSE}}$", fontsize=7.5, color=INK, labelpad=1)
    ax.set_xlabel("progress through the sampler", fontsize=7, color=INK, labelpad=1)
    ax.text(0.98, 1.03, "extrapolation, $w>1$", transform=ax.get_yaxis_transform(), fontsize=5.8,
            color=EXTRA_INK, va="bottom", ha="right")
    ax.text(0.02, 0.97, f"every published scale is above this range ({min(pubs):g} to {max(pubs):g})",
            transform=ax.transAxes, fontsize=5.6, color=MUTED, va="top")
    fig.subplots_adjust(left=0.13, right=0.74, bottom=0.2, top=0.97)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor="white")
    print("wrote", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
