#!/usr/bin/env python
"""Appendix figure fig:mean-worse: guidance moves the prediction away from the posterior mean while the samples
improve (EDM2-S guided by XS on ImageNet-64), three panels stacked on one w axis.

    python reproduce/figures/fig_mean_worse.py --out paper-figures/mean-worse [--dir DIR]

Reads, from reproduce/experiments/errors_in64.py's folder (default $VMG_RUNS/errors-in64), analysis.json and
the sweep's sweep/metrics/w*.json (ens_edm2.py eval).
(a) E|D_w - D*|^2 relative to S alone (w = 1), XL standing in for D*: median over the 32 noise levels (line)
    and their range (band). The change of E|D_w - x0|^2 with w equals that of E|D_w - D*|^2, so where the
    distance rises needs no stand-in; only its size relative to S does. w* is the closest grid point.
(b, c) FID and FD_DINOv2 of 10,000 samples at each constant w, the best one starred, the value at w* in red.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.experiments.errors_in64 import MAIN, W_GRID, distance_ratio, folder  # noqa: E402
from reproduce.style import COND as BLUE, rc  # noqa: E402

RED, DARK, GRAY, LINE, GRID = "#d1495b", "#222222", "#6b6b6b", "#c9c9c4", "#e6e6e3"
FS = 7                                     # annotations and tick labels; titles and the axis label 7.5
# panel: (y axis top, y tick step, height of the red label's base in axes units)
PANELS = {"dist": (6, 2, None), "fid": (12, 4, 0.032), "fd_dinov2": (360, 100, 0.258)}


def sweep(d):
    """(w, FID, FD_DINOv2) of the sweep's arms, in order of w."""
    rs = [json.loads(p.read_text()) for p in (d / "sweep" / "metrics").glob("w*.json")]
    assert rs, f"no sweep metrics in {d / 'sweep' / 'metrics'}"
    rs.sort(key=lambda r: float(r["args"]["w"]))
    return tuple(np.array([f(r) for r in rs]) for f in (lambda r: float(r["args"]["w"]), lambda r: r["fid"],
                                                         lambda r: r["fd_dinov2"]))


def red_label(ax, w, y, s, va):
    """Right-aligned red text left of the w* line, at weight w and height y in axes units."""
    ax.text(w, y, s, transform=ax.get_xaxis_transform(), fontsize=FS, color=RED, ha="right", va=va,
            multialignment="right", linespacing=0.95)


def frame(ax, key, title, wstar):
    top, step, _ = PANELS[key]
    ax.set_xlim(W_GRID[0], W_GRID[-1])
    ax.set_xticks(np.arange(-0.5, 3.01, 0.5))
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 1e-9, step))
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for w in (0, 1):                                            # XS and S alone
        ax.axvline(w, color=LINE, lw=0.8, zorder=0.6)
    ax.axvline(wstar, color=RED, lw=1.0, ls=(0, (3, 1.6)), zorder=0.7)
    ax.set_title(title, loc="left", fontsize=7.5, pad=3)


def distance_panel(ax, P, best):
    R = distance_ratio(P["kappa"]["value"], P["q"]["value"], W_GRID)
    med = np.median(R, 0)
    j = int(np.argmin(med))
    ws = W_GRID[j]
    at = lambda w: float(np.interp(w, W_GRID, med))  # noqa: E731
    frame(ax, "dist", "(a) distance to the posterior mean, $\\mathbb{E}\\|D_w-D^\\star\\|^2$ rel. to S", ws)
    ax.fill_between(W_GRID, R.min(0), R.max(0), color=BLUE, alpha=0.16, lw=0, zorder=0.8)
    ax.plot(W_GRID, med, color=BLUE, lw=1.3, zorder=3)
    for w, name in ((0, "XS"), (1, "S")):
        ax.plot(w, at(w), "o", ms=3.6, mfc="white", mec=GRAY, mew=0.8, zorder=4)
        ax.text(w + 0.04, 0.97, name, transform=ax.get_xaxis_transform(), fontsize=FS, color=GRAY, va="top")
    for w, what in best:
        ax.plot(w, at(w), "o", ms=3.6, color=DARK, mec="white", mew=0.5, zorder=4)
        ax.annotate(f"$w$ of best {what}", (w, at(w)), xytext=(5, -3), textcoords="offset points", fontsize=FS,
                    color=DARK, va="top")
    ax.plot(ws, med[j], "*", ms=9, color=RED, mec="white", mew=0.5, zorder=5)
    red_label(ax, ws - 0.05, 0.697, f"closest\nto the mean\n$w^\\star={ws:.2f}$", "center")
    return ws, R


def metric_panel(ax, key, title, w, y, ws):
    i = int(np.argmin(y))
    yw = float(np.interp(ws, w, y))
    frame(ax, key, title, ws)
    ax.axhline(y[i], color=DARK, lw=0.6, ls=(0, (1, 1.5)), zorder=0.9)
    ax.plot(w, y, "-o", color=BLUE, lw=1.3, ms=2.8, mec="white", mew=0.4, zorder=3)
    ax.plot(w[i], y[i], "*", ms=9, color=DARK, mec="white", mew=0.5, zorder=5)
    ax.annotate(f"best, $w={w[i]:g}$", (w[i], y[i]), xytext=(0, 6), textcoords="offset points", fontsize=FS,
                color=DARK, ha="center", va="bottom")
    ax.plot(ws, yw, "o", ms=4.2, color=RED, mec="white", mew=0.5, zorder=5)
    val = f"{yw:.1f}" if yw < 10 else f"{yw:.0f}"
    red_label(ax, ws - 0.13, PANELS[key][2], f"$w^\\star$: {val}\n$\\times${yw / y[i]:.1f} best", "bottom")
    return yw, y[i], w[i]


def figure(res, sw, plt):
    fig, axes = plt.subplots(3, 1, figsize=(3.0446, 3.7257))                # the paper's 219 x 268 pt
    fig.subplots_adjust(left=0.0864, right=0.9264, bottom=0.0997, top=0.9574, hspace=0.48)
    w, fid, dino = sw
    best = [(w[np.argmin(fid)], "FID"), (w[np.argmin(dino)], "FD")]
    ws, R = distance_panel(axes[0], res["pairs"][MAIN], best)
    k = int(np.argmin(np.abs(W_GRID - 1.7)))
    print(f"w* = {ws:.2f}; at w = 1.7 the distance is {R[:, k].min():.2f}-{R[:, k].max():.2f} times S's")
    for ax, key, title, y in ((axes[1], "fid", "(b) FID, 10k samples", fid),
                              (axes[2], "fd_dinov2", "(c) FD$_{\\mathrm{DINOv2}}$, 10k samples", dino)):
        yw, yb, wb = metric_panel(ax, key, title, w, y, ws)
        print(f"{key}: best {yb:.2f} at w = {wb:g}; at w* {yw:.2f} ({yw / yb:.2f} x best)")
    axes[2].set_xlabel("guidance weight $w$")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", help="errors_in64.py's output folder (default $VMG_RUNS/errors-in64)")
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    a = ap.parse_args()
    d = folder(a.dir)
    plt = rc(size=7.5, small=FS)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
    fig = figure(json.loads((d / "analysis.json").read_text()), sweep(d), plt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=300)
    print(f"wrote {a.out}.pdf")


if __name__ == "__main__":
    main()
