#!/usr/bin/env python
"""Appendix figure fig:shared-error: the errors of EDM2-S and its guide XS on ImageNet-64, XL standing in for D*.

    python reproduce/figures/fig_shared_error.py --out paper-figures/shared-error [--dir DIR]

Reads analysis.json of reproduce/experiments/errors_in64.py (default $VMG_RUNS/errors-in64). With E_i = D_i - D*
and the guide's error regressed on S's, E0 = kappa E1 + E0perp, the weight closest to D* exceeds 1 iff kappa > 1.
(a) a hypothetical pair that fits the shared-error view (kappa = 2, |E0perp| = 0.35 |E1|)
(b) the measured triangle D*, D0, D1 at sigma = 1.2, with the weight Karras et al. use (1.7)
(c) kappa(sigma), with the middle 50% and 90% of the per-input ratio alpha_0 / alpha_1 = <E0, E1> / |E1|^2
(d) the weight closest to D* against sigma, with its 90% bootstrap band over images
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.experiments.errors_in64 import MAIN, folder  # noqa: E402
from reproduce.style import ACC as ORANGE, COND as BLUE, INK, MUTED, rc  # noqa: E402

FAINT, GRID = "#a9a8a2", "#e4e3df"
USED = 1.70          # NVIDIA's FID preset for exactly these checkpoints (Karras et al. 2024)
SIGMA_B = 1.2        # the noise level of panel (b)


def triangle(ax, kappa, perp, title, d1_off, star_off, used=None, k_off=(0, -7)):
    """D* at the origin, S's error E1 = (1, 0), the guide's E0 = (kappa, perp), and the guidance line D_w."""
    d1, d0 = np.array([1.0, 0.0]), np.array([kappa, perp])
    L = lambda w: d0 + w * (d1 - d0)  # noqa: E731
    ws = float(d0 @ (d0 - d1) / ((d1 - d0) @ (d1 - d0)))
    ax.plot(*np.array([L(-0.2), L(2.3)]).T, color=FAINT, lw=0.9, zorder=1)
    ax.plot([0, 2.2], [0, 0], color=FAINT, lw=0.6, ls=(0, (2, 2)), zorder=0)          # the direction of S's error
    ax.plot([kappa, kappa], [0, perp], color=ORANGE, lw=0.8, ls=":")                # E0perp, down to kappa E1
    ax.plot(kappa, 0, "|", ms=7, color=ORANGE, mew=1.4)
    ax.annotate("$\\kappa E_1$", (kappa, 0), xytext=k_off, textcoords="offset points", color=ORANGE, fontsize=7.5,
                ha="center" if k_off[0] == 0 else "left", va="top" if k_off[1] < 0 else "bottom")
    for p, c in ((d0, ORANGE), (d1, BLUE)):
        ax.annotate("", p, (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, shrinkA=0, shrinkB=2.5,
                                                   mutation_scale=8))
    ax.plot(*d0, "o", ms=4.5, color=ORANGE)
    ax.plot(*d1, "o", ms=4.5, color=BLUE)
    ax.plot(*L(ws), "*", ms=9, color=INK, mec="white", mew=0.4, zorder=3)
    ax.plot([0, L(ws)[0]], [0, L(ws)[1]], color=INK, lw=0.6, ls=":")
    ax.plot(0, 0, "+", ms=8, color=INK, mew=1.2)
    ax.annotate("$D^\\star$", (0, 0), xytext=(-14, 3), textcoords="offset points", fontsize=8)
    ax.annotate("$D_0$ ($w{=}0$)", d0, xytext=(4, 1), textcoords="offset points", fontsize=7.5, color=ORANGE)
    ax.annotate("$D_1$ ($w{=}1$)", d1, xytext=d1_off, textcoords="offset points", fontsize=7.5, color=BLUE)
    ax.annotate(f"$w^\\star={ws:.2f}$", L(ws), xytext=star_off, textcoords="offset points", fontsize=7.5)
    if used is not None:
        ax.plot(*L(used), "x", ms=5, color=INK, mew=1.1, zorder=3)
        ax.annotate(f"$w={used:g}$, used", L(used), xytext=(5, -3), textcoords="offset points", fontsize=7.5,
                    color=INK)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.25, 2.5)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(title, loc="left")


def axis_noise(ax, ylim, title):
    ax.axhline(1, color=FAINT, lw=0.9)
    ax.set_xscale("log")
    ax.set_xlim(100, 0.0015)                       # sampling order: high noise on the left
    ax.set_xlabel("noise level $\\sigma$")
    ax.grid(color=GRID, lw=0.5)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left")


def figure(res, plt):
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.95), gridspec_kw={"width_ratios": [0.9, 0.9, 1, 1]})
    triangle(axes[0], 2.0, 0.35, "(a) hypothetical, $\\kappa=2$", (2, -11), (2, -13), k_off=(4, 1))
    sig, P = np.array(res["sigma"]), res["pairs"][MAIN]
    k = int(np.argmin(np.abs(sig - SIGMA_B)))
    kappa, q = P["kappa"]["value"][k], P["q"]["value"][k]
    triangle(axes[1], kappa, np.sqrt(q), f"(b) measured, $\\kappa={kappa:.2f}$", (6, 3), (6, -1), used=USED)
    ax = axes[2]
    A = P["alpha_ratio"]
    ax.fill_between(sig, A["p5"], A["p95"], color=BLUE, alpha=0.13, lw=0)
    ax.fill_between(sig, A["p25"], A["p75"], color=BLUE, alpha=0.28, lw=0)
    ax.plot(sig, P["kappa"]["value"], color=BLUE, lw=1.3)
    axis_noise(ax, (-0.2, 1.45), "(c) share of S's error, $\\kappa$")
    ax.text(0.0017, 1.03, "$\\kappa>1$: $w^\\star>1$", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.plot(sig[k], kappa, "o", ms=4, mfc="white", mec=BLUE, mew=1.0, zorder=4)
    ax.annotate("(b)", (sig[k], kappa), xytext=(0, 3), textcoords="offset points", fontsize=7, ha="center",
                va="bottom")
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.3, label="average"),
                       plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.28, lw=0, label="50% of inputs"),
                       plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.13, lw=0, label="90% of inputs")],
              frameon=False, loc="lower left", fontsize=6.5, handlelength=1.6, borderaxespad=0.2)
    ax = axes[3]
    ax.fill_between(sig, P["w_ref"]["lo"], P["w_ref"]["hi"], color=BLUE, alpha=0.15, lw=0)
    ax.plot(sig, P["w_ref"]["value"], color=BLUE, lw=1.3)
    axis_noise(ax, (0, 2.1), "(d) weight closest to $D^\\star$, $w^\\star$")
    ax.text(0.0017, 1.03, "no guidance", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    ax.axhline(USED, color=MUTED, lw=0.9)
    ax.text(0.0017, USED + 0.04, f"used, $w={USED:g}$", fontsize=6.5, color=MUTED, ha="right", va="bottom")
    fig.tight_layout(w_pad=0.8)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", help="errors_in64.py's output folder (default $VMG_RUNS/errors-in64)")
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    a = ap.parse_args()
    plt = rc(size=8, small=7)
    plt.rcParams.update({"axes.titlesize": 8, "axes.spines.top": False, "axes.spines.right": False})
    res = json.loads((folder(a.dir) / "analysis.json").read_text())
    fig = figure(res, plt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf", bbox_inches="tight")
    fig.savefig(a.out + ".png", bbox_inches="tight", dpi=220)
    print(f"wrote {a.out}.pdf")


if __name__ == "__main__":
    main()
