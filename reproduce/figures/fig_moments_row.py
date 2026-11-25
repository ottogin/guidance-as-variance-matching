#!/usr/bin/env python
"""fig:moments as one row: M(1), M(2) and M(inf) against sampling steps on datasets with exact posteriors.

    python reproduce/figures/fig_moments_row.py --out paper-figures/moment-samplers-row

The paper's panels (--panels default) are the 2-D mixture and ImageNet-64; CIFAR-10 is available as a third.
The datasets: the 2-D mixture of ten Gaussians, 10,000 CIFAR-10 images (d = 3072), 10,000 ImageNet-64 training images,
10 per class (d = 12,288). The image sets are point masses, so the posterior is a softmax over the images; both sets come
from reproduce/toys/moments_highd.py (datasets cifar and in64), cached as $VMG_RUNS/_toys/moments_highd/res/conv_<ds>.json,
and plot its primary MMD (small kernel, half the median nearest-neighbour distance). Data loading, the noise floor
and the printed numbers are fig_moments_grid.py's; this is the paper's row layout (M(inf) drawn like M(1) and M(2), in grey) with the
ImageNet-64 panel added.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce import style as F  # noqa: E402
from reproduce.figures.fig_moments_grid import KS, euler_steps, load  # noqa: E402

COL = {"1": "#2569ad", "2": "#d15b26", "inf": "#7a7a78"}
MARK = {"1": "o", "2": "s", "inf": "s"}
LAB = {"1": "mean", "2": "mean + covariance", "inf": "full posterior"}      # the samplers M(1), M(2), M(inf)
TITLES = {"2d": ("2D mixture of ten Gaussians", "2D mixture"),
          "cifar": ("CIFAR-10 images, $d=3072$", "CIFAR-10"),
          "in64": ("ImageNet-64 images, $d=12{,}288$", "ImageNet-64")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--moments", default=None, help="fig_moments.py cache (default: $VMG_RUNS/_toys/moments)")
    ap.add_argument("--highd", default=None,
                    help="reproduce/toys/moments_highd.py's res/ folder (default: $VMG_RUNS/_toys/moments_highd/res)")
    ap.add_argument("--panels", default="2d,in64", help="comma-separated datasets, lettered in this order (default: "
                    "the paper's two panels; 2d,cifar,in64 is the earlier three-panel row)")
    ap.add_argument("--wide", action="store_true", help="the full text width with long titles (default: one column, "
                    "as in the paper: short titles, a shared y axis, the datasets spelled out in the caption)")
    a = ap.parse_args()
    if a.highd is None:
        import os
        a.highd = str(Path(os.environ.get("VMG_RUNS", REPO / "runs")) / "_toys" / "moments_highd" / "res")
    if a.moments is None:
        from vmg import runs
        a.moments = str(runs.root() / "_toys" / "moments")
    PANELS = [(src, f"({chr(97 + j)}) {TITLES[src][0]}", f"({chr(97 + j)}) {TITLES[src][1]}")
              for j, src in enumerate(a.panels.split(","))]
    plt = F.rc(7, 6)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    data = [load(src, Path(a.moments), Path(a.highd)) for src, _, _ in PANELS]
    top = False
    if a.wide:
        fig, axs = plt.subplots(1, len(PANELS), figsize=(2 * F.W_COL + 0.33, 1.55))
        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.30, top=0.86, wspace=0.16)
        xt, ms, tfs, xl, lg, edge = [1, 4, 16, 64, 256, 1024], 1.0, 6.8, (0.53, 0.115), (0.53, -0.04), "white"
    else:                                                    # one column: y shared
        fig, axs = plt.subplots(1, len(PANELS), figsize=(F.W_COL, 1.31), sharey=True)
        fig.subplots_adjust(left=0.115, right=0.985, bottom=0.33, top=0.87, wspace=0.15)
        xt, ms, tfs, xl, lg, edge = [1, 32, 1024], 0.8, 6.6, (0.55, 0.15), (0.53, -0.035), None   # white edges read as dashes
        top = len(PANELS) <= 2
        if top:                                              # two wider panels: every other power of two, and the
            xt = [1, 4, 16, 64, 256, 1024]                   # legend in a row above them, away from the caption
            fig.set_size_inches(F.W_COL, 1.30)
            fig.subplots_adjust(bottom=0.2, top=0.78)
            xl, lg = (0.55, 0.03), (0.53, 1.0)
        lo = min(fl for _, _, fl in data) * 0.3
        hi = max(float(np.nanmax(m[k])) for _, m, _ in data for k in KS) * 2.2
    for j, (src, title, short) in enumerate(PANELS):
        ax = axs[j]
        ns, mmd, floor = data[j]
        ns_ = np.array(ns)
        ax.axhspan(1e-12, floor, color="#ececea", lw=0, zorder=0)
        for k in KS:
            y = np.maximum(mmd[k], floor)
            hol = mmd[k] <= floor
            if k == "inf":                                   # exact posterior draws: at the floor up to seed noise
                y, hol = np.full_like(y, floor), np.ones_like(hol)   # (2-D, N=2 is 1.03x floor); keep all hollow
            ax.plot(ns_, y, color=COL[k], lw=1.1 * ms, zorder=3 - 1.5 * (k == "inf"), marker=MARK[k], ms=2.4 * ms, mec=edge or COL[k], mew=0.3,
                    markevery=[int(i) for i in np.flatnonzero(~hol)])
            ax.plot(ns_[hol], y[hol], MARK[k], ms=2.4 * ms, mfc="white", mec=COL[k], mew=0.6, zorder=4 - 1.5 * (k == "inf"))   # M(inf) under the others
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(xt)
        ax.set_xticklabels([str(x) for x in xt])
        ax.set_xlim(0.75, ns[-1] * 1.35)
        if a.wide:
            ax.set_ylim(floor * 0.3, max(float(np.nanmax(mmd[k])) for k in KS) * 2.2)
        else:
            ax.set_ylim(lo, hi)
            ax.set_yticks([1e0, 1e-2, 1e-4])
        F.style(ax)
        ax.grid(False, which="minor")
        if not a.wide:
            ax.grid(False, axis="x")
        ax.tick_params(axis="both", labelsize=5.8, pad=1.5)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_minor_locator(plt.NullLocator())
        if j == 0:
            ax.set_ylabel("MMD to data", labelpad=1)
        ax.set_title(title if a.wide else short, fontsize=tfs, color=F.INK, pad=3)

        hit = {k: next((N for N, x in zip(ns, mmd[k]) if x <= floor), None) for k in KS}
        print(f"{src:6s} floor {floor:.2e}; first at the floor: M(1) {hit['1']}, M(2) {hit['2']}, "
              f"M(inf) {hit['inf']}; M(1) at {ns[-1]} = {mmd['1'][-1] / floor:.1f} x floor; Euler-equivalent steps "
              f"of M(2): " + ", ".join(f"{N}: {euler_steps(ns, mmd['1'], mmd['2'][ns.index(N)]):.0f}"
                                       for N in (4, 8, 16, 32) if N in ns))
    fig.text(*xl, "sampling steps $K$", ha="center", fontsize=7, color=F.INK)
    h = [Line2D([], [], color=COL["1"], lw=1.1, marker="o", ms=2.4, mec=edge or COL["1"], mew=0.3),
         Line2D([], [], color=COL["2"], lw=1.1, marker="s", ms=2.4, mec=edge or COL["2"], mew=0.3),
         Line2D([], [], color=COL["inf"], lw=1.1, marker="s", ms=2.4, mfc="white", mec=COL["inf"], mew=0.6),
         Patch(color="#ececea", lw=0)]
    labs = [LAB["1"], LAB["2"], LAB["inf"], "noise floor"]
    fig.legend(h, labs, frameon=False, loc="upper center" if top else "lower center", ncol=4,
               bbox_to_anchor=lg, handlelength=1.8 if a.wide else 1.4, columnspacing=1.4 if a.wide else 0.8,
               handletextpad=0.4 if a.wide else 0.3, fontsize=6 if a.wide else 5.6)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf", dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
