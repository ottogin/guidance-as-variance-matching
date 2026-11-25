#!/usr/bin/env python
"""fig:moments as a 2x2 grid: M(1), M(2) and M(inf) on four datasets whose posterior is exact.

    python reproduce/figures/fig_moments_grid.py --out paper-figures/moment-samplers-2x2

(a) the 2-D mixture of ten Gaussians: reproduce/figures/fig_moments.py's cache ($VMG_RUNS/_toys/moments).
(b) a union of 20 random 8-dimensional subspaces (d = 1024), (c) 10,000 CIFAR-10 images (d = 3072), (d) a
    100-component Gaussian mixture fit to CIFAR-10 (d = 3072): reproduce/toys/moments_highd.py, cached in
    $VMG_RUNS/_toys/moments_highd/res/conv_<ds>.json. (Its sparse-sources set is left out: M(1)'s error rises from 2 to 6 steps.)
Each panel plots its dataset's own MMD (the one the high-d script marks primary). Values at or below the noise floor
(twice the standard deviation of the null runs, in MMD^2) are drawn hollow, on the floor. Prints the numbers quoted
in the caption and the appendix.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce import style as F  # noqa: E402

KS = ["1", "2", "inf"]
COL = {"1": "#2569ad", "2": "#d15b26", "inf": "#666666"}
MARK = {"1": "o", "2": "s", "inf": "o"}
LAB = {"1": "$M(1)$: mean prediction", "2": "$M(2)$: mean + covariance", "inf": r"$M(\infty)$: full posterior"}
PANELS = [("2d", "(a) 2D mixture of ten Gaussians"),
          ("subsp", "(b) union of subspaces, $d=1024$"),
          ("cifar", "(c) CIFAR-10 images, $d=3072$"),
          ("cgmm", "(d) CIFAR-10 mixture, $d=3072$")]


def load(src, moments, highd):
    """(NS, {k: MMD per N}, floor) for one panel."""
    if src == "2d":
        c = json.load(open(moments / "conv.json"))
        ns = c["NS"]
        mmd = {k: np.array([math.sqrt(max(c["mmd2"][k][str(N)], 0)) for N in ns]) for k in KS}
        return ns, mmd, math.sqrt(2 * np.std(c["null"], ddof=1))
    r = json.load(open(highd / f"conv_{src}.json"))
    p = r["measures"].index(r["primary"])
    return r["NS"], {k: np.array(r["mmd"][k][p]) for k in KS}, r["floors"][p]


def euler_steps(ns, m1, v):
    """Steps at which M(1) reaches the error v (log-log interpolation); nan if it never does by the last N."""
    for i in range(len(ns) - 1):
        if m1[i] >= v >= m1[i + 1]:
            f = (math.log(v) - math.log(m1[i])) / (math.log(m1[i + 1]) - math.log(m1[i]))
            return math.exp(math.log(ns[i]) + f * (math.log(ns[i + 1]) - math.log(ns[i])))
    return math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--moments", default=None, help="fig_moments.py cache (default: $VMG_RUNS/_toys/moments)")
    ap.add_argument("--highd", default=None,
                    help="reproduce/toys/moments_highd.py's res/ folder (default: $VMG_RUNS/_toys/moments_highd/res)")
    a = ap.parse_args()
    if a.highd is None:
        import os
        a.highd = str(Path(os.environ.get("VMG_RUNS", REPO / "runs")) / "_toys" / "moments_highd" / "res")
    if a.moments is None:
        from vmg import runs
        a.moments = str(runs.root() / "_toys" / "moments")
    plt = F.rc(7, 6)

    fig, axs = plt.subplots(2, 2, figsize=(F.W_COL, 2.85))
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.115, top=0.86, wspace=0.3, hspace=0.5)
    for j, (src, title) in enumerate(PANELS):
        ax = axs.flat[j]
        ns, mmd, floor = load(src, Path(a.moments), Path(a.highd))
        ns_ = np.array(ns)
        ax.axhspan(1e-12, floor, color="#ececea", lw=0, zorder=0)
        for k in KS:
            y = np.maximum(mmd[k], floor)
            hol = mmd[k] <= floor
            z = 2.5 if k == "inf" else 3                    # the reference sits under the samplers on the floor
            ax.plot(ns_, y, color=COL[k], lw=1.0, ls=(0, (3, 1.6)) if k == "inf" else "-", zorder=z, label=LAB[k],
                    marker=MARK[k], ms=2.3, mec="white", mew=0.3, markevery=[int(i) for i in np.flatnonzero(~hol)])
            ax.plot(ns_[hol], y[hol], MARK[k], ms=2.3, mfc="white", mec=COL[k], mew=0.6, zorder=z + 1)

        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
        ax.set_xlim(0.75, ns[-1] * 1.35)
        top = max(float(np.nanmax(mmd[k])) for k in KS)
        ax.set_ylim(floor * 0.3, top * 2.2)
        F.style(ax)
        ax.grid(False, which="minor")
        ax.tick_params(axis="both", labelsize=5.8, pad=1.5)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        if j % 2 == 0:
            ax.set_ylabel("MMD to data", labelpad=1)
        if j >= 2:
            ax.set_xlabel("Sampling steps $N$", labelpad=1)
        if j == 0:
            ax.text(0.85, floor * 0.47, "noise floor", color=F.MUTED, fontsize=5.4, va="center")
        ax.text(0, 1.06, title, transform=ax.transAxes, fontsize=6.6, color=F.INK)

        hit = {k: next((N for N, x in zip(ns, mmd[k]) if x <= floor), None) for k in KS}
        print(f"{src:6s} floor {floor:.2e}; first at the floor: M(1) {hit['1']}, M(2) {hit['2']}, "
              f"M(inf) {hit['inf']}; M(1) at {ns[-1]} = {mmd['1'][-1] / floor:.1f} x floor; Euler-equivalent steps "
              f"of M(2): " + ", ".join(f"{N}: {euler_steps(ns, mmd['1'], mmd['2'][ns.index(N)]):.0f}"
                                       for N in (4, 8, 16, 32)))

    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.0), handlelength=1.8,
               columnspacing=0.9, handletextpad=0.4, fontsize=6)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf", dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
