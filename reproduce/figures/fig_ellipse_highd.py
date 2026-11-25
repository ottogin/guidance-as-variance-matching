#!/usr/bin/env python
"""fig:ellipse at CIFAR-10 size: how likely the final samples are under p(x | c), against the guidance strength.

    python reproduce/figures/fig_ellipse_highd.py --res RES --out OUT/ellipse-highd

Reads RES/<ds>.json (ellipse_highd.py run) and RES/<ds>-sweep.json, or the -sweep-{scaled,const}.json halves, from
reproduce/toys/ellipse_highd.py sweep [--group scaled|const] (ds = cls: the ten CIFAR-10
classes; clu: every cluster its own class). Columns: the two labellings. Top row: the matched schedule scaled,
w(t) = 1 + lam (w_matched(t) - 1), lam = 0 being plain conditional sampling and lam = 1 the matched schedule (star).
Bottom row: constant weights, with the matched schedule's value dashed. y: mean log p(x | c) per coordinate of the
16-step samples (left), share of samples denser than 99% of real data (right axis, light).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce import style as F  # noqa: E402

COLS = [("cls", "CIFAR-10 classes"), ("clu", "clusters as classes")]
LL, SH = "#2569ad", "#d15b26"


STEPS = (1, 4, 7, 15)


def steps_figure(res, out, plt):
    """fig:ellipse's share row at CIFAR-10 size: at a few steps of the matched run, the mean log-density of
    D_w = D_0 + w (D_1 - D_0) under p(x | c), against w (RES/<ds>.json of ellipse_highd.py run). The share above
    real data's density lines is 100% for every w at these steps, so the mean is plotted instead."""
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, len(STEPS), figsize=(5.5, 2.75))
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.88, wspace=0.42, hspace=0.75)
    for r_, (ds, name) in enumerate(COLS):
        R = json.load(open(Path(res) / f"{ds}.json"))
        wg = np.array(R["wgrid"])
        for k, step in enumerate(STEPS):
            ax, s = axs[r_, k], R["steps"][step - 1]
            sel = wg <= 6.0
            m = np.array(s["mean_cond"])[sel]
            x = wg[sel]
            wv = s["w_matched"]
            i1, iv = int(np.argmin(np.abs(x - 1))), int(np.argmin(np.abs(x - wv)))
            ax.plot(x, m, color=F.INK, lw=1.1, zorder=2)
            if wv <= 6:
                ax.axvline(wv, color=F.ACC, lw=0.8, ls=(0, (3, 1.6)), zorder=1)
                ax.plot([wv], [m[iv]], "*", ms=8, color=F.ACC, mec="white", mew=0.5, zorder=5, clip_on=False)
            ax.plot([0], [m[0]], "o", ms=4, color=F.UNCOND, mec="white", mew=0.6, zorder=4, clip_on=False)
            ax.plot([1], [m[i1]], "o", ms=4, color=F.COND, mec="white", mew=0.6, zorder=4, clip_on=False)
            j = int(np.argmax(m))
            ax.plot([x[j]], [m[j]], "o", ms=6, mfc="none", mec=F.INK, mew=0.7, zorder=3, clip_on=False)
            span = m.max() - m.min()
            if span < 1e-4:                                          # D_1 = D_0: w changes nothing
                ax.set_ylim(m.mean() - 1e-3, m.mean() + 1e-3)
                ax.text(0.55, 0.12, "$D_1 = D_0$:\n$w$ has no effect", transform=ax.transAxes, ha="center",
                        fontsize=5.8, color=F.MUTED, linespacing=1.05)
            ax.set_xlim(0, 6)
            ax.set_xticks(range(0, 7, 2))
            F.style(ax)
            ax.tick_params(labelsize=5.6, pad=1.2, length=2)
            ax.yaxis.get_major_formatter().set_useOffset(False)
            ax.text(0, 1.06, f"step {step} of 16, $t={s['t']:.2f}$" + (f", $w_{{VM}}={wv:.1f}$" if wv <= 6 else
                                                                          f", $w_{{VM}}>6$"),
                    transform=ax.transAxes, fontsize=6.0, color=F.INK)
            if k == 0:
                ax.set_ylabel(f"{name}\nmean $\\log p(D_w|c)/d$", fontsize=6.2, labelpad=2)
            if r_ == 1:
                ax.set_xlabel("guidance weight $w$", fontsize=6.2, labelpad=1)
            print(f"{ds} step {step:2d} t={s['t']:.3f}: matched {wv:.2f}, best w {x[j]:.2f}, "
                  f"mean log p at D0 {m[0]:.4f} D1 {m[i1]:.4f} matched {m[iv] if wv <= 6 else float('nan'):.4f} best {m[j]:.4f}")
    hs = [Line2D([], [], marker="*", ls="none", ms=7, color=F.ACC, mec="white", mew=0.5),
          Line2D([], [], marker="o", ls="none", ms=4, color=F.COND, mec="white", mew=0.6),
          Line2D([], [], marker="o", ls="none", ms=4, color=F.UNCOND, mec="white", mew=0.6),
          Line2D([], [], marker="o", ls="none", ms=5, mfc="none", mec=F.INK, mew=0.7)]
    fig.legend(hs, [r"variance-matched $w^\star_{\mathrm{VM}}(t)$", "$D_1$ ($w=1$)", "$D_0$ ($w=0$)", "most likely $w$"],
               loc="lower center", ncol=4, frameon=False, fontsize=6.2, bbox_to_anchor=(0.55, 0.0), handletextpad=0.3)
    fig.savefig(out + "-steps.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + "-steps.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}-steps.pdf/.png")


def sweep(res, ds, kind):
    """RES/<ds>-sweep-<kind>.json (sweep --group <kind>), else that kind's arms of RES/<ds>-sweep.json (--group all)."""
    p = Path(res) / f"{ds}-sweep-{kind}.json"
    if p.exists():
        return json.load(open(p))
    R = json.load(open(Path(res) / f"{ds}-sweep.json"))
    return dict(R, arms=[r for r in R["arms"] if r["kind"] == kind])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt = F.rc(7, 6)
    steps_figure(a.res, a.out, plt)
    fig, axs = plt.subplots(2, 2, figsize=(F.W_COL, 2.9))
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.12, top=0.9, wspace=0.75, hspace=0.62)
    for j, (ds, name) in enumerate(COLS):
        sc, co = sweep(a.res, ds, "scaled"), sweep(a.res, ds, "const")
        ref = next(r for r in sc["arms"] if r["value"] == 1.0)
        for i, (R, xlab) in enumerate([(sc, r"scale $\lambda$ of the matched schedule"), (co, "constant weight $w$")]):
            ax = axs[i, j]
            arms = [r for r in R["arms"] if not (R is co and r["value"] == 0.0)]   # w = 0 ignores the label
            x = np.array([r["value"] for r in arms])
            ll = np.array([r["mean_cond"] for r in arms])
            sh = np.array([100 * r["share_cond"]["0.99"] for r in arms])
            if R is co:                                                    # w = 1 is lam = 0 of the scaled family
                r0 = next(r for r in sc["arms"] if r["value"] == 0.0)
                x, ll = np.r_[1.0, x], np.r_[r0["mean_cond"], ll]
                sh = np.r_[100 * r0["share_cond"]["0.99"], sh]
            o = np.argsort(x)
            x, ll, sh = x[o], ll[o], sh[o]
            ax2 = ax.twinx()
            ax2.plot(x, sh, color=SH, lw=0.9, alpha=0.55, marker="s", ms=2, zorder=2)
            ax2.tick_params(colors=F.MUTED, labelcolor=SH, labelsize=5.6, width=0.5, length=2)
            for s in ("top", "left", "bottom"):
                ax2.spines[s].set_visible(False)
            ax2.spines["right"].set_color(F.MUTED); ax2.spines["right"].set_linewidth(0.6)
            if j == 1:
                ax2.set_ylabel("% above data's\n99% line", color=SH, fontsize=6, labelpad=2)
            ax.plot(x, ll, color=LL, lw=1.1, marker="o", ms=2.4, mec="white", mew=0.3, zorder=3)
            ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
            if R is sc:
                ax.plot([1.0], [ref["mean_cond"]], marker="*", ms=7.5, color=F.INK, zorder=5, ls="none")
            else:
                ax.axhline(ref["mean_cond"], color=F.INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
            k = int(np.argmax(ll))
            ax.plot([x[k]], [ll[k]], marker="o", ms=4.5, mfc="none", mec=LL, mew=0.8, zorder=4)
            F.style(ax)
            ax.tick_params(labelsize=5.8, pad=1.5)
            ax.set_xlabel(xlab, labelpad=1, fontsize=6.3)
            if j == 0:
                ax.set_ylabel(r"mean $\log p(x|c)$ / $d$", color=LL, labelpad=1, fontsize=6.3)
            pad = 0.15 * (ll.max() - ll.min() + 1e-4)
            ax.set_ylim(ll.min() - pad, ll.max() + pad)
            if i == 0:
                ax.text(0, 1.1, f"({'ab'[j]}) {name}", transform=ax.transAxes, fontsize=6.8, color=F.INK)
            print(f"{ds} {'scaled' if R is sc else 'const '}: best {x[k]:.2f} -> {ll[k]:.4f}; matched "
                  f"{ref['mean_cond']:.4f}; no guidance {ll[np.argmin(np.abs(x - (0 if R is sc else 1)))]:.4f}; "
                  f"data {sc['real_mean_cond']:.4f}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}.pdf/.png")


if __name__ == "__main__":
    main()
