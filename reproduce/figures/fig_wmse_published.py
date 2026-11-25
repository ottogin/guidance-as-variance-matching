#!/usr/bin/env python
"""Paper Fig. 2 (fig:wmse-published), one column: MSE-optimal weights against published weights.

    python reproduce/figures/fig_wmse_published.py --out paper-figures/wmse-published [--layout paper] [--allow-missing]

(a) One stick per pair of reproduce/rosters/wmse_pairs.py: the median over noise levels of w*_MSE(t) (dot)
    joined to the published weight (ring), grouped by guidance method; pairs with no published weight are dots
    only. The SD 3.5 Medium COCO pair is outlined in ink.
(b) That pair's w*_MSE(t) at every step of its 40-step Euler sampler, with the 95% bootstrap band and the median.
(c) Its denoising error per step in velocity space, E|v_w - v|^2 / d = E|D_w - x0|^2 / (d s^2) at flow time
    s = sigma / (1 + sigma): at w = 1, at w*_MSE(t) and at the published weight, with 95% bootstrap bands.
Reads runs/<model>/<dataset>/wmse.json (compute_wmse.py). `--layout paper` (default) is the paper's 3.25 in
square: (a) on top, (b) and (c) side by side, (c) without the w*_MSE curve (it lies within 0.4%
of w = 1). `--layout column` stacks (b) and (c) under (a) with their bands and both curves in (c).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.rosters.wmse_pairs import AUTOG, AVM, CFG, HIGHLIGHT, PAIRS, T2I  # noqa: E402
from vmg import runs  # noqa: E402

INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#c3c2b7"
# shown left to right: (group, colour, tick label)
ORDER = [(T2I, "#1baf7a", "text-to-image\nCFG"), (AUTOG, "#2a78d6", "autoguidance"),
         (CFG, "#eb6834", "class CFG"), (AVM, "#e87ba4", "audio, video,\nmotion")]


def load(p):
    path = runs.root() / p.model / p.dataset / "wmse.json"
    return json.load(open(path)) if path.exists() else None


def frame(ax, fs):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.6)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=fs, colors=INK, length=2.2, width=0.5, color=BASE)


def velocity(r):
    """[3, S] arrays (value, 2.5%, 97.5%) of E|v_w - v|^2 / d at w = 1, w*_MSE and the published weight."""
    s = np.asarray(r["sigma"], float)
    s2 = (1 / (1 + 1 / s)) ** 2                               # s = 1 at sigma = inf
    band = lambda k: np.vstack([np.asarray(r[k], float), np.asarray(r[k + "_ci"], float)]) / s2   # noqa: E731
    return band("mse_w1"), band("mse_wmse"), band("mse_published")


def dumbbells(ax, data):
    """(a): one stick per pair, from the median of w*_MSE(t) (dot) to the published weight (ring)."""
    from matplotlib.lines import Line2D
    width, gap, x0, centers = 0.78, 0.22, 0.0, []
    for group, col, _ in ORDER:
        ds = sorted([(p, float(np.nanmedian(r["w_mse"]))) for p, r in data if r is not None and p.group == group],
                    key=lambda d: (d[0].pub if d[0].pub is not None else -1, d[1]))
        k = len(ds)
        w = width * min(1.0, 0.25 + k / 20)                  # narrow groups stay narrow
        xs = x0 + (np.linspace(0, w, k) if k > 1 else np.array([w / 2]))
        for (p, med), x in zip(ds, xs):
            h = p == HIGHLIGHT
            if p.pub is not None:
                ax.plot([x, x], [med, p.pub], color=INK if h else col, lw=0.8 if h else 0.6, alpha=1 if h else 0.6,
                        zorder=5 if h else 2)
                ax.scatter([x], [p.pub], s=11, facecolor="white", edgecolor=INK if h else col,
                           linewidth=1.0 if h else 0.8, zorder=6 if h else 3)
            ax.scatter([x], [med], s=22 if h else 11, color=col, edgecolor=INK if h else "white",
                       linewidth=0.9 if h else 0.4, zorder=7 if h else 4)
            if h:
                ax.hl_xy = (x, med)                          # where the compact layout's callout starts
        centers.append(x0 + w / 2)
        x0 += w + gap
    ax.axhline(1, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(-0.12, x0 - gap + 0.12)
    ax.set_ylim(-0.25, 8.1)
    ax.set_yticks(range(0, 9, 2))
    ax.set_xticks(centers)
    ax.set_xticklabels([g[2] for g in ORDER], fontsize=6.5, color=INK, linespacing=1.05)
    ax.set_ylabel("guidance weight $w$", fontsize=7.2, color=INK, labelpad=2)
    frame(ax, 6.5)
    ax.tick_params(axis="x", length=0, pad=3)
    items = [(r"median over $t$ of $w^\star_{\mathrm{MSE}}(t)$",
              dict(marker="o", ls="none", ms=3.6, mfc=MUTED, mec="white", mew=0.4)),
             ("published $w$", dict(marker="o", ls="none", ms=3.6, mfc="white", mec=MUTED, mew=0.8)),
             ("no guidance ($w=1$)", dict(ls=(0, (4, 3)), lw=0.8, color=INK))]
    ax.legend([Line2D([], [], **kw) for _, kw in items], [t for t, _ in items], loc="lower left",
              bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=6.2, handletextpad=0.25, columnspacing=0.9,
              borderaxespad=0.1, labelcolor=INK)
    ax.set_title("(a)", fontsize=7, color=INK, loc="left", x=-0.13, y=1.0, pad=2)


def column(data, hl):
    """(a), then (b) and (c) stacked on a shared step axis."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3.3, 4.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 2.5], hspace=0.36)
    sub = gs[1].subgridspec(2, 1, height_ratios=[1, 1.15], hspace=0.38)   # (b) and (c) share the step axis
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(sub[0])
    cx = fig.add_subplot(sub[1], sharex=bx)
    dumbbells(ax, data)

    # (b) the expanded pair at every step
    t2i, pub = ORDER[0][1], HIGHLIGHT.pub
    w = np.asarray(hl["w_mse"], float)
    x = np.arange(1, len(w) + 1)
    med = float(np.median(w))
    bx.axhline(1, color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
    bx.axhline(med, color=t2i, lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    bx.fill_between(x, hl["ci_lo"], hl["ci_hi"], color=t2i, alpha=0.25, lw=0, zorder=2)
    bx.plot(x, w, color=t2i, lw=1.0, zorder=3)
    bx.scatter(x, w, s=3, color=t2i, zorder=4, lw=0)
    bx.text(len(x) / 2, med - 0.05, f"median {med:.2f}", fontsize=6.0, color=t2i, ha="center", va="top")
    bx.text(0.99, 1.04, f"$\\uparrow$ published $w={pub:g}$", transform=bx.get_yaxis_transform(), fontsize=6.0,
            color=INK, ha="right", va="bottom")
    bx.set_xlim(0, len(x) + 1)
    bx.set_xticks([1, 10, 20, 30, 40])
    bx.set_ylim(0, 1.25)
    bx.set_yticks([0, 0.5, 1.0])
    bx.set_yticklabels(["0", "0.5", "1"])
    bx.set_ylabel(r"$w^\star_{\mathrm{MSE}}(t)$", fontsize=7.2, color=INK, labelpad=2)
    frame(bx, 6.5)
    bx.tick_params(axis="x", labelbottom=False)
    bx.set_title("(b) the outlined pair: SD 3.5 Medium, COCO, 40 Euler steps", fontsize=6.5, color=INK, loc="left",
                 x=-0.13, pad=3)

    # (c) its denoising error in velocity space
    one, best, pb = velocity(hl)
    cx.fill_between(x, best[1], best[2], color=t2i, alpha=0.25, lw=0, zorder=2)
    cx.plot(x, best[0], color=t2i, lw=1.3, zorder=3)
    cx.plot(x, one[0], color=INK, lw=0.8, ls=(0, (4, 3)), zorder=4)       # on top of w*: they differ by < 0.4%
    cx.fill_between(x, pb[1], pb[2], color=INK, alpha=0.15, lw=0, zorder=2)
    cx.plot(x, pb[0], color=INK, lw=0.8, zorder=4)
    cx.scatter(x, pb[0], s=5, facecolor="white", edgecolor=INK, linewidth=0.6, zorder=5)
    cx.text(9, float(pb[0][8]) * 1.25, f"published $w={pub:g}$", fontsize=6.0, color=INK, ha="left", va="bottom")
    cx.text(20, float(one[0][19]) * 0.82, r"$w=1$ (dashed) and $w^\star_{\mathrm{MSE}}(t)$", fontsize=6.0, color=t2i,
            ha="center", va="top")
    cx.set_yscale("log")
    cx.set_ylim(0.13, 2.6)
    cx.set_yticks([0.2, 0.5, 1, 2])
    cx.set_yticklabels(["0.2", "0.5", "1", "2"])
    cx.minorticks_off()
    cx.set_xlabel("sampling step", fontsize=7.0, color=INK, labelpad=1.5)
    cx.set_ylabel(r"$\mathbb{E}\,\|v_w-v\|^2/d$", fontsize=7.2, color=INK, labelpad=2)
    frame(cx, 6.5)
    cx.set_title("(c) its denoising error in velocity space", fontsize=6.5, color=INK, loc="left", x=-0.13, pad=3)
    fig.align_ylabels([ax, bx, cx])
    return fig


# the paper layout's lower panels
P_INK, P_GREEN, P_SPINE, P_GRID = "#202124", "#19856c", "#666666", "#e3e3e0"


def pframe(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(P_SPINE)
        ax.spines[s].set_linewidth(0.6)
    ax.yaxis.grid(True, color=P_GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5, colors=P_INK, length=2.2, width=0.5, color=P_SPINE)


def paper(data, hl):
    """The paper's 3.25 in square: (a) on top, the outlined pair's (b) and (c) side by side."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(3.25, 3.25))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.62, 1.0], hspace=0.56)
    sub = gs[1].subgridspec(1, 2, wspace=0.42)
    ax, bx, cx = fig.add_subplot(gs[0]), fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    dumbbells(ax, data)
    pub = HIGHLIGHT.pub
    w = np.asarray(hl["w_mse"], float)
    x = np.arange(1, len(w) + 1)
    med = float(np.median(w))

    # (b) MSE-optimal weight
    bx.axhline(1, color=P_SPINE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    bx.axhline(med, color=P_GREEN, lw=0.8, ls=(0, (1, 1.4)), zorder=1)
    bx.fill_between(x, hl["ci_lo"], hl["ci_hi"], color=P_GREEN, alpha=0.2, lw=0, zorder=2)
    bx.plot(x, w, color=P_GREEN, lw=1.0, zorder=3)
    bx.scatter(x, w, s=2.5, color=P_GREEN, zorder=4, lw=0)
    bx.text(0.97, 1.1, f"Pub. $w={pub:g}$ $\\uparrow$", transform=bx.get_yaxis_transform(), fontsize=6.0,
            color=P_INK, ha="right", va="bottom")
    bx.text(len(x) * 0.55, med - 0.12, f"Median {med:.2f}", fontsize=6.0, color=P_GREEN, ha="center", va="top")
    bx.set_xlim(-0.5, len(x) + 1.5)
    bx.set_xticks([1, 20, 40])
    bx.set_ylim(0, 1.25)
    bx.set_yticks([0, 0.5, 1.0])
    bx.set_yticklabels(["0", "0.5", "1"])
    bx.set_ylabel(r"$w^\star_{\mathrm{MSE}}(t)$", fontsize=7.2, color=P_INK, labelpad=1)
    pframe(bx)
    bx.set_title("(b) MSE-optimal weight", fontsize=7, color=P_INK, pad=3)

    # (c) denoising error: the published weight and w = 1 (w*_MSE lies within 0.4% of w = 1)
    one, _, pb = velocity(hl)
    cx.plot(x, pb[0], color=P_INK, lw=1.0, zorder=4)
    cx.scatter(x, pb[0], s=4, facecolor="white", edgecolor=P_INK, linewidth=0.5, zorder=5)
    cx.plot(x, one[0], color=P_GREEN, lw=1.2, zorder=3)
    cx.text(len(x) * 0.72, float(pb[0][int(len(x) * 0.72)]) * 2.3, f"$w={pub:g}$", fontsize=6.5, color=P_INK,
            ha="center", va="bottom")
    cx.text(len(x) * 0.62, float(one[0].min()) * 0.86, "$w=1$", fontsize=6.5, color=P_GREEN, ha="center", va="top")
    cx.set_yscale("log")
    cx.set_ylim(0.13, 2.6)
    cx.set_yticks([0.2, 0.5, 1, 2])
    cx.set_yticklabels(["0.2", "0.5", "1", "2"])
    cx.minorticks_off()
    cx.set_xlim(-0.5, len(x) + 1.5)
    cx.set_xticks([1, 20, 40])
    cx.set_ylabel(r"$\mathbb{E}\|v_w-v\|^2/d$", fontsize=7.2, color=P_INK, labelpad=1)
    pframe(cx)
    cx.set_title("(c) Denoising error", fontsize=7, color=P_INK, pad=3)

    fig.canvas.draw()
    top = max(bx.get_position().y1, cx.get_position().y1)
    fig.text(0.5, top + 0.062, "Outlined pair: SD 3.5 Medium, COCO, 40 Euler steps", fontsize=6.5, color=P_INK,
             ha="center", va="bottom")
    fig.text(0.5, bx.get_position().y0 - 0.062, "Sampling step", fontsize=7, color=P_INK, ha="center", va="top")
    return fig


def compact(data, hl):
    """The paper's Fig. 4: a 3.2 x 2.6 in figure with the legend across the top, (a) under
    it, and the outlined pair's (b) and (c) in a callout box tied to its dot in (a)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(3.2, 2.62))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.78, left=0.115, right=0.975, top=0.905,
                          bottom=0.115)
    sub = gs[1].subgridspec(1, 2, wspace=0.42)
    ax, bx, cx = fig.add_subplot(gs[0]), fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    dumbbells(ax, data)
    ax.get_legend().remove()
    ax.tick_params(labelsize=7.5)
    for t in ax.get_xticklabels():
        t.set_fontsize(7.8)
    ax.set_ylabel("guidance weight $w$", fontsize=8, color=INK, labelpad=2)
    ax.set_title("(a)", fontsize=8, color=INK, loc="left", x=-0.16, y=1.0, pad=-4)
    items = [(r"median over $t$ of $w^\star_{\mathrm{MSE}}(t)$",
              dict(marker="o", ls="none", ms=3.6, mfc=MUTED, mec="white", mew=0.4)),
             ("published $w$", dict(marker="o", ls="none", ms=3.6, mfc="white", mec=MUTED, mew=0.8)),
             ("no guidance ($w=1$)", dict(ls=(0, (4, 3)), lw=0.8, color=INK))]
    fig.legend([Line2D([], [], **kw) for _, kw in items], [t for t, _ in items], loc="upper center",
               bbox_to_anchor=(0.53, 1.0), ncol=3, frameon=False, fontsize=7.4, handletextpad=0.25,
               columnspacing=1.0, borderaxespad=0.0, labelcolor=INK)
    pub = HIGHLIGHT.pub
    w = np.asarray(hl["w_mse"], float)
    x = np.arange(1, len(w) + 1)
    med = float(np.median(w))

    # (b) MSE-optimal weight
    bx.axhline(1, color=P_SPINE, lw=0.8, ls=(0, (4, 3)), zorder=1)
    bx.axhline(med, color=P_GREEN, lw=0.8, ls=(0, (1, 1.4)), zorder=1)
    bx.fill_between(x, hl["ci_lo"], hl["ci_hi"], color=P_GREEN, alpha=0.2, lw=0, zorder=2)
    bx.plot(x, w, color=P_GREEN, lw=1.0, zorder=3)
    bx.scatter(x, w, s=2.5, color=P_GREEN, zorder=4, lw=0)
    bx.text(0.97, 1.04, f"Pub. $w={pub:g}$ $\\uparrow$", transform=bx.get_yaxis_transform(), fontsize=7.4,
            color=P_INK, ha="right", va="bottom")
    bx.text(len(x) * 0.55, med - 0.14, f"Median {med:.2f}", fontsize=7.4, color=P_GREEN, ha="center", va="top")
    bx.set_xlim(-0.5, len(x) + 1.5)
    bx.set_xticks([1, 20, 40])
    bx.set_ylim(0, 1.32)
    bx.set_yticks([0, 0.5, 1.0])
    bx.set_yticklabels(["0", "0.5", "1"])
    bx.set_ylabel(r"$w^\star_{\mathrm{MSE}}(t)$", fontsize=8, color=P_INK, labelpad=1)
    pframe(bx)
    bx.tick_params(labelsize=7.5)
    bx.set_title("(b) MSE-optimal weight", fontsize=8, color=P_INK, pad=4)

    # (c) denoising error: the published weight and w = 1 (w*_MSE lies within 0.4% of w = 1)
    one, _, pb = velocity(hl)
    cx.plot(x, pb[0], color=P_INK, lw=1.0, zorder=4)
    cx.scatter(x, pb[0], s=4, facecolor="white", edgecolor=P_INK, linewidth=0.5, zorder=5)
    cx.plot(x, one[0], color=P_GREEN, lw=1.2, zorder=3)
    cx.text(len(x) * 0.72, float(pb[0][int(len(x) * 0.72)]) * 2.0, f"$w={pub:g}$", fontsize=7.6, color=P_INK,
            ha="center", va="bottom")
    cx.text(len(x) * 0.62, float(one[0].min()) * 0.93, "$w=1$", fontsize=7.6, color=P_GREEN, ha="center", va="top")
    cx.set_yscale("log")
    cx.set_ylim(0.125, 2.6)
    cx.set_yticks([0.2, 0.5, 1, 2])
    cx.set_yticklabels(["0.2", "0.5", "1", "2"])
    cx.minorticks_off()
    cx.set_xlim(-0.5, len(x) + 1.5)
    cx.set_xticks([1, 20, 40])
    cx.set_ylabel(r"$\mathbb{E}\|v_w-v\|^2/d$", fontsize=8, color=P_INK, labelpad=1)
    pframe(cx)
    cx.tick_params(labelsize=7.5)
    cx.set_title("(c) Denoising error", fontsize=8, color=P_INK, pad=3)

    # the callout: a box around (b) and (c), titled on its top edge, tied to the outlined dot in (a)
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    bb = [inv.transform(a.get_tightbbox(fig.canvas.get_renderer())) for a in (bx, cx)]
    x0, x1 = 0.035, 0.995
    y1 = max(b[1][1] for b in bb) + 0.045
    y0 = min(b[0][1] for b in bb) - 0.05
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, transform=fig.transFigure, fill=False, ec="#9a9a9a",
                             lw=0.7, zorder=0))
    box = dict(facecolor="white", edgecolor="none", pad=1.2)
    fig.text(0.5 * (x0 + x1), y1, "Outlined pair: SD 3.5 Medium, COCO, 40 Euler steps", fontsize=7.8, color=P_INK,
             ha="center", va="center", bbox=box)
    fig.text(0.5 * (x0 + x1), y0, "Sampling step", fontsize=8, color=P_INK, ha="center", va="center", bbox=box)
    hx, hy = ax.hl_xy
    dot = inv.transform(ax.transData.transform((hx, hy)))
    axb = inv.transform(ax.transAxes.transform((0, 0)))[1]
    fig.add_artist(Line2D([dot[0], dot[0] - 0.04, x0 + 0.06], [dot[1] - 0.012, axb - 0.03, y1], color="#9a9a9a",
                          lw=0.7, transform=fig.transFigure, zorder=0))
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--layout", choices=("compact", "paper", "column"), default="compact")
    ap.add_argument("--allow-missing", action="store_true", help="draw the pairs that exist (smoke runs)")
    a = ap.parse_args()
    data = [(p, load(p)) for p in PAIRS]
    missing = [f"{p.model}/{p.dataset}" for p, r in data if r is None]
    assert not missing or a.allow_missing, f"no wmse.json for {len(missing)} pairs: {', '.join(missing)}"
    hl = load(HIGHLIGHT)
    assert hl is not None, f"the expanded pair {HIGHLIGHT.model}/{HIGHLIGHT.dataset} is required"
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                                "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = dict(compact=compact, paper=paper, column=column)[a.layout](data, hl)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}.pdf ({len(PAIRS) - len(missing)} of {len(PAIRS)} pairs, {a.layout} layout)")


if __name__ == "__main__":
    main()
