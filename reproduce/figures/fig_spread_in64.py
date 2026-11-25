#!/usr/bin/env python
"""The 2-D story figure's "Spread size" panel, redone on ImageNet-64 (reproduce/experiments/spread_in64.py).

    python reproduce/figures/fig_spread_in64.py --dir RUNS/spread_in64 --out paper-figures/spread-in64 [--sigma 0.5]

Writes <out>.pdf/png, one panel shaped like the 2-D one at --sigma, and <out>-sigmas.pdf/png, the same panel at
every noise level. y is tr Cov over retrained models of D(x_t) over tr Sigma(x_t), the median over held-out probes;
the band is a 5-95% bootstrap over models, as the 2-D panel's 90% band. The 2-D panel's dashed bootstrap line,
1/(kappa N), comes from a delta-method expansion that needs many training points to share the posterior weight. On
ImageNet-64 about one point carries it (effective number ~1 up to sigma = 2, in the analysis summary), so kappa has
no meaningful value here and the dashed line is only a 1/N slope through the largest N. Reads DIR/analysis.npz and
DIR/config.json, written by the experiment's analyze stage.
"""
import argparse
import json
from pathlib import Path

import numpy as np

INK, GRID = "#1f1f1e", "#e4e3dc"
PURPLE, EXACT = "#7d4fb3", "#bcc4cf"


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.7,
        "pdf.fonttype": 42})
    return plt


def load(d, kimg):
    res = dict(np.load(Path(d) / "analysis.npz"))
    cfg = json.loads((Path(d) / "config.json").read_text())
    e = f"ema{cfg['ema'][0]:.3f}"
    ns = np.asarray(cfg["ns"])
    ks = sorted({int(k.rsplit("_k", 1)[1]) for k in res if k.startswith("trained_")})
    common = [k for k in ks if all(f"trained_n{n}_{e}_k{k}" in res for n in ns)]
    k = kimg or common[-1]
    return res, cfg, e, ns, k


def nfmt(n):
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else f"{n:,}"


def panel(ax, res, cfg, e, ns, k, l):
    tr = np.array([res[f"trained_n{n}_{e}_k{k}"][:, l] for n in ns])          # [N, (point, lo, hi)]
    ex = np.array([res[f"exact_n{n}"][l] for n in ns])
    ax.plot(ns, ex, color=EXACT, lw=5, solid_capstyle="round", zorder=1)
    ax.plot(ns, tr[-1, 0] * ns[-1] / ns, color=INK, ls=(0, (4, 2)), lw=0.9, zorder=2)
    ax.fill_between(ns, tr[:, 1], tr[:, 2], color=PURPLE, alpha=0.22, lw=0, zorder=3)
    ax.plot(ns, tr[:, 0], color=PURPLE, lw=1.6, marker="o", ms=3.8, zorder=4)
    for n2, _, _ in cfg["seed_only"]:
        so = res.get(f"seedonly_n{n2}_{e}_k{k}")
        if so is not None:
            ax.plot([n2 * 1.18], [so[0, l]], ls="none", marker="D", ms=4, mfc="white", mec=PURPLE, mew=1.1, zorder=5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(ns[0] / 1.6, ns[-1] * 1.6)
    ax.set_xticks(ns, [nfmt(n) for n in ns])
    ax.minorticks_off()
    ax.set_yticks([10.0 ** p for p in range(-4, 3)])
    ax.set_ylim(10 ** np.floor(np.log10(tr[:, 1].min()) - 0.15), 10 ** np.ceil(np.log10(ex.max()) + 0.15))
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend(fig, plt, ncol, y):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [(Line2D([], [], color=PURPLE, lw=1.6, marker="o", ms=3.8), "Adam, own subset and seed"),
         (Line2D([], [], ls="none", marker="D", ms=4, mfc="white", mec=PURPLE, mew=1.1), "Same subset, other seeds"),
         (Line2D([], [], color=EXACT, lw=5), "Exact posterior mean of $N$ points"),
         (Line2D([], [], color=INK, ls=(0, (4, 2)), lw=0.9), r"$\propto 1/N$"),
         (Patch(color=PURPLE, alpha=0.22, lw=0), "5-95% over models")]
    fig.legend(*zip(*h), loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False, handlelength=2.2,
               columnspacing=1.2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="the experiment's output directory (analysis.npz, config.json)")
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--kimg", type=int, help="training length; default the longest every N has reached")
    a = ap.parse_args()
    plt = setup()
    res, cfg, e, ns, k = load(a.dir, a.kimg)
    sig = res["sigmas"]
    l = int(np.argmin(np.abs(sig - a.sigma)))
    mi = k * 1000 / 2 ** 20
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(2.35, 2.95))
    fig.subplots_adjust(left=0.24, right=0.96, top=0.89, bottom=0.42)
    panel(ax, res, cfg, e, ns, k, l)
    ax.set_title("(c) Spread size", loc="left", fontweight="bold")
    ax.text(0.04, 0.04, f"ImageNet-64, $\\sigma={sig[l]:g}$\n{mi:.3g} Mi training images", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.5, color="#6b6b66")
    ax.set_xlabel("Training-set size $N$")
    ax.set_ylabel(r"Spread / tr $\Sigma$")
    legend(fig, plt, 1, 0.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, len(sig), figsize=(1.45 * len(sig), 2.25))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.33, wspace=0.42)
    for i, (ax, s) in enumerate(zip(axs, sig)):
        panel(ax, res, cfg, e, ns, k, i)
        ax.set_title(f"$\\sigma={s:g}$", fontsize=8.5)
        ax.set_xlabel("$N$")
    axs[0].set_ylabel(r"Spread / tr $\Sigma$")
    fig.suptitle(f"Spread size on ImageNet-64, every noise level ({mi:.3g} Mi training images, EMA {e[3:]})",
                 fontsize=9, fontweight="bold", x=0.06, ha="left")
    legend(fig, plt, 5, 0.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}-sigmas.{ext}", dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf and {out}-sigmas.pdf ({mi:.3g} Mi, sigma {sig[l]:g})")


if __name__ == "__main__":
    main()
