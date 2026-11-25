#!/usr/bin/env python
"""fig:vgap: SD 3.5 Medium's E||v_c - v_u||^2 per sampling step, against the guidance interval of Kynkaanniemi
et al. (2024).

    python reproduce/figures/fig_vgap.py --out paper-figures/vgap-interval

From the calibration moments of compute_w.py (runs/sd35m/coco/moments/): with v = (x_t - D) / s at flow time s,
E||v_c - v_u||^2 / d = E||D_1 - D_0||^2 / (d s^2), which is the coefficient b of eq. wconstant over s^2. The right
axis divides by E||v*||^2 / d = 1 + E||x0||^2 / d, the energy of the target velocity v* = eps - x0. Their SD-XL
interval sigma in (0.28, 5.42] is in the variance-exploding level the paper calls t; SD 3.5 M's flow time s maps
to it by t = s / (1 - s), so the interval is guided steps 15-36 of 40. The top axis shows that t.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

INK, MUTED, GRID, BASE = "#1f1f1e", "#6b6a63", "#e4e3dc", "#c3c2b7"
GREEN, WARM = "#1baf7a", "#eb6834"
SIG_HI, SIG_LO = 5.42, 0.28              # SD-XL guidance interval, Kynkaanniemi et al. (2024), EDM sigma


def from_moments(model="sd35m", dataset="coco"):
    """(flow time s [S], E||v_c - v_u||^2 / d [S], E||v*||^2 / d) from the stored moments pass."""
    from vmg import moments as M, runs
    sums, meta = M.load(runs.moments_shards(model, dataset))
    m = M.reduce(sums, meta)
    sigma = np.asarray(m.sigma, float)
    s = np.where(np.isfinite(sigma), sigma / (1 + np.where(np.isfinite(sigma), sigma, 0)), 1.0)
    b = m.rr[:, 2] - 2 * m.rr[:, 1] + m.rr[:, 0]
    ex0 = float(sums["q_xx"].sum() / sums["cnt"].sum() / m.d)
    return s, b / s ** 2, 1 + ex0


def plot(t, gap, ref, out, col=True):
    # font sizes: annotation, small annotation, axis label, top-axis label, ticks
    fa, fs, fl, ft, fk = (6.3, 6.0, 7.3, 7.0, 6.6) if col else (7.5, 7, 8.5, 8, 7.5)
    L = gap
    step = np.arange(1, len(L) + 1)
    with np.errstate(divide="ignore"):
        sig = t / (1 - t)
    on = (sig > SIG_LO) & (sig <= SIG_HI)
    first, last = step[on][0], step[on][-1]
    assert on[first - 1:last].all() and (first, last) == (15, 36), (first, last)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                                "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.3, 2.15) if col else (5.5, 2.7))
    x0, x1 = 0.5, len(L) + 0.5
    ax.axvspan(x0, first - 0.5, color=WARM, alpha=0.09, lw=0, zorder=0)
    ax.axvspan(last + 0.5, x1, color=MUTED, alpha=0.12, lw=0, zorder=0)
    for xb in (first - 0.5, last + 0.5):
        ax.axvline(xb, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.plot(step, gap, color=GREEN, lw=1.3 if col else 1.6, zorder=3)
    ax.scatter(step, gap, s=7, color=GREEN, lw=0, zorder=4)
    ax.set_yscale("log")
    ax.set_xlim(x0, x1)
    lo, hi = gap.min() / 2.5, gap.max() * 4.5
    ax.set_ylim(lo, hi)
    ytxt = hi / 1.35
    kw = dict(fontsize=fa, ha="center", va="top", linespacing=1.1)
    ax.text((x0 + first - 0.5) / 2, ytxt, "guidance harmful\n(high noise)", color="#b04a1e", **kw)
    sep = "\n" if col else ", "
    ax.text((first + last) / 2, ytxt, f"guidance interval{sep}$t\\in({SIG_LO:g}, {SIG_HI:g}]$\n"
            "(Kynkäänniemi et al., 2024)", color=INK, **kw)
    if col:     # four steps wide: too narrow for horizontal text
        ax.text((last + 0.5 + x1) / 2, ytxt, "largely unneeded", color=MUTED, rotation=90,
                **dict(kw, fontsize=fs))
    else:
        ax.text((last + 0.5 + x1) / 2, ytxt, "largely\nunneeded", color=MUTED, **dict(kw, fontsize=fs))

    ax.set_xticks([1, 10, first, 20, 30, last, 40])
    ax.set_xlabel("sampling step", fontsize=fl, color=INK, labelpad=2 if col else 4)
    ax.set_ylabel("$\\mathbb{E}\\,\\|v_c-v_u\\|^2$ per coordinate", fontsize=fl, color=INK)
    top = ax.secondary_xaxis("top")
    top.set_xticks([1, first, 25, last, 40])
    top.set_xticklabels(["$\\infty$" if not np.isfinite(sig[s - 1]) else f"{sig[s - 1]:.2g}"
                         for s in [1, first, 25, last, 40]])
    top.set_xlabel("noise level $t$", fontsize=ft, color=INK, labelpad=2 if col else 3)
    top.tick_params(labelsize=fk, colors=INK, length=2.5, width=0.6, color=BASE, pad=1.5 if col else 3.5)
    top.spines["top"].set_color(BASE)
    right = ax.secondary_yaxis("right", functions=(lambda y: 100 * y / ref, lambda p: p * ref / 100))
    right.set_ylabel("% of $\\mathbb{E}\\,\\|v^\\star\\|^2$", fontsize=fl, color=INK)
    right.tick_params(labelsize=fk, colors=INK, length=2.5, width=0.6, color=BASE, pad=1.5 if col else 3.5)
    right.spines["right"].set_color(BASE)
    from matplotlib.ticker import FuncFormatter
    right.yaxis.set_major_formatter(FuncFormatter(lambda p, _: f"{p:g}%"))
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(labelsize=fk, colors=INK, length=2.5, width=0.6, color=BASE, pad=1.5 if col else 3.5)
    fig.tight_layout(pad=0.25 if col else 0.4)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, facecolor="white")
    return step, first, last, on


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    a = ap.parse_args()
    t, gap, ref = from_moments()
    step, first, last, on = plot(t, gap, ref, a.out)
    for name, m in (("harmful", step < first), ("interval", on), ("unneeded", step > last)):
        g = gap[m]
        print(f"{name:9s} steps {step[m][0]:2d}-{step[m][-1]:2d}  E|dv|^2 {g.min():.2e}..{g.max():.2e}  "
              f"= {100 * g.min() / ref:.3f}..{100 * g.max() / ref:.2f}% of E|v*|^2  share of summed gap "
              f"{100 * g.sum() / gap.sum():.1f}%")
    print(f"wrote {a.out}.pdf")


if __name__ == "__main__":
    main()
