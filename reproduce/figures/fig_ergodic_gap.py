#!/usr/bin/env python
"""Cov_{p,x_t}(D) - Cov_{x_t}(D^g), averaged over nets g, for the weak and the strong pool in one plot.

    python reproduce/figures/fig_ergodic_gap.py --npz DIR/ergodic-mat.npz --ns 128,2048 --out fig.png

The averaged difference equals tr Cov_g(mean_x D^g) >= 0. (a) as is, with tr Cov(x0)/N, the sigma -> inf limit, as
dotted guides; (b) divided by tr Cov_{p,x_t}(D).
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#2a78d6", "#eb6834"]
INK, MUTED = "#0b0b0b", "#8a8984"


def paper(z, ns, out):
    """The paper's single-column version of panel (a): reproduce/style.py look, PDF."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from reproduce import style as F
    plt = F.rc()
    Cx0 = z[f"{ns[0]}/Cx0"].mean(0)
    s = z["sigmas"] / float(np.sqrt(np.trace(Cx0) / 2))
    fig, A = plt.subplots(figsize=(F.W_COL, 2.0), constrained_layout=True)
    F.style(A)
    for n, col, lab in zip(ns, (F.COND, F.ACC), ("weak", "strong")):
        gap = np.trace(z[f"{n}/Cpool"] - z[f"{n}/C"].mean(1), axis1=1, axis2=2)
        A.axhline(np.trace(Cx0) / n, color=col, lw=0.8, ls=":")
        A.plot(s, gap, color=col, lw=1.2, marker="o", ms=2.8, label=f"$N={n:,}$ ({lab})".replace(",", "{,}"))
    A.plot([], [], color=F.MUTED, lw=0.8, ls=":", label=r"$\mathrm{tr}\,\mathrm{Cov}(x_0)/N$")
    A.set(xscale="log", yscale="log", xlabel=r"$\sigma/\sigma_{\mathrm{data}}$",
          ylabel=r"$\mathrm{tr}\,\mathrm{Cov}_{p,x_t}(D)-\mathrm{tr}\,\mathrm{Cov}_{x_t}(D)$")
    A.legend(frameon=False, loc="upper left", handlelength=1.6)
    fig.savefig(out)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ns", default="128,2048")
    ap.add_argument("--out", required=True)
    ap.add_argument("--paper", action="store_true", help="single-column paper PDF of panel (a)")
    a = ap.parse_args()
    if a.paper:
        return paper(np.load(a.npz), [int(v) for v in a.ns.split(",")], a.out)
    z = np.load(a.npz)
    ns = [int(v) for v in a.ns.split(",")]
    Cx0 = z[f"{ns[0]}/Cx0"].mean(0)
    sd = float(np.sqrt(np.trace(Cx0) / 2))
    s = z["sigmas"] / sd
    lx0 = np.linalg.eigvalsh(Cx0)
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "lines.linewidth": 1.6})
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0), constrained_layout=True)
    for i, n in enumerate(ns):
        lab = f"$N={n}$" + (" ($D_0$)" if i == 0 else " ($D_1$)")
        gap = np.trace(z[f"{n}/Cpool"] - z[f"{n}/C"].mean(1), axis1=1, axis2=2)          # [S]
        ax[0].plot(s, gap, color=C[i], marker="o", ms=3, label=lab)
        ax[0].axhline(lx0.sum() / n, color=C[i], lw=0.7, ls=":", alpha=0.8)
        ax[1].plot(s, gap / np.trace(z[f"{n}/Cpool"], axis1=1, axis2=2), color=C[i], marker="o", ms=3, label=lab)
    ax[0].plot([], [], color=MUTED, ls=":", lw=0.7, label=r"$\mathrm{tr\,Cov}(x_0)/N$")
    ax[0].set(ylabel=r"$\mathrm{tr}\,\mathrm{Cov}_{p,x_t}(D)-\mathrm{tr}\,\mathrm{Cov}_{x_t}(D^g)$",
              title="(a) gap")
    ax[1].set(ylabel=r"$\dfrac{\mathrm{tr}\,\mathrm{Cov}_{p,x_t}(D)-\mathrm{tr}\,\mathrm{Cov}_{x_t}(D^g)}{\mathrm{tr}\,\mathrm{Cov}_{p,x_t}(D)}$",
              title=r"(b) gap / $\mathrm{tr}\,\mathrm{Cov}_{p,x_t}(D)$")
    for A in ax:
        A.set(xscale="log", yscale="log", xlabel=r"$\sigma/\sigma_{\rm data}$")
        A.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(a.out, dpi=200)
    print("sigma/sd", np.array2string(s, precision=3, max_line_width=250))


if __name__ == "__main__":
    main()
