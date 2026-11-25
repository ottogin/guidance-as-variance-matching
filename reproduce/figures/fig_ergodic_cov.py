#!/usr/bin/env python
"""Eq. (16), first line, on the 2-D mixture: one guided pair's covariance over inputs vs the covariance over
inputs and retrainings. Reads ergodic.npz from reproduce/toys/ergodic_cov.py.

    python reproduce/figures/fig_ergodic_cov.py --npz DIR/ergodic.npz --out fig_ergodic_cov.pdf

The default --pairs need pools N = 128, 512, 2048 and 8192 (ergodic_cov.py train --out DIR --n N for each, then
measure --out DIR); fig:ergodic-gap trains only 128 and 2048, which give --pairs 128-2048.

(a) The share of the pooled excess covariance, tr Cov_{p,x}(D_w) - tr Cov(E[x0|x_t]), that one pair sees over x_t,
    at w = w_pool: mean over pairs / pooled (the first line), and the 25-75 % band of single pairs, per (N0, N1).
(b, c) The weights per noise level: w_pool (matches the pooled covariance), w_post (retraining spread = posterior
    spread, eq. 15's c = 1) and w_pair (VMG on one pair, eq. 17): median and 5-95 % band over pairs.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
INK, MUTED = "#0b0b0b", "#8a8984"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default="128-2048,512-8192")
    a = ap.parse_args()
    z = np.load(a.npz)
    sig = z["sigmas"]
    pairs = a.pairs.split(",")
    sd = float(np.sqrt(z[f"{pairs[0]}/tr_x0"].mean() / 2))
    s = sig / sd
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "lines.linewidth": 1.6})
    fig, ax = plt.subplots(1, 1 + len(pairs), figsize=(2.6 * (1 + len(pairs)), 2.3), constrained_layout=True)
    for i, p in enumerate(pairs):
        r = z[f"{p}/at_pool/excess_l"] / z[f"{p}/at_pool/excess_r"][:, None]
        lo, hi = np.nanpercentile(r, [25, 75], axis=1)
        md = z[f"{p}/at_pool/excess_l"].mean(1) / z[f"{p}/at_pool/excess_r"]
        ax[0].fill_between(s, lo, hi, color=C[i], alpha=0.18, lw=0)
        ax[0].plot(s, md, color=C[i], marker="o", ms=3, label=f"$N_0,N_1$ = {p.replace('-', ', ')}")
    ax[0].axhline(1, color=MUTED, lw=0.8, ls=":")
    ax[0].set(xscale="log", xlabel=r"$\sigma/\sigma_{\rm data}$", ylabel="one pair / pooled (excess cov.)",
              ylim=(0, 1.15), title="(a) eq. 16, first line")
    ax[0].legend(frameon=False, fontsize=7, loc="lower left")
    for i, p in enumerate(pairs):
        A = ax[1 + i]
        w = z[f"{p}/w_pair"]
        lo, md, hi = np.nanpercentile(w, [5, 50, 95], axis=1)
        A.fill_between(s, lo, hi, color=C[0], alpha=0.18, lw=0)
        A.plot(s, md, color=C[0], marker="o", ms=3, label=r"$w_{\rm pair}$ (VMG, one pair)")
        A.plot(s, z[f"{p}/w_pool"], color=C[1], label=r"$w_{\rm pool}$ (pooled cov.)")
        A.plot(s, z[f"{p}/w_post"], color=C[2], ls="--", label=r"$w_{\rm post}$ ($c(w)=1$)")
        A.set(xscale="log", yscale="log", xlabel=r"$\sigma/\sigma_{\rm data}$", ylabel="$w$",
              title=f"({'bc'[i]}) $N_0,N_1$ = {p.replace('-', ', ')}")
        if i == 0:
            A.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(a.out, dpi=200)
    print(a.out)


if __name__ == "__main__":
    main()
