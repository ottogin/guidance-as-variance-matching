#!/usr/bin/env python
"""tr[Cov_{x_t}(D^g) - Cov_{p,x_t}(D)] per pair g, for D0, D1 and D_w at w = w_pool, on the 2-D toy.

    python reproduce/figures/fig_ergodic_diff.py --npz DIR/ergodic.npz --pair 128-2048 --out fig.png

The mean over pairs is exactly -tr Cov_g(E_x[D^g]) (the input-averaged prediction's spread over retrainings).
(a) -mean difference against sigma, with E tr Cov(x0|x_t) (what guidance must add) and tr Cov(x0)/N (the spread
    of an N-point sample mean, the sigma -> inf limit of the offset) for reference.
(b) the per-pair difference over E tr Cov(x0|x_t): mean, 25-75 % and 5-95 % bands over pairs.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED = "#0b0b0b", "#8a8984"


def diffs(z, p):
    B = z[f"{p}/B"][0]
    t00, t01, t11 = (z[f"{p}/{k}"] * (B - 1) / B for k in ("t00", "t01", "t11"))       # [S,G] population
    P = z[f"{p}/pooled"]                                                               # [S,3]
    w = z[f"{p}/w_pool"][:, None]
    tw = (1 - w) ** 2 * t00 + 2 * w * (1 - w) * t01 + w * w * t11
    Pw = (1 - w[:, 0]) ** 2 * P[:, 0] + 2 * w[:, 0] * (1 - w[:, 0]) * P[:, 1] + w[:, 0] ** 2 * P[:, 2]
    return {"$D_0$": t00 - P[:, :1], "$D_1$": t11 - P[:, 2:], r"$D_{w_{\rm pool}}$": tw - Pw[:, None]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pair", default="128-2048")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.npz)
    p = a.pair
    n0, n1 = (int(v) for v in p.split("-"))
    trx0 = z[f"{p}/tr_x0"]
    sd = float(np.sqrt(trx0.mean() / 2))
    s = z["sigmas"] / sd
    post = z[f"{p}/tr_post"]
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "lines.linewidth": 1.6})
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7), constrained_layout=True)
    d = diffs(z, p)
    for i, (k, v) in enumerate(d.items()):
        ax[0].plot(s, -v.mean(1), color=C[i], marker="o", ms=3, label=k)
    ax[0].plot(s, post, color=INK, lw=1.0, label=r"$\mathbb{E}\,\mathrm{tr\,Cov}(x_0|x_t)$")
    for n, ls in ((n0, "--"), (n1, ":")):
        ax[0].axhline(trx0.mean() / n, color=MUTED, lw=0.9, ls=ls)
        ax[0].text(s[0], trx0.mean() / n * 1.25, f"tr Cov$(x_0)/{n}$", color=MUTED, fontsize=6.5)
    ax[0].set(xscale="log", yscale="log", ylim=(1e-6, 5), xlabel=r"$\sigma/\sigma_{\rm data}$",
              ylabel=r"$-\,\mathrm{tr}[\mathrm{Cov}_{x_t}(D)-\mathrm{Cov}_{p,x_t}(D)]$, mean over pairs",
              title=f"(a) $N_0,N_1$ = {n0}, {n1}")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    v = d[r"$D_{w_{\rm pool}}$"] / post[:, None]
    q5, q25, q75, q95 = np.percentile(v, [5, 25, 75, 95], axis=1)
    ax[1].fill_between(s, q5, q95, color=C[2], alpha=0.12, lw=0, label="5-95 % of pairs")
    ax[1].fill_between(s, q25, q75, color=C[2], alpha=0.3, lw=0, label="25-75 %")
    ax[1].plot(s, v.mean(1), color=C[2], marker="o", ms=3, label="mean")
    ax[1].axhline(0, color=MUTED, lw=0.8, ls=":")
    ax[1].axhline(-1, color=INK, lw=0.8, ls=":")
    ax[1].set(xscale="log", ylim=(-1.6, 1.6), xlabel=r"$\sigma/\sigma_{\rm data}$",
              ylabel=r"$\mathrm{tr}[\mathrm{Cov}_{x_t}(D_w)-\mathrm{Cov}_{p,x_t}(D_w)]\;/\;\mathbb{E}\,\mathrm{tr\,Cov}(x_0|x_t)$",
              title=r"(b) per pair, $w=w_{\rm pool}$")
    ax[1].yaxis.label.set_size(7)
    ax[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(a.out, dpi=200)
    for k, v in d.items():
        print(k, "mean diff:", np.array2string(v.mean(1), precision=4, max_line_width=200))
    print("E tr Cov(x0|xt):", np.array2string(post, precision=4, max_line_width=200))
    print("sigma/sd:", np.array2string(s, precision=3, max_line_width=200))


if __name__ == "__main__":
    main()
