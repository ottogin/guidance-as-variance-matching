#!/usr/bin/env python
"""Pairs of guided trajectories of one class: corr(Delta(x), Delta(x')) against |x - x'|, one panel per noise level.

    python reproduce/figures/fig_errfield_pairs.py DIR [DIR2 ...] [--tag pairs-c1] [--x now|final]

DIR is an output folder of `errfield_edm2.py pairs --classes 1 --tag pairs-c1` (pass the same --tag here; the
pair subsample sub_r / sub_raw / sub_cen at every
sampler step). Delta = D1 - D0, the two denoisers' x0 predictions; at a fixed sigma its correlation is the same as that
of the eps predictions' difference (-Delta / sigma). One panel per DIR, one curve per sampler step (colour = sigma):
the mean correlation of the pairs in 20 equal-count distance bins, distance in units of the median nearest-image
distance. --x final puts the distance between the two trajectories' final images on x instead.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tag", default="pairs-c1")
    ap.add_argument("--field", default="delta", choices=("delta", "d1", "d0", "dw"),
                    help="delta = D1 - D0; d1, d0 = each denoiser's x0 prediction; dw = the guided one (needs --denoised)")
    ap.add_argument("--levels", help="comma list of step indices: one panel per sigma, the DIRs overlaid")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--kind", default="raw", choices=("raw", "cen"))
    ap.add_argument("--x", default="now", choices=("now", "final"),
                    help="now: |x - x'| at that step; final: distance between the two trajectories' final images")
    ap.add_argument("--out")
    a = ap.parse_args()
    runs = []
    for d in a.dirs:
        D = Path(d)
        runs.append((np.load(D / f"{a.tag}.npz"), json.loads((D / f"{a.tag}-meta.json").read_text()), D.name))
    if a.levels:
        return panels(a, runs)
    fig, axs = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 3.8), sharey=True, constrained_layout=True,
                            squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, (z, meta, name) in zip(axs[0], runs):
        sig = z["sigma"]
        ls = np.log(sig)
        for k in range(len(sig)):
            r = z["sub_r"][k] if a.x == "now" else z["sub_rfinal"]
            c = z[f"sub_{a.kind}" if a.field == "delta" else f"sub_{a.field}_{a.kind}"][k]
            edges = np.quantile(r, np.linspace(0, 1, a.bins + 1))
            b = np.clip(np.digitize(r, edges) - 1, 0, a.bins - 1)
            xm = np.array([r[b == j].mean() for j in range(a.bins)])
            ym = np.array([c[b == j].mean() for j in range(a.bins)])
            ax.plot(xm, ym, "-", lw=1.4, color=cmap((ls[k] - ls.min()) / (ls.max() - ls.min())))
        ax.axhline(0, color="#bbbbbb", lw=0.6)
        if a.x == "now":
            ax.set_xscale("log")
        cls = np.unique(z["cls"])
        ax.set_title(f"{name}: D1 = {meta['d1']}, D0 = {meta['d0']}, class {cls[0] if len(cls) == 1 else 'mixed'}, "
                     f"{meta['n']} trajectories", fontsize=9)
        ax.set_xlabel("|x − x′| at that step (nearest-image units)" if a.x == "now"
                      else "distance between the two final images (nearest-image units)", fontsize=9)
    sym = dict(delta="Δ", d1="D₁", d0="D₀", dw="D_w")[a.field]
    axs[0, 0].set_ylabel(f"corr({sym}(x), {sym}(x′)){'' if a.kind == 'raw' else ', centred'}, mean over pairs", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.LogNorm(sig.min(), sig.max()))
    fig.colorbar(sm, ax=axs[0].tolist(), label="σ", pad=0.01)
    out = a.out or str(Path(a.dirs[0]) / f"{a.tag}-{a.field}-corr-vs-{'distance' if a.x == 'now' else 'final'}-{a.kind}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)
    for z, meta, name in runs:
        print(name)
        for k in range(len(z["sigma"])):
            r, c = z["sub_r"][k], z[f"sub_{a.kind}"][k]
            q = np.quantile(r, [0.1, 0.5, 0.9])
            lo, hi = c[r <= q[0]].mean(), c[r >= q[2]].mean()
            print(f"  sigma {z['sigma'][k]:8.4f}  r 10/50/90 % {q[0]:.3g} {q[1]:.3g} {q[2]:.3g}  corr mean "
                  f"{c.mean():.4f}  closest 10 % {lo:.4f}  farthest 10 % {hi:.4f}  spearman "
                  f"{np.corrcoef(np.argsort(np.argsort(r)), np.argsort(np.argsort(c)))[0, 1]:+.3f}")


def binned(r, c, bins):
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    b = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    return (np.array([r[b == j].mean() for j in range(bins)]), np.array([c[b == j].mean() for j in range(bins)]),
            np.array([c[b == j].std() / np.sqrt((b == j).sum()) for j in range(bins)]))


def panels(a, runs):
    ks = [int(k) for k in a.levels.split(",")]
    nc = 4
    nr = -(-len(ks) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.3 * nc, 2.9 * nr), constrained_layout=True, squeeze=False)
    sym = dict(delta="Δ", d1="D₁", d0="D₀", dw="D_w")[a.field]
    styles = [dict(color="#2f5fb8", ls="-"), dict(color="#c8562a", ls="--"), dict(color="#3a8a4f", ls=":")]
    for ax, k in zip(axs.flat, ks):
        for (z, meta, name), st in zip(runs, styles):
            r = z["sub_r"][k] if a.x == "now" else z["sub_rfinal"]
            c = z[f"sub_{a.kind}" if a.field == "delta" else f"sub_{a.field}_{a.kind}"][k]
            lab = f"{name}: {sym} = " + {"delta": f"{meta['d1']} − {meta['d0']}", "d1": meta["d1"], "d0": meta["d0"],
                                         "dw": f"{meta['d0']} + {meta['w']:g} ({meta['d1']} − {meta['d0']})"}[a.field]
            # random-x runs: same-image pairs (filled markers) and other-image pairs (open) apart
            groups = ([("same image", z["sub_sameimg"], "o"), ("other image", ~z["sub_sameimg"], "s")]
                      if "sub_sameimg" in z else [("", np.ones(len(r), bool), "o")])
            for gl, gm, mk in groups:
                xm, ym, se = binned(r[gm], c[gm], a.bins)
                ax.plot(xm, ym, marker=mk, ms=2.5, lw=1.5, label=f"{lab}, {gl}" if gl else lab,
                        mfc="white" if gl == "other image" else st["color"], **st)
                ax.fill_between(xm, ym - 2 * se, ym + 2 * se, color=st["color"], alpha=0.2, lw=0)
            if "sub_sameimg" in z and r.max() / r.min() > 20:
                ax.set_xscale("log")
        ax.set_title(f"σ = {runs[0][0]['sigma'][k]:.3g}", fontsize=10)
        ax.set_xlabel("|x − x′| (nearest-image units)" if a.x == "now" else "distance of the final images", fontsize=8)
        ax.tick_params(labelsize=7.5)
    for ax in axs.flat[len(ks):]:
        ax.axis("off")
    for row in axs:
        row[0].set_ylabel(f"corr({sym}(x), {sym}(x′)){'' if a.kind == 'raw' else ', centred'}", fontsize=9)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(runs), fontsize=9, frameon=False)
    out = a.out or str(Path(a.dirs[0]) / f"{a.tag}-{a.field}-panels-{a.kind}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
