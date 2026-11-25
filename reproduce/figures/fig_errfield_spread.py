#!/usr/bin/env python
"""Random x of one class, partners at every distance: corr(D1(x), D1(x')) and corr(Delta(x), Delta(x')) against
|x - x'|, 2 rows x one column per sigma.

    python reproduce/figures/fig_errfield_spread.py DIR [--tag spread-c850] [--kind raw|cen] [--out PNG]

DIR is an output folder of `errfield_edm2.py spread`: base x = x0 + sigma n of the class, partner
x' = x0 + t (x0j - x0) + sigma (cos(t pi/2) n + sin(t pi/2) m) with t log-uniform (t = 1: an independent random x
of the class). Every (base, partner) pair is a dot, coloured by t; the black line is the mean in equal-count
distance bins. Distance in units of the median nearest-image distance (46.97 px).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def binned(r, c, bins):
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    b = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    return np.array([r[b == j].mean() for j in range(bins)]), np.array([c[b == j].mean() for j in range(bins)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="spread-c850")
    ap.add_argument("--kind", default="raw", choices=("raw", "cen"))
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--out")
    a = ap.parse_args()
    D = Path(a.dir)
    z = np.load(D / f"{a.tag}.npz")
    meta = json.loads((D / f"{a.tag}-meta.json").read_text())
    sig, t = z["sigma"], z["t"].ravel()
    L = len(sig)
    norm = matplotlib.colors.LogNorm(t.min(), 1)
    rows = [("d1", f"corr(D_S(x), D_S(x′))"), ("delta", "corr(Δ(x), Δ(x′)),  Δ = D_S − D_XS")]
    fig, axs = plt.subplots(2, L, figsize=(min(3.2, 21 / L) * L, 5.9), constrained_layout=True, squeeze=False)
    for i, (f, ylab) in enumerate(rows):
        for j in range(L):
            ax = axs[i, j]
            r, c = z["r"][j].ravel(), z[f"{f}_{a.kind}"][j].ravel()
            sc = ax.scatter(r, c, c=t, cmap="viridis", norm=norm, s=1.5, alpha=0.35, lw=0, rasterized=True)
            xm, ym = binned(r, c, a.bins)
            ax.plot(xm, ym, "-", color="black", lw=1.6)
            ax.set_xscale("log")
            ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
            ax.tick_params(labelsize=7.5)
            if i == 0:
                ax.set_title(f"σ = {sig[j]:.3g}", fontsize=10)
            else:
                ax.set_xlabel("|x − x′|  (nearest-image units)", fontsize=8)
            q = np.quantile(r, [0.1, 0.9])
            print(f"{f:5s} sigma {sig[j]:8.4f}  |x-x'| {r.min():.3g}..{r.max():.3g}  corr closest 10 % "
                  f"{c[r <= q[0]].mean():+.3f}  farthest 10 % {c[r >= q[1]].mean():+.3f}  (t > 0.9: "
                  f"{c[t > 0.9].mean():+.3f})")
        axs[i, 0].set_ylabel(ylab + ("" if a.kind == "raw" else ", centred"), fontsize=8.5)
    fig.colorbar(sc, ax=axs.ravel().tolist(), pad=0.01, shrink=0.8,
                 label="t  (t = 1: x′ an independent random x)")
    fig.suptitle(f"EDM2 ImageNet-64, class {meta['cls']}, label fixed: {meta['n_base']} random x × "
                 f"{meta['per_base']} partners each; D_S = {Path(meta['net']).stem}, "
                 f"D_XS = {Path(meta['guide']).stem}", fontsize=9)
    out = a.out or str(D / f"{a.tag}-{a.kind}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
