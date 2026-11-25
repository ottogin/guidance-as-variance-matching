#!/usr/bin/env python
"""Random x of one class: corr(D1(x), D1(x')) and corr(Delta(x), Delta(x')) against |x - x'|, 2 rows x 5 sigmas.

    python reproduce/figures/fig_errfield_random.py DIR [--tag random-c850] [--levels 0,8,14,20,26] [--kind raw]

DIR is an output folder of `errfield_edm2.py random` (x = x0 + sigma n, x0 = the 50 val images of the class, 80
fresh noises each, the class label fixed; D1 = EDM2-S, D0 = EDM2-XS for --pair ag; Delta = D1 - D0). Each panel is
a scatter of 20,000 of the stored pairs (same-image pairs and other-image pairs coloured apart) with the binned
mean of each group on top; distance in units of the median nearest-image distance (46.97 px).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SAME, OTHER = "#2f5fb8", "#c8562a"


def binned(r, c, bins):
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    b = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    return np.array([r[b == j].mean() for j in range(bins)]), np.array([c[b == j].mean() for j in range(bins)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="random-c850")
    ap.add_argument("--levels", default="0,8,14,20,26", help="step indices of the 32 EDM2 levels")
    ap.add_argument("--kind", default="raw", choices=("raw", "cen"))
    ap.add_argument("--n-show", type=int, default=20000)
    ap.add_argument("--bins", type=int, default=12)
    ap.add_argument("--out")
    a = ap.parse_args()
    D = Path(a.dir)
    z = np.load(D / f"{a.tag}.npz")
    meta = json.loads((D / f"{a.tag}-meta.json").read_text())
    ks = [int(k) for k in a.levels.split(",")]
    same = z["sub_sameimg"]
    rng = np.random.default_rng(0)
    show = np.zeros(len(same), bool)
    for g in (same, ~same):  # half the dots from each group, so the rarer same-image pairs stay visible
        idx = np.flatnonzero(g)
        show[rng.choice(idx, min(len(idx), a.n_show // 2), replace=False)] = True
    rows = [("d1", f"corr(D_S(x), D_S(x′))"), ("delta", f"corr(Δ(x), Δ(x′)),  Δ = D_S − D_XS")]
    fig, axs = plt.subplots(2, len(ks), figsize=(3.2 * len(ks), 5.9), constrained_layout=True)
    for i, (f, ylab) in enumerate(rows):
        for j, k in enumerate(ks):
            ax = axs[i, j]
            r, c = z["sub_r"][k], z[f"sub_{f}_{a.kind}"][k]
            for g, col, lab in ((~same, OTHER, "other image (x0 ≠ x0′)"), (same, SAME, "same image (x0 = x0′)")):
                m = g & show
                ax.scatter(r[m], c[m], s=1.5, color=col, alpha=0.12, lw=0, rasterized=True)
                xm, ym = binned(r[g], c[g], a.bins)
                ax.plot(xm, ym, "-o", color=col, ms=3, lw=1.6, mec="white", mew=0.5, label=lab)
                print(f"{f:5s} sigma {z['sigma'][k]:8.4f}  {lab:24s} |x-x'| 1/50/99 % "
                      f"{np.percentile(r[g], 1):.3g} {np.median(r[g]):.3g} {np.percentile(r[g], 99):.3g}  "
                      f"corr mean {c[g].mean():+.3f} sd {c[g].std():.3f}")
            if r.max() / r.min() > 3:
                ax.set_xscale("log")
            ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
            ax.tick_params(labelsize=7.5)
            if i == 0:
                ax.set_title(f"σ = {z['sigma'][k]:.3g}", fontsize=10)
            else:
                ax.set_xlabel("|x − x′|  (nearest-image units)", fontsize=8)
        axs[i, 0].set_ylabel(ylab + ("" if a.kind == "raw" else ", centred"), fontsize=8.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
    fig.suptitle(f"EDM2 ImageNet-64, class {meta['cls']}: x = x0 + σ n, {meta['images']} val images × "
                 f"{meta['per_image']} noises; D_S = {Path(meta['net']).stem}, D_XS = {Path(meta['guide']).stem}",
                 fontsize=9, y=1.06)
    out = a.out or str(D / f"{a.tag}-d1-delta-{a.kind}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
