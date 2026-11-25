#!/usr/bin/env python
"""Correlation of an EDM2 pair's Delta = D1 - D0 at two inputs against their distance, no kernel assumed.

    python reproduce/figures/fig_errfield_edm2.py DIR [--out FIG.png] [--levels 0,5,10,15,20,25]

DIR is an output folder of reproduce/experiments/errfield_edm2.py. One panel per noise level; x = |x - x'| in units of
the median nearest-image distance (1,000 train images), y = correlation of Delta.
sigma and the class label are fixed within a panel: every x' is evaluated with x's label. Drawn, when present:
  pairs-c8.npz  pairs of guided trajectories at that step, same class (mean, 10-90 % band): errfield_edm2.py pairs
                --classes 8 --tag pairs-c8 (--tag here for another)
  field.npz     the designed perturbations of x = x0 + sigma n: rotated noise (same image), and the image moved toward
                its nearest image / a random image (same noise); pooled correlation
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
    ap.add_argument("dir")
    ap.add_argument("--out")
    ap.add_argument("--levels", default="0,5,10,15,20,25,31")
    ap.add_argument("--kind", default="raw", choices=("raw", "cen"))
    ap.add_argument("--tag", default="pairs-c8")
    a = ap.parse_args()
    D = Path(a.dir)
    ks = [int(k) for k in a.levels.split(",")]
    pz = np.load(D / f"{a.tag}.npz") if (D / f"{a.tag}.npz").exists() else None
    fz = np.load(D / "field.npz") if (D / "field.npz").exists() else None
    meta = json.loads(((D / "meta.json") if (D / "meta.json").exists() else (D / f"{a.tag}-meta.json")).read_text())
    unit = meta["unit"]
    sig = (pz if pz is not None else fz)["sigma"]
    nc = min(4, len(ks))
    nr = -(-len(ks) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.7 * nr), sharey=True, constrained_layout=True, squeeze=False)
    for ax in axs.flat[len(ks):]:
        ax.axis("off")
    axs = axs.flat[:len(ks)]
    C = dict(same="#c8562a", diff="#8a8f98", noise="#2f5fb8", image="#7a4bb5")
    for ax, k in zip(np.atleast_1d(axs), ks):
        ax.axhline(0, color="#cccccc", lw=0.8)
        if pz is not None:
            centers = 10 ** (0.5 * (pz["logr_bins"][1:] + pz["logr_bins"][:-1]))
            for g, lab in (("same", "two sampler trajectories, same class"),):
                m = pz[f"{g}_count"][k] >= 30
                if not m.any():
                    continue
                ax.plot(centers[m], pz[f"{g}_{a.kind}_mean"][k, m], "-o", ms=2.5, color=C[g], label=lab)
                if g == "same":
                    ax.fill_between(centers[m], pz[f"{g}_{a.kind}_q10"][k, m], pz[f"{g}_{a.kind}_q90"][k, m],
                                    color=C[g], alpha=0.2, lw=0)
        if fz is not None:
            P = list(fz["names"])
            pre = "" if a.kind == "raw" else "c"

            def pt(name):
                j = P.index(name)
                ok = ~np.isnan(fz["dot"][k, j])
                if ok.sum() < 0.5 * ok.size:
                    return None
                rho = fz[pre + "dot"][k, j, ok].sum() / np.sqrt(fz[pre + "a2"][k, j, ok].sum() *
                                                                fz[pre + "b2"][k, j, ok].sum())
                return float(np.nanmedian(fz["r"][k, j])) / unit, float(rho)
            for fam, names, lab in (
                    ("noise", [p for p in P if p.startswith("onshell")] + ["indep"], "same image, rotated noise σn → σn′"),
                    ("image", [p for p in P if p.startswith("nnlerp")] + ["nnimg", "rndimg"],
                     "same noise, image moved toward another")):
                xy = sorted(filter(None, (pt(p) for p in names if p in P)))
                if xy:
                    ax.plot(*zip(*xy), "-s", ms=3, color=C[fam], label=lab)
        ax.set_xscale("log")
        ax.set_title(f"σ = {sig[k]:.3g}", fontsize=10)
        ax.set_xlabel("|x − x′| (nearest-image units)", fontsize=8)
        ax.tick_params(labelsize=8)
    np.atleast_1d(axs)[0].set_ylabel(f"corr(Δ(x), Δ(x′)){'' if a.kind == 'raw' else ', centred'}", fontsize=9)
    np.atleast_1d(axs)[0].set_ylim(-0.15, 1.02)
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    for ax in np.atleast_1d(axs)[1:]:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0), frameon=False)
    out = a.out or str(D / f"corr-vs-distance-{a.kind}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
