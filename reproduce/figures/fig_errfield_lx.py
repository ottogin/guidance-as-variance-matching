#!/usr/bin/env python
"""corr(Delta(x), Delta(x')) against |x - x'| at four noise levels (2 x 2), with the measured l_x and l_t of
App. C.4's error field (eq. error-field) on each panel.

    python reproduce/figures/fig_errfield_lx.py DIR [--tag spread-c850] [--ttag spreadt-c850] [--out PDF]

DIR holds `errfield_edm2.py spread` (random x of one class, partners at every distance) and `spreadt` (the same x,
the networks told sigma' = sigma exp(delta)) outputs. Delta = D1 - D0 (EDM2-S minus EDM2-XS), raw cosine.
l_x: the distance at which the binned mean correlation falls to exp(-1/2), the kernel's value at |x - x'| = l_x
(log-interpolated), marked by the dotted line. l_t: the same crossing of the mean
correlation against |log sigma' - log sigma| (both signs averaged). Distances in units of the median nearest-image
distance (46.97 px), as in fig:diversity-lx.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproduce.experiments.errfield_edm2 import E_HALF, crossing  # noqa: E402


N_STEPS = 32   # EDM2's sampler (ens_edm2.STEPS); titles count steps from 1


def binned(r, c, bins):
    edges = np.quantile(r, np.linspace(0, 1, bins + 1))
    b = np.clip(np.digitize(r, edges) - 1, 0, bins - 1)
    return np.array([r[b == j].mean() for j in range(bins)]), np.array([c[b == j].mean() for j in range(bins)])


def l_t(zt, j, field="delta", kind="raw"):
    """l_t at level j of a spreadt file: crossing of the mean corr against |delta|, +delta and -delta averaged;
    also each side alone."""
    dl, m = zt["dlogs"], zt[f"{field}_{kind}"][j].mean(1)
    pos = dl > 0
    ad = dl[pos]
    up = m[pos]
    down = np.array([m[dl == -v][0] for v in ad])
    return crossing(ad, (up + down) / 2), crossing(ad, up), crossing(ad, down)


def fmt(c):
    s, v = c
    return {"=": "", "<": "<", ">": ">"}.get(s, "?") + f"{v:.2g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="spread-c850")
    ap.add_argument("--ttag", default="spreadt-c850")
    ap.add_argument("--sigmas", default="16.6,3.83,0.598,0.0474")
    ap.add_argument("--xlim", default="2e-4,70", help="shared x range, units")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--paper", action="store_true", help="single-column size for the paper (3.3 in wide)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.paper:   # match the paper's other figures: Times-like serif text and STIX math
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                             "mathtext.fontset": "stix"})
    F = (dict(size=(3.3, 2.75), title=7, text=6.5, tick=5.5, label=6.5, lw=1.1, ms=0.6) if a.paper else
         dict(size=(6.6, 5.4), title=10, text=9.5, tick=8, label=9, lw=1.6, ms=1))
    D = Path(a.dir)
    z = np.load(D / f"{a.tag}.npz")
    zt = np.load(D / f"{a.ttag}.npz")
    meta = json.loads((D / f"{a.tag}-meta.json").read_text())
    want = [float(s) for s in a.sigmas.split(",")]
    js = [int(np.argmin(abs(np.log(z["sigma"] / s)))) for s in want]
    jt = [int(np.argmin(abs(np.log(zt["sigma"] / s)))) for s in want]
    fig, axs = plt.subplots(2, 2, figsize=F["size"], sharex=True, sharey=True,
                            constrained_layout=True)
    out_json = {}
    for ax, j, k in zip(axs.flat, js, jt):
        s = z["sigma"][j]
        assert abs(np.log(zt["sigma"][k] / s)) < 1e-4, (s, zt["sigma"][k])
        r, c = z["r"][j].ravel(), z["delta_raw"][j].ravel()
        xm, ym = binned(r, c, a.bins)
        lx = crossing(xm, ym)
        lt, ltp, ltm = l_t(zt, k)
        ax.scatter(r, c, s=F["ms"], color="#9aa7b8", alpha=0.25, lw=0, rasterized=True)
        ax.plot(xm, ym, "-", color="black", lw=F["lw"])   # measured range only: no extension beyond the data
        ax.axvline(lx[1], color="#c8562a", lw=0.8, ls=":")
        ax.axhline(0, color="#bbbbbb", lw=0.6, zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(-0.25, 1.05)
        ax.set_xlim(*[float(v) for v in a.xlim.split(",")])
        ax.set_title(f"step {int(z['levels'][j]) + 1}/{N_STEPS} (σ = {s:.3g})", fontsize=F["title"], pad=3)
        ax.text(0.04, 0.06, f"$\\ell_x$ = {fmt(lx)}\n$\\ell_t$ = {fmt(lt)}", transform=ax.transAxes, fontsize=F["text"],
                va="bottom", ha="left")
        ax.tick_params(labelsize=F["tick"])
        out_json[f"{s:.4g}"] = dict(l_x=lx, l_t=lt, l_t_up=ltp, l_t_down=ltm)
        print(f"sigma {s:8.4f}: l_x {fmt(lx)} units, l_t {fmt(lt)} (sigma' > sigma {fmt(ltp)}, < {fmt(ltm)})")
    for ax in axs[1]:
        ax.set_xlabel("$\\|x-x'\\|$" + ("" if a.paper else "  (nearest-image units)"), fontsize=F["label"])
    for ax in axs[:, 0]:
        ax.set_ylabel("corr$(\\Delta(x), \\Delta(x'))$", fontsize=F["label"])
    out = a.out or str(D / "errfield-lx.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(Path(out).with_suffix(".png")), dpi=170, bbox_inches="tight")
    (Path(out).with_suffix(".json")).write_text(json.dumps(dict(meta=meta, levels=out_json), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
