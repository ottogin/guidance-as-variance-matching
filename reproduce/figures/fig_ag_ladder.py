#!/usr/bin/env python
"""fig:diversity-ladder: M(2) with a fixed error field of growing (l_x, l_t), above guidance of a trained pair with growing w.

    python reproduce/figures/fig_ag_ladder.py --simulate --out diversity-ladder.pdf

Top row: reproduce/toys/corr_flow_sim.js with field seed --seed, 512 steps, 5,000 points, at --levels; --simulate
writes them to --m2dir first (needs node). Bottom row: reproduce/toys/ag_diversity.py samples (--guided) of pair --pair
at the constant scales --ws. The data density is drawn underneath.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.style import INK, GRID, COND, ACC, MUTED, rc  # noqa: E402
from vmg.runs import root  # noqa: E402

TH = 2 * np.pi * np.arange(10) / 10
MU = np.stack([2 * np.cos(TH), np.sin(TH)], 1)
S = 0.15
XL, YL = (-2.75, 2.75), (-1.6, 1.6)


def stats(x):
    """Effective number of modes among samples within 3 s of a mode mean."""
    x = x[np.isfinite(x).all(1)]
    d2 = ((x[:, None] - MU) ** 2).sum(-1)
    on = d2.min(1) < (3 * S) ** 2
    w = np.bincount(d2.argmin(1)[on], minlength=10) / max(on.sum(), 1)
    w = w[w > 0]
    return float(np.exp(-(w * np.log(w)).sum()))


def density():
    gx, gy = np.meshgrid(np.linspace(*XL, 400), np.linspace(*YL, 240))
    g = np.stack([gx, gy], -1)
    p = np.exp(-((g[..., None, :] - MU) ** 2).sum(-1) / (2 * S * S)).sum(-1)
    return p / p.max()


def panel(ax, x, color, dens, note, size, alpha, gray):
    x = x[np.isfinite(x).all(1)]
    ax.imshow(dens, extent=(*XL, *YL), origin="lower", cmap="Greys", vmin=0, vmax=1 / gray, interpolation="bilinear",
              zorder=0)
    ax.scatter(*x.T, s=size, color=color, alpha=alpha, linewidths=0, rasterized=True, zorder=1)
    ax.set(xlim=XL, ylim=YL, xticks=[], yticks=[], aspect="equal")
    if note:
        ax.text(0.97, 0.05, note, transform=ax.transAxes, ha="right", fontsize=6.5, color=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
        s.set_linewidth(0.8)


def level_title(lx, lt):
    return r"$\ell_x=\ell_t=0$" if lx == 0 and lt == 0 else rf"$\ell_x={lx:g},\ \ell_t={lt:g}$"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2dir", default=str(root() / "_toys" / "ag_diversity" / "m2ladder"))
    ap.add_argument("--guided", default=str(root() / "_toys" / "ag_diversity" / "fresh" / "samples-w64d4s40k-n256-i1.npz"))
    ap.add_argument("--levels", nargs="+", default=["0,0", "2,0.3", "5,1"], help="l_x,l_t per column")
    ap.add_argument("--seed", type=int, default=193, help="field seed of the top row")
    ap.add_argument("--n", type=int, default=5000, help="points per M(2) panel")
    ap.add_argument("--ws", nargs="+", default=["1", "6", "12"])
    ap.add_argument("--pair", type=int, default=15)
    ap.add_argument("--counts", action="store_true", help="print the effective number of modes in each panel")
    ap.add_argument("--no-mirror-top", dest="mirror_top", action="store_false",
                    help="keep the top row as simulated (by default it is mirrored left to right)")
    ap.add_argument("--dot-size", type=float, default=0.3, help="marker area in pt^2")
    ap.add_argument("--dot-alpha", type=float, default=0.7)
    ap.add_argument("--gray", type=float, default=0.15, help="Greys level at a mode's peak density")
    ap.add_argument("--out", required=True)
    ap.add_argument("--simulate", action="store_true")
    a = ap.parse_args()
    levels = [tuple(float(v) for v in L.split(",")) for L in a.levels]
    m2 = Path(a.m2dir)

    def m2file(lx, lt):
        return m2 / (f"s{a.seed}_lx{lx:g}_lt{lt:g}" + ("" if a.n == 5000 else f"_n{a.n}") + ".bin")

    if a.simulate:
        m2.mkdir(parents=True, exist_ok=True)
        js = REPO / "reproduce/toys/corr_flow_sim.js"
        for lx, lt in levels:
            if not m2file(lx, lt).exists():
                subprocess.run(["node", str(js), str(a.seed), f"{lx:g}", f"{lt:g}", "512", str(a.n), str(m2file(lx, lt))],
                               check=True)
    z = np.load(a.guided)
    dens = density()
    plt = rc()
    nc = len(levels)
    fig, axes = plt.subplots(2, nc, figsize=(3.3, 1.62), constrained_layout=True)     # one paper column
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.015, hspace=0.04)
    for c, (lx, lt) in enumerate(levels):
        x = np.fromfile(m2file(lx, lt), dtype="<f4").reshape(-1, 2)
        if a.mirror_top:                # x -> -x maps the mixture and the law of g onto themselves: an equally likely field
            x = x * np.array([-1.0, 1.0], dtype=x.dtype)
        panel(axes[0, c], x, COND, dens, f"{stats(x):.1f} modes" if a.counts else None,
              a.dot_size, a.dot_alpha, a.gray)
        axes[0, c].set_title(level_title(lx, lt), pad=2, fontsize=6.5, color=INK)
    for c, w in enumerate(a.ws):
        x = z[f"w{w}"][a.pair]
        panel(axes[1, c], x, ACC, dens, f"{stats(x):.1f} modes" if a.counts else None,
              a.dot_size, a.dot_alpha, a.gray)
        axes[1, c].set_title(rf"$w={w}$", pad=2, fontsize=6.5, color=INK)
    axes[0, 0].set_ylabel(r"$\mathcal{M}^{(2)}$, field", fontsize=6.5, labelpad=2)
    axes[1, 0].set_ylabel("guidance", fontsize=6.5, labelpad=2)
    fig.savefig(a.out, dpi=400)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
