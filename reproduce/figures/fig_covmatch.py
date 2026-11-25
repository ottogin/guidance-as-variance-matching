#!/usr/bin/env python
"""fig:covmatch (App. D.1): SD 3.5 Medium's guidance weight from matching the trace vs the whole 16 x 16 channel covariance.

    python reproduce/figures/fig_covmatch.py --out paper-figures/covmatch-pair-sd35m

Per-step weight from matching the trace of the second-moment matrix (the paper's data-mean rule) and from fitting
one w to the whole C x C matrix: the argmin on [0, 40] of ||M(w) - T||_F^2, no hold, no clip. The last step, where
the trace rule has no root in range, is left out.
Both from the calibration pass run with --extras chancov (vmg/extras.py), whose raw channel Grams of (x0, r0, r1)
are centred here with the pass's per-coordinate first moments. With X = x0 - xbar, R_i = D_i - x0 centred on
their means and G_ab = sum over samples and positions of a b^T / (n P):
    M(w) - T = E0 + w E1 + w^2 E2,   E0 = sym(G_X0) + G_00,   E1 = sym(G_X1) - sym(G_X0) + sym(G_10) - 2 G_00,
    E2 = G_11 - sym(G_10) + G_00,    sym(G) = G + G^T,
and around the data mean (the paper's rule) each E_k gains the pooled outer product of the mean errors m(w) =
rbar0 + w (rbar1 - rbar0).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

INK, GRID, BASE, ORANGE, BLUE = "#1f1f1e", "#e4e3dc", "#c3c2b7", "#eb6834", "#2a78d6"
LO, HI = 0.0, 40.0


def sym(G):
    return G + np.swapaxes(G, -1, -2)


def matrices(sums, meta):
    """(E [S, 3, C, C] centred, MM [S, 3, C, C] the data-mean extra) from a chancov moments pass."""
    from vmg.moments import chan_matrices
    return chan_matrices(sums, meta)


def argmin_poly(J, lo=LO, hi=HI):
    """The minimiser of an ascending-coefficient polynomial on [lo, hi] (vmg.solve.argmin_poly)."""
    from vmg.solve import argmin_poly as am
    return am(J, lo, hi)


def trace_root(t, lo=LO, hi=HI):
    """The largest root in [lo, hi] of t2 w^2 + t1 w + t0, NaN if there is none."""
    a, b, c = t[2], t[1], t[0]
    disc = b * b - 4 * a * c
    if a == 0 or disc < 0:
        return np.nan
    r = np.array([(-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a)])
    inside = r[(r >= lo) & (r <= hi)]
    return float(inside.max()) if len(inside) else np.nan


def fits(E, MM):
    """(trace rule, whole-matrix argmin) per level, both around the data mean."""
    Ed = E + MM
    tr = np.array([trace_root([np.trace(e) for e in Ed[j]]) for j in range(len(E))])
    cm = []
    for j in range(len(E)):
        K = np.array([[np.vdot(a, b) for b in Ed[j]] for a in Ed[j]])
        J = np.zeros(5)
        for k in range(3):
            for m in range(3):
                J[k + m] += K[k, m]
        cm.append(argmin_poly(J))
    return tr, np.array(cm)


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(BASE)
        ax.spines[sp].set_linewidth(0.6)
    ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.2, colors=INK, length=2.0, width=0.5, color=BASE)


def plot(tr, cm, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                                "pdf.fonttype": 42, "ps.fonttype": 42})
    ok = np.isfinite(tr)
    st = np.arange(1, len(tr) + 1)
    fig, b = plt.subplots(1, 1, figsize=(3.3, 1.45))
    b.plot(st[ok], tr[ok], color=INK, lw=1.0, zorder=3, label="trace")
    b.plot(st[ok], cm[ok], color=BLUE, lw=0.9, marker="o", ms=1.8, mec="white", mew=0.3, zorder=2,
           label="whole matrix")
    b.set_ylim(0, 22.5)
    b.set_yticks([0, 5, 10, 15, 20])
    b.set_ylabel(r"guidance scale $w$", fontsize=6.8, color=INK, labelpad=1.5)
    b.legend(loc="upper left", frameon=False, fontsize=6.0, handlelength=1.4, handletextpad=0.35,
             borderaxespad=0.1, labelspacing=0.25, labelcolor=INK)
    b.set_xlim(0, 41)
    b.set_xticks([1, 10, 20, 30, 40])
    b.set_xlabel("sampling step", fontsize=6.8, color=INK, labelpad=1)
    style(b)
    fig.tight_layout(pad=0.2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--model", default="sd35m")
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--tag")
    a = ap.parse_args()
    from vmg import moments as M, runs
    sums, meta = M.load(runs.moments_shards(a.model, a.dataset, a.tag))
    assert "chancov" in meta.get("extras", []), "run the moments pass with --extras chancov"
    E, MM = matrices(sums, meta)
    tr, cm = fits(E, MM)
    plot(tr, cm, a.out)
    print(f"whole-matrix / trace weight median {np.nanmedian(cm / tr):.3f}")
    print(f"wrote {a.out}.pdf")


if __name__ == "__main__":
    main()
