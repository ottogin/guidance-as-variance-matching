#!/usr/bin/env python
"""fig:cov-trajectory: the two predictions of a guided pair along the unguided sampler, three pairs side by side.

    python reproduce/figures/fig_cov_trajectory.py --sd35m DIR --cfg DIR --ag DIR --out paper-figures/cov-trajectory

Columns: CFG on SD 3.5 Medium (one prompt, reproduce/experiments/seedcov.py), then EDM2-S on one ImageNet-64 class
(reproduce/experiments/seedcov_edm2.py) guided by CFG with the unconditional EDM2-XS and by autoguidance with EDM2-XS.
Each DIR holds meta.json, vgap.npz and align.npz. One figure at the CVPR text width, fonts at print size, so the
rows line up by construction. Rows, on one step axis per column:

  (a) E||v_1 - v_0||^2 per coordinate, over the states x_t at each step (vgap.npz "total"): one log axis for all;
  (b) direction alignment S_K, the mean cos^2 of the principal angles between top-K subspaces (align.npz), of
      Cov(D_1) and Cov(D_0) (blue) and of Cov(D_1) and Cov(x_0) (orange); one axis for all.

D_1 is the strong prediction (conditional for CFG, EDM2-S for autoguidance), D_0 the weak one. Shading in the CFG
columns: the guidance interval of Kynkaanniemi et al. (2024), EDM sigma in (0.28, 5.42], in each sampler's steps;
guidance harms before it and is unnecessary after. The top axis is that noise level t: meta "sigma" for EDM2, and
s / (1 - s) for SD 3.5's flow time s.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reproduce import style as st  # noqa: E402
from reproduce.figures.fig_vgap import SIG_HI, SIG_LO  # noqa: E402

COLS = (("sd35m", "CFG: SD 3.5 Medium", True),
        ("cfg", r"CFG: EDM2-S $\leftarrow$ EDM2-XS (uncond.)", True),
        ("ag", r"Autoguidance: EDM2-S $\leftarrow$ EDM2-XS", False))      # key, title, CFG's interval shaded
WEAK, X0 = "#2a78d6", "#eb6834"                                             # D_1 vs D_0, D_1 vs x_0


def load(run, K):
    """One column's curves: noise level t and the zone steps (first, last of the guidance interval) per step."""
    run = Path(run)
    meta = json.load(open(run / "meta.json"))
    if "sigma" in meta:                                     # EDM2: sigma itself
        t = np.asarray(meta["sigma"], float)
    else:                                                   # SD 3.5: flow time s -> t = s / (1 - s), inf at s = 1
        s = np.asarray(meta["s"], float)
        with np.errstate(divide="ignore"):
            t = s / (1 - s)
    L = np.load(run / "align.npz")
    if not (run / "vgap.npz").exists():                     # SD 3.5: seedcov.py caches it from its batch sums
        import importlib.util
        spec = importlib.util.spec_from_file_location("seedcov", REPO / "reproduce" / "experiments" / "seedcov.py")
        seedcov = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(seedcov)
        assert seedcov.vgap_run(run) is not None, f"{run}: no vgap.npz and no batches/b*-sums.npz to make it"
    gap = np.load(run / "vgap.npz")["total"]
    S = len(t)
    assert len(gap) == S and L["DcDu_paired_sub"].shape[0] == S, f"{run}: step counts disagree"
    ks = np.arange(1, S + 1)
    on = (t > SIG_LO) & (t <= SIG_HI)
    first, last = ks[on][0], ks[on][-1]
    assert on[first - 1:last].all(), f"{run}: the guidance interval is not one run of steps"
    return dict(t=t, S=S, gap=gap, weak=L["DcDu_paired_sub"][:, K - 1], x0=L["Dc_paired_sub"][:, K - 1],
                first=int(first), last=int(last))


def fit(options, width_in, size_pt):
    """The first of `options` (zone labels, longest lines first) whose widest line fits width_in at size_pt."""
    for o in options:
        if max(len(line) for line in o.split("\n")) * 0.5 * size_pt / 72 <= width_in:
            return o
    return options[-1]


def draw(cols, out, K):
    plt = st.rc(size=7, small=6)
    import matplotlib.ticker
    from matplotlib.transforms import blended_transform_factory

    W = 6.875                                               # CVPR \textwidth, included at 100%
    Lm, Rm, gc = 0.47, 0.12, 0.17                           # left (y labels), right (the last top label), between columns
    Tt, Tx, ha, gab, hb, B = 0.17, 0.31, 0.80, 0.06, 0.92, 0.31   # title, top axis, (a), gap, (b), bottom axis
    wc = (W - Lm - Rm - 2 * gc) / 3
    H = Tt + Tx + ha + gab + hb + B
    fig = plt.figure(figsize=(W, H))

    def rect(x, ytop, w, h):                                # inches from the top left -> figure fractions
        return [x / W, (H - ytop - h) / H, w / W, h / H]

    gmin, gmax = min(c["gap"].min() for c in cols), max(c["gap"].max() for c in cols)
    lo, hi = gmin / 2, gmax * 8                             # room above for the zone labels
    amin = min(min(c["weak"].min(), c["x0"].min()) for c in cols)
    y0 = min(0.6, np.floor(amin * 10) / 10)

    av0 = ab0 = None
    for j, (c, (key, title, zones)) in enumerate(zip(cols, COLS)):
        x = Lm + j * (wc + gc)
        av = fig.add_axes(rect(x, Tt + Tx, wc, ha), sharey=av0)
        ab = fig.add_axes(rect(x, Tt + Tx + ha + gab, wc, hb), sharex=av, sharey=ab0)
        av0, ab0 = av0 or av, ab0 or ab
        S, first, last = c["S"], c["first"], c["last"]
        ks = np.arange(1, S + 1)
        tk = [1, first, last, S]

        for ax in (av, ab):
            st.style(ax)
            if zones:
                ax.axvspan(0.5, first - 0.5, color=st.ACC, alpha=0.09, lw=0, zorder=0)
                ax.axvspan(last + 0.5, S + 0.5, color=st.MUTED, alpha=0.12, lw=0, zorder=0)
            if j:
                ax.tick_params(labelleft=False)

        # (a) the velocity gap
        av.plot(ks, c["gap"], color=st.UNCOND, lw=1.3, zorder=3)
        av.set_yscale("log")
        av.set_ylim(lo, hi)
        av.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=12))
        av.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        av.tick_params(labelbottom=False)
        top = av.secondary_xaxis("top")
        ttk = [q for q in tk if q != S or (S - last) * wc / S >= 0.3]   # no last label when it would crowd
        top.set_xticks(ttk)
        top.set_xticklabels([r"$\infty$" if not np.isfinite(c["t"][q - 1]) else
                             f"${c['t'][q - 1]:.0f}$" if c["t"][q - 1] >= 10 else f"${c['t'][q - 1]:.2g}$"
                             for q in ttk])
        top.set_xlabel(r"noise level $t$", labelpad=2, color=st.MUTED)
        top.tick_params(colors=st.MUTED, labelcolor=st.MUTED, width=0.6, length=2.5, pad=1.5)
        top.spines["top"].set_color(st.MUTED)
        fs = 5.8
        kw = dict(fontsize=fs, ha="center", va="top", linespacing=1.0,
                  transform=blended_transform_factory(av.transData, av.transAxes))
        step_in = wc / S
        if zones:
            for (a_, b_), opts, col in ((((0.5, first - 0.5)), ("guidance harmful", "guidance\nharmful"), "#b04a1e"),
                                        (((first - 0.5, last + 0.5)), ("useful guidance interval",
                                                                       "useful guidance\ninterval"), st.INK),
                                        (((last + 0.5, S + 0.5)), ("unnecessary", "unneces-\nsary",
                                                                   "un-\nneces-\nsary"), st.MUTED)):
                av.text((a_ + b_) / 2, 0.96, fit(opts, (b_ - a_) * step_in - 0.04, fs), color=col, **kw)
        else:
            av.text((S + 1) / 2, 0.96, "useful guidance at every step", color=st.INK, **kw)

        # (b) direction alignment
        ab.plot(ks, c["weak"], color=WEAK, lw=1.3, zorder=4, label=r"$D_1$ vs $D_0$")
        ab.plot(ks, c["x0"], color=X0, lw=1.3, zorder=4, label=r"$D_1$ vs $x_0$")
        ab.axhline(1, color=st.INK, lw=0.7, ls=(0, (4, 2)), zorder=2)
        ab.set_ylim(y0, 1.04)
        ab.set_yticks(np.round(np.arange(y0, 1.001, 0.1), 1))
        ab.set_xlim(0.5, S + 0.5)
        ab.set_xticks(tk)
        ab.set_xlabel("sampling step", labelpad=1.5)
        fig.text((x + wc / 2) / W, 1 - 0.02 / H, title, ha="center", va="top", fontsize=8, color=st.INK)
        if j == 0:
            av.set_ylabel(r"(a) $\mathbb{E}\,\|v_1 - v_0\|^2$")
            ab.set_ylabel("(b) direction alignment")
            ab.text(1.2, 1.006, "1: the same directions", fontsize=fs, color=st.INK, va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
            ab.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", handlelength=1.6,
                      borderpad=0.3, labelspacing=0.25)
    fig.align_ylabels([av0, ab0])

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf")                               # no cropping: exactly the text width
    fig.savefig(f"{out}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out}.{{pdf,png}} ({W:.3f} x {H:.3f} in)")


def report(cols):
    """The numbers App. C.2's text quotes: the gap's fall per zone, alignment at the zone edges, AG over CFG."""
    for (key, *_), c in zip(COLS, cols):
        g, f, l = c["gap"], c["first"], c["last"]
        print(f"{key}: steps {c['S']}, interval {f}-{l}; gap step 1 {g[0]:.3g}, step {f} {g[f - 1]:.3g} "
              f"(x{g[0] / g[f - 1]:.1f} down), step {l} {g[l - 1]:.3g} (x{g[f - 1] / g[l - 1]:.1f} down); "
              f"S_K vs D_0 at 1/{f}/{l}: {c['weak'][0]:.2f}/{c['weak'][f - 1]:.2f}/{c['weak'][l - 1]:.2f}, "
              f"vs x_0: {c['x0'][0]:.2f}/{c['x0'][f - 1]:.2f}/{c['x0'][l - 1]:.2f}")
    cfg, ag = cols[1], cols[2]
    if cfg["S"] == ag["S"]:
        h = slice(0, cfg["first"] - 1)
        r = cfg["gap"][h] / ag["gap"][h]
        print(f"CFG over AG gap at the high-noise steps 1-{cfg['first'] - 1}: median x{np.median(r):.1f} "
              f"(range {r.min():.1f}-{r.max():.1f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sd35m", required=True, help="seedcov.py run: CFG on SD 3.5 Medium")
    ap.add_argument("--cfg", required=True, help="seedcov_edm2.py run: CFG, EDM2-S with the unconditional EDM2-XS")
    ap.add_argument("--ag", required=True, help="seedcov_edm2.py run: autoguidance, EDM2-S with EDM2-XS")
    ap.add_argument("--k", type=int, default=50, help="the subspace size of (b)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cols = [load(getattr(a, key), a.k) for key, *_ in COLS]
    report(cols)
    draw(cols, a.out, a.k)


if __name__ == "__main__":
    main()
