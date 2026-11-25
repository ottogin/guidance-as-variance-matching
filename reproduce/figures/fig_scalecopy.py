#!/usr/bin/env python
"""fig:scalecopy: the weak models' disagreement is a scaled copy of the strong models' (16-D Gaussian toy).

    python reproduce/figures/fig_scalecopy.py --out paper-figures/scalecopy [--src scalecopy.npz]

(a) Mean over the 69 points x_t per noise level t of ||C_weak - a C_strong||^2 / ||C_weak||^2 with a fitted per
point, per noise level or once overall (+-1 SE); (b)-(d) the two covariances at one x_t each (t = 1, 0.1, 0.05),
projected onto the data's longest and shortest axes as 1-sd ellipses, with C_strong times the a fitted in that
panel. One CVPR column, 2 x 2. Reads only reproduce/toys/bigboot16.py's scalecopy.npz.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce import style as F  # noqa: E402

CELLS = [(1.0, 0), (0.1, 2), (0.05, 5)]          # (t, place) of panels (b)-(d): the centre, the short axis, a draw
WEAK, STRONG = F.COND, F.ACC


def ellipse(Mx, n=240):
    lam, U = np.linalg.eigh(Mx)
    th = np.linspace(0, 2 * np.pi, n)
    return (U * np.sqrt(np.clip(lam, 0, None))) @ np.stack([np.cos(th), np.sin(th)])


def load(path):
    Z = np.load(path)
    sig, gi = Z["sigmas"], np.flatnonzero(Z["good"])
    loss = {k: (Z[f"{k}_mean"][gi], Z[f"{k}_se"][gi]) for k in ("L_loc", "L_sig", "L_uni")}
    cells = []
    for t, k in CELLS:
        i = int(np.flatnonzero(np.isclose(sig, t))[0])
        cells.append(dict(t=t, W=Z["W2"][i, k], S=Z["S2"][i, k], a=float(Z["a2"][i, k])))
    return dict(t=sig[gi], loss=loss, a_uni=float(Z["a_uni"]), cells=cells)


def loss_panel(ax, d):
    t = d["t"]
    spec = [("L_loc", "#aeaca6", "one $a$ per $x_t$"), ("L_sig", "#6f6d68", "one $a$ per $t$"),
            ("L_uni", F.INK, rf"one $a$ for all ($a={d['a_uni']:.2f}$)")]          # more sharing = darker
    for key, col, lab in spec:
        m, se = d["loss"][key]
        ax.errorbar(t, m, yerr=se, color=col, lw=1.1, marker="o", ms=2.6, capsize=1.4, elinewidth=0.6, capthick=0.6,
                    label=lab, zorder=3)
    F.style(ax)
    ax.set_xscale("log")
    ax.set_xticks(t, [f"{x:g}" for x in t])
    ax.minorticks_off()
    ax.set_xlim(t.max() * 1.35, t.min() / 1.35)                       # sampling order: noisy on the left
    ax.set_ylim(0, 0.1)
    ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.1], ["0", ".02", ".04", ".06", ".08", ".10"])
    ax.set_xlabel("noise level $t$", labelpad=1.5)
    ax.set_ylabel(r"$\|C_{\mathrm{weak}}-a\,C_{\mathrm{strong}}\|^2 / \|C_{\mathrm{weak}}\|^2$", labelpad=2)
    for j, c in enumerate(d["cells"]):
        ax.axvline(c["t"], color=F.GRID, lw=0.8, ls=(0, (2, 1.5)), zorder=1)
        ax.text(c["t"], 0.098, f"({'bcd'[j]})", ha="center", va="top", fontsize=6, color=F.MUTED,
                bbox=dict(fc="white", ec="none", pad=0.6))
    leg = ax.legend(loc="upper left", frameon=True, fontsize=6.2, handlelength=1.6, borderaxespad=0.2, borderpad=0.25,
                    bbox_to_anchor=(0.0, 0.9), ncol=1, columnspacing=1.2)
    leg.get_frame().set(facecolor="white", edgecolor="none", alpha=1)
    leg.set_zorder(5)


def ellipse_panel(ax, c):
    W, S, a = c["W"], c["S"], c["a"]
    f = 1 / (np.abs(ellipse(W)).max() * 1.15)                        # each panel on its own scale
    ew, es, eas = ellipse(W) * f, ellipse(S) * f, ellipse(a * S) * f
    ax.plot(ew[0], ew[1], color=WEAK, lw=2.0, zorder=2)
    ax.plot(eas[0], eas[1], color=STRONG, lw=1.1, ls=(0, (3, 1.6)), zorder=3)
    ax.plot(es[0], es[1], color=STRONG, lw=0.8, zorder=3)
    L = (np.abs(ellipse(W)).max() * 1.15) * f
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color(F.MUTED)
        s.set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axhline(0, color=F.GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=F.GRID, lw=0.6, zorder=0)
    ax.set_xlabel("long data axis", labelpad=2)
    ax.set_ylabel("short data axis", labelpad=2)


def build(d, out):
    plt = F.rc()
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    ttl = lambda ax, s: ax.set_title(s, fontsize=7.5, color=F.INK, pad=3)       # noqa: E731
    fig = plt.figure(figsize=(3.3, 3.15))
    gs = GridSpec(2, 2, figure=fig, hspace=0.46, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    loss_panel(ax, d)
    ttl(ax, "(a) distance from a scaled copy")
    for j, c in enumerate(d["cells"]):
        ax = fig.add_subplot(gs[(j + 1) // 2, (j + 1) % 2])
        ellipse_panel(ax, c)
        ttl(ax, f"({'bcd'[j]}) $t={c['t']:g}$")
    fig.subplots_adjust(left=0.15, right=0.99, top=0.95, bottom=0.1)
    hs = [Line2D([], [], color=WEAK, lw=2.0), Line2D([], [], color=STRONG, lw=0.8),
          Line2D([], [], color=STRONG, lw=1.1, ls=(0, (3, 1.6)))]
    fig.legend(hs, ["weak ($n=10$k)", "strong ($n=30$k)", r"$a\times$ strong"], frameon=False, loc="upper center",
               bbox_to_anchor=(0.57, 0.035), ncol=3, fontsize=6.5, handlelength=2.0, columnspacing=1.6)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(str(out) + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", f"{out}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--src", help="scalecopy.npz (default $VMG_RUNS/_toys/bigboot16/scalecopy.npz)")
    a = ap.parse_args()
    from vmg import runs
    build(load(a.src or runs.root() / "_toys" / "bigboot16" / "scalecopy.npz"), a.out)


if __name__ == "__main__":
    main()
