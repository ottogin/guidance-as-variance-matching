#!/usr/bin/env python
"""Appendix fig:cross-pairs, panel (c): guidance in both directions between two EDM2-XS runs on ImageNet-64
that differ only in their training seed, drawn exactly like the SD panels (fig_cross_pairs.panel).

    python reproduce/figures/fig_cross_seedpair.py --out paper-figures/ [--ema 0.050]

Reads $VMG_RUNS/edm2-64-seedpair/tables/arms.csv (seedpair_in64.py table; --arms another) (one row per arm; 10,000 samples each, EDM Heun 32 steps):
D = D_B + w (D_A - D_B), so B (seed 31415926) sits at w = 0 and A (seed 20261117) at w = 1, both stock
`train_edm2.py --preset=edm2-img64-xs --duration=32Mi`, at the same EMA length. Writes cross-edm2xs-seeds-e<ema>.pdf.
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.figures import fig_cross_pairs as fc  # noqa: E402

ARMS = None     # default: $VMG_RUNS/edm2-64-seedpair/tables/arms.csv (seedpair_in64.py table)
N = 10000
fc.NAME.update({"A": "seed A", "B": "seed B"})


def fine_ticks(lo, hi, n=4):
    """Like fig_cross_pairs.nice_ticks but allows 0.25 and 0.5 steps: these FID ranges span ~1."""
    import math
    for step in (0.25, 0.5, 1, 2, 2.5, 5, 10, 20, 25, 50, 100):
        first = math.ceil(lo / step - 1e-9) * step
        ticks = [round(first + k * step, 2) for k in range(int((hi - first) / step + 1e-9) + 1)]
        if len(ticks) <= n + 1:
            return [int(t) if t == int(t) else t for t in ticks]
    return [lo, hi]


fc.nice_ticks = fine_ticks


def load(ema, wmin=-2.0, wmax=2.0):
    rows = []
    for r in csv.DictReader(open(ARMS)):
        if r["d1"] == "A" and r["guide"] == "B" and float(r["ema"]) == float(ema):
            w = float(r["w"])
            assert int(r["n"]) == N, r
            if wmin <= w <= wmax:
                rows.append((w, float(r["fid"]), float(r["fd_dinov2"])))
    assert rows, f"no A-guided-by-B arms at EMA {ema} in {ARMS}"
    return sorted(rows)


def row(emas, out, w_in, h_in, png=False, wrange=(-2, 2)):
    """All EMAs as one row of panels (figure* width): twin FID / FD_DINOv2 axes per panel, the FID label on
    the leftmost panel and the FD label on the rightmost only, one shared header for the two directions."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wlo, whi = wrange
    k = len(emas)
    fig = plt.figure(figsize=(w_in, h_in))
    L, R, gap, B, T = 0.068, 0.062, 0.078, 0.175, 0.6
    pw = (1 - L - R - gap * (k - 1)) / k
    for i, ema in enumerate(emas):
        ws, fid, fd = zip(*load(ema, wlo, whi))
        ax = fig.add_axes([L + i * (pw + gap), B, pw, T])
        ax2 = ax.twinx()
        ax.axvspan(0, 1, color=fc.BAND, zorder=0, lw=0)
        for x in (0, 1):
            ax.axvline(x, color=fc.EDGE, lw=0.7, zorder=0.5)
        ax.plot(ws, fid, color=fc.FIDC, lw=1.2, marker="o", ms=3.4, zorder=3, clip_on=False)
        ax2.plot(ws, fd, color=fc.FDC, lw=1.2, marker="s", ms=3.2, zorder=3, clip_on=False)
        for a_, ys, col in ((ax, fid, fc.FIDC), (ax2, fd, fc.FDC)):
            lo, hi = min(ys), max(ys)
            pad = 0.06 * (hi - lo)
            a_.set_ylim(lo - pad, hi + pad)
            a_.set_yticks(fine_ticks(lo, hi))
            a_.tick_params(axis="y", colors=col, labelcolor=col, labelsize=7, length=2.5, width=0.6, pad=1.5)
        ax2.set_zorder(ax.get_zorder() + 1)
        ax2.patch.set_visible(False)
        if i == 0:
            ax.set_ylabel(r"FID $\downarrow$", color=fc.FIDC, fontsize=8, labelpad=2)
        if i == k - 1:
            ax2.set_ylabel(r"FD$_{\mathrm{DINOv2}}$ $\downarrow$", color=fc.FDC, fontsize=8, labelpad=3)
        xt = [w for w in range(int(wlo), int(whi) + 1)]
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{w:g}".replace("-", "−") for w in xt])
        ax.set_xlim(wlo - 0.3, whi + 0.3)
        ax.tick_params(axis="x", colors=fc.MUTED, labelcolor=fc.MUTED, labelsize=7, length=2.5, width=0.6, pad=1.5)
        for a_ in (ax, ax2):
            a_.spines["top"].set_visible(False)
            for sp in ("left", "right", "bottom"):
                a_.spines[sp].set_color(fc.MUTED)
                a_.spines[sp].set_linewidth(0.6)
        ax.set_title(f"EMA length {float(ema):.3f}", fontsize=8, color=fc.INK, pad=3)
        if i == k // 2:
            ax.set_xlabel(r"Guidance weight $w$", fontsize=8, labelpad=1, color=fc.INK)
    # one header line: the metrics (left) and what each side of the shaded band does (right)
    h = [Line2D([], [], color=fc.FIDC, marker="o", ms=3.4, lw=1.2, label="FID"),
         Line2D([], [], color=fc.FDC, marker="s", ms=3.2, lw=1.2, label=r"FD$_{\mathrm{DINOv2}}$")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(L - 0.01, 1.0), ncol=2, frameon=False,
               fontsize=7.5, handlelength=1.6, columnspacing=1.2, borderaxespad=0.15)
    fig.text(1 - R, 0.975, r"$w=0$: seed B,  $w=1$: seed A;   $w<0$: B away from A,  $w>1$: A away from B",
             ha="right", va="top", fontsize=7.5, color=fc.INK)
    fig.savefig(out + ".pdf")
    if png:
        fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(out + ".pdf")


def column(emas, out, w_in, h_in, png=False, wrange=(-2, 2)):
    """All EMAs stacked in one column (CVPR column width), sharing the w axis: x tick labels on the bottom
    panel only, FID / FD_DINOv2 labels once on the middle panel, legend and w key as a two-line header."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wlo, whi = wrange
    k = len(emas)
    fig = plt.figure(figsize=(w_in, h_in))
    L, R, B, top, gap = 0.135, 0.145, 0.085, 0.865, 0.06    # top = upper edge of the first panel
    ph = (top - B - gap * (k - 1)) / k
    for i, ema in enumerate(emas):
        ws, fid, fd = zip(*load(ema, wlo, whi))
        ax = fig.add_axes([L, top - ph - i * (ph + gap), 1 - L - R, ph])
        ax2 = ax.twinx()
        ax.axvspan(0, 1, color=fc.BAND, zorder=0, lw=0)
        for x in (0, 1):
            ax.axvline(x, color=fc.EDGE, lw=0.7, zorder=0.5)
        ax.plot(ws, fid, color=fc.FIDC, lw=1.2, marker="o", ms=3.2, zorder=3, clip_on=False)
        ax2.plot(ws, fd, color=fc.FDC, lw=1.2, marker="s", ms=3.0, zorder=3, clip_on=False)
        for a_, ys, col in ((ax, fid, fc.FIDC), (ax2, fd, fc.FDC)):
            lo, hi = min(ys), max(ys)
            pad = 0.07 * (hi - lo)
            a_.set_ylim(lo - pad, hi + pad)
            a_.set_yticks(fine_ticks(lo, hi, n=3))
            a_.tick_params(axis="y", colors=col, labelcolor=col, labelsize=7, length=2.5, width=0.6, pad=1.5)
        ax2.set_zorder(ax.get_zorder() + 1)
        ax2.patch.set_visible(False)
        if i == k // 2:
            ax.set_ylabel(r"FID $\downarrow$", color=fc.FIDC, fontsize=8, labelpad=3)
            ax2.set_ylabel(r"FD$_{\mathrm{DINOv2}}$ $\downarrow$", color=fc.FDC, fontsize=8, labelpad=4)
        xt = list(range(int(wlo), int(whi) + 1))
        ax.set_xticks(xt)
        ax.set_xticklabels([f"{w:g}".replace("-", "\u2212") for w in xt] if i == k - 1 else [])
        ax.set_xlim(wlo - 0.3, whi + 0.3)
        ax.tick_params(axis="x", colors=fc.MUTED, labelcolor=fc.MUTED, labelsize=7, length=2.5, width=0.6, pad=1.5)
        for a_ in (ax, ax2):
            a_.spines["top"].set_visible(False)
            for sp in ("left", "right", "bottom"):
                a_.spines[sp].set_color(fc.MUTED)
                a_.spines[sp].set_linewidth(0.6)
        ax.set_title(f"EMA length {float(ema):.3f}", fontsize=7.5, color=fc.INK, pad=2)
        if i == k - 1:
            ax.set_xlabel(r"Guidance weight $w$", fontsize=8, labelpad=1, color=fc.INK)
    h = [Line2D([], [], color=fc.FIDC, marker="o", ms=3.2, lw=1.2, label="FID"),
         Line2D([], [], color=fc.FDC, marker="s", ms=3.0, lw=1.2, label=r"FD$_{\mathrm{DINOv2}}$")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=7.5, handlelength=1.6, columnspacing=1.2, borderaxespad=0.1)
    fig.text(0.5, 0.945, r"$w=0$: seed B,  $w=1$: seed A;  $w<0$: B away from A,  $w>1$: A away from B",
             ha="center", va="top", fontsize=7, color=fc.INK)
    fig.savefig(out + ".pdf")
    if png:
        fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(out + ".pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", nargs="+", default=["0.035", "0.050", "0.100"],
                    help="EMA lengths of both models (0, 0.035, 0.050, ..., 0.200); one PDF each")
    ap.add_argument("--layout", choices=["column", "row", "panels"], default="column",
                    help="column: panels stacked in one column-wide PDF; row: one figure*-wide PDF; panels: one PDF per EMA")
    ap.add_argument("--width", type=float, default=None, help="inches; default 3.25 (column, panels) or 6.875 (row)")
    ap.add_argument("--height", type=float, default=None, help="inches; default 3.9 (column), 2.0 (row) or 1.65 (panels)")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--arms", help="the arms table (default: $VMG_RUNS/edm2-64-seedpair/tables/arms.csv)")
    a = ap.parse_args()
    global ARMS
    from vmg import runs
    ARMS = a.arms or str(runs.root() / "edm2-64-seedpair" / "tables" / "arms.csv")
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                                "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                                "axes.unicode_minus": True})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.layout == "column":
        column(a.ema, str(Path(a.out) / "cross-edm2xs-seeds"), a.width or 3.25, a.height or 3.9, a.png)
        return
    if a.layout == "row":
        row(a.ema, str(Path(a.out) / "cross-edm2xs-seeds"), a.width or 6.875, a.height or 2.0, a.png)
        return
    a.width, a.height = a.width or 3.25, a.height or 1.65
    for ema in a.ema:
        tag = f"{float(ema):.3f}".replace("0.", "").replace(".", "")
        fc.panel(load(ema), "B", "A", str(Path(a.out) / f"cross-edm2xs-seeds-e{tag}"), a.width, a.height, a.png,
             wrange=(-2, 2))


if __name__ == "__main__":
    main()
