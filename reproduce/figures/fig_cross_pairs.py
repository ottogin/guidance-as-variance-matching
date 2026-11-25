#!/usr/bin/env python
"""Appendix figures fig:cross-pairs: guidance in both directions for the two SD 2.1 pairs, drawn like Fig. 1(c).

    python reproduce/figures/fig_cross_pairs.py --out paper-figures/

Writes cross-sd21-cc.pdf (SD 2.1-base x CommonCanvas) and cross-sd15-sd21.pdf (SD 1.5 x SD 2.1-base). Each panel
puts FID (left axis, circles) and FD_DINOv2 (right axis, squares) on one guidance-weight axis w, the weight on
the model at w = 1: D = D_0 + w (D_1 - D_0), with the two base models at w = 0 and w = 1 and the band between
them shaded. CommonCanvas is always at w = 0 and SD 1.5 always at w = 1, as in Fig. 1(c).

Each point is an evaluate.py record on the first 1,000 COCO evaluation rows (tag 1k), sampled as the paper's were:
w > 0 on the model at 1's own pipeline (the cross adapter cross-<m1>-<m0>, at --w w), w <= 0 on the model at 0's
(cross-<m0>-<m1>, at --w 1 - w, so w = 0 is that model alone). See reproduce/plan.py (fig:cross-pairs).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

INK, MUTED, BAND, EDGE = "#000000", "#6b6a63", "#efefed", "#c9c9c5"
FIDC, FDC = "#2a78d6", "#4a3aa7"
NAME = {"sd15": "SD 1.5", "sd21": "SD 2.1", "cc": "CommonCanvas"}
# (output stem, model at w = 0, model at w = 1)
PAIRS = [("cross-sd15-sd21", "sd21", "sd15"),       # fig:cross-sd15-sd21
         ("cross-sd21-cc", "cc", "sd21")]           # fig:cross-sd21-cc
POS = (0.5, 1, 2, 3, 4, 5)               # w on the w = 1 model's pipeline
NEG = (0, -1, -2, -3, -4)                # w on the w = 0 model's pipeline (its --w is 1 - w)
TAG, N = "1k", 1000


def arms(m0, m1):
    """[(adapter, --w value, w on the figure's axis)] behind one panel."""
    return [(f"cross-{m1}-{m0}", w, w) for w in POS] + [(f"cross-{m0}-{m1}", 1 - w, w) for w in NEG]


def load(m0, m1):
    from reproduce.rosters.sweeps import arm
    rows = []
    for key, v, w in arms(m0, m1):
        path = runs.metrics_json(key, "coco", arm(v), TAG)
        assert path.exists(), f"{path} missing: sample and evaluate {key} at --w {v:g} (reproduce/plan.py)"
        r = json.load(open(path))
        assert r["n"] == N, f"{path}: {r['n']} rows, not {N}"
        rows.append((w, r["fid"], r["fd_dinov2"]))
    return sorted(rows)


def nice_ticks(lo, hi, n=4):
    """The finest round step (1, 2, 2.5 or 5 times a power of ten; integer ticks only) giving at
    most n + 1 ticks inside [lo, hi]."""
    import math
    for mag in (10 ** k for k in range(0, 6)):
        for step in (mag, 2 * mag, 2.5 * mag, 5 * mag):
            if step != int(step):
                continue
            first = math.ceil(lo / step) * step
            ticks = [int(first + k * step) for k in range(int((hi - first) / step + 1e-9) + 1)]
            if len(ticks) <= n + 1:
                return ticks
    return [lo, hi]


def panel(rows, m0, m1, out, w_in, h_in, png=False, wrange=None):
    """wrange = (lo, hi) fixes the w axis (to share it across panels); default: the data's own range."""
    import matplotlib.pyplot as plt

    ws, fid, fd = zip(*rows)
    wlo, whi = wrange or (min(ws), max(ws))
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0.13, 0.285, 0.72, 0.53])
    ax2 = ax.twinx()
    ax.axvspan(0, 1, color=BAND, zorder=0, lw=0)
    for x in (0, 1):
        ax.axvline(x, color=EDGE, lw=0.7, zorder=0.5)
    ax.plot(ws, fid, color=FIDC, lw=1.3, marker="o", ms=4.2, zorder=3, clip_on=False)
    ax2.plot(ws, fd, color=FDC, lw=1.3, marker="s", ms=4.0, zorder=3, clip_on=False)

    for a_, ys, col in ((ax, fid, FIDC), (ax2, fd, FDC)):
        lo, hi = min(ys), max(ys)
        pad = 0.06 * (hi - lo)
        a_.set_ylim(lo - pad, hi + pad)
        a_.set_yticks(nice_ticks(lo, hi))
        a_.tick_params(axis="y", colors=col, labelcolor=col, labelsize=7, length=2.5, width=0.6, pad=1.5)
    ax2.set_zorder(ax.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax.set_ylabel(r"FID $\downarrow$", color=FIDC, fontsize=8, labelpad=2)
    ax2.set_ylabel(r"FD$_{\mathrm{DINOv2}}$ $\downarrow$", color=FDC, fontsize=8, labelpad=4)

    xt = [w for w in (-4, -3, -2, -1, 0, 1, 2, 3, 4, 5) if wlo <= w <= whi]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{w:g}".replace("-", "−") for w in xt])
    ax.set_xlim(wlo - 0.35, whi + 0.35)
    ax.tick_params(axis="x", colors=MUTED, labelcolor=MUTED, labelsize=7, length=2.5, width=0.6, pad=1.5)
    for a_ in (ax, ax2):
        a_.spines["top"].set_visible(False)
        for s in ("left", "right", "bottom"):
            a_.spines[s].set_color(MUTED)
            a_.spines[s].set_linewidth(0.6)
    ax.set_xlabel(r"Guidance weight $w$", fontsize=8, labelpad=1, color=INK)
    ax.annotate(rf"$w=0$: {NAME[m0]};  $w=1$: {NAME[m1]}", xy=(0.5, 0), xycoords=ax.xaxis.label,
                xytext=(0, -0.5), textcoords="offset points", ha="center", va="top", fontsize=7, color=INK)

    # what each side of the band does, over the left and right halves of the axis
    tr = ax.get_xaxis_transform()
    ax.text((wlo - 0.35 + 0) / 2, 1.04, f"{NAME[m0]} away\nfrom {NAME[m1]}", transform=tr,
            ha="center", va="bottom", fontsize=7, color=INK, linespacing=1.05)
    ax.text((1 + whi + 0.35) / 2, 1.04, f"{NAME[m1]} away\nfrom {NAME[m0]}", transform=tr,
            ha="center", va="bottom", fontsize=7, color=INK, linespacing=1.05)
    fig.savefig(out + ".pdf")
    if png:
        fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(out + ".pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--width", type=float, default=3.25, help="inches; CVPR column = 3.25")
    ap.add_argument("--height", type=float, default=1.65)
    ap.add_argument("--png", action="store_true", help="also write a 300 dpi PNG next to each PDF")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                                "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                                "axes.unicode_minus": True})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for stem, m0, m1 in PAIRS:
        panel(load(m0, m1), m0, m1, str(Path(a.out) / stem), a.width, a.height, a.png)


if __name__ == "__main__":
    main()
