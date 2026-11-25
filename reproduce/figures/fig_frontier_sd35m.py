#!/usr/bin/env python
"""Paper figure fig:frontier-sd35m: SD 3.5 Medium at 40 steps, the constant-w curve against our schedule.

    python reproduce/figures/fig_frontier_sd35m.py --out paper-figures/frontier-sd35m

Three panels from runs/sd35m/coco/metrics/ (evaluate.py; COCO, 10,000 images per arm, the arms w1 ... w8 of
reproduce/rosters/sweeps.py and vm-schedule): FID vs FD_DINOv2, and precision vs recall (k = 3) in DINOv2 and
in Inception features.
The constant arms are joined in order of w, from unguided w = 1 to 8. w = 4.5 (the published weight) is left
out: it sits between 4 and 5 and only crowds the curve.  Writes <out>.{pdf,png}: `--layout shade` (default, the paper's)
is one column with w as a light-to-dark ramp; `--layout col` one CVPR column (3.3 in), a 2x2 grid with the legend in the fourth cell; `--layout wide` a 5.5 in row.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"
CONST, OURS = "#2a78d6", "#eb6834"
SKIP = (4.5,)  # constants not drawn

# per panel: x key, y key, axis labels, limits, w-label offsets and the "ours" label offset (points)
PANELS = [
    dict(x="fid", y="fd_dinov2", xl="FID $\\downarrow$", yl="FD$_\\mathrm{DINOv2}$ $\\downarrow$",
         title="FID vs FD$_\\mathrm{DINOv2}$", ours_off=(6, 0, "left"),
         off={1: (-6, 0), 2: (-7, -2), 3: (0, -9), 4: (0, 7), 5: (5, -4), 6: (5, -4), 7: (5, -4), 8: (5, 3)}),
    dict(x="recall_dinov2", y="precision_dinov2", xl="recall (DINOv2) $\\uparrow$",
         yl="precision (DINOv2) $\\uparrow$",
         title="Precision vs recall (DINOv2)", ours_off=(6, 0, "left"),
         off={1: (6, 0), 2: (5, -5), 3: (5, -5), 4: (0, 6), 5: (5, 6), 6: (-2, 7), 7: (1, -8), 8: (-6, -6)}),
    dict(x="recall_inception", y="precision_inception", xl="recall (Inception) $\\uparrow$",
         yl="precision (Inception) $\\uparrow$",
         title="Precision vs recall (Inception)", ours_off=(5, -6, "left"),
         off={1: (-6, 0), 2: (-3, 6), 3: (3, 6), 4: (0, 7), 5: (-1, 7), 6: (-2, 7), 7: (0, -8), 8: (-6, -6)}),
]


CONSTS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)


def load(model="sd35m", dataset="coco"):
    get = lambda arm: json.load(open(runs.metrics_json(model, dataset, arm)))     # noqa: E731
    consts = [dict(get(f"w{w:g}"), w_const=w, name=f"w{w:g}") for w in CONSTS if w not in SKIP]
    return consts, dict(get("vm-schedule"), name="vm-schedule")


# font sizes and marker areas per layout
SIZES = {"wide": dict(w=6.5, ours=7, lab=8, title=7.5, tick=7, dot=20, star=70),
         "col": dict(w=6, ours=6.5, lab=7, title=7, tick=6.5, dot=11, star=48)}
# label offsets (points) that differ in the column layout, where the axes are smaller
COL_OFF = [dict(), dict(), dict()]


def panel(ax, p, consts, ours, z=SIZES["wide"], off=None):
    x = [c[p["x"]] for c in consts]
    y = [c[p["y"]] for c in consts]
    ax.plot(x, y, color=CONST, lw=1.2, zorder=2)
    for c in consts:
        w = c["w_const"]
        ax.scatter(c[p["x"]], c[p["y"]], s=z["dot"], zorder=4, facecolor=CONST, edgecolor=CONST, linewidth=0.8)
        dx, dy = {**p["off"], **(off or {})}[w]
        ax.annotate(f"{w:g}", (c[p["x"]], c[p["y"]]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=z["w"], color=INK, ha="center", va="center")
    ax.scatter(ours[p["x"]], ours[p["y"]], s=z["star"], marker="*", color=OURS, edgecolor="white", linewidth=0.6,
               zorder=5)
    dx, dy, ha = (off or {}).get("ours", p["ours_off"])
    ax.annotate("ours", (ours[p["x"]], ours[p["y"]]), xytext=(dx, dy), textcoords="offset points",
                fontsize=z["ours"], color=INK, ha=ha, va="center")
    ax.margins(x=0.12, y=0.1)
    ax.set_xlabel(p["xl"], fontsize=z["lab"], color=INK)
    ax.set_ylabel(p["yl"], fontsize=z["lab"], color=INK)
    ax.set_title(p["title"], fontsize=z["title"], color=INK, loc="left", pad=4)
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=z["tick"], length=2.5, width=0.6, labelcolor=INK)
    return ax


# --layout shade: w = 1..8 as an ordinal ramp, light to dark.  Blue steps interpolated evenly in OKLab; gray = the
# same for the Inception curve.  Adjacent steps are close in lightness, so the ends are also labelled "1" and "8".
BLUES = ["#86b6ef", "#74a3db", "#628fc8", "#517db5", "#406aa2", "#2f598f", "#1e477d", "#0d366b"]
GRAYS = ["#c4c3bb", "#b1b0a8", "#9e9d96", "#8b8a83", "#797872", "#66655f", "#53524c", "#3a3935"]


def shade_figure(plt, consts, ours):
    import numpy as np
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MultipleLocator
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6})

    def style(ax, label):
        ax.set_title(label, fontsize=8, loc="left", pad=2, color=INK)
        ax.margins(x=0.08)
        ax.grid(True, color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=7, length=2, width=0.5, labelcolor=INK, pad=1.5)

    def curve(ax, kx, ky, ramp, offs):
        x = np.array([c[kx] for c in consts])
        y = np.array([c[ky] for c in consts])
        seg = np.stack([np.c_[x[:-1], y[:-1]], np.c_[x[1:], y[1:]]], 1)
        ax.add_collection(LineCollection(seg, colors=ramp[1:], lw=1.1, zorder=2, capstyle="round"))
        ax.scatter(x, y, s=15, c=ramp, zorder=3, edgecolor="white", linewidth=0.5)
        for i, w in ((0, 1), (len(x) - 1, 8)):
            ax.annotate(f"{w}", (x[i], y[i]), xytext=offs[w], textcoords="offset points", fontsize=6.5,
                        color=MUTED, ha="center", va="center")
        ax.scatter(ours[kx], ours[ky], s=55, marker="*", color=OURS, edgecolor="white", linewidth=0.5, zorder=5)

    assert [c["w_const"] for c in consts] == list(CONSTS), "the ramp assumes w = 1..8"
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.25, 2.5))
    curve(a, "fid", "fd_dinov2", BLUES, {1: (-6, 3), 8: (2, 6)})
    a.annotate("ours", (ours["fid"], ours["fd_dinov2"]), xytext=(10, 0), textcoords="offset points",
               fontsize=7, color=INK, ha="center", va="center")
    a.set_xlabel("FID $\\downarrow$", labelpad=1)
    a.set_ylabel("FD$_\\mathrm{DINOv2}$ $\\downarrow$", labelpad=1)
    a.autoscale_view()
    a.xaxis.set_major_locator(MultipleLocator(3))
    a.yaxis.set_major_locator(MultipleLocator(50))
    curve(b, "recall_dinov2", "precision_dinov2", BLUES, {1: (5, -4), 8: (-5, -2)})
    curve(b, "recall_inception", "precision_inception", GRAYS, {1: (-6, -1), 8: (1, -7)})
    b.text(0.555, 0.785, "DINOv2", color=BLUES[5], fontsize=7.5, ha="center")
    b.text(0.47, 0.625, "Inception", color=GRAYS[6], fontsize=7.5, ha="center")
    b.set_xlabel("recall $\\uparrow$", labelpad=1)
    b.set_ylabel("precision $\\uparrow$", labelpad=1)
    b.set_ylim(0.44, 0.82)
    b.autoscale_view(scaley=False)
    b.xaxis.set_major_locator(MultipleLocator(0.1))
    b.yaxis.set_major_locator(MultipleLocator(0.1))
    style(a, "(a)")
    style(b, "(b)")
    fig.tight_layout(pad=0.15, w_pad=0.9, rect=(0, 0, 1, 0.904))

    # key: the two ramps as swatches between "1" and "8", then the star
    k = fig.add_axes([0.0, 0.926, 1, 0.07])
    k.axis("off")
    k.set_xlim(0, 1)
    k.set_ylim(0, 1)
    x0, dw = 0.255, 0.024
    k.text(x0 - 0.012, 0.5, "constant $w$:  1", ha="right", va="center", fontsize=7.5, color=INK)
    for ramp, yy in ((BLUES, 0.76), (GRAYS, 0.24)):
        for i, c in enumerate(ramp):
            k.add_patch(plt.Rectangle((x0 + i * dw, yy - 0.22), dw * 0.88, 0.44, color=c, lw=0))
    k.text(x0 + 8 * dw + 0.006, 0.5, "8", ha="left", va="center", fontsize=7.5, color=INK)
    k.scatter([0.535], [0.5], s=55, marker="*", color=OURS, edgecolor="white", linewidth=0.5)
    k.text(0.552, 0.5, "ours, variance-matched $w(\\sigma)$", ha="left", va="center", fontsize=7.5, color=INK)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--layout", choices=("shade", "col", "wide"), default="shade")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                                "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt

    consts, ours = load()
    print("w      " + "  ".join(f"{k:>10s}" for k in ("fid", "fd_dinov2", "P_dino", "R_dino", "P_inc", "R_inc")))
    for c in consts + [ours]:
        print(f"{c['name']:14s}" + "  ".join(f"{c[k]:10.4f}" for k in ("fid", "fd_dinov2", "precision_dinov2",
                                                                       "recall_dinov2", "precision_inception",
                                                                       "recall_inception")))
    hs = [plt.Line2D([], [], color=CONST, lw=1.2, marker="o", ms=4, label="constant $w$ (label = $w$)"),
          plt.Line2D([], [], color=OURS, lw=0, marker="*", ms=9, mec="white", mew=0.6,
                     label="ours, variance-matched $w(\\sigma)$")]
    lg = None
    if args.layout == "shade":
        fig = shade_figure(plt, consts, ours)
    elif args.layout == "wide":
        fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15))
        for ax, p in zip(axes, PANELS):
            panel(ax, p, consts, ours)
        lg = fig.legend(handles=hs, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
                        bbox_to_anchor=(0.5, -0.01), handletextpad=0.4, columnspacing=1.6)
        fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.2)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.9))
        for ax, p, off in zip(axes.flat, PANELS, COL_OFF):
            panel(ax, p, consts, ours, SIZES["col"], off)
        axes[1, 1].axis("off")
        lg = axes[1, 1].legend(handles=hs, loc="center", frameon=False, fontsize=6.8, handletextpad=0.4,
                               labelspacing=0.9)
        fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.8)
    for t in (lg.get_texts() if lg else ()):
        t.set_color(INK)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor="white")
        print(f"wrote {args.out}.{ext}", flush=True)


if __name__ == "__main__":
    main()
