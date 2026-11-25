#!/usr/bin/env python
"""fig:wvar for EDM2 on ImageNet-64: Var(D_w(x_t, t)) along the sampling path, CFG next to autoguidance.

    python reproduce/figures/fig_wvar_edm2.py --out paper-figures/wvar-in64

Reads reproduce/experiments/wvar_edm2.py's wvar.json of each pair (default: $VMG_RUNS/<model>/imagenet64/wvar/,
or pass --json). Panel per pair, styled as the SD 3.5 / AudioLDM 2 figure:
constant weights in blue by w, the published weight in green, the variance-matched schedule in orange, and the
calibration images' own coordinate-averaged variance dotted. --weights adds a row with each arm's w per step.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reproduce import style as ST  # noqa: E402

PAIRS = [("edm2-64-cfg-s-xsu-fid", "(a) CFG, S $\\leftarrow$ XS-uncond"),
         ("edm2-64-ag-s-xs-fid", "(b) autoguidance, S $\\leftarrow$ XS")]
CONST = {"w1": "#86b6ef", "w2": "#3987e5", "w3": "#1c5cab", "w5": "#0d366b"}
PUB, VM, DATA = ST.UNCOND, ST.ACC, ST.MUTED


def default_json(model):
    from vmg import runs
    return runs.root() / model / "imagenet64" / "wvar" / "wvar.json"


def panel(ax, d, title, ticks):
    arms = d["arms"]
    S = len(next(iter(arms.values()))["var"]) - 1
    x = np.arange(1, S + 2)
    for k, c in CONST.items():
        if k in arms:
            ax.plot(x, arms[k]["var"], color=c, lw=1.1, zorder=2)
    if "pub" in arms:
        ax.plot(x, arms["pub"]["var"], color=PUB, lw=1.1, zorder=3)
    if "vm" in arms:
        ax.plot(x, arms["vm"]["var"], color=VM, lw=1.5, zorder=4)
    if d.get("var_x0"):
        ax.axhline(d["var_x0"], color=DATA, lw=0.8, ls=(0, (1.5, 1.5)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlim(0.5, S + 1.5)
    ax.set_xticks(ticks + [S + 1], [str(t) for t in ticks] + ["final"])
    ST.style(ax)
    ST.title(ax, title)
    print(f"{title}: n={next(iter(arms.values()))['n']}, var_x0 {d.get('var_x0')}")
    for k, v in arms.items():
        var = np.asarray(v["var"])
        print(f"  {k:>9}: step 1 {var[0]:.4f}, peak {var.max():.4f} at {int(var.argmax()) + 1}, final {var[-1]:.4f}"
              f"  (w {min(v['w']):.2f}..{max(v['w']):.2f})")


def weights_panel(ax, d, ticks):
    arms = d["arms"]
    S = len(next(iter(arms.values()))["w"])
    x = np.arange(1, S + 1)
    for k, c in CONST.items():
        if k in arms:
            ax.plot(x, arms[k]["w"], color=c, lw=0.9)
    if "pub" in arms:
        ax.plot(x, arms["pub"]["w"], color=PUB, lw=0.9)
    if "vm" in arms:
        ax.plot(x, arms["vm"]["w"], color=VM, lw=1.4, marker="o", ms=1.6)
    ax.set_xlim(0.5, S + 1.5)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.set_xlabel("sampling step")
    ST.style(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", nargs=2, help="wvar.json of the CFG pair and of the autoguidance pair")
    ap.add_argument("--out", help="output path without extension (default $VMG_RUNS/paper/wvar-in64)")
    ap.add_argument("--weights", action="store_true", help="add a row with w per step")
    ap.add_argument("--pair", choices=["both", "cfg", "ag"], default="both",
                    help="draw both pairs side by side (default) or one pair as a figure of its own")
    a = ap.parse_args()
    paths = a.json or [default_json(m) for m, _ in PAIRS]
    keep = {"both": [0, 1], "cfg": [0], "ag": [1]}[a.pair]
    data = [json.load(open(paths[j])) for j in keep]
    pairs = [PAIRS[j] for j in keep]
    if len(keep) == 1:      # one pair alone: drop the (a)/(b) prefix
        pairs = [(m, t.split(") ", 1)[1]) for m, t in pairs]
    pubs = [d["published_w"] for d in data]

    plt = ST.rc(7, 6.3)
    from matplotlib.lines import Line2D
    rows = 2 if a.weights else 1
    cols = len(data)
    width = 3.3 if cols == 2 else 1.9
    fig, axs = plt.subplots(rows, cols, figsize=(width, 1.6 if rows == 1 else 2.45), squeeze=False, sharex="col",
                            gridspec_kw=dict(wspace=0.3, hspace=0.18, height_ratios=[2.2, 1][:rows]))
    ticks = [1, 8, 16, 24]
    for j, (d, (_, title)) in enumerate(zip(data, pairs)):
        panel(axs[0, j], d, title, ticks)
        if a.weights:
            weights_panel(axs[1, j], d, ticks)
            axs[1, j].set_xticks(ticks + [len(d["arms"]["w1"]["var"])], [str(t) for t in ticks] + ["final"])
        else:
            axs[0, j].set_xlabel("sampling step")
    lo = min(min(min(v["var"]) for v in d["arms"].values()) for d in data)
    hi = max(max(max(v["var"]) for v in d["arms"].values()) for d in data)
    for ax in axs[0]:
        ax.set_ylim(lo / 1.3, hi * 1.3)
        decades = [10.0 ** k for k in range(-4, 4) if lo / 1.3 <= 10.0 ** k <= hi * 1.3]
        if len(decades) < 2:        # under two labelled decades: label 1-3-10 steps instead
            from matplotlib.ticker import FixedLocator, NullFormatter
            ys = [m * 10.0 ** k for k in range(-4, 4) for m in (1, 3) if lo / 1.3 <= m * 10.0 ** k <= hi * 1.3]
            ax.yaxis.set_major_locator(FixedLocator(ys))
            ax.set_yticklabels([f"{y:g}" for y in ys])
            ax.yaxis.set_minor_formatter(NullFormatter())
    axs[0, 0].set_ylabel(r"Var$(D_w(x_t,t))$ along the path")
    if a.weights:
        axs[1, 0].set_ylabel("$w$")
    pub_label = (f"published ($w={pubs[0]:g}$ CFG, ${pubs[1]:g}$ autog.)" if cols == 2
                 else f"published ($w={pubs[0]:g}$)")
    keys = [("w1", "$w=1$"), ("w2", "$w=2$"), ("w3", "$w=3$"), ("w5", "$w=5$"), ("pub", pub_label),
            ("vm", r"variance-matched $w^\star_{\mathrm{VM}}(t)$"), ("data", "ImageNet val images")]
    col = {**CONST, "pub": PUB, "vm": VM, "data": DATA}
    hs = [Line2D([], [], color=col[k], lw=1.5 if k == "vm" else (0.8 if k == "data" else 1.1),
                 ls=(0, (1.5, 1.5)) if k == "data" else "-") for k, _ in keys]
    if cols == 2:
        ncol, anchor = 3, (0.5, -0.36 if rows == 1 else -0.2)
    else:
        ncol, anchor = 2, (0.5, -0.5 if rows == 1 else -0.3)
    fig.legend(hs, [l for _, l in keys], loc="lower center", ncol=ncol, frameon=False,
               bbox_to_anchor=anchor, handlelength=1.6, columnspacing=0.9)
    from vmg import runs
    out = Path(a.out) if a.out else runs.root() / "paper" / "wvar-in64"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out}.png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}.{{pdf,png}}")


if __name__ == "__main__":
    main()
