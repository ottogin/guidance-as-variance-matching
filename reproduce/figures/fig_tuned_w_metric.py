#!/usr/bin/env python
"""Paper fig:tuned-w-metric: the best constant guidance scale for each metric, against the released scale and ours.

    python reproduce/figures/fig_tuned_w_metric.py --out paper-figures/tuned-w-metric-models [--allow-missing]

One panel per model (SD 3.5 Medium and SD 1.5, both on COCO), one row per metric, log axis. For each metric m,
w*_m is the best weight of the model's sweep (reproduce/rosters/sweeps.py; ties go to the smaller w): SD 3.5
Medium's 42 weights and SD 1.5's 0.5 grid from w = 1. An open marker is a best weight at the lower end of the grid
(w = 1). Two vertical lines per panel: the released scale (the `pub` arm, a pale band) and our variance-matched
constant (the `vm-global` arm, black), with their values in a strip under the header. The released scale is a band
because the two can coincide on this axis (SD 1.5: 7.5 and 7.44), where a second thin line would be hidden.
Reads runs/<model>/<dataset>/metrics/<arm>.json.

The layout is the paper's (one column, 3.16 x 1.735 in), set in pixels of that page at 600 dpi. Text is Times New
Roman where it is installed and STIX otherwise.
"""
import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.rosters import sweeps  # noqa: E402
from vmg import runs  # noqa: E402

# metric key -> sign: the best w minimises sign * value
METRICS = {"fid": 1, "fd_dinov2": 1, "precision_dinov2": -1, "recall_dinov2": -1, "precision_inception": -1,
           "recall_inception": -1, "clip": -1}
INK, RULE, GRID, MUTED = "#252525", "#92928e", "#e4e4e1", "#686868"
BLUE, ORANGE, GREEN, PURPLE = "#2569ad", "#c05727", "#19856c", "#77549a"      # distance, precision, recall, alignment
TINT = {BLUE: "#f5f8fb", ORANGE: "#fcf7f5", GREEN: "#f5f9f8", PURPLE: "#f9f7fa"}
# rows, top to bottom: (metric key, label, colour)
ROWS = [("fid", "FID", BLUE), ("fd_dinov2", r"FD$_{\rm DINOv2}$", BLUE),
        ("precision_inception", r"$P_{\rm Inc}$", ORANGE), ("recall_inception", r"$R_{\rm Inc}$", GREEN),
        ("precision_dinov2", r"$P_{\rm DINOv2}$", ORANGE), ("recall_dinov2", r"$R_{\rm DINOv2}$", GREEN),
        ("clip", "CLIP", PURPLE)]
# (model, dataset, panel title, sweep, marker, marker size in pt)
MODELS = [("sd35m", "coco", "SD 3.5 Medium", sweeps.SD35M, "o", 4.0),
          ("sd15", "coco", "SD 1.5", sweeps.SD15, "s", 3.7)]

DPI = 600
PT = DPI / 72                            # pixels per point
W, H = 1896, 1041                        # the page
X0, X1, GAP = 429, 1874, 80              # the panels' span (the row labels are to its left) and the gap between panels
LO, HI = math.log2(0.855), math.log2(18.06)
TOP, BOT = 84.5, 867.5                   # the rules above and below the rows
BAND0, BAND1 = 89, 846.5                 # the strip and the tinted rows
STRIP = 62                               # the strip that carries the two lines' values
FS_LAB, FS_VAL = 7.2, 6.8                # row and tick labels; values
OURS, PUB, PUB_BAND = INK, "#77776f", "#b4b4ad"
LW, LW_BAND = 0.9, 2.6


def load(model, dataset, ws):
    rows = {}
    for w in ws:
        p = runs.root() / model / dataset / "metrics" / f"{sweeps.arm(w)}.json"
        if p.exists():
            rows[w] = json.load(open(p))
    return rows


def best(rows):
    """{metric: w*}; ties go to the smaller w."""
    out = {}
    for k, sign in METRICS.items():
        have = [w for w in rows if k in rows[w]]
        if have:
            out[k] = min(have, key=lambda w: (sign * rows[w][k], w))
    return out


def lines(model, dataset):
    """(released scale, our constant): the mean weights of the pub and vm-global arms, None where an arm is missing."""
    out = []
    for arm in ("pub", "vm-global"):
        p = runs.root() / model / dataset / "metrics" / f"{arm}.json"
        out.append(json.load(open(p))["w_mean"] if p.exists() else None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--allow-missing", action="store_true",
                    help="draw the models whose sweeps exist, without the lines of a model whose pub or vm-global arm "
                         "is missing")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                                "mathtext.fontset": "cm", "pdf.fonttype": 42, "ps.fonttype": 42})
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    done = []
    for model, ds, name, ws, mk, ms in MODELS:
        rows = load(model, ds, ws)
        missing = len(ws) - len(rows)
        assert not missing or a.allow_missing, f"{model}/{ds}: {missing} of {len(ws)} sweep arms have no metrics"
        pub, ours = lines(model, ds)
        assert (pub is not None and ours is not None) or a.allow_missing, f"{model}/{ds}: no pub or vm-global arm"
        if len(rows) >= 3:
            b = best(rows)
            done.append((name, mk, ms, b, pub, ours))
            print(f"{name} ({len(rows)} weights): " + ", ".join(f"{k} {w:g}" for k, w in b.items())
                  + f"; released {pub}, ours {ours}")
    assert done, "no sweep to draw"

    fig = plt.figure(figsize=(W / DPI, H / DPI))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    def extent(t):
        """A text's horizontal extent, in page pixels."""
        fig.canvas.draw()
        p = ax.transData.inverted().transform(t.get_window_extent().get_points())
        return min(p[:, 0]), max(p[:, 0])

    ys = BAND0 + STRIP / 2 + 3
    b0 = BAND0 + STRIP
    bh = (BAND1 - b0) / len(ROWS)
    yrow = [b0 + (i + 0.5) * bh for i in range(len(ROWS))]
    for i, (_, lab, col) in enumerate(ROWS):
        ax.add_patch(Rectangle((6, b0 + i * bh), W - 14, bh, fc=TINT[col], ec="none", zorder=0))
        ax.text(35, yrow[i] - 7, lab, color=col, fontsize=FS_LAB, ha="left", va="center_baseline", zorder=5)

    pw = (X1 - X0 - GAP * (len(done) - 1)) / len(done)
    for j, (name, mk, ms, b, pub, ours) in enumerate(done):
        left = X0 + j * (pw + GAP)

        def px(w, left=left):
            return left + (math.log2(w) - LO) / (HI - LO) * pw

        for y in (TOP, BOT):
            ax.plot([left, left + pw], [y, y], color=RULE, lw=0.5, solid_capstyle="butt", zorder=4)
        for t in (1, 4, 16):
            ax.plot([px(t)] * 2, [TOP + 2, BOT - 2], color=GRID, lw=0.42, solid_capstyle="butt", zorder=1)
            ax.plot([px(t)] * 2, [BOT, BOT + 16], color=RULE, lw=0.5, solid_capstyle="butt", zorder=4)
            ax.text(px(t), BOT + 28, str(t), color=INK, fontsize=FS_LAB, ha="center", va="top")
        ax.text(left + pw / 2, TOP - 36, name, color=INK, fontsize=7.5, fontweight="bold", ha="center", va="baseline")

        taken = []                                              # the x-intervals the two lines occupy
        if pub is not None and ours is not None:
            xo, xp = px(ours), px(pub)
            ax.plot([xp, xp], [BAND0 + 6, BAND1], color=PUB_BAND, lw=LW_BAND, solid_capstyle="butt", zorder=2)
            ax.plot([xo, xo], [BAND0 + 6, BAND1], color=OURS, lw=LW, solid_capstyle="butt", zorder=3)
            taken = [(xp - LW_BAND * PT / 2, xp + LW_BAND * PT / 2), (xo - LW * PT / 2, xo + LW * PT / 2)]
            near = abs(xo - xp) < 9 * PT                        # close lines: one value on each side of the pair
            so = (-1 if ours <= pub else 1) if near else 1
            for x, txt, col, wt, side, lw in ((xo, f"{ours:.2f}", OURS, "bold", so, LW),
                                              (xp, f"{pub:g}", PUB, "normal", -so if near else 1, LW_BAND)):
                ax.text(x + side * (lw / 2 + 1.2) * PT, ys, txt, color=col, fontsize=FS_VAL, fontweight=wt,
                        ha="left" if side > 0 else "right", va="center_baseline", zorder=5)

        def clear(t, pad=3):
            lo, hi = extent(t)
            return all(hi + pad < u or lo - pad > v for u, v in taken) and hi <= left + pw + GAP / 2

        for i, (k, _, col) in enumerate(ROWS):
            if k not in b:
                ax.text(px(4), yrow[i] - 5, "n/a", color=MUTED, fontsize=FS_VAL, ha="center", va="center_baseline", zorder=5)
                continue
            w = b[k]
            edge = w == 1                                       # the lower end of the searched grid
            ax.plot([px(w)], [yrow[i]], mk, ms=ms - (0.9 if edge else 0), mfc="white" if edge else col, mec=col,
                    mew=0.9 if edge else 0, zorder=6, clip_on=False)

            def value(side, w=w, i=i, col=col):
                return ax.text(px(w) + side * 33, yrow[i] - 5, f"{w:g}", color=col, fontsize=FS_VAL,
                               ha="right" if side < 0 else "left", va="center_baseline", zorder=5)

            # the value sits left of its marker, or right of it where a line would run through it
            t = value(-1)
            if not clear(t):
                r = value(1)
                (t if clear(r) else r).remove()                 # neither side is clear: it stays on the left

    if any(pub is not None and ours is not None for *_, pub, ours in done):     # the key, in the label column
        x = 35
        for lab, col, wt, lc, lw in (("Ours", OURS, "bold", OURS, LW), ("Pub.", PUB, "normal", PUB_BAND, LW_BAND)):
            t = ax.text(x, ys, lab, color=col, fontsize=FS_VAL, fontweight=wt, ha="left", va="center_baseline")
            x = extent(t)[1] + 2 * PT
            ax.plot([x, x + 8.5 * PT], [ys - 2, ys - 2], color=lc, lw=lw, solid_capstyle="butt")
            x += 8.5 * PT + 5 * PT

    ax.text((X0 + X1) / 2, 1006, r"Metric-optimal constant $w_m^\star$ (log scale)", color=INK, fontsize=7.6,
            ha="center", va="baseline")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out + ".pdf")
    fig.savefig(a.out + ".png", dpi=DPI)
    print(f"wrote {a.out}.pdf")


if __name__ == "__main__":
    main()
