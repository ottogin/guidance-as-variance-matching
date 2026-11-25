#!/usr/bin/env python
"""Fig. 1 (fig:teaser; panels fig:wone, fig:teaser-samples, fig:random_guidance, fig:teaser-nested).

    python reproduce/figures/fig_samples_sd35m.py --out paper-figures/ --teaser    # teaser/p2-*.jpg
    python reproduce/figures/fig_teaser.py --out paper-figures/ --photos           # everything else

Writes teaser-B.pdf, the whole figure except its photos on a 495 x 185.76 pt canvas (\\textwidth), and teaser-B.tex,
a picture environment that places the PDF and puts each photo in the cell left for it, so the photos keep their own
JPEG encodes; it also sets the four subfigure labels. --photos writes the photo rows that fig_samples_sd35m.py does
not: samples/<row>-{fid,fdd,ours}.jpg, from the user study's renders (sample.py --tag userstudy --fp32-state, or
the study's own JPEGs with --renders).
  (a) our constant weight (compute_w.py's global trace fit, mean pool: tab_objective.fits) against the published
      one (the model Specs) for the eleven models of tab:objective;
  (b) the column heads, the frame around ours and the subcaption of the photos (SD 3.5 Medium, COCO rows PHOTOS);
  (c) SD 1.5 and CommonCanvas guiding each other: D = D_CC + w (D_SD15 - D_CC), FID and FD_DINOv2 on the first
      1,000 COCO evaluation rows (tag 1k), every point sampled on the cross-sd15-cc adapter at --w w;
  (d) w*_checkpoint / w*_independent on EDM2-XS (reproduce/experiments/nested_in64.py analyze, fit): the mean of
      the four measured ratios per metric and r, their range as bars, and the fitted moving-average model.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

W, H = 495.0, 185.76                     # pt; 6.875 x 2.58 in
GRID, AX = "#e4e3dc", "#6b6a63"
# orange is ours (the samples); (c) and (d) sit side by side and share no color or marker, so a reader cannot map
# FID / FD_DINOv2 onto checkpoint / independent
BLUE, GREEN, PURPLE, ORANGE, MAGENTA = "#2a78d6", "#1baf7a", "#4a3aa7", "#eb6834", "#b8407f"
C_FID, C_FD = BLUE, PURPLE
C_NEST, C_IND = MAGENTA, GREEN

# (a): (model, dataset, label, modality, label position (left, baseline) in pt)
MODELS = [("sd15", "coco", "SD 1.5", "image", (142.5, 143.8)), ("sd35m", "coco", "SD 3.5 M", "image", (99.2, 110.7)),
          ("sd35l", "coco", "SD 3.5 L", "image", (103.4, 96.1)),
          ("flux2-klein", "mjhq", "FLUX.2 klein", "image", (65.7, 127.7)),
          ("svd", "ucf101", "SVD", "video", (74.7, 63.7)), ("wan", "openvid", "Wan 2.1", "video", (121.7, 118.8)),
          ("cogvideox", "openvid", "CogVideoX", "video", (91.3, 139.7)),
          ("audioldm2", "audiocaps", "AudioLDM 2", "audio", (59.7, 119.3)),
          ("sao", "songdescriber", "Stable Audio Open", "audio", (81.6, 153.8)),
          ("mdm", "kitml", "MDM", "motion", (86.2, 84.1)), ("cosmos", "bridge", "Cosmos", "world model", (40.2, 43.9))]
MOD = {"image": (BLUE, "o"), "video": (ORANGE, "o"), "audio": (GREEN, "o"), "motion": (PURPLE, "o"),
       "world model": (ORANGE, "D")}
MS = {"o": 4.6, "D": 4.2}

# (b): the photo cells (rows top to bottom, as Figures/<stem>-<arm>.jpg), their columns and size, in inches
PHOTOS = ("samples/5789", "teaser/p2", "samples/6198")
PHOTO_X, PHOTO_Y, PHOTO_SIDE = (2.4311, 3.0761, 3.7211), (1.5900, 0.9450, 0.3000), 0.62
HEAD_X = (197.3, 243.8, 290.3)           # pt, the centres of the column heads

# (c)
CROSS, TAG_C, N_C = "cross-sd15-cc", "1k", 1000
WS_C = (-4, -2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 5)
LIM_C = ((70, 98, [72, 80, 88, 96]), (860, 1100, [900, 1000, 1100]))     # the paper's axes, kept if the data fit


def cross_arms():
    """[(adapter, --w)] of panel (c), for reproduce/plan._arm."""
    return [(CROSS, float(w)) for w in WS_C]


def panel_a():
    from reproduce.tables.tab_objective import fits, published
    return [(label, published(m), fits(m, d)[0], mod, pos) for m, d, label, mod, pos in MODELS]


def panel_c():
    from reproduce.rosters.sweeps import arm
    rows = []
    for _, w in cross_arms():
        p = runs.metrics_json(CROSS, "coco", arm(w), TAG_C)
        assert p.exists(), f"{p} missing: sample and evaluate {CROSS} at --w {w:g} (--n {N_C} --tag {TAG_C})"
        r = json.loads(p.read_text())
        assert r["n"] == N_C, f"{p}: {r['n']} rows, not {N_C}"
        rows.append((w, r["fid"], r["fd_dinov2"]))
    return np.array(rows)


def panel_d():
    from reproduce.experiments import nested_in64 as NI
    summary = json.loads((NI.root() / "summary.json").read_text())
    fit = json.loads((NI.root() / "fit.json").read_text())
    return {m: [NI.ratios(summary, m, r) for r in NI.RS] for m in NI.METRICS}, fit["L_steps"]


def draw(out, png=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reproduce.experiments import nested_in64 as NI
    from reproduce.figures import fig_samples_sd35m as FS
    from reproduce.figures.fig_cross_pairs import nice_ticks
    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.2,
                         "axes.linewidth": 0.6, "axes.edgecolor": AX, "xtick.color": AX, "ytick.color": AX,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "pdf.fonttype": 42, "text.color": "black"})
    fig = plt.figure(figsize=(W / 72, H / 72))

    def axes_pt(x0, y0, x1, y1):
        return fig.add_axes([x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H])

    def text(x, y, s, **kw):
        fig.text(x / W, y / H, s, **kw)

    def open_frame(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # (a) log-log, the y = x line and a +-15% band
    rows = panel_a()
    from scipy.stats import spearmanr
    print(f"(a) Spearman rho = {spearmanr([r[1] for r in rows], [r[2] for r in rows])[0]:.2f}")
    a = axes_pt(28.8, 41.04, 146.88, 159.12)
    open_frame(a)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(0.9, 9)
    a.set_ylim(0.9, 9)
    t = [1, 1.5, 2, 3, 4, 5, 6, 8]
    for set_t, set_l in ((a.set_xticks, a.set_xticklabels), (a.set_yticks, a.set_yticklabels)):
        set_t(t)
        set_l([f"{v:g}" for v in t])
    a.minorticks_off()
    xx = np.geomspace(0.8, 10, 50)
    a.fill_between(xx, 0.85 * xx, 1.15 * xx, color=GRID, lw=0, zorder=0)
    a.plot(xx, xx, color=AX, lw=0.8, zorder=1)
    for label, pub, ours, mod, (lx, ly) in rows:
        col, mk = MOD[mod]
        a.plot(pub, ours, mk, ms=MS[mk], mfc=col, mec="white", mew=0.5, zorder=3)
        text(lx, ly, label, fontsize=6.2, ha="left", va="baseline")
    for k, (mod, (col, mk)) in enumerate(MOD.items()):
        y = 82.0 - 8.32 * k
        fig.add_artist(plt.Line2D([111.77 / W], [y / H], marker=mk, ms=MS[mk], mfc=col, mec="white", mew=0.5, ls=""))
        text(118.8, y - 2.2, mod, fontsize=6.4, ha="left", va="baseline")
    text(62.5, 84.3, "$\\pm15\\%$", fontsize=6.2, rotation=45, ha="left", va="baseline", color="#555555")
    a.set_xlabel("Published $w$", labelpad=2)
    a.set_ylabel("Predicted $w$", labelpad=2)
    text(87.8, 5, "(a) Predicted versus published guidance scale", fontsize=8.5, ha="center", va="baseline")

    # (b) column heads over the photo cells, a frame around ours
    ours = dict((r[0], r[2]) for r in rows)["SD 3.5 M"]
    wcol = {name: float(arm[1:]) for name, arm in FS.ARMS if arm.startswith("w")}
    heads = (("best FID", f"$w = {wcol['fid']:g}$", "black"),
             ("best FD$_\\mathrm{DINOv2}$", f"$w = {wcol['fdd']:g}$", "black"), ("ours", f"$w = {ours:.2f}$", ORANGE))
    for cx, (top, bottom, col) in zip(HEAD_X, heads):
        text(cx, 172.8, top, fontsize=6.6, ha="center", va="baseline", color=col)
        text(cx, 166.3, bottom, fontsize=6.6, ha="center", va="baseline", color=col)
    fig.add_artist(plt.Rectangle((266.91 / W, 20.59 / H), 46.66 / W, 139.54 / H, fill=False, ec=ORANGE, lw=0.9))
    text(244.4, 5, "(b) Samples at each choice of $w$", fontsize=8.5, ha="center", va="baseline")

    # (c) both directions, FID (left axis) and FD_DINOv2 (right) on one w axis, the base models' interval shaded
    both = panel_c()
    X0, X1 = 353, 461
    c = axes_pt(X0, 116, X1, 157)
    c2 = c.twinx()
    c.axvspan(0, 1, color="#efefed", lw=0, zorder=0)
    for v in (0, 1):
        c.axvline(v, color="#c9c9c5", lw=0.55, zorder=0)
    c.plot(both[:, 0], both[:, 1], "-o", color=C_FID, ms=2.8, lw=1.05, zorder=3)
    c2.plot(both[:, 0], both[:, 2], "-s", color=C_FD, ms=2.5, lw=1.05, zorder=3)
    c.set_xlim(-4.5, 5.5)
    c.set_xticks([-4, -2, 0, 1, 3, 5])
    for ax, ys, (lo, hi, ticks) in ((c, both[:, 1], LIM_C[0]), (c2, both[:, 2], LIM_C[1])):
        if not lo <= ys.min() <= ys.max() <= hi:
            pad = 0.08 * (ys.max() - ys.min())
            lo, hi, ticks = ys.min() - pad, ys.max() + pad, nice_ticks(ys.min(), ys.max(), 3)
        ax.set_ylim(lo, hi)
        ax.set_yticks(ticks)
    c.tick_params(axis="y", colors=C_FID)
    c2.tick_params(axis="y", colors=C_FD)
    c.tick_params(axis="x", pad=2)
    c.set_ylabel("FID $\\downarrow$", color=C_FID, labelpad=2)
    c2.set_ylabel("FD$_\\mathrm{DINOv2}$ $\\downarrow$", color=C_FD, labelpad=2)
    for ax in (c, c2):
        ax.spines["top"].set_visible(False)
    c2.spines["right"].set_color(AX)
    c.spines["left"].set_color(AX)
    c.set_xlabel("Guidance weight $w$", labelpad=1.5, fontsize=6.8)
    text(377, 160, "CommonCanvas away\nfrom SD 1.5", fontsize=6.2, ha="center", va="bottom", linespacing=1.05)
    text(447, 160, "SD 1.5 away from\nCommonCanvas", fontsize=6.2, ha="center", va="bottom", linespacing=1.05)
    text(408, 93, "$w{=}0$: CommonCanvas;  $w{=}1$: SD 1.5", fontsize=6.2, ha="center", va="baseline")
    text(408, 84, "(c) Guiding in both directions", fontsize=8.5, ha="center", va="baseline")

    # (d) measured ratios (mean of four, range as bars) and the moving-average model with its fitted L
    q, L = panel_d()
    rs = list(NI.RS)
    d = axes_pt(X0, 34, X1, 72)
    open_frame(d)
    k = np.arange(12, 101)
    d.plot(k / NI.T, [NI.ema_ratio(NI.T, int(x), np.exp(-1 / L)) for x in k], "--", color=C_NEST, lw=1.0,
           label="Theory", zorder=2)
    for m, mk, mfc, lab in (("fd_dinov2", "D", C_IND, "Measured, FD$_\\mathrm{DINOv2}$"),
                            ("fid", "o", "white", "Measured, FID")):
        mean = [x.mean() for x in q[m]]
        err = [[x.mean() - x.min() for x in q[m]], [x.max() - x.mean() for x in q[m]]]
        d.errorbar(rs, mean, yerr=err, fmt=mk + "-", ms=2.8, lw=0.9, mfc=mfc, mec=C_IND, mew=0.7, color=C_IND,
                   ecolor=C_IND, elinewidth=0.6, capsize=1.4, label=lab, zorder=4)
    d.axhline(1, color="#c9c9c5", lw=0.55, zorder=0)
    d.set_xticks(rs)
    d.set_xticklabels(["1/8", "1/4", "1/2", "3/4"])
    d.set_xlim(0.08, 0.8)
    d.set_ylim(0.85, 2.8)
    d.set_yticks([1, 2])
    d.grid(axis="y", color=GRID, lw=0.5)
    d.set_axisbelow(True)
    d.set_ylabel("$w^\\star_{\\mathrm{checkpoint}}/w^\\star_{\\mathrm{indep.}}$", labelpad=2)
    d.set_xlabel("Training fraction $r$ (weak / strong)", labelpad=1.5, fontsize=6.8)
    d.legend(frameon=False, fontsize=6.2, loc="upper left", bbox_to_anchor=(-0.02, 1.12), handlelength=1.3,
             borderaxespad=0, labelspacing=0.1, handletextpad=0.35)
    text(408, 5, "(d) Checkpoint vs. retrained weak model", fontsize=8.5, ha="center", va="baseline")
    print(f"(d) moving-average model at L = {L / NI.T:.2f} T")

    fig.savefig(out / "teaser-B.pdf", transparent=True)
    if png:
        fig.savefig(out / "teaser-B-layer.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(out / "teaser-B.pdf")


def tex():
    """The picture environment that places teaser-B.pdf and the photos (Figures/teaser-B.tex)."""
    from reproduce.figures import fig_samples_sd35m as FS
    arms = [name for name, _ in FS.ARMS]
    lines = ["% Fig. 1: generated by reproduce/figures/fig_teaser.py (the vector layer teaser-B.pdf and the photos).",
             r"\setlength{\unitlength}{1in}%", r"\begin{picture}(%.4f,%.4f)%%" % (W / 72, H / 72),
             r"\put(0.0000,0){\includegraphics{Figures/teaser-B.pdf}}%"]
    for x, name in zip(PHOTO_X, arms):
        for y, stem in zip(PHOTO_Y, PHOTOS):
            lines.append(r"\put(%.4f,%.4f){\includegraphics[width=%.4fin,height=%.4fin]{Figures/%s-%s.jpg}}%%"
                         % (x, y, PHOTO_SIDE, PHOTO_SIDE, stem, name))
    lines += [r"\end{picture}%",
              r"{\phantomsubcaption\label{fig:wone}}{\phantomsubcaption\label{fig:teaser-samples}}"
              r"{\phantomsubcaption\label{fig:random_guidance}}{\phantomsubcaption\label{fig:teaser-nested}}%", ""]
    return "\n".join(lines)


def photos(out, renders=None):
    """samples/<row>-<arm>.jpg of the PHOTOS rows under samples/, from the user study's renders."""
    from PIL import Image
    from reproduce.figures import fig_samples_sd35m as FS
    study = {"fid": "c1.6", "fdd": "c2.4", "ours": "c4.14"}       # the study's arms behind fig_samples_sd35m.ARMS
    for stem in (s for s in PHOTOS if s.startswith("samples/")):
        row = int(stem.split("/")[1])
        for name, _ in FS.ARMS:
            arm = study[name]
            study_dir = runs.root() / "sd35m" / f"coco-{FS.USERSTUDY_TAG}" / "samples"
            src = (Path(renders) / arm / f"{row:06d}.jpg" if renders
                   else study_dir / FS.SAMPLE_ARM[arm] / f"{row:05d}.png")
            dst = out / f"{stem}-{name}.jpg"
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.open(src).convert("RGB").save(dst, quality=FS.QUALITY)
    print(f"photo rows {[s for s in PHOTOS if s.startswith('samples/')]} -> {out}/samples/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the paper's Figures folder (or a scratch copy)")
    ap.add_argument("--photos", action="store_true", help="also write the samples/ photo rows")
    ap.add_argument("--renders", help="--photos from the study's own JPEGs (<dir>/<arm>/<coco id>.jpg)")
    ap.add_argument("--png", action="store_true", help="also a 300 dpi PNG of the vector layer")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    draw(out, a.png)
    (out / "teaser-B.tex").write_text(tex())
    if a.photos:
        photos(out, a.renders)


if __name__ == "__main__":
    main()
