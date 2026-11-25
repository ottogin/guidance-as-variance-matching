#!/usr/bin/env python
"""Fig. 2 (fig:concept, fig:story): guidance as variance matching, with 1,024 trained 2-D denoisers per N.

    python reproduce/figures/fig_concept_story.py --src $VMG_RUNS/_toys/story2d --out concept_story

Reads reproduce/toys/story2d.py's <optim>-n<N>.npz (Adam and SGD, N = 64 .. 4,096) and writes <out>.pdf, <out>.png.
Everything is at one input x_t, input 41 of the 256 at sigma = 0.4. Left: the data with x_t; (a) the posterior
p(x0 | x_t), mean mu and 2-s.d. ellipse of Sigma. (b) The predictions of the 1,024 Adam models at x_t for
N = 256, 64, 2,048, 512 (insets: magnified, with Sigma rescaled to the cloud's trace). (c) Model 657 of the weak
N = 64 and the strong N = 512 pools (circled in (b)): the difference, then the guided prediction at w*, with the
2-s.d. ellipse of all 1,024 guided predictions. (d) The guided predictions of the pairs 2,048 / 256 and 512 / 64,
each at the larger w* with tr Cov over the 1,024 pairs equal to tr Sigma. (e) The clouds' trace over tr Sigma and
long-axis angle against N, for Adam, SGD and the exact posterior mean of each model's training set (gray); the
bands are 5-95% bootstrap intervals (300 resamples) of the angle, the dotted line is Sigma's axis.
"""
import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.mathtext import MathTextParser  # noqa: E402
from matplotlib.patches import ConnectionPatch, Ellipse, FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.toys.story2d import MEANS, NS, S_COMP, posterior, posterior_mean_cov  # noqa: E402

SIGMA, INPUT, RUN = 0.4, 41, 657            # the noise level, the input (of 256) and the circled model index
OPTS = ("adam", "sgd")
PAIRS = [(2048, 256), (512, 64)]            # (strong, weak) training-set sizes of (d); the second is (c)'s pair
BLOCK = [256, 64, 2048, 512]                # (b), row by row

INK, FRAME, PANEL_FRAME = "#25252A", "#C8CBD1", "#666971"
BLACK, WEAK, STRONG, ORANGE = "#000000", "#927BB9", "#512185", "#D65E18"
OPTC = {"adam": "#8E44AD", "sgd": "#3B6FB6"}
GREY_ARROW, GREY_DASH, DASH = "#868A91", (0, (2.6, 1.7)), (0, (3.4, 2.0))
POSTMAP = LinearSegmentedColormap.from_list("post", ["#FFFFFF", "#BBC3CB"])
HEAT = LinearSegmentedColormap.from_list("heat", ["#FFFFFF", "#A7AFBA"])
EXL = dict(color="#C3C8CE", lw=2.2, solid_capstyle="round", zorder=1)
POSTL = dict(color=INK, lw=0.7, ls=":")
W, H = 6.875, 1.54                          # inches
PANEL_W, PANEL_H = 0.70, 0.56
ROW_Y = (0.735, 0.13)
TT_Y1, TT_DY, TT_FS = 1.415, 0.092, 6.5
MECH_X, GUIDED_X = 3.91, 4.81
BOX = 1.0                                   # half-width of the small panels, data units
GUIDED = r"D_{w^\star}"
MATH = MathTextParser("path")

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 6.5,
                     "axes.titlesize": 6.8, "legend.fontsize": 5.6, "axes.linewidth": 0.5, "xtick.labelsize": 5.8,
                     "ytick.labelsize": 5.8, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                     "xtick.minor.size": 1.2, "ytick.minor.size": 1.2, "xtick.major.width": 0.5,
                     "ytick.major.width": 0.5, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.edgecolor": PANEL_FRAME, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "figure.dpi": 180, "savefig.pad_inches": 0})


# ------------------------------------------------------------------ data
def shape(S):
    """Long-axis angle (deg, mod 180) and aspect ratio of a 2x2 covariance."""
    e, V = np.linalg.eigh(S)
    return math.degrees(math.atan2(V[1, -1], V[0, -1])) % 180, e[-1] / e[0]


def spread(D, trC, rng=None, nb=300):
    """Size tr Cov(D) / tr Sigma and long-axis angle of a cloud D [R, 2]; with rng, the angle's 5-95% bootstrap."""
    S = np.cov(D.T)
    a = shape(S)[0]
    out = dict(size=np.trace(S) / trC, angle=a)
    if rng is not None:
        an = [((shape(np.cov(D[rng.integers(0, len(D), len(D))].T))[0] - a + 90) % 180) - 90 + a
              for _ in range(nb)]
        out["angle_ci"] = np.percentile(an, [5, 95])
    return out


def ring_box(m, S):
    """Corners of the bounding box of the 2-s.d. ellipse of S around m."""
    e, V = np.linalg.eigh(S)
    t = np.linspace(0, 2 * np.pi, 720)
    P = m + (V @ (2 * np.sqrt(e)[:, None] * np.stack([np.cos(t), np.sin(t)]))).T
    return P.min(0), P.max(0)


def load(src):
    """Every quantity the figure draws, from the Adam and SGD pools at input INPUT, sigma SIGMA."""
    z = {(o, n): np.load(Path(src) / f"{o}-n{n}.npz") for o in OPTS for n in NS}
    S = SimpleNamespace()
    S.xs = z["adam", NS[0]][f"xt_{SIGMA}"][INPUT].astype(float)
    S.cloud = {k: v[f"pred_{SIGMA}"][:, INPUT].astype(float) for k, v in z.items()}
    exact = {n: z["adam", n][f"ideal_{SIGMA}"][:, INPUT].astype(float) for n in NS}   # the same sets for SGD
    mu, C = posterior_mean_cov(S.xs[None], SIGMA)
    S.mu, S.C, S.pa = mu[0], C[0], shape(C[0])[0]
    S.post = posterior(S.xs[None], SIGMA)
    trC = np.trace(S.C)
    rng = np.random.default_rng(0)
    S.st = {o: {n: spread(S.cloud[o, n], trC, rng) for n in NS} for o in OPTS}
    S.si = {n: spread(exact[n], trC) for n in NS}
    S.gp = []
    for n1, n0 in PAIRS:                    # the larger root of tr Cov_r(D0 + w (D1 - D0)) = tr Sigma
        P0, P1 = S.cloud["adam", n0], S.cloud["adam", n1]
        c0, cd = P0 - P0.mean(0), (P1 - P0) - (P1 - P0).mean(0)
        a, b, c = (cd * cd).sum(1).mean(), 2 * (c0 * cd).sum(1).mean(), (c0 * c0).sum(1).mean() - trC
        w = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        S.gp.append(dict(ns=n1, nw=n0, w=w, G=P0 + w * (P1 - P0)))
    S.S0, S.S1, S.Gp = S.cloud["adam", 64], S.cloud["adam", 512], S.gp[1]["G"]
    S.D0, S.D1, S.DW = S.S0[RUN], S.S1[RUN], S.Gp[RUN]
    S.cen = 0.5 * (S.mu + S.xs) + [0, 0.2]
    bxs = [ring_box(S.mu, S.C), ring_box(S.Gp.mean(0), np.cov(S.Gp.T))]
    lo, hi = np.min([b[0] for b in bxs], 0), np.max([b[1] for b in bxs], 0)
    pad = np.array([0.05, 0.08]) * (hi - lo)
    S.LO, S.HI = lo - pad, hi + pad
    S.LO[1] = min(S.LO[1], S.xs[1] - 0.22)
    S.ASP = (S.HI[1] - S.LO[1]) / (S.HI[0] - S.LO[0])
    return S


# ------------------------------------------------------------------ drawing primitives
def ellipse(ax, mu, cov, k=2.0, **kw):
    e, V = np.linalg.eigh(cov)
    ax.add_patch(Ellipse(mu, 2 * k * math.sqrt(e[1]), 2 * k * math.sqrt(e[0]),
                         angle=math.degrees(math.atan2(V[1, 1], V[0, 1])), fill=False, **kw))


def clean(ax, c):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(c)
        sp.set_linewidth(0.5)


def density(S, kind, G):
    """The posterior density at x_t ("post") or the data density ("data") on a grid G [..., 2]."""
    if kind == "post":
        wk, cmk, vk = S.post
        return (wk[0] * np.exp(-0.5 * ((G[..., None, :] - cmk[0]) ** 2).sum(-1) / vk)).sum(-1) / (2 * math.pi * vk)
    v = S_COMP ** 2
    return np.exp(-0.5 * ((G[..., None, :] - MEANS) ** 2).sum(-1) / v).mean(-1) / (2 * math.pi * v)


def heat(ax, S, kind, lo, hi, n, cmap, gamma):
    g = np.linspace(0, 1, n)
    X, Y = np.meshgrid(lo[0] + (hi[0] - lo[0]) * g, lo[1] + (hi[1] - lo[1]) * g)
    Z = density(S, kind, np.stack([X, Y], -1))
    ax.imshow(Z, extent=[lo[0], hi[0], lo[1], hi[1]], origin="lower", cmap=cmap, norm=PowerNorm(gamma, 0, Z.max()),
              interpolation="bilinear", zorder=0)


def big_panel(ax, S, kind, lo, hi):
    """Density on [lo, hi] as the left panels draw it, limits set, framed."""
    heat(ax, S, kind, lo, hi, 420, POSTMAP, 0.55)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    clean(ax, FRAME)
    for sp in ax.spines.values():
        sp.set_linewidth(0.45)


def small_panel(ax, S):
    """The shared view of (b)-(d): the posterior density around mu and the 2-s.d. ellipse of Sigma."""
    view = np.array([S.mu[0], S.cen[1]])
    heat(ax, S, "post", view - BOX, view + BOX, 400, HEAT, 0.5)
    ellipse(ax, S.mu, S.C, 2, color=INK, lw=0.5, ls="--", zorder=2)
    ax.set_xlim(S.mu[0] - BOX, S.mu[0] + BOX)
    ax.set_ylim(S.cen[1] - BOX * PANEL_H / PANEL_W, S.cen[1] + BOX * PANEL_H / PANEL_W)
    ax.set_aspect("equal")
    clean(ax, FRAME)


def axat(fig, x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def title(fig, x, y, letter, text, size):
    """'(letter) text' with its baseline at (x, y) inches, the letter in bold."""
    r = fig.canvas.get_renderer()
    for t, kw in (((f"({letter})" if letter else ""), dict(fontweight="bold")), (text, {})):
        if t:
            h = fig.text(x / W, y / H, t, va="baseline", fontsize=size, **kw)
            x += h.get_window_extent(r).width / fig.dpi + 0.035


def title2(fig, x, letter, l1, l2):
    """Two-line panel title; the second line starts under the first line's text."""
    title(fig, x, TT_Y1, letter, l1, TT_FS)
    h = fig.text(0, 0, f"({letter})", fontsize=TT_FS, fontweight="bold")
    w = h.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    h.remove()
    fig.text((x + w + 0.035) / W, (TT_Y1 - TT_DY) / H, l2, fontsize=TT_FS, va="baseline")


def arrow(ax, p, q, col, lw, shrink):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=3.2,head_width=1.6", color=col, lw=lw,
                                 shrinkA=0, shrinkB=shrink, zorder=7, mutation_scale=1))


def dashed_arrow(fig, a, b, z=9):
    """A dashed grey shaft and a small solid head, from a to b in figure inches."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    hb = b - (b - a) / np.linalg.norm(b - a) * 0.035
    for p, q, style, ls in ((a, hb, "-", GREY_DASH), (hb, b, "-|>", "-")):
        fig.add_artist(FancyArrowPatch((p[0] / W, p[1] / H), (q[0] / W, q[1] / H), transform=fig.transFigure,
                                       arrowstyle=style, mutation_scale=4.5, color=GREY_ARROW, lw=0.65, ls=ls,
                                       shrinkA=0, shrinkB=0, zorder=z))


def dashed_line(fig, p, q):
    fig.add_artist(Line2D([p[0] / W, q[0] / W], [p[1] / H, q[1] / H], transform=fig.transFigure, color=GREY_ARROW,
                          lw=0.65, ls=GREY_DASH, zorder=6))


def mpoint(ax, p, col, ms):
    ax.plot(*p, "o", ms=ms, color=col, mec="white", mew=0.4, zorder=10)


def ring_text(ax, mu, C, k, pieces, fs, center_dir, gap_pt, kern_pt):
    """Write `pieces` one after another just outside the k-s.d. ellipse of C around mu, centred at the ring point in
    direction center_dir, each rotated to the local tangent (reading left to right)."""
    e, V = np.linalg.eigh(C)
    a, b = k * math.sqrt(e[1]), k * math.sqrt(e[0])
    th = np.linspace(0, 2 * math.pi, 4001)
    P = mu[None] + a * np.cos(th)[:, None] * V[:, 1][None] + b * np.sin(th)[:, None] * V[:, 0][None]
    d0, d1 = ax.transData.transform(mu), ax.transData.transform(mu + np.array([1.0, 0.0]))
    pt_per_data = float(np.hypot(*(d1 - d0))) * 72.0 / ax.figure.dpi                   # equal aspect
    arc = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    dirs = P - mu[None]
    ic = int(np.argmax(dirs @ (center_dir / np.linalg.norm(center_dir)) / np.linalg.norm(dirs, axis=1)))
    widths = [TextPath((0, 0), t, size=fs, prop=FontProperties()).get_extents().width + kern_pt for t in pieces]
    tang = P[min(ic + 1, len(P) - 1)] - P[max(ic - 1, 0)]
    sign = 1 if tang[0] > 0 else -1
    s = arc[ic] - sign * sum(widths) / pt_per_data / 2
    for t, w in zip(pieces, widths):
        wd = w / pt_per_data
        j = int(np.argmin(np.abs(arc - ((s + sign * wd / 2) % arc[-1]))))
        tg = P[min(j + 1, len(P) - 1)] - P[max(j - 1, 0)]
        tg = tg / np.linalg.norm(tg) * sign
        nrm = np.array([tg[1], -tg[0]])
        if np.dot(nrm, P[j] - mu) < 0:
            nrm = -nrm
        ax.annotate(t, P[j], xytext=gap_pt * nrm, textcoords="offset points",
                    rotation=math.degrees(math.atan2(tg[1], tg[0])), rotation_mode="anchor", ha="center",
                    va="top", fontsize=fs, zorder=9, annotation_clip=False)
        s += sign * wd


def ctext(fig, xc, y, expression, colors, fs):
    """Typeset a formula once, centred at (xc, y) inches, and draw each glyph in its own color."""
    parsed = MATH.parse(expression, 72, FontProperties(family="STIXGeneral", size=fs))
    assert len(colors) == len(parsed.glyphs) and not parsed.rects
    origin = np.array([xc - parsed.width / 144, y]) * fig.dpi
    for g, col in zip(parsed.glyphs, colors):
        font, size, char = g[:3]                     # matplotlib >= 3.11 adds a glyph index before the offsets
        pos = fig.transFigure.inverted().transform(origin + np.array(g[-2:]) * fig.dpi / 72)
        fig.text(*pos, chr(char), fontproperties=FontProperties(fname=font.fname, size=size), color=col,
                 ha="left", va="baseline", zorder=12)


# ------------------------------------------------------------------ panels
def left(fig, S):
    """The data with x_t and its zoom window, and (a) the posterior."""
    yb, blk, dw, aw = 0.19, 1.165, 0.76, 1.32
    ah, dh = aw * S.ASP, dw * 0.8
    ya = yb + (blk - ah) / 2
    ad = axat(fig, 0.02, ya + (ah - dh) / 2, dw, dh)
    big_panel(ad, S, "data", np.array([-2.5, -2.0]), np.array([2.5, 2.0]))
    ad.plot(*S.xs, "o", color=INK, ms=2.4, zorder=5)
    ad.add_patch(Rectangle(S.LO, *(S.HI - S.LO), fill=False, lw=0.6, ec=INK, zorder=6))
    title(fig, 0.02, ya + (ah - dh) / 2 + dh + 0.05, "", r"Data $p_{\mathrm{data}}$", 6.9)
    aa = axat(fig, 0.87, ya, aw, ah)
    big_panel(aa, S, "post", S.LO, S.HI)
    ellipse(aa, S.mu, S.C, 2, color=INK, lw=0.7, ls=DASH, zorder=3)
    aa.plot(*S.mu, marker="x", color=INK, ms=3.4, mew=0.8, zorder=6)
    aa.plot(*S.xs, "o", color=INK, ms=2.6, zorder=9)
    aa.annotate("$x_t$", S.xs, xytext=(-6, 1), textcoords="offset points", fontsize=6.6, color=INK, zorder=9,
                ha="right", va="center")
    arrow(aa, S.xs, S.mu, "#55575E", 0.7, 3.5)
    e, V = np.linalg.eigh(S.C)
    u = V[:, 1] * (-1 if V[0, 1] > 0 else 1)                                       # long axis, to the upper left
    ang = math.degrees(math.atan2(u[1], u[0]))
    n = np.array([-u[1], u[0]])
    aa.annotate(r"$\mu=\mathbb{E}[x_0|x_t]$", S.mu, xytext=4.0 * n * np.sign(n[1]), textcoords="offset points",
                rotation=ang - 180 if abs(ang) > 90 else ang, rotation_mode="anchor", ha="center", va="bottom",
                fontsize=6.0, zorder=9)
    ring_text(aa, S.mu, S.C, 2, ["$\\Sigma$", "$=$", "Cov", "(", "$x_0$", "$|$", "$x_t$", ")"], fs=6.0,
              center_dir=np.array([0.88, -0.85]), gap_pt=2.4, kern_pt=0.5)
    title(fig, 0.87, ya + ah + 0.05, "a", r"Posterior $p(x_0\mid x_t)$", 6.9)
    fig.text((0.87 + aw / 2) / W, (ya - 0.13) / H, r"a diffusion model predicts only $\mu$", ha="center",
             va="baseline", fontsize=5.9)
    for (xa, ya_), xyb in (((S.HI[0], S.HI[1]), (0, 1)), ((S.HI[0], S.LO[1]), (0, 0))):
        fig.add_artist(ConnectionPatch((xa, ya_), xyb, "data", "axes fraction", axesA=ad, axesB=aa,
                                       color="#90949B", lw=0.45))


def models(fig, S):
    """(b): the 1,024 Adam models per N, true scale, with magnified insets; returns the axes by N."""
    xb, sg, axes = 2.30, 0.045, {}
    for k, n in enumerate(BLOCK):
        row, col = divmod(k, 2)
        ax = axat(fig, xb + col * (PANEL_W + sg), ROW_Y[row], PANEL_W, PANEL_H)
        small_panel(ax, S)
        P, color = S.cloud["adam", n], WEAK if n in (64, 256) else STRONG
        ax.scatter(*P.T, s=0.5, color=color, lw=0, zorder=3, rasterized=True)
        ax.text(0.04, 0.95, f"$N={n}$", transform=ax.transAxes, fontsize=5.4, va="top")
        Sm, c0 = np.cov(P.T), P.mean(0)
        half = 3.6 * math.sqrt(np.linalg.eigvalsh(Sm)[1])
        iw = 0.31
        ins = ax.inset_axes([0.02, 0.03, iw, iw * PANEL_W / PANEL_H])
        ins.scatter(*P.T, s=0.4, color=color, lw=0, rasterized=True)
        ellipse(ins, c0, S.C * np.trace(Sm) / np.trace(S.C), 2, color=INK, lw=0.5, ls="--")
        ins.set_xlim(c0[0] - half, c0[0] + half)
        ins.set_ylim(c0[1] - half, c0[1] + half)
        ins.set_aspect("equal")
        clean(ins, "#92969E")
        ins.text(0.05, 0.95, f"×{BOX / half:.0f}", transform=ins.transAxes, fontsize=4.8, va="top")
        if n in (64, 512):
            ax.plot(*P[RUN], "o", ms=4.6, mfc="none", mec=INK, mew=0.7, zorder=8)
        axes[n] = ax
    title2(fig, xb, "b", "Prediction errors follow", r"scaled-down $\Sigma$")
    return axes


def curves(fig, S):
    """(e): spread over tr Sigma and long-axis angle against N."""
    x, yl, ph = 5.62, 0.28, 0.3525
    pw = W - x - yl - 0.03
    b1, b2 = axat(fig, x + yl, 1.285 - (ph - 0.05), pw, ph - 0.05), axat(fig, x + yl, 0.58, pw, ph)
    for o in OPTS:
        b1.plot(NS, [S.st[o][n]["size"] for n in NS], "o-", color=OPTC[o], ms=1.6, lw=0.7, zorder=3)
    b1.plot(NS, [S.si[n]["size"] for n in NS], **EXL)
    b1.set_yscale("log")
    for o in OPTS:
        b2.fill_between(NS, *np.array([S.st[o][n]["angle_ci"] for n in NS]).T, color=OPTC[o], alpha=0.15, lw=0)
        b2.plot(NS, [S.st[o][n]["angle"] for n in NS], "o-", color=OPTC[o], ms=1.6, lw=0.7, zorder=3)
    b2.plot(NS, [S.si[n]["angle"] for n in NS], **EXL)
    b2.axhline(S.pa, **POSTL)
    b2.set_ylim(S.pa - 11, S.pa + 5)
    for b in (b1, b2):
        b.set_xscale("log", base=2)
        b.set_xticks(NS[::2])
        b.set_xticklabels([str(n) if n < 1000 else f"{n // 1024}k" for n in NS[::2]])
        b.minorticks_off()
        b.tick_params(labelsize=5.2, length=1.8, width=0.4, pad=1.2)
        b.patch.set_alpha(0.92)
        for sp in b.spines.values():
            sp.set_linewidth(0.45)
    b1.set_ylabel("spread / tr Σ", labelpad=1, fontsize=5.4)
    b1.set_xticklabels([])
    b1.set_yticks([0.1, 0.01])
    b1.set_yticklabels(["0.1", "0.01"])
    b2.set_yticks([150, 160])
    b2.set_ylabel("axis (deg)", labelpad=1, fontsize=5.4)
    b2.set_xlabel("training-set size N", fontsize=5.6, labelpad=0.5)
    title2(fig, x, "e", r"Errors shrink as $1/N$,", r"aligned with $\Sigma$")
    hh = [Line2D([], [], color=OPTC[o], marker="o", ms=2, lw=0.7) for o in OPTS]
    fig.legend(hh + [Line2D([], [], **EXL), Line2D([], [], **POSTL)],
               ["Adam", "SGD", "exact mean of N points", "posterior"], loc="upper left",
               bbox_to_anchor=((x - 0.04) / W, 0.39 / H), ncol=2, columnspacing=0.5, handletextpad=0.3,
               handlelength=1.1, fontsize=5.0, frameon=False, borderaxespad=0, labelspacing=0.2)


def guided(fig, S):
    """(d): the 1,024 guided predictions of each pair at its w*."""
    for gp, y in zip(S.gp, ROW_Y):
        ax = axat(fig, GUIDED_X, y, PANEL_W, PANEL_H)
        small_panel(ax, S)
        ax.scatter(*gp["G"].T, s=0.5, color=ORANGE, lw=0, zorder=3, rasterized=True)
        ax.text(0.96, 0.95, fr"$w^\star={gp['w']:.2f}$", transform=ax.transAxes, fontsize=5.4, va="top",
                ha="right", zorder=6)
        ax.text(0.04, 0.05, f"$N$: {gp['ns']} / {gp['nw']}", transform=ax.transAxes, fontsize=5.0, va="bottom",
                color="#555555", zorder=6)
    title2(fig, GUIDED_X, "d", "So that it", r"matches $\Sigma$")


def guided_label(fig, ax, S, x, y, width, height):
    """The D_w* label next to the guided point: the first nearby spot clear of every ellipse, arrow and point, of
    the formula strip, the right edge and the step title."""
    fig.canvas.draw()
    r, dpi = fig.canvas.get_renderer(), fig.dpi

    def hits(bb, p):
        if isinstance(p, Ellipse):
            V = p.get_path().interpolated(80).transformed(p.get_transform()).vertices
            return bool(((V[:, 0] >= bb.x0) & (V[:, 0] <= bb.x1) & (V[:, 1] >= bb.y0) & (V[:, 1] <= bb.y1)).any())
        return p.get_path().transformed(p.get_transform()).intersects_bbox(bb, filled=False)

    for rad in (7.0, 9.0):
        for t in np.radians([135, 90, 180, 45, 225, 270]):
            lab = ax.annotate(f"${GUIDED}$", S.DW, xytext=(rad * math.cos(t), rad * math.sin(t)),
                              textcoords="offset points", color=ORANGE, fontsize=5.9, ha="center", va="center",
                              zorder=11)
            bb = lab.get_window_extent(r).expanded(1.06, 1.10)
            hit = any(hits(bb, p) for p in ax.patches) or any(hits(bb, ln) for ln in ax.lines) or \
                bb.y0 < (y + 0.10) * dpi or bb.x1 > (x + width - 0.02) * dpi or bb.y1 > (y + height - 0.02) * dpi \
                or (bb.x0 < (x + 0.40) * dpi and bb.y1 > (y + height - 0.12) * dpi)
            if not hit:
                return
            lab.remove()
    v = S.DW - S.D0
    n = np.array([-v[1], v[0]]) / np.linalg.norm(v)
    ax.annotate(f"${GUIDED}$", S.D0 + 0.60 * v, xytext=tuple(4.6 * n * np.sign(n[1])), textcoords="offset points",
                color=ORANGE, fontsize=5.0, ha="center", va="center", zorder=11)


def mechanism(fig, S):
    """(c): step 1, the difference of the circled pair; step 2, the guided prediction. Returns the two axes."""
    x, (y1, y2), w, h = MECH_X, ROW_Y, PANEL_W, PANEL_H
    ax = axat(fig, x, y1, w, h)
    small_panel(ax, S)
    fig.text((x + 0.028) / W, (y1 + h - 0.085) / H, r"$\mathbf{1}$  subtract", fontsize=5.6, va="baseline")
    ax.add_patch(FancyArrowPatch(S.D0, S.D1, arrowstyle="-|>", mutation_scale=4.0, color=INK, lw=0.7, shrinkA=0.9,
                                 shrinkB=1.3, zorder=8))
    mpoint(ax, S.D0, WEAK, 2.25)
    mpoint(ax, S.D1, STRONG, 2.25)
    ctext(fig, x + w * 0.66, y1 + 0.03, r"$\Delta=D_1-D_0$", [BLACK, BLACK, STRONG, STRONG, BLACK, WEAK, WEAK], 4.6)
    ax2 = axat(fig, x, y2, w, h)
    small_panel(ax2, S)
    fig.text((x + 0.028) / W, (y2 + h - 0.085) / H, r"$\mathbf{2}$  scale by $w^\star$", fontsize=5.6,
             va="baseline")
    ax2.add_patch(FancyArrowPatch(S.D0, S.DW, arrowstyle="-|>", mutation_scale=4.6, color=ORANGE, lw=0.85,
                                  shrinkA=1.2, shrinkB=1.5, zorder=8))
    ellipse(ax2, S.Gp.mean(0), np.cov(S.Gp.T, bias=True), 2, color=ORANGE, lw=0.75, zorder=4)
    mpoint(ax2, S.D0, WEAK, 2.6)
    mpoint(ax2, S.D1, STRONG, 2.6)
    mpoint(ax2, S.DW, ORANGE, 2.8)
    guided_label(fig, ax2, S, x, y2, w, h)
    ng = len(MATH.parse(f"${GUIDED}$", 72, FontProperties(size=6.3)).glyphs)
    ctext(fig, x + w / 2, y2 + 0.03, f"${GUIDED}\\!=\\!D_0\\!+\\!w^\\star(D_1\\!-\\!D_0)$",
          [ORANGE] * ng + [BLACK, WEAK, WEAK, BLACK, BLACK, BLACK, BLACK, STRONG, STRONG, BLACK, WEAK, WEAK, BLACK],
          4.6)
    return ax, ax2


def connectors(fig, S, axes, ax1):
    """Dashed arrows from the circled models of (b) to the pair in (c), labelled in the gutter, and from step 2 to
    both panels of (d)."""
    inv = fig.transFigure.inverted()
    for n, P, target, col, name in ((64, S.S0, S.D0, WEAK, r"$D_0$"), (512, S.S1, S.D1, STRONG, r"$D_1$")):
        pa = inv.transform(axes[n].transData.transform(P[RUN])) * [W, H]
        pb = inv.transform(ax1.transData.transform(target)) * [W, H]
        u = (pb - pa) / np.linalg.norm(pb - pa)
        dashed_arrow(fig, pa + u * (3.0 / 72), pb - u * (1.8 / 72))
        xg = 3.828
        slope = (pb[1] - pa[1]) / (pb[0] - pa[0])
        yg = pa[1] + (xg - pa[0]) * slope
        fig.text(xg / W, (yg + 0.032 + max(slope, 0) * 0.04) / H, name, fontsize=6.0, color=col, ha="center",
                 va="bottom", zorder=12)
    ytop, ybot = [y + PANEL_H / 2 for y in ROW_Y]
    x0 = MECH_X + PANEL_W
    xj, ym = x0 + 0.09, ROW_Y[1] + PANEL_H / 2
    dashed_line(fig, (x0, ym), (xj, ym))
    dashed_line(fig, (xj, ybot), (xj, ytop))
    for yy in (ytop, ybot):
        dashed_arrow(fig, (xj, yy), (GUIDED_X - 0.008, yy), z=6)


def draw(S, out):
    fig = plt.figure(figsize=(W, H))
    left(fig, S)
    axes = models(fig, S)
    curves(fig, S)
    guided(fig, S)
    title2(fig, MECH_X, "c", "Guidance scales", "it back up")
    ax1, _ = mechanism(fig, S)
    connectors(fig, S, axes, ax1)
    fig.savefig(out + ".pdf", dpi=450, metadata={"Title": "concept_story", "Author": ""})
    fig.savefig(out + ".png", dpi=300)
    plt.close(fig)
    print(out + ".pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", help="reproduce/toys/story2d.py's folder (default $VMG_RUNS/_toys/story2d)")
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    a = ap.parse_args()
    src = a.src or str(Path(os.environ.get("VMG_RUNS", REPO / "runs")) / "_toys" / "story2d")
    S = load(src)
    for gp in S.gp:
        print(f"pair {gp['ns']}/{gp['nw']}: w* = {gp['w']:.2f}")
    print("spread / exact spread (Adam):", {n: round(float(S.st["adam"][n]["size"] / S.si[n]["size"]), 3) for n in NS})
    draw(S, a.out)


if __name__ == "__main__":
    main()
