#!/usr/bin/env python
"""fig:ellipse (App. 2-D example): with exact denoisers, the variance-matched weight puts the most guided
predictions in the data's high-density region; D_0 and D_1 put fewer.

    python reproduce/figures/fig_ellipse.py --out OUT/ellipse-share        # CPU, ~20 s

Data: K = 20 isotropic Gaussians (std 0.15) with means evenly spaced in angle on an ellipse with semi-axes 4 and
1 (the toy model of vmg/models/toy.py). D_1 = E[x0 | x_t, c] and D_0 = E[x0 | x_t] in closed form. Noising is
x_t = (1 - t) x0 + t eps, written as x0 + sigma eps with sigma = t / (1 - t), on the SD 3 16-step Euler grid
(shift 3). The matched schedule is the paper's rule (vmg.solve) at each of the sampler's levels, from 100,000
forward-noised calibration points. Panels: steps 10, 15 and 16. The input is the sampler's own state, a 16-step
Euler run guided by the matched schedule. For each w, the curve is the share of D_w = D_0 + w (D_1 - D_0) whose
log-density under the true mixture exceeds the q-quantile of real data's log-density; real data puts 1 - q there.
Writes <out>.{pdf,png} (the share row), <out>-points.{pdf,png} (the clouds) and <out>-merged.{pdf,png}, the
paper's figure: the clouds (a)-(e) over the share curves (f)-(h), one column wide.
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from reproduce import style as F  # noqa: E402
from vmg import solve  # noqa: E402

STEPS = (10, 15, 16)


class Mix:
    def __init__(self, K=20, ax=4.0, ay=1.0, s=0.15):
        th = 2 * np.pi * np.arange(K) / K
        self.mu = np.stack([ax * np.cos(th), ay * np.sin(th)], 1)
        self.K, self.s2 = K, s * s

    def denoise(self, x, c, sig):
        """exact D_1(x, c) and D_0(x) at VE level sig (inf: the component mean and the overall mean)."""
        if np.isinf(sig):
            return self.mu[c], np.broadcast_to(self.mu.mean(0), (len(c), 2))
        v = self.s2 + sig * sig
        d = x[:, None, :] - self.mu[None]
        m = self.mu[None] + (self.s2 / v) * d
        lg = -0.5 * (d ** 2).sum(-1) / v
        g = np.exp(lg - lg.max(1, keepdims=True))
        g /= g.sum(1, keepdims=True)
        return m[np.arange(len(x)), c], (g[..., None] * m).sum(1)

    def logp(self, x):
        q = -0.5 * ((x[:, None, :] - self.mu[None]) ** 2).sum(-1) / self.s2
        mx = q.max(1)
        return mx + np.log(np.exp(q - mx[:, None]).sum(1)) - np.log(self.K * 2 * np.pi * self.s2)


def flow_grid(N, shift=3.0):
    u = np.linspace(1, 1 / N, N)
    t = shift * u / (1 + (shift - 1) * u)
    with np.errstate(divide="ignore"):
        return np.where(t >= 1, np.inf, t / (1 - t)), t


def tvar(z):
    return z.var(0).mean()


def tcov(a, b):
    return ((a - a.mean(0)) * (b - b.mean(0))).sum(1).mean() / 2


def schedule(mix, sigs, x0, c, eps):
    """The paper's per-level weight (eq. wstar) from forward-noised calibration points, via vmg.solve."""
    xc = x0 - x0.mean(0)
    dot = lambda u, v: (u * v).sum(1).mean() / 2                      # E<u, v> per coordinate
    cdot = lambda u, v: dot(u - u.mean(0), v - v.mean(0))
    xr, rr, rrc = [], [], []
    for s in sigs:
        d1, d0 = mix.denoise(x0 + (0 if np.isinf(s) else s) * eps, c, s)
        r0, r1 = d0 - x0, d1 - x0
        xr.append([dot(xc, r0), dot(xc, r1)])
        rr.append([dot(r0, r0), dot(r0, r1), dot(r1, r1)])
        rrc.append([cdot(r0, r0), cdot(r0, r1), cdot(r1, r1)])
    m = solve.Moments(sigma=sigs, t=sigs, var_x0=tvar(x0), xr=xr, rr=rr, rr_centred=rrc, n=len(x0), d=2)
    return solve.fit_schedule(m)["w"]


def own_states(mix, sigs, ws, n, seed, keep):
    """16-step Euler (DDIM) run guided by the schedule ws; the state entering each step in keep."""
    rng = np.random.default_rng(seed)
    c = np.arange(n) % mix.K
    z = rng.standard_normal((n, 2))
    x = z.copy()
    lv = list(sigs) + [0.0]
    out = {}
    for j in range(len(sigs)):
        s, sn = lv[j], lv[j + 1]
        if j + 1 in keep:
            out[j + 1] = x.copy()
        d1, d0 = mix.denoise(x, c, s)
        dw = d0 + ws[j] * (d1 - d0)
        x = dw + sn * z if np.isinf(s) else (sn / s) * x + (1 - sn / s) * dw
    return out, c


def shares(mix, d0, d1, wgrid, thr):
    return np.array([(mix.logp(d0 + w * (d1 - d0)) > thr).mean() for w in wgrid])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension")
    ap.add_argument("--q", type=float, default=0.9, help="probability line: real data's log-density quantile")
    ap.add_argument("--ncal", type=int, default=100_000)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--wmax", type=float, default=4.0)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--s", type=float, default=0.15)
    ap.add_argument("--ax", type=float, default=4.0, help="ellipse semi-axis along x")
    ap.add_argument("--ay", type=float, default=1.0, help="ellipse semi-axis along y")
    ap.add_argument("--pstep", type=int, default=15, help="step shown in the point-cloud figure")
    ap.add_argument("--layout", choices=("col", "wide"), default="col")
    args = ap.parse_args()
    col = args.layout == "col"

    mix, N = Mix(args.K, args.ax, args.ay, args.s), 16
    sigs, ts = flow_grid(N)
    rng = np.random.default_rng(0)
    cc = np.arange(args.ncal) % mix.K
    x0 = mix.mu[cc] + np.sqrt(mix.s2) * rng.standard_normal((args.ncal, 2))
    eps = rng.standard_normal((args.ncal, 2))
    ws = schedule(mix, sigs, x0, cc, eps)
    thr = np.quantile(mix.logp(x0), args.q)
    vb = tvar(mix.mu)
    print(f"K {mix.K}, Var x0 {tvar(x0):.3f}, between share {100 * vb / tvar(x0):.1f}%")
    print("matched schedule:", " ".join(f"{w:.2f}" for w in ws))

    st, c = own_states(mix, sigs, ws, args.n, 1, set(STEPS) | {args.pstep})
    wgrid = np.round(np.arange(0, args.wmax + 1e-9, 0.005), 3)
    panels = []
    for i in STEPS:
        s = sigs[i - 1]
        d1, d0 = mix.denoise(st[i], c, s)
        H = shares(mix, d0, d1, wgrid, thr)
        hv = shares(mix, d0, d1, [ws[i - 1]], thr)[0]
        j = H.argmax()
        near = wgrid[H >= H[j] - 0.02]
        print(f"step {i} (t = {ts[i - 1]:.3f}, sigma = {s:.3f}), {100 * args.q:g}% line: matched w {ws[i - 1]:.3f} -> "
              f"{100 * hv:.1f}%; best w {wgrid[j]:.3f} -> {100 * H[j]:.1f}% (within 2 points: {near[0]:.3f}-{near[-1]:.3f}); "
              f"D_1 {100 * H[wgrid == 1.0][0]:.1f}%, D_0 {100 * H[0]:.1f}%")
        panels.append((i, ts[i - 1], H, hv, ws[i - 1]))

    import matplotlib
    plt = F.rc()

    bb = None if col else "tight"         # col: the canvas is laid out in inches, exactly 3.3 in wide
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pts = (mix, x0[:args.n], st[args.pstep], c, sigs[args.pstep - 1], ws[args.pstep - 1], thr)

    if col:
        fig = plt.figure(figsize=(W_COL, SHARE_H))
        share_row(fig, None, panels, wgrid, args, "abc", Hf=SHARE_H)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(5.5, 1.85), gridspec_kw=dict(wspace=0.32))
        share_row(fig, axs, panels, wgrid, args, "abc")
    fig.savefig(args.out + ".pdf", bbox_inches=bb, pad_inches=0.02)
    fig.savefig(args.out + ".png", dpi=260, bbox_inches=bb, pad_inches=0.02)
    print("wrote", args.out + ".{pdf,png}")

    if col:
        fig = plt.figure(figsize=(W_COL, points_h()))
    else:
        fig = plt.figure(figsize=(5.5, 2.45))
    points_block(fig, *pts, args, Hf=points_h() if col else None)
    fig.savefig(args.out + "-points.pdf", bbox_inches=bb, pad_inches=0.02, dpi=400)
    fig.savefig(args.out + "-points.png", dpi=260, bbox_inches=bb, pad_inches=0.02)
    print("wrote", args.out + "-points.{pdf,png}")

    if col:     # both in one column figure: the clouds (a)-(e) on top, the share curves (f)-(h) below
        Hf = SHARE_H + MERGE_GAP + points_h()
        fig = plt.figure(figsize=(W_COL, Hf))
        points_block(fig, *pts, args, Hf=Hf, y0=SHARE_H + MERGE_GAP, quiet=True)
        share_row(fig, None, panels, wgrid, args, "fgh", Hf=Hf)
        fig.savefig(args.out + "-merged.pdf", pad_inches=0.02, dpi=400)
        fig.savefig(args.out + "-merged.png", dpi=260, pad_inches=0.02)
        print("wrote", args.out + "-merged.{pdf,png}")


W_COL = F.W_COL
SHARE_H = 1.5                                    # col: height of the share row (in)
MERGE_GAP = 0.04                                 # col: space between the clouds and the share row in -merged


def points_h():
    """col: height of the points block (in) -- the (b)-(e) row, the gap, the (a) strip."""
    return 0.215 + 0.765 + 0.29 + 0.66 + 0.02


def share_row(fig, axs, panels, wgrid, args, letters, Hf=None, y0=0.0):
    """The share-vs-w curves at the three steps. col layout (axs None): placed in inches on a canvas Hf tall, with the
    row's bottom edge y0 in above the canvas bottom; one shared x and y label and the legend below."""
    from matplotlib.lines import Line2D
    col = axs is None
    if col:
        W, l, r, g, b, h = W_COL, 0.36, 0.03, 0.07, 0.5, 0.76
        pw = (W - l - r - 2 * g) / 3
        axs = [fig.add_axes([(l + k * (pw + g)) / W, (y0 + b) / Hf, pw / W, h / Hf]) for k in range(3)]
    ylab = f"% of $D_w$ denser than {100 * args.q:g}% of data"
    for k, (ax, (i, t, H, hv, wv)) in enumerate(zip(axs, panels)):
        ms, sm = (3.2, 6.5) if col else (4, 8)
        ax.axhline(100 * (1 - args.q), color=F.INK, lw=0.8, ls=(0, (1, 1.5)), zorder=1)
        ax.plot(wgrid, 100 * H, color=F.INK, lw=1.0 if col else 1.2, zorder=2)
        ax.axvline(wv, color=F.ACC, lw=0.8 if col else 0.9, ls=(0, (3, 1.6)), zorder=3)
        ax.plot([0], [100 * H[0]], "o", ms=ms, color=F.UNCOND, mec="white", mew=0.6, zorder=4, clip_on=False)
        ax.plot([1], [100 * H[wgrid == 1.0][0]], "o", ms=ms, color=F.COND, mec="white", mew=0.6, zorder=4, clip_on=False)
        ax.plot([wv], [100 * hv], "*", ms=sm, color=F.ACC, mec="white", mew=0.5, zorder=5, clip_on=False)
        ax.set_xlim(0, args.wmax)
        ax.set_ylim(0, 104)
        ax.set_xticks(range(int(args.wmax) + 1))
        if not col:
            ax.set_xlabel("guidance weight $w$")
            if k == 0:
                ax.set_ylabel(ylab)
        F.style(ax)
        if col:     # one x label and one y label for the row; y tick labels on the first panel only
            ax.set_yticks([0, 50, 100])
            ax.tick_params(pad=1.2, labelsize=5.8, length=2)
            if k:
                ax.tick_params(labelleft=False)
            ax.text(0, 1.04, f"({letters[k]}) step {i} of 16,\n$t={t:.2f}$", transform=ax.transAxes, fontsize=6.2,
                    color=F.INK, va="bottom", linespacing=1.05)
        else:
            F.title(ax, f"({letters[k]}) step {i} of 16, $t={t:.2f}$")
    hs = [Line2D([], [], marker="*", ls="none", ms=7, color=F.ACC, mec="white", mew=0.5),
          Line2D([], [], marker="o", ls="none", ms=4, color=F.COND, mec="white", mew=0.6),
          Line2D([], [], marker="o", ls="none", ms=4, color=F.UNCOND, mec="white", mew=0.6),
          Line2D([], [], color=F.INK, lw=0.8, ls=(0, (1, 1.5)))]
    labs = [r"variance-matched $w^\star_{\mathrm{VM}}(t)$", "$D_1$ ($w=1$)", "$D_0$ ($w=0$)",
            f"real data ({100 * (1 - args.q):g}% by definition)"]
    if col:
        fig.text(0.04 / W, (y0 + b + h / 2) / Hf, f"% of $D_w$ denser\nthan {100 * args.q:g}% of data", rotation=90,
                 ha="left", va="center", fontsize=6.2, color=F.INK, linespacing=1.05, multialignment="center")
        fig.text((l + (W - l - r) / 2) / W, (y0 + b - 0.18) / Hf, "guidance weight $w$", ha="center", va="center",
                 fontsize=6.2, color=F.INK)
        hs[0] = Line2D([], [], marker="*", ls="none", ms=6, color=F.ACC, mec="white", mew=0.5)
        hs[1].set_markersize(3.2); hs[2].set_markersize(3.2)
        fig.legend(hs, labs, frameon=False, loc="lower center", bbox_to_anchor=((l + (W - l - r) / 2) / W, y0 / Hf),
                   ncol=2, fontsize=5.8, handlelength=1.5, columnspacing=1.2, labelspacing=0.25, borderaxespad=0.1,
                   handletextpad=0.4)
    else:
        fig.legend(hs, labs, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=6.5,
                   handlelength=1.6, columnspacing=1.2)


def points_block(fig, mix, x0, x, c, sig, wv, thr, args, Hf=None, y0=0.0, quiet=False):
    """overview of the data + four zoomed clouds: x0, D_0, D_1, D_w at the matched w (all at one step). col layout
    (Hf given): placed in inches on a canvas Hf tall, the block's bottom edge y0 in above the canvas bottom."""
    from matplotlib.patches import Rectangle
    d1, d0 = mix.denoise(x, c, sig)
    clouds = [("data $x_0$", x0, F.INK), ("$D_0$ ($w=0$)", d0, F.UNCOND), ("$D_1$ ($w=1$)", d1, F.COND),
              (rf"$D_w$, matched $w={wv:.2f}$", d0 + wv * (d1 - d0), F.ACC)]
    xr = mix.mu[:, 0].max()
    box = (xr - 1.35, xr + 0.4, -0.875, 0.875)                 # the right end, where curvature is highest

    col = Hf is not None

    def region(ax, lim, n=360):
        gx, gy = np.meshgrid(np.linspace(lim[0], lim[1], n), np.linspace(lim[2], lim[3], n))
        lp = mix.logp(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(gx.shape)
        ax.contourf(gx, gy, lp, levels=[thr, lp.max() + 1], colors=[F.PHOTO], alpha=0.3, zorder=0)
        ax.contour(gx, gy, lp, levels=[thr], colors=[F.INK], linewidths=0.45 if col else 0.55, zorder=4)

    full = (-xr - 0.5, xr + 0.5, -mix.mu[:, 1].max() - 0.45, mix.mu[:, 1].max() + 0.45)
    if col:     # (a) strip top right with its title beside it, (b)-(e) in a row; heights must match points_h()
        W, sq, g, yb = W_COL, 0.765, 0.05, y0 + 0.215
        xl = (W - 4 * sq - 3 * g) / 2
        hs = 0.66
        ws = hs * (full[1] - full[0]) / (full[3] - full[2])
        ys = yb + sq + 0.29
        top = fig.add_axes([(W - xl - ws) / W, ys / Hf, ws / W, hs / Hf])
        cells = [[(xl + k * (sq + g)) / W, yb / Hf, sq / W, sq / Hf] for k in range(4)]
    else:
        gs = fig.add_gridspec(2, 4, height_ratios=[0.58, 1], hspace=0.3, wspace=0.12)
        cells = [gs[1, k] for k in range(4)]
        top = fig.add_subplot(gs[0, :])
    region(top, full, 500)
    top.scatter(x0[:, 0], x0[:, 1], s=0.07 if col else 0.15, color=F.INK, alpha=0.35, lw=0, rasterized=True, zorder=2)
    top.add_patch(Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=F.ACC, lw=0.8, zorder=3))
    top.set_xlim(full[:2]); top.set_ylim(full[2:]); top.set_aspect("equal")
    top.set_xticks([]); top.set_yticks([])
    for sp in top.spines.values():
        sp.set_color(F.MUTED); sp.set_linewidth(0.6)
    if col:
        fig.text(xl / W, (ys + hs) / Hf, f"(a) data, {mix.K} Gaussians;\nbox: the zoomed end", ha="left", va="top",
                 fontsize=6.2, color=F.INK, linespacing=1.1)
    else:
        F.title(top, f"(a) data, {mix.K} Gaussians; box: the zoomed end")
    for k, (lab, P, pc) in enumerate(clouds):
        ax = fig.add_axes(cells[k]) if col else fig.add_subplot(cells[k])
        region(ax, box)
        ax.scatter(P[:, 0], P[:, 1], s=0.25 if col else 0.6, color=pc, alpha=0.5, lw=0, rasterized=True, zorder=2)
        ax.plot(mix.mu[:, 0], mix.mu[:, 1], "+", color=F.INK, ms=2.6 if col else 3.5, mew=0.5 if col else 0.6, zorder=3)
        ax.set_xlim(box[:2]); ax.set_ylim(box[2:]); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(F.MUTED); sp.set_linewidth(0.6)
        share = (mix.logp(P) > thr).mean()
        if col:     # wrapped to fit 0.77 in panels
            ax.text(0.5, -0.04, f"{100 * share:.1f}% inside\nthe outline", transform=ax.transAxes, ha="center",
                    va="top", fontsize=5.8, color=F.INK, linespacing=1.05)
            ax.text(0, 1.03, f"({'bcde'[k]}) {lab}".replace(", matched", ", matched\n"), transform=ax.transAxes,
                    fontsize=6.2, color=F.INK, va="bottom", linespacing=1.05)
        else:
            ax.text(0.5, -0.05, f"{100 * share:.1f}% inside the outline", transform=ax.transAxes, ha="center",
                    va="top", fontsize=6.5, color=F.INK)
            F.title(ax, f"({'bcde'[k]}) {lab}")
        if not quiet:
            print(f"points step {args.pstep}: {lab}: {100 * share:.1f}% denser than {100 * args.q:g}% of data")


if __name__ == "__main__":
    main()
