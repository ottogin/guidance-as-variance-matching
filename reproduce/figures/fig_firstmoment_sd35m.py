#!/usr/bin/env python
"""SD 3.5 Medium on COCO: the guidance weight from first-moment matching, from ours, and the published one.

    python reproduce/figures/fig_firstmoment_sd35m.py --out results/firstmoment-sd35m

First-moment matching asks for E[D_w | t] = xbar0 over forward-noised calibration photos, with
D_w = D0 + w Delta and Delta = D1 - D0. Two readings of it are drawn:

* the closed form, w(t) = (xbar0 - E[D0 | t]) / E[Delta | t], applied to the image mean (the
  average over all latent coordinates, i.e. over the 16 equal-sized channels). The band is a 95% bootstrap
  over photos. Points beyond the axis are drawn as triangles at the edge with their value.
* its least-squares version over all d coordinates, w = <xbar0 - E D0, E Delta> / |E Delta|^2, from the inner
  products of the mean residuals (raw minus centred residual products in the test fixture).

Ours is the schedule of compute_w.py (its default rule, tr Cov); its last level has no admissible root
and holds step 39's weight (hollow marker).

Data: tests/fixtures/sd35m-coco-s40.npz, and results/meanshift/ (per-photo
channel means of x0, D1 - x0 and D0 - x0 from the raw residual dump). Both are 5,705 COCO photos, 40 levels,
seed 0. Writes <out>.{pdf,png}.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reproduce.style import ACC, COND, GRID, INK, MUTED, UNCOND, W_COL, rc, style  # noqa: E402
from vmg import solve  # noqa: E402

PUBLISHED = 4.5        # SD 3.5 Medium's released guidance scale (vmg.models.sd.SD35Medium)
YLIM = (-40.0, 30.0)
LABEL_LEFT = {24}     # off-scale value labels drawn left of their triangle


def least_squares_w(rr, rr_centred):
    """Least-squares w with E[D_w] = xbar0 at each level, from raw and centred residual products (00, 01, 11)."""
    m00, m01, m11 = (np.asarray(rr) - np.asarray(rr_centred)).T     # <E r_i, E r_j> / d
    return (m00 - m01) / (m11 - 2 * m01 + m00)


def closed_form_w(chan_dir, reps=2000, seed=0):
    """(w [S], lo [S], hi [S], w over all levels at once): -E[r0] / E[r1 - r0] on the image mean, with a
    95% photo bootstrap. r1 = D1 - x0, r0 = D0 - x0, per photo averaged over channels."""
    Z = [np.load(p) for p in sorted(glob.glob(str(Path(chan_dir) / "part*.npz")))]
    assert Z, f"no part*.npz in {chan_dir}"
    o = np.argsort(np.concatenate([z["levels"] for z in Z]))
    r1 = np.concatenate([z["r1c"] for z in Z])[o].astype(np.float64)     # [S, n, C]
    r0 = np.concatenate([z["r2c"] for z in Z])[o].astype(np.float64)
    U, D = r0.mean(2), (r1 - r0).mean(2)                                 # [S, n]
    n = U.shape[1]
    rng = np.random.default_rng(seed)
    boot = np.empty((reps, U.shape[0]))
    for b in range(reps):
        i = rng.integers(0, n, n)
        boot[b] = -U[:, i].mean(1) / D[:, i].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5], 0)
    return -U.mean(1) / D.mean(1), lo, hi, -U.mean(1).sum() / D.mean(1).sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixture", default=str(REPO / "tests" / "fixtures" / "sd35m-coco-s40.npz"))
    ap.add_argument("--chan", default=str(REPO / "results" / "meanshift" / "chan"))
    ap.add_argument("--out", default=str(REPO / "results" / "firstmoment-sd35m"))
    args = ap.parse_args()

    z = np.load(args.fixture)
    m = solve.Moments(sigma=z["sigma"], t=z["t"], var_x0=float(z["var_x0"]), xr=z["xr"], rr=z["rr"],
                      rr_centred=z["rr_centred"], n=int(z["n"]), d=int(z["d"]))
    ours = solve.fit_schedule(m)
    w_ls = least_squares_w(m.rr, m.rr_centred)
    w_rv, lo, hi, w_rv_all = closed_form_w(args.chan)
    steps = np.arange(1, len(m.t) + 1)
    for k in range(len(steps)):
        print(f"step {steps[k]:2d}  t {m.t[k]:.4f}  closed form {w_rv[k]:8.2f} [{lo[k]:8.2f}, {hi[k]:8.2f}]  "
              f"least squares {w_ls[k]:6.2f}  ours {ours['w'][k]:6.2f}")
    print(f"closed form averaged over all steps first: w = {w_rv_all:.3f}")

    plt = rc(7.5, 6.5)
    fig, ax = plt.subplots(figsize=(W_COL, 2.4))
    style(ax, grid=False)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.text(40.6, 1.0, "$w=1$", color=MUTED, fontsize=6, va="center", ha="left")
    ax.axhline(PUBLISHED, color=MUTED, lw=1.0, ls=(0, (4, 2)), zorder=2, label=f"published ($w={PUBLISHED:g}$)")

    ok = ours["admissible"]
    ax.plot(steps, ours["w"], color=COND, lw=1.4, zorder=4, label="second moment (ours)")
    ax.plot(steps[ok], ours["w"][ok], "o", color=COND, ms=2.4, mec="white", mew=0.4, zorder=5)
    ax.plot(steps[~ok], ours["w"][~ok], "o", color="white", ms=2.6, mec=COND, mew=0.8, zorder=5)

    ax.fill_between(steps, np.clip(lo, *YLIM), np.clip(hi, *YLIM), color=ACC, alpha=0.18, lw=0, zorder=2)
    ax.plot(steps, w_rv, color=ACC, lw=1.4, marker="s", ms=2.2, mec="white", mew=0.4, zorder=3,
            label="first moment (closed form)")
    for s, v in zip(steps, w_rv):                     # off-scale points: a triangle at the edge and the value
        if YLIM[0] < v < YLIM[1]:
            continue
        top = v >= YLIM[1]
        y = YLIM[1] if top else YLIM[0]
        ax.plot(s, y, "^" if top else "v", color=ACC, ms=3.6, mec="white", mew=0.4, zorder=6, clip_on=False)
        left = s in LABEL_LEFT                        # where the next steps' points would collide
        ax.annotate(f"{v:.0f}".replace("-", "\u2212"), (s, y), xytext=(-5 if left else 4, -3 if top else 3),
                    textcoords="offset points", fontsize=5.8, color=INK, ha="right" if left else "left",
                    va="top" if top else "bottom")

    ax.plot(steps, w_ls, color=UNCOND, lw=1.0, marker="D", ms=1.8, mec="white", mew=0.3, zorder=3,
            label="first moment (least squares)")
    ax.set_xlim(0, 42)
    ax.set_xticks([1, 10, 20, 30, 40])
    ax.set_ylim(*YLIM)
    ax.set_yticks([-40, -30, -20, -10, 0, 10, 20, 30])
    ax.set_xlabel("sampler step (40-step Euler, noisy $\\to$ clean)")
    ax.set_ylabel("guidance weight $w$")
    h, lab = ax.get_legend_handles_labels()
    order = [1, 2, 3, 0]                                 # ours, closed form, least squares, published
    ax.legend([h[i] for i in order], [lab[i] for i in order], loc="upper left", frameon=False,
              handlelength=2.2, borderaxespad=0.2, labelcolor=INK, fontsize=5.8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        print(f"wrote {args.out}.{ext}")


if __name__ == "__main__":
    main()
