#!/usr/bin/env python
"""App. C "Guidance reduces diversity", fig:diversity-lx: fidelity (exact log-likelihood) and diversity (coverage) of
M(2) with a correlated error field on 1,000 ImageNet-64 Gaussians, against l_x, at l_t = 0 and 0.1
(K = 128 steps, 100,000 real vs 100,000 generated samples, 3 fields).

    python reproduce/toys/corr_diversity_in64.py work --model gauss --sweep big        # on GPUs; writes the units
    python reproduce/figures/fig_diversity_lx_in64.py --collect --out diversity-lx-in64.pdf   # units -> points.json, figure
    python reproduce/figures/fig_diversity_lx_in64.py --out diversity-lx-in64.pdf             # from points.json

--units and --data default to $VMG_RUNS/_toys/corr_diversity_in64/gauss/big/{units, points.json}.

Panels: one per l_t; both curves relative to real samples. Log-likelihood: the Gaussians are ~1.4e5 nats apart, so log p(x) = -|x - nearest centre|^2 / 2 s^2 + const exactly,
and the gap to real samples is -(spread - spread_real) / 2 nats per dimension, spread = E|x - centre|^2 / (s^2 d);
plotted as the per-dimension likelihood ratio exp(gap).
Coverage (k = 3) is in the fixed random 64-D projection, divided by its value on real samples.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import sys
from pathlib import Path as _P

sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from reproduce.style import COND, INK, MUTED, UNCOND, W_COL, rc, style  # noqa: E402
from vmg.runs import root  # noqa: E402

BIG = root() / "_toys" / "corr_diversity_in64" / "gauss" / "big"     # corr_diversity_in64.py work --sweep big
ap = argparse.ArgumentParser()
ap.add_argument("--units", default=str(BIG / "units"))
ap.add_argument("--data", default=str(BIG / "points.json"))
ap.add_argument("--out", required=True)
ap.add_argument("--collect", action="store_true", help="rebuild --data from the unit JSONs")
ap.add_argument("--lts", default="0,0.1", help="one panel per l_t")
a = ap.parse_args()
SEEDS, N, LTS, KEYS = (0, 1, 2), 128, tuple(a.lts.split(",")), ("spread", "coverage_64", "precision_64", "recall_64")

if a.collect:
    runs = {}
    for f in glob.glob(f"{a.units}/*.json"):
        m = re.fullmatch(r"(lx[\d.]+_lt[\d.]+|data|lx0)_N(\d+)_r(\d+)", os.path.basename(f)[:-5])
        if m and int(m[3]) in SEEDS and int(m[2]) in (0, N):
            runs.setdefault(m[1], {})[int(m[3])] = json.load(open(f))
    pts = {"seeds": SEEDS, "N": N, "n": runs["data"][0]["n"], "lx": {}, "ref": {}}
    for arm in ("data", "lx0"):
        pts["ref"][arm] = {k: [runs[arm][r][k] for r in SEEDS] for k in KEYS}
    for lt in LTS:
        lxs = sorted(float(x[2:].split("_lt")[0]) for x in runs if x.endswith(f"_lt{lt}") and len(runs[x]) == 3)
        pts["lx"][lt] = {f"{lx:g}": {k: [runs[f"lx{lx:g}_lt{lt}"][r][k] for r in SEEDS] for k in KEYS} for lx in lxs}
    os.makedirs(os.path.dirname(a.data), exist_ok=True)
    json.dump(pts, open(a.data, "w"), indent=1)
    print("wrote", a.data, {lt: list(v) for lt, v in pts["lx"].items()})

pts = json.load(open(a.data))
ref = {k: np.mean(pts["ref"]["data"][k]) for k in KEYS}
CURVES = (("Likelihood", COND, lambda v: np.exp(-(v["spread"] - ref["spread"]) / 2)),
          ("Coverage", UNCOND, lambda v: v["coverage_64"] / ref["coverage_64"]))

LOWS = []
plt = rc()
fig, axes = plt.subplots(1, len(LTS), figsize=(W_COL, 1.6), sharey=True, constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.03)
for ax, lt in zip(axes, LTS):
    lxs = sorted(pts["lx"][lt], key=float)
    x = np.array([float(v) for v in lxs])
    ax.axhline(1, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ends = {}
    for lab, c, f in CURVES:
        v = np.array([f({k: np.array(pts["lx"][lt][lx][k]) for k in KEYS}) for lx in lxs])
        mu, sd = v.mean(1), v.std(1, ddof=1)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.16, lw=0)
        LOWS.append(float((mu - sd).min()))
        ax.plot(x, mu, color=c, lw=1.2, marker="o", ms=2.0, mec="white", mew=0.35, zorder=3)
        ends[lab] = (mu[-1], c)
    for lab, (y, c) in ends.items():                       # direct labels; keep them off the dashed line at 1
        dy = 5 if lab == "Likelihood" and y > 0.995 else (-5 if lab == "Likelihood" else 0)
        ax.annotate(lab, (x[-1], y), xytext=(3, dy), textcoords="offset points", va="center", fontsize=6 if len(LTS) <= 2 else 5.5, color=c)
    ax.set_xscale("log")
    xt = [0.03, 0.1, 0.3, 1, 3, 10] if len(LTS) <= 2 else [0.03, 0.3, 3]
    ax.set_xticks(xt, [f"{t:g}" for t in xt])
    ax.minorticks_off()
    ax.set_xlim(0.025, 60 if len(LTS) <= 2 else 120)
    if len(LTS) <= 2:
        ax.set_xlabel(r"$\ell_x$ / nearest-image distance")
    ax.set_title(r"$\ell_t=%s$" % lt, pad=2, color=INK)
    style(ax)
if len(LTS) > 2:                                              # narrow panels: one shared axis label
    fig.supxlabel(r"$\ell_x$ / nearest-image distance", fontsize=plt.rcParams["axes.labelsize"])
axes[0].set_ylabel("Relative to real samples")
axes[0].set_ylim(np.floor(min(LOWS) * 20) / 20, 1.04)
fig.savefig(a.out)
print("wrote", a.out)
