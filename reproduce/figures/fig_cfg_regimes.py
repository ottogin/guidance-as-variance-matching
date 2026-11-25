#!/usr/bin/env python
"""fig:cfg-regimes: where CFG helps, its difference is model error rather than label information (EDM2-XS, ImageNet-64).

    python reproduce/figures/fig_cfg_regimes.py --out paper-figures/cfg-regimes-img64 [--summary PATH]

Reads $VMG_RUNS/edm2-64-xs-2e27/cfg-regimes/summary.json (reproduce/experiments/cfg_regimes_in64.py analyze). Per
image, at the sampler levels with sigma >= 0.09, the mean over the two label-dropout seeds of |Delta|^2, Delta =
D(x_t, c) - D(x_t), and of the conditional error |D(x_t, c) - D_XL(x_t, c)|^2. Each sigma window is shaded by the
FID change of guidance with w = 2 inside it only, relative to no guidance, averaged over the two seeds: green
lowers FID, red raises it. Writes <out>.pdf and <out>.png.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce import style  # noqa: E402

PURPLE, ORANGE, GRAY, GREEN, RED = "#4a3aa7", style.ACC, "#8a8a84", "#1baf7a", "#c0392b"
XLO, XHI = 0.09, 90                     # the plotted sigma range; the outer windows extend to its ends


def load(path):
    S = json.loads(Path(path).read_text())
    lv = {k: np.array(v) for k, v in S["levels"].items()}
    keep = lv["sigma"] >= XLO
    runs = sorted(S["fid"])
    assert len(runs) == 2, f"window FIDs of {runs}, expected both seeds"
    wins = sorted(S["windows"].items(), key=lambda kv: kv[1][0])            # low sigma first
    change = np.array([[100 * (S["fid"][r][w] / S["fid"][r]["none"] - 1) for w, _ in wins] for r in runs])
    return dict(sigma=lv["sigma"][keep], delta2=lv["delta2"][keep].mean(1), err=lv["err"][keep].mean(1),
                edges=[lo for _, (lo, _) in wins] + [wins[-1][1][1]], windows=[e for _, e in wins],
                change=change.mean(0), w=S["w"])


def figure(d):
    plt = style.rc(size=8.5, small=8.5)
    plt.rcParams.update({"axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
    fig, a = plt.subplots(figsize=(3.35, 1.85))
    plt.subplots_adjust(0.17, 0.225, 0.985, 0.88)
    for k, ((lo, hi), v) in enumerate(zip(d["windows"], d["change"])):
        lo_, hi_ = XLO if k == 0 else lo, XHI if k == len(d["windows"]) - 1 else hi
        good = v < 0
        a.axvspan(lo_, hi_, color=GREEN if good else RED, alpha=0.11 if good else 0.09, lw=0)
        a.text(np.sqrt(lo_ * hi_), 1.0, f"{v:+.0f}%", ha="center", va="top", fontsize=9.5,
               color=GREEN if good else RED, transform=a.get_xaxis_transform(), fontweight="bold")
    a.plot(d["sigma"], d["delta2"], "-o", color=PURPLE, ms=2.3, lw=1.1,
           label="$\\|\\Delta\\|^2$, $\\Delta=D(x_t,c)-D(x_t)$")
    a.plot(d["sigma"], d["err"], "-s", color=ORANGE, ms=2.3, lw=1.1,
           label="model error $\\|D(x_t,c)-D_{\\mathrm{XL}}(x_t,c)\\|^2$")
    a.set_xscale("log")
    a.set_xlim(XHI, XLO)
    a.set_yscale("log")
    a.set_ylim(0.05, 5000)
    inner = d["edges"][1:-1]
    ticks = inner + [80]
    a.set_xticks(ticks)
    a.set_xticklabels([f"{t:g}" for t in ticks], fontsize=7.5)
    a.set_xticks([], minor=True)
    for e in inner:
        a.axvline(e, color=GRAY, lw=0.5, ls=":")
    a.set_xlabel("noise level $\\sigma$ (sampling runs left to right)", fontsize=8)
    a.set_ylabel("per image (sum over pixels)", fontsize=7.5)
    a.legend(frameon=False, fontsize=6.2, loc="lower left", handlelength=1.5, borderaxespad=0.2)
    a.set_title(f"Shading: FID change with guidance ($w={d['w']:g}$) in that window only", loc="left", fontsize=7.2)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output path without extension (.pdf and .png are written)")
    ap.add_argument("--summary", help="default: $VMG_RUNS/edm2-64-xs-2e27/cfg-regimes/summary.json")
    a = ap.parse_args()
    from vmg import runs
    d = load(a.summary or runs.root() / "edm2-64-xs-2e27" / "cfg-regimes" / "summary.json")
    print("FID change per window (low sigma first), mean of the two seeds, %:", np.round(d["change"], 1))
    fig = figure(d)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=220))):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", pad_inches=0.01, **kw)
    print("wrote", f"{a.out}.pdf")


if __name__ == "__main__":
    main()
