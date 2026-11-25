#!/usr/bin/env python
"""l_x and l_t of the error field (App. C.3, eq. error-field) at every step t of EDM2's sampler (or against sigma), for
Delta = D1 - D0 and for D1 and D0 themselves.

    python reproduce/figures/fig_errfield_lsigma.py DIR [--tag spread-cfg-c850] [--ttag spreadt-cfg-c850] [--paper]

DIR holds `errfield_edm2.py spread` and `spreadt` outputs, possibly split by level into several files
(TAG*.npz, TTAG*.npz; a level present in more than one file must agree). l_x and l_t are measured as in
fig_errfield_lx.py (fig:errfield-lx): the exp(-1/2) crossing of the binned mean (raw) correlation of each field
against |x - x'| (units: median nearest-image distance), and against |log sigma' - log sigma| at a fixed x.
Levels where the correlation never falls that low within the measured range (l_x at the first five steps) have no
crossing and are left out; --bounds draws them as open triangles (lower bounds at the largest distance reached).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproduce.experiments.errfield_edm2 import crossing  # noqa: E402
from reproduce.figures.fig_errfield_lx import binned, l_t  # noqa: E402


FIELDS = (("delta", "$\\Delta=D_1-D_0$", "black"), ("d1", "$D_1$", "#2a6fc8"), ("d0", "$D_0$", "#d9822b"))


PANEL_COLORS = ("#2a78d6", "#eb6834")   # l_x, l_t when one field is drawn (validated categorical slots 1-2)
N_STEPS = 32   # EDM2's sampler; steps counted from 1, as in fig:errfield-lx
XKEY = {"step": "step", "sigma": "sigma"}
PLAIN = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")   # 100, 10, 1, 0.1 rather than 10^k


def load_levels(D, tag, keys):
    """{level: {key: array at that level}} from every DIR/TAG*.npz; duplicated levels must match."""
    out = {}
    files = sorted(D.glob(f"{tag}*.npz"))
    assert files, f"no {tag}*.npz in {D}"
    for f in files:
        z = np.load(f)
        for j, lev in enumerate(z["levels"]):
            rec = {"sigma": float(z["sigma"][j]), **{k: z[k][j] for k in keys}}
            for k in ("dlogs", "t"):
                if k in z:
                    rec[k] = z[k]
            lev = int(lev)
            if lev in out:
                for k in keys:
                    assert np.allclose(out[lev][k], rec[k], atol=1e-4), (f, lev, k)
            out[lev] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--tag", default="spread-cfg-c850")
    ap.add_argument("--ttag", default="spreadt-cfg-c850")
    ap.add_argument("--bins", type=int, default=60,
                    help="equal-count distance bins; 60 resolves the crossing at step 5 that 30 (fig:errfield-lx) merges away")
    ap.add_argument("--fields", default="delta", help="any of delta, d1, d0")
    ap.add_argument("--lx-scale", choices=("log", "symlog", "linear"), default="symlog")
    ap.add_argument("--x", choices=("step", "sigma"), default="sigma", help="x axis: sampler step t = 1..32, or sigma")
    ap.add_argument("--all-steps", action="store_true",
                    help="keep the steps before the first measured l_x (default: both panels start there)")
    ap.add_argument("--bounds", action="store_true", help="also draw levels with no crossing (open triangles)")
    ap.add_argument("--paper", action="store_true", help="single-column size for the paper (3.3 in wide)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.paper:
        plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
                             "mathtext.fontset": "stix"})
    F = (dict(size=(3.3, 1.6), title=6.5, text=5.5, tick=5.5, label=6.5, lw=1.0, ms=2.2) if a.paper else
         dict(size=(6.6, 3.0), title=10, text=8.5, tick=8, label=9.5, lw=1.6, ms=4.0))
    D = Path(a.dir)
    fields = [fd for fd in FIELDS if fd[0] in a.fields.split(",")]
    keys = [f"{f}_raw" for f, _, _ in fields]
    S = load_levels(D, a.tag, ["r"] + keys)
    T = load_levels(D, a.ttag, keys)
    rows = []
    for lev in sorted(set(S) | set(T)):
        row = {"level": lev, "step": lev + 1}
        msg = []
        for f, _, _ in fields:
            if lev in S:
                s = S[lev]
                xm, ym = binned(s["r"].ravel(), s[f"{f}_raw"].ravel(), a.bins)
                row.update(sigma=s["sigma"], r_max=float(xm[-1]))
                row[f"l_x_{f}"] = crossing(xm, ym)
                rnd = s["t"] > 0.9   # ~independent partners: another image of the class, an independent noise
                row[f"corr_rand_{f}"] = float(s[f"{f}_raw"][rnd].mean())
                row["r_rand"] = float(np.median(s["r"][rnd]))
            if lev in T:
                zt = {"dlogs": T[lev]["dlogs"], f"{f}_raw": T[lev][f"{f}_raw"][None]}
                row["sigma"] = T[lev]["sigma"]
                row[f"l_t_{f}"], row[f"l_t_up_{f}"], row[f"l_t_down_{f}"] = l_t(zt, 0, field=f)
            msg.append(f"{f}: " + " ".join(f"{k} {row[f'{k}_{f}'][0].strip('=')}{row[f'{k}_{f}'][1]:.3g}"
                                           for k in ("l_x", "l_t") if f"{k}_{f}" in row))
        rows.append(row)
        print(f"step {lev + 1:2d}  sigma {row['sigma']:9.4f}  " + " | ".join(msg))

    if not a.all_steps:   # start both panels at the first step with a measured l_x (no crossing before it)
        first = min(row["step"] for row in rows if row.get(f"l_x_{fields[0][0]}", ("",))[0] == "=")
        rows = [row for row in rows if row["step"] >= first]
    fig, axs = plt.subplots(1, 2, figsize=F["size"], sharex=True, constrained_layout=True)
    panels = ((axs[0], "l_x", "$\\ell_x$", "(a) across inputs $x$"),
              (axs[1], "l_t", "$\\ell_t$", "(b) across noise levels $\\log\\sigma$"))
    single = len(fields) == 1
    for (ax, key, lab, title), pc in zip(panels, PANEL_COLORS):
        for f, name, col in fields:
            col = pc if single else col
            pts = [(row[XKEY[a.x]], *row[f"{key}_{f}"]) for row in rows if f"{key}_{f}" in row]
            eq = [(sg, v) for sg, k, v in pts if k == "="]
            if eq:
                ax.plot(*zip(*eq), "-o", color=col, lw=F["lw"], ms=F["ms"], mew=0, label=name,
                        zorder=3)
            for kind, marker in ((">", "^"), ("<", "v")) if a.bounds else ():
                b = [(sg, v) for sg, k, v in pts if k == kind]
                if b:
                    ax.plot(*zip(*b), marker, ls="none", color=col, ms=F["ms"] * 1.2, mfc="white", mew=0.8,
                            zorder=3)
        if a.x == "sigma":
            ax.set_xscale("log")
        if key == "l_x" and a.lx_scale == "log":     # spans three decades; l_t stays within ~0.1-1.1
            ax.set_yscale("log")
            ax.set_ylim(0.1, 400)
            ax.yaxis.set_major_formatter(PLAIN)
        elif key == "l_x" and a.lx_scale == "symlog":   # linear on [0, 1], log above: starts at 0, keeps the floor
            ax.set_yscale("symlog", linthresh=1, linscale=1)
            ax.set_ylim(0, 400)
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.5, 1, 10, 100]))
            ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(
                [0.25, 0.75] + [k * 10 ** e for e in (0, 1, 2) for k in range(2, 10)]))
            ax.yaxis.set_major_formatter(PLAIN)
        elif key == "l_x":
            ax.set_ylim(0, 250)
        else:
            ax.set_ylim(0, 1.0)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
        if a.x == "sigma":
            ax.xaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="major", color="#e4e4e1", lw=0.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#52514e")
            ax.spines[sp].set_linewidth(0.6)
        ax.tick_params(which="both", labelsize=F["tick"], labelcolor="#0b0b0b", length=0, pad=2)   # no tick marks
        ax.set_ylabel(lab, fontsize=F["label"])
        ax.set_xlabel("noise level $\\sigma$  (sampling $\\rightarrow$)" if a.x == "sigma" else
                      f"sampling step $t$  (of {N_STEPS})", fontsize=F["label"])
        ax.set_title(title, fontsize=F["title"], loc="left", pad=3)
    if single and a.bounds:   # one series per panel: the panel names it; the key explains the open markers
        h = [matplotlib.lines.Line2D([], [], color="#52514e", marker="o", ms=F["ms"], lw=F["lw"], mew=0,
                                     label="measured"),
             matplotlib.lines.Line2D([], [], color="#52514e", marker="^", ms=F["ms"] * 1.2, mfc="white", mew=0.8,
                                     ls="none", label="lower bound")]
        axs[0].legend(handles=h, fontsize=F["text"], frameon=False, handlelength=1.6, loc="upper right",
                      borderaxespad=0.2)
    elif not single:
        axs[0].legend(fontsize=F["text"], frameon=False, handlelength=1.4)
    if a.x == "sigma":
        axs[0].set_xlim(rows[0]["sigma"] * 1.35, rows[-1]["sigma"] / 1.35)   # inverted: sampling runs left to right
    else:
        axs[0].set_xlim(0, N_STEPS + 1)
        axs[0].xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 8, 16, 24, 32]))
        axs[0].xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    out = a.out or str(D / "errfield-lsigma.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(str(Path(out).with_suffix(".png")), dpi=200, bbox_inches="tight")
    meta = json.loads(sorted(D.glob(f"{a.tag}*-meta.json"))[0].read_text())
    Path(out).with_suffix(".json").write_text(json.dumps(dict(unit=meta["unit"], d1=meta["d1"], d0=meta["d0"],
                                                              bins=a.bins, rows=rows), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
