#!/usr/bin/env python
"""tab:pub-sched: published schedules on SD 3.5 Medium at the released weight and at our constant.

    python reproduce/tables/tab_pub_sched.py --out paper-tables/

Rows come in four groups (constant, interval, TV-CFG, linear: vmg/schedules.py), each at the released weight
(Pub.) and at our global weight (Ours), then our schedule. Mean w is over sampling steps. Our rows are shaded
(tours) and every Ours cell is coloured by its relative change against its Pub. cell, on the printed values:
vmgood where better, vmbad where worse, bold beyond 1%, black only on a tie. \\best (paper preamble) underlines the
best printed value of each column over all rows. The columns follow tab:pub-all's image block: Inception (FID,
precision, recall), DINOv2 (FD, precision, recall), CLIP. Reads runs/sd35m/coco/metrics/<arm>.json (arms pub,
vm-global, pub-<kind>, gw-<kind>, vm-schedule) and writes pub-sched.tex, everything between the table's \\label
and its \\vspace.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

GROUPS = [("Constant", "pub", "vm-global"), ("Interval", "pub-interval", "gw-interval"),
          ("TV-CFG", "pub-tvcfg", "gw-tvcfg"), ("Linear", "pub-linear", "gw-linear")]
SCHEDULE = "vm-schedule"
U, D = r"$\uparrow$", r"$\downarrow$"
# key, +1 if larger is better (-1 smaller), decimals, header
COLS = [("fid", -1, 2, f"FID{D}"), ("precision_inception", 1, 3, f"P{U}"), ("recall_inception", 1, 3, f"R{U}"),
        ("fd_dinov2", -1, 1, f"FD{D}"), ("precision_dinov2", 1, 3, f"P{U}"), ("recall_dinov2", 1, 3, f"R{U}"),
        ("clip", 1, 4, f"CLIP{U}")]
BOLD = 0.01          # relative change beyond which an Ours cell is bold
HEAD = [r"\scriptsize\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{0.9}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{ll" + "r" * (len(COLS) + 1) + "}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Inception} & \multicolumn{3}{c}{DINOv2} & \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"\multicolumn{2}{l}{Schedule} & \hd{mean $w$} & " + " & ".join(rf"\hd{{{h}}}" for *_, h in COLS) + r" \\",
        r"\midrule"]


def render(m):
    """The table body. m: {arm: its metrics JSON}."""
    arms = [x for _, p, o in GROUPS for x in (p, o)] + [SCHEDULE]
    r = {x: {k: round(m[x][k], d) for k, _, d, _ in COLS} for x in arms}
    best = {k: (max if s > 0 else min)(r[x][k] for x in arms) for k, s, _, _ in COLS}

    def cells(x, pub=None):
        out = []
        for k, s, d, _ in COLS:
            t = f"{m[x][k]:.{d}f}"
            if pub and r[x][k] != r[pub][k]:
                rel = (r[x][k] - r[pub][k]) / abs(r[pub][k])
                col = "vmgood" if rel * s > 0 else "vmbad"
                t = rf"{{\bfseries\color{{{col}}}{t}}}" if abs(rel) >= BOLD else rf"{{\color{{{col}}}{t}}}"
            if r[x][k] == best[k]:
                t = rf"\best{{{t}}}"
            out.append(t)
        return " & ".join(out)

    rows = list(HEAD)
    for lab, p, o in GROUPS:
        rows.append(f"{lab} & Pub. & {m[p]['w_mean']:.2f} & {cells(p)}" + r" \\")
        rows.append(rf"\rowcolor{{tours}} & Ours & {m[o]['w_mean']:.2f} & {cells(o, p)}" + r" \\")
    rows += [r"\midrule",
             rf"\rowcolor{{tours}} Our schedule & $w(\sigma)$ & {m[SCHEDULE]['w_mean']:.2f} & {cells(SCHEDULE)}" + r" \\",
             r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    return "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sd35m")
    ap.add_argument("--dataset", default="coco")
    a = ap.parse_args()
    arms = [x for _, p, o in GROUPS for x in (p, o)] + [SCHEDULE]
    body = render({x: json.load(open(runs.metrics_json(a.model, a.dataset, x))) for x in arms})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pub-sched.tex").write_text(body)
    print(body, end="")


if __name__ == "__main__":
    main()
