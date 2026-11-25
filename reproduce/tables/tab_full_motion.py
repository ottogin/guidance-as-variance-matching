#!/usr/bin/env python
"""tab:full-motion: MDM on KIT-ML, every metric: the real test motions, Pub. (2.5) and our schedule.

    python reproduce/tables/tab_full_motion.py --out paper-tables/ [--allow-missing]

Reads runs/mdm/kitml/metrics/compare-vm-schedule-vs-pub.json (MDM's wo_mm evaluation, 20 replications; the Real
column is each metric's gt_<name>, the evaluator on the real motions) and runs/mdm/kitml-mm/metrics/
compare-vm-schedule-vs-pub.json (MultiModality, mm_short, 5 replications). Diversity is best closest to the real
value, so it is neither coloured nor bolded. Writes full-motion.tex, the tabular body between \\midrule and \\bottomrule.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import compare_json, fmt, pair  # noqa: E402

ROWS = [("R-precision", [("r_precision1", r"Top-1 $\uparrow$", 3), ("r_precision2", r"Top-2 $\uparrow$", 3),
                         ("r_precision3", r"Top-3 $\uparrow$", 3)]),
        (None, [("mdm_fid", r"FID $\downarrow$", 3), ("mm_dist", r"MM-Dist $\downarrow$", 3),
                ("diversity", r"Diversity $\rightarrow$", 2), ("multimodality", r"MMod. $\uparrow$", 2)])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    rec, mm = compare_json("mdm", "kitml"), compare_json("mdm", "kitml-mm")
    missing = [d for d, r in (("mdm/kitml", rec), ("mdm/kitml-mm", mm)) if r is None]
    assert not missing or a.allow_missing, f"no compare-vm-schedule-vs-pub.json for {', '.join(missing)}"
    both = {**(rec or {}), **(mm or {})}
    lines = []
    for g, (head, rows) in enumerate(ROWS):
        if g:
            lines.append(r"\addlinespace")
        if head:
            lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{head}}}}} \\")
        for key, label, d in rows:
            real = both.get(f"gt_{key}")
            cells = [fmt(round(real["a"], d), d) if real else r"\na",
                     *(pair(both[key], key, d) if key in both else (r"\na", r"\na"))]
            lines.append(("\\quad " if head else "") + f"{label} & " + " & ".join(cells) + r" \\")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "full-motion.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'full-motion.tex'}" + (f" (missing: {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
