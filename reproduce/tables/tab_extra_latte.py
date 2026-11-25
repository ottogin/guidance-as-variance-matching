#!/usr/bin/env python
"""tab:extra-latte: Latte-XL/2 on UCF-101, Pub. (7.0) against our schedule and the best constant of a sweep.

    python reproduce/tables/tab_extra_latte.py --out paper-tables/ [--allow-missing]

Pub. and Ours (and Ours' paired intervals) come from runs/latte/ucf101/metrics/compare-vm-schedule-vs-pub.json.
The constant column is the sweep arm w in {1, 3, 5, 7, 9} (7 is Pub.) with the lowest FVD, read from that arm's
evaluate.py record; the choice is printed and written as a comment at the top of the fragment, since the paper's
header names it. Cells are shaded against Pub. on the printed values, bold marks the best of the three columns.
Writes extra-latte.tex, the tabular body between \\midrule and \\bottomrule.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import LOWER_BETTER, compare_json, fmt, shade  # noqa: E402
from vmg import runs  # noqa: E402

MODEL, DATASET, PUB_W = "latte", "ucf101", 7.0
SWEEP = (1, 3, 5, 9)                     # with Pub. (7), the constants the best one is chosen from
ROWS = [("I3D features", [("fvd", r"FVD $\downarrow$", 1), ("precision_i3d", r"Precision $\uparrow$", 3),
                          ("recall_i3d", r"Recall $\uparrow$", 3), ("density_i3d", r"Density $\uparrow$", 3),
                          ("coverage_i3d", r"Coverage $\uparrow$", 3)]),
        (None, [("is_c3d", r"Inception Score (C3D) $\uparrow$", 1)])]


def arm_metrics(arm):
    path = runs.metrics_json(MODEL, DATASET, arm)
    return json.load(open(path)) if path.exists() else None


def best_constant(pub_fvd):
    """(w, metrics) of the lowest-FVD constant among the sweep and Pub."""
    arms = {w: arm_metrics(f"w{w:g}") for w in SWEEP}
    missing = [w for w, m in arms.items() if m is None]
    assert not missing, f"no evaluated sweep arm for w = {missing} (reproduce/plan.py: tab:extra-latte)"
    arms[PUB_W] = {"fvd": pub_fvd}
    w = min(arms, key=lambda k: arms[k]["fvd"])
    return w, arms[w]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    rec = compare_json(MODEL, DATASET)
    if rec is None:
        assert a.allow_missing, f"no compare-vm-schedule-vs-pub.json for {MODEL}/{DATASET}"
        print("extra-latte: no runs yet, nothing written")
        return
    w, const = best_constant(rec["fvd"]["a"])
    if w == PUB_W:                                       # the constant column then repeats Pub.
        const = {k: r["a"] for k, r in rec.items()}
    lines = [f"% Const. ({w:g}): the lowest-FVD constant of w in {{1, 3, 5, 7, 9}}"]
    for head, group in ROWS:
        if head:
            lines.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{head}}}}} \\")
        for key, label, d in group:
            pub, ours, c = round(rec[key]["a"], d), round(rec[key]["b"], d), round(const[key], d)
            best = (min if key in LOWER_BETTER else max)(pub, ours, c)
            bf = lambda v: rf"\textbf{{{fmt(v, d)}}}" if v == best else fmt(v, d)     # noqa: E731
            star = r"\sig" if rec[key].get("excludes_zero") else ""
            lines.append(("\\quad " if head else "") + f"{label} & {bf(pub)} & "
                         rf"\cellcolor{{{shade(key, pub, ours)}}}{bf(ours)}{star} & "
                         rf"\cellcolor{{{shade(key, pub, c)}}}{bf(c)} \\")
        lines.append(r"\addlinespace")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "extra-latte.tex").write_text("\n".join(lines[:-1]) + "\n")
    print(f"wrote {out / 'extra-latte.tex'} (best constant on FVD: w = {w:g})")


if __name__ == "__main__":
    main()
