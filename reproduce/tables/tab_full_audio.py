#!/usr/bin/env python
"""tab:full-audio: every audio metric, Pub. against our schedule, for AudioLDM 2 and Stable Audio Open.

    python reproduce/tables/tab_full_audio.py --out paper-tables/ [--allow-missing]

Reads runs/<model>/<dataset>/metrics/compare-vm-schedule-vs-pub.json (reproduce/experiments/compare.py) and
writes full-audio.tex, the tabular body between \\midrule and \\bottomrule: one block per model, its metrics
grouped as in the paper (vmg/metrics/audio.py names them).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import compare_json, pair  # noqa: E402

# (model, dataset, block head, [(group head or None, [(key, label, decimals)])])
BLOCKS = [
    ("audioldm2", "audiocaps", r"\textbf{AudioLDM 2} \data{$w=3.5$}", [
        ("Own suite", [("fad", r"FAD (VGGish) $\downarrow$", 2), ("kl", r"KL (softmax) $\downarrow$", 2),
                       ("is_passt", r"IS $\uparrow$", 2), ("clap", r"CLAP $\uparrow$", 3)]),
        ("Independent", [("kl_sigmoid", r"KL (sigmoid) $\downarrow$", 2), ("clap_laion", r"CLAP (LAION) $\uparrow$", 3),
                         ("fd_passt", r"FD (PaSST) $\downarrow$", 1), ("precision_passt", r"P $\uparrow$", 3),
                         ("recall_passt", r"R $\uparrow$", 3), ("density_passt", r"Dens. $\uparrow$", 3),
                         ("coverage_passt", r"Cov. $\uparrow$", 3)])]),
    ("sao", "songdescriber", r"\textbf{Stable Audio Open} \data{$w=7$}", [
        (None, [("fd_openl3", r"FD (OpenL3) $\downarrow$", 2), ("kl", r"KL $\downarrow$", 3),
                ("clap", r"CLAP $\uparrow$", 3)])])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    lines, missing = [], []
    for b, (model, ds, head, groups) in enumerate(BLOCKS):
        rec = compare_json(model, ds)
        if rec is None:
            missing.append(f"{model}/{ds}")
        if b:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{3}}{{@{{}}l}}{{{head}}} \\")
        for g, (sub, rows) in enumerate(groups):
            if g:
                lines.append(r"\addlinespace")
            if sub:
                lines.append(rf"\multicolumn{{3}}{{@{{}}l}}{{\textit{{{sub}}}}} \\")
            for key, label, d in rows:
                cells = pair(rec[key], key, d) if rec is not None and key in rec else (r"\na", r"\na")
                lines.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
    assert not missing or a.allow_missing, f"no compare-vm-schedule-vs-pub.json for {', '.join(missing)}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "full-audio.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'full-audio.tex'}" + (f" (missing: {', '.join(missing)})" if missing else ""))


if __name__ == "__main__":
    main()
