#!/usr/bin/env python
"""tab:baselines-sd35m (not in the paper): other methods' baselines on SD 3.5 Medium beside ours.

    python reproduce/tables/tab_baselines_sd35m.py --out paper-tables/

Every arm is sampled with the same code and the same float32 latent state as the paper's SD 3.5 runs (sample.py
--fp32-state, tag "baselines"), on the 10,000 COCO evaluation rows at 40 Euler steps:

  references  Pub. 4.5, ours constant (4.13, printed 4.14) and our schedule w(sigma), re-sampled
  SIMCA       simulation-free (= w_MSE), simulation-based in the CFG form, and with SIMCA's two free weights
  Step AG     guidance on the first 50% / 30% of the steps at Pub.; ours pruned at 50%; our constant refit for it
  CFG-OEC     orthogonal error correction (tau 0.9) on Pub. and on our schedule

A star marks a paired 95% interval (compare.py) of the k-NN metrics that excludes zero, against the row's
reference: Pub. for the published-weight rows and the SIMCA rows, our schedule for the rows built on it, and our
constant for the refit. NFE is what the rule needs over the 40 steps. Missing arms print as "-", so partial runs
tabulate. Also writes the CPU analysis of reproduce/baselines/analysis.py (figures, analysis-sd35m.json) into
<out>/baselines-sd35m/ when its inputs exist (without them the var. excess column is empty). Writes
baselines-sd35m.md (with the paper's printed reference rows) and
baselines-sd35m.tex (the tabular body).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import compare_json  # noqa: E402
from vmg import runs  # noqa: E402

TAG = "baselines"
PUB, VMG, VMS = "pub-f32", "vm-global-f32", "vm-schedule-f32"
# (group, label, arm, reference arm for the stars)
ROWS = [("Reference", "Pub. 4.5", PUB, None),
        ("Reference", "Ours, constant", VMG, PUB),
        ("Reference", "Ours, schedule w(σ)", VMS, PUB),
        ("SIMCA", "simulation-free (= w_MSE)", "simca-sf-f32", PUB),
        ("SIMCA", "simulation-based, CFG form", "simca-sb1-f32", PUB),
        ("SIMCA", "simulation-based, free α, β", "simca-sb2-f32", PUB),
        ("Step AG", "p = 0.5, Pub.", "stepag50-pub-f32", PUB),
        ("Step AG", "p = 0.3, Pub.", "stepag30-pub-f32", PUB),
        ("Step AG", "p = 0.5, our w(σ)", "stepag50-vms-f32", VMS),
        ("Step AG", "p = 0.5, our constant refit", "stepag50-vmr-f32", VMG),
        ("CFG-OEC", "τ = 0.9, Pub.", "oec90-pub-f32", PUB),
        ("CFG-OEC", "τ = 0.9, our w(σ)", "oec90-vms-f32", VMS)]
COLS = [("fid", "FID", 2), ("fd_dinov2", "FD_DINOv2", 1), ("precision_dinov2", "P_DINO", 3),
        ("recall_dinov2", "R_DINO", 3), ("precision_inception", "P_Inc", 3), ("recall_inception", "R_Inc", 3),
        ("clip", "CLIP", 4)]
# the paper's printed SD 3.5 M rows (tab:pub-sched, fp32 state, the paper's own seeds)
PRINTED = [("Pub. 4.5 (printed)", 4.50, [20.66, 242.5, 0.741, 0.550, 0.691, 0.458, 0.2716]),
           ("Ours, constant (printed)", 4.14, [20.15, 240.0, 0.748, 0.570, 0.684, 0.454, 0.2713]),
           ("Ours, schedule (printed)", 7.82, [20.01, 226.0, 0.759, 0.579, 0.676, 0.444, 0.2741])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sd35m")
    ap.add_argument("--dataset", default="coco")
    ap.add_argument("--no-analysis", action="store_true", help="skip the CPU analysis and its figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    excess = {}
    if not a.no_analysis:
        from reproduce.baselines import analysis
        try:
            res = analysis.run(analysis.default_run(), out / "baselines-sd35m")
            excess = {k: v["mean"] for k, v in res["excess"].items()}
        except (AssertionError, FileNotFoundError) as e:      # neither its calibration pass nor the stored products
            print(f"skipping the CPU analysis and the var. excess column: {e}")
    md = ["| Group | Rule | mean w | NFE | " + " | ".join(c for _, c, _ in COLS) + " | var. excess |",
          "|---|---|---|---|" + "---|" * len(COLS) + "---|"]
    tex = []
    group = None
    for g, label, arm, ref in ROWS:
        p = runs.root() / a.model / f"{a.dataset}-{TAG}" / "metrics" / f"{arm}.json"
        m = json.load(open(p)) if p.exists() else None
        cmp = compare_json(a.model, a.dataset, ref, arm, TAG) if (m and ref) else None
        cells = []
        for k, _, d in COLS:
            if m is None or k not in m:
                cells.append("-")
                continue
            star = "*" if cmp and k in cmp and cmp[k].get("excludes_zero") else ""
            cells.append(f"{m[k]:.{d}f}{star}")
        ex = excess.get(arm.replace("-f32", ""))      # analysis.py's arm names
        ex = None if ex is None else (0.0 if abs(ex) < 5e-4 else ex)          # no "-0.000"
        wm = f"{m['w_mean']:.2f}" if m and "w_mean" in m else "-"
        nfe = str(m["nfe"]) if m and "nfe" in m else "-"
        md.append(f"| {g if g != group else ''} | {label} | {wm} | {nfe} | " + " | ".join(cells) +
                  f" | {'' if ex is None else format(ex, '+.3f')} |")
        if g != group and group is not None:
            tex.append(r"\addlinespace[1pt]")
        tex.append(f"{g if g != group else ''} & {label} & {wm} & {nfe} & " +
                   " & ".join(c.replace("*", r"\sig") if c != "-" else r"\na" for c in cells) + r" \\")
        group = g
    md += ["", "Printed in the paper (tab:pub-sched; same protocol, the old runs' own initial noise):", "",
           "| Rule | mean w | " + " | ".join(c for _, c, _ in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for label, w, v in PRINTED:
        md.append(f"| {label} | {w:.2f} | " + " | ".join(f"{x:.{d}f}" for x, (_, _, d) in zip(v, COLS)) + " |")
    md += ["", "var. excess: tr Cov(D_w) / tr Cov(x0) - 1 averaged over the 40 levels (the default rule, stored "
           "calibration moments); 0 is matched. Defined for rules of the CFG form only."]
    (out / "baselines-sd35m.md").write_text("\n".join(md) + "\n")
    (out / "baselines-sd35m.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
