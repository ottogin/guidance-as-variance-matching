#!/usr/bin/env python
"""tab:noise-control (App. D.4): the variance our schedule adds, added instead as noise that ignores x_t.

    python reproduce/tables/tab_noise_control.py --out paper-tables/

SD 3.5 Medium on COCO, 10,000 images per row: no guidance (arm w1), no guidance plus isotropic or data-shaped
Gaussian noise on the prediction at every step, sized so that it meets the variance target our schedule meets
(sample.py --isonoise / --shapenoise, vmg.guidance.matched_noise_std), and our schedule (vm-schedule). Reads
runs/sd35m/coco/metrics/<arm>.json (reproduce/experiments/evaluate.py) and writes noise-control.tex, the body
between \\midrule and \\bottomrule.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

ROWS = [("w1", r"No guidance ($w=1$)"), ("isonoise", r"\quad + isotropic noise"),
        ("shapenoise", r"\quad + data-shaped noise"), ("vm-schedule", r"Ours, $w^\star_{\mathrm{VM}}(t)$")]
# key, decimals, in the paper's column order
COLS = [("fid", 2), ("fd_dinov2", 1), ("precision_dinov2", 3), ("recall_dinov2", 3), ("precision_inception", 3),
        ("recall_inception", 3), ("clip", 4)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sd35m")
    ap.add_argument("--dataset", default="coco")
    a = ap.parse_args()
    lines = []
    for arm, name in ROWS:
        m = json.load(open(runs.metrics_json(a.model, a.dataset, arm)))
        lines.append(f"{name} & " + " & ".join(f"{float(m[k]):.{d}f}" for k, d in COLS) + r" \\")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "noise-control.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
