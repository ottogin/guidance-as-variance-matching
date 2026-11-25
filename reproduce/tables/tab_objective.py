#!/usr/bin/env python
"""tab:objective (App. D.2): the constant scale under three objectives, next to the published one.

    python reproduce/tables/tab_objective.py --out paper-tables/

With r_t(w) = Var[D_w] - Var[x0] at each calibrated level: Ours makes the level average of r_t vanish (eq. wconstant),
L1 minimises the mean |r_t| and L2 the mean r_t^2, both over w in (0, 40] (vmg.solve, pools mean / l1 / l2). Reads the
three global trace fits from each model's compute_w.py result, runs/<model>/<dataset>/w.json ("fits"; refit an older
w.json with compute_w.py --stage fit). Bold marks the candidate closest to Pub. (none on a tie); the last row is the
median relative distance to Pub. SVD's Pub. is the mean of its per-frame ramp, 2.0. Writes objective.tex, the body
between \\midrule and \\bottomrule.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

# (model, dataset, name as printed)
MODELS = [("sd15", "coco", "SD~1.5"), ("sd35m", "coco", "SD~3.5 Medium"), ("sd35l", "coco", "SD~3.5 Large"),
          ("flux2-klein", "mjhq", "FLUX.2 klein"), ("svd", "ucf101", "SVD"), ("wan", "openvid", "Wan~2.1"),
          ("cogvideox", "openvid", "CogVideoX"), ("cosmos", "bridge", "Cosmos"), ("audioldm2", "audiocaps", "AudioLDM~2"),
          ("sao", "songdescriber", "Stable Audio Open"), ("mdm", "kitml", "MDM")]
POOLS = ("mean", "l1", "l2")
PUB = {"svd": 2.0}                       # the mean of SVD's 1 -> 3 ramp


def published(model):
    if model in PUB:
        return PUB[model]
    from vmg.models import get
    return float(get(model).spec.published_w)


def fits(model, dataset):
    j = json.load(open(runs.w_json(model, dataset)))
    assert "fits" in j, f"{runs.w_json(model, dataset)} predates the pooled fits: refit with compute_w.py --stage fit"
    return [float(j["fits"]["trace"]["global"][p]["w"]) for p in POOLS]


def body(rows):
    """rows: [(name, [ours, l1, l2], pub)] -> the tabular body."""
    out, errs = [], []
    for name, ws, pub in rows:
        d = [abs(w - pub) for w in ws]
        best = int(np.argmin(d)) if sorted(d)[0] < sorted(d)[1] - 1e-12 else None
        cells = [rf"\textbf{{{w:.2f}}}" if i == best else f"${w:.2f}$" for i, w in enumerate(ws)]
        out.append(f"{name} & " + " & ".join(cells) + f" & ${pub:.1f}$ \\\\")
        errs.append([x / pub for x in d])
    med = np.median(np.array(errs), 0)
    i = int(np.argmin(med))
    cells = [rf"\textbf{{{100 * m:.0f}\%}}" if k == i else f"${100 * m:.0f}\\%$" for k, m in enumerate(med)]
    out += [r"\midrule", "Median error & " + " & ".join(cells) + r" & \\"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [(name, fits(m, d), published(m)) for m, d, name in MODELS]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "objective.tex").write_text(body(rows))
    print(body(rows))


if __name__ == "__main__":
    main()
