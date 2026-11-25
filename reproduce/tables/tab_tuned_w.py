#!/usr/bin/env python
"""tab:tuned-w: constants tuned to single metrics on SD 3.5 Medium, against our constant.

    python reproduce/tables/tab_tuned_w.py --out paper-tables/

For each metric m, w*_m is the best of the swept constants (reproduce/rosters/sweeps.py SD35M; ties go to the
smaller w). One row per distinct w*_m lists every metric at that constant: bold on the metric it was tuned to,
and each cell shaded against Ours (vm-global): green (vmbetter) where ours is better, red (vmworse) where the
tuned constant is, faint (xl) below 1%, light (lt) for 1-5%, full beyond 5%. Unshaded rows follow: the published
constant (its sweep arm), ours, and in gray our schedule (vm-schedule).
Reads runs/sd35m/coco/metrics/<arm>.json (evaluate.py) and writes tuned-w.tex (the tabular body) and
tuned-w.json (every w*_m and the off-target win count).
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.rosters.sweeps import SD35M, arm  # noqa: E402
from vmg import runs  # noqa: E402

# key: (+1 if smaller is better, label, format)
COLS = {"fid": (+1, "FID", "{:.2f}"), "fd_dinov2": (+1, r"FD$_\mathrm{DINOv2}$", "{:.1f}"),
        "precision_dinov2": (-1, r"P$_\mathrm{DINOv2}$", "{:.3f}"),
        "precision_inception": (-1, r"P$_\mathrm{Inc}$", "{:.3f}"), "clip": (-1, "CLIP", "{:.4f}")}


def better(a, b, key, printed=False):
    sign, _, f = COLS[key]
    if printed:
        a, b = float(f.format(a)), float(f.format(b))
    return sign * a < sign * b


def cell(key, v, o, target):
    rel = round(100 * abs(v - o) / abs(o), 1)          # percent, rounded as printed (0.995% counts as 1.0%)
    tone = "xl" if rel < 1 else ("lt" if rel < 5 else "")
    col = ("vmworse" if better(v, o, key) else "vmbetter") + tone
    txt = COLS[key][2].format(v)
    txt = rf"\textbf{{{txt}}}" if target else txt
    return rf"\cellcolor{{{col}}}{txt}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="sd35m")
    ap.add_argument("--dataset", default="coco")
    a = ap.parse_args()
    load = lambda name: json.load(open(runs.metrics_json(a.model, a.dataset, name)))   # noqa: E731
    ev = {w: load(arm(w)) for w in SD35M}
    ours, sched = load("vm-global"), load("vm-schedule")
    from vmg.models import get
    pub_w = float(get(a.model).spec.published_w)
    best = {k: min(SD35M, key=lambda w: (COLS[k][0] * ev[w][k], w)) for k in COLS}
    groups = {}
    for k in COLS:
        groups.setdefault(best[k], []).append(k)
    rows = []
    for w in sorted(groups, key=lambda w: list(COLS).index(groups[w][0])):
        cells = [cell(k, ev[w][k], ours[k], k in groups[w]) for k in COLS]
        rows.append(" / ".join(COLS[k][1] for k in groups[w]) + f" & {w:g} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
    if pub_w in ev:                    # the released constant is a point of the sweep (SD 3.5 Medium: 4.5)
        rows.append(f"Pub. & {pub_w:g} & " + " & ".join(COLS[k][2].format(ev[pub_w][k]) for k in COLS) + r" \\")
        rows.append(r"\midrule")
    rows.append(f"Ours & {ours['w_mean']:.2f} & " + " & ".join(COLS[k][2].format(ours[k]) for k in COLS) + r" \\")
    g = r"\color{black!60}"
    rows.append(rf"{g}Ours ($w(\sigma)$) & {g} \ \ \ -- & "
                + " & ".join(g + COLS[k][2].format(sched[k]) for k in COLS) + r" \\")
    wins = sum(better(ours[k], ev[best[m]][k], k, printed=True) for m in COLS for k in COLS if k != m)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tuned-w.tex").write_text("\n".join(rows) + "\n")
    json.dump(dict(best=best, ours_w=ours["w_mean"], off_target_wins=[wins, len(COLS) * (len(COLS) - 1)]),
              open(out / "tuned-w.json", "w"), indent=1)
    print("\n".join(rows))
    print("w*: " + ", ".join(f"{k} {best[k]:g}" for k in COLS) + f"; ours wins {wins} of {len(COLS) * (len(COLS) - 1)}")


if __name__ == "__main__":
    main()
