#!/usr/bin/env python
"""tab:pub-sched-all: the published schedules against ours on nine models.

    python reproduce/tables/tab_pub_sched_all.py --out paper-tables/ [--allow-missing]

Nine columns per row: the constant, Interval, TV-CFG and Linear (vmg/schedules.py), each at the released weight
(Pub. w) and at our global weight (Our w), then our schedule. Each model's first row is the weight averaged over
sampling steps. Each Our-w cell is shaded against the Pub.-w cell to its left on the printed values, bold marks
the best printed value in each row, and MDM's diversity (best closest to the real data) is neither. Reads
runs/<model>/<dataset>[-<tag>]/metrics/<arm>.json (evaluate.py) for the arms of reproduce/plan.py's sched_jobs;
AudioLDM 2 runs on its first 1,024 test captions (tag sub1024). Writes pub-sched-all.tex, the tabular body.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import LOWER_BETTER, UNRANKED, fmt, shade  # noqa: E402
from vmg import runs  # noqa: E402

ARMS = ["pub", "vm-global", "pub-interval", "gw-interval", "pub-tvcfg", "gw-tvcfg", "pub-linear", "gw-linear",
        "vm-schedule"]
IMAGE = [("fid", r"FID $\downarrow$", 2), ("precision_inception", r"Precision (Inception) $\uparrow$", 3),
         ("recall_inception", r"Recall (Inception) $\uparrow$", 3),
         ("fd_dinov2", r"FD$_{\mathrm{DINOv2}}$ $\downarrow$", 1),
         ("precision_dinov2", r"Precision (DINOv2) $\uparrow$", 3), ("recall_dinov2", r"Recall (DINOv2) $\uparrow$", 3),
         ("clip", r"CLIP score $\uparrow$", 4)]
VIDEO = [("fvd", r"FVD $\downarrow$", 1), ("precision_i3d", r"Precision $\uparrow$", 3),
         ("recall_i3d", r"Recall $\uparrow$", 3), ("density_i3d", r"Density $\uparrow$", 3),
         ("coverage_i3d", r"Coverage $\uparrow$", 3), ("clip_t", r"CLIP-T $\uparrow$", 2)]
# (group, [(model, dataset, tag, shown name, rows)])
BLOCKS = [
    ("Text-to-image", [("sd15", "coco", None, "SD 1.5", IMAGE), ("sd35m", "coco", None, "SD 3.5 Medium", IMAGE),
                       ("sd35l", "coco", None, "SD 3.5 Large", IMAGE),
                       ("flux2-klein", "mjhq", None, "FLUX.2 [klein] 4B", IMAGE)]),
    ("Video", [("wan", "openvid", None, "Wan2.1 1.3B", VIDEO), ("cogvideox", "openvid", None, "CogVideoX-2B", VIDEO)]),
    ("Text-to-audio", [
        ("audioldm2", "audiocaps", "sub1024", "AudioLDM 2",
         [("fad", r"FAD (VGGish) $\downarrow$", 2), ("kl", r"KL (PaSST) $\downarrow$", 3),
          ("is_passt", r"Inception Score (PaSST) $\uparrow$", 2), ("clap", r"CLAP (AudioLDM 2) $\uparrow$", 3),
          ("clap_laion", r"CLAP (LAION) $\uparrow$", 3), ("fd_passt", r"FD (PaSST) $\downarrow$", 1),
          ("precision_passt", r"Precision (PaSST) $\uparrow$", 3), ("recall_passt", r"Recall (PaSST) $\uparrow$", 3)]),
        ("sao", "songdescriber", None, "Stable Audio Open",
         [("fd_openl3", r"FD (OpenL3) $\downarrow$", 2), ("kl", r"KL (PaSST) $\downarrow$", 3),
          ("clap", r"CLAP $\uparrow$", 4)])]),
    ("Text-to-motion", [("mdm", "kitml", None, "MDM",
                         [("mdm_fid", r"FID $\downarrow$", 3), ("r_precision3", r"R-precision top-3 $\uparrow$", 3),
                          ("mm_dist", r"MM-Dist $\downarrow$", 3), ("diversity", r"Diversity $\rightarrow$", 2)])])]
SUBSET = {"audioldm2": 1024}             # tag sub<N>: the first N evaluation rows


def arm_metrics(model, dataset, tag, arm):
    path = runs.metrics_json(model, dataset, arm, tag)
    return json.load(open(path)) if path.exists() else None


def block(model, dataset, tag, name, rows, allow_missing):
    m = {x: arm_metrics(model, dataset, tag, x) for x in ARMS}
    missing = [x for x, v in m.items() if v is None]
    assert not missing or allow_missing, f"{model}/{dataset}: no evaluated arm {missing} (plan.py sched_jobs)"
    grey = lambda x: (rf"{{\color{{black!60}}{m[x]['w_mean']:.2f}}}" if m[x] and "w_mean" in m[x]  # noqa: E731
                      else r"\na")
    lines = [rf"\enspace\textbf{{{name}}} \data{{mean $w$}} & " + " & ".join(grey(x) for x in ARMS) + r" \\"]
    for key, label, d in rows:
        v = {x: round(m[x][key], d) for x in ARMS if m[x] and key in m[x]}
        best = None if key in UNRANKED or not v else (min if key in LOWER_BETTER else max)(v.values())
        cells = []
        for i, x in enumerate(ARMS):
            if x not in v:
                cells.append(r"\na")
                continue
            t = rf"\textbf{{{fmt(v[x], d)}}}" if v[x] == best else fmt(v[x], d)
            left = ARMS[i - 1]
            if i % 2 == 1 and i < len(ARMS) - 1 and key not in UNRANKED and left in v:   # an Our-w column
                t = rf"\cellcolor{{{shade(key, v[left], v[x])}}}" + t
            cells.append(t)
        lines.append(rf"\qquad {label} & " + " & ".join(cells) + r" \\")
    return lines, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    lines, missing = [], []
    for g, (group, models) in enumerate(BLOCKS):
        if g:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textit{{{group}}}}} \\")
        for j, spec in enumerate(models):
            if j:
                lines.append(r"\addlinespace[3pt]")
            body, miss = block(*spec, a.allow_missing)
            lines += body
            missing += [f"{spec[0]}:{x}" for x in miss]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pub-sched-all.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'pub-sched-all.tex'}" + (f" (missing {len(missing)} arms)" if missing else ""))


if __name__ == "__main__":
    main()
