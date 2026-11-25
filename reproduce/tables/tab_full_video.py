#!/usr/bin/env python
"""tab:full-video: every video and world-model metric, Pub. against our schedule.

    python reproduce/tables/tab_full_video.py --out paper-tables/ [--allow-missing]

Reads runs/<model>/<dataset>/metrics/compare-vm-schedule-vs-pub.json (reproduce/experiments/compare.py) for SVD,
Wan2.1, CogVideoX and Cosmos-Predict2.5, and writes full-video.tex, the tabular body between \\midrule and
\\bottomrule. Latte-XL/2 has its own table (tab_extra_latte.py).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import full_table  # noqa: E402

MODELS = [("svd", "ucf101"), ("wan", "openvid"), ("cogvideox", "openvid"), ("cosmos", "bridge")]
ROWS = [("I3D", [("fvd", r"FVD $\downarrow$", 1), ("precision_i3d", r"P $\uparrow$", 3),
                 ("recall_i3d", r"R $\uparrow$", 3), ("density_i3d", r"Dens. $\uparrow$", 3),
                 ("coverage_i3d", r"Cov. $\uparrow$", 3)]),
        (None, [("clip_t", r"CLIP-T $\uparrow$", 2)]),
        ("Frame accuracy", [("psnr", r"PSNR $\uparrow$", 2), ("ssim", r"SSIM $\uparrow$", 4)])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    full_table(MODELS, ROWS, a.out, "full-video.tex", a.allow_missing)


if __name__ == "__main__":
    main()
