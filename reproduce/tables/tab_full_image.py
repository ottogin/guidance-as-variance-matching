#!/usr/bin/env python
"""tab:full-image: every image metric, Pub. against our schedule, for the four text-to-image models.

    python reproduce/tables/tab_full_image.py --out paper-tables/ [--allow-missing]

Reads runs/<model>/<dataset>/metrics/compare-vm-schedule-vs-pub.json (reproduce/experiments/compare.py) and
writes full-image.tex, the tabular body between \\midrule and \\bottomrule.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.tables._tex import full_table  # noqa: E402

MODELS = [("sd15", "coco"), ("sd35m", "coco"), ("sd35l", "coco"), ("flux2-klein", "mjhq")]
ROWS = [("Inception", [("fid", r"FID $\downarrow$", 2), ("precision_inception", r"P $\uparrow$", 3),
                       ("recall_inception", r"R $\uparrow$", 3), ("density_inception", r"Dens. $\uparrow$", 3),
                       ("coverage_inception", r"Cov. $\uparrow$", 3)]),
        ("DINOv2", [("fd_dinov2", r"FD $\downarrow$", 1), ("precision_dinov2", r"P $\uparrow$", 3),
                    ("recall_dinov2", r"R $\uparrow$", 3), ("density_dinov2", r"Dens. $\uparrow$", 3),
                    ("coverage_dinov2", r"Cov. $\uparrow$", 3)]),
        (None, [("clip", r"CLIP $\uparrow$", 4)])]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    full_table(MODELS, ROWS, a.out, "full-image.tex", a.allow_missing)


if __name__ == "__main__":
    main()
