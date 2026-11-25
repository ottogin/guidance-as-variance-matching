#!/usr/bin/env python
r"""tab:ncalib (App. E.2): the constant scale fitted from n calibration examples, SD 3.5 Medium and AudioLDM 2.

    python reproduce/tables/tab_ncalib.py --out paper-tables/

Reads $VMG_RUNS/<model>/<dataset>/ncalib/ncalib.json (reproduce/experiments/ncalib.py fit) of each block and writes
ncalib.tex, the table body between \midrule and \bottomrule: per size n, the network evaluations (n times the
levels times 2 predictions), the SD of w and its central 95% range over the random subsets, both corrected for the
overlap of large subsets.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

TABLE = (16, 32, 64, 128, 256, 512, 1024, 2048)     # ncalib.py fit's sizes the table prints
BLOCKS = [("sd35m", "coco", r"SD~3.5 Medium, COCO photos"), ("audioldm2", "audiocaps", r"AudioLDM~2, AudioCaps captions")]


def num(x):
    """An integer as the paper prints it: $1{,}280$."""
    return "$" + f"{x:,}".replace(",", "{,}") + "$"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lines = []
    for i, (model, dataset, head) in enumerate(BLOCKS):
        r = json.load(open(runs.root() / model / dataset / "ncalib" / "ncalib.json"))
        lines += ([r"\midrule"] if i else []) + [r"\multicolumn{4}{@{}l}{\emph{%s}} \\" % head]
        for s in (s for s in r["sizes"] if s["n"] in TABLE):
            lines.append(f"{num(s['n'])} & {num(s['n'] * r['levels'] * 2)} & ${s['sd']:.2f}$ & "
                         f"${s['lo']:.2f}$--${s['hi']:.2f}$ \\\\")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ncalib.tex").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
