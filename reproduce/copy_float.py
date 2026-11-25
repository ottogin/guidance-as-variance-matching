#!/usr/bin/env python
"""Copy a figure an experiment script drew into its own folder to the paper's file name (run_all.py's make step).

    python reproduce/copy_float.py SRC DST      # SRC.pdf -> DST.pdf, and SRC.png -> DST.png when it exists

SRC and DST are paths without the extension (one with an extension is used as given).
"""
import shutil
import sys
from pathlib import Path


def main():
    src, dst = (Path(p) for p in sys.argv[1:3])
    pairs = [(src, dst)] if src.suffix else [(src.with_suffix(e), dst.with_suffix(e)) for e in (".pdf", ".png")]
    assert pairs[0][0].exists(), f"{pairs[0][0]} missing"
    for s, d in pairs:
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s, d)
            print(f"{s} -> {d}")


if __name__ == "__main__":
    main()
