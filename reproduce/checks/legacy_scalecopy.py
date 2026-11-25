#!/usr/bin/env python
"""Check fig:scalecopy's port against the stored run behind the paper (results/ of the earlier pipeline).

    python reproduce/checks/legacy_scalecopy.py export --out /tmp/legacy-scalecopy.npz     # gate 1 input
    python reproduce/figures/fig_scalecopy.py --src /tmp/legacy-scalecopy.npz --out /tmp/scalecopy
    python reproduce/checks/legacy_scalecopy.py compare                                     # gate 2

export: the stored reductions (results/bigboot16/mean81k/mean81k.npz and ellipses/ellipses-axes.npz) rewritten as
the scalecopy.npz that reproduce/toys/bigboot16.py writes, so the figure script can be compared with the paper's.
compare: reproduce/toys/bigboot16.py's reduce stage on the stored per-shard moments (results/gaussboot/main, the
first 1,000 nets per family, then results/big16, the other 80,000), against those stored reductions.
Only this check reads the old results/ tree, which is not part of this repository.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RES = REPO / "results"
MEAN = RES / "bigboot16" / "mean81k" / "mean81k.npz"
AXES = RES / "bigboot16" / "ellipses" / "ellipses-axes.npz"
SHARDS = [RES / "gaussboot" / "main", RES / "big16"]
LOSS = [f"{k}_{s}" for k in ("L_loc", "L_sig", "L_uni") for s in ("mean", "se")]


def export(out):
    Z, E = np.load(MEAN), np.load(AXES)
    assert np.allclose(Z["sigmas"], E["sigmas"])
    np.savez(out, sigmas=Z["sigmas"], good=Z["good"], a_uni=Z["weak:strong|a_uni"],
             **{k: Z[f"weak:strong|{k}"] for k in LOSS}, W2=E["W2"], S2=E["S2"], a2=E["a2"])
    print(f"wrote {out}")


def compare():
    from reproduce.toys import bigboot16 as B
    with tempfile.TemporaryDirectory() as tmp:
        R = B.reduce(SHARDS, Path(tmp) / "scalecopy.npz")
    Z, E = np.load(MEAN), np.load(AXES)
    worst = 0.0
    rows = [(k, R[k], Z[f"weak:strong|{k}"]) for k in LOSS + ["a_uni", "a_sig"]] + \
           [(k, R[k], E[k]) for k in ("W2", "S2", "a2")]
    for k, new, old in rows:
        new, old = np.asarray(new, float), np.asarray(old, float)
        rel = float(np.max(np.abs(new - old)) / np.max(np.abs(old)))
        worst = max(worst, rel)
        print(f"  {k:12s} max |new - stored| / max |stored| = {rel:.1e}")
    assert worst < 1e-10, f"the reduce stage differs from the stored reductions by {worst:.1e}"
    print(f"GATE 2 PASS: reductions reproduced to {worst:.1e} (relative)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("export", "compare"))
    ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "legacy-scalecopy.npz"))
    a = ap.parse_args()
    export(a.out) if a.cmd == "export" else compare()


if __name__ == "__main__":
    main()
