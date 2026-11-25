#!/usr/bin/env python
"""Paired comparison of two evaluated arms: every metric's difference with its paired 95% interval.

    python reproduce/experiments/compare.py --model sd35m --dataset coco --a pub --b vm-schedule [--fd]

Reads the per-row features and scores that evaluate.py kept beside each arm's metrics, and the shared real
reference, and writes runs/<model>/<dataset>/metrics/compare-<b>-vs-<a>.json (vmg/metrics/paired.py: 100 paired
half-samples). CPU only.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import cli, runs  # noqa: E402
from vmg.metrics import common as C, for_modality  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--a", required=True, help="the baseline arm, e.g. pub")
    ap.add_argument("--b", required=True, help="the arm compared against it, e.g. vm-schedule")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--fd", action="store_true", help="also interval the Frechet distances (slow)")
    ap.add_argument("--tag")
    a = ap.parse_args()
    from vmg.models import get
    model = get(a.model)
    (fa, ra, ma), (fb, rb, mb) = (C.load(runs.metrics_json(a.model, a.dataset, arm, a.tag).with_suffix(".feats.npz"))
                                  for arm in (a.a, a.b))
    assert ma["rows"] == mb["rows"], "the two arms were evaluated on different rows"
    real, _, meta = C.load(C.ref_path(model, a.dataset, ma["rows"]))
    assert meta["rows"] == ma["rows"], "the reference holds other rows"
    mod = for_modality(model.spec.modality)
    from vmg.metrics.paired import compare
    res = compare(real, fa, fb, mod.spaces(model, a.dataset), reps=a.reps, fd=a.fd, n=len(ma["rows"]),
                  per_row={k: (ra[k], rb[k]) for k in ra if k in rb})
    out = dict(model=a.model, dataset=a.dataset, a=a.a, b=a.b, n=len(ma["rows"]), reps=a.reps, metrics=res,
               git=cli.git_sha())
    path = runs.metrics_json(a.model, a.dataset, f"compare-{a.b}-vs-{a.a}", a.tag)
    cli.dump(path, out)
    for m, r in res.items():
        ci = f"  [{r['lo']:+.4f}, {r['hi']:+.4f}]{' *' if r['excludes_zero'] else ''}" if "lo" in r else ""
        print(f"  {m:24s} {r['a']:10.4f} {r['b']:10.4f} {r['diff']:+10.4f}{ci}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
