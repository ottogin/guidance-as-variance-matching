"""The paper's motion metrics: MDM's own text-to-motion evaluation on KIT-ML (eval/eval_humanml.py, wo_mm).

An evaluation row of MDM is one replication of that protocol (seed 10 + row): MDM generates the test loader's
motions and scores them with its T2M evaluator, in one loop, and MDM.sample() writes the replication's numbers
to rep<row>.json. Every metric is therefore a per-row value, its mean over the replications is MDM's reported
mean, and arms pair replication by replication (the same seed gives the same captions, lengths and noise):
    mdm_fid                          FID of the T2M evaluator's motion embeddings
    r_precision1, 2, 3               R-precision, top 1 / 2 / 3 (batches of 32)
    mm_dist                          MM-Dist (MDM's "Matching Score")
    diversity                        Diversity (300 pairs)
and the same for the real test motions, gt_<name> (identical in every arm at a seed; the paper's Real column).
MultiModality is MDM's mm_short, its own 5 replications (seeds 10-14) of 100 prompts generated 30 times each: the
dataset "kitml-mm", whose rows are those replications (mm<row>.json) and whose one metric is
    multimodality                    MultiModality (10 draws of 10 pairs per prompt)
There are no feature sets: the evaluator's FID is computed inside each replication.
"""
import json

import numpy as np

NAMES = {"FID": "mdm_fid", "Matching Score": "mm_dist", "Diversity": "diversity"}


def spaces(model, dataset):
    return {}


def _values(m):
    out = {}
    for key, name in NAMES.items():
        out[name] = float(m[key])
    for j, v in enumerate(m["R_precision"]):
        out[f"r_precision{j + 1}"] = float(v)
    return out


def _rep(src, i, kind="rep"):
    path = src / f"{kind}{int(i):02d}.json"
    assert path.exists(), f"{path} missing: run sample.py for this arm (one replication per row)"
    return json.load(open(path))


def reference(model, dataset, ids):
    """Nothing to precompute: the real test motions are scored inside every replication (gt_* in score_arm)."""
    return {}, {}


def score_arm(model, dataset, ids, src, ref):
    if dataset == "kitml-mm":
        vals = []
        for i in ids:
            r = _rep(src, i, "mm")
            assert r["seed"] == 10 + int(i) and r.get("mm"), f"{src}/mm{int(i):02d}.json is not mm_short at seed {10 + int(i)}"
            vals.append(float(r["metrics"]["vald"]["Multimodality"]))
        return {}, {"multimodality": np.asarray(vals, dtype=np.float64)}
    rows = {}
    for i in ids:
        r = _rep(src, i)
        assert r["seed"] == 10 + int(i), f"{src}/rep{int(i):02d}.json was run at seed {r['seed']}"
        vals = _values(r["metrics"]["vald"])
        vals.update({f"gt_{k}": v for k, v in _values(r["metrics"]["ground truth"]).items()})
        for k, v in vals.items():
            rows.setdefault(k, []).append(v)
    return {}, {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
