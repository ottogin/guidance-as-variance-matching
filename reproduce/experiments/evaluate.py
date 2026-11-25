#!/usr/bin/env python
"""Score one sampled arm against its real reference with the paper's metrics for the model's modality.

    python reproduce/experiments/evaluate.py --model sd35m --dataset coco --arm pub [--n N] [--tag T]

Reads runs/<model>/<dataset>/samples/<arm>/ (sample.py) and writes runs/<model>/<dataset>/metrics/<arm>.json, with
the arm's per-row features and per-row scores beside it (<arm>.feats.npz) so that compare.py and any rescoring
never touch the GPU. The metrics come from the modality's module in vmg/metrics (for images: FID, FD_DINOv2,
precision / recall / density / coverage with k = 3 in both feature spaces, and CLIP score, at a 256 px metric
size). The real reference is the same evaluation rows' own data; its features are computed once per (modality,
dataset, rows) under $VMG_RUNS/_refs/ and shared by every model and arm.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import cli, runs  # noqa: E402
from vmg.metrics import common as C, for_modality  # noqa: E402


def reference(mod, model, dataset, ids):
    path = C.ref_path(model, dataset, ids)
    if path.exists():
        feats, per_row, meta = C.load(path)
        assert meta.get("rows") == list(map(int, ids)), f"{path} holds other rows"
        return feats, per_row
    path.parent.mkdir(parents=True, exist_ok=True)
    feats, per_row = mod.reference(model, dataset, ids)
    C.save(path, feats, per_row, rows=list(map(int, ids)))
    return feats, per_row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--arm", required=True, help="a folder under runs/<model>/<dataset>/samples/")
    ap.add_argument("--n", type=int, help="only the first N evaluation rows")
    ap.add_argument("--tag")
    a = ap.parse_args()
    from vmg.models import get
    model = get(a.model)
    dataset = a.dataset or model.spec.default_dataset
    ids = model.rows(dataset, "eval")
    ids = ids[:a.n] if a.n else ids
    mod = for_modality(model.spec.modality)
    real, real_rows = reference(mod, model, dataset, ids)
    src = runs.samples_dir(a.model, dataset, a.arm, a.tag)
    fake, per_row = mod.score_arm(model, dataset, ids, src, (real, real_rows))
    out = runs.metrics_json(a.model, dataset, a.arm, a.tag)
    C.save(out.with_suffix(".feats.npz"), fake, per_row, rows=list(map(int, ids)), modality=model.spec.modality)
    res = C.scores(fake, real, mod.spaces(model, dataset))
    res.update({k: float(np.mean(v)) for k, v in per_row.items()}, n=len(ids), prdc_k=C.PRDC_K)
    if (src / "weights.json").exists():              # written by sample.py: the weight of every sampler level
        res.update(json.load(open(src / "weights.json")))
    res = dict(model=a.model, dataset=dataset, arm=a.arm, tag=a.tag, created=time.strftime("%Y-%m-%d %H:%M:%S"),
               git=cli.git_sha(), **res)
    cli.dump(out, res)
    print(" ".join(f"{k}={v:.4f}" for k, v in res.items() if isinstance(v, float)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
