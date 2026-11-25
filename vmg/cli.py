"""Shared command-line plumbing for compute_w.py and compute_wmse.py."""
import argparse
import json
import subprocess
import time

import numpy as np

from . import moments as M
from . import runs
from . import sources
from .models import base

DEFAULT_N = 1024


def parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("the guided pair")
    g.add_argument("--model", required=True,
                   help="a preset (one of: " + ", ".join(base.REGISTRY) + "); with --guidance cfg a diffusers repo id or "
                        "folder; with --guidance cfg-pair / autoguidance an EDM or EDM2 pickle (path or URL), the "
                        "primary model D1; or module:Class, an adapter of your own (vmg.models.base.Model)")
    g.add_argument("--guidance", choices=sources.KINDS, default="auto",
                   help="how D0 is made: cfg = the same network with the empty prompt (SD 3.5, ...); cfg-pair = a "
                        "separately trained unconditional network (EDM2); autoguidance = a weaker model with the same "
                        "conditioning (Karras et al. 2024). Default auto: a preset's own pair; cfg without --reference; "
                        "with one, cfg-pair if the reference is unconditional, else autoguidance")
    g.add_argument("--reference", help="cfg-pair / autoguidance: the reference model D0, an EDM or EDM2 pickle "
                                       "(default: none, i.e. CFG with the empty prompt)")
    g.add_argument("--published-w", type=float,
                   help="the pair's published guidance weight, printed beside ours (default: the preset's, else none)")
    g.add_argument("--steps", type=int, help="sampler steps (default: the model's released setting)")
    g.add_argument("--size", type=int, help="render size of a folder dataset's images (default: the model's)")
    d = ap.add_argument_group("calibration data")
    d.add_argument("--dataset", help="calibration dataset: one the model lists (default: its first), or "
                                     "folder:<dir> (images, with captions or labels in <dir>/index.jsonl)")
    d.add_argument("--n", default=str(DEFAULT_N),
                   help=f"calibration rows: the first N, or 'all' (default: {DEFAULT_N}). A subset run is "
                        "kept apart from the full one (run folder <dataset>-n<N>)")
    r = ap.add_argument_group("running")
    r.add_argument("--shard", type=int, default=0, help="this job's shard (default: %(default)s)")
    r.add_argument("--shards", type=int, default=1, help="shards of the moments pass (default: %(default)s)")
    r.add_argument("--stage", choices=("all", "moments", "fit"), default="all",
                   help="moments = the GPU pass for this shard; fit = merge every shard and solve (CPU); "
                        "all = both (default)")
    r.add_argument("--batch", type=int, help="calibration rows per forward batch (default: the model's)")
    r.add_argument("--boot", type=int, default=200,
                   help="row-bootstrap replicates for the standard errors (default: %(default)s)")
    r.add_argument("--tag", help="keep this run apart from the default one, e.g. a step-count variant (default: none; "
                                 "n<N> for a subset)")
    r.add_argument("--smoke", action="store_true",
                   help="check the adapter against its pipeline, then run everything on 2 rows (tag 'smoke')")
    r.add_argument("--redo", action="store_true", help="recompute moments shards that already exist")
    r.add_argument("--extras", default="", help="extra sums of the moments pass, e.g. chancov (vmg/extras.py; default: "
                                                "those --match needs)")
    return ap


def finish_args(a):
    """Settle the row count, the run tag and the model. Sets a.model_obj, a.key (the run folder's model name),
    a.dataset and a.n (None = every calibration row)."""
    model = sources.resolve(a)
    a.model_obj, a.key = model, model.spec.key
    a.dataset = a.dataset or model.spec.default_dataset
    if a.smoke:
        a.n, a.tag, a.shard, a.shards = 2, a.tag or "smoke", 0, 1
        return a
    n = str(a.n).strip().lower()
    if n in ("all", "0", "none", ""):
        a.n = None
    else:
        a.n = int(n)
        assert a.n > 0, "--n must be positive (or 'all')"
        total = len(model.rows(a.dataset, "calib"))
        if a.n >= total:
            a.n = None                         # the whole calibration set: the default run folder
        elif not a.tag:
            a.tag = f"n{a.n}"
    return a


def model_and_dataset(a):
    return a.model_obj, a.dataset


def calib_rows(a, model, dataset):
    rows = model.rows(dataset, "calib")
    return rows[: a.n] if a.n else rows


def moments_stage(a, model, dataset, extras=()):
    """Run (or reuse) this shard's moments pass; `extras` are the extra sums it must hold."""
    from . import extras as X
    names = list(dict.fromkeys([e for e in a.extras.split(",") if e] + list(extras)))
    path = runs.moments_shard(a.key, dataset, a.shard, a.shards, a.tag)
    if path.exists() and not a.redo:
        have = set(json.loads(str(np.load(path)["meta"])).get("extras", []))
        missing = [e for e in names if e not in have]
        assert not missing, (f"{path} exists but lacks the {', '.join(missing)} sums this fit needs: rerun the "
                             f"moments stage with --redo")
        print(f"moments: {path} exists (use --redo to recompute)")
        return path
    rows = calib_rows(a, model, dataset)
    al = max(int(model.row_align), 1)
    cut = lambda k: len(rows) if k == a.shards else (len(rows) * k // a.shards) // al * al     # noqa: E731
    lo, hi = cut(a.shard), cut(a.shard + 1)
    model.load()
    if a.smoke:
        print("smoke: checking the adapter against the released pipeline")
        model.smoke()
    print(f"moments: {model.spec.name} on {dataset}, rows {lo}..{hi - 1} of {len(rows)}, "
          f"{len(model.calib_levels())} levels{', extras ' + ','.join(names) if names else ''}")
    sums, meta = M.accumulate(model, dataset, rows[lo:hi], batch=a.batch, extras=X.make(names))
    meta["git"] = git_sha()
    meta["n_rows"] = len(rows)
    M.save(path, sums, meta)
    print(f"wrote {path}")
    return path


def load_moments(a, dataset):
    paths = runs.moments_shards(a.key, dataset, a.tag)
    assert paths, f"no moments under {runs.run_dir(a.key, dataset, a.tag)}; run the moments stage first"
    K = int(paths[0].stem.rsplit("-of-", 1)[1])
    assert len(paths) == K, f"{len(paths)} of {K} moments shards present"
    sums, meta = M.load(paths)
    want = int(meta.get("n_rows", len(sums["rows"])))
    assert len(sums["rows"]) == want, f"the shards hold {len(sums['rows'])} rows, the pass was cut for {want}"
    if a.n and len(sums["rows"]) != a.n:
        raise SystemExit(f"{runs.run_dir(a.key, dataset, a.tag)} holds a pass on {len(sums['rows'])} rows, not --n "
                         f"{a.n}: use another --tag or --redo")
    return sums, meta


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(runs.REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def header(model, dataset, meta):
    return dict(model=model.spec.key, name=model.spec.name, checkpoint=model.spec.checkpoint,
                reference=model.spec.reference, dataset=dataset, n=None, d=meta["d"],
                shards=meta.get("shards", 1), seed=meta["seed"], levels=meta["levels"],
                sigma=meta["sigma"], t=meta["t"], published_w=model.spec.published_w,
                published_note=model.spec.published_note, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                git=git_sha())


def dump(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=conv)
