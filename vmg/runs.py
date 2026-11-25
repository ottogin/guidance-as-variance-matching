"""Where every pipeline output lives. The only module that builds paths.

    $VMG_RUNS/<model>/<dataset>/moments/shard-<k>-of-<K>.npz    compute_w / compute_wmse, GPU stage
                               /w.json  /wmse.json              CPU fit
                               /samples/<arm>/                  sample.py
                               /metrics/<arm>.json              reproduce/experiments/evaluate.py
                               /ref/                            real-data features for the metrics

VMG_RUNS defaults to ./runs in the repository.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def root():
    return Path(os.environ.get("VMG_RUNS", REPO / "runs"))


def run_dir(model, dataset, tag=None):
    """tag separates variants of one pair, e.g. a smoke run ("smoke") or a different step count."""
    d = root() / model / (dataset if tag is None else f"{dataset}-{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def moments_shard(model, dataset, shard, shards, tag=None):
    d = run_dir(model, dataset, tag) / "moments"
    d.mkdir(exist_ok=True)
    return d / f"shard-{shard:03d}-of-{shards:03d}.npz"


def moments_shards(model, dataset, tag=None):
    return sorted((run_dir(model, dataset, tag) / "moments").glob("shard-*-of-*.npz"))


def w_json(model, dataset, tag=None, rule=None):
    """compute_w.py's result: w.json for the default rule (tr Cov), w-datamean.json for the --datamean fit."""
    from .solve import DEFAULT
    rule = rule or DEFAULT
    return run_dir(model, dataset, tag) / ("w.json" if rule == DEFAULT else f"w-{rule}.json")


def wmse_json(model, dataset, tag=None):
    return run_dir(model, dataset, tag) / "wmse.json"


def samples_dir(model, dataset, arm, tag=None):
    d = run_dir(model, dataset, tag) / "samples" / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_image(im, path):
    """Write a sample image (PIL image or uint8 array) atomically. Sampling resumes by skipping rows whose file
    exists, so a job preempted mid-write must not leave a truncated file behind."""
    from PIL import Image
    im = im if isinstance(im, Image.Image) else Image.fromarray(im)
    path = Path(path)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    im.save(tmp)
    os.replace(tmp, path)


def metrics_json(model, dataset, arm, tag=None):
    d = run_dir(model, dataset, tag) / "metrics"
    d.mkdir(exist_ok=True)
    return d / f"{arm}.json"


def ref_dir(model, dataset, tag=None):
    d = run_dir(model, dataset, tag) / "ref"
    d.mkdir(exist_ok=True)
    return d
