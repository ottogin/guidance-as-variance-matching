"""Modality-independent pieces of the metrics: Frechet distance, the k-NN metrics per space, feature files."""
import json
import os
import socket

import numpy as np

from .. import runs
from .prdc import prdc

PRDC_K = 3


def frechet(fa, fb):
    """Frechet distance between two feature sets [n, d], as EDM2's calculate_metrics_from_stats."""
    import scipy.linalg
    fa, fb = (np.asarray(f, dtype=np.float64) for f in (fa, fb))
    ma, mb = fa.mean(0), fb.mean(0)
    sa, sb = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    s = scipy.linalg.sqrtm(sa @ sb)
    return float(np.real(np.square(ma - mb).sum() + np.trace(sa + sb - 2 * s)))


def frechet_stats(fa, mu_b, sigma_b):
    """Frechet distance between a feature set [n, d] and a reference given by its mean and covariance."""
    import scipy.linalg
    fa = np.asarray(fa, dtype=np.float64)
    ma, sa = fa.mean(0), np.cov(fa, rowvar=False)
    s = scipy.linalg.sqrtm(sa @ np.asarray(sigma_b, dtype=np.float64))
    return float(np.real(np.square(ma - np.asarray(mu_b)).sum() + np.trace(sa + sigma_b - 2 * s)))


def rows_of(feats, sp, how, idx=None):
    """Space sp's features for the evaluation rows idx (all if None). A space with rows=<key> holds several
    features per evaluation row, and feats[key] gives each feature's row position."""
    X = np.asarray(feats[sp])
    if idx is None:
        return X
    g = how.get("rows")
    return X[np.isin(np.asarray(feats[g]), idx)] if g else X[idx]


def scores(fake, real, spaces, k=PRDC_K, idx=None, fd=True, r_real=None):
    """Every Frechet and k-NN metric of `spaces` for one arm against the real set (rows `idx`, or all). Each
    space is a dict of
        fd=<name>        the Frechet distance's name (None: no distance)
        prdc=True        precision / recall / density / coverage, named <metric>_<space>
        fd_stats=(mu, sigma)  take the Frechet distance against these reference statistics (e.g. a published
                         training-set reference) instead of the real rows, which then serve the k-NN metrics only
        fixed_ref=True   the real features are one fixed set, not one per evaluation row (e.g. the 884 AudioCaps
                         test clips behind 4,411 captions): used whole, never resampled
        rows=<key>       several features per evaluation row (frames, windows; see rows_of)
        stats={name: f}  statistics of the arm's own features, f(features) -> float (an Inception Score)"""
    from .prdc import kth_sqdist
    out = {}
    for sp, how in spaces.items():
        fixed = how.get("fixed_ref", False)
        F = rows_of(fake, sp, how, idx)
        R = None if sp not in real else np.asarray(real[sp]) if fixed else rows_of(real, sp, how, idx)
        if R is not None and not fixed and not how.get("rows"):
            assert len(F) == len(R), f"{sp}: {len(F)} generated vs {len(R)} real rows"
        if fd and how.get("fd"):
            st = how.get("fd_stats")
            out[how["fd"]] = frechet_stats(F, *st) if st is not None else frechet(F, R)
        for name, f in how.get("stats", {}).items():
            out[name] = float(f(F))
        if how.get("prdc"):
            rr = r_real[sp] if r_real is not None and sp in r_real else kth_sqdist(R, k)
            for name, v in prdc(R, F, k=k, r_real=rr).items():
                out[f"{name}_{sp}"] = v
    return out


def ref_path(model, dataset, ids):
    """Where the real reference of these evaluation rows is cached (shared by every model of a modality, unless
    an adapter names its own reference with `metric_ref`)."""
    key = getattr(model, "metric_ref", model.spec.modality)
    return runs.root() / "_refs" / f"{key}-{dataset}-{ids[0]}-{len(ids)}.npz"


def save(path, feats, per_row=None, **meta):
    tmp = f"{path}.{socket.gethostname()}-{os.getpid()}.tmp.npz"    # jobs may write one shared reference at once
    np.savez(tmp, **{f"feat_{k}": v for k, v in feats.items()}, **{f"row_{k}": v for k, v in (per_row or {}).items()},
             meta=json.dumps(meta))
    os.replace(tmp, path)


def load(path):
    """(feats, per_row, meta) of a file written by save()."""
    z = np.load(path, allow_pickle=True)
    feats = {k[5:]: z[k] for k in z.files if k.startswith("feat_")}
    per_row = {k[4:]: z[k] for k in z.files if k.startswith("row_")}
    return feats, per_row, json.loads(str(z["meta"])) if "meta" in z.files else {}
