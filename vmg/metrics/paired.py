"""Paired 95% intervals for the difference of a metric between two arms that share rows.

Arms are paired: row i of every arm and of the real reference has the same prompt (or conditioning) and the
same initial noise. Each replicate draws one half-sample of the rows WITHOUT replacement and applies it to both
arms and to the real set, so the row-to-row spread cancels from the difference. Half-samples rather than a
with-replacement bootstrap, because a duplicated row is its own nearest neighbour at distance 0, which shrinks
every k-NN radius and biases precision and recall. For m = n / 2 the finite-population correction makes a
half-sample statistic's spread about its full-sample standard error (Var = s^2/m (1 - m/n) = s^2/n), so the
interval is the full-n difference plus the half-sample differences' own 2.5 / 97.5% deviations. Only
differences get intervals: the k-NN metrics and Frechet distances also move in level with n, and that shift is
common to both arms.
"""
import numpy as np

from .common import PRDC_K, rows_of, scores
from .prdc import kth_sqdist


def compare(real, a, b, spaces, k=PRDC_K, reps=100, seed=0, fd=False, per_row=None, n=None):
    """{metric: dict(a, b, diff[, lo, hi, excludes_zero])} for arm b minus arm a. `real`, `a`, `b` map feature
    space -> features in row order; `spaces` is the modality's spaces() (common.scores). per_row: {name:
    (values_a [n], values_b [n])} for per-row metrics (CLIP score, PSNR, ...), whose difference is a plain mean on
    the same half-samples. n is the number of evaluation rows (default: a per-row metric's length, else that of
    a space with one feature per row). fd=True also intervals the Frechet distances (slow: two matrix square
    roots per replicate and space); without it they are reported without an interval. A fixed_ref space keeps
    its whole real set, and its k-NN radii, in every half-sample."""
    if n is None:
        n = len(next(iter(per_row.values()))[0]) if per_row else \
            next(len(a[sp]) for sp, how in spaces.items() if not how.get("rows"))
    full_a, full_b = scores(a, real, spaces, k), scores(b, real, spaces, k)
    for name, (va, vb) in (per_row or {}).items():
        full_a[name], full_b[name] = float(np.mean(va)), float(np.mean(vb))
    rng = np.random.default_rng(seed)
    diffs = {m: [] for m in full_a}
    fixed = {sp: kth_sqdist(np.asarray(real[sp]), k) for sp, how in spaces.items()
             if how.get("prdc") and how.get("fixed_ref")}
    for _ in range(reps):
        idx = np.sort(rng.choice(n, n // 2, replace=False))
        rr = {sp: fixed[sp] if sp in fixed else kth_sqdist(rows_of(real, sp, how, idx), k)
              for sp, how in spaces.items() if how.get("prdc")}
        sa = scores(a, real, spaces, k, idx=idx, fd=fd, r_real=rr)
        sb = scores(b, real, spaces, k, idx=idx, fd=fd, r_real=rr)
        for m in sa:
            diffs[m].append(sb[m] - sa[m])
        for name, (va, vb) in (per_row or {}).items():
            diffs[name].append(float(np.mean(np.asarray(vb)[idx] - np.asarray(va)[idx])))
    out = {}
    for m in full_a:
        d = full_b[m] - full_a[m]
        rec = dict(a=full_a[m], b=full_b[m], diff=d)
        if diffs[m]:
            x = np.asarray(diffs[m])
            lo, hi = d + np.percentile(x - x.mean(), [2.5, 97.5])
            rec.update(lo=float(lo), hi=float(hi), excludes_zero=bool(lo > 0 or hi < 0))
        out[m] = rec
    return out
