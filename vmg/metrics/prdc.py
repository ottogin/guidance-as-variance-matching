"""Precision, recall, density and coverage between two sets of detector features.

    from vmg.metrics.prdc import prdc
    prdc(real, fake, k=3)      # -> dict(precision, recall, density, coverage)

FID folds fidelity and diversity into one distance, so it cannot say WHICH of the two
guidance trades away.  These split it, on the same Inception / DINOv2 features as the
Frechet distances (vmg/metrics/image.py):

  precision (Kynkaanniemi et al. 2019)  fraction of generated samples inside the real
                                        manifold: the union of balls around each real
                                        point, out to its k-th nearest real neighbour
  recall                                the same with the roles swapped -- fraction of
                                        real samples inside the generated manifold
  density (Naeem et al. 2020)           how many real balls a generated sample sits in,
                                        divided by k.  About 1 when the two sets come from
                                        one distribution, and NOT capped at 1, so it sees
                                        samples piling into dense regions, which
                                        precision cannot (it saturates)
  coverage                              fraction of real samples whose OWN ball holds at
                                        least one generated sample; unlike recall, one
                                        generated outlier cannot inflate it

All four depend on n and k, so compare runs only at the same (n_real, n_fake, k).  k = 3
is the value both papers use.

The ball test is `<=`, as in Kynkaanniemi's reference code.  For two distinct continuous
sets the choice is immaterial (ties have probability zero); it matters only when a
generated sample IS a real one, where `<=` counts the k-th neighbour on the boundary.  So
fake == real gives density (k+1)/k here, and 1 under Naeem's `<`.

Distances are float64 through |a|^2 + |b|^2 - 2 a.b, clamped at 0, and compared SQUARED
so that no square root rounds a boundary point across the ball's edge.  Nothing N x M is
ever resident: the fake rows go through in chunks, and each chunk updates every per-row
and per-column quantity it touches.
"""
import numpy as np


def _sq(x):
    return np.einsum("ij,ij->i", x, x)


def _sqdist(a, b, a2, b2):
    """Squared Euclidean distances [len(a), len(b)], float64, never negative."""
    return np.maximum(a2[:, None] + b2[None, :] - 2.0 * (a @ b.T), 0.0)


def kth_sqdist(x, k, chunk=1024):
    """Squared distance from each row of x to its k-th nearest OTHER row.

    A row's distance to itself is 0 only up to rounding, and a true duplicate elsewhere in
    the set can come out smaller, so self is forced below everything rather than trusted
    to sort first; the k-th smallest of the rest is then index k of the partition.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    assert 0 < k < n, f"k = {k} needs more than {n} points"
    x2 = _sq(x)
    out = np.empty(n)
    for i in range(0, n, chunk):
        d = _sqdist(x[i:i + chunk], x, x2[i:i + chunk], x2)
        b = len(d)
        d[np.arange(b), np.arange(i, i + b)] = -1.0
        out[i:i + b] = np.partition(d, k, axis=1)[:, k]
    return out


def prdc(real, fake, k=3, chunk=1024, r_real=None):
    """dict(precision, recall, density, coverage) of `fake` against `real` ([n, d] each).

    r_real: kth_sqdist(real, k), when several fake sets are scored against one real set.
    """
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    assert real.ndim == 2 and fake.ndim == 2 and real.shape[1] == fake.shape[1], (
        f"feature shapes {real.shape} and {fake.shape} do not pair up")
    n, m = len(real), len(fake)
    r_real = kth_sqdist(real, k, chunk) if r_real is None else r_real
    assert len(r_real) == n, "r_real is not this real set's radii"
    r_fake = kth_sqdist(fake, k, chunk)
    x2, f2 = _sq(real), _sq(fake)

    in_real = np.zeros(m, dtype=bool)      # fake j inside the real manifold
    balls = np.zeros(m, dtype=np.int64)    # how many real balls hold fake j
    in_fake = np.zeros(n, dtype=bool)      # real i inside the fake manifold
    nearest = np.full(n, np.inf)           # real i's squared distance to the nearest fake
    for j in range(0, m, chunk):
        d = _sqdist(fake[j:j + chunk], real, f2[j:j + chunk], x2)     # [b, n]
        inside = d <= r_real[None, :]
        in_real[j:j + len(d)] = inside.any(1)
        balls[j:j + len(d)] = inside.sum(1)
        in_fake |= (d <= r_fake[j:j + len(d), None]).any(0)
        np.minimum(nearest, d.min(0), out=nearest)
    return dict(precision=float(in_real.mean()), recall=float(in_fake.mean()),
                density=float(balls.sum() / (k * m)),
                coverage=float((nearest <= r_real).mean()))
