"""vmg/extras.py chancov: raw channel Grams plus the pass's first moments give the centred matrices of fig:covmatch."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduce.figures import fig_covmatch as F  # noqa: E402


def test_centring_from_raw_grams_matches_direct_products():
    rng = np.random.default_rng(0)
    n, C, P, S = 50, 4, 5, 2
    x0 = rng.normal(size=(n, C, P)) + rng.normal(size=(C, P))
    r = rng.normal(size=(S, 2, n, C, P)) + rng.normal(size=(S, 2, 1, C, P))       # columns (r0, r1)
    G = np.stack([np.einsum("ncp,ndp->cd", u, u) for u in (np.concatenate([x0, r[j, 0], r[j, 1]], 1)
                                                            for j in range(S))])
    sums = dict(x_chancov_G=G, cnt=np.ones(n), s_x=x0.sum(0).ravel(), s_r=r.sum(2).reshape(S, 2, -1))
    E, MM = F.matrices(sums, dict(d=C * P))
    X = x0 - x0.mean(0)
    cp = lambda a, b: np.tensordot(a, b, axes=([0, 2], [0, 2])) / (n * P)       # noqa: E731
    sym = lambda g: g + g.T                                                      # noqa: E731
    for j in range(S):
        R0, R1 = r[j, 0] - r[j, 0].mean(0), r[j, 1] - r[j, 1].mean(0)
        want = [sym(cp(X, R0)) + cp(R0, R0),
                sym(cp(X, R1)) - sym(cp(X, R0)) + sym(cp(R1, R0)) - 2 * cp(R0, R0),
                cp(R1, R1) - sym(cp(R1, R0)) + cp(R0, R0)]
        assert np.allclose(E[j], want, atol=1e-12)
        m0, d = r[j, 0].mean(0), r[j, 1].mean(0) - r[j, 0].mean(0)
        assert np.allclose(MM[j], [m0 @ m0.T / P, sym(m0 @ d.T / P), d @ d.T / P], atol=1e-12)
    # B = Cov(x0) - Cov(D0) is -E0, A = Cov(D1 - D0) is E2
    D0, D1 = x0 + r[0, 0], x0 + r[0, 1]
    cov = lambda a: cp(a - a.mean(0), a - a.mean(0))                             # noqa: E731
    assert np.allclose(-E[0, 0], cov(x0) - cov(D0)) and np.allclose(E[0, 2], cov(D1 - D0))
