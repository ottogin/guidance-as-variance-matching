"""The solver against the SD 3.5 Medium calibration pass behind the paper, and on hand-built cases."""
from pathlib import Path

import numpy as np
import pytest

from vmg import solve

FIX = Path(__file__).with_name("fixtures") / "sd35m-coco-s40.npz"


@pytest.fixture(scope="module")
def sd35m():
    z = np.load(FIX)
    m = solve.Moments(sigma=z["sigma"], t=z["t"], var_x0=float(z["var_x0"]), xr=z["xr"], rr=z["rr"],
                      rr_centred=z["rr_centred"], n=int(z["n"]), d=int(z["d"]))
    return m, z


def test_sd35m_schedules_match_the_stored_fits(sd35m):
    m, z = sd35m
    for rule, ref in (("centred", z["w_centred"]), ("datamean", z["w_datamean"])):
        s = solve.fit_schedule(m, rule)
        assert np.abs(s["w"] - ref).max() <= 1e-9, rule
        assert list(np.where(~s["admissible"])[0]) == [39], "only the last level has no root in [1, 40]"
        assert s["w"][39] == s["w"][38], "an inadmissible level holds its nearest neighbour"


def test_sd35m_constant_weights(sd35m):
    m, _ = sd35m
    w, ok = solve.fit_global(m)                     # the default, tr Cov: the fit behind the paper's printed 4.14 +- 0.008
    assert ok and abs(w - 4.13408) < 1e-5
    assert solve.fit_global(m, "centred") == (w, ok)
    w, ok = solve.fit_global(m, "datamean")         # the second-moment variant about the data mean
    assert ok and abs(w - 3.98630) < 1e-5


def test_sd35m_wmse(sd35m):
    m, _ = sd35m
    w = solve.w_mse(m)
    assert np.allclose(w[:3], [1.0743, 1.0339, 1.0153], atol=1e-4), "paper Fig. 2b: above 1 at steps 1-3"
    assert (w[3:] < 1).all()
    assert np.all(solve.mse(m, w) <= solve.mse(m, 1.0) + 1e-15), "w_MSE minimises the error"


def test_root_cases():
    # b w^2 - 2 a w - c = 0 with roots 3 and -1: a = 1, b = 1, c = 3
    assert solve.root(1.0, 1.0, 3.0) == (3.0, True)
    assert solve.root(1.0, 1.0, 3.0, clip=(4.0, 40.0)) == (4.0, False), "outside: clipped, not admissible"
    w, ok = solve.root(0.0, 1.0, -1.0)                 # w^2 = -1: no real root
    assert np.isnan(w) and not ok
    w, ok = solve.root(1.0, 0.0, 1.0)
    assert np.isnan(w) and not ok
    plus, minus = solve.roots(1.0, 1.0, 3.0)
    assert (plus, minus) == (3.0, -1.0)


def test_hold_nearest_prefers_the_noisier_level_on_ties():
    w = solve.hold_nearest([5.0, 0.0, 7.0, 0.0, 0.0], [True, False, True, False, False])
    assert list(w) == [5.0, 5.0, 7.0, 7.0, 7.0]


def test_rules_agree_when_residual_means_vanish():
    rng = np.random.default_rng(0)
    S = 6
    rr = np.abs(rng.normal(size=(S, 3))) + np.array([2.0, 1.0, 3.0])
    mx0 = -rr[:, 0]
    xr = np.stack([mx0, mx0 - 2 * rr[:, 0]], 1)        # a = 3 m00 - m01 > 0, c = m00 > 0: a root near 3.5
    m = solve.Moments(sigma=np.arange(S) + 1.0, t=np.arange(S) + 1.0, var_x0=1.0, xr=xr, rr=rr,
                      rr_centred=rr, n=10, d=1)
    a = solve.fit_schedule(m, "datamean")["raw"]
    b = solve.fit_schedule(m, "centred")["raw"]
    assert np.isfinite(a).all() and np.array_equal(a, b)


def test_matching_identity_holds_at_the_root(sd35m):
    """At w(t) the guided trace covariance equals the data's (the boxed covariance equation, per level)."""
    m, _ = sd35m
    s = solve.fit_schedule(m)
    a, b, c = solve.coefficients(m)
    w = s["raw"][s["admissible"]]
    resid = b[s["admissible"]] * w * w - 2 * a[s["admissible"]] * w - c[s["admissible"]]
    assert np.abs(resid).max() < 1e-10


def test_interp_log_sigma():
    s = np.array([10.0, 1.0, 0.1])
    w = np.array([2.0, 4.0, 6.0])
    got = solve.interp_log_sigma(s, w, np.array([np.inf, 10.0, np.sqrt(10.0), 1.0, 0.01]))
    assert np.allclose(got, [2.0, 2.0, 3.0, 4.0, 6.0])
