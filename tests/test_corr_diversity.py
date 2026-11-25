"""reproduce/toys/corr_diversity.py: the frozen-error-field toy (Sec. 7, "Guidance reduces diversity")."""
import json
import math

import numpy as np
import pytest

from reproduce.figures import fig_moments as FM
from reproduce.toys import corr_diversity as C


@pytest.fixture
def mix():
    return FM.Mix()


def _xT(n, seed=0):
    return np.random.default_rng(seed).standard_normal((n, 2))


def test_iid_arm_is_m2_bitwise(mix):
    xT = _xT(500)
    for N in (1, 4, 16):
        a = FM.run(mix, 2, N, xT, np.random.default_rng(11))
        b, _ = C.run_field(mix, N, xT, 1.0, C.IID(np.random.default_rng(11)))
        assert np.array_equal(a, b)


def test_alpha_zero_is_ddim_bitwise(mix):
    xT = _xT(300)
    a = FM.run(mix, 1, 8, xT, np.random.default_rng(0))
    for nz in (None, C.IID(np.random.default_rng(1)), C.Field(0.3, 1.0, (5, 0), 256), C.Field(C.INF, 0.0, (5, 1), 64),
               C.TrajField(1.0, C.INF, (5, 2), 300, 32)):
        b, _ = C.run_field(mix, 8, xT, 0.0, nz)
        assert np.array_equal(a, b)


def test_field_is_unit_gaussian():
    x = np.random.default_rng(0).uniform(-2, 2, (20000, 2))       # far apart relative to lx: ~ independent draws
    g = C.Field(0.02, 1.0, (9, 0), 4096)(x, 0.5, 0)
    assert abs(g.mean()) < 0.03
    assert np.allclose(g.var(0), 1.0, atol=0.06)
    assert abs(np.corrcoef(g.T)[0, 1]) < 0.03
    kurt = ((g - g.mean(0)) ** 4).mean(0) / g.var(0) ** 2 - 3
    assert np.all(np.abs(kurt) < 0.2)


@pytest.mark.parametrize("lx,d", [(0.3, 0.15), (0.3, 0.3), (1.0, 0.5)])
def test_field_correlation_matches_kernel_in_x(lx, d):
    p, q = np.array([[0.2, -0.1]]), np.array([[0.2 + d, -0.1]])
    a = np.array([C.Field(lx, 1.0, (3, s), 1024)(p, 0.5, 0)[0] for s in range(600)])
    b = np.array([C.Field(lx, 1.0, (3, s), 1024)(q, 0.5, 0)[0] for s in range(600)])
    rho = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    assert abs(rho - math.exp(-d * d / (2 * lx * lx))) < 0.08


def test_field_correlation_in_log_sigma_and_limits():
    p = np.array([[0.3, 0.1]])
    t1, t2 = 0.5, 0.6                                              # log sigma 0 and 0.405
    ds = C.lsig(t2) - C.lsig(t1)
    a = np.array([C.Field(1.0, 0.5, (4, s), 1024)(p, t1, 0)[0] for s in range(600)])
    b = np.array([C.Field(1.0, 0.5, (4, s), 1024)(p, t2, 1)[0] for s in range(600)])
    rho = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    assert abs(rho - math.exp(-ds * ds / (2 * 0.25))) < 0.08
    f = C.Field(0.5, C.INF, (4, 0), 256)                           # lt = inf: the same function at every t and step
    assert np.allclose(f(p, 0.2, 0), f(p, 0.9, 7))
    f = C.Field(C.INF, 1.0, (4, 0), 256)                           # lx = inf: constant in x
    x = np.random.default_rng(1).normal(size=(50, 2))
    assert np.allclose(f(x, 0.4, 0), f(x[:1], 0.4, 0))
    a = np.array([C.Field(0.5, 0.0, (4, s), 512)(p, 0.5, 0)[0] for s in range(400)])   # lt = 0: new field per step
    b = np.array([C.Field(0.5, 0.0, (4, s), 512)(p, 0.5, 1)[0] for s in range(400)])
    assert abs((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())) < 0.12


def test_shared_randomness_across_lx():
    x = np.random.default_rng(2).normal(size=(40, 2))
    fa, fb = C.Field(0.3, 1.0, (6, 0), 512), C.Field(1.2, 1.0, (6, 0), 512)
    assert np.allclose(fa(x, 0.4, 0), fb(x * (1.2 / 0.3), 0.4, 0))


def test_explicit_pair_equals_alpha_form():
    for n1, n0 in [(64, 16), (1000, 250)]:
        wp, wm = C.w_star(n1, n0)
        assert abs(C.c_w(wp, n1, n0) - 1) < 1e-12 and abs(C.c_w(wm, n1, n0) - 1) < 1e-12
        assert wm < 0 < 1 < wp
    x = np.random.default_rng(0).uniform(-3, 3, (40000, 2))
    n1, n0, w = 64, 16, 3.0
    g = C.Pair(C.Field(0.02, 1.0, (7, 1), 4096), C.Field(0.02, 1.0, (7, 0), 4096), n1, n0, w)(x, 0.5, 0)
    assert np.allclose(g.var(0), C.c_w(w, n1, n0), rtol=0.06)


def test_fresh_field_per_step_matches_m2_in_law(mix):
    n, N = 4000, 8
    xT = _xT(n, 3)
    a = FM.run(mix, 2, N, xT, np.random.default_rng(4))
    b, _ = C.run_field(mix, N, xT, 1.0, C.TrajField(0.3, 0.0, (8, 0), n))
    pa = np.bincount(C.mode_of(mix, a), minlength=10) / n
    pb = np.bincount(C.mode_of(mix, b), minlength=10) / n
    assert np.abs(pa - pb).max() < 0.03
    assert abs(np.trace(np.cov(a.T)) - np.trace(np.cov(b.T))) / np.trace(np.cov(a.T)) < 0.04


def test_metrics_on_data(mix):
    n = 4000
    x = mix.sample(n, np.random.default_rng(12))[0]
    m = C.metrics(mix, x, C.data_ref(mix, n))
    assert abs(m["Hn"] - 1) < 0.01 and m["tv"] < 0.05 and m["dropped"] == 0
    exact = -np.mean(mix.logp(mix.sample(200000, np.random.default_rng(1))[0]))
    assert abs(m["knn3"] - exact) < 0.1
    assert abs(m["spread"] - 1) < 0.08 and abs(m["trcov"] - 1) < 0.05
    assert m["recall"] > 0.9 and m["precision"] > 0.9


def test_commit_t():
    modes = np.array([[0, 0], [1, 2], [2, 2], [2, 2]], np.int8)    # N = 4; trajectory 0 settles at step 2, 1 at 1
    t = C.commit_t(modes, np.array([2, 2]), 4)
    ts = FM.flow_grid(4)
    assert t[0] == ts[2] and t[1] == ts[1]


def test_compute_writes_resumes_and_summarizes(tmp_path, monkeypatch):
    monkeypatch.setenv("VMG_RUNS", str(tmp_path))
    monkeypatch.setattr(C, "slices", lambda n: dict(T=[C.cell(1.0, 0.3, 1.0, 3, n=n), C.cell(1.0, "iid", N=3, n=n),
                                                       C.cell(0.0, N=3, n=n), C.cell(1.0, "exact", N=3, n=n)]))
    keys = C.compute(["T"], procs=1, R=2, RF=2, n=120, verbose=False)
    d = tmp_path / "_toys" / "corr_diversity"
    assert all((d / "cells" / f"{k}.json").exists() and (d / "cells" / f"{k}.npz").exists() for k in keys)
    assert (d / "null_n120.json").exists()
    before = {k: (d / "cells" / f"{k}.json").stat().st_mtime_ns for k in keys}
    C.compute(["T"], procs=1, R=2, RF=2, n=120, verbose=False)
    assert before == {k: (d / "cells" / f"{k}.json").stat().st_mtime_ns for k in keys}
    s = C.summarize(120, verbose=False)
    assert len(s["rows"]) == 4 and {"I", "Hn", "recall", "knn3", "vm"} <= set(s["rows"][0])
    o = json.loads((d / "cells" / f"{C.key(C.cell(0.0, N=3, n=120))}.json").read_text())
    assert abs(o["I"][0]) < 1e-12                                  # DDIM: every pair is the same function


def test_shards_partition_the_cells():
    cells = [c for v in C.slices(4000).values() for c in v]
    keys = {C.key(c) for c in cells}
    parts = [{C.key(c) for c in C.shard_cells(list({C.key(c): c for c in cells}.values()), s, 4)} for s in range(4)]
    assert set().union(*parts) == keys and sum(len(p) for p in parts) == len(keys)
