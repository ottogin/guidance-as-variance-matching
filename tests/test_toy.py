"""End to end on the exact 2-D mixture: the closed forms of App. 2-D hold, and sharding changes nothing."""
import numpy as np
import pytest
import torch

from vmg import moments as M
from vmg import noise, solve
from vmg.models import get


@pytest.fixture(scope="module")
def toy():
    return get("toy")


@pytest.fixture(scope="module")
def pass_(toy):
    rows = toy.rows("mixture")
    sums, meta = M.accumulate(toy, "mixture", rows, log=lambda *a: None)
    return sums, meta


def test_smoke(toy):
    toy.smoke()


def test_exact_predictors(toy, pass_):
    """a(t) = 0, w_MSE(t) = 1 and w(t) = sqrt(V / (V - V_c)), within 4 bootstrap standard errors.

    The row bootstrap holds the data and residual means fixed, so it misses the O(1/n) product of two
    sample-mean errors in a(t); that term is bounded separately by sqrt(Var[x0] b) / n."""
    sums, meta = pass_
    m = M.reduce(sums, meta)
    a, b, c = solve.coefficients(m)
    sch = solve.fit_schedule(m)
    boot = list(M.bootstrap(sums, meta, reps=100))
    se_a = np.std([solve.coefficients(r)[0] for r in boot], 0)
    se_mse = np.std([solve.w_mse(r) for r in boot], 0)
    se_w = np.std([solve.fit_schedule(r)["raw"] for r in boot], 0)
    mean_term = np.sqrt(m.var_x0 * b) / m.n
    assert (np.abs(a) <= 4 * (se_a + mean_term)).all(), "a(t) = 0 for exact predictors"
    assert (np.abs(solve.w_mse(m) - 1) <= 4 * se_mse).all(), "w_MSE(t) = 1 for exact predictors"
    # the closed-form posterior traces at the very inputs the pass used
    x0, _ = toy.draw(meta_rows := list(sums["rows"].astype(int)))
    for j, k in enumerate(meta["levels"]):
        sig = float(toy.sig[k])
        if np.isinf(sig):
            continue
        x = toy.noise(x0, k, noise.eps_batch(toy.seed, meta_rows, k, (2,)))
        V, Vc = (v.double().mean().item() for v in toy.posterior_trace(x, sig))
        want = np.sqrt(V / (V - Vc))
        assert abs(sch["raw"][j] - want) <= 4 * se_w[j] + 1e-6, (k, sch["raw"][j], want, se_w[j])


def test_shards_and_batches_give_the_same_moments(toy, pass_):
    sums, meta = pass_
    rows = toy.rows("mixture")
    parts = []
    for k in range(4):
        sl = rows[len(rows) * k // 4: len(rows) * (k + 1) // 4]
        parts.append(M.accumulate(toy, "mixture", sl, batch=7 if k % 2 else 64, log=lambda *a: None))
    for key in ("q_xx", "q_xr", "q_rr", "cnt"):
        got = np.concatenate([p[0][key] for p in parts])
        assert np.array_equal(got, sums[key]), f"per-row {key} must not depend on sharding or batching"
    whole = M.reduce(sums, meta)
    merged = {key: np.concatenate([p[0][key] for p in parts]) for key in M.ROW_KEYS}
    merged.update({key: sum(p[0][key] for p in parts) for key in M.FIRST_KEYS})
    split = M.reduce(merged, meta)
    assert np.allclose(split.rr, whole.rr, rtol=1e-12) and np.allclose(split.xr, whole.xr, rtol=1e-9, atol=1e-15)
    assert abs(solve.fit_global(split)[0] - solve.fit_global(whole)[0]) < 1e-10


def test_save_load_round_trip(toy, pass_, tmp_path):
    sums, meta = pass_
    half = len(sums["rows"]) // 2
    a = {k: sums[k][:half] for k in M.ROW_KEYS}
    b = {k: sums[k][half:] for k in M.ROW_KEYS}
    # first moments are additive; split them as the two halves' own sums would be
    rows = toy.rows("mixture")
    sa, _ = M.accumulate(toy, "mixture", rows[:half], log=lambda *x: None)
    a.update({k: sa[k] for k in M.FIRST_KEYS})
    b.update({k: sums[k] - sa[k] for k in M.FIRST_KEYS})
    M.save(tmp_path / "s0.npz", a, meta)
    M.save(tmp_path / "s1.npz", b, meta)
    got, gm = M.load([tmp_path / "s0.npz", tmp_path / "s1.npz"])
    assert gm["shards"] == 2 and np.array_equal(got["rows"], sums["rows"])
    assert abs(solve.fit_global(M.reduce(got, gm))[0] - solve.fit_global(M.reduce(sums, meta))[0]) < 1e-10
    with pytest.raises(AssertionError, match="overlap"):
        M.load([tmp_path / "s0.npz", tmp_path / "s0.npz"])


def test_guided_sampling_runs(toy, tmp_path):
    rows = toy.rows("mixture", "eval")[:512]
    x1 = toy.sample("mixture", rows, None, tmp_path)
    x3 = toy.sample("mixture", rows, 3.0, tmp_path)
    assert x1.shape == (512, 2) and torch.isfinite(x3).all()
    assert not torch.equal(x1, x3)
