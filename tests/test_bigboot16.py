"""fig:scalecopy's 16-D Gaussian toy (reproduce/toys/bigboot16.py): the shard plan, the reduction's closed forms
and a few-step training run, all on the CPU in seconds."""
import json

import numpy as np
import pytest
import torch

from reproduce.toys import bigboot16 as B


def test_plan_is_the_papers_81k_nets():
    seen, seeds = set(), set()
    per = {nm: 0 for nm, _ in B.ARMS}
    for k in range(len(B.plan())):
        for g, nm, _ in B.shard_models(k):
            assert g not in seen, f"training set {g} appears twice"
            seen.add(g)
            per[nm] += 1
        seeds.add(B.plan()[k][4])
    assert len(B.plan()) == 170 and per == {nm: 81000 for nm, _ in B.ARMS} and len(seeds) == 170
    assert B.shard_models(0)[:2] == [(0, "ind10k", 10000), (1, "ind10k", 10000)]           # the first run's sets
    assert B.shard_models(10)[0] == (3000, "ind10k", 10000) and B.shard_models(10)[500] == (103000, "ind30k", 30000)


def test_posterior_is_gaussian_conditioning():
    g = B.EmbeddedGaussian()
    Sig = g.A @ g.A.T
    x = torch.randn(5, 16, dtype=torch.float64)
    for s in (0.05, 1.0):
        R = Sig @ torch.linalg.inv(Sig + s * s * torch.eye(16, dtype=torch.float64))
        assert torch.allclose(g.denoise(x, s), x @ R.T, atol=1e-10)
        assert torch.allclose(g.post_cov(s), Sig - R @ Sig, atol=1e-10)


def _moments(C, n):
    """One part whose sample covariance is exactly C: zero first moment, s2 = (n - 1) C."""
    return (n, np.zeros(C.shape[:-1]), (n - 1) * C)


def test_reduce_recovers_an_exact_scaled_copy(tmp_path):
    rng = np.random.default_rng(0)
    S_, Q, d, a = len(B.SIGMAS), 7, 16, 2.7
    X = rng.standard_normal((S_, Q, d, d))
    S0 = X @ X.transpose(0, 1, 3, 2) / d                    # the strong data part
    Y = rng.standard_normal((S_, Q, d, d))
    Fr = Y @ Y.transpose(0, 1, 3, 2) / d                    # training noise
    q = rng.standard_normal((S_, Q, d))
    for j in range(4):                                       # four equal shards (so the jackknife runs)
        mom = {"q": q, "sigmas": np.array(B.SIGMAS)}
        for fam, C in (("ind10k", Fr + a * S0), ("ind30k", Fr + S0), ("inf", Fr)):
            n, s1, s2 = _moments(C, 1001)
            mom.update({f"{fam}_n": 250, f"{fam}_s1": s1, f"{fam}_s2": s2 * 250 / 1001})
        np.savez(tmp_path / f"shard-{j:03d}.npz", meta=json.dumps({}), **mom)
    R = B.reduce([tmp_path], tmp_path / "out.npz", blocks=4)
    # the pooled covariance of four parts of 250 with these sums is C * (1000 / 999); the scale cancels in W ~ a S
    assert np.allclose(R["a_loc"], a) and np.allclose(R["a_uni"], a) and np.allclose(R["a2"], a)
    for key in ("L_loc", "L_sig", "L_uni"):
        assert np.abs(R[key]).max() < 1e-10 and np.abs(R[key + "_se_nets"]).max() < 1e-10


def test_a_few_nets_train_and_write_moments(tmp_path, monkeypatch):
    monkeypatch.setenv("VMG_RUNS", str(tmp_path))
    models = [m for m in B.shard_models(0) if m[0] % 100 < 2]                      # 2 nets per family
    path = B.stage_train(0, "cpu", steps=20, models=models)
    z = np.load(path)
    assert [int(z[f"{nm}_n"]) for nm, _ in B.ARMS] == [2, 2, 2]
    assert z["ind10k_s2"].shape == (9, 69, 16, 16) and np.isfinite(z["ind30k_s2"]).all()
    R = B.reduce([path.parent], tmp_path / "out.npz")
    assert R["M"] == 2 and R["W2"].shape == (9, 69, 2, 2)


def test_queries_are_the_papers_points():
    q = B.queries(B.EmbeddedGaussian())
    assert len(q) == 9 and q[0].shape == (69, 16)
    assert torch.allclose(q[4] - q[3], (B.SIGMAS[4] - B.SIGMAS[3]) * (q[1] - q[0]) / (B.SIGMAS[1] - B.SIGMAS[0]))


@pytest.mark.skipif(not (B.REPO / "results" / "gaussboot" / "main" / "shard-0" / "eval.npz").exists(),
                    reason="the earlier run's stored outputs are not here")
def test_points_match_the_stored_run():
    e = np.load(B.REPO / "results" / "gaussboot" / "main" / "shard-0" / "eval.npz", allow_pickle=True)
    g = B.EmbeddedGaussian()
    assert np.allclose(g.U.numpy(), e["U"], atol=1e-12)
    assert np.allclose(np.stack([q.numpy() for q in B.queries(g)]), e["q"], atol=1e-12)
