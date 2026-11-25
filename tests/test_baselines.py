"""The other papers' guidance rules (vmg/baselines.py) and how they reach the SD 3 sampler (no GPU, no weights)."""
from pathlib import Path

import numpy as np
import pytest
import torch

from vmg import baselines as B
from vmg import solve

FIX = Path(__file__).with_name("fixtures") / "sd35m-coco-s40.npz"


def pair(shape=(3, 4, 6, 6), seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(shape, generator=g).to(dtype) for _ in range(2))


def test_mix_is_diffusers_guidance_bitwise():
    vu, vc = pair()
    w = np.linspace(0.3, 9.0, 5)
    for k in range(5):
        assert torch.equal(B.Mix(w)(k, vu, vc), vu + float(w[k]) * (vc - vu))


def test_free2_and_step_ag():
    vu, vc = pair(dtype=torch.float32)
    f = B.Free2([0.7, 2.0], [0.4, -1.5])
    assert torch.allclose(f(1, vu, vc), 2.0 * vc - 1.5 * vu)
    s = B.StepAG(np.full(10, 4.5), B.k_stop(0.5, 10))
    assert torch.equal(s(4, vu, vc), vu + 4.5 * (vc - vu)) and torch.equal(s(5, vu, vc), vc)
    assert s.nfe(10) == 15 and B.Mix(np.ones(10)).nfe(10) == 20
    assert B.k_stop(0.3, 40) == 12 and B.k_stop(0.5, 40) == 20


def test_oec_matches_algorithm_1():
    """Yang et al., Alg. 1 (Dynamic), written out step by step, against the closed form vu - (1 - s) proj_A(B)."""
    pu, pc = pair(seed=1, dtype=torch.float32)
    vu, vc = pair(seed=2, dtype=torch.float32)
    w = 3.0
    oec = B.OEC(np.full(4, w), tau=0.9)
    first = oec(0, pu, pc)
    assert torch.equal(first, pu + w * (pc - pu)), "the first level is plain CFG"
    got = oec(1, vu, vc)
    for i in range(len(vu)):
        e_c, e_u, p_c, p_u = vc[i].flatten(), vu[i].flatten(), pc[i].flatten(), pu[i].flatten()
        hat_c, hat_u = 2 * e_c - p_c, 2 * e_u - p_u
        A, Bv = e_c - hat_c, e_u - hat_u
        s = float(A @ Bv / (A.norm() * Bv.norm()))
        if s < 0.9:
            u = Bv - (A @ Bv) / (A @ A) * A + hat_u
            u = (1 - s) * u + s * e_u
        else:
            u = e_u
        ref = u + w * (e_c - u)
        assert torch.allclose(got[i].flatten(), ref, atol=1e-5)
    assert oec.calls == len(vu) and 0 <= oec.fired <= oec.calls


def test_oec_leaves_parallel_changes_alone():
    pu, pc = pair(seed=3, dtype=torch.float32)
    d = torch.randn(pu.shape)
    oec = B.OEC(np.full(3, 4.5), tau=0.9)
    oec(0, pu, pc)
    vu, vc = pu - 0.1 * d, pc - 0.3 * d          # A and B parallel: cos = 1, no correction
    assert torch.allclose(oec(1, vu, vc), vu + 4.5 * (vc - vu), atol=1e-6) and oec.fired == 0
    oec.reset()
    assert oec.prev is None


def test_guide_hook_output_survives_the_pipelines_mix():
    """The hook writes the guide's output into both halves of the CFG batch; the pipeline's own
    v_u' + g (v_c' - v_u') must then return it bitwise, for any g."""
    from vmg.models.sd import SD35Medium
    m = SD35Medium()
    m.ts = np.linspace(1000, 25, 40).astype(np.float32)
    seen = []
    w = np.linspace(0.2, 12, 40)
    hook = m.guide_hook(B.Mix(w), log=seen)
    vu, vc = pair()
    out = hook(None, (), dict(timestep=torch.full((6,), float(m.ts[7]))), (torch.cat([vu, vc]),))
    u2, c2 = out[0].chunk(2)
    want = vu + float(w[7]) * (vc - vu)
    for g in (1.000001, 2.0, 4.5, 37.0):
        assert torch.equal(u2 + g * (c2 - u2), want)
    assert seen == [7]


def test_fp32_state_wrapper_installs_and_restores():
    from vmg.models.sd import Fp32State

    class Net(torch.nn.Module):
        dtype = torch.bfloat16

        def forward(self, hidden_states=None):
            self.got = hidden_states.dtype
            return (hidden_states.to(torch.bfloat16),)

    class Sched:
        def step(self, model_output, timestep, sample):
            return (sample + model_output).to(model_output.dtype)

    class Pipe:
        def __init__(self):
            self.transformer, self.scheduler = Net(), Sched()

        def prepare_latents(self, x):
            return x.to(torch.bfloat16)

    p = Pipe()
    with Fp32State(p) as f:
        z = p.prepare_latents(torch.randn(2, 3))
        assert z.dtype == torch.float32
        v = p.transformer(hidden_states=z)[0]
        assert p.transformer.got == torch.bfloat16
        assert p.scheduler.step(v, 0, z).dtype == torch.float32
        assert f.seen == {"torch.float32"}
    assert p.prepare_latents(torch.randn(2)).dtype == torch.bfloat16
    assert p.scheduler.step(torch.zeros(2, dtype=torch.bfloat16), 0, torch.zeros(2)).dtype == torch.bfloat16
    assert not p.transformer._forward_pre_hooks


def test_simca_solve_recovers_known_weights():
    g = np.random.default_rng(0)
    n, d = 64, 50
    vu, vc = g.normal(size=(n, d)), g.normal(size=(n, d))

    def P(psi):
        ip = lambda a, b: (a * b).sum(1)                                     # noqa: E731
        return np.stack([ip(vu, vu), ip(vu, vc), ip(vc, vc), ip(vu, psi), ip(vc, psi)], 1)
    assert np.allclose(B.simca_solve(P(1.7 * vc - 0.4 * vu), 2), [1.7, -0.4])
    assert np.allclose(B.simca_solve(P(vu + 3.2 * (vc - vu)), 1), [3.2])


@pytest.fixture(scope="module")
def sd35m():
    fx = np.load(FIX)
    return solve.Moments(sigma=fx["sigma"], t=fx["t"], var_x0=float(fx["var_x0"]), xr=fx["xr"], rr=fx["rr"],
                         rr_centred=fx["rr_centred"], n=int(fx["n"]), d=int(fx["d"]))


def test_refit_pruned(sd35m):
    a, b, c = solve.coefficients(sd35m, "centred")
    L = len(a)
    w_all, ok = B.refit_pruned(a, b, c, np.ones(L, bool))
    assert ok and w_all == solve.fit_global(sd35m, "centred")[0]
    keep = np.arange(L) < 20
    w, ok = B.refit_pruned(a, b, c, keep)
    ws = np.where(keep, w, 1.0)
    assert ok and abs((b * ws * ws - 2 * a * ws - c).mean()) < 1e-12, "the level-averaged match holds again"


def test_simca_simulation_free_is_w_mse():
    """In the CFG form, SIMCA's least squares in velocity space against psi = (x_s - x0) / s is
    E||D_w - x0||^2 / s^2: the same minimiser as w_MSE = E<x0 - D_u, D_c - D_u> / E||D_c - D_u||^2."""
    g = np.random.default_rng(1)
    n, d, s = 200, 30, 0.37
    x0, eps = g.normal(size=(n, d)), g.normal(size=(n, d))
    x = (1 - s) * x0 + s * eps
    Du, Dc = x0 + 0.5 * g.normal(size=(n, d)), x0 + 0.4 * g.normal(size=(n, d))
    vu, vc, psi = (x - Du) / s, (x - Dc) / s, (x - x0) / s
    ip = lambda a, b: (a * b).sum(1)                                        # noqa: E731
    P = np.stack([ip(vu, vu), ip(vu, vc), ip(vc, vc), ip(vu, psi), ip(vc, psi)], 1)
    w_mse = ip(x0 - Du, Dc - Du).sum() / ip(Dc - Du, Dc - Du).sum()
    assert np.allclose(B.simca_solve(P, 1), [w_mse])
