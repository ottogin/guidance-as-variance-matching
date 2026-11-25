"""The posterior-DDPM sampler (vmg/ddpm.py): its reverse step is exact, and its clean draws have the stated spread."""
import numpy as np
import torch

from vmg import ddpm as DD
from vmg.ddpm import PosteriorDDPM


def test_flow_posterior_preserves_the_marginal():
    """A x_s + B x0 + C z with x_s = (1 - s) x0 + s eps is (1 - s') x0 + s' eps' for every s' < s."""
    for s, sn in [(1.0, 0.97), (0.9, 0.5), (0.5, 0.49), (0.2, 0.01), (0.3, 0.0)]:
        A, B, C = DD.flow_posterior(s, sn)
        assert np.isclose(A * (1 - s) + B, 1 - sn), (s, sn)
        assert np.isclose((A * s) ** 2 + C ** 2, sn ** 2), (s, sn)
    assert DD.flow_posterior(1.0, 0.6)[0] == 0.0, "the first step forgets the pure-noise input"


def test_flow_posterior_is_the_conditional():
    """Monte Carlo on one coordinate: the step's mean and variance given (x_s, x0) are the Gaussian conditional
    of x_s' given x_s and x0 under the forward process (x_s built from x_s' by adding noise)."""
    g = torch.Generator().manual_seed(0)
    n, s, sn = 400_000, 0.8, 0.45
    x0 = torch.randn(n, generator=g) * 0.7 + 0.3
    xsn = (1 - sn) * x0 + sn * torch.randn(n, generator=g)
    a = (1 - s) / (1 - sn)                                   # x_s = a x_s' + sqrt(s^2 - a^2 s'^2) noise
    xs = a * xsn + np.sqrt(s ** 2 - a ** 2 * sn ** 2) * torch.randn(n, generator=g)
    A, B, C = DD.flow_posterior(s, sn)
    res = xsn - (A * xs + B * x0)                           # the step's residual must be N(0, C^2), independent
    assert abs(float(res.mean())) < 3e-3
    assert np.isclose(float(res.var()), C ** 2, rtol=1e-2)
    assert abs(float(torch.corrcoef(torch.stack([res, xs]))[0, 1])) < 5e-3


def test_exact_posterior_draws_give_exact_marginals():
    """Gaussian data: with D1 the posterior mean and an iso spread equal to the posterior std, the sampler draws
    x~0 from the exact posterior, so every intermediate marginal is exactly (1 - s) x0 + s eps (M(infinity))."""
    g = torch.Generator().manual_seed(1)
    n, d, mu, v0 = 100_000, 4, 0.5, 0.6 ** 2
    grid = [1.0, 0.9, 0.6, 0.35, 0.15, 0.05]
    kappa = 2.0
    smp = PosteriorDDPM("iso", kappa)
    x = torch.randn(n, d, generator=g)
    for s, sn in zip(grid[:-1], grid[1:]):
        a = 1 - s                                       # x_s = a x0 + s eps: posterior of x0 per coordinate
        pv = 1 / (1 / v0 + a * a / (s * s))
        pm = pv * (mu / v0 + a * x / (s * s))
        d0 = pm - np.sqrt(pv) / kappa                   # rms(D1 - D0) = posterior std / kappa
        x = smp.step(x, s, sn, d0, pm, torch.randn(n, d, generator=g), torch.randn(n, generator=g),
                     torch.randn(n, d, generator=g))
        m_want, v_want = (1 - sn) * mu, (1 - sn) ** 2 * v0 + sn ** 2
        assert abs(float(x.mean()) - m_want) < 0.01, (sn, float(x.mean()), m_want)
        assert np.isclose(float(x.var()), v_want, rtol=0.02), (sn, float(x.var()), v_want)


def test_shapes_add_the_same_energy():
    """E||x~0 - D1||^2 = kappa^2 ||D1 - D0||^2 for iso, diag and rank1; none is D1 itself; dw centres on D_w."""
    g = torch.Generator().manual_seed(2)
    B = 4096
    d1, d0 = torch.randn(B, 4, 4, 4, generator=g), torch.randn(B, 4, 4, 4, generator=g) * 0.3
    d0 = d0 * torch.rand(B, 1, 1, 1, generator=g)                    # a spread that varies by sample
    xi, gg = torch.randn(B, 4, 4, 4, generator=g), torch.randn(B, generator=g)
    want = 3.5 ** 2 * float((d1 - d0).pow(2).sum())
    for sh in ("iso", "diag", "rank1"):
        got = float((PosteriorDDPM(sh, 3.5).draw(d0, d1, xi, gg) - d1).pow(2).sum())
        assert abs(got / want - 1) < 0.06, (sh, got / want)
    iso = PosteriorDDPM("iso", 3.5).draw(d0, d1, xi, gg) - d1        # per sample, exactly kappa rms(D1 - D0) xi
    assert torch.allclose(iso, 3.5 * (d1 - d0).flatten(1).pow(2).mean(1).sqrt().view(-1, 1, 1, 1) * xi, atol=1e-5)
    r1 = PosteriorDDPM("rank1", 3.5).draw(d0, d1, xi, gg) - d1       # along D1 - D0 only
    assert torch.allclose(r1, 3.5 * gg.view(-1, 1, 1, 1) * (d1 - d0), atol=1e-5)
    assert torch.equal(PosteriorDDPM("none", 0.0).draw(d0, d1, xi, gg), d1)
    assert torch.allclose(PosteriorDDPM("none", 0.0, "dw", 4.5).draw(d0, d1, xi, gg), d0 + 4.5 * (d1 - d0))


def test_last_step_returns_the_mean():
    g = torch.Generator().manual_seed(3)
    x, d1, d0, xi, z = (torch.randn(2, 3, generator=g) for _ in range(5))
    gg = torch.randn(2, generator=g)
    assert torch.equal(PosteriorDDPM("iso", 3.5).step(x, 0.07, 0.0, d0, d1, xi, gg, z), d1)
    assert torch.allclose(PosteriorDDPM("none", 0.0, "dw", 4.5).step(x, 0.07, 0.0, d0, d1, xi, gg, z),
                          d0 + 4.5 * (d1 - d0))


def test_names():
    assert PosteriorDDPM("iso", 3.5).name() == "ddpm-iso-k3.5"
    assert PosteriorDDPM("none", 0.0).name() == "ddpm"
    assert PosteriorDDPM("none", 0.0, "dw", 4.5).name() == "ddpm-dw4.5"
    assert PosteriorDDPM("rank1", 4.5).name() == "ddpm-rank1-k4.5"
