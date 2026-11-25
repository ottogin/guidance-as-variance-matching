"""The per-step weight reaches the step it belongs to (no GPU, no weights).

diffusers calls callback_on_step_end AFTER the update of step i and re-reads the scale at the top of every
step, so a hook that sets the weight of the timestep it was handed is off by one: step 0 runs at the
pipeline's argument and w[-1] never applies. These tests replay diffusers' ordering against the real hook.
"""
import numpy as np

from vmg import guidance as G
from vmg import noise


class FakePipe:
    def __init__(self, timesteps, initial):
        self.scheduler = type("S", (), {"timesteps": timesteps})()
        self._guidance_scale = initial


def effective(w, t_levels, visited=None):
    w0, cb, _ = G.diffusers_hook(w, t_levels)
    visited = list(t_levels) if visited is None else list(visited)
    p = FakePipe(visited, w0)
    seen = []
    for i, t in enumerate(visited):
        seen.append(p._guidance_scale)      # read at the top of the step
        cb(p, i, t, {})                     # fired after the scheduler step
    return np.asarray(seen)


def test_descending_float_grid():
    t = np.linspace(1000.0, 25.0, 40)
    w = np.linspace(1.5, 6.5, 40)
    assert np.allclose(effective(w, t), w)


def test_first_and_last_level_are_applied():
    t = np.linspace(1000.0, 25.0, 8)
    w = np.array([1.5, 2.0, 3.0, 20.0, 3.0, 2.0, 1.5, 7.0])
    got = effective(w, t)
    assert got[0] == 1.5 and got[-1] == 7.0 and int(np.argmax(got)) == 3


def test_repeated_timestep_is_keyed_on_t():
    """PNDM revisits a timestep, so step index != level index."""
    t_levels = np.array([961, 941, 921, 901], dtype=float)
    visited = np.array([961, 941, 961, 921, 901], dtype=float)
    got = effective([2.0, 3.0, 4.0, 5.0], t_levels, visited)
    assert list(got) == [2.0, 3.0, 2.0, 4.0, 5.0]


def test_never_exactly_one():
    got = effective(np.ones(12), np.linspace(1000.0, 25.0, 12))
    assert (got > 1.0).all()


def test_live_scale_reads_the_hook():
    p = FakePipe([3.0, 2.0, 1.0], 4.0)
    s = G.LiveScale(p, 4.0)
    assert s * 2.0 == 8.0 and float(s) == 4.0
    p._guidance_scale = 6.0
    assert s * 2.0 == 12.0 and 2.0 * s == 12.0 and s > 1


def test_weights_from_json_interpolates_a_calibrated_subset(tmp_path):
    import json
    levels = dict(sigma=np.array([10.0, 3.0, 1.0, 0.3, 0.1]), t=np.arange(5.0))
    j = dict(levels=[0, 2, 4], sigma=[10.0, 1.0, 0.1], schedule=dict(w=[2.0, 4.0, 6.0]), **{"global": dict(w=3.5)})
    (tmp_path / "w.json").write_text(json.dumps(j))
    assert np.allclose(G.from_json(tmp_path / "w.json", "global", levels), 3.5)
    got = G.from_json(tmp_path / "w.json", "schedule", levels)
    assert np.allclose(got[[0, 2, 4]], [2.0, 4.0, 6.0]) and 2.0 < got[1] < 4.0


def test_noise_is_the_legacy_formula():
    import torch
    for seed, row, lv in ((0, 0, 0), (0, 5704, 39), (3, 17, 999)):
        want = torch.randn((4, 6), generator=torch.Generator().manual_seed(seed * 1_000_003 + row * 1009 + lv))
        assert torch.equal(noise.eps(seed, row, lv, (4, 6)), want)


def _moments(x0, D0, D1, sigma):
    """solve.Moments of synthetic predictions, from the per-coordinate definitions in vmg.solve."""
    from vmg import solve
    N, d = x0.shape
    dx = x0 - x0.mean(0)

    def ip(u, v):
        return float((u * v).sum(1).mean() / d)
    xr, rr, rc = [], [], []
    for d0, d1 in zip(D0, D1):
        r0, r1 = d0 - x0, d1 - x0
        c0, c1 = r0 - r0.mean(0), r1 - r1.mean(0)
        xr.append([ip(dx, r0), ip(dx, r1)])
        rr.append([ip(r0, r0), ip(r0, r1), ip(r1, r1)])
        rc.append([ip(c0, c0), ip(c0, c1), ip(c1, c1)])
    return solve.Moments(sigma=sigma, t=sigma, var_x0=ip(dx, dx), xr=xr, rr=rr, rr_centred=rc, n=N, d=d)


def test_iso_noise_meets_the_target(tmp_path):
    """D1 + eps, eps ~ N(0, std^2 I), has the data's second moment wherever D1 falls short of it, and no noise
    is added where D1 already exceeds it; for both rules."""
    import json
    from vmg import solve
    rng = np.random.default_rng(0)
    N, d = 20000, 6
    x0 = rng.normal(size=(N, d)) * np.arange(1, d + 1) + 0.7
    xb = x0.mean(0)
    shrink = [0.3, 0.6, 0.9, 1.3]                    # the last level overshoots the data's moment
    D1 = [xb + s * (x0 - xb) + 0.2 * rng.normal(size=(N, d)) + 0.1 for s in shrink]
    D0 = [xb + 0.5 * s * (x0 - xb) + 0.4 * rng.normal(size=(N, d)) for s in shrink]
    sigma = np.array([10.0, 3.0, 1.0, 0.1])
    m = _moments(x0, D0, D1, sigma)
    for rule in solve.RULES:
        a, b, c = solve.coefficients(m, rule)
        p = tmp_path / f"w-{rule}.json"
        p.write_text(json.dumps(dict(levels=list(range(4)), sigma=sigma.tolist(),
                                     coefficients=dict(a=a.tolist(), b=b.tolist(), c=c.tolist()))))
        std = G.matched_noise_std(p, dict(sigma=sigma, t=sigma))
        ctr = (lambda D: D.mean(0)) if rule == "centred" else (lambda D: xb)
        target = ((x0 - xb) ** 2).sum(1).mean() / d
        have = np.array([((D - ctr(D)) ** 2).sum(1).mean() / d for D in D1])
        assert np.allclose(std ** 2, np.clip(target - have, 0, None), rtol=1e-9, atol=1e-12), rule
        assert std[-1] == 0 and (std[:-1] > 0).all(), rule
