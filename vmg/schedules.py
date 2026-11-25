"""Published guidance schedules on a model's own sampler levels (paper tab:pub-sched).

Each takes one base weight w (in the convention D0 + w (D1 - D0)) and returns a weight per sampler level, on a
normalised time u = t / max(t) in [0, 1] (1 = noise) and the EDM sigma of each level (levels() of an adapter):

  interval  Kynkaanniemi et al. 2024: guidance only for EDM sigma in (0.28, 5.42] (their SD-XL interval),
            w = 1 outside. The value inside keeps the total sum_i w_i (u_i - u_{i+1}) at w, as Jin et al.
            2026 ran it on SD 3.5.
  linear    Wang et al. 2024: omega(t) = 2 (1 - t/T) omega in their eps_c + omega (eps_c - eps_u) convention,
            i.e. w = 1 + 2 (w - 1)(1 - t/T), T = 1000 training timesteps.
  tvcfg     Jin et al. 2026 (TV-CFG): linspace 1 -> 2w - 1 -> 1 over the steps, rescaled so the same total is w
            (their guidance_schedulers.symmetric, github.com/sqt24/tvcfg).
"""
import numpy as np

KINDS = ("interval", "linear", "tvcfg")
INTERVAL = (0.28, 5.42)


def widths(u):
    """Width of each step in u, the last one ending at 0 (TV-CFG's delta_timesteps)."""
    return u - np.append(u[1:], 0.0)


def interval(u, sigma, w, lo=INTERVAL[0], hi=INTERVAL[1]):
    inside = (sigma > lo) & (sigma <= hi)
    d = widths(u)
    return np.where(inside, (w - d[~inside].sum()) / d[inside].sum(), 1.0)


def linear(t_over_T, w):
    return 1 + 2 * (w - 1) * (1 - t_over_T)


def tvcfg(u, w):
    n, half = len(u), len(u) // 2
    if n % 2 == 0:
        g = np.concatenate([np.linspace(1, 2 * w - 1, half, endpoint=False), np.linspace(2 * w - 1, 1, half)])
    else:
        g = np.concatenate([np.linspace(1, 2 * w - 1, half + 1), np.linspace(2 * w - 1, 1, half + 1)[1:]])
    return g * w / np.sum(g * widths(u))


def published(kind, levels, w, T=1000):
    """The weights of schedule `kind` at every level of levels() = dict(sigma, t), with base weight w."""
    t = np.asarray(levels["t"], dtype=np.float64)
    sigma = np.asarray(levels["sigma"], dtype=np.float64)
    u = t / t.max()
    assert u[0] > u[-1], "levels must run from noise to data"
    if kind == "interval":
        return interval(u, sigma, w)
    if kind == "linear":
        return linear(t / T, w)
    if kind == "tvcfg":
        return tvcfg(u, w)
    raise ValueError(f"unknown schedule {kind!r}; one of {KINDS}")
