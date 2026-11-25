"""Variance-matched guidance weights (paper Sec. 4) and the MSE-optimal weight (paper eq. wmse).

Notation follows the paper. D1 is the primary prediction (conditional / strong model), D0 the reference
(unconditional / weak model), r_i = D_i - x0, and xbar0 the data mean. Every statistic is an expectation over
forward-noised calibration examples, summed over coordinates and divided by the number of coordinates d, so
values are per coordinate (the weights do not depend on that scale).

At each noise level the paper's matching condition is the trace of the boxed covariance equation,
tr Cov_{x_t}(D_w) = tr Cov(x0), which with D_w = x0 + r_w, r_w = r0 + w (r1 - r0), reads

    b w^2 - 2 a w - c = 0,    a = -tr Cov(D0, D1 - D0),  b = tr Cov(D1 - D0),  c = tr Cov(x0) - tr Cov(D0),

and in terms of the centred residual moments c_ij = tr Cov(r_i, r_j), mx_i = tr Cov(x0, r_i) = E<x0 - xbar0, r_i>:

    b = c11 - 2 c01 + c00,    a = -[(mx1 - mx0) + (c01 - c00)],    c = -(2 mx0 + c00).

The schedule solves it per level (eq. wstar); the constant weight solves it once with a, b, c averaged over
the calibrated levels (eq. wconstant). At each level we take the larger real root in [1, 40]; a level with no
admissible root holds the weight of the nearest level that has one (paper App. "Fitting"). This is the default
rule, "centred", and the one behind the paper's printed weights (SD 3.5 Medium 4.14, ...).

rule="datamean" is the second-moment variant E||D_w - xbar0||^2 = E||x0 - xbar0||^2: the same equation with
the raw residual products m_ij = E<r_i, r_j> in place of the centred ones. The two differ by the squared gap
between the mean guided prediction and the data mean, ||E_{x_t} D_w - xbar0||^2, which the covariance form
does not count (3.99 against 4.14 on SD 3.5 Medium).
"""
from dataclasses import dataclass

import numpy as np

CLIP = (1.0, 40.0)
RULES = ("centred", "datamean")
DEFAULT = "centred"


@dataclass
class Moments:
    """Per-level second moments of one (D1, D0) pair on calibration data, per coordinate."""

    sigma: np.ndarray        # [S] EDM-equivalent noise level of each calibrated level (sampler order)
    t: np.ndarray            # [S] the sampler's own timestep values at those levels
    var_x0: float            # E||x0 - xbar0||^2 / d
    xr: np.ndarray           # [S, 2] E<x0 - xbar0, r_i> / d, columns (D0, D1)
    rr: np.ndarray           # [S, 3] E<r_i, r_j> / d, raw, columns (00, 01, 11)
    rr_centred: np.ndarray   # [S, 3] the same with each r_i centred on its mean
    n: int                   # calibration samples
    d: int                   # coordinates per sample

    def __post_init__(self):
        for k in ("sigma", "t", "xr", "rr", "rr_centred"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.float64))
        S = len(self.sigma)
        assert self.t.shape == (S,) and self.xr.shape == (S, 2), (self.t.shape, self.xr.shape)
        assert self.rr.shape == (S, 3) and self.rr_centred.shape == (S, 3)


def coefficients(m, rule=DEFAULT):
    """(a, b, c), each [S], of b w^2 - 2 a w - c = 0 at every level."""
    assert rule in RULES, rule
    m00, m01, m11 = (m.rr_centred if rule == "centred" else m.rr).T
    mx0, mx1 = m.xr.T
    b = m11 - 2 * m01 + m00
    a = -((mx1 - mx0) + (m01 - m00))
    c = -(2 * mx0 + m00)
    return a, b, c


def roots(a, b, c):
    """Both roots (a +- sqrt(a^2 + b c)) / b, larger first; NaN where b <= 0 or the discriminant is negative."""
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
    disc = a * a + b * c
    ok = (b > 0) & (disc >= 0)
    s = np.sqrt(np.where(ok, disc, 0.0))
    bb = np.where(ok, b, 1.0)
    return np.where(ok, (a + s) / bb, np.nan), np.where(ok, (a - s) / bb, np.nan)


def root(a, b, c, clip=CLIP):
    """(w, admissible): the larger real root inside `clip`.

    Without an admissible root, w is the larger root clipped to the interval (or NaN if there is no real
    root) and admissible is False. Computed as in the code that produced the paper's weights
    (ta w^2 + tb w + tc with ta = b, tb = -2a, tc = -c), so it reproduces them bit for bit.
    """
    lo, hi = clip
    ta, tb, tc = float(b), -2.0 * float(a), -float(c)
    disc = tb * tb - 4 * ta * tc
    if ta == 0 or not np.isfinite(disc) or disc < 0:
        return np.nan, False
    r = np.array([(-tb + np.sqrt(disc)) / (2 * ta), (-tb - np.sqrt(disc)) / (2 * ta)])
    inside = r[(r >= lo) & (r <= hi)]
    if len(inside):
        return float(inside.max()), True
    return float(np.clip(r.max(), lo, hi)), False


def hold_nearest(w, ok):
    """Levels without an admissible root take the weight of the nearest level with one (ties: the noisier,
    i.e. lower, index)."""
    w = np.asarray(w, dtype=np.float64).copy()
    ok = np.asarray(ok, dtype=bool)
    assert ok.any(), "no level has an admissible root"
    idx = np.arange(len(w))
    good = idx[ok]
    nearest = np.array([good[np.argmin(np.abs(good - i))] for i in idx])
    w[~ok] = w[nearest[~ok]]
    return w


def fit_schedule(m, rule=DEFAULT, clip=CLIP):
    """The per-level weight w(t) (eq. wstar). Returns dict(w, raw, admissible, a, b, c)."""
    a, b, c = coefficients(m, rule)
    got = [root(a[i], b[i], c[i], clip) for i in range(len(a))]
    raw = np.array([g[0] for g in got])
    ok = np.array([g[1] for g in got])
    # with no admissible level at all (e.g. a 2-row smoke run) there is nothing to hold: report the raw roots
    w = hold_nearest(np.where(np.isfinite(raw), raw, 0.0), ok) if ok.any() else raw.copy()
    return dict(w=w, raw=raw, admissible=ok, a=a, b=b, c=c)


def fit_global(m, rule=DEFAULT, clip=CLIP, levels=None):
    """The constant weight (eq. wconstant): one root of the matching equation averaged over levels.

    `levels` selects which calibrated levels enter the average (default: all, weighted equally).
    Returns (w, admissible).
    """
    a, b, c = coefficients(m, rule)
    sel = slice(None) if levels is None else np.asarray(levels)
    return root(a[sel].mean(), b[sel].mean(), c[sel].mean(), clip)


def w_mse(m):
    """The MSE-optimal weight per level (eq. wmse): E<x0 - D0, D1 - D0> / E||D1 - D0||^2."""
    m00, m01, m11 = m.rr.T
    return (m00 - m01) / (m11 - 2 * m01 + m00)


def mse(m, w):
    """E||D_w - x0||^2 / d per level for a weight `w` (scalar or [S])."""
    m00, m01, m11 = m.rr.T
    w = np.asarray(w, dtype=np.float64)
    return m00 + 2 * w * (m01 - m00) + w * w * (m11 - 2 * m01 + m00)


def interp_log_sigma(sigma_from, w_from, sigma_to):
    """Weights at the noise levels `sigma_to`, linear in log sigma between calibrated levels, held at the ends.

    Used when a model is calibrated on a subset of its sampler's levels (MDM: 50 of 1,000; Latte: 51 of 250),
    as the code behind the paper did. Levels with infinite sigma take the weight of the noisiest finite one."""
    s_from, w_from, s_to = (np.asarray(v, dtype=np.float64) for v in (sigma_from, w_from, sigma_to))
    fin = np.isfinite(s_from)
    assert (s_from[fin] > 0).all() and (s_to[np.isfinite(s_to)] > 0).all(), "needs sigma > 0"
    o = np.argsort(s_from[fin])
    x = np.log(np.where(np.isfinite(s_to), s_to, s_from[fin].max()))
    return np.interp(x, np.log(s_from[fin][o]), w_from[fin][o])


# --- what to match and how to pool: the general fits (compute_w.py --match / --mode / --pool) ----------------
#
# Every matching statistic is a residual polynomial per level, R_t(w) = E0 + w E1 + w^2 E2, the guided prediction's
# statistic minus the data's (around its own mean for the default rule, around the data mean for "datamean"):
#
#   trace  a scalar: tr Cov(D_w) - tr Cov(x0) per coordinate, = b w^2 - 2 a w - c (E = [-c, -2a, b])
#   cov    a C x C matrix: the channel covariance of D_w minus that of x0, summed over positions (App. D.1)
#   diag   a d-vector: each coordinate's variance of D_w minus that of x0
#
# E has shape [S, 3, ...]. A schedule fits one w per level, a global weight one w for all levels, pooled as in
# App. D.2: "mean" makes the level-averaged residual vanish (the paper's eq. wconstant), "l1" minimises the mean
# norm of the residual over levels and "l2" its mean square. For the trace, the schedule and the mean pool keep the
# paper's closed-form root (fit_schedule / fit_global above); every other fit is an argmin over DOMAIN, where a
# matrix or vector residual is measured in the Frobenius / Euclidean norm.

MATCHES = ("trace", "cov", "diag")
MODES = ("schedule", "global")
POOLS = ("mean", "l1", "l2")
DOMAIN = (0.0, 40.0)        # argmin fits: App. D.2 takes L1 and L2 over w > 0, App. D.1 the whole matrix on [0, 40]


def residual_trace(m, rule=DEFAULT):
    """E [S, 3] of the trace match from solve.Moments: r_t(w) = E0 + w E1 + w^2 E2 = b w^2 - 2 a w - c."""
    a, b, c = coefficients(m, rule)
    return np.stack([-c, -2.0 * a, b], 1)


def quartic(E):
    """[S, 5] ascending coefficients in w of ||E0 + w E1 + w^2 E2||^2 at each level (E [S, 3, ...])."""
    E = np.asarray(E, dtype=np.float64)
    F = E.reshape(E.shape[0], 3, -1)
    K = np.einsum("skd,smd->skm", F, F)
    J = np.zeros((E.shape[0], 5))
    for k in range(3):
        for j in range(3):
            J[:, k + j] += K[:, k, j]
    return J


def argmin_poly(J, lo=DOMAIN[0], hi=DOMAIN[1]):
    """The minimiser on [lo, hi] of the polynomial with ascending coefficients J: a stationary point or an end."""
    J = np.asarray(J, dtype=np.float64)
    dJ = np.arange(1, len(J)) * J[1:]
    r = np.roots(dJ[::-1]) if np.any(dJ) else np.array([])
    real = np.array([x.real for x in np.atleast_1d(r) if abs(x.imag) <= 1e-8 * max(abs(x), 1.0)])
    cand = np.concatenate([real[(real >= lo) & (real <= hi)], [lo, hi]])
    return float(cand[np.argmin(np.polyval(J[::-1], cand))])


def _norm_objective(E):
    """w -> E_t ||R_t(w)|| (the l1 pool), vectorised over an array of w."""
    J = quartic(E)

    def f(w):
        w = np.atleast_1d(np.asarray(w, dtype=np.float64))
        sq = np.stack([np.polyval(j[::-1], w) for j in J])          # [S, len(w)]
        return np.sqrt(np.clip(sq, 0.0, None)).mean(0)
    return f


def _grid_argmin(f, lo, hi, n=8001):
    """Minimise a scalar function on [lo, hi]: a grid, then bounded refinement between the best point's neighbours."""
    from scipy.optimize import minimize_scalar
    g = np.linspace(lo, hi, n)
    v = f(g)
    i = int(np.argmin(v))
    a, b = g[max(i - 1, 0)], g[min(i + 1, n - 1)]
    if b <= a:
        return float(g[i])
    r = minimize_scalar(lambda w: float(f(w)[0]), bounds=(a, b), method="bounded", options=dict(xatol=1e-10))
    return float(r.x) if r.fun <= v[i] else float(g[i])


def fit_residual(E, mode="schedule", pool="mean", domain=DOMAIN):
    """Argmin fits of a residual E [S, 3, ...]: a schedule [S] (argmin ||R_t||^2 per level) or one weight."""
    assert mode in MODES and pool in POOLS, (mode, pool)
    lo, hi = domain
    J = quartic(E)
    if mode == "schedule":
        return np.array([argmin_poly(j, lo, hi) for j in J])
    if pool == "mean":
        return argmin_poly(quartic(np.asarray(E).mean(0, keepdims=True))[0], lo, hi)
    if pool == "l2":
        return argmin_poly(J.mean(0), lo, hi)
    return _grid_argmin(_norm_objective(E), lo, hi)


def fit(match, mode, pool="mean", m=None, E=None, rule=DEFAULT, clip=CLIP, domain=DOMAIN):
    """One fit: dict(w, admissible). The trace takes solve.Moments `m`; cov and diag their residual `E`.

    For the trace, the schedule and the mean pool are the paper's admissible roots (eq. wstar, eq. wconstant);
    everything else is fit_residual. `admissible` is False only where the paper's root does not exist."""
    assert match in MATCHES and mode in MODES and pool in POOLS, (match, mode, pool)
    if match == "trace":
        if mode == "schedule":
            s = fit_schedule(m, rule, clip)
            return dict(w=s["w"], admissible=s["admissible"])
        if pool == "mean":
            w, ok = fit_global(m, rule, clip)
            return dict(w=w, admissible=ok)
        E = residual_trace(m, rule)
    assert E is not None, f"match {match!r} needs its residual E"
    w = fit_residual(E, mode, pool, domain)
    return dict(w=w, admissible=np.ones(len(w), bool) if mode == "schedule" else True)


def fit_all(m=None, residuals=None, rule=DEFAULT, clip=CLIP, domain=DOMAIN):
    """Every fit the inputs allow: {match: {"schedule": fit, "global": {pool: fit}}}. `residuals` maps "cov" /
    "diag" to their E; the trace comes from `m`."""
    have = {"trace": None} if m is not None else {}
    have.update(residuals or {})
    out = {}
    for match, E in have.items():
        out[match] = {"schedule": fit(match, "schedule", m=m, E=E, rule=rule, clip=clip, domain=domain),
                      "global": {p: fit(match, "global", p, m=m, E=E, rule=rule, clip=clip, domain=domain)
                                 for p in POOLS}}
    return out
