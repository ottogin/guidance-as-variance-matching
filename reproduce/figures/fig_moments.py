#!/usr/bin/env python
"""fig:moments (Sec. 3.2) and fig:moments-density (App.): sampling with the first k moments of the posterior,
on a 2-D mixture with exact moments.

    python reproduce/figures/fig_moments.py --show 1,2,inf --out OUT/moment-samplers-m12inf           # fig:moments
    python reproduce/figures/fig_moments.py --show 1,2 --density 1,2 --out OUT/moment-samplers-m12    # fig:moments-density
    python reproduce/figures/fig_moments.py --compute [--procs 10]      # (re)build the cache; ~40 CPU-min at 10 procs
    python reproduce/figures/fig_moments.py --compute --ks 3,4,5 [--until-floor]   # optional extra samplers
    python reproduce/figures/fig_moments.py --test                      # exactness of q_4, q_5

Plotting builds the cache first if it is missing (M(1), M(2), M(3), M(inf): what the paper's figures need).

Data: K = 10 isotropic Gaussians (std s = 0.15), means evenly spaced in angle on an ellipse with semi-axes 2 and 1.
Noising x_t = (1 - t) x0 + t eps on the SD3 grid (shift 3, from t = 1). p(x0 | x_t) is a Gaussian mixture, so every
moment is exact. A step of M(k) replaces the posterior mean in the Euler step by a draw from q_k, a distribution with
the posterior's first k cumulants:  x_t' = (1 - a) x_t + a xh,  a = (t - t') / t,  xh ~ q_k(. | x_t).
    q_1 = the posterior mean (Euler);  q_2 = N(mean, Cov);  q_inf = the posterior itself (exact at any step size);
    q_3 = Gaussian + independent two-point variables along three directions 120 deg apart (whitened coordinates,
          aligned with the third harmonic of the third cumulant) that carry the third cumulant exactly; each
          two-point variance nu = |lambda| / sqrt 2 gives it zero excess kurtosis.
    q_4, q_5 = rho * N(0, tau2 I), the form of the posterior itself, with rho on the conic through the component means
          (forced once degree-4 moments match) carrying their trigonometric moments 1..k: the Szego quadrature with a
          uniformly random free parameter, i.e. exact draws from the Burg maximum-entropy density on the conic.
Panel (a): MMD to the true mixture (Gaussian kernel width 0.3, exact on the mixture side via its characteristic
function on a fixed 240-frequency quadrature, debiased) on 10^6 samples (5 chunks of 2e5, chunk-seeded initial noise
shared by every run), N = 1 .. 1024; values within twice the null standard deviation (8 fresh 10^6 data draws) sit
on the floor. Panels (b-e): sample-minus-true density at N = 32 (bins 0.05, exact bin masses).
Cache: $VMG_RUNS/_toys/moments/{conv.json, n32.npz, laststep.json}. Prints the numbers quoted in the paper.
"""
import argparse
import json
import math
import os

import numpy as np

import sys  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from vmg import runs  # noqa: E402

CACHE = str(runs.root() / "_toys" / "moments")
NS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024]
KS = ["1", "2", "3", "inf"]   # what a fresh --compute builds: the paper's panels, plus M(3), whose histogram
                              # sets the colour range of the density figure
KS_DENS = ["1", "2", "3", "inf"]                 # the density figure keeps its 2x2 grid
KIDX = {"1": 0, "2": 1, "3": 2, "inf": 4, "4": 5, "5": 6}
CH, NCH, ELL, NSHOW = 200000, 5, 0.3, 32
XE, YE = np.linspace(-2.7, 2.7, 109), np.linspace(-1.7, 1.7, 69)
SQ2 = math.sqrt(2.0)


# ---------------------------------------------------------------- model and samplers

class Mix:
    def __init__(self, K=10, ax=2.0, ay=1.0, s=0.15):
        th = 2 * np.pi * np.arange(K) / K
        self.mu = np.stack([ax * np.cos(th), ay * np.sin(th)], 1)
        self.K, self.s, self.s2 = K, s, s * s

    def sample(self, n, rng):
        c = rng.integers(0, self.K, n)
        return self.mu[c] + self.s * rng.standard_normal((n, 2)), c

    def post(self, x, t):
        """p(x0 | x_t): weights g (n,K), means m (n,K,2), shared variance tau2."""
        n = len(x)
        if t >= 1:
            return np.full((n, self.K), 1.0 / self.K), np.broadcast_to(self.mu, (n, self.K, 2)), self.s2
        sig = t / (1 - t)
        y = x / (1 - t)
        v = self.s2 + sig * sig
        d = y[:, None, :] - self.mu[None]
        lg = -0.5 * (d * d).sum(-1) / v
        g = np.exp(lg - lg.max(1, keepdims=True))
        g /= g.sum(1, keepdims=True)
        return g, self.mu[None] + (self.s2 / v) * d, self.s2 * sig * sig / v

    def logp(self, x):
        q = -0.5 * ((x[:, None, :] - self.mu[None]) ** 2).sum(-1) / self.s2
        mx = q.max(1)
        return mx + np.log(np.exp(q - mx[:, None]).sum(1)) - np.log(self.K * 2 * np.pi * self.s2)


def moments(g, m, tau2):
    mean = np.einsum("nk,nki->ni", g, m)
    d = m - mean[:, None, :]
    dx, dy = d[..., 0], d[..., 1]
    return mean, ((g * dx * dx).sum(1) + tau2, (g * dx * dy).sum(1), (g * dy * dy).sum(1) + tau2), d


def sqrt2(a, b, c):
    s = np.sqrt(np.maximum(a * c - b * b, 0.0))
    t = np.sqrt(np.maximum(a + c + 2 * s, 1e-300))
    return (a + s) / t, b / t, (c + s) / t


def inv2(a, b, c):
    det = a * c - b * b
    return c / det, -b / det, a / det


def mv(Mt, v):
    a, b, c = Mt
    return np.stack([a * v[..., 0] + b * v[..., 1], b * v[..., 0] + c * v[..., 1]], -1)


def third_dirs(g, dw):
    z = dw[..., 0] + 1j * dw[..., 1]
    H1 = 0.75 * (g * (z.real ** 2 + z.imag ** 2) * z).sum(1)
    H3 = 0.25 * (g * z ** 3).sum(1)
    a3 = np.abs(H3)
    phi1 = np.where(a3 > 1e-14, np.angle(H3) / 3, np.angle(H1))
    P = (4 / 3) * H1 * np.exp(-1j * phi1)
    om = np.exp(-2j * np.pi / 3)
    lam = np.stack([(4 * a3 + 2 * np.real(P * om ** r)) / 3 for r in range(3)], 1)
    ph = phi1[:, None] + 2 * np.pi / 3 * np.arange(3)[None]
    return lam, np.cos(ph), np.sin(ph)


def _verblunsky(g, zt, k):
    """Szego recursion on the circle measure sum_j g_j delta_{zt_j}: Verblunsky coefficients alpha_0 .. alpha_{k-1}.
    Evaluated on the atoms (Stieltjes-style), so a measure with <= k atoms just returns |alpha| = 1 where it ends."""
    n = len(g)
    P = np.ones((n, len(zt)), complex)
    Ps = P.copy()
    al = np.zeros((n, k), complex)
    for j in range(k):
        nr = np.maximum((g * (P.real ** 2 + P.imag ** 2)).sum(1), 1e-300)
        ab = (g * zt * P).sum(1) / nr                                   # conj(alpha_j)
        r = np.abs(ab)
        ab = np.where(r > 1, ab / np.maximum(r, 1e-300), ab)
        a = np.conj(ab)
        al[:, j] = a
        P, Ps = zt * P - ab[:, None] * Ps, Ps - a[:, None] * zt * P
    return al


def _ggt(al):
    """Unitary upper-Hessenberg (GGT) matrix of the Verblunsky coefficients al (n, m), with |al[:, -1]| = 1."""
    n, m = al.shape
    rho = np.sqrt(np.maximum(1 - (al.real ** 2 + al.imag ** 2), 0.0))
    am = np.concatenate([-np.ones((n, 1)), al[:, :-1]], 1)             # alpha_{i-1}, alpha_{-1} = -1
    G = np.zeros((n, m, m), complex)
    for l in range(m):
        prod = np.ones(n)
        for i in range(l, -1, -1):                                     # prod = rho_i ... rho_{l-1}
            G[:, i, l] = -np.conj(al[:, l]) * am[:, i] * prod
            if i > 0:
                prod = prod * rho[:, i - 1]
        if l + 1 < m:
            G[:, l + 1, l] = rho[:, l]
    return G


def conic_nodes(k, g, m, rng):
    """The mixing part of q_k for k >= 4: a (k+1)-point measure on the conic through the component means.

    The posterior is mu * N(0, tau2 I) with mu = sum_j g_j delta_{m_j}, and every m_j lies on the conic
    m_j = mbar + U (cos th_j, sin th_j), th_j = 2 pi j / K. Matching mu's moments up to degree k >= 4 forces the
    measure onto that conic (E[Q^2] = 0 for its quadratic Q), where they are the trigonometric moments c_1 .. c_k.
    The Szego quadrature with a uniformly random free parameter alpha_k = tau, |tau| = 1, matches them exactly;
    averaged over tau it is the Bernstein-Szego (Burg maximum-entropy) density with those moments.
    Returns nodes (n, k+1, 2) and weights (n, k+1)."""
    n, K = g.shape
    th = 2 * np.pi * np.arange(K) / K
    mbar = m.mean(1)
    U = (2.0 / K) * np.einsum("nki,kj->nij", m - mbar[:, None, :], np.stack([np.cos(th), np.sin(th)], 1))
    al = _verblunsky(g, np.exp(1j * th), k)
    tau = np.exp(2j * np.pi * rng.random(n))
    lam, V = np.linalg.eig(_ggt(np.concatenate([al, tau[:, None]], 1)))
    w = V[:, 0, :].real ** 2 + V[:, 0, :].imag ** 2
    w /= w.sum(1, keepdims=True)
    e = np.stack([lam.real, lam.imag], -1) / np.abs(lam)[..., None]
    return mbar[:, None, :] + np.einsum("nij,nqj->nqi", U, e), w


def conic_draw(k, g, m, tau2, rng, batch=50000):
    n = len(g)
    out = np.empty((n, 2))
    for s in range(0, n, batch):
        P, w = conic_nodes(k, g[s:s + batch], m[s:s + batch], rng)
        cw = np.cumsum(w, 1)
        j = (cw < rng.random(len(w))[:, None] * cw[:, -1:]).sum(1).clip(0, k)
        out[s:s + batch] = P[np.arange(len(w)), j]
    return out + np.sqrt(tau2) * rng.standard_normal((n, 2))


def draw(k, g, m, tau2, rng, eps=0.02):
    n = len(g)
    if k in (4, 5):
        return conic_draw(k, g, m, tau2, rng)
    if k == "inf":
        u = rng.random(n)[:, None]
        j = (np.cumsum(g, 1) < u).sum(1).clip(0, g.shape[1] - 1)
        return m[np.arange(n), j] + np.sqrt(tau2) * rng.standard_normal((n, 2))
    mean, C, d = moments(g, m, tau2)
    if k == 1:
        return mean
    L = sqrt2(*C)
    z = rng.standard_normal((n, 2))
    if k == 2:
        return mean + mv(L, z)
    Li = inv2(*L)
    dw = np.stack([Li[0][:, None] * d[..., 0] + Li[1][:, None] * d[..., 1],
                   Li[1][:, None] * d[..., 0] + Li[2][:, None] * d[..., 1]], -1)
    lam, vx, vy = third_dirs(g, dw)
    nu = np.abs(lam) / SQ2
    Ma, Mb, Mc = (nu * vx * vx).sum(1), (nu * vx * vy).sum(1), (nu * vy * vy).sum(1)
    top = 0.5 * (Ma + Mc) + np.sqrt(0.25 * (Ma - Mc) ** 2 + Mb * Mb)
    f = np.minimum(1.0, (1 - eps) / np.maximum(top, 1e-300))
    nu = nu * f[:, None]
    u = mv(sqrt2(1 - Ma * f, -Mb * f, 1 - Mc * f), z)
    ok = nu > 1e-14
    with np.errstate(invalid="ignore", divide="ignore"):
        gs = np.where(ok, lam / np.where(ok, nu, 1.0) ** 1.5, 0.0)
        p = 0.5 * (1 - np.abs(gs) / np.sqrt(gs * gs + 4))
        Lr = np.where(ok, np.sqrt(nu / (p * (1 - p))), 0.0)
    far = rng.random(lam.shape) < p
    xi = np.sign(gs) * np.where(far, (1 - p) * Lr, -p * Lr)
    u = u + np.stack([(xi * vx).sum(1), (xi * vy).sum(1)], -1)
    return mean + mv(L, u)


def flow_grid(N, shift=3.0):
    u = np.linspace(1, 1 / N, N)
    return np.append(shift * u / (1 + (shift - 1) * u), 0.0)


def run(mix, k, N, xT, rng, exact_last=False):
    """M(k) sampler on the N-step grid; the step is the Euler step with the posterior mean replaced by xh ~ q_k."""
    ts = flow_grid(N)
    x = xT.copy()
    for i in range(N):
        t, ts_ = ts[i], ts[i + 1]
        g, m, tau2 = mix.post(x, t)
        xh = draw("inf" if (exact_last and i == N - 1) else k, g, m, tau2, rng)
        if ts_ == 0:
            x = xh
        else:
            eh = (x - (1 - t) * xh) / t
            x = (1 - ts_) * xh + ts_ * eh                      # = (1 - a) x + a xh, a = (t - ts_) / t
    return x


# ---------------------------------------------------------------- measures

def fourier_grid(ell, nr=10, na=24):
    xr, wr = np.polynomial.laguerre.laggauss(nr)
    r = np.sqrt(2 * xr) / ell
    a = np.pi * np.arange(na) / na
    W = np.stack([np.outer(r, np.cos(a)).ravel(), np.outer(r, np.sin(a)).ravel()], 1)
    return W, np.repeat(wr, na) / na


def phi_sum(x, W):
    acc = np.zeros(len(W), complex)
    for i in range(0, len(x), 50000):
        acc += np.exp(1j * (x[i:i + 50000] @ W.T)).sum(0)
    return acc


def mmd2_from(acc, n, mix, W, w):
    php = np.exp(-0.5 * mix.s2 * (W * W).sum(1)) * np.exp(1j * (W @ mix.mu.T)).mean(1)
    phq = acc / n
    return float((w * (np.abs(phq - php) ** 2 - (1 - np.abs(phq) ** 2) / (n - 1))).sum())


def binmass(mix, xe, ye):
    from scipy.special import erf
    s = mix.s * SQ2
    px = np.diff(0.5 * (1 + erf((xe[None, :] - mix.mu[:, 0:1]) / s)), axis=1)
    py = np.diff(0.5 * (1 + erf((ye[None, :] - mix.mu[:, 1:2]) / s)), axis=1)
    return np.einsum("ki,kj->ij", px, py) / mix.K


# ---------------------------------------------------------------- cache

def _job(a):
    kind, k, N, c = a
    mix = Mix()
    W, _ = fourier_grid(ELL)
    if kind == "null":
        x, _ = mix.sample(NCH * CH, np.random.default_rng([c, 31]))
        return a, phi_sum(x, W), None
    xT = np.random.default_rng([c, 7]).standard_normal((CH, 2))
    if kind == "cell":
        x = run(mix, k if k == "inf" else int(k), N, xT, np.random.default_rng([c, N, 0, 0, KIDX[k]]))
    else:                                                       # last step exact
        x = run(mix, int(k), N, np.random.default_rng([c, 7]).standard_normal((CH, 2)),
                np.random.default_rng([c, N, int(k), 5]), exact_last=True)
    extra = None
    if kind == "cell" and N == NSHOW:
        h, _, _ = np.histogram2d(x[:, 0], x[:, 1], [XE, YE])
        d2 = ((x[:, None, :] - mix.mu[None]) ** 2).sum(-1).min(1)
        extra = (h, float((mix.logp(x) < _Q01).sum()), float(d2.sum() / (2 * mix.s2)))
    return a, phi_sum(x, W), extra


_Q01 = None


def _init(q01):
    global _Q01
    _Q01 = q01


def compute(procs, ks=None):
    """ks=None rebuilds the whole cache; a list of keys computes only those cells and merges them into it."""
    from multiprocessing import Pool
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mix = Mix()
    W, w = fourier_grid(ELL)
    q01 = float(np.quantile(mix.logp(mix.sample(4000000, np.random.default_rng(999))[0]), 0.01))
    merge = ks is not None
    ks = ks or KS
    jobs = [("cell", k, N, c) for k in ks for N in NS for c in range(NCH)]
    if not merge:
        jobs += [("null", None, None, r) for r in range(8)]
        jobs += [("last", k, 8, c) for k in ks if k in ("1", "2", "3") for c in range(3)]
    jobs.sort(key=lambda a: -(a[2] or 50) * {"3": 2, "4": 12, "5": 13}.get(a[1], 1))
    acc, null, hist = {}, [], {k: [np.zeros((len(XE) - 1, len(YE) - 1)), 0.0, 0.0] for k in ks}
    with Pool(procs, initializer=_init, initargs=(q01,)) as pool:
        for i, (a, ph, extra) in enumerate(pool.imap_unordered(_job, jobs)):
            print(f"{i + 1}/{len(jobs)} done: {a}", flush=True)
            if a[0] == "null":
                null.append(mmd2_from(ph, NCH * CH, mix, W, w))
                continue
            key = (a[0], a[1], a[2])
            s = acc.get(key, [0, 0]); acc[key] = [s[0] + ph, s[1] + CH]
            if extra is not None:
                hist[a[1]][0] += extra[0]; hist[a[1]][1] += extra[1]; hist[a[1]][2] += extra[2]
    os.makedirs(CACHE, exist_ok=True)
    n = NCH * CH
    new = {k: {str(N): mmd2_from(*acc[("cell", k, N)], mix, W, w) for N in NS} for k in ks}
    per_k = {}
    for k in ks:
        per_k[f"h{k}"], per_k[f"rare_{k}"], per_k[f"within_{k}"] = hist[k][0], hist[k][1] / n, hist[k][2] / n
    if merge:                                                   # keep every existing cell, add or replace ks
        conv = json.load(open(os.path.join(CACHE, "conv.json")))
        assert conv["NS"] == NS and conv["n"] == n
        conv["mmd2"].update(new)
        json.dump(conv, open(os.path.join(CACHE, "conv.json"), "w"))
        old = dict(np.load(os.path.join(CACHE, "n32.npz")))
        old.update(per_k)
        np.savez(os.path.join(CACHE, "n32.npz"), **old)
        print(f"merged {ks} into {CACHE}", flush=True)
        return
    conv = {"NS": NS, "n": n, "null": null, "mmd2": new}
    json.dump(conv, open(os.path.join(CACHE, "conv.json"), "w"))
    np.savez(os.path.join(CACHE, "n32.npz"), xe=XE, ye=YE, n=n, **per_k)
    last = {k: {"full": math.sqrt(max(conv["mmd2"][k]["8"], 0)),
                "last_exact": math.sqrt(max(mmd2_from(*acc[("last", k, 8)], mix, W, w), 0))}
            for k in ks if k in ("1", "2", "3")}
    json.dump(last, open(os.path.join(CACHE, "laststep.json"), "w"))


def test(n=2000, seed=3):
    """Exactness of q_4 / q_5: the conic measure's moments vs the posterior mixing measure's, in units of the conic's
    radius, for every monomial up to degree k+1 (degree k+1 is free, and differs where > k components carry weight)."""
    mix = Mix()
    rng = np.random.default_rng(seed)
    th = 2 * np.pi * np.arange(mix.K) / mix.K
    ok = True
    for t in [1.0, 0.9, 0.5, 0.2, 0.05, 0.01]:
        x0, _ = mix.sample(n, rng)
        g, m, tau2 = mix.post((1 - t) * x0 + t * rng.standard_normal((n, 2)), t)
        m = np.ascontiguousarray(m)
        mbar = m.mean(1)
        U = (2.0 / mix.K) * np.einsum("nki,kj->nij", m - mbar[:, None], np.stack([np.cos(th), np.sin(th)], 1))
        fit = np.abs(mbar[:, None] + np.einsum("nij,kj->nki", U, np.stack([np.cos(th), np.sin(th)], 1)) - m).max()
        r = np.sqrt(np.abs(np.linalg.det(U)))[:, None, None]
        mean = np.einsum("nk,nki->ni", g, m)[:, None]
        for k in (4, 5):
            P, w = conic_nodes(k, g, m, rng)
            a_, b_ = (m - mean) / r, (P - mean) / r
            err = [max(np.abs((g * a_[..., 0] ** i * a_[..., 1] ** (d - i)).sum(1)
                              - (w * b_[..., 0] ** i * b_[..., 1] ** (d - i)).sum(1)).max() for i in range(d + 1))
                   for d in range(1, k + 2)]
            ok &= max(err[:k]) < 1e-9 and fit < 1e-12
            print(f"t={t:<5} M({k}): conic fit {fit:.0e}; max moment error, degrees 1..{k}: {max(err[:k]):.0e}; "
                  f"degree {k + 1} (free): {err[k]:.0e}")
    print("PASS" if ok else "FAIL")
    return ok


def compute_until_floor(procs, ks, max_n=None):
    """Add samplers ks to the cache one step count at a time, in increasing N, and stop each one at the first N whose
    MMD is at the noise floor of the cached null runs; larger N stay out of the cache and the plot. With max_n, fill
    every N <= max_n instead, without stopping at the floor. Cells already cached are kept and not recomputed.
    Same seeds as compute(). The cache is saved after every N, so an interrupted run keeps what it finished."""
    from multiprocessing import Pool
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mix = Mix()
    W, w = fourier_grid(ELL)
    q01 = float(np.quantile(mix.logp(mix.sample(4000000, np.random.default_rng(999))[0]), 0.01))
    conv = json.load(open(os.path.join(CACHE, "conv.json")))
    npz = dict(np.load(os.path.join(CACHE, "n32.npz")))
    assert conv["NS"] == NS and conv["n"] == NCH * CH
    floor = math.sqrt(2 * np.std(conv["null"], ddof=1))
    n = NCH * CH
    active = list(ks)
    for k in ks:
        conv["mmd2"].setdefault(k, {})
    with Pool(procs, initializer=_init, initargs=(q01,)) as pool:
        for N in NS:
            if not active or (max_n is not None and N > max_n):
                break
            todo = [k for k in active if str(N) not in conv["mmd2"][k]]
            acc = {k: [0, 0] for k in todo}
            hist = {k: [0, 0.0, 0.0] for k in todo}
            for a, ph, extra in pool.imap_unordered(_job, [("cell", k, N, c) for k in todo for c in range(NCH)]):
                k = a[1]
                acc[k] = [acc[k][0] + ph, acc[k][1] + CH]
                if extra is not None:
                    hist[k] = [hist[k][0] + extra[0], hist[k][1] + extra[1], hist[k][2] + extra[2]]
            for k in list(active):
                if k in todo:
                    conv["mmd2"][k][str(N)] = mmd2_from(*acc[k], mix, W, w)
                    if N == NSHOW:
                        npz[f"h{k}"], npz[f"rare_{k}"], npz[f"within_{k}"] = hist[k][0], hist[k][1] / n, hist[k][2] / n
                v = conv["mmd2"][k][str(N)]
                at = math.sqrt(max(v, 0)) <= floor and max_n is None
                print(f"M({k}) N={N}: MMD {math.sqrt(max(v, 0)):.5f}, floor {floor:.5f}"
                      + ("" if k in todo else " (cached)") + ("  -> at the floor, stop" if at else ""), flush=True)
                if at:
                    active.remove(k)
            json.dump(conv, open(os.path.join(CACHE, "conv.json"), "w"))
            np.savez(os.path.join(CACHE, "n32.npz"), **npz)


# ---------------------------------------------------------------- figure

def save(fig, out):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out + ".pdf", dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}.pdf/.png", flush=True)


def plot(out, layout="col", show=None, dens=None):
    """layout "wide": a five-panel wide figure.  layout "col" (CVPR one column): <out> holds the error-vs-steps
    panel alone, <out>-density the four density panels as a 2x2 grid with the colour bar below.
    show: draw only these samplers (e.g. ["1", "2"]) and write the error panel alone, no density figure.
    dens: with show, also write <out>-density with only these samplers' density panels, side by side."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from reproduce import style as F
    plt = F.rc()

    COL = {"1": "#2a78d6", "2": "#eb6834", "3": "#1baf7a", "4": "#eda100", "5": "#e87ba4", "inf": "#4b515a"}
    LAB = {"1": "$M(1)$ = Euler", "2": "$M(2)$", "3": "$M(3)$", "4": "$M(4)$", "5": "$M(5)$",
           "inf": r"$M(\infty)$, posterior draws"}
    conv = json.load(open(os.path.join(CACHE, "conv.json")))
    H = np.load(os.path.join(CACHE, "n32.npz"))
    last = json.load(open(os.path.join(CACHE, "laststep.json")))
    floor = math.sqrt(2 * np.std(conv["null"], ddof=1))
    NS_ = conv["NS"]
    KC = [k for k in KS if k in conv["mmd2"] and (show is None or k in show)]   # cached samplers, legend order
    curve = {k: np.array([math.sqrt(max(conv["mmd2"][k].get(str(N), math.nan), 0)) for N in NS_]) for k in KC}

    def stat(name, k):                                          # per-key N=32 statistics (old caches: arrays)
        if f"{name}_{k}" in H:
            return float(H[f"{name}_{k}"])
        return float(H[name][KS_DENS.index(k)]) if k in KS_DENS else math.nan

    def eq_euler(v):
        ys = curve["1"]
        for i in range(len(NS_) - 1):
            if ys[i] >= v >= ys[i + 1]:
                f = (math.log(v) - math.log(ys[i])) / (math.log(ys[i + 1]) - math.log(ys[i]))
                return math.exp(math.log(NS_[i]) + f * (math.log(NS_[i + 1]) - math.log(NS_[i])))
        return float("nan")

    wide = layout == "wide"
    if wide:
        fig = plt.figure(figsize=(5.5, 2.3))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.32, 1, 1], wspace=0.06, hspace=0.32, left=0.075, right=0.985,
                              bottom=0.17, top=0.9)
        ax = fig.add_subplot(gs[:, 0])
    else:
        fig = plt.figure(figsize=(3.3, 2.0))
        ax = fig.add_subplot(111)
    ax.axhspan(1e-6, floor, color="#ececea", lw=0, zorder=0)
    for k in KC:
        y = np.maximum(curve[k], floor)
        hol = curve[k] <= floor
        z = 2.5 if k == "inf" else 3                           # the reference sits under the samplers on the floor
        ax.plot(NS_, y, color=COL[k], lw=1.2, ls=(0, (3, 1.6)) if k == "inf" else "-", zorder=z, label=LAB[k])
        done = np.flatnonzero(~np.isnan(curve[k]))
        if done[-1] < len(NS_) - 1 and hol[done[-1]]:          # stopped once at the floor: carry on, no markers
            ax.plot([NS_[done[-1]], NS_[-1]], [floor, floor], color=COL[k], lw=1.2, zorder=z)
        ax.plot(np.array(NS_)[~hol], y[~hol], "o", ms=2.6, color=COL[k], mec="white", mew=0.4, zorder=z + 1)
        ax.plot(np.array(NS_)[hol], y[hol], "o", ms=2.6, mfc="white", mec=COL[k], mew=0.7, zorder=z + 1)
    eqs = {k: eq_euler(curve[k][NS_.index(32)]) for k in ["2", "3", "4", "5"] if k in curve}   # printed only
    ax.text(1.1, floor * 0.68, "noise floor", color=F.MUTED, fontsize=6, va="center")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 4, 16, 64, 256, 1024]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1024"])
    ax.set_xlim(0.8, 1300); ax.set_ylim(floor * 0.5, 1.6)
    ax.set_xlabel("sampling steps $N$")
    ax.set_ylabel("MMD to the data", labelpad=1)
    F.style(ax)
    ax.grid(False, which="minor")
    ax.legend(frameon=False, loc="upper right", handlelength=1.8, borderaxespad=0.2, ncol=2 if len(KC) > 4 else 1,
              columnspacing=1.0)
    if wide:
        F.title(ax, "(a) error against steps")
    else:
        save(fig, out)
        if show is not None and dens is None:
            return
        if dens is None:
            fig = plt.figure(figsize=(3.3, 2.3))
            gs = fig.add_gridspec(2, 2, wspace=0.04, hspace=0.2, left=0.01, right=0.99, bottom=0.2, top=0.94)
        else:
            fig = plt.figure(figsize=(3.3, 1.42))
            gs = fig.add_gridspec(1, len(dens), wspace=0.04, left=0.01, right=0.99, bottom=0.3, top=0.92)

    mix = Mix()
    xe, ye = H["xe"], H["ye"]
    pm = binmass(mix, xe, ye)
    area = (xe[1] - xe[0]) * (ye[1] - ye[0])
    KD = dens or KS_DENS
    diffs = {k: (H[f"h{k}"] / float(H["n"]) - pm) / area for k in KS_DENS}
    lim = float(np.quantile(np.abs(np.concatenate([diffs[k].ravel() for k in ["1", "2", "3"]])), 0.995))
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f4f4f2", "#f0a07c", "#c4371a"])
    th = np.linspace(0, 2 * np.pi, 60)
    for j, k in enumerate(KD):
        a2 = fig.add_subplot(gs[0, j] if dens else gs[j // 2, (1 if wide else 0) + j % 2])
        im = a2.imshow(diffs[k].T, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap=cmap,
                       norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest", rasterized=True)
        for mu in mix.mu:
            a2.plot(mu[0] + 2 * mix.s * np.cos(th), mu[1] + 2 * mix.s * np.sin(th), color="#9a9994", lw=0.35)
        a2.set_xticks([]); a2.set_yticks([]); a2.set_aspect("equal")
        for s_ in a2.spines.values():
            s_.set_visible(False)
        F.title(a2, f"({('bcde' if wide else 'abcd')[j]}) " + LAB[k].replace(", posterior draws", "") + ", $N=32$")
    cax = fig.add_axes([0.5, 0.075, 0.4, 0.02] if wide else [0.2, 0.21, 0.6, 0.035] if dens else [0.2, 0.1, 0.6, 0.025])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=5.8, length=1.5, width=0.4, colors=F.MUTED, labelcolor=F.INK)
    cb.set_ticks([-lim, 0, lim]); cb.set_ticklabels(["too few", "0", "too many"])
    cb.set_label("sample density $-$ true density at $N=32$", fontsize=6.2, labelpad=1.5)
    save(fig, out if wide else out + "-density")

    names = {"1": "M(1)", "2": "M(2)", "3": "M(3)", "4": "M(4)", "5": "M(5)", "inf": "M(inf)"}
    print(f"floor (MMD) {floor:.2e}; colour limit +-{lim:.3f}")
    for k in KC:
        hit = next((N for N, v in zip(NS_, curve[k]) if v <= floor), None)
        print(f"{names[k]:7s} MMD at 16/32/1024 steps: {curve[k][NS_.index(16)]:.4f} / {curve[k][NS_.index(32)]:.4f} / "
              f"{curve[k][-1]:.2e}; first at the floor: N = {hit}")
    for Ns in [4, 8, 16, 32]:
        print(f"at {Ns} steps, Euler-equivalent steps: " + ", ".join(
            f"{names[k]} {eq_euler(curve[k][NS_.index(Ns)]):.0f}" for k in eqs))
    for k in KC:
        print(f"{names[k]:7s} N=32: off the data (below real data's 1% log-density quantile) {100 * stat('rare', k):.2f}%; "
              f"mean squared distance to the nearest mean / 2s^2 = {stat('within', k):.3f}")
    for k in [k for k in ("1", "2", "3") if k in last]:
        print(f"{names[k]:7s} N=8: full {last[k]['full']:.4f} -> exact last step {last[k]['last_exact']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=False, help="output path without extension")
    ap.add_argument("--compute", action="store_true")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--ks", default=None, help="comma list, e.g. 4,5: compute only these samplers, merge into the cache")
    ap.add_argument("--until-floor", action="store_true",
                    help="with --compute --ks: increasing N, stop each sampler at the first N at the noise floor")
    ap.add_argument("--max-n", type=int, default=None, help="with --until-floor: fill every N up to this instead")
    ap.add_argument("--test", action="store_true", help="check that q_4 / q_5 match the posterior's moments, then exit")
    ap.add_argument("--layout", choices=["col", "wide"], default="col",
                    help="col (default): error panel alone + <out>-density 2x2; wide: the old five-panel figure")
    ap.add_argument("--show", default=None,
                    help="comma list, e.g. 1,2: plot only these samplers, error panel alone (col layout)")
    ap.add_argument("--density", default=None,
                    help="with --show: also write <out>-density with only these samplers' panels, e.g. 1,2")
    args = ap.parse_args()
    if args.test:
        sys.exit(0 if test() else 1)
    if args.compute and args.until_floor:
        compute_until_floor(args.procs, args.ks.split(","), args.max_n)
    elif args.compute or not os.path.exists(os.path.join(CACHE, "conv.json")):
        compute(args.procs, args.ks.split(",") if args.ks else None)
    if not args.out:
        sys.exit(0)
    if args.show and args.layout != "col":
        ap.error("--show needs --layout col")
    if args.density and not args.show:
        ap.error("--density needs --show")
    plot(args.out, args.layout, args.show.split(",") if args.show else None,
         args.density.split(",") if args.density else None)
