#!/usr/bin/env python
"""Sec. 7, "Guidance reduces diversity": what errors correlated across nearby x_t do to the samples (toy T1).

    python reproduce/toys/corr_diversity.py compute --slice S1 S2 S3 S4 [--procs 16] [--shard k --shards K]
                                            # ~8 core-hours in all: a CPU array of K = 4 x 16 cores, or one box
    python reproduce/toys/corr_diversity.py report     # after every shard: summary.json + tables, fig.png / -scatter
    python reproduce/toys/corr_diversity.py test       # self-checks, seconds

Sec. 5.2 idealises a guided pair at w* as D_w(x_t) = E[x0|x_t] + Cov(x0|x_t)^{1/2} g with g ~ N(0, I). Redrawing g at
every call is the second-moment sampler M(2) of fig:moments. A trained pair is one fixed function, though, so its g
is a frozen field of (x_t, t), smooth on some scale. This toy keeps what the variance-matching rule sees, the
per-point covariance of D_w, and changes only how g is correlated:

    D_w = E[x0|x_t] + Cov(x0|x_t)^{1/2} alpha g(x_t, t),   g a unit-variance Gaussian random field (Fourier features),
    Cov(g(x, t), g(x', t')) = exp(-|x - x'|^2 / 2 lx^2 - (s - s')^2 / 2 lt^2) I_2,   s = log sigma = log(t / (1 - t)),

on the flow input x_t (marginal std 0.7-1.4, about a network's input scale), with t clamped at 0.999.
  lx = "iid"    g redrawn at every call: M(2) exactly (bitwise, with the same RNG)
  lx = inf      constant in x: every trajectory gets the same push at a given step
  lx = "kt"     lx = sqrt(2) t, the correlation length of Sec. 5.2's ideal denoiser fit to ONE fixed training set
  lx = "exact"  reference: draws from the posterior itself, M(inf)
  lt = 0        an independent field at every step; lt = inf: the same function of x at every step
  alpha         sqrt(c(w, n1, n0)) of eq. (cov-delta): 1 is w*, 1/sqrt(n1) is w = 1, 0 is DDIM (M(1))

Data, grid and sampler are fig:moments (a)'s: K = 10 Gaussians, s = 0.15, on a 2 x 1 ellipse, and the SD3 flow
grid with shift 3. A cell runs R = 32 frozen models (pairs) on the same n = 4000 initial noises, plus R_F = 4 runs of a
fresh model per trajectory (one independent field each). The fresh runs are the ensemble over pairs. Diversity per
model is read against them:
  I(model; mode) = H(pi_fresh) - mean_m H(pi_m): what the pair's identity says about which mode a trajectory reaches
  (0 if the pair does not matter, log K if it alone decides). The mode is the nearest centre. The entropies are
  Miller-Madow corrected.
Per model it also records:
  - H(pi) / log K, TV from uniform, and the modes below 1/(4K)
  - the spread within each blob about its own mean and the blob's offset from its centre (both / 2 s^2)
  - k-NN (Kozachenko-Leonenko) and leave-one-out KDE entropy
  - precision / recall / density / coverage (k = 3, against one fixed n-point data sample) and exact-side MMD
  - tr Cov / tr Cov(x0), and the t at which each trajectory commits to its final mode
  - VM's own statistic: the mean over the grid's t of tr Cov_{x_t ~ p_t}(D_w) / tr Cov(x0), the quantity
    eq. (varmatch) sets to 1

Outputs under $VMG_RUNS/_toys/corr_diversity/: cells/<key>.json and .npz (samples), null.json, summary.json, fig.*.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.figures import fig_moments as FM  # noqa: E402
from vmg import runs  # noqa: E402

INF = math.inf
R_MODELS, R_FRESH, N_TRAJ = 32, 4, 4000
R_VM = 8                                   # models whose VM statistic is evaluated (it is a mean; 8 is plenty)
T_CLAMP = 0.999
F_TRAJ = 512                               # features per trajectory in the fresh-model arm
SEED_X, SEED_FIELD, SEED_FRESH, SEED_IID, SEED_DATA, SEED_CAL = 7, 1000, 2000, 3000, 31, 41
KNN_K = (1, 3, 10)
KDE_H = (0.05, 0.15)


def out_dir():
    return runs.root() / "_toys" / "corr_diversity"


# ---------------------------------------------------------------- the error field

def lsig(t):
    t = min(float(t), T_CLAMP)
    return math.log(t / (1.0 - t))


def lx_at(lx, t):
    return math.sqrt(2.0) * float(t) if lx == "kt" else float(lx)


def f_rule(lx):
    """Features per frozen field: a finite F leaves spurious long-range correlation ~ 1/sqrt(F), which matters once the
    cloud covers many correlation cells (short lx)."""
    if lx in ("iid", "exact"):
        return 0
    return 16384 if (lx == "kt" or float(lx) < 0.1) else 4096


class Field:
    """One frozen unit-variance field (x, t) -> R^2, i.e. one model pair, from F random Fourier features.

    Given its frequencies the field is exactly Gaussian (the amplitudes are). Its covariance is
    (2/F) sum_f cos(w_f.(x - x') + v_f (s - s')), which tends to the kernel at rate 1/sqrt(F). The base draws (W0, v0,
    phases, amplitudes) do not depend on lx, lt, alpha or N, so cells that differ only in those deform one field.
    lt = 0 draws new base parameters at every step index i."""

    def __init__(self, lx, lt, seed, F):
        self.lx, self.lt, self.F = lx, float(lt), int(F)
        self.seed = tuple(int(s) for s in np.atleast_1d(seed))
        self._cache = {}
        self._p = None if self.lt == 0 else self._draw(self.seed)

    def _draw(self, seed):
        r = np.random.default_rng(list(seed))
        W0 = r.standard_normal((self.F, 2))
        v0 = r.standard_normal(self.F)
        ph = r.uniform(0.0, 2 * np.pi, self.F)
        A = r.standard_normal((self.F, 2)) * math.sqrt(2.0 / self.F)
        return W0, v0, ph, A

    def params(self, i):
        if self._p is not None:
            return self._p
        if i not in self._cache:
            self._cache = {i: self._draw(self.seed + (int(i),))}      # keep the current step only
        return self._cache[i]

    def phase(self, ph, v0, t):
        return ph if self.lt in (0.0, INF) else ph + v0 * (lsig(t) / self.lt)

    def __call__(self, x, t, i):
        W0, v0, ph, A = self.params(i)
        lx = lx_at(self.lx, t)
        c = self.phase(ph, v0, t)
        z = np.broadcast_to(c, (len(x), self.F)) if lx == INF else x @ (W0.T / lx) + c
        return np.cos(z) @ A


class TrajField(Field):
    """A fresh model per trajectory: trajectory j has its own field (F_TRAJ features). This is the ensemble over
    pairs, and the reference against which one pair's diversity is read."""

    def __init__(self, lx, lt, seed, n, F=F_TRAJ):
        self.n = int(n)
        super().__init__(lx, lt, seed, F)

    def _draw(self, seed):
        r = np.random.default_rng(list(seed))
        n, F = self.n, self.F
        return (r.standard_normal((n, F, 2)), r.standard_normal((n, F)), r.uniform(0.0, 2 * np.pi, (n, F)),
                r.standard_normal((n, F, 2)) * math.sqrt(2.0 / F))

    def __call__(self, x, t, i):
        W0, v0, ph, A = self.params(i)
        lx = lx_at(self.lx, t)
        c = self.phase(ph, v0, t)
        z = c if lx == INF else np.einsum("nd,nfd->nf", x, W0) / lx + c
        return np.einsum("nf,nfd->nd", np.cos(z), A)


class IID:
    """g redrawn at every call: M(2)."""

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, x, t, i):
        return self.rng.standard_normal((len(x), 2))


def c_w(w, n1, n0):
    """Eq. (cov-delta): Cov(D_w) = c(w) Cov(x0|x_t) for independent ideal denoisers on n1 and n0 points."""
    return w * w / n1 + (1 - w) ** 2 / n0


def w_star(n1, n0):
    """Eq. (wstar-boot): the two roots of c(w) = 1."""
    r = math.sqrt(n0 * n1 * (n0 + n1 - 1))
    return (n1 + r) / (n0 + n1), (n1 - r) / (n0 + n1)


class Pair:
    """The explicit pair of Sec. 5.2: D_m = E[x0|x_t] + Cov^{1/2} g_m / sqrt(n_m), g1 and g0 independent fields, guided
    at w. Its noise w g1/sqrt(n1) + (1 - w) g0/sqrt(n0) is sqrt(c(w)) times a unit field of the same kernel, so it equals
    the alpha form in law (tested; the sampler uses the alpha form)."""

    def __init__(self, f1, f0, n1, n0, w):
        self.f1, self.f0, self.a1, self.a0 = f1, f0, w / math.sqrt(n1), (1 - w) / math.sqrt(n0)

    def __call__(self, x, t, i):
        return self.a1 * self.f1(x, t, i) + self.a0 * self.f0(x, t, i)


# ---------------------------------------------------------------- the sampler

def run_field(mix, N, xT, alpha, noise, last=None):
    """fig_moments.run with M(2)'s z replaced by alpha * noise(x, t, i); noise None (or alpha 0) is M(1). `last` is a
    noise for the final step only (e.g. IID), else the same noise. Returns x0 and the posterior's most likely component
    at every step, [N, n]."""
    ts = FM.flow_grid(N)
    x = xT.copy()
    modes = np.empty((N, len(x)), np.int8)
    for i in range(N):
        t, ts_ = ts[i], ts[i + 1]
        g, m, tau2 = mix.post(x, t)
        modes[i] = g.argmax(1)
        mean, C, _ = FM.moments(g, m, tau2)
        nz = last if (last is not None and i == N - 1) else noise
        if nz is None or alpha == 0:
            xh = mean
        else:
            z = nz(x, t, i)
            xh = mean + FM.mv(FM.sqrt2(*C), z if alpha == 1 else alpha * z)
        if ts_ == 0:
            x = xh
        else:
            eh = (x - (1 - t) * xh) / t
            x = (1 - ts_) * xh + ts_ * eh
    return x, modes


# ---------------------------------------------------------------- measures

def tr_x0(mix):
    return float(2 * mix.s2 + ((mix.mu - mix.mu.mean(0)) ** 2).sum(1).mean())


def mode_of(mix, x):
    return ((x[:, None, :] - mix.mu[None]) ** 2).sum(-1).argmin(1)


def entropy(p, n):
    """Plug-in entropy (nats) of a histogram from n samples, Miller-Madow corrected."""
    q = p[p > 0]
    return float(-(q * np.log(q)).sum() + (len(q) - 1) / (2.0 * n))


def knn_entropy(x, k):
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    r = cKDTree(x).query(x, k + 1)[0][:, k]
    return float(digamma(len(x)) - digamma(k) + math.log(math.pi) + 2 * np.log(np.maximum(r, 1e-300)).mean())


def kde_entropy(x, h, chunk=1000):
    n = len(x)
    x2 = (x * x).sum(1)
    tot = np.empty(n)
    for a in range(0, n, chunk):
        d2 = np.maximum(x2[a:a + chunk, None] + x2[None] - 2.0 * x[a:a + chunk] @ x.T, 0.0)
        k = np.exp(-0.5 * d2 / (h * h))
        k[np.arange(len(k)), np.arange(a, a + len(k))] = 0.0
        tot[a:a + chunk] = k.sum(1)
    dens = tot / ((n - 1) * 2 * math.pi * h * h)
    return float(-np.log(np.maximum(dens, 1e-300)).mean())


def blob_stats(mix, x, k):
    """Spread about each blob's own sample mean, and the blobs' offsets from their centres, both / 2 s^2 per sample.
    They add up to the mean squared distance to the nearest centre / 2 s^2 (1 on data)."""
    own = off = 0.0
    for j in range(mix.K):
        sel = x[k == j]
        if len(sel):
            mj = sel.mean(0)
            own += float(((sel - mj) ** 2).sum())
            off += len(sel) * float(((mj - mix.mu[j]) ** 2).sum())
    z = len(x) * 2 * mix.s2
    return own / z, off / z


def data_ref(mix, n):
    return mix.sample(n, np.random.default_rng([SEED_DATA, 999]))[0]


def commit_t(modes, final, N):
    """t from which the posterior's top component stays at the trajectory's final mode (step 0, the prior, skipped).
    Larger t = earlier commitment; 0 = only at the end."""
    ts = FM.flow_grid(N)
    diff = modes[1:] != final[None].astype(modes.dtype)
    n1 = diff.shape[0]
    idx = np.where(diff.any(0), n1 + 1 - np.argmax(diff[::-1], 0), 1)
    return ts[np.minimum(idx, N)]


def metrics(mix, x, ref, modes=None, N=None):
    from vmg.metrics.prdc import prdc
    n = len(x)
    k = mode_of(mix, x)
    pi = np.bincount(k, minlength=mix.K) / n
    own, off = blob_stats(mix, x, k)
    W, wq = FM.fourier_grid(FM.ELL)
    out = dict(pi=pi.tolist(), H=entropy(pi, n), Hn=entropy(pi, n) / math.log(mix.K),
               tv=float(0.5 * np.abs(pi - 1.0 / mix.K).sum()), dropped=int((pi < 0.25 / mix.K).sum()),
               spread_own=own, offset=off, spread=own + off,
               mmd=FM.mmd2_from(FM.phi_sum(x, W), n, mix, W, wq),
               trcov=float(np.trace(np.cov(x.T)) / tr_x0(mix)),
               mean_shift=float(np.linalg.norm(x.mean(0) - mix.mu.mean(0))))
    for kk in KNN_K:
        out[f"knn{kk}"] = knn_entropy(x, kk)
    for h in KDE_H:
        out[f"kde{h:g}"] = kde_entropy(x, h)
    out.update({key: float(v) for key, v in prdc(ref, x, k=3).items()})
    if modes is not None:
        tc = commit_t(modes, k, N)
        out.update(tcommit_mean=float(tc.mean()), tcommit_q=np.quantile(tc, [0.1, 0.5, 0.9]).tolist(),
                   last_vs_nearest=float((modes[-1] == k).mean()))
    return out


def vm_stat(mix, noise, alpha, N, n=N_TRAJ):
    """VM's own statistic for one model: tr Cov over x_t ~ p_t of D_w(x_t) / tr Cov(x0), at the grid's t, and its mean
    (eq. (varmatch) with t on the sampling grid instead of U[0, 1])."""
    rng = np.random.default_rng([SEED_CAL])
    x0 = mix.sample(n, rng)[0]
    eps = rng.standard_normal((n, 2))
    ts = FM.flow_grid(N)[:-1]
    vals = []
    for i, t in enumerate(ts):
        xt = (1 - t) * x0 + t * eps
        g, m, tau2 = mix.post(xt, t)
        mean, C, _ = FM.moments(g, m, tau2)
        D = mean if (noise is None or alpha == 0) else mean + FM.mv(FM.sqrt2(*C), alpha * noise(xt, t, i))
        vals.append(float(np.trace(np.cov(D.T)) / tr_x0(mix)))
    return float(np.mean(vals)), vals


# ---------------------------------------------------------------- cells

def cell(alpha=1.0, lx="iid", lt=0.0, N=16, F=None, last="field", n=N_TRAJ):
    """Canonical cell dict: alpha 0 is DDIM whatever the field; iid and exact ignore lt, F and last."""
    alpha = float(alpha)
    if alpha == 0 or lx in ("iid", "exact"):
        lx = "iid" if alpha == 0 else lx
        lt, F, last = 0.0, 0, "field"
    else:
        lx = lx if lx == "kt" else float(lx)
        F = f_rule(lx) if F is None else int(F)
    return dict(alpha=alpha, lx=lx, lt=float(lt), N=int(N), F=int(F), last=last, n=int(n))


def _fmt(v):
    return v if isinstance(v, str) else ("inf" if v == INF else f"{v:g}")


def key(c):
    return f"a{_fmt(c['alpha'])}_lx{_fmt(c['lx'])}_lt{_fmt(c['lt'])}_N{c['N']}_F{c['F']}_{c['last']}_n{c['n']}"


LXS = ["iid", 0.03, 0.1, 0.3, 1.0, 3.0, INF, "kt"]
LTS = [0.0, 0.3, 1.0, INF]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
NS = [4, 8, 16, 32, 64, 128]


def slices(n=N_TRAJ):
    S1 = [cell(1.0, lx, lt, 16, last=last, n=n) for lx in LXS for lt in LTS for last in ("field", "iid")]
    S1 += [cell(1.0, "exact", N=16, n=n), cell(0.0, N=16, n=n)]
    S2 = [cell(a, lx, lt, 16, n=n) for a in ALPHAS for lx, lt in [("iid", 0), (0.3, 1.0), (1.0, 1.0), (INF, INF)]]
    S3 = [cell(1.0, lx, lt, N, n=n) for N in NS for lx, lt in [("iid", 0), (1.0, 0.0), (1.0, 1.0), (1.0, INF),
                                                               (0.3, 1.0)]]
    S3 += [cell(1.0, "exact", N=N, n=n) for N in NS] + [cell(0.0, N=N, n=n) for N in NS]
    S4 = [cell(1.0, lx, INF, 16, F=F, n=n) for F in (1024, 4096, 16384) for lx in (0.03, 0.1)]
    # l_x at lt = 0 and N = 128, where M(2) has converged (fig:moments (a): MMD^2 5e-7 there, the n = 1e6 null).
    # F = 4096 throughout: S4 shows lx = 0.03 identical at F = 4096 and 16384 (P .945/.946, R .975/.975).
    S5 = [cell(1.0, lx, 0.0, 128, F=4096, n=n) for lx in (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3)]
    # checks on S5's short-lx clumping: F = 16384 at lx = 0.03, and the step trend at lx = 0.03 and 0.1
    S6 = [cell(1.0, 0.03, 0.0, 128, F=16384, n=n)] + [cell(1.0, lx, 0.0, N, F=4096, n=n) for lx in (0.03, 0.1)
                                                      for N in (32, 64, 256)]
    return dict(S1=S1, S2=S2, S3=S3, S4=S4, S5=S5, S6=S6)


def make_noise(c, arm, r):
    """The noise of one run: arm 'model' (one frozen field per model r) or 'fresh' (a field per trajectory)."""
    lx, lt, F, n = c["lx"], c["lt"], c["F"], c["n"]
    if c["alpha"] == 0:
        return None, None
    last = IID(np.random.default_rng([SEED_IID, 500, r, arm == "fresh"])) if c["last"] == "iid" else None
    if lx == "iid":
        return IID(np.random.default_rng([SEED_IID, r, arm == "fresh"])), last
    if arm == "model":
        return Field(lx, lt, (SEED_FIELD, r), F), last
    return TrajField(lx, lt, (SEED_FRESH, r), n), last


def run_one(c, arm, r):
    mix = FM.Mix()
    n, N = c["n"], c["N"]
    xT = np.random.default_rng([SEED_X]).standard_normal((n, 2))
    if c["lx"] == "exact":
        x = FM.run(mix, "inf", N, xT, np.random.default_rng([SEED_IID, 77, r, arm == "fresh"]))
        modes = None
    else:
        noise, last = make_noise(c, arm, r)
        x, modes = run_field(mix, N, xT, c["alpha"], noise, last)
    m = metrics(mix, x, data_ref(mix, n), modes, N)
    if c["lx"] != "exact" and (arm == "fresh" and r == 0 or arm == "model" and r < R_VM):
        noise, _ = make_noise(c, arm, r)
        if isinstance(noise, TrajField):              # per-trajectory fields: evaluate field j at calibration point j
            noise = TrajField(c["lx"], c["lt"], (SEED_FRESH, r), n)
        m["vm"], m["vm_t"] = vm_stat(mix, noise, c["alpha"], N, n)
    return m, x.astype(np.float32)


def _job(a):
    kind, c, arm, r = a
    t0 = time.time()
    if kind == "null":
        mix = FM.Mix()
        x = mix.sample(c["n"], np.random.default_rng([SEED_DATA, r]))[0]
        return a, metrics(mix, x, data_ref(mix, c["n"])), None, time.time() - t0
    m, x = run_one(c, arm, r)
    return a, m, x, time.time() - t0


def _mse(v):
    v = np.asarray(v, float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def aggregate(c, models, fresh):
    """models, fresh: lists of metric dicts (index r). Means and SEs over models, and the Jensen gaps."""
    out = dict(cell=c, key=key(c), R=len(models), RF=len(fresh), model={}, fresh={})
    scal = [k for k, v in models[0].items() if isinstance(v, (int, float))]
    for k in scal:
        out["model"][k] = _mse([m[k] for m in models if k in m])
        vals = [f[k] for f in fresh if k in f]
        if vals:
            out["fresh"][k] = _mse(vals)
    pis = np.array([m["pi"] for m in models])
    pif = np.array([f["pi"] for f in fresh]).mean(0)
    n, K = c["n"], pis.shape[1]
    Hm = np.array([m["H"] for m in models])
    Hf = np.array([f["H"] for f in fresh])
    out["I"] = (float(Hf.mean() - Hm.mean()), float(math.hypot(Hm.std(ddof=1) / math.sqrt(len(Hm)),
                                                                Hf.std(ddof=1) / math.sqrt(len(Hf)) if len(Hf) > 1 else 0.0)))
    out["I_mix"] = float(entropy(pis.mean(0), n * len(pis)) - Hm.mean())
    out["pi_model"], out["pi_fresh"] = pis.tolist(), pif.tolist()
    out["K"] = K
    if "tcommit_q" in models[0]:
        out["tcommit_q_model"] = np.mean([m["tcommit_q"] for m in models], 0).tolist()
        out["tcommit_q_fresh"] = np.mean([f["tcommit_q"] for f in fresh], 0).tolist()
    if "vm_t" in models[0]:
        out["vm_t_model"] = np.mean([m["vm_t"] for m in models if "vm_t" in m], 0).tolist()
    return out


def _write(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_fmt))
    os.replace(tmp, path)


def cell_cost(c, R=R_MODELS, RF=R_FRESH):
    """Relative CPU cost of a cell: the field evaluations dominate (n F N per run, twice for the VM statistic)."""
    if c["alpha"] == 0 or c["lx"] in ("iid", "exact"):
        return c["N"] * 64 * (R + RF + R_VM)
    return c["N"] * ((c["F"] + 64) * (R + R_VM) + (2 * F_TRAJ + 64) * (RF + 1))


def shard_cells(cells, shard, shards):
    """Longest-processing-time split: each cell, most expensive first, goes to the least loaded shard."""
    load, own = [0.0] * shards, [[] for _ in range(shards)]
    for c in sorted(cells, key=lambda c: (-cell_cost(c), key(c))):
        j = min(range(shards), key=lambda s: (load[s], s))
        load[j] += cell_cost(c)
        own[j].append(c)
    return own[shard]


def compute(names, procs=10, R=R_MODELS, RF=R_FRESH, n=N_TRAJ, redo=False, verbose=True, shard=0, shards=1):
    S = slices(n)
    cells, seen = [], set()
    for s in names:
        for c in S[s]:
            if key(c) not in seen:
                seen.add(key(c))
                cells.append(c)
    cells = shard_cells(cells, shard, shards)
    d = out_dir() / "cells"
    d.mkdir(parents=True, exist_ok=True)
    todo = [c for c in cells if redo or not (d / f"{key(c)}.json").exists()]
    jobs = [("cell", c, "model", r) for c in todo for r in range(R)]
    jobs += [("cell", c, "fresh", r) for c in todo for r in range(RF)]
    nullc = dict(n=n, R=R)
    null_path = out_dir() / f"null_n{n}.json"
    if shard == 0 and (redo or not null_path.exists()):
        jobs += [("null", nullc, "data", r) for r in range(R)]

    def cost(j):
        c = j[1]
        if j[0] == "null":
            return 0
        f = F_TRAJ * 2 if j[2] == "fresh" else c.get("F", 0)
        return c["N"] * (f + 64) * (1 + (j[3] < R_VM))
    jobs.sort(key=cost, reverse=True)
    if verbose:
        print(f"[corr_diversity] shard {shard}/{shards}: {len(todo)} of {len(cells)} cells to run, {len(jobs)} jobs, "
              f"{procs} procs", flush=True)
    buf, nulls, t0, done = {}, [], time.time(), 0

    def take(res):
        nonlocal done
        (kind, c, arm, r), m, x, dt = res
        done += 1
        if kind == "null":
            nulls.append((r, m))
            if len(nulls) == R:
                ms = [mm for _, mm in sorted(nulls, key=lambda z: z[0])]
                pis = np.array([mm["pi"] for mm in ms])
                Hs = np.array([mm["H"] for mm in ms])
                obj = dict(n=n, R=R, model={k: _mse([mm[k] for mm in ms]) for k, v in ms[0].items()
                                            if isinstance(v, (int, float))},
                           I_mix=float(entropy(pis.mean(0), n * R) - Hs.mean()),
                           H_split=float(Hs[: R // 2].mean() - Hs[R // 2:].mean()),
                           knn_exact=float(-np.mean(FM.Mix().logp(FM.Mix().sample(10 ** 6, np.random.default_rng(5))[0]))))
                _write(null_path, obj)
            return
        b = buf.setdefault(key(c), dict(c=c, model={}, fresh={}, xm={}, xf={}))
        b[arm][r] = m
        (b["xm"] if arm == "model" else b["xf"])[r] = x
        if len(b["model"]) == R and len(b["fresh"]) == RF:
            agg = aggregate(c, [b["model"][i] for i in range(R)], [b["fresh"][i] for i in range(RF)])
            np.savez_compressed(d / f"{key(c)}.npz", x_model=np.stack([b["xm"][i] for i in range(R)]),
                                x_fresh=np.stack([b["xf"][i] for i in range(RF)]))
            _write(d / f"{key(c)}.json", agg)
            del buf[key(c)]
            if verbose:
                el = time.time() - t0
                print(f"  {el / 60:6.1f} min  {done}/{len(jobs)}  {key(c)}  I = {agg['I'][0]:.3f} +- {agg['I'][1]:.3f}"
                      f"  Hn(model) = {agg['model']['Hn'][0]:.3f}  recall = {agg['model']['recall'][0]:.3f}", flush=True)

    if procs <= 1:
        for j in jobs:
            take(_job(j))
    else:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[v] = "1"
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(procs) as pool:
            for res in pool.imap_unordered(_job, jobs):
                take(res)
    return [key(c) for c in cells]


# ---------------------------------------------------------------- summary

def load_cells():
    d = out_dir() / "cells"
    out = {}
    for p in sorted(d.glob("*.json")):
        o = json.loads(p.read_text())
        out[o["key"]] = o
    return out


def find(cells, **kw):
    want = key(cell(**kw))
    return cells.get(want)


def summarize(n=N_TRAJ, verbose=True):
    cells = load_cells()
    null_path = out_dir() / f"null_n{n}.json"
    null = json.loads(null_path.read_text()) if null_path.exists() else None
    rows = []
    for o in cells.values():
        c = o["cell"]
        row = dict(key=o["key"], **c, I=o["I"], I_mix=o["I_mix"])
        for k in ("Hn", "tv", "dropped", "recall", "coverage", "precision", "density", "knn3", "kde0.05", "spread_own",
                  "offset", "trcov", "mmd", "vm", "tcommit_mean", "mean_shift"):
            if k in o["model"]:
                row[k] = o["model"][k]
            if k in o["fresh"]:
                row[k + "_fresh"] = o["fresh"][k]
        rows.append(row)
    summ = dict(meta=dict(R=R_MODELS, RF=R_FRESH, n=n, F_traj=F_TRAJ, time="log sigma, t <= 0.999",
                          F_rule="16384 if lx < 0.1 or lx = sqrt(2) t else 4096", seeds=dict(
                              x=SEED_X, field=SEED_FIELD, fresh=SEED_FRESH, iid=SEED_IID, data=SEED_DATA, cal=SEED_CAL)),
                null=null, rows=rows)
    _write(out_dir() / "summary.json", summ)
    if verbose:
        _print_tables(cells, null, n)
    return summ


def _print_tables(cells, null, n):
    def g(o, k, side="model"):
        return o[side][k][0] if (o and k in o[side]) else float("nan")

    if null:
        print(f"data null (n = {n}, {null['R']} batches): I_mix {null['I_mix']:.4f}  Hn {null['model']['Hn'][0]:.4f}  "
              f"recall {null['model']['recall'][0]:.3f}  knn3 {null['model']['knn3'][0]:.3f}  "
              f"(exact entropy {null['knn_exact']:.3f})  spread {null['model']['spread'][0]:.3f}")
    ref = {nm: find(cells, alpha=a, lx=lx, N=16, n=n) for nm, a, lx in
           [("DDIM", 0.0, "iid"), ("M(2)", 1.0, "iid"), ("M(inf)", 1.0, "exact")]}
    print("\nreferences at N = 16:  " + "  ".join(
        f"{nm}: Hn {g(o, 'Hn'):.3f} recall {g(o, 'recall'):.3f} knn3 {g(o, 'knn3'):.2f}" for nm, o in ref.items() if o))
    for last in ("field", "iid"):
        print(f"\nS1 (alpha = 1, N = 16, last step = {last}): I(model; mode) [nats] / per-model Hn / recall / knn3")
        print("lx \\ lt      " + "".join(f"{_fmt(lt):>28s}" for lt in LTS))
        for lx in LXS[1:]:
            line = f"{_fmt(lx):10s}"
            for lt in LTS:
                o = find(cells, alpha=1.0, lx=lx, lt=lt, N=16, last=last, n=n)
                line += (f"  {o['I'][0]:.3f}/{g(o, 'Hn'):.3f}/{g(o, 'recall'):.2f}/{g(o, 'knn3'):5.2f}" if o
                         else f"{'-':>28s}")
            print(line)
    print("\nS2 (N = 16): I vs alpha")
    for lx, lt in [("iid", 0.0), (0.3, 1.0), (1.0, 1.0), (INF, INF)]:
        os_ = [find(cells, alpha=a, lx=lx, lt=lt, N=16, n=n) for a in ALPHAS]
        print(f"  ({_fmt(lx)}, {_fmt(lt)}): " + " ".join(f"{a:g}:{o['I'][0]:.3f}" for a, o in zip(ALPHAS, os_) if o))
    print("\nS3 (alpha = 1): I vs N")
    for lx, lt in [("iid", 0.0), (1.0, 0.0), (1.0, 1.0), (1.0, INF), (0.3, 1.0)]:
        os_ = [find(cells, alpha=1.0, lx=lx, lt=lt, N=N, n=n) for N in NS]
        print(f"  ({_fmt(lx)}, {_fmt(lt)}): " + " ".join(f"{N}:{o['I'][0]:.3f}" for N, o in zip(NS, os_) if o))
    print("\nS4 (alpha = 1, N = 16, lt = inf): I vs F")
    for lx in (0.03, 0.1):
        os_ = [find(cells, alpha=1.0, lx=lx, lt=INF, N=16, F=F, n=n) for F in (1024, 4096, 16384)]
        print(f"  lx {lx:g}: " + " ".join(f"F{F}:{o['I'][0]:.3f}" for F, o in zip((1024, 4096, 16384), os_) if o))


# ---------------------------------------------------------------- self-test

def test(n=400):
    mix = FM.Mix()
    xT = np.random.default_rng([SEED_X]).standard_normal((n, 2))
    a = FM.run(mix, 2, 8, xT, np.random.default_rng(3))
    b, _ = run_field(mix, 8, xT, 1.0, IID(np.random.default_rng(3)))
    assert np.array_equal(a, b), "iid arm is not M(2)"
    a = FM.run(mix, 1, 8, xT, np.random.default_rng(3))
    for nz in (None, Field(0.3, 1.0, (1, 0), 256), TrajField(0.3, 0.0, (1, 0), n, 64)):
        b, _ = run_field(mix, 8, xT, 0.0, nz)
        assert np.array_equal(a, b), "alpha = 0 is not DDIM"
    wp, wm = w_star(64, 16)
    assert abs(c_w(wp, 64, 16) - 1) < 1e-12 and abs(c_w(wm, 64, 16) - 1) < 1e-12
    print("ok: iid == M(2) bitwise, alpha 0 == M(1) bitwise, c(w*) = 1")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["compute", "summarize", "plot", "report", "test"])
    ap.add_argument("--slice", nargs="+", default=["S1", "S2", "S3", "S4"])
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 10)))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--R", type=int, default=R_MODELS)
    ap.add_argument("--RF", type=int, default=R_FRESH)
    ap.add_argument("--n", type=int, default=N_TRAJ)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.stage == "compute":
        compute(a.slice, a.procs, a.R, a.RF, a.n, a.redo, shard=a.shard, shards=a.shards)
        if a.shards == 1:
            summarize(a.n)
    elif a.stage == "summarize":
        summarize(a.n)
    elif a.stage == "plot":
        plot(a.out, a.n)
    elif a.stage == "report":                         # after every shard: tables, then figures
        summarize(a.n)
        plot(a.out, a.n)
    else:
        test()


LX_PLOT = ["iid", 0.03, 0.1, 0.3, 1.0, 3.0, INF, "kt"]
LX_LAB = ["iid", ".03", ".1", ".3", "1", "3", r"$\infty$", r"$\sqrt{2}t$"]
LX_POS = [0, 1, 2, 3, 4, 5, 6, 7.4]                     # a gap before the sqrt(2) t arm (a schedule, not a number)


def plot(out=None, n=N_TRAJ):
    from reproduce import style as S
    plt = S.rc()
    cells = load_cells()
    null_path = out_dir() / f"null_n{n}.json"
    null = json.loads(null_path.read_text()) if null_path.exists() else None
    out = Path(out) if out else out_dir() / "fig"
    lt_col = dict(zip(LTS, S.SIGRAMP[1:]))               # light = short memory across steps, dark = long
    lt_lab = {0.0: r"$\ell_t=0$ (new field per step)", 0.3: r"$\ell_t=0.3$", 1.0: r"$\ell_t=1$",
              INF: r"$\ell_t=\infty$"}
    cat = [S.COND, S.ACC, S.UNCOND, "#eda100"]
    marks = ["o", "s", "^", "D"]

    def get(lx, lt, metric, side="model", alpha=1.0, N=16, last="field"):
        o = find(cells, alpha=alpha, lx=lx, lt=lt, N=N, last=last, n=n)
        if o is None:
            return np.nan, np.nan
        if metric == "I":
            return tuple(o["I"])
        v = o[side].get(metric)
        return tuple(v) if v else (np.nan, np.nan)

    def lx_axis(ax):
        ax.set_xticks(LX_POS)
        ax.set_xticklabels(LX_LAB)
        ax.set_xlabel(r"correlation length in $x_t$, $\ell_x$")

    def ref(ax, y, label, ls=":"):
        if y is not None and np.isfinite(y):
            ax.axhline(y, color=S.MUTED, lw=0.8, ls=ls, zorder=1)
            ax.text(LX_POS[-1] + 0.3, y, label, color=S.MUTED, fontsize=6, va="center", ha="left")

    def by_lx(ax, metric, fresh=True):
        for lt in LTS:
            ys = np.array([get(lx, lt, metric) for lx in LX_PLOT])
            ax.errorbar(LX_POS, ys[:, 0], ys[:, 1], color=lt_col[lt], lw=1.4, marker="o", ms=3, capsize=0,
                        label=lt_lab[lt], zorder=3)
        if fresh:
            ys = np.array([get(lx, INF, metric, "fresh") for lx in LX_PLOT])
            ax.plot(LX_POS, ys[:, 0], color=S.INK, lw=1.0, ls="--", marker="x", ms=3, label=r"fresh pair per "
                    r"trajectory ($\ell_t=\infty$)", zorder=2)
        lx_axis(ax)

    fig, axs = plt.subplots(2, 4, figsize=(2 * S.W_COL * 1.45, 4.6))
    axs = axs.ravel()
    # (a) I over (lx, lt)
    ax = axs[0]
    M = np.array([[get(lx, lt, "I")[0] for lt in LTS] for lx in LX_PLOT[1:]])
    im = ax.imshow(M, cmap=S.sig_cmap(), aspect="auto", vmin=0, vmax=np.nanmax(M))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] > 0.55 * np.nanmax(M) else S.INK)
    ax.set_xticks(range(len(LTS)))
    ax.set_xticklabels([_fmt(v).replace("inf", r"$\infty$") for v in LTS])
    ax.set_yticks(range(len(LX_PLOT) - 1))
    ax.set_yticklabels(LX_LAB[1:])
    ax.set_xlabel(r"correlation length in $\log\sigma$, $\ell_t$")
    ax.set_ylabel(r"$\ell_x$")
    iid = get("iid", 0.0, "I")[0]
    S.title(ax, rf"(a) $I(\mathrm{{pair}};\mathrm{{mode}})$ [nats], $w^\star$, 16 steps (iid: {iid:.3f})")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    # (b) per-model mode entropy
    ax = axs[1]
    S.style(ax)
    by_lx(ax, "Hn")
    ref(ax, null["model"]["Hn"][0] if null else None, "data")
    ax.set_ylabel(r"$H(\pi)/\log K$ per pair")
    S.title(ax, "(b) mode entropy")
    ax.legend(frameon=False, loc="lower left", fontsize=5.5)
    # (c) recall
    ax = axs[2]
    S.style(ax)
    by_lx(ax, "recall")
    ref(ax, null["model"]["recall"][0] if null else None, "data")
    ref(ax, get("iid", 0.0, "recall")[0], "M(2)", ls="--")
    ax.set_ylabel("recall per pair (k = 3)")
    S.title(ax, "(c) recall")
    # (d) k-NN entropy
    ax = axs[3]
    S.style(ax)
    by_lx(ax, "knn3")
    ref(ax, null["knn_exact"] if null else None, "data")
    ref(ax, get("iid", 0.0, "knn3")[0], "M(2)", ls="--")
    ax.set_ylabel("k-NN entropy per pair [nats]")
    S.title(ax, "(d) differential entropy")
    # (e) within-blob spread and blob offset (same units)
    ax = axs[4]
    S.style(ax)
    for metric, ls, lab in (("spread_own", "-", "spread about own blob mean"), ("offset", "--", "blob offset")):
        ys = np.array([get(lx, INF, metric) for lx in LX_PLOT])
        ax.errorbar(LX_POS, ys[:, 0], ys[:, 1], color=S.SIGRAMP[-1], lw=1.4, ls=ls, marker="o", ms=3, label=lab)
    ref(ax, null["model"]["spread"][0] if null else None, "data")
    lx_axis(ax)
    ax.set_ylabel(r"per sample $/\,2s^2$ ($\ell_t=\infty$)")
    S.title(ax, "(e) within-mode spread")
    ax.legend(frameon=False, fontsize=5.5)
    # (f) VM's own statistic
    ax = axs[5]
    S.style(ax)
    by_lx(ax, "vm", fresh=False)
    ref(ax, 1.0, "target")
    ax.set_ylabel(r"$\overline{\mathrm{tr\,Cov}_{x_t}D_w}\,/\,\mathrm{tr\,Cov}\,x_0$")
    S.title(ax, "(f) what VM matches")
    # (g) I vs steps
    ax = axs[6]
    S.style(ax)
    arms = [((1.0, 0.0), r"$\ell_x=1,\ell_t=0$"), ((1.0, 1.0), r"$\ell_x=1,\ell_t=1$"),
            ((1.0, INF), r"$\ell_x=1,\ell_t=\infty$"), ((0.3, 1.0), r"$\ell_x=.3,\ell_t=1$")]
    for (lx, lt), lab, col, mk in [(a, b, c, m) for (a, b), c, m in zip(arms, cat, marks)]:
        ys = np.array([get(lx, lt, "I", N=N) for N in NS])
        ax.errorbar(NS, ys[:, 0], ys[:, 1], color=col, marker=mk, ms=3, lw=1.4, label=lab)
    ys = np.array([get("iid", 0.0, "I", N=N) for N in NS])
    ax.plot(NS, ys[:, 0], color=S.MUTED, ls=":", marker="x", ms=3, lw=1.0, label="iid (M(2))")
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS)
    ax.set_xticklabels([str(v) for v in NS])
    ax.set_xlabel("sampling steps N")
    ax.set_ylabel("I [nats]")
    S.title(ax, r"(g) $I$ vs steps, $w^\star$")
    ax.legend(frameon=False, fontsize=5.5)
    # (h) I vs alpha
    ax = axs[7]
    S.style(ax)
    arms = [((0.3, 1.0), r"$\ell_x=.3,\ell_t=1$"), ((1.0, 1.0), r"$\ell_x=1,\ell_t=1$"),
            ((INF, INF), r"$\ell_x=\ell_t=\infty$")]
    for (lx, lt), lab, col, mk in [(a, b, c, m) for (a, b), c, m in zip(arms, cat, marks)]:
        ys = np.array([get(lx, lt, "I", alpha=a) for a in ALPHAS])
        ax.errorbar(ALPHAS, ys[:, 0], ys[:, 1], color=col, marker=mk, ms=3, lw=1.4, label=lab)
    ys = np.array([get("iid", 0.0, "I", alpha=a) for a in ALPHAS])
    ax.plot(ALPHAS, ys[:, 0], color=S.MUTED, ls=":", marker="x", ms=3, lw=1.0, label="iid")
    ax.axvline(1.0, color=S.MUTED, lw=0.6, ls="--")
    ax.text(1.02, ax.get_ylim()[1] * 0.95, r"$w^\star$", color=S.MUTED, fontsize=6, va="top")
    ax.set_xlabel(r"error size $\alpha=\sqrt{c(w)}$")
    ax.set_ylabel("I [nats]")
    S.title(ax, r"(h) $I$ vs guidance strength, 16 steps")
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout(w_pad=1.2, h_pad=1.6)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    _plot_scatter(cells, out, n, S, plt)
    print(f"wrote {out}.png/.pdf and {out}-scatter.png")


def _plot_scatter(cells, out, n, S, plt):
    mix = FM.Mix()
    panels = [("data", None), ("M(2): g redrawn every call", dict(lx="iid", lt=0.0)),
              (r"$\ell_x=.1,\ell_t=\infty$", dict(lx=0.1, lt=INF)), (r"$\ell_x=.3,\ell_t=\infty$", dict(lx=0.3, lt=INF)),
              (r"$\ell_x=1,\ell_t=\infty$", dict(lx=1.0, lt=INF)), (r"$\ell_x=\ell_t=\infty$", dict(lx=INF, lt=INF)),
              (r"$\ell_x=\sqrt{2}t,\ell_t=1$", dict(lx="kt", lt=1.0))]
    rows = [("one pair (model 0)", "x_model"), ("another pair (model 1)", "x_model1"), ("fresh pair per trajectory",
                                                                                         "x_fresh")]
    fig, axs = plt.subplots(len(rows), len(panels), figsize=(2 * S.W_COL * 1.45, 3.4), sharex=True, sharey=True)
    for j, (lab, kw) in enumerate(panels):
        for i, (rlab, which) in enumerate(rows):
            ax = axs[i, j]
            if kw is None:
                x = mix.sample(n, np.random.default_rng([SEED_DATA, 100 + i]))[0]
                txt = ""
            else:
                o = find(cells, alpha=1.0, N=16, n=n, **kw)
                if o is None:
                    ax.set_axis_off()
                    continue
                z = np.load(out_dir() / "cells" / f"{o['key']}.npz")
                x = z["x_fresh"][0] if which == "x_fresh" else z["x_model"][1 if which == "x_model1" else 0]
                pi = np.bincount(mode_of(mix, x), minlength=mix.K) / len(x)
                txt = rf"$H/\log K$ {entropy(pi, len(x)) / math.log(mix.K):.2f}"
            ax.scatter(x[:, 0], x[:, 1], s=0.15, color=S.SIGRAMP[-1], alpha=0.5, lw=0, rasterized=True)
            ax.scatter(mix.mu[:, 0], mix.mu[:, 1], s=4, color=S.ACC, lw=0)
            ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=5.5, color=S.INK)
            ax.set_xlim(-2.8, 2.8)
            ax.set_ylim(-1.8, 1.8)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(S.GRID)
            if i == 0:
                ax.set_title(lab, fontsize=6.5, color=S.INK)
            if j == 0:
                ax.set_ylabel(rlab, fontsize=6, color=S.INK)
    fig.suptitle(r"samples at $w^\star$ (16 steps): one frozen error field per pair vs a fresh field per trajectory",
                 fontsize=7, color=S.INK, y=1.0)
    fig.tight_layout(w_pad=0.3, h_pad=0.3)
    fig.savefig(f"{out}-scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
