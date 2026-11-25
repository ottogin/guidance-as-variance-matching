#!/usr/bin/env python
"""M(2) on ImageNet-64 point masses with its noise correlated across x_t (the high-dimensional corr_diversity.py).

    python reproduce/toys/corr_diversity_in64.py setup     # CPU: 10,000 images from the EDM2 zip, then the self-checks
    python reproduce/toys/corr_diversity_in64.py work   [--model dots|gauss]   # GPU: claims units until none is left
    python reproduce/toys/corr_diversity_in64.py report [--model dots|gauss]   # tables from units/*.json (no GPU)

Data as in fig:moments (ImageNet-64): 10 training images per class (d = 12,288), each a point mass, scaled to unit mean
per-coordinate variance, so p(x0 | x_t) is a softmax over the images. They are drawn by moments_highd.build's recipe from
the seed-pair zip (seedpair_in64.py prep), whose row order need not be the train array's, so they can be other images.

One M(2) step draws x0_hat = m + sum_k sqrt(g_k) xi_k (X_k - m), with g the posterior weights, m the posterior mean and
xi ~ N(0, I_K); this has exactly the posterior covariance. Here xi may be a field of x_t, new at every step (l_t = 0):

    Cov(xi_k(x), xi_j(x')) = delta_kj exp(-|x - x'|^2 / 2 l^2),   l = l_x * (median nearest-neighbour image distance).

In the 2-D toy the modes are about 1 apart, so its l_x is already a fraction of the distance between neighbouring
modes. Here every image is a mode, hence the unit. l_x = 0 is vanilla M(2): xi independent per sample and per step.

A run is one model (one field per step) on n = 10,000 shared initial noises. Precision and recall (k = 3, pixel space)
are against the 10,000 images; `data` is 10,000 draws with replacement from them, the ceiling at this n.

--model gauss: every sample of a dot model ends exactly on an image, so precision is 1 by construction. Here each image
becomes N(X_k, s^2 I), and p(x0 | x_t) = sum_k g_k N(m_k, tau^2 I) with v = s^2 + sigma^2, f = s^2 / v,
m_k = f y + (1 - f) X_k, tau^2 = s^2 (1 - f). The M(2) draw gains tau z, with z ~ N(0, I_d) and z a field too:
    x0_hat = m + (1 - f) sum_k sqrt(g_k) xi_k (X_k - Xbar) + tau z,   Xbar = sum_k g_k X_k.
K = 1,000 images (the first of each class), so that n = 10,000 points put about 10 in every Gaussian and the k = 3 balls
measure a Gaussian rather than the gap between images. s sets the pair distance inside a Gaussian, s sqrt(2 d), to 0.3
of the median nearest-image distance, the ratio of the 2-D toy (0.3 against modes about 1 apart). The reference is
n fresh draws from the mixture; `data` is n further draws.

--sweep lt (gauss only): l_x fixed at 1 and the field correlated across noise levels too,
    Cov(xi(x, s), xi(x', s')) = exp(-|x - x'|^2 / 2 l^2 - (s - s')^2 / 2 l_t^2),   s = log sigma, t clamped at 0.999.
l_t = 0 is a new field at every step, l_t = inf one field for the whole trajectory. Because l_t is measured in log sigma,
the correlation between two noise levels does not depend on N, unlike l_t = 0. These runs also record the within-Gaussian
variance a model keeps and precision / recall in random projections, since recall in 12,288 dimensions saturates.

--sweep sched (gauss only): the field's lengths follow EDM2's measured l_x(sigma), l_t(sigma) (fig:errfield-lx, CFG
S - XSu), read from fig_errfield_lsigma.py's JSON (--schedule). The toy's images are EDM2's (the first of each class of
in64-10k.npz), centred and scaled by 1 / c, c = (EDM2 unit) / nn, so sigma_EDM2 = c sigma here and lengths in
nearest-image units are the same in both. EDM2's l_x is a distance between noisy inputs x0 + sigma n; this sampler's
state is the flow x_t = (1 - t)(x0 + sigma n), so the field takes x_t with the length (1 - t) l_x(sigma) nn. In time, a
warped coordinate tau(log sigma) = int dlog sigma / l_t, so two levels correlate as exp(-(tau - tau')^2 / 2), i.e. with
the local l_t. Arms: sched (l_x of steps 1-4, which have no crossing, set so the kernel at the distance of two
independent inputs equals their measured mean correlation; above sigma_EDM2 = 80 l_x grows with sigma, holding that
correlation), schedhold (l_x of step 5 held above sigma_EDM2 = 38), schedlt0 (sched's l_x, a new field at every step).
l_t is held at step 5's value above sigma_EDM2 = 38 (its sigma' > sigma side probes the networks above sigma_max), and
both lengths at their last values below 0.002. Interpolation is linear in log sigma and log l.
"""
import argparse
import io
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from reproduce.figures.fig_moments import flow_grid  # noqa: E402
from vmg import runs  # noqa: E402

F64 = torch.float64
NS = (128, 512)
LXS = (0.03, 0.05, 0.1)                     # the requested lengths
LXS_WIDE = (0.3, 1.0, 3.0)                  # the same axis further out, run last
R, N_SAMPLES, F, PER_CLASS, KNN, SUB = 8, 10000, 2048, 10, 3, 2500
SEED_X, SEED_FIELD, SEED_IID, SEED_DATA, SEED_REAL = 7, 1000, 3000, 31, 37
STALE_S = 900
S_PAIR = 0.3                                # gauss: s sqrt(2 d) / (median nearest-image distance)
LX_FIX, LTS, PROJ = 1.0, (0.0, 0.1, 0.3, 1.0, 3.0, math.inf), (16, 64, 256, 1024)   # --sweep lt
LS_MAX = math.log(0.999 / 0.001)            # log sigma at t = 0.999
# --sweep big: 100k real and 100k generated, metrics in random projections only, plus exact per-Gaussian balance
N_BIG, PROJ_BIG, CH_BIG, R_BIG, STALE_BIG = 100_000, (16, 64, 256), 20_000, 3, 3600
LXS_DENSE = (0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
SCHED_ARMS = ("sched", "schedhold", "schedlt0")   # --sweep sched
SCHEDULE = None                                 # its JSON path (--schedule)


def out_dir(model="dots"):
    d = runs.root() / "_toys" / "corr_diversity_in64"
    return d if model == "dots" else d / model


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gen(dev, *key):
    g = torch.Generator(device=dev)
    g.manual_seed(int(np.random.SeedSequence(list(key)).generate_state(1, np.uint64)[0] >> 1))
    return g


def normalize(X):
    X = X.astype(np.float64) / 127.5 - 1
    X = X - X.mean(0)
    return X / math.sqrt((X ** 2).mean())


# ---------------------------------------------------------------- data, posterior, noise

class Mix:
    """The uniform mixture of N(X_k, s^2 I) over the rows of X; s = 0 is the point masses."""

    def __init__(self, X, dev, dt=torch.float32, s=0.0):
        self.dev, self.dt, self.s = dev, dt, float(s)
        self.X64 = torch.as_tensor(X, device=dev, dtype=F64)
        self.X = self.X64.to(dt)
        self.K, self.d = self.X.shape
        self.X2 = (self.X64 ** 2).sum(1)

    def post(self, y, s2):
        n = len(y)
        if math.isinf(s2):
            return torch.full((n, self.K), 1.0 / self.K, device=self.dev, dtype=F64)
        y64 = y.to(F64)
        v = self.s ** 2 + s2
        return torch.softmax(-0.5 * ((y64 ** 2).sum(1, keepdim=True) - 2 * y64 @ self.X64.T + self.X2) / v, 1)

    def combine(self, w):
        return w.to(self.dt) @ self.X

    def sample(self, n, rng):
        x = self.X[torch.randint(0, self.K, (n,), device=self.dev, generator=rng)]
        return x + self.s * torch.randn(n, self.d, device=self.dev, dtype=self.dt, generator=rng) if self.s else x


class IID:
    """xi (and z) independent per sample and per step: vanilla M(2)."""

    def __init__(self, rng):
        self.rng = rng

    def __call__(self, x, g, i, dz=0, ls=0.0):
        xi = torch.randn(g.shape, device=g.device, dtype=F64, generator=self.rng)
        return xi, torch.randn(len(x), dz, device=g.device, dtype=x.dtype, generator=self.rng) if dz else None


class Field:
    """A unit-variance Gaussian field of (x, log sigma) with K + dz outputs, from F Fourier features.
    lt = 0: a new field at every step; lt = inf: one field for all steps."""

    def __init__(self, ell, seed, d, K, dev, F=F, dz=0, lt=0.0):
        self.ell, self.seed, self.d, self.K, self.dev, self.F, self.dz = float(ell), tuple(seed), d, K, dev, F, dz
        self.lt = float(lt)
        self.i, self.p = None, None

    def params(self, i):
        i = i if self.lt == 0 else 0
        if self.i != i:
            r = gen(self.dev, *self.seed, i)
            W = torch.randn(self.F, self.d, device=self.dev, generator=r) / self.ell
            ph = torch.rand(self.F, device=self.dev, generator=r) * 2 * math.pi
            A = torch.randn(self.F, self.K + self.dz, device=self.dev, generator=r) * math.sqrt(2.0 / self.F)
            v = torch.randn(self.F, device=self.dev, generator=r)
            self.i, self.p = i, (W, ph, A, v)
        return self.p

    def __call__(self, x, g, i, dz=0, ls=0.0):
        W, ph, A, v = self.params(i)
        if 0 < self.lt < math.inf:
            ph = ph + v * (ls / self.lt)
        o = torch.cos(x.to(W.dtype) @ W.T + ph) @ A
        return o[:, :self.K].to(F64), o[:, self.K:].to(x.dtype) if dz else None


class Schedule:
    """EDM2's l_x(sigma) (nearest-image units) and l_t(sigma) (log sigma), as functions of this toy's log sigma."""

    def __init__(self, path, nn, hold=False):
        J = json.loads(Path(path).read_text())
        rows = sorted(J["rows"], key=lambda r: -r["sigma"])
        self.c = J["unit"] / nn
        ls = np.log([r["sigma"] for r in rows])
        first = next(i for i, r in enumerate(rows) if r["l_x_delta"][0] == "=")
        lx = np.array([r["l_x_delta"][1] for r in rows], dtype=float)
        for i in range(first):
            lx[i] = lx[first] if hold else rows[i]["r_rand"] / math.sqrt(-2 * math.log(rows[i]["corr_rand_delta"]))
        lt = np.array([r["l_t_delta"][1] for r in rows], dtype=float)
        lt[:first] = lt[first]
        self.ls, self.llx, self.llt, self.hold = ls[::-1], np.log(lx[::-1]), np.log(lt[::-1]), hold   # ascending
        u = np.linspace(-12.0, 12.0, 24001)                     # tau on a fine log sigma_EDM2 grid
        dt = np.exp(-np.interp(u, self.ls, self.llt))
        self.u, self.tau = u, np.concatenate([[0.0], np.cumsum(0.5 * (dt[1:] + dt[:-1]) * np.diff(u))])
        self.table = [dict(sigma=float(np.exp(a)), l_x=float(np.exp(b)), l_t=float(np.exp(c)))
                      for a, b, c in zip(ls, np.log(lx), np.log(lt))]

    def lx(self, lsE):
        """l_x at log sigma_EDM2 = lsE, in nearest-image units."""
        if lsE > self.ls[-1] and not self.hold:                 # above sigma_max: l_x / sigma held
            return float(np.exp(self.llx[-1] + lsE - self.ls[-1]))
        return float(np.exp(np.interp(lsE, self.ls, self.llx)))

    def t(self, lsE):
        return float(np.interp(lsE, self.u, self.tau))


class SchedField:
    """Field's sibling with the length (1 - t) l_x(sigma) nn at step i of the N-step flow grid and the warped time
    tau(sigma); lt0: a new field at every step instead (tau unused)."""

    def __init__(self, sch, N, nn, seed, d, K, dev, F=F, dz=0, lt0=False):
        self.seed, self.d, self.K, self.dev, self.F, self.dz, self.lt0 = tuple(seed), d, K, dev, F, dz, lt0
        ts = flow_grid(N)
        self.ell, self.tau = [], []
        for i in range(N):
            t = min(float(ts[i]), 0.999)                        # sigma clamped at t = 0.999, as Field's l_t (LS_MAX)
            lsE = math.log(sch.c * t / (1 - t))
            self.ell.append((1 - t) * sch.lx(lsE) * nn)
            self.tau.append(sch.t(lsE))
        self.i, self.p = None, None

    def params(self, i):
        i = i if self.lt0 else 0
        if self.i != i:
            r = gen(self.dev, *self.seed, i)
            W = torch.randn(self.F, self.d, device=self.dev, generator=r)
            ph = torch.rand(self.F, device=self.dev, generator=r) * 2 * math.pi
            A = torch.randn(self.F, self.K + self.dz, device=self.dev, generator=r) * math.sqrt(2.0 / self.F)
            v = torch.randn(self.F, device=self.dev, generator=r)
            self.i, self.p = i, (W, ph, A, v)
        return self.p

    def __call__(self, x, g, i, dz=0, ls=0.0):
        W, ph, A, v = self.params(i)
        arg = (x.to(W.dtype) @ W.T) / self.ell[i] + ph
        if not self.lt0:
            arg = arg + v * self.tau[i]
        o = torch.cos(arg) @ A
        return o[:, :self.K].to(F64), o[:, self.K:].to(x.dtype) if dz else None


def sched_field(arm, M, N, nn, r):
    sch = Schedule(SCHEDULE, nn, hold=arm == "schedhold")
    return SchedField(sch, N, nn, (SEED_FIELD, r), M.d, M.K, M.dev, dz=M.d, lt0=arm == "schedlt0")


def draw(M, y, s2, x, noise, i):
    """x0_hat at noise s2 (inf: the prior): the posterior mean, plus the M(2) term unless noise is None."""
    g = M.post(y, s2)
    f = 0.0 if math.isinf(s2) else M.s ** 2 / (M.s ** 2 + s2)
    bar = M.combine(g)
    mean = bar if f == 0 else (1 - f) * bar + f * y
    if noise is None:
        return mean
    ls = LS_MAX if math.isinf(s2) else min(0.5 * math.log(s2), LS_MAX)
    xi, z = noise(x, g, i, M.d if M.s else 0, ls)
    c = torch.sqrt(g) * xi
    xh = mean + (1 - f) * (M.combine(c) - bar * c.sum(1).to(M.dt)[:, None])
    return xh + M.s * math.sqrt(1 - f) * z if M.s else xh


def run(M, N, xT, noise, sub=SUB):
    """M(2) on the N-step SD3 grid with xi (and z) from `noise`; noise None is M(1)."""
    ts = flow_grid(N)
    x = xT.clone()
    for i in range(N):
        t, t2 = float(ts[i]), float(ts[i + 1])
        s2 = math.inf if t >= 1 else (t / (1 - t)) ** 2
        a = (t - t2) / t
        for s in range(0, len(x), sub):
            xs = x[s:s + sub]
            xh = draw(M, torch.zeros_like(xs) if math.isinf(s2) else xs / (1 - t), s2, xs, noise, i)
            x[s:s + sub] = xh if t2 == 0 else (1 - a) * xs + a * xh
    return x


# ---------------------------------------------------------------- metrics (vmg.metrics.prdc, on the GPU)

def _sqd(a, b, a2, b2):
    return (a2[:, None] + b2[None] - 2 * a @ b.T).clamp(min=0)


def kth_sqdist(x, k, chunk=1024):
    x2 = (x * x).sum(1)
    out = torch.empty(len(x), device=x.device, dtype=F64)
    for i in range(0, len(x), chunk):
        d = _sqd(x[i:i + chunk], x, x2[i:i + chunk], x2)
        b = torch.arange(len(d), device=x.device)
        d[b, b + i] = -1.0
        out[i:i + len(d)] = d.kthvalue(k + 1, dim=1).values
    return out


def prdc(real, fake, k=KNN, r_real=None, chunk=1024):
    """vmg.metrics.prdc.prdc on the GPU."""
    r_real = kth_sqdist(real, k, chunk) if r_real is None else r_real
    r_fake = kth_sqdist(fake, k, chunk)
    x2, f2 = (real * real).sum(1), (fake * fake).sum(1)
    n, m, dev = len(real), len(fake), real.device
    in_real, balls = torch.zeros(m, dtype=torch.bool, device=dev), torch.zeros(m, dtype=torch.int64, device=dev)
    in_fake = torch.zeros(n, dtype=torch.bool, device=dev)
    nearest = torch.full((n,), math.inf, dtype=F64, device=dev)
    for j in range(0, m, chunk):
        d = _sqd(fake[j:j + chunk], real, f2[j:j + chunk], x2)
        inside = d <= r_real[None]
        in_real[j:j + len(d)] = inside.any(1)
        balls[j:j + len(d)] = inside.sum(1)
        in_fake |= (d <= r_fake[j:j + len(d), None]).any(0)
        nearest = torch.minimum(nearest, d.min(0).values)
    return dict(precision=float(in_real.double().mean()), recall=float(in_fake.double().mean()),
                density=float(balls.sum().double() / (k * m)), coverage=float((nearest <= r_real).double().mean()))


def assign(M, x, chunk=1024):
    """Each row's nearest image and its squared distance."""
    x = x.to(F64)
    x2 = (x * x).sum(1)
    out = [_sqd(x[j:j + chunk], M.X64, x2[j:j + chunk], M.X2).min(1) for j in range(0, len(x), chunk)]
    return torch.cat([o.indices for o in out]), torch.cat([o.values for o in out])


# ---------------------------------------------------------------- units of work

def units():
    """(arm, N, r), most wanted first. Arms: data, m1, or the l_x as a float (0 is vanilla M(2))."""
    u = [("data", 0, r) for r in range(R)]
    u += [(lx, N, r) for N in NS for lx in (0.0,) + LXS for r in range(R)]
    u += [("m1", N, 0) for N in NS]
    u += [(lx, N, r) for N in NS for lx in LXS_WIDE for r in range(R)]
    return u


def units_lt():
    """The l_t sweep: arms are (l_x, l_t); 0.0 is vanilla M(2)."""
    u = [("data", 0, r) for r in range(R)] + [(0.0, N, r) for N in NS for r in range(R)]
    u += [((LX_FIX, lt), N, r) for N in NS for lt in LTS for r in range(R)]
    u += [((lx, 0.0), N, r) for N in NS for lx in LXS + (0.3, 3.0) for r in range(R)]         # the l_x axis at l_t = 0
    u += [((lx, math.inf), N, r) for N in NS for lx in LXS + (0.3, 3.0) for r in range(R)]    # and at l_t = inf
    u += [((lx, 0.1), N, r) for N in NS for lx in LXS + (0.3, 3.0) for r in range(R)]         # and at l_t = 0.1
    u += [((lx, lt), 128, r) for lt in (0.0, 0.1) for lx in (0.07, 0.15, 0.2, 0.4, 0.5, 0.7, 1.5, 2.0, 5.0, 10.0)
          for r in range(3)]                                                                    # denser l_x, N = 128, 3 seeds
    return u + [((lx, 1.0), N, r) for N in NS for lx in LXS + (0.3, 3.0) for r in range(R)]   # and at l_t = 1


def label(arm):
    if isinstance(arm, tuple):
        return f"lx{arm[0]:g}_lt{arm[1]:g}"
    return arm if isinstance(arm, str) else f"lx{arm:g}"


def name(arm, N, r):
    return f"{label(arm)}_N{N}_r{r}"


def claim(d, nm, stale=STALE_S):
    lock = d / f"{nm}.lock"
    for _ in range(2):
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return lock
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime < stale:
                    return None
                lock.unlink()                                   # left by a preempted job
            except FileNotFoundError:
                pass
    return None


def write_json(path, obj):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def load_mix(model, dev):
    z = np.load(out_dir() / "in64-10k.npz")
    u8 = z["u8"][np.unique(z["labels"], return_index=True)[1]] if model == "gauss" else z["u8"]
    M = Mix(normalize(u8.reshape(len(u8), -1)), dev)
    nn = float(kth_sqdist(M.X64, 1).sqrt().median())
    if model == "gauss":
        M.s = S_PAIR * nn / math.sqrt(2 * M.d)
    return M, nn


def pair_ratio(M, x, ids):
    """Within-Gaussian variance about each Gaussian's own sample mean, / s^2 d: 1 for independent draws."""
    x = x.to(F64)
    cnt = torch.bincount(ids, minlength=M.K).to(F64)
    bar = torch.zeros(M.K, M.d, device=M.dev, dtype=F64).index_add_(0, ids, x) / cnt.clamp(min=1)[:, None]
    return float(((x - bar[ids]) ** 2).sum() / (cnt[cnt > 0] - 1).sum() / (M.s ** 2 * M.d))


def one(M, arm, N, r, real, r3, nn, Q=None):
    dev, n = M.dev, N_SAMPLES
    if arm == "data":
        x = M.sample(n, gen(dev, SEED_DATA, r))
    else:
        xT = torch.randn(n, M.d, device=dev, generator=gen(dev, SEED_X))
        lx, lt = arm if isinstance(arm, tuple) else (arm, 0.0)
        noise = (None if arm == "m1" else IID(gen(dev, SEED_IID, N, r)) if arm == 0
                 else Field(lx * nn, (SEED_FIELD, r), M.d, M.K, dev, dz=M.d if M.s else 0, lt=lt))
        x = run(M, N, xT, noise)
    m = prdc(real, x.to(F64), r_real=r3)
    ids, d2 = assign(M, x)
    counts = torch.bincount(ids, minlength=M.K)
    m.update(distinct=float((counts > 0).double().mean()), max_hits=int(counts.max()),
             on_image=float((d2.sqrt() < 0.01 * nn).double().mean()), dist_nn=float((d2.sqrt() / nn).mean()),
             spread=float(d2.mean() / (M.s ** 2 * M.d)) if M.s else 0.0, nan=int(torch.isnan(x).any(1).sum()))
    if Q is not None:
        m["pair_ratio"] = pair_ratio(M, x, ids)
        for k in PROJ:
            q = prdc(real @ Q[:k].T, x.to(F64) @ Q[:k].T)
            m[f"precision_{k}"], m[f"recall_{k}"] = q["precision"], q["recall"]
    return m


def units_big():
    """--sweep big, N = 128 only: data null, vanilla M(2), and the dense l_x axis at l_t = 0 and 0.1."""
    u = [("data", 0, r) for r in range(R_BIG)] + [(0.0, 128, r) for r in range(R_BIG)]
    return u + [((lx, lt), 128, r) for lt in (0.0, 0.1, 1.0) for lx in LXS_DENSE for r in range(R_BIG)]


def units_sched():
    """--sweep sched, N = 128: the three schedule arms (data and vanilla M(2) are big/'s units, same seeds)."""
    return [(arm, 128, r) for arm in SCHED_ARMS for r in range(R_BIG)]


def units_kde():
    """--sweep kde: the same units as units_big (l_x up to 10)."""
    return units_big()


def big_dir(model, n, kde=False, sched=False):
    base = "big-sched" if sched else "big-kde" if kde else "big"
    return out_dir(model) / (base if n == N_BIG else f"{base}-n{n}")


KDE_TOP = 16


def kde_stats(M, real, ids_r, G, ids_g, chunk=1024, sub=128):
    """Each real point under the generated samples blurred by the data's own s, in the full d dimensions:
    ll(x) = log (1/m) sum_j N(x; g_j, s^2 I) + (d/2) log(2 pi s^2), in nats; plus which Gaussian the nearest
    generated sample belongs to. float32 |x|^2 + |g|^2 - 2 x.g is ~2 nats off here (norms ~1e4), so float32 only
    picks each point's KDE_TOP nearest samples and those distances are recomputed exactly in float64; the rest sit
    >~100 nats further out (the within-Gaussian spread of |x - g|^2 / 2 s^2 alone is ~300 nats).
    real: list of float32 chunks; G: (m, d) float32; ids_*: nearest centres, in the same order."""
    dev, s2, m = M.dev, 2 * M.s ** 2, len(G)
    g2 = (G * G).sum(1)
    ll, args, best_all = [], [], []
    for x in real:
        for j in range(0, len(x), chunk):
            xj = x[j:j + chunk]
            x2 = (xj * xj).sum(1)
            tv = torch.full((len(xj), KDE_TOP), math.inf, device=dev)
            ti = torch.zeros((len(xj), KDE_TOP), dtype=torch.long, device=dev)
            for o in range(0, m, 20_000):
                dd = x2[:, None] + g2[None, o:o + 20_000] - 2 * xj @ G[o:o + 20_000].T
                v, i = torch.cat([tv, dd], 1).topk(KDE_TOP, 1, largest=False)
                ti = torch.gather(torch.cat([ti, torch.arange(o, o + dd.shape[1], device=dev).expand_as(dd)], 1), 1, i)
                tv = v
            ex = torch.empty((len(xj), KDE_TOP), dtype=F64, device=dev)
            for b in range(0, len(xj), sub):
                ex[b:b + sub] = ((G[ti[b:b + sub]].double() - xj[b:b + sub, None].double()) ** 2).sum(-1)
            ll.append(torch.logsumexp(-ex / s2, 1) - math.log(m))
            mn, am = ex.min(1)
            args.append(torch.gather(ti, 1, am[:, None])[:, 0])
            best_all.append(mn)
    ll, arg, best = torch.cat(ll), torch.cat(args), torch.cat(best_all)
    q = torch.quantile(ll[torch.randperm(len(ll), device=dev)[:min(len(ll), 2 ** 24)]], torch.tensor(
        [0.01, 0.1, 0.5], dtype=F64, device=dev))
    return dict(kde_ll_mean=float(ll.mean()), kde_ll_q01=float(q[0]), kde_ll_q10=float(q[1]), kde_ll_median=float(q[2]),
                kde_nn_other=float((ids_g[arg] != ids_r).double().mean()),
                kde_nn_d2=float(best.double().mean() / (M.s ** 2 * M.d))), ll


def chunks_of(M, arm, N, r, n, nn):
    """The n samples of one unit, CH_BIG at a time (the field is a function of (x_t, step), so chunking is exact)."""
    if arm == "data":
        rng = gen(M.dev, SEED_DATA, r)
        for s in range(0, n, CH_BIG):
            yield M.sample(min(CH_BIG, n - s), rng)
        return
    lx, lt = arm if isinstance(arm, tuple) else (arm, 0.0) if arm not in SCHED_ARMS else (None, None)
    rx = gen(M.dev, SEED_X)
    noise = (IID(gen(M.dev, SEED_IID, N, r)) if arm == 0 else sched_field(arm, M, N, nn, r) if arm in SCHED_ARMS
             else Field(lx * nn, (SEED_FIELD, r), M.d, M.K, M.dev, dz=M.d, lt=lt))
    for s in range(0, n, CH_BIG):
        yield run(M, N, torch.randn(min(CH_BIG, n - s), M.d, device=M.dev, generator=rx), noise)


def reduce_big(M, xs, Q):
    """Project, assign to the nearest centre and accumulate per-Gaussian sums, one chunk at a time."""
    P, ids, d2 = [], [], []
    S1 = torch.zeros(M.K, M.d, device=M.dev, dtype=F64)
    S2, nan = 0.0, 0
    for x in xs:
        nan += int(torch.isnan(x).any(1).sum())
        x = x.to(F64)
        P.append(x @ Q.T)
        ic = []
        for j in range(0, len(x), 2048):
            xj = x[j:j + 2048]
            m, i = _sqd(xj, M.X64, (xj * xj).sum(1), M.X2).min(1)
            ic.append(i)
            d2.append(m)
        ic = torch.cat(ic)
        ids.append(ic)
        S1.index_add_(0, ic, x)
        S2 += float((x * x).sum())
        del x
    ids, d2 = torch.cat(ids), torch.cat(d2)
    cnt = torch.bincount(ids, minlength=M.K)
    c = cnt.to(F64)
    within = S2 - float(((S1 ** 2).sum(1) / c.clamp(min=1)).sum())
    pr = within / float((c[cnt > 0] - 1).sum()) / (M.s ** 2 * M.d)
    return torch.cat(P), ids, d2, cnt, pr, nan


def balance(cnt, K):
    p = cnt.to(F64) / cnt.sum()
    h = float(-(p[p > 0] * p[p > 0].log()).sum())
    return dict(eff_modes=math.exp(h), eff_modes_frac=math.exp(h) / K, distinct=float((cnt > 0).double().mean()),
                frac_lt3=float((cnt < 3).double().mean()), max_hits=int(cnt.max()),
                tv_uniform=float(0.5 * (p - 1.0 / K).abs().sum()), cnt_std_over_mean=float(cnt.double().std() / cnt.double().mean()))


def work_big(model, n=N_BIG, kde=False, sched=False):
    dev = device()
    assert dev.type == "cuda", "no usable GPU on this node"
    assert model == "gauss"
    torch.backends.cuda.matmul.allow_tf32 = False
    root = big_dir(model, n, kde, sched)
    d = root / "units"
    d.mkdir(parents=True, exist_ok=True)
    M, nn = load_mix(model, dev)
    Q = (torch.randn(max(PROJ), M.d, device=dev, dtype=F64, generator=gen(dev, 51)) / math.sqrt(M.d))[:max(PROJ_BIG)]
    rr = gen(dev, SEED_REAL)
    real_x = [M.sample(min(CH_BIG, n - s), rr) for s in range(0, n, CH_BIG)] if kde else None   # same draws as below
    Pr, ids_r, _, cnt_r, pr_r, _ = reduce_big(
        M, iter(real_x) if kde else (M.sample(min(CH_BIG, n - s), rr) for s in range(0, n, CH_BIG)), Q)
    r3 = {k: kth_sqdist(Pr[:, :k].contiguous(), KNN) for k in PROJ_BIG}
    print(f"{torch.cuda.get_device_name()}: n = {n}, nn {nn:.2f}, s = {M.s:.4f}; real set: pair_ratio {pr_r:.4f}, "
          f"{balance(cnt_r, M.K)}", flush=True)
    extra = {}
    if sched:
        sch = Schedule(SCHEDULE, nn)
        extra = dict(schedule=SCHEDULE, c=sch.c, edm2=sch.table, **{a: dict(ell_flow=f.ell, tau=f.tau) for a, f in
                     ((a, sched_field(a, M, 128, nn, 0)) for a in SCHED_ARMS)})
    write_json(root / "meta.json", dict(nn=nn, s=M.s, K=M.K, d=M.d, n=n, F=F, k=KNN, proj=PROJ_BIG, chunk=CH_BIG,
                                        real_pair_ratio=pr_r, real_balance=balance(cnt_r, M.K), **extra))
    todo = units_sched() if sched else units_kde() if kde else units_big()
    if n != N_BIG:                                                                       # smoke: a few arms
        todo = [u for u in todo if u[2] == 0 and u[0] in ("data", 0.0, (0.03, 0.0), (1.0, 0.0), (1.0, 0.1)) + SCHED_ARMS]
    for arm, N, r in todo:
        nm = name(arm, N, r)
        f = d / f"{nm}.json"
        if f.exists():
            continue
        lock = claim(d, nm, STALE_BIG)
        if lock is None:
            continue
        t0 = time.time()
        xs = list(chunks_of(M, arm, N, r, n, nn)) if kde else chunks_of(M, arm, N, r, n, nn)
        P, ids, d2, cnt, pr, nan = reduce_big(M, iter(xs) if kde else xs, Q)
        t1 = time.time()
        m = dict(pair_ratio=pr, spread=float(d2.mean() / (M.s ** 2 * M.d)), nan=nan, **balance(cnt, M.K))
        if kde:
            G = torch.cat(xs)
            del xs
            km, ll = kde_stats(M, real_x, ids_r, G, ids)
            m.update(km)
            if n <= CH_BIG:                             # smoke: top-k exact log-sum-exp == the direct float64 one
                X, Gd = real_x[0].double(), G.double()           # float64: |x|^2 + |g|^2 - 2 x.g is exact to ~1e-12
                dd = ((X * X).sum(1)[:, None] + (Gd * Gd).sum(1)[None] - 2 * X @ Gd.T).clamp(min=0)
                ref = torch.logsumexp(-dd / (2 * M.s ** 2), 1) - math.log(len(Gd))
                assert torch.allclose(ll, ref, rtol=0, atol=1e-6), float((ll - ref).abs().max())
                m["smoke_kde_maxdiff"] = float((ll - ref).abs().max())
                del dd, Gd
            del G
        if n <= CH_BIG:                                 # smoke runs: the chunked reductions == the 10k-path ones
            x = next(chunks_of(M, arm, N, r, n, nn))
            i2, e2 = assign(M, x)
            assert torch.equal(i2, ids) and torch.allclose(e2, d2), "assign"
            assert abs(pair_ratio(M, x, ids) - pr) < 1e-9, (pair_ratio(M, x, ids), pr)
            assert torch.allclose(x.to(F64) @ Q.T, P), "projection"
            q = prdc(Pr[:, :64].contiguous(), (x.to(F64) @ Q[:64].T))
            assert all(abs(q[k] - prdc(Pr[:, :64].contiguous(), P[:, :64].contiguous(), r_real=r3[64])[k]) < 1e-12 for k in q)
            m["smoke_checks"] = "ok"
        for k in PROJ_BIG:
            q = prdc(Pr[:, :k].contiguous(), P[:, :k].contiguous(), r_real=r3[k])
            m.update({f"{key}_{k}": v for key, v in q.items()})
        write_json(f, dict(arm=label(arm) if isinstance(arm, tuple) else arm, N=N, r=r, n=n,
                           seconds_sample=t1 - t0, seconds=time.time() - t0, **m))
        lock.unlink(missing_ok=True)
        print(f"{nm}: P64 {m['precision_64']:.3f} R64 {m['recall_64']:.3f} C64 {m['coverage_64']:.3f} eff "
              f"{m['eff_modes']:.0f} pair {pr:.3f}" + (f" kde {m['kde_ll_mean']:.1f} other {m['kde_nn_other']:.4f}"
              if kde else "") + f" ({t1 - t0:.0f} + {time.time() - t1:.0f} s)", flush=True)
        del P, ids, d2
    print("no unclaimed units left", flush=True)


def work(model, sweep=None, n=N_BIG):
    if sweep in ("big", "kde", "sched"):
        return work_big(model, n, kde=sweep == "kde", sched=sweep == "sched")
    dev = device()
    assert dev.type == "cuda", "no usable GPU on this node"
    torch.backends.cuda.matmul.allow_tf32 = False
    root = out_dir(model) / "ltsweep" if sweep else out_dir(model)
    d = root / "units"
    d.mkdir(parents=True, exist_ok=True)
    M, nn = load_mix(model, dev)
    Q = torch.randn(max(PROJ), M.d, device=dev, dtype=F64, generator=gen(dev, 51)) / math.sqrt(M.d) if sweep else None
    real = M.X64 if model == "dots" else M.sample(N_SAMPLES, gen(dev, SEED_REAL)).to(F64)
    r3 = kth_sqdist(real, KNN)
    print(f"{torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu'}: {M.K} images, d = {M.d}, median "
          f"nearest-image distance {nn:.2f}, s = {M.s:.4f}", flush=True)
    write_json(root / "meta.json", dict(nn=nn, s=M.s, K=M.K, d=M.d, n=N_SAMPLES, F=F, k=KNN, R=R))
    for arm, N, r in (units_lt() if sweep else units()):
        nm = name(arm, N, r)
        f = d / f"{nm}.json"
        if f.exists():
            continue
        lock = claim(d, nm)
        if lock is None:
            continue
        t0 = time.time()
        m = one(M, arm, N, r, real, r3, nn, Q)
        write_json(f, dict(arm=label(arm) if sweep else arm, N=N, r=r, seconds=time.time() - t0, **m))
        lock.unlink(missing_ok=True)
        print(f"{nm}: P {m['precision']:.3f} R {m['recall']:.3f} distinct {m['distinct']:.3f} spread "
              f"{m['spread']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    print("no unclaimed units left", flush=True)


# ---------------------------------------------------------------- setup, checks, report

def prep():
    """10 images per class, by the recipe of moments_highd.build, from the seed-pair EDM2 zip."""
    import PIL.Image
    f = out_dir() / "in64-10k.npz"
    if f.exists():
        return
    f.parent.mkdir(parents=True, exist_ok=True)
    root = runs.root() / "edm2-64-seedpair" / "img64"
    y = np.load(root / "labels.npy")
    assert y.min() == 0 and y.max() == 999, (y.min(), y.max())
    rng = np.random.default_rng(0)
    rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), PER_CLASS, replace=False) for c in range(1000)]))
    with zipfile.ZipFile(root / "img64.zip") as z:
        ims = [np.asarray(PIL.Image.open(io.BytesIO(z.read(f"{r:08d}"[:5] + f"/img{r:08d}.png")))) for r in rows]
    u8 = np.stack(ims).transpose(0, 3, 1, 2)
    assert u8.shape == (10000, 3, 64, 64) and u8.dtype == np.uint8, (u8.shape, u8.dtype)
    tmp = f.with_name("tmp-in64-10k.npz")
    np.savez(tmp, u8=u8, rows=rows, labels=y[rows])
    os.replace(tmp, f)
    print(f"wrote {f}", flush=True)


def test():
    from vmg.metrics.prdc import prdc as prdc_np
    from reproduce.toys import moments_highd as FH
    dev = torch.device("cpu")
    X = normalize(np.random.default_rng(0).normal(size=(60, 16)))
    M, H = Mix(X, dev), FH.Points({"X": torch.as_tensor(X)}, dev, torch.float32)
    xT = torch.randn(40, 16, generator=gen(dev, 1))
    for N in (1, 4, 16):
        a = FH.run(H, "2", N, xT, gen(dev, 2), 15)
        assert torch.equal(run(M, N, xT, IID(gen(dev, 2)), 15), a), "vanilla is not moments_highd's M(2)"
        assert torch.equal(run(M, N, xT, None, 15), FH.run(H, "1", N, xT, None, 15)), "m1 is not its M(1)"
    d, K, ell = 20, 4000, 2.0
    p = torch.zeros(1, d)
    q = p.clone()
    q[0, 0] = ell
    far = p.clone()
    far[0, 0] = 12 * ell
    A, B, C = [], [], []
    for s in range(12):
        f = Field(ell, (5, s), d, K, dev)
        A.append(f(p, None, 0)[0]); B.append(f(q, None, 0)[0]); C.append(f(far, None, 0)[0])
    A, B, C = torch.cat(A).flatten(), torch.cat(B).flatten(), torch.cat(C).flatten()
    assert abs(float(A.var()) - 1) < 0.03, float(A.var())
    assert abs(float((A * B).mean()) - math.exp(-0.5)) < 0.03, float((A * B).mean())
    assert abs(float((A * C).mean())) < 0.03, float((A * C).mean())
    g = torch.randn(2, K)
    assert not torch.equal(Field(ell, (5, 0), d, K, dev)(p, g, 0)[0], Field(ell, (5, 0), d, K, dev)(p, g, 1)[0])
    a_, b_, c_, e_ = [], [], [], []                             # correlation across noise levels
    for s in range(12):
        f = Field(ell, (6, s), d, K, dev, lt=0.5)
        a_.append(f(p, None, 0, 0, 0.0)[0]); b_.append(f(p, None, 1, 0, 0.4)[0]); c_.append(f(p, None, 2, 0, 5.0)[0])
        h = Field(ell, (6, s), d, K, dev, lt=math.inf)
        e_.append((h(p, None, 0, 0, 0.0)[0] - h(p, None, 7, 0, 3.0)[0]).abs().max())
    a_, b_, c_ = torch.cat(a_).flatten(), torch.cat(b_).flatten(), torch.cat(c_).flatten()
    assert abs(float((a_ * b_).mean()) - math.exp(-0.16 / 0.5)) < 0.03, float((a_ * b_).mean())
    assert abs(float((a_ * c_).mean())) < 0.03 and float(max(e_)) == 0.0
    xi, z = Field(ell, (5, 1), d, 3000, dev, dz=3000)(p, None, 0, 3000)
    assert abs(float(z.var()) - 1) < 0.08 and abs(float((xi.flatten().float() * z.flatten()).mean())) < 0.08
    # the Gaussian-mixture draw has the posterior's mean and covariance (dense fp64 formulas)
    Kg, dg, sg = 5, 6, 0.4
    Xg = torch.randn(Kg, dg, dtype=F64, generator=gen(dev, 8))
    Mg = Mix(Xg, dev, F64, sg)
    x0 = Xg[1] + sg * torch.randn(dg, dtype=F64, generator=gen(dev, 9))
    for s2 in (math.inf, 0.3, 0.02):
        y = torch.zeros(dg, dtype=F64) if math.isinf(s2) else x0 + math.sqrt(s2) * torch.randn(dg, dtype=F64, generator=gen(dev, 10))
        gk = torch.full((Kg,), 1.0 / Kg, dtype=F64) if math.isinf(s2) else torch.softmax(
            -0.5 * ((y - Xg) ** 2).sum(1) / (sg ** 2 + s2), 0)
        fk = 0.0 if math.isinf(s2) else sg ** 2 / (sg ** 2 + s2)
        mk = fk * y + (1 - fk) * Xg
        mean = gk @ mk
        cov = (mk - mean).T @ (gk[:, None] * (mk - mean)) + sg ** 2 * (1 - fk) * torch.eye(dg, dtype=F64)
        Y = y[None].expand(400000, -1)
        Z = draw(Mg, Y, s2, Y, IID(gen(dev, 11)), 0)
        assert float((Z.mean(0) - mean).abs().max()) < 0.01 and float((torch.cov(Z.T, correction=0) - cov).abs().max()) < 0.015
        assert float((draw(Mg, y[None], s2, y[None], None, 0)[0] - mean).abs().max()) < 1e-10
    r = np.random.default_rng(3)
    real, fake = r.normal(size=(300, 8)), r.normal(0.2, 1.1, size=(250, 8))
    fake[:12] = real[:12]
    fake[20:24] = fake[30]                                      # exact duplicates among the samples
    a = prdc_np(real, fake, k=3)
    b = prdc(torch.as_tensor(real), torch.as_tensor(fake), chunk=64)
    ids, d2 = assign(Mix(real, dev), torch.as_tensor(fake), chunk=64)
    assert all(abs(a[k] - b[k]) < 1e-12 for k in a), (a, b)
    assert ids[:12].tolist() == list(range(12)) and float(d2[:12].max()) < 1e-10   # |a|^2 + |b|^2 - 2 a.b rounds
    u = units()
    assert len({name(*x) for x in u}) == len(u)
    assert len({name(*x) for x in units_lt()}) == len(units_lt())
    print("ok: vanilla == M(2) and m1 == M(1) bitwise, field has the kernel, prdc matches vmg.metrics.prdc", flush=True)


def report(model):
    root = out_dir(model)
    rows = [json.loads(f.read_text()) for f in sorted((root / "units").glob("*.json"))]
    meta = json.loads((root / "meta.json").read_text())
    keys = ["precision", "recall", "coverage", "distinct", "on_image", "dist_nn"] + (["spread"] if model == "gauss" else [])
    groups = {}
    for x in rows:
        groups.setdefault((x["N"], str(x["arm"])), []).append(x)
    out = []
    for (N, arm), v in groups.items():
        c = dict(N=N, arm=arm, runs=len(v), nan=sum(x["nan"] for x in v))
        for k in keys:
            a = np.array([x[k] for x in v])
            c[k], c[k + "_se"] = float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
        out.append(c)
    out.sort(key=lambda c: (c["N"], {"data": -2.0, "m1": -1.0}.get(c["arm"], float(c["arm"]) if c["arm"][0].isdigit() else 0)))
    write_json(root / "summary.json", dict(meta=meta, rows=out))
    print(f"{meta['K']} images, median nearest-image distance {meta['nn']:.2f} (l = l_x x that), s = {meta.get('s', 0):.4f}")
    print(f"{'N':>4} {'arm':>6} {'runs':>4} {'precision':>14} {'recall':>14} {'coverage':>14} {'distinct':>7} {'spread':>7}")
    for c in out:
        print(f"{c['N']:>4} {c['arm']:>6} {c['runs']:>4} " + " ".join(
            f"{c[k]:>7.4f}+-{c[k + '_se']:.4f}" for k in keys[:3]) + f" {c['distinct']:>7.3f} {c.get('spread', 0):>7.3f}")


def report_lt(model):
    root = out_dir(model) / "ltsweep"
    rows = [json.loads(f.read_text()) for f in sorted((root / "units").glob("*.json"))]
    groups = {}
    for x in rows:
        groups.setdefault((x["N"], x["arm"]), []).append(x)
    keys = ["precision", "recall", "pair_ratio"] + [f"{m}_{k}" for k in PROJ for m in ("precision", "recall")]
    out = []
    for (N, arm), v in groups.items():
        c = dict(N=N, arm=arm, runs=len(v), nan=sum(x["nan"] for x in v))
        for k in keys:
            a = np.array([x[k] for x in v if k in x])
            c[k], c[k + "_se"] = float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
        out.append(c)
    def rank(arm):
        if arm in ("data", "lx0"):
            return (-1.0, 0.0) if arm == "data" else (0.0, 0.0)
        lx, lt = arm[2:].split("_lt")
        return float(lx), float(lt)
    out.sort(key=lambda c: (c["N"], rank(c["arm"])))
    write_json(root / "summary.json", dict(rows=out))
    print(f"{'N':>4} {'arm':>13} {'runs':>4} {'var lost':>9} {'P':>6} {'R':>6} {'P 64-d':>7} {'R 64-d':>14} {'P 16-d':>7} {'R 16-d':>7}")
    for c in out:
        print(f"{c['N']:>4} {c['arm']:>13} {c['runs']:>4} {100 * (1 - c['pair_ratio']):>8.2f}% {c['precision']:>6.3f} {c['recall']:>6.3f} "
              f"{c['precision_64']:>7.3f} {c['recall_64']:>7.3f}+-{c['recall_64_se']:.3f} {c['precision_16']:>7.3f} {c['recall_16']:>7.3f}")


def diag(model="gauss"):
    """Why recall falls: within-Gaussian geometry of one model per arm, recall at inflated radii and in random
    projections, and a control with a known shared offset per Gaussian (spread stays exactly 1)."""
    dev = device()
    torch.backends.cuda.matmul.allow_tf32 = False
    M, nn = load_mix(model, dev)
    real = M.sample(N_SAMPLES, gen(dev, SEED_REAL)).to(F64)
    r3 = kth_sqdist(real, KNN)
    cs = (1.0, 1.01, 1.02, 1.05, 1.1, 1.2)
    Q = torch.randn(1024, M.d, device=dev, dtype=F64, generator=gen(dev, 51)) / math.sqrt(M.d)

    def stats(x):
        x = x.to(F64)
        ids, d2 = assign(M, x)
        cnt = torch.bincount(ids, minlength=M.K).to(F64)
        bar = torch.zeros(M.K, M.d, device=dev, dtype=F64).index_add_(0, ids, x) / cnt.clamp(min=1)[:, None]
        within = float(((x - bar[ids]) ** 2).sum() / (cnt[cnt > 0] - 1).sum() / (M.s ** 2 * M.d))
        rf = kth_sqdist(x, KNN)
        x2, y2 = (real * real).sum(1), (x * x).sum(1)
        to_real = torch.full((len(real),), math.inf, device=dev, dtype=F64)
        to_fake = torch.full((len(x),), math.inf, device=dev, dtype=F64)
        for j in range(0, len(x), 1024):
            d = _sqd(x[j:j + 1024], real, y2[j:j + 1024], x2)
            to_real = torch.minimum(to_real, (d / rf[j:j + 1024, None]).min(0).values)
            to_fake[j:j + len(d)] = (d / r3[None]).min(1).values
        out = dict(spread=float(d2.mean() / (M.s ** 2 * M.d)), pair_ratio=within,
                   knn_ratio=float((rf.median() / r3.median()).sqrt()),
                   recall={str(c): float((to_real <= c * c).double().mean()) for c in cs},
                   precision={str(c): float((to_fake <= c * c).double().mean()) for c in cs}, proj={})
        for m in (16, 64, 256, 1024):
            q = prdc(real @ Q[:m].T, x @ Q[:m].T)
            out["proj"][str(m)] = dict(precision=q["precision"], recall=q["recall"])
        return out

    res = dict(data=stats(M.sample(N_SAMPLES, gen(dev, SEED_DATA, 0))), runs={}, control={})
    xT = torch.randn(N_SAMPLES, M.d, device=dev, generator=gen(dev, SEED_X))
    for N in NS:
        for lx in (0.0,) + LXS + LXS_WIDE:
            nz = IID(gen(dev, SEED_IID, N, 0)) if lx == 0 else Field(lx * nn, (SEED_FIELD, 0), M.d, M.K, dev, dz=M.d)
            res["runs"][f"N{N}_lx{lx:g}"] = stats(run(M, N, xT, nz))
            print(f"N{N}_lx{lx:g}", json.dumps(res["runs"][f"N{N}_lx{lx:g}"])[:150], flush=True)
    r = gen(dev, 52)
    ids = torch.randint(0, M.K, (N_SAMPLES,), device=dev, generator=r)
    u, e = torch.randn(M.K, M.d, device=dev, generator=r), torch.randn(N_SAMPLES, M.d, device=dev, generator=r)
    for rho in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        res["control"][f"{rho:g}"] = stats(M.X[ids] + M.s * (math.sqrt(rho) * u[ids] + math.sqrt(1 - rho) * e))
    write_json(out_dir(model) / "diag.json", res)
    print("wrote diag.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["setup", "work", "report", "test", "diag"])
    ap.add_argument("--model", choices=["dots", "gauss"], default="dots")
    ap.add_argument("--sweep", choices=["lt", "big", "kde", "sched"], default=None)
    ap.add_argument("--schedule", help="--sweep sched: fig_errfield_lsigma.py's JSON (l_x, l_t along EDM2's sampler)")
    ap.add_argument("--n", type=int, default=N_BIG, help="--sweep big: samples per side (smoke: <= 20000)")
    a = ap.parse_args()
    SCHEDULE = a.schedule
    if a.stage == "setup":
        prep()
        test()
    elif a.stage == "test":
        test()
    elif a.stage == "work":
        work(a.model, a.sweep, a.n)
    elif a.stage == "report" and a.sweep:
        report_lt(a.model)
    else:
        dict(report=report, diag=diag)[a.stage](a.model)
