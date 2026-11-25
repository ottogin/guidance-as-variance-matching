#!/usr/bin/env python
"""fig:moments panel (b): M(1) = Euler vs M(2) vs M(inf) on high-dimensional datasets with exact posteriors.

    python reproduce/toys/moments_highd.py test   [--work W]               # exactness checks (small d, dense fp64)
    python reproduce/toys/moments_highd.py prep   [--work W] [--cifar TAR] [--in64 DIR]   # datasets, frequencies, null
                                                                                         # runs, real-size check (GPU)
    python reproduce/toys/moments_highd.py cell   --ds DS --k K --chunk C  # one sampler x one chunk, every N (GPU, resumable)
    python reproduce/toys/moments_highd.py figure [--work W] [--out PATH]  # MMD per cell -> res/conv_<ds>.json (+ a 1x4 plot)

The paper's panel is in64 (reproduce/figures/fig_moments_row.py reads W/res/conv_in64.json); W defaults to
$VMG_RUNS/_toys/moments_highd.

Datasets (each scaled to unit mean per-coordinate variance; p(x0 | y) exact at every noise level):
    cifar   10,000 CIFAR-10 training images (1,000 per class, d = 3072); the data IS that finite set, so the posterior
            is a softmax over the images.
    cgmm    a Gaussian mixture fit to all 50,000 CIFAR-10 training images: 100 k-means clusters, each a PPCA Gaussian
            (top 32 eigenvalues + the mean of the rest, isotropic), d = 3072.
    subsp   a union of 20 random 8-dimensional subspaces of R^1024, a zero-mean Gaussian on each (no thickness).
    sparse  x = Q s, s_i iid 0.9 N(0, (0.05c)^2) + 0.1 N(0, c^2), Q Haar-orthogonal, d = 1024. The dynamics and the
            isotropic MMD are rotation-equivariant, so this is a product of 1-D mixtures seen in a rotated basis.
    in64    10,000 ImageNet-64 training images (10 per class, d = 12,288, CHW like cifar), treated exactly as cifar:
            the data IS that finite set. Read from img64_train_u8.npy / img64_train_labels.npy in --in64 (default
            $VMG_IMAGENET), which reproduce/experiments/spread_in64.py prep builds from the ImageNet train JPEGs.

Sampler (as reproduce/figures/fig_moments.py): SD3 Euler grid (shift 3), x_t = (1 - t) x0 + t eps, y = x_t / (1 - t),
sigma = t / (1 - t). A step of M(k) is x' = (1 - a) x + a xh with a = (t - t') / t and xh drawn from q_k(. | y):
    q_1 = posterior mean;  q_2 = N(posterior mean, posterior covariance);  q_inf = the posterior itself.
Every posterior is a mixture of Gaussians (degenerate for cifar and subsp), so all three come from one linear
combination of the component posterior means with weights w: w = g (mean), w = sqrt(g) xi (the between-component part
of q_2), w = one-hot k ~ g (q_inf), plus the within-component noise. No d x d matrix is ever formed.
Metric: MMD with a Gaussian kernel, evaluated as E_omega |phi_q - phi_p|^2 on F = 4096 fixed frequencies
omega ~ N(0, I / ell^2), ell = median pairwise data distance; phi_p exact (closed form for every dataset), the same
debiasing as fig_moments.mmd2_from. A second set at ell / 2 is printed as a check. 10^6 samples per cell
(5 chunks of 2e5, chunk-seeded x_T shared by all samplers); the noise floor is from 8 runs of 10^6 exact data draws.
"""
import argparse
import json
import math
import os
import sys
import tarfile
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from reproduce.figures.fig_moments import NS as NS_ALL, flow_grid  # noqa: E402

DSS = ["cifar", "cgmm", "subsp", "sparse", "in64"]   # append only: the index seeds every run
KS = ["1", "2", "inf"]
TITLE = {"cifar": "CIFAR-10 images", "cgmm": "CIFAR-10 mixture", "subsp": "union of subspaces",
         "sparse": "sparse sources", "in64": "ImageNet-64 images"}
POINTS = ("cifar", "in64")                  # finite image sets: the posterior is a softmax over the images
F64 = torch.float64


def cfg(a):
    """Run sizes; the smoke run shrinks them on the command line."""
    return dict(ch=a.ch, nch=a.nch, F=a.F, nnull=a.nnull, ns=[int(s) for s in a.ns.split(",")] if a.ns else NS_ALL)


def gen(dev, *key):
    g = torch.Generator(device=dev)
    g.manual_seed(int(np.random.SeedSequence(list(key)).generate_state(1, np.uint64)[0] >> 1))
    return g


def pick(g, u):
    """Index j ~ g per row from uniforms u."""
    return (torch.cumsum(g, 1) < u[:, None]).sum(1).clamp(max=g.shape[1] - 1)


# ---------------------------------------------------------------- datasets

class GMM:
    """sum_k pi_k N(mu_k, C_k), C_k = U_k diag(lam_k) U_k^T + psi_k (I - U_k U_k^T); U_k (d, r) orthonormal.
    psi_k = 0 is allowed (data on the subspaces). Component posteriors at noise s2: mean
    m_k = mu_k + f_k (y - mu_k) + U_k ((h_k - f_k) * p_k), p_k = U_k^T (y - mu_k), f = psi/(psi+s2), h = lam/(lam+s2);
    covariance a2_k (I - U U^T) + U diag(b2_k) U^T, a2 = psi s2/(psi+s2) <= b2 = lam s2/(lam+s2) (lam >= psi)."""

    def __init__(self, P, dev, dt):
        self.dev, self.dt = dev, dt
        self.pi, self.mu, self.U = P["pi"].to(dev, F64), P["mu"].to(dev, F64), P["U"].to(dev, F64)
        self.lam, self.psi = P["lam"].to(dev, F64), P["psi"].to(dev, F64)
        self.K, self.d, self.r = self.U.shape
        assert bool((self.lam >= self.psi[:, None] - 1e-12).all())
        self.Ucat64 = self.U.permute(1, 0, 2).reshape(self.d, self.K * self.r).contiguous()
        self.Umu = torch.einsum("kdr,kd->kr", self.U, self.mu)
        self.mu2 = (self.mu ** 2).sum(1)
        self.Ucat, self.muw = self.Ucat64.to(dt), self.mu.to(dt)

    def post(self, y, s2):
        """Weights g (n, K) and what combine/noise need. s2 = inf is the prior (y unused)."""
        n = len(y)
        if math.isinf(s2):
            g = self.pi.expand(n, -1)
            z = torch.zeros(self.K, device=self.dev, dtype=F64)
            return g, dict(p=None, y=None, f=z, h=torch.zeros_like(self.lam), a2=self.psi, b2=self.lam)
        lg, p = self.logits(y, s2)
        g = torch.softmax(lg, 1)
        return g, dict(p=p.to(self.dt), y=y, f=self.psi / (self.psi + s2), h=self.lam / (self.lam + s2),
                       a2=self.psi * s2 / (self.psi + s2), b2=self.lam * s2 / (self.lam + s2))

    def logits(self, y, s2):
        """log pi_k + log N(y; mu_k, C_k + s2 I) up to a shared constant, in fp64; and the projections p."""
        n = len(y)
        y64 = y.to(F64)
        p = (y64 @ self.Ucat64).view(n, self.K, self.r) - self.Umu
        d2 = (y64 ** 2).sum(1, keepdim=True) - 2 * y64 @ self.mu.T + self.mu2
        pp = (p ** 2).sum(-1)
        q = ((p ** 2) / (self.lam + s2)).sum(-1) + (d2 - pp) / (self.psi + s2)
        ld = torch.log(self.lam + s2).sum(-1) + (self.d - self.r) * torch.log(self.psi + s2)
        return torch.log(self.pi) - 0.5 * (q + ld), p

    def logp(self, x):
        """Exact data log-density (needs psi > 0)."""
        return torch.logsumexp(self.logits(x, 0.0)[0], 1) - 0.5 * self.d * math.log(2 * math.pi)

    def combine(self, w, S):
        """sum_k w_k m_k for any weights w (n, K) (linear in w)."""
        w64 = w.to(F64)
        out = (w64 * (1 - S["f"])).to(self.dt) @ self.muw
        if S["y"] is not None:
            out = out + (w64 @ S["f"]).to(self.dt)[:, None] * S["y"]
            c = (w64[:, :, None] * (S["h"] - S["f"][:, None])[None]).to(self.dt) * S["p"]
            out = out + c.reshape(len(w), -1) @ self.Ucat.T
        return out

    def noise(self, w, S, rng):
        """A draw with covariance sum_k w_k P_k (w >= 0)."""
        n = len(w)
        w64 = w.to(F64)
        z = torch.randn(n, self.d, device=self.dev, dtype=self.dt, generator=rng)
        eta = torch.randn(n, self.K, self.r, device=self.dev, dtype=self.dt, generator=rng)
        iso = torch.sqrt(w64 @ S["a2"]).to(self.dt)[:, None] * z
        c = (torch.sqrt(w64)[:, :, None] * torch.sqrt((S["b2"] - S["a2"][:, None]).clamp(min=0))[None]).to(self.dt)
        return iso + (c * eta).reshape(n, -1) @ self.Ucat.T

    def sample(self, n, rng):
        g, S = self.post(torch.zeros(n, self.d, device=self.dev), math.inf)
        w = torch.nn.functional.one_hot(pick(g, torch.rand(n, device=self.dev, dtype=F64, generator=rng)),
                                        self.K).to(F64)
        return self.combine(w, S) + self.noise(w, S, rng)

    def charfn(self, W):
        """phi_p(omega) for rows of W (F, d), complex128."""
        W = W.to(self.dev, F64)
        Wu = (W @ self.Ucat64).view(len(W), self.K, self.r)
        quad = self.psi * (W ** 2).sum(1, keepdim=True) + ((self.lam - self.psi[:, None])[None] * Wu ** 2).sum(-1)
        return (self.pi * torch.exp(1j * (W @ self.mu.T) - 0.5 * quad)).sum(1)


class Points:
    """The uniform distribution on a finite set X (N, d): the posterior is a softmax over the points."""

    def __init__(self, P, dev, dt):
        self.dev, self.dt = dev, dt
        self.X64 = P["X"].to(dev, F64)
        self.X = self.X64.to(dt)
        self.K, self.d = self.X.shape
        self.X2 = (self.X64 ** 2).sum(1)

    def post(self, y, s2):
        n = len(y)
        if math.isinf(s2):
            return torch.full((n, self.K), 1.0 / self.K, device=self.dev, dtype=F64), {}
        y64 = y.to(F64)
        lg = -0.5 * ((y64 ** 2).sum(1, keepdim=True) - 2 * y64 @ self.X64.T + self.X2) / s2
        return torch.softmax(lg, 1), {}

    def combine(self, w, S):
        return w.to(self.dt) @ self.X

    def noise(self, w, S, rng):
        return 0.0

    def sample(self, n, rng):
        return self.X[torch.randint(0, self.K, (n,), device=self.dev, generator=rng)]

    def charfn(self, W):
        W = W.to(self.dev, F64)
        out = torch.zeros(len(W), device=self.dev, dtype=torch.complex128)
        for i in range(0, self.K, 2000):
            out += torch.exp(1j * (W @ self.X64[i:i + 2000].T)).sum(1)
        return out / self.K


class Sparse:
    """x = Q s, s_i iid (1 - rho) N(0, v0) + rho N(0, v1). The posterior factorizes over s-coordinates."""

    def __init__(self, P, dev, dt):
        self.dev, self.dt = dev, dt
        self.Q64 = P["Q"].to(dev, F64)
        self.Q = self.Q64.to(dt)
        self.d = self.Q.shape[0]
        self.rho, self.v0, self.v1 = float(P["rho"]), float(P["v0"]), float(P["v1"])

    def coord_post(self, y, s2):
        """Per-coordinate (in s) mixture: weight of the slab g1, component means m0, m1 and variances c0, c1."""
        if math.isinf(s2):
            n = len(y)
            z = torch.zeros(n, self.d, device=self.dev, dtype=F64)
            return z + self.rho, z, z, z + self.v0, z + self.v1
        ys = y.to(F64) @ self.Q64
        l0 = math.log(1 - self.rho) - 0.5 * ys ** 2 / (self.v0 + s2) - 0.5 * math.log(self.v0 + s2)
        l1 = math.log(self.rho) - 0.5 * ys ** 2 / (self.v1 + s2) - 0.5 * math.log(self.v1 + s2)
        g1 = torch.sigmoid(l1 - l0)
        return (g1, self.v0 / (self.v0 + s2) * ys, self.v1 / (self.v1 + s2) * ys,
                torch.full_like(ys, self.v0 * s2 / (self.v0 + s2)), torch.full_like(ys, self.v1 * s2 / (self.v1 + s2)))

    def draw(self, k, y, s2, rng):
        g1, m0, m1, c0, c1 = self.coord_post(y, s2)
        mean = (1 - g1) * m0 + g1 * m1
        if k == "1":
            s = mean
        elif k == "2":
            var = (1 - g1) * (c0 + m0 ** 2) + g1 * (c1 + m1 ** 2) - mean ** 2
            s = mean + torch.sqrt(var.clamp(min=0)) * torch.randn(mean.shape, device=self.dev, dtype=F64, generator=rng)
        else:
            slab = torch.rand(mean.shape, device=self.dev, dtype=F64, generator=rng) < g1
            s = torch.where(slab, m1, m0) + torch.sqrt(torch.where(slab, c1, c0)) * torch.randn(
                mean.shape, device=self.dev, dtype=F64, generator=rng)
        return s.to(self.dt) @ self.Q.T

    def sample(self, n, rng):
        return self.draw("inf", torch.zeros(n, self.d, device=self.dev), math.inf, rng)

    def charfn(self, W):
        Ws = W.to(self.dev, F64) @ self.Q64
        lp = torch.logaddexp(math.log(1 - self.rho) - 0.5 * self.v0 * Ws ** 2, math.log(self.rho) - 0.5 * self.v1 * Ws ** 2)
        return torch.exp(lp.sum(1)).to(torch.complex128)


def draw(M, k, y, s2, rng):
    """xh ~ q_k(. | y) at noise s2 (s2 = inf: the prior)."""
    if isinstance(M, Sparse):
        return M.draw(k, y, s2, rng)
    g, S = M.post(y, s2)
    if k == "1":
        return M.combine(g, S)
    if k == "2":
        mean = M.combine(g, S)
        c = torch.sqrt(g) * torch.randn(g.shape, device=M.dev, dtype=F64, generator=rng)
        between = M.combine(c, S) - mean * c.sum(1).to(M.dt)[:, None]
        return mean + between + M.noise(g, S, rng)
    w = torch.nn.functional.one_hot(pick(g, torch.rand(len(g), device=M.dev, dtype=F64, generator=rng)),
                                    g.shape[1]).to(F64)
    return M.combine(w, S) + M.noise(w, S, rng)


def run(M, k, N, xT, rng, sub):
    """M(k) on the N-step grid, in sub-batches of `sub` rows."""
    ts = flow_grid(N)
    x = xT.clone()
    for i in range(N):
        t, t2 = float(ts[i]), float(ts[i + 1])
        s2 = math.inf if t >= 1 else (t / (1 - t)) ** 2
        a = (t - t2) / t
        for s in range(0, len(x), sub):
            xs = x[s:s + sub]
            y = torch.zeros_like(xs) if math.isinf(s2) else xs / (1 - t)
            xh = draw(M, k, y, s2, rng)
            x[s:s + sub] = xh if t2 == 0 else (1 - a) * xs + a * xh
    return x


# ---------------------------------------------------------------- dataset construction

def cifar_u8(tar):
    """CIFAR-10 training images (50000, 3072) uint8 (CHW order) + labels, from the binary release (no pickle), or
    with no tar from vmg.data.cifar10's arrays (python -m vmg.data.cifar10 fetch: the same images)."""
    if tar is None:
        from vmg.data import cifar10
        X = np.load(cifar10.root() / "cifar10_train_u8.npy")
        y = np.load(cifar10.root() / "cifar10_train_labels.npy").astype(np.int64)
        assert X.shape == (50000, 32, 32, 3), X.shape
        return np.ascontiguousarray(X.transpose(0, 3, 1, 2)).reshape(50000, 3072), y
    imgs, labs = [], []
    with tarfile.open(tar) as t:
        for i in range(1, 6):
            b = t.extractfile(f"cifar-10-batches-bin/data_batch_{i}.bin").read()
            assert len(b) == 10000 * 3073, len(b)
            a = np.frombuffer(b, np.uint8).reshape(10000, 3073)
            labs.append(a[:, 0].astype(np.int64)); imgs.append(a[:, 1:])
    X, y = np.concatenate(imgs), np.concatenate(labs)
    assert X.shape == (50000, 3072) and np.bincount(y).tolist() == [5000] * 10
    return X, y


def in64_u8(root):
    """ImageNet-64 training images (M, 12288) uint8 as a memmap view in HWC order, and labels (M,)."""
    assert root and os.path.exists(os.path.join(root, "img64_train_u8.npy")), (
        f"no img64_train_u8.npy in --in64 {root}: build it with reproduce/experiments/spread_in64.py prep --src "
        "<ILSVRC2012 train> (it writes $VMG_IMAGENET/img64_train_u8.npy and img64_train_labels.npy)")
    X = np.load(os.path.join(root, "img64_train_u8.npy"), mmap_mode="r")
    y = np.load(os.path.join(root, "img64_train_labels.npy"))
    assert X.shape[1:] == (64, 64, 3) and len(y) == len(X) and y.min() == 0 and y.max() == 999, (X.shape, y.shape)
    return X, y


def normalize(X):
    X = X.astype(np.float64) / 127.5 - 1
    X = X - X.mean(0)
    return X / math.sqrt((X ** 2).mean())


def kmeans(X, K, iters, dev, seed=0):
    """Lloyd's with k-means++ seeding (torch, fp32 distances). Returns labels."""
    rng = np.random.default_rng(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    x2 = (Xt ** 2).sum(1)
    C = [Xt[rng.integers(len(X))]]
    dmin = ((Xt - C[0]) ** 2).sum(1)
    for _ in range(1, K):
        p = (dmin / dmin.sum()).double().cpu().numpy()
        C.append(Xt[rng.choice(len(X), p=p / p.sum())])
        dmin = torch.minimum(dmin, ((Xt - C[-1]) ** 2).sum(1))
    C = torch.stack(C)
    for it in range(iters):
        lab = (x2[:, None] - 2 * Xt @ C.T + (C ** 2).sum(1)).argmin(1)
        cnt = torch.bincount(lab, minlength=K)
        newC = torch.zeros_like(C).index_add_(0, lab, Xt) / cnt.clamp(min=1)[:, None].float()
        far = ((Xt - newC[lab]) ** 2).sum(1)
        for j in torch.nonzero(cnt == 0).flatten().tolist():           # empty cluster: reseed at the worst point
            i = int(far.argmax()); newC[j] = Xt[i]; far[i] = 0
        if torch.equal(newC, C):
            break
        C = newC
    return (x2[:, None] - 2 * Xt @ C.T + (C ** 2).sum(1)).argmin(1).cpu().numpy()


def build(ds, tar, dev, kmeans_iters=50, in64=None):
    if ds == "in64":
        X, y = in64_u8(in64)
        rng = np.random.default_rng(0)
        idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), 10, replace=False) for c in range(1000)]))
        return {"X": torch.as_tensor(normalize(np.ascontiguousarray(X[idx]).transpose(0, 3, 1, 2).reshape(len(idx), -1))),
                "rows": torch.as_tensor(idx)}
    if ds == "cifar":
        X, y = cifar_u8(tar)
        rng = np.random.default_rng(0)
        idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), 1000, replace=False) for c in range(10)]))
        return {"X": torch.as_tensor(normalize(X[idx]))}
    if ds == "cgmm":
        X = normalize(cifar_u8(tar)[0])
        r, d = 32, X.shape[1]
        for seed in range(10):                                          # every cluster must support a rank-32 fit
            lab = kmeans(X, 100, kmeans_iters, dev, seed)
            if np.bincount(lab, minlength=100).min() > 2 * r:
                break
            print(f"cgmm: k-means seed {seed} left a cluster of {np.bincount(lab, minlength=100).min()}; reseeding")
        pi, mu, U, lam, psi = [], [], [], [], []
        for k in range(100):
            Xk = X[lab == k]
            assert len(Xk) > 2 * r, (k, len(Xk))
            m = Xk.mean(0)
            _, s, Vt = np.linalg.svd(Xk - m, full_matrices=False)
            ev = s ** 2 / len(Xk)
            pi.append(len(Xk) / len(X)); mu.append(m); U.append(Vt[:r].T); lam.append(ev[:r])
            psi.append((ev.sum() - ev[:r].sum()) / (d - r))
        P = {k: torch.as_tensor(np.array(v)) for k, v in dict(pi=pi, mu=mu, U=U, lam=lam, psi=psi).items()}
        print(f"cgmm: cluster sizes {np.bincount(lab).min()}..{np.bincount(lab).max()}, psi "
              f"{min(psi):.4f}..{max(psi):.4f}, top-32 share of within variance "
              f"{np.mean([l.sum() / (l.sum() + (d - r) * p) for l, p in zip(lam, psi)]):.3f}")
        return P
    if ds == "subsp":
        K, d, r = 20, 1024, 8
        g = torch.Generator().manual_seed(11)
        U = torch.stack([torch.linalg.qr(torch.randn(d, r, dtype=F64, generator=g))[0] for _ in range(K)])
        return dict(pi=torch.full((K,), 1.0 / K, dtype=F64), mu=torch.zeros(K, d, dtype=F64), U=U,
                    lam=torch.full((K, r), d / r, dtype=F64), psi=torch.zeros(K, dtype=F64))
    if ds == "sparse":
        d = 1024
        g = torch.Generator().manual_seed(13)
        Q, R = torch.linalg.qr(torch.randn(d, d, dtype=F64, generator=g))
        Q = Q * torch.sign(torch.diagonal(R))[None]                     # Haar
        c2 = 1 / (0.9 * 0.05 ** 2 + 0.1)
        return dict(Q=Q, rho=torch.tensor(0.1), v0=torch.tensor(0.05 ** 2 * c2), v1=torch.tensor(c2))
    raise ValueError(ds)


def load(ds, work, dev, dt=torch.float32):
    P = torch.load(os.path.join(work, "models", f"{ds}.pt"), map_location="cpu", weights_only=True)
    return {"cifar": Points, "in64": Points, "cgmm": GMM, "subsp": GMM, "sparse": Sparse}[ds](P, dev, dt)


# ---------------------------------------------------------------- MMD

def phisum(x, W, sub=20000):
    """sum_rows exp(i W x) for rows of x; W (F, d)."""
    acc = torch.zeros(len(W), device=x.device, dtype=torch.complex128)
    for i in range(0, len(x), sub):
        P = x[i:i + sub] @ W.T
        acc += torch.complex(torch.cos(P).sum(0, dtype=F64), torch.sin(P).sum(0, dtype=F64))
    return acc


def mmd2(acc, n, php, n_p=None):
    """Debiased MMD^2 on each half of the frequency rows (two kernel widths). n_p: the data side php is itself an
    average of n_p exact draws (debiased too); None: php is exact."""
    phq = acc / n
    e = (phq - php).abs() ** 2 - (1 - phq.abs() ** 2) / (n - 1)
    if n_p is not None:
        e = e - (1 - php.abs() ** 2) / (n_p - 1)
    F = len(e) // 2
    return float(e[:F].mean()), float(e[F:].mean())


# Structure metrics: each dataset is scored where its structure is visible, with an exact data side.
#   cifar   small-kernel MMD to the 10k images, computed exactly by pairs (widths 1/4, 1/2, 1 x the median distance
#           from an image to its nearest other image; 1/2 is the figure's)
#   cgmm    MMD of the standardized exact log-density log p(x) (1-D)
#   subsp   MMD of T(x) = (|U_k^T x|, k = 1..20; distance to the nearest subspace) (21-D)
#   sparse  MMD of the source coordinates Q^T x, pooled (the samplers act on them independently and identically)
# cgmm / subsp / sparse: Gaussian kernel of width 1/2 (figure) and 1 x the median distance between independent data
# features, as F random frequencies per width.
SP_LO, SP_HI, SP_NB = -30.0, 30.0, 24000                               # sparse: histogram of source coordinates


def subsp_T(M, x):
    x64 = x.to(F64)
    e = ((x64 @ M.Ucat64).view(len(x), M.K, M.r) ** 2).sum(-1)
    dist = ((x64 ** 2).sum(1, keepdim=True) - e).clamp(min=0).min(1).values.sqrt()
    return torch.cat([e.sqrt(), dist[:, None]], 1).float()


def feat(ds, M, x, st):
    """Low-dimensional structure features of rows x (cgmm, subsp)."""
    if ds == "cgmm":
        return ((M.logp(x) - st["lp_m"]) / st["lp_s"]).float()[:, None]
    return subsp_T(M, x)


def kpairs(A, B, ells, same, blk=20000):
    """sum of exp(-|a - b|^2 / 2 ell^2) over pairs (a in A, b in B); same=True: A is B, pairs i < j only."""
    out = torch.zeros(len(ells), device=A.device, dtype=F64)
    a2, b2 = (A * A).sum(1), (B * B).sum(1)
    for i in range(0, len(A), blk):
        for j in range(i if same else 0, len(B), blk):
            d2 = (a2[i:i + blk, None] + b2[None, j:j + blk] - 2 * A[i:i + blk] @ B[j:j + blk].T).clamp(min=0)
            for e, l in enumerate(ells):
                Kb = torch.exp(-d2 / (2 * l * l))
                s = Kb.sum(dtype=F64)
                if same and i == j:
                    s = (s - Kb.diagonal().sum(dtype=F64)) / 2
                out[e] += s
    return out


def struct_setup(ds, M, F, dev, key, nd, bt):
    """Frequencies / widths / exact data side of the structure metric; nd exact data draws in batches of bt
    for a sampled data side (4x the samples per cell)."""
    rng = gen(dev, 201, key)
    if ds in POINTS:
        X = M.X
        x2 = (X * X).sum(1)
        nn = []
        for i in range(0, len(X), 2000):
            d2 = (x2[i:i + 2000, None] + x2[None] - 2 * X[i:i + 2000] @ X.T).clamp(min=0)
            d2[torch.arange(len(d2)), torch.arange(i, i + len(d2))] = math.inf
            nn.append(d2.min(1).values.sqrt())
        nnmed = float(torch.cat(nn).median())
        ells = [0.25 * nnmed, 0.5 * nnmed, nnmed]
        Epp = (2 * kpairs(X, X, ells, True) + len(X)) / len(X) ** 2
        return {"ells": torch.tensor(ells), "Epp": Epp.cpu(), "nnmed": nnmed}
    if ds == "sparse":
        s = (M.sample(20000, rng) @ M.Q).flatten()
        a, b = s[torch.randperm(len(s), device=dev, generator=rng)[:200000]], s[:200000]
        med = float((a - b).abs().median())
        Ws = torch.randn(F, device=dev, dtype=F64, generator=rng)
        Ws = torch.cat([Ws / (0.5 * med), Ws / med])
        lp = torch.logaddexp(math.log(1 - M.rho) - 0.5 * M.v0 * Ws ** 2, math.log(M.rho) - 0.5 * M.v1 * Ws ** 2)
        return {"W": Ws.cpu(), "php": torch.exp(lp).to(torch.complex128).cpu(), "med": med}
    st = {}
    if ds == "cgmm":
        lp = torch.cat([M.logp(M.sample(bt, rng)) for _ in range(max(1, nd // (4 * bt)))])
        st = {"lp_m": float(lp.mean()), "lp_s": float(lp.std())}
    Ta, Tb = feat(ds, M, M.sample(20000, rng), st), feat(ds, M, M.sample(20000, rng), st)
    med = float(torch.sqrt(((Ta - Tb) ** 2).sum(1)).median())
    W = torch.randn(F, Ta.shape[1], device=dev, dtype=torch.float32, generator=rng)
    W = torch.cat([W / (0.5 * med), W / med])
    acc, n_p = 0, 0
    for _ in range(max(1, nd // bt)):                                   # data side: nd exact draws
        acc = acc + phisum(feat(ds, M, M.sample(bt, rng), st), W)
        n_p += bt
    st.update({"W": W.cpu(), "php": (acc / n_p).cpu(), "n_p": n_p, "med": med})
    return st


def stats(ds, M, x, fr, dev):
    """Per-chunk sufficient statistics of rows x: generic phi sums + the structure metric's."""
    z = {"acc": phisum(x, fr["W"].to(dev)).cpu(), "n": len(x), "nan": int(torch.isnan(x).any(1).sum())}
    st = fr["st"]
    if ds in POINTS:
        ells = st["ells"].tolist()
        z["kqq"] = kpairs(x, x, ells, True).cpu()
        z["kqp"] = (kpairs(x, M.X, ells, False) / M.K).cpu()
        z["npairs"] = len(x) * (len(x) - 1) // 2
    elif ds == "sparse":
        h = torch.zeros(SP_NB, device=dev, dtype=torch.int64)
        for i in range(0, len(x), 20000):
            s = (x[i:i + 20000] @ M.Q).flatten()
            idx = ((s - SP_LO) / ((SP_HI - SP_LO) / SP_NB)).floor().clamp(0, SP_NB - 1).long()
            h += torch.bincount(idx, minlength=SP_NB)
        z["hist"] = h.cpu()
    else:
        z["acc_st"] = phisum(feat(ds, M, x, st), st["W"].to(dev)).cpu()
    return z


def addd(a, b):
    return b if a is None else {k: a[k] + b[k] for k in a}


def struct_mmd2(ds, z, fr):
    """Structure MMD^2 (figure width first, then the others) from summed statistics z."""
    st = fr["st"]
    if ds in POINTS:
        return [float(v) for v in z["kqq"] / z["npairs"] - 2 * z["kqp"] / z["n"] + st["Epp"]]
    if ds == "sparse":
        w = (SP_HI - SP_LO) / SP_NB
        c = SP_LO + w * (torch.arange(SP_NB, dtype=F64) + 0.5)
        h = z["hist"].to(F64)
        nz = torch.nonzero(h).flatten()
        acc = torch.zeros(len(st["W"]), dtype=torch.complex128)
        for i in range(0, len(nz), 1000):                               # occupied bins only, in blocks
            b = nz[i:i + 1000]
            acc += (h[b][None] * torch.exp(1j * st["W"][:, None] * c[b][None])).sum(1)
        return list(mmd2(acc, float(h.sum()), st["php"]))
    return list(mmd2(z["acc_st"], z["n"], st["php"], st["n_p"]))


# ---------------------------------------------------------------- subcommands

def device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.device("cuda")
    return torch.device("cpu")


def dense_check(M, y, s2, comps):
    """Dense fp64 posterior mean and covariance for a small model; comps = (pi, mu, C) or None for Points."""
    if comps is None:
        X = M.X64
        lg = -0.5 * ((y[None] - X) ** 2).sum(1) / s2
        g = torch.softmax(lg, 0)
        m = g @ X
        return m, ((X - m) * g[:, None]).T @ (X - m)
    pi, mu, C = comps
    d = len(y)
    lg, ms, Ps = [], [], []
    for k in range(len(pi)):
        A = C[k] + s2 * torch.eye(d, dtype=F64)
        dist = torch.distributions.MultivariateNormal(mu[k], covariance_matrix=A)
        lg.append(torch.log(pi[k]) + dist.log_prob(y))
        G = C[k] @ torch.linalg.inv(A)
        ms.append(mu[k] + G @ (y - mu[k])); Ps.append(C[k] - G @ C[k])
    g = torch.softmax(torch.stack(lg), 0)
    m = sum(g[k] * ms[k] for k in range(len(pi)))
    Cov = sum(g[k] * (Ps[k] + torch.outer(ms[k] - m, ms[k] - m)) for k in range(len(pi)))
    return m, Cov


def test():
    """Small-d exactness: mean, q_2 covariance and q_inf moments against dense fp64 formulas; phi_p vs samples."""
    torch.manual_seed(0)
    dev, ok = torch.device("cpu"), True
    gq = torch.Generator().manual_seed(1)
    d, K, r = 10, 4, 3
    U = torch.stack([torch.linalg.qr(torch.randn(d, r, dtype=F64, generator=gq))[0] for _ in range(K)])
    lam = torch.rand(K, r, dtype=F64, generator=gq) * 2 + 0.6
    mu = torch.randn(K, d, dtype=F64, generator=gq)
    pi = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=F64)
    models = []
    for name, psi in [("gmm", torch.tensor([0.05, 0.2, 0.1, 0.5], dtype=F64)), ("gmm-psi0", torch.zeros(K, dtype=F64))]:
        P = dict(pi=pi, mu=mu, U=U, lam=lam, psi=psi)
        C = [U[k] @ torch.diag(lam[k]) @ U[k].T + psi[k] * (torch.eye(d, dtype=F64) - U[k] @ U[k].T) for k in range(K)]
        models.append((name, GMM(P, dev, F64), (pi, mu, C)))
    models.append(("points", Points({"X": torch.randn(40, d, dtype=F64, generator=gq)}, dev, F64), None))
    Qs, Rs = torch.linalg.qr(torch.randn(d, d, dtype=F64, generator=gq))
    sp = Sparse(dict(Q=Qs, rho=torch.tensor(0.2), v0=torch.tensor(0.01), v1=torch.tensor(2.0)), dev, F64)
    n = 200000
    for name, M, comps in models + [("sparse", sp, "sparse")]:
        rng = torch.Generator().manual_seed(5)
        x0 = M.sample(3, rng)
        for s2 in [math.inf, 4.0, 0.3, 0.02, 1e-4]:
            for j in range(2):
                y = torch.zeros(d, dtype=F64) if math.isinf(s2) else x0[j] + math.sqrt(s2) * torch.randn(d, dtype=F64, generator=rng)
                if comps == "sparse":
                    g1, m0, m1, c0, c1 = M.coord_post(y[None], s2)
                    ms = (1 - g1) * m0 + g1 * m1
                    vs = (1 - g1) * (c0 + m0 ** 2) + g1 * (c1 + m1 ** 2) - ms ** 2
                    # brute force: 1-D quadrature per s-coordinate
                    ys = (y @ M.Q64)
                    grid = torch.linspace(-12, 12, 200001, dtype=F64)
                    prior = ((1 - M.rho) * torch.exp(-0.5 * grid ** 2 / M.v0) / math.sqrt(M.v0)
                             + M.rho * torch.exp(-0.5 * grid ** 2 / M.v1) / math.sqrt(M.v1))
                    bm, bv = [], []
                    for i in range(d):
                        wq = prior * (1.0 if math.isinf(s2) else torch.exp(-0.5 * (ys[i] - grid) ** 2 / s2))
                        wq = wq / wq.sum()
                        bm.append((wq * grid).sum()); bv.append((wq * grid ** 2).sum() - bm[-1] ** 2)
                    em = (ms[0] - torch.stack(bm)).abs().max().item()
                    ev = (vs[0] - torch.stack(bv)).abs().max().item() / max(torch.stack(bv).max().item(), 1e-12)
                    m_ex = ms[0] @ M.Q64.T
                    C_ex = M.Q64 @ torch.diag(vs[0]) @ M.Q64.T
                    if s2 > 1e-3:
                        ok &= em < 1e-6 and ev < 1e-4
                elif math.isinf(s2):                                    # prior moments directly
                    if comps is None:
                        m_ex = M.X64.mean(0); C_ex = torch.cov(M.X64.T, correction=0)
                    else:
                        m_ex = sum(pi[k] * mu[k] for k in range(K))
                        C_ex = sum(pi[k] * (comps[2][k] + torch.outer(mu[k] - m_ex, mu[k] - m_ex)) for k in range(K))
                else:
                    m_ex, C_ex = dense_check(M, y, s2, comps)
                Y = y[None].expand(n, -1)
                m1 = draw(M, "1", Y[:2], s2, rng)[0]
                e1 = (m1 - m_ex).abs().max().item() / (1 + m_ex.abs().max().item())
                sc = max(C_ex.diagonal().max().item(), 1e-10)
                res = [f"{name:9s} s2={s2:<7g} mean err {e1:.1e}"]
                ok &= e1 < 1e-8
                # a near one-hot finite-set posterior has a covariance 2e5 exact draws cannot resolve (rare jumps)
                rare = comps is None and not math.isinf(s2) and 1 - M.post(y[None], s2)[0].max().item() < 1e-3
                for k in ["2", "inf"]:
                    Z = draw(M, k, Y, s2, rng)
                    em = ((Z.mean(0) - m_ex).abs().max() / math.sqrt(sc)).item()
                    ec = ((torch.cov(Z.T, correction=0) - C_ex).abs().max() / sc).item()
                    skip = k == "inf" and rare
                    res.append(f"q_{k}: mean {em:.3f} sd, cov {ec:.3f}" + (" (near one-hot: not checked)" if skip else ""))
                    ok &= skip or (em < 0.02 and ec < 0.03)
                if j == 0:
                    print("; ".join(res))
        # phi_p vs empirical
        W = torch.randn(16, d, dtype=F64, generator=rng) * 0.7
        xs = M.sample(1000000, rng)
        emp = torch.exp(1j * (xs @ W.T)).mean(0)
        e = (emp - M.charfn(W)).abs().max().item()
        print(f"{name:9s} phi_p vs 1e6 samples: max |diff| {e:.4f} (MC sd ~ {1 / math.sqrt(1e6):.4f})")
        ok &= e < 5e-3
    print("PASS" if ok else "FAIL", flush=True)
    return ok


def prep(a):
    C = cfg(a)
    dev = device()
    W_ = a.work
    for sub in ["models", "freq", "null", "cells", "res", "fig"]:
        os.makedirs(os.path.join(W_, sub), exist_ok=True)
    for ds in a.dss.split(","):
        t0 = time.time()
        key = DSS.index(ds)
        fm = os.path.join(W_, "models", f"{ds}.pt")
        if not os.path.exists(fm):
            torch.save(build(ds, a.cifar, dev, a.kmeans_iters, a.in64), fm)
        M = load(ds, W_, dev)
        # generic metric: median distance between independent data draws, F frequencies at ell and ell / 2
        rng = gen(dev, 101, key)
        xa, xb = M.sample(20000, rng), M.sample(20000, rng)
        ell = float(torch.sqrt(((xa - xb) ** 2).sum(1)).median())
        tot = float((M.sample(100000, rng).to(F64) ** 2).sum(1).mean() / xa.shape[1])
        fq = os.path.join(W_, "freq", f"{ds}.pt")
        if os.path.exists(fq):                  # resumed: keep it, cells may already be reading these frequencies
            fr = torch.load(fq, weights_only=True)
        else:
            Wf = torch.randn(C["F"], xa.shape[1], device=dev, dtype=F64, generator=gen(dev, 102, key)) / ell
            Wf = torch.cat([Wf, 2 * Wf])
            fr = {"W": Wf.float().cpu(), "php": M.charfn(Wf).cpu(), "ell": ell, "d": int(xa.shape[1])}
            fr["st"] = struct_setup(ds, M, C["F"], dev, key, 4 * C["nch"] * C["ch"], min(C["ch"], 200000))
            torch.save(fr, fq + ".tmp")
            os.replace(fq + ".tmp", fq)
        for r in range(C["nnull"]):                                     # noise floor: exact data draws
            fn = os.path.join(W_, "null", f"{ds}_r{r}.pt")
            if os.path.exists(fn):
                continue
            z, g = None, gen(dev, 31, key, r)
            for c in range(C["nch"]):
                z = addd(z, stats(ds, M, M.sample(C["ch"], g), fr, dev))
            torch.save(z, fn + ".tmp")
            os.replace(fn + ".tmp", fn)
        # real-size precision check: the fp32 path against an all-fp64 model, mean prediction, several noise levels
        M64 = load(ds, W_, dev, F64)
        x0 = M64.sample(2000, gen(dev, 41, key))
        errs = []
        for t in [0.95, 0.6, 0.2, 0.05, float(flow_grid(1024)[-2])]:
            s2 = (t / (1 - t)) ** 2
            y = x0 + math.sqrt(s2) * torch.randn(x0.shape, device=dev, dtype=F64, generator=gen(dev, 42))
            e = (draw(M, "1", y.float(), s2, None).to(F64) - draw(M64, "1", y, s2, None)).abs().max().item()
            errs.append(f"t={t:.4g}: {e:.1e}")
        stt = {k: v for k, v in fr["st"].items() if isinstance(v, float)}
        print(f"{ds}: d={xa.shape[1]} mean per-coord var {tot:.3f} ell {ell:.2f} structure {stt}; "
              f"fp32 vs fp64 posterior mean max|diff| " + ", ".join(errs) + f"  ({time.time() - t0:.0f} s)", flush=True)
    print("exit=0", flush=True)


def cell(a):
    C = cfg(a)
    dev = device()
    ds, k, c = a.ds, a.k, a.chunk
    M = load(ds, a.work, dev)
    fr = torch.load(os.path.join(a.work, "freq", f"{ds}.pt"), weights_only=True)
    od = os.path.join(a.work, "cells", ds)
    os.makedirs(od, exist_ok=True)
    xT = torch.randn(C["ch"], M.d, device=dev, generator=gen(dev, 7, DSS.index(ds), c))
    sub = a.sub or {"cifar": 25000, "cgmm": 25000, "in64": 25000}.get(ds, 100000)
    print(f"{ds} M({k}) chunk {c}: {torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu'}", flush=True)
    for N in C["ns"]:
        f = os.path.join(od, f"k{k}_c{c}_N{N}.pt")
        if os.path.exists(f):
            continue
        t0 = time.time()
        x = run(M, k, N, xT, gen(dev, DSS.index(ds), N, c, KS.index(k)), sub)
        if c == 0:                                                      # raw samples kept for later metrics
            torch.save(x[:a.keep].half().cpu().clone(), os.path.join(od, f"x_k{k}_N{N}.pt"))
        z = stats(ds, M, x, fr, dev)
        torch.save(z, f + ".tmp")
        os.replace(f + ".tmp", f)
        print(f"  N={N}: {time.time() - t0:.1f} s", flush=True)
    print("exit=0", flush=True)


def figure(a):
    C = cfg(a)
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                                "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.5, "axes.labelsize": 7.5,
                                "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5})
    import matplotlib.pyplot as plt

    class Fm:                                     # the figure style, as far as used here
        MUTED = "#6b6a63"

        @staticmethod
        def style(ax):
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(length=2.0, width=0.5)

        @staticmethod
        def title(ax, s):
            ax.set_title(s, fontsize=7, loc="left", pad=2)

    COL = {"1": "#2a78d6", "2": "#eb6834", "inf": "#4b515a"}
    LAB = {"1": "$M(1)$ = Euler", "2": "$M(2)$", "inf": r"$M(\infty)$, posterior draws"}
    dss = a.dss.split(",")
    fig, axs = plt.subplots(1, len(dss), figsize=(1.72 * len(dss), 1.75), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.2, top=0.86, wspace=0.3)
    report = {}
    for j, ds in enumerate(dss):
        fr = torch.load(os.path.join(a.work, "freq", f"{ds}.pt"), weights_only=True)

        def measures(z):
            return list(mmd2(z["acc"], z["n"], fr["php"])) + struct_mmd2(ds, z, fr)

        null = [measures(torch.load(os.path.join(a.work, "null", f"{ds}_r{r}.pt"), weights_only=True))
                for r in range(C["nnull"])]
        nm = len(null[0])                                               # generic ell, ell/2, structure widths...
        floors = [math.sqrt(2 * np.std([v[i] for v in null], ddof=1)) for i in range(nm)]
        vals, missing, nans = {}, [], 0
        for k in KS:
            vals[k] = np.full((len(C["ns"]), nm), math.nan)
            for i, N in enumerate(C["ns"]):
                z = None
                for c in range(C["nch"]):
                    f = os.path.join(a.work, "cells", ds, f"k{k}_c{c}_N{N}.pt")
                    if not os.path.exists(f):
                        missing.append(f"M({k}) N={N} c{c}")
                        z = None
                        break
                    z = addd(z, torch.load(f, weights_only=True))
                if z is not None:
                    nans += z["nan"]
                    vals[k][i] = [math.sqrt(max(v, 0)) for v in measures(z)]
        names = ["generic ell", "generic ell/2"] + (
            [f"small-kernel {w}xNN" for w in (0.25, 0.5, 1)] if ds in POINTS else ["structure 0.5 med", "structure 1 med"])
        prim = 3 if ds in POINTS else 2                                  # the figure's measure
        report[ds] = {"NS": C["ns"], "d": fr["d"], "measures": names, "primary": names[prim], "floors": floors,
                      "mmd": {k: vals[k].T.tolist() for k in KS}, "missing": missing, "nan_rows": nans,
                      "setup": {kk: v for kk, v in fr["st"].items() if isinstance(v, float)} | {"ell": fr["ell"]}}
        json.dump(report[ds], open(os.path.join(a.work, "res", f"conv_{ds}.json"), "w"))
        ax, floor = axs[0, j], floors[prim]
        ax.axhspan(1e-12, floor, color="#ececea", lw=0, zorder=0)
        nsa = np.array(C["ns"])
        for k in KS:
            cur = vals[k][:, prim]
            y = np.maximum(cur, floor)
            hol = cur <= floor
            z_ = 2.5 if k == "inf" else 3
            ax.plot(nsa, y, color=COL[k], lw=1.2, ls=(0, (3, 1.6)) if k == "inf" else "-", zorder=z_, label=LAB[k])
            ax.plot(nsa[~hol], y[~hol], "o", ms=2.4, color=COL[k], mec="white", mew=0.4, zorder=z_ + 1)
            ax.plot(nsa[hol], y[hol], "o", ms=2.4, mfc="white", mec=COL[k], mew=0.7, zorder=z_ + 1)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ticks = [t for t in [1, 4, 16, 64, 256, 1024] if t <= max(C["ns"])]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlim(0.8, max(C["ns"]) * 1.27)
        top = np.nanmax([np.nanmax(vals[k][:, prim]) for k in KS] + [floor * 2])
        ax.set_ylim(floor * 0.5, top * 1.6)
        ax.set_xlabel("sampling steps $N$")
        if j == 0:
            ax.set_ylabel("MMD to the data", labelpad=1)
            ax.text(1.05, floor * 0.7, "noise floor", color=Fm.MUTED, fontsize=5.8, va="center")
        Fm.style(ax)
        ax.grid(False, which="minor")
        Fm.title(ax, f"({'abcd'[j]}) {TITLE[ds]}, $d={fr['d']}$")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03), handlelength=1.8,
               columnspacing=1.2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out + ".pdf", dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out + ".png", dpi=260, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}.pdf/.png")

    def eq_euler(ys, ns, v):
        for i in range(len(ns) - 1):
            if ys[i] >= v >= ys[i + 1]:
                f = (math.log(v) - math.log(ys[i])) / (math.log(ys[i + 1]) - math.log(ys[i]))
                return math.exp(math.log(ns[i]) + f * (math.log(ns[i + 1]) - math.log(ns[i])))
        return math.nan

    for ds, R in report.items():
        ns = R["NS"]
        print(f"\n== {ds} (d={R['d']}) setup {R['setup']}"
              + (f"; MISSING {len(R['missing'])} chunks: {R['missing'][:4]}" if R["missing"] else "")
              + (f"; NaN rows {R['nan_rows']}" if R["nan_rows"] else ""))
        for m, name in enumerate(R["measures"]):
            fl = R["floors"][m]
            print(f"  [{name}{' *figure*' if name == R['primary'] else ''}] floor {fl:.2e}")
            for k in KS:
                ys = np.array(R["mmd"][k][m])
                hit = next((N for N, v in zip(ns, ys) if v <= fl), None)
                print(f"    M({k:3s}) " + " ".join(f"{N}:{v:.2e}" for N, v in zip(ns, ys) if N in (1, 4, 8, 16, 32, 64, 256, 1024))
                      + f"  first at floor N={hit}")
            e = np.array(R["mmd"]["1"][m])
            print("    Euler-equivalent steps of M(2): " + ", ".join(
                f"N={N}: {eq_euler(e, ns, np.array(R['mmd']['2'][m])[ns.index(N)]):.0f}" for N in (4, 8, 16, 32, 64) if N in ns))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["test", "prep", "cell", "figure"])
    ap.add_argument("--work", default=os.environ.get("MKHD") or os.path.join(
        os.environ.get("VMG_RUNS", os.path.join(REPO, "runs")), "_toys", "moments_highd"))
    ap.add_argument("--cifar", default=None, help="cifar-10-binary.tar.gz (prep; default vmg.data.cifar10's arrays)")
    ap.add_argument("--in64", default=os.environ.get("VMG_IMAGENET"),
                    help="directory with img64_train_u8.npy and img64_train_labels.npy (prep)")
    ap.add_argument("--dss", default=",".join(DSS))
    ap.add_argument("--ds", choices=DSS)
    ap.add_argument("--k", choices=KS)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--ch", type=int, default=200000)
    ap.add_argument("--nch", type=int, default=5)
    ap.add_argument("--F", type=int, default=4096)
    ap.add_argument("--nnull", type=int, default=8)
    ap.add_argument("--ns", default=None, help="comma list of step counts (default: fig_moments.NS)")
    ap.add_argument("--sub", type=int, default=None)
    ap.add_argument("--keep", type=int, default=10000, help="raw samples saved per cell (chunk 0, fp16)")
    ap.add_argument("--kmeans-iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(a.work, "fig", "moment-samplers-highd")
    if a.cmd == "test":
        sys.exit(0 if test() else 1)
    {"prep": prep, "cell": cell, "figure": figure}[a.cmd](a)
