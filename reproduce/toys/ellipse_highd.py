#!/usr/bin/env python
"""fig:ellipse at CIFAR-10 size: does the variance-matched weight put guided predictions where p(x | c) is high?

    python reproduce/toys/ellipse_highd.py test                                   # exactness at small d (CPU, s)
    python reproduce/toys/ellipse_highd.py prep --work W --cifar TAR --cgmm PT    # the two labelled mixtures
    python reproduce/toys/ellipse_highd.py run  --work W --ds clu|cls             # -> W/res/<ds>.json
    python reproduce/toys/ellipse_highd.py sweep --work W --ds clu|cls            # -> W/res/<ds>-sweep.json (after run)

Data (d = 3072, CIFAR-10 training images scaled as in fig:moments): Gaussian mixtures whose components are
probabilistic-PCA Gaussians (the top 32 principal directions, plus the average of the other variances on the rest),
each component carrying a class label.
    clu  the 100-cluster mixture of fig:moments (d), every cluster its own class: as in the 2-D toy, each class is
         one Gaussian.
    cls  the ten CIFAR-10 classes, each a mixture of 10 clusters fit to that class's 5,000 images.
Protocol of fig:ellipse (reproduce/figures/fig_ellipse.py): D_1 = E[x0 | x_t, c] and D_0 = E[x0 | x_t] in closed
form; the matched schedule from vmg.solve on forward-noised calibration points at the 16 levels of the SD 3 Euler
grid (shift 3); a 16-step run guided by that schedule; at every step, for every w on a grid, the log-density of
D_w = D_0 + w (D_1 - D_0) under p(x | c) and under p(x). Along the line D_0 + w (D_1 - D_0) each component's
log-density is a quadratic in w, so the whole grid costs three projections. Results: mean log-density per
coordinate, and the share of D_w above real data's q-quantile of the same log-density (q = 0.5, 0.9, 0.99).
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
from vmg import solve  # noqa: E402

F64 = torch.float64
STEPS = 16
QS = (0.5, 0.9, 0.99)
SUB = 2500                                                   # rows per batch


def flow_grid(N, shift=3.0):
    """VE levels sigma of the SD 3 Euler grid (inf at t = 1) and the flow times t, as in fig_ellipse.py."""
    u = np.linspace(1, 1 / N, N)
    t = shift * u / (1 + (shift - 1) * u)
    with np.errstate(divide="ignore"):
        return np.where(t >= 1, np.inf, t / (1 - t)), t


# ---------------------------------------------------------------- the labelled mixture

class Mix:
    """sum_k pi_k N(mu_k, C_k), C_k = U_k diag(lam_k) U_k^T + psi_k (I - U_k U_k^T), component k in class lab_k."""

    def __init__(self, P):
        self.pi, self.mu, self.U, self.lam, self.psi = (P[k].to(F64) for k in ("pi", "mu", "U", "lam", "psi"))
        self.lab = P["lab"].long()
        self.K, self.d, self.r = self.U.shape
        assert bool((self.psi > 0).all()) and bool((self.lam >= self.psi[:, None] - 1e-12).all())
        self.Ucat = self.U.permute(1, 0, 2).reshape(self.d, self.K * self.r).contiguous()
        self.Umu = torch.einsum("kdr,kd->kr", self.U, self.mu)
        self.mu2 = (self.mu ** 2).sum(1)
        self.C = int(self.lab.max()) + 1
        self.pic = torch.zeros(self.C, dtype=F64).index_add_(0, self.lab, self.pi)
        self.cmean = torch.zeros(self.C, self.d, dtype=F64).index_add_(0, self.lab, self.pi[:, None] * self.mu)
        self.cmean /= self.pic[:, None]
        self.mean = self.pi @ self.mu
        self.logdet0 = torch.log(self.lam).sum(1) + (self.d - self.r) * torch.log(self.psi)

    def proj(self, y):
        """U_k^T (y - mu_k) (n, K, r) and ||y - mu_k||^2 (n, K)."""
        p = (y @ self.Ucat).view(len(y), self.K, self.r) - self.Umu
        d2 = (y ** 2).sum(1, keepdim=True) - 2 * y @ self.mu.T + self.mu2
        return p, d2

    def logits(self, y, s2):
        """log pi_k + log N(y; mu_k, C_k + s2 I), without the shared -d/2 log 2 pi; and the projections."""
        p, d2 = self.proj(y)
        pp = (p ** 2).sum(-1)
        q = ((p ** 2) / (self.lam + s2)).sum(-1) + (d2 - pp) / (self.psi + s2)
        ld = torch.log(self.lam + s2).sum(-1) + (self.d - self.r) * torch.log(self.psi + s2)
        return torch.log(self.pi) - 0.5 * (q + ld), p

    def logp(self, x, c):
        """(log p(x | c), log p(x)), each per sample, without the -d/2 log 2 pi."""
        lg, _ = self.logits(x, 0.0)
        mask = self.lab[None, :] == c[:, None]
        return (torch.logsumexp(lg.masked_fill(~mask, -math.inf), 1) - torch.log(self.pic[c]),
                torch.logsumexp(lg, 1))

    def denoise(self, y, c, s2):
        """D_1 = E[x0 | y, c] and D_0 = E[x0 | y] at VE noise variance s2 (inf: class mean and overall mean)."""
        if math.isinf(s2):
            return self.cmean[c], self.mean.expand(len(y), -1)
        lg, p = self.logits(y, s2)
        f, h = self.psi / (self.psi + s2), self.lam / (self.lam + s2)
        mask = self.lab[None, :] == c[:, None]

        def comb(g):          # sum_k g_k m_k,  m_k = (1 - f_k) mu_k + f_k y + U_k ((h_k - f_k) * p_k)
            out = (g * (1 - f)) @ self.mu + (g @ f)[:, None] * y
            return out + ((g[:, :, None] * (h - f[:, None])[None]) * p).reshape(len(y), -1) @ self.Ucat.T

        return comb(torch.softmax(lg.masked_fill(~mask, -math.inf), 1)), comb(torch.softmax(lg, 1))

    def sample(self, n, gen):
        """(x0, c) from the mixture."""
        k = torch.multinomial(self.pi, n, replacement=True, generator=gen)
        z = torch.randn(n, self.d, dtype=F64, generator=gen)
        x = torch.empty(n, self.d, dtype=F64)
        for j in torch.unique(k).tolist():
            i = torch.nonzero(k == j).flatten()
            zj = z[i]
            a = (torch.sqrt(self.lam[j]) - torch.sqrt(self.psi[j])) * (zj @ self.U[j])
            x[i] = self.mu[j] + torch.sqrt(self.psi[j]) * zj + a @ self.U[j].T
        return x, self.lab[k]

    def line_logp(self, D0, Dl, c, wgrid):
        """log p(D0 + w Dl | c) and log p(D0 + w Dl), each (n, W): every component's log-density is a quadratic
        A + 2 w B + w^2 Q in w, from the projections of D0 and Dl alone."""
        pe, ee = self.proj(D0)
        pd = (Dl @ self.Ucat).view(len(Dl), self.K, self.r)
        ed = (Dl * D0).sum(1, keepdim=True) - Dl @ self.mu.T
        dd = (Dl ** 2).sum(1, keepdim=True)
        il, ip = 1 / self.lam, 1 / self.psi
        A = (pe ** 2 * il).sum(-1) + (ee - (pe ** 2).sum(-1)) * ip
        B = (pe * pd * il).sum(-1) + (ed - (pe * pd).sum(-1)) * ip
        Q = (pd ** 2 * il).sum(-1) + (dd - (pd ** 2).sum(-1)) * ip
        base = torch.log(self.pi) - 0.5 * self.logdet0
        mask = self.lab[None, :] == c[:, None]
        lpc = torch.log(self.pic[c])
        cond, marg = [], []
        for w in wgrid:
            lk = base - 0.5 * (A + 2 * w * B + w * w * Q)
            cond.append(torch.logsumexp(lk.masked_fill(~mask, -math.inf), 1) - lpc)
            marg.append(torch.logsumexp(lk, 1))
        return torch.stack(cond, 1), torch.stack(marg, 1)


def batched(fn, n, *args):
    """fn over row batches of the tensors in args; concatenates each output."""
    outs = [fn(*(a[i:i + SUB] for a in args)) for i in range(0, n, SUB)]
    return tuple(torch.cat([o[j] for o in outs]) for j in range(len(outs[0])))


# ---------------------------------------------------------------- the protocol of fig_ellipse.py

def schedule(mix, sigs, x0, c, eps):
    """The paper's per-level weight (eq. wstar) from forward-noised calibration points, via vmg.solve."""
    d, n = mix.d, len(x0)
    xc = x0 - x0.mean(0)
    dot = lambda u, v: float((u * v).sum(1).mean() / d)                 # noqa: E731   E<u, v> per coordinate
    cdot = lambda u, v: dot(u - u.mean(0), v - v.mean(0))               # noqa: E731
    xr, rr, rrc = [], [], []
    for s in sigs:
        y = x0 + (0.0 if np.isinf(s) else float(s)) * eps
        d1, d0 = batched(lambda yy, cc: mix.denoise(yy, cc, float(s) ** 2), n, y, c)
        r0, r1 = d0 - x0, d1 - x0
        xr.append([dot(xc, r0), dot(xc, r1)])
        rr.append([dot(r0, r0), dot(r0, r1), dot(r1, r1)])
        rrc.append([cdot(r0, r0), cdot(r0, r1), cdot(r1, r1)])
    var_x0 = float(x0.var(0, unbiased=False).mean())
    m = solve.Moments(sigma=sigs, t=sigs, var_x0=var_x0, xr=xr, rr=rr, rr_centred=rrc, n=n, d=d)
    fit = solve.fit_schedule(m)
    return fit["w"], fit, solve.w_mse(m)


def run(mix, args, out):
    t0 = time.time()
    sigs, ts = flow_grid(STEPS)
    gen = torch.Generator().manual_seed(0)
    x0, cc = mix.sample(args.ncal, gen)
    eps = torch.randn(args.ncal, mix.d, dtype=F64, generator=gen)
    ws, fit, wmse = schedule(mix, sigs, x0, cc, eps)
    print(f"matched schedule ({time.time() - t0:.0f} s): " + " ".join(f"{w:.2f}" for w in ws), flush=True)
    print("admissible: " + "".join("y" if a else "n" for a in fit["admissible"]), flush=True)

    real_c, real_m = batched(lambda x, c: mix.logp(x, c), args.ncal, x0, cc)       # the density lines
    thr = {"cond": {q: float(torch.quantile(real_c, q)) for q in QS},
           "marg": {q: float(torch.quantile(real_m, q)) for q in QS}}

    gen = torch.Generator().manual_seed(1)
    _, c = mix.sample(args.n, gen)                                       # labels with the data's p(c)
    z = torch.randn(args.n, mix.d, dtype=F64, generator=gen)
    x = z.clone()
    wmax = max(args.wmax, 1.5 * float(np.max(ws)))
    wgrid = np.round(np.arange(0, wmax + 1e-9, args.dw), 4)
    lv = list(sigs) + [0.0]
    steps = []
    for j in range(STEPS):
        s, sn = float(lv[j]), float(lv[j + 1])
        d1, d0 = batched(lambda yy, c_: mix.denoise(yy, c_, s * s), args.n, x, c)
        lc, lm = batched(lambda a, b, c_: mix.line_logp(a, b, c_, wgrid), args.n, d0, d1 - d0, c)
        rec = {"step": j + 1, "t": float(ts[j]), "sigma": s if np.isfinite(s) else None, "w_matched": float(ws[j]),
               "w_mse": float(wmse[j]),
               "mean_cond": (lc.mean(0) / mix.d).tolist(), "mean_marg": (lm.mean(0) / mix.d).tolist(),
               "share_cond": {str(q): (lc > thr["cond"][q]).double().mean(0).tolist() for q in QS},
               "share_marg": {str(q): (lm > thr["marg"][q]).double().mean(0).tolist() for q in QS}}
        steps.append(rec)
        best = int(np.argmax(rec["mean_cond"]))
        i90 = np.array(rec["share_cond"]["0.9"])
        wm = int(np.argmin(np.abs(wgrid - ws[j])))
        print(f"step {j + 1:2d} t={ts[j]:.3f}: matched w {ws[j]:.2f}; mean log p(D_w|c)/d best at w={wgrid[best]:.2f} "
              f"({rec['mean_cond'][best]:.4f}) vs matched {rec['mean_cond'][wm]:.4f}, D_1 "
              f"{rec['mean_cond'][int(np.argmin(np.abs(wgrid - 1)))]:.4f}, D_0 {rec['mean_cond'][0]:.4f}; "
              f"share>90%: best {100 * i90.max():.1f}% at w={wgrid[int(i90.argmax())]:.2f}, matched {100 * i90[wm]:.1f}%"
              f" ({time.time() - t0:.0f} s)", flush=True)
        dw = d0 + float(ws[j]) * (d1 - d0)
        x = dw + sn * z if np.isinf(s) else (sn / s) * x + (1 - sn / s) * dw

    fc, fm = batched(lambda a, b: mix.logp(a, b), args.n, x, c)            # the final samples themselves
    res = {"ds": args.ds, "d": mix.d, "K": mix.K, "classes": mix.C, "n": args.n, "ncal": args.ncal,
           "sigma": [float(s) if np.isfinite(s) else None for s in sigs], "t": ts.tolist(), "w": [float(w) for w in ws],
           "w_raw": [float(w) for w in fit["raw"]], "admissible": [bool(a) for a in fit["admissible"]],
           "w_mse": [float(w) for w in wmse], "wgrid": wgrid.tolist(),
           "thr": {k: {str(q): v / mix.d for q, v in d.items()} for k, d in thr.items()},
           "real": {"mean_cond": float(real_c.mean() / mix.d), "mean_marg": float(real_m.mean() / mix.d)},
           "final": {"mean_cond": float(fc.mean() / mix.d), "mean_marg": float(fm.mean() / mix.d),
                     "share_cond": {str(q): float((fc > thr["cond"][q]).double().mean()) for q in QS}},
           "steps": steps}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"))
    print(f"wrote {out} ({time.time() - t0:.0f} s)", flush=True)


LAMS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)        # w(t) = 1 + lam (w_matched(t) - 1)
CONSTS = (0.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)                     # constant weights (1.0 is lam = 0)


def class_scores(mix, x):
    """log p(x | c') for every class c' (n, C), without the -d/2 log 2 pi."""
    lg, _ = mix.logits(x, 0.0)
    return torch.stack([torch.logsumexp(lg[:, mix.lab == k], 1) for k in range(mix.C)], 1) - torch.log(mix.pic)


def sweep(mix, a, out):
    """Whole 16-step runs at a family of guidance strengths, same noise and labels for every arm; scores of the
    final samples against the true p(x | c)."""
    t0 = time.time()
    R = json.load(open(os.path.join(a.work, "res", f"{a.ds}.json")))
    ws = np.array(R["w"])
    sigs, _ = flow_grid(STEPS)
    gen = torch.Generator().manual_seed(0)
    x0, cc = mix.sample(a.ncal, gen)                                      # real data: the same draw as `run`
    sc = batched(lambda x: (class_scores(mix, x),), a.ncal, x0)[0]
    real = sc.gather(1, cc[:, None])[:, 0]
    thr = {q: float(torch.quantile(real, q)) for q in QS}
    rq = torch.quantile(real / mix.d, torch.linspace(0.005, 0.995, 199, dtype=F64))
    gen = torch.Generator().manual_seed(1)
    _, c = mix.sample(a.n, gen)
    z = torch.randn(a.n, mix.d, dtype=F64, generator=gen)
    arms = [("scaled", lam, 1 + lam * (ws - 1)) for lam in LAMS] + [("const", w, np.full(STEPS, w)) for w in CONSTS]
    arms = [arm for arm in arms if a.group in ("all", arm[0])]
    lv = list(sigs) + [0.0]
    res = []
    head = {"ds": a.ds, "n": a.n, "real_mean_cond": float(real.mean() / mix.d),
            "thr": {str(q): v / mix.d for q, v in thr.items()}, "w_matched": ws.tolist()}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for kind, val, wt in arms:
        x = z.clone()
        for j in range(STEPS):
            s, sn = float(lv[j]), float(lv[j + 1])
            d1, d0 = batched(lambda yy, c_: mix.denoise(yy, c_, s * s), a.n, x, c)
            dw = d0 + float(wt[j]) * (d1 - d0)
            x = dw + sn * z if np.isinf(s) else (sn / s) * x + (1 - sn / s) * dw
        S = batched(lambda xx: (class_scores(mix, xx),), a.n, x)[0]
        lc = S.gather(1, c[:, None])[:, 0]
        lpost = lc + torch.log(mix.pic[c]) - torch.logsumexp(S + torch.log(mix.pic), 1)
        sq = torch.quantile(lc / mix.d, torch.linspace(0.005, 0.995, 199, dtype=F64))
        rec = {"kind": kind, "value": val, "w": [float(v) for v in wt],
               "mean_cond": float(lc.mean() / mix.d), "sd_cond": float(lc.std() / mix.d),
               "share_cond": {str(q): float((lc > thr[q]).double().mean()) for q in QS},
               "class_acc": float((S.argmax(1) == c).double().mean()), "mean_log_post": float(lpost.mean()),
               "w1_logp": float((sq - rq).abs().mean()),              # W1 between the per-coordinate log-densities
               "var_ratio": float(x.var(0).mean() / x0.var(0).mean())}
        res.append(rec)
        print(f"{kind:6s} {val:5.2f}: mean log p(x|c)/d {rec['mean_cond']:.4f} (data {float(real.mean() / mix.d):.4f}); "
              f"above the data's 50/90/99% lines {100 * rec['share_cond']['0.5']:.1f}/{100 * rec['share_cond']['0.9']:.1f}/"
              f"{100 * rec['share_cond']['0.99']:.1f}%; class acc {100 * rec['class_acc']:.1f}%; W1 {rec['w1_logp']:.4f}; "
              f"var ratio {rec['var_ratio']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        json.dump(head | {"arms": res}, open(out + ".tmp", "w"))         # after every arm: nothing is lost
        os.replace(out + ".tmp", out)
    print(f"wrote {out}", flush=True)


# ---------------------------------------------------------------- building the mixtures

def cifar_u8(tar):
    """CIFAR-10 training images (50000, 3072) uint8 (CHW) and labels, from the binary release, or with no tar from
    vmg.data.cifar10's arrays (python -m vmg.data.cifar10 fetch: the same images)."""
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
            a = np.frombuffer(b, np.uint8).reshape(10000, 3073)
            labs.append(a[:, 0].astype(np.int64)); imgs.append(a[:, 1:])
    X, y = np.concatenate(imgs), np.concatenate(labs)
    assert X.shape == (50000, 3072) and np.bincount(y).tolist() == [5000] * 10
    return X, y


def normalize(X):
    """As fig:moments: [-1, 1], centred per coordinate, unit mean per-coordinate variance."""
    X = X.astype(np.float64) / 127.5 - 1
    X = X - X.mean(0)
    return X / math.sqrt((X ** 2).mean())


def kmeans(X, K, iters=50, seed=0):
    """Lloyd's with k-means++ seeding; returns labels."""
    rng = np.random.default_rng(seed)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    x2 = (Xt ** 2).sum(1)
    C = [Xt[rng.integers(len(X))]]
    dmin = ((Xt - C[0]) ** 2).sum(1)
    for _ in range(1, K):
        p = (dmin / dmin.sum()).double().numpy()
        C.append(Xt[rng.choice(len(X), p=p / p.sum())])
        dmin = torch.minimum(dmin, ((Xt - C[-1]) ** 2).sum(1))
    C = torch.stack(C)
    for _ in range(iters):
        lab = (x2[:, None] - 2 * Xt @ C.T + (C ** 2).sum(1)).argmin(1)
        cnt = torch.bincount(lab, minlength=K)
        newC = torch.zeros_like(C).index_add_(0, lab, Xt) / cnt.clamp(min=1)[:, None].float()
        if torch.equal(newC, C):
            break
        C = newC
    return (x2[:, None] - 2 * Xt @ C.T + (C ** 2).sum(1)).argmin(1).numpy()


def ppca(Xk, r):
    m = Xk.mean(0)
    _, s, Vt = np.linalg.svd(Xk - m, full_matrices=False)
    ev = s ** 2 / len(Xk)
    return m, Vt[:r].T, ev[:r], (ev.sum() - ev[:r].sum()) / (Xk.shape[1] - r)


def build_cls(tar, per_class=10, r=32):
    X, y = cifar_u8(tar)
    X = normalize(X)
    out = {k: [] for k in ("pi", "mu", "U", "lam", "psi", "lab")}
    for c in range(10):
        Xc = X[y == c]
        for seed in range(10):
            lab = kmeans(Xc, per_class, seed=seed)
            if np.bincount(lab, minlength=per_class).min() > 2 * r:
                break
        sizes = np.bincount(lab, minlength=per_class)
        print(f"class {c}: cluster sizes {sizes.min()}..{sizes.max()}", flush=True)
        for k in range(per_class):
            m, U, lam, psi = ppca(Xc[lab == k], r)
            for key, v in zip(("pi", "mu", "U", "lam", "psi", "lab"), (sizes[k] / len(X), m, U, lam, psi, c)):
                out[key].append(v)
    return {k: torch.as_tensor(np.array(v)) for k, v in out.items()}


def prep(a):
    os.makedirs(os.path.join(a.work, "models"), exist_ok=True)
    P = torch.load(a.cgmm, map_location="cpu", weights_only=True)
    P["lab"] = torch.arange(len(P["pi"]))
    torch.save(P, os.path.join(a.work, "models", "clu.pt"))
    print(f"clu: {len(P['pi'])} clusters, each its own class", flush=True)
    fm = os.path.join(a.work, "models", "cls.pt")
    if not os.path.exists(fm):
        torch.save(build_cls(a.cifar), fm)
    print("exit=0", flush=True)


# ---------------------------------------------------------------- exactness at small d

def test():
    """Small-d checks against dense formulas: D_1, D_0, the class and marginal log-densities, and the line."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(3)
    d, K, r = 9, 6, 3
    U = torch.stack([torch.linalg.qr(torch.randn(d, r, dtype=F64, generator=g))[0] for _ in range(K)])
    P = dict(pi=torch.tensor([.1, .2, .15, .25, .2, .1], dtype=F64), mu=torch.randn(K, d, dtype=F64, generator=g),
             U=U, lam=torch.rand(K, r, dtype=F64, generator=g) * 2 + 0.6, psi=torch.rand(K, dtype=F64, generator=g) * .3 + .05,
             lab=torch.tensor([0, 0, 1, 1, 1, 2]))
    M = Mix(P)
    Cs = [U[k] @ torch.diag(P["lam"][k]) @ U[k].T + P["psi"][k] * (torch.eye(d, dtype=F64) - U[k] @ U[k].T)
          for k in range(K)]
    ok = True
    x, c = M.sample(5, g)
    for s2 in (4.0, 0.3, 0.01):
        y = x + math.sqrt(s2) * torch.randn(x.shape, dtype=F64, generator=g)
        d1, d0 = M.denoise(y, c, s2)
        for i in range(len(y)):
            lw, ms = [], []
            for k in range(K):
                A = Cs[k] + s2 * torch.eye(d, dtype=F64)
                lw.append(torch.log(P["pi"][k]) + torch.distributions.MultivariateNormal(P["mu"][k], A).log_prob(y[i]))
                ms.append(P["mu"][k] + Cs[k] @ torch.linalg.solve(A, y[i] - P["mu"][k]))
            lw, ms = torch.stack(lw), torch.stack(ms)
            m0 = torch.softmax(lw, 0) @ ms
            m1 = torch.softmax(lw.masked_fill(P["lab"] != c[i], -math.inf), 0) @ ms
            ok &= bool(torch.allclose(d0[i], m0, atol=1e-9) and torch.allclose(d1[i], m1, atol=1e-9))
    W = [0.0, 0.7, 1.0, 2.5]
    D0, Dl = torch.randn(4, d, dtype=F64, generator=g), torch.randn(4, d, dtype=F64, generator=g)
    lc, lm = M.line_logp(D0, Dl, c[:4], W)
    for j, w in enumerate(W):
        pc, pm = M.logp(D0 + w * Dl, c[:4])
        ok &= bool(torch.allclose(lc[:, j], pc, atol=1e-9) and torch.allclose(lm[:, j], pm, atol=1e-9))
        for i in range(4):                                               # against dense densities
            y = D0[i] + w * Dl[i]
            dens = torch.stack([torch.log(P["pi"][k]) + torch.distributions.MultivariateNormal(P["mu"][k], Cs[k]).log_prob(y)
                                for k in range(K)])
            const = 0.5 * d * math.log(2 * math.pi)
            ok &= abs(float(torch.logsumexp(dens, 0)) + const - float(lm[i, j])) < 1e-8
    xs, cs = M.sample(400000, g)                                          # sampler: moments of one class
    sel = cs == 1
    k = torch.tensor([2, 3, 4])
    wk = P["pi"][k] / P["pi"][k].sum()
    mean = wk @ P["mu"][k]
    ok &= float((xs[sel].mean(0) - mean).abs().max()) < 0.02
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("test", "prep", "run", "sweep"))
    ap.add_argument("--work", default=None)
    ap.add_argument("--ds", choices=("clu", "cls"))
    ap.add_argument("--cifar", help="cifar-10-binary.tar.gz (prep; default vmg.data.cifar10's arrays)")
    ap.add_argument("--cgmm", help="fig:moments' 100-cluster mixture, models/cgmm.pt of reproduce/toys/moments_highd.py (prep)")
    ap.add_argument("--n", type=int, default=20000, help="sampler states")
    ap.add_argument("--ncal", type=int, default=20000, help="calibration points for the schedule")
    ap.add_argument("--wmax", type=float, default=4.0)
    ap.add_argument("--dw", type=float, default=0.01)
    ap.add_argument("--res", default=None, help="output folder (default: <work>/res); sweep reads <work>/res/<ds>.json")
    ap.add_argument("--group", choices=("all", "scaled", "const"), default="all", help="sweep: which arms")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    if a.cmd == "test":
        sys.exit(0 if test() else 1)
    if a.cmd == "prep":
        prep(a)
    else:
        M = Mix(torch.load(os.path.join(a.work, "models", f"{a.ds}.pt"), map_location="cpu", weights_only=True))
        res = a.res or os.path.join(a.work, "res")
        if a.cmd == "run":
            run(M, a, os.path.join(res, f"{a.ds}.json"))
        else:
            sweep(M, a, os.path.join(res, f"{a.ds}-sweep" + ("" if a.group == "all" else f"-{a.group}") + ".json"))
