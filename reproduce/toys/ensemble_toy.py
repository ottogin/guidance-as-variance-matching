#!/usr/bin/env python
"""Guidance with an ensemble of models, on the 2-D mixture with trained MLP denoisers (a side study).

    python reproduce/toys/ensemble_toy.py train   --out DIR       # trains every pool (one GPU, minutes)
    python reproduce/toys/ensemble_toy.py moments --out DIR       # E1: ensemble mean / covariance vs the posterior
    python reproduce/toys/ensemble_toy.py sample  --out DIR       # E2: samplers, metrics against the true mixture

Theory (Sec. 5.2): an ideal denoiser fit to n posterior samples predicts mu + e, e ~ (0, Sigma / n), independent
across retrainings. An ensemble of K such models therefore gives
  (1) a better posterior mean, Dbar = mean_k D^k, error covariance Sigma / (K n);
  (2) an estimate of the posterior covariance's shape, S(x_t) = Cov_k D^k(x_t) ~ Sigma / n;
  (3) guidance without a weak model: Dtilde = Dbar + alpha sum_k z_k (D^k - Dbar), sum of coefficients 1, and with
      alpha set by variance matching, tr Cov_{x_t, z}(Dtilde) = tr Cov(x0). With z redrawn at every call this is the
      second-moment sampler M(2) with a rank-(K-1) covariance estimate; with z frozen it is ordinary (deterministic)
      guidance by a fixed linear combination of the K models.
Data: fig:moments (a)'s mixture, K = 10 Gaussians, s = 0.15, means on a 2 x 1 ellipse.
Pools: G nets per training-set size N, each on its own N points ("indep"), or all on one shared set ("shared").
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
KMIX, AX, AY, S = 10, 2.0, 1.0, 0.15
TH = 2 * math.pi * torch.arange(KMIX) / KMIX
MU = torch.stack([AX * torch.cos(TH), AY * torch.sin(TH)], 1).to(DEV)          # [K, 2]
VAR_X0 = 2 * S * S + float((MU - MU.mean(0)).pow(2).sum(1).mean())            # tr Cov(x0)
SD = math.sqrt(VAR_X0 / 2)
H, FREQS = 128, torch.tensor([0.5, 1, 2, 4, 8, 16.0], device=DEV)
POOLS = [("indep", 128), ("indep", 512), ("indep", 2048), ("indep", 8192), ("shared", 512), ("shared", 2048)]
G = 32
WIDE = True


# ------------------------------------------------------------------ data and exact posterior
def draw(n, gen):
    c = torch.randint(KMIX, (n,), generator=gen, device=DEV)
    return MU[c] + S * torch.randn(n, 2, generator=gen, device=DEV), c


def post(x, sig):
    """Exact posterior of the mixture at VE level sig: (mean [B,2], cov [B,2,2], comp means [B,K,2], weights, v)."""
    v = S * S * sig * sig / (S * S + sig * sig)
    dl = x[:, None] - MU[None]
    m = MU[None] + (S * S / (S * S + sig * sig)) * dl
    g = torch.softmax(-0.5 * dl.pow(2).sum(-1) / (S * S + sig * sig), 1)
    mu = (g[..., None] * m).sum(1)
    dm = m - mu[:, None]
    cov = v * torch.eye(2, device=DEV) + torch.einsum("bk,bki,bkj->bij", g, dm, dm)
    return mu, cov, m, g, v


def post_sample(x, sig, gen):
    _, _, m, g, v = post(x, sig)
    k = torch.multinomial(g, 1, generator=gen)[:, 0]
    return m[torch.arange(len(x)), k] + math.sqrt(v) * torch.randn(len(x), 2, generator=gen, device=DEV)


def logp(x):
    d = x[:, None] - MU[None]
    return torch.logsumexp(-0.5 * d.pow(2).sum(-1) / (S * S), 1) - math.log(KMIX) - math.log(2 * math.pi * S * S)


def emp_post_mean(x, data, sig):
    """Exact posterior mean of each net's training set: x [B,2], data [G,N,2] -> [G,B,2]."""
    d2 = torch.cdist(x[None].expand(len(data), -1, -1), data).pow(2)            # [G,B,N]
    return torch.softmax(-0.5 * d2 / (sig * sig), -1) @ data


# ------------------------------------------------------------------ batched MLP denoisers (EDM preconditioning)
class Pool(torch.nn.Module):
    def __init__(self, g, gen):
        super().__init__()
        dims = [2 + 2 * len(FREQS), H, H, H, H, 2]
        self.W = torch.nn.ParameterList()
        self.b = torch.nn.ParameterList()
        for i, (a, o) in enumerate(zip(dims[:-1], dims[1:])):
            w = torch.randn(g, a, o, generator=gen, device=DEV) * math.sqrt(1.0 / a)
            if i == len(dims) - 2:
                w = w * 0.1
            self.W.append(torch.nn.Parameter(w))
            self.b.append(torch.nn.Parameter(torch.zeros(g, 1, o, device=DEV)))

    def forward(self, x, sig, idx=None):
        """x [G,B,2] (or [B,2], broadcast), sig [G,B] or float -> D [G,B,2]. idx: subset of nets."""
        W = [w if idx is None else w[idx] for w in self.W]
        b = [c if idx is None else c[idx] for c in self.b]
        g = W[0].shape[0]
        if x.dim() == 2:
            x = x[None].expand(g, -1, -1)
        if not torch.is_tensor(sig):
            sig = torch.full(x.shape[:2], float(sig), device=DEV)
        s = sig[..., None]
        cin, cskip = 1 / torch.sqrt(s * s + SD * SD), SD * SD / (s * s + SD * SD)
        cout = s * SD / torch.sqrt(s * s + SD * SD)
        cn = torch.log(s) / 4 * FREQS
        h = torch.cat([x * cin, torch.cos(cn), torch.sin(cn)], -1)
        for i in range(len(W)):
            h = torch.baddbmm(b[i], h, W[i])
            if i < len(W) - 1:
                h = torch.nn.functional.silu(h)
        return cskip * x + cout * h


def train_pool(kind, n, out, steps=40000, batch=512, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(1000 * n + (7 if kind == "shared" else 0) + seed)
    if kind == "indep":
        data = torch.stack([draw(n, gen)[0] for _ in range(G)])
    else:
        data = draw(n, gen)[0][None].expand(G, -1, -1).contiguous()
    pool = Pool(G, gen)
    opt = torch.optim.Adam(pool.parameters(), lr=2e-3, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * min(i / steps, 1.0))))
    t0 = time.time()
    for it in range(steps):
        idx = torch.randint(n, (G, batch), generator=gen, device=DEV)
        x0 = torch.gather(data, 1, idx[..., None].expand(-1, -1, 2))
        sig = torch.exp(-0.6 + 1.6 * torch.randn(G, batch, generator=gen, device=DEV))
        if WIDE:      # half the batch log-uniform on [0.002, 40]: the sampler's whole range is trained
            lu = torch.exp(math.log(0.002) + math.log(40 / 0.002) * torch.rand(G, batch, generator=gen, device=DEV))
            sig = torch.where(torch.rand(G, batch, generator=gen, device=DEV) < 0.5, sig, lu)
        xt = x0 + sig[..., None] * torch.randn(G, batch, 2, generator=gen, device=DEV)
        lam = (sig * sig + SD * SD) / (sig * SD) ** 2
        loss = (lam[..., None] * (pool(xt, sig) - x0).pow(2)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if it % 10000 == 0:
            print(f"  {kind} N={n} it {it} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(state=pool.state_dict(), data=data, kind=kind, n=n), out / f"pool-{kind}-{n}.pt")
    print(f"{kind} N={n}: {time.time() - t0:.0f}s", flush=True)


def load_pool(out, kind, n):
    ck = torch.load(out / f"pool-{kind}-{n}.pt", map_location=DEV)
    p = Pool(G, torch.Generator(device=DEV).manual_seed(0))
    p.load_state_dict(ck["state"])
    p.eval()
    for q in p.parameters():
        q.requires_grad_(False)
    return p, ck["data"]


# ------------------------------------------------------------------ E1: moments at fixed x_t
def eig2(c):
    """Principal axis angle (rad) and aspect ratio sqrt(l1/l2) of [B,2,2] covariances."""
    ev, evec = torch.linalg.eigh(c)
    ang = torch.atan2(evec[:, 1, 1], evec[:, 0, 1])
    return ang, torch.sqrt(ev[:, 1].clamp_min(1e-20) / ev[:, 0].clamp_min(1e-20))


def axis_gap(a, b):
    d = torch.remainder(a - b, math.pi)
    return torch.minimum(d, math.pi - d)


def moments(out):
    gen = torch.Generator(device=DEV).manual_seed(5)
    res = {}
    for kind, n in POOLS:
        pool, data = load_pool(out, kind, n)
        r = {}
        for sig in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
            x0, _ = draw(512, gen)
            x = x0 + sig * torch.randn_like(x0)
            mu, cov, *_ = post(x, sig)
            D = pool(x, sig)                                   # [G,B,2]
            E = emp_post_mean(x, data, sig)                    # exact posterior means of the nets' training sets
            trS = cov.diagonal(dim1=1, dim2=2).sum(-1)         # [B]
            one = ((D - mu) ** 2).sum(-1).mean(0)              # single-net MSE per x
            mse_k = {}
            for K in (1, 2, 4, 8, 16, 32):
                avg = D.reshape(G // K, K, *D.shape[1:]).mean(1)       # disjoint ensembles
                mse_k[K] = float((((avg - mu) ** 2).sum(-1).mean(0) / trS).median())
                if K == 32:
                    bias = float((((avg[0] - mu) ** 2).sum(-1) / trS).median())
            mse_exactk = {K: float((((E.reshape(G // K, K, *E.shape[1:]).mean(1) - mu) ** 2).sum(-1).mean(0) /
                                    trS).median()) for K in (1, 4, 32)}
            Dc = D - D.mean(0)
            Sk = torch.einsum("gbi,gbj->bij", Dc, Dc) / (G - 1)                 # ensemble covariance per x
            Ec = E - E.mean(0)
            Se = torch.einsum("gbi,gbj->bij", Ec, Ec) / (G - 1)
            a_post, ar_post = eig2(cov)
            a_ens, ar_ens = eig2(Sk)
            a_emp, ar_emp = eig2(Se)
            elong = ar_post > 2.0                                              # only where the axis is defined
            trSk = Sk.diagonal(dim1=1, dim2=2).sum(-1)
            # how much of the ensemble spread lies along Sigma's shape: tr(Sk Sigma^-1)/2 vs tr Sk / (tr Sigma/2)
            shape_err = ((Sk / trSk[:, None, None] - cov / trS[:, None, None]) ** 2).sum((1, 2)).sqrt()
            r[str(sig)] = dict(
                mse_single=float((one / trS).median()), mse_K=mse_k, mse_exact_K=mse_exactk, bias32=bias,
                n_eff=float((trS / trSk).median()), n_eff_exact=float((trS / Se.diagonal(dim1=1, dim2=2).sum(-1))
                                                                      .median()),
                frac_elong=float(elong.float().mean()),
                axis_gap_deg=float(torch.rad2deg(axis_gap(a_ens, a_post))[elong].median()) if elong.any() else None,
                axis_gap_exact_deg=float(torch.rad2deg(axis_gap(a_emp, a_post))[elong].median())
                if elong.any() else None,
                aspect_ens_over_post=float((ar_ens / ar_post)[elong].median()) if elong.any() else None,
                aspect_exact_over_post=float((ar_emp / ar_post)[elong].median()) if elong.any() else None,
                shape_err=float(shape_err.median()),
                net_vs_exact=float(((((D - E) ** 2).sum(-1)).mean(0) / trSk).median()))
            print(kind, n, sig, json.dumps(r[str(sig)]), flush=True)
        res[f"{kind}-{n}"] = r
    json.dump(res, open(out / "moments.json", "w"), indent=1)


# ------------------------------------------------------------------ E2: samplers
def karras(steps, smin=0.002, smax=20.0, rho=7.0):
    i = torch.arange(steps, dtype=torch.float64)
    s = (smax ** (1 / rho) + i / (steps - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    return [float(v) for v in s] + [0.0]


def tr_var(x):
    """tr Cov of [B,2] (or [G,B,2] -> per G)."""
    return (x - x.mean(-2, keepdim=True)).pow(2).sum(-1).mean(-1)


def vm_pair(d1, d0):
    """(a, b, c) of the paper's matching condition from calibration predictions d1, d0 [B,2]."""
    a = tr_var(d1 - d0)
    c0, c1 = d0 - d0.mean(0), d1 - d1.mean(0)
    b = tr_var(d0) - (c0 * c1).sum(-1).mean()
    c = tr_var(d0) - VAR_X0
    return ("pair", float(a), float(b), float(c))


def vm_ens(D):
    """alpha for Dbar + alpha sum_k z_k (D^k - Dbar), z ~ N(0, I): matches tr Cov to VAR_X0. D [K,B,2]."""
    Db = D.mean(0)
    spread = (D - Db).pow(2).sum(-1).sum(0).mean()
    return ("ens", float((VAR_X0 - tr_var(Db)).clamp_min(0)), float(spread))


def finalize(stats, wmode):
    """Per-level constants from per-level stats; wmode 'level' (eq. wstar) or 'global' (eq. wconstant: the
    coefficients averaged over the levels)."""
    if stats[0] is None:
        return [None] * len(stats)
    kind = stats[0][0]
    arr = np.array([s[1:] for s in stats])
    if wmode == "global":
        arr = np.broadcast_to(arr.mean(0), arr.shape)
    if kind == "pair":
        a, b, c = arr.T
        return list((b + np.sqrt(np.clip(b * b - a * c, 0, None))) / a)
    num, den = arr.T
    return list(np.sqrt(num / den))


class Sampler:
    """One method: predict(x, i, sig, state) -> x0 prediction. calib(i, sig, xcal) fits its per-level constant."""


WMODE = "level"


def run_method(name, fn_pred, calib, grid, ntraj, gen_seed):
    gen = torch.Generator(device=DEV).manual_seed(gen_seed)
    xcal0, _ = draw(8192, torch.Generator(device=DEV).manual_seed(11))
    ecal = torch.randn(8192, 2, generator=torch.Generator(device=DEV).manual_seed(12), device=DEV)
    consts = finalize([calib(xcal0 + s * ecal, s) for s in grid[:-1]], WMODE)
    x = grid[0] * torch.randn(ntraj, 2, generator=gen, device=DEV)
    for i in range(len(grid) - 1):
        s, sn = grid[i], grid[i + 1]
        d = fn_pred(x, s, consts[i], gen)
        x = (sn / s) * x + (1 - sn / s) * d
    return x, consts


def metrics(x, truth, train=None):
    out = {}
    lp = logp(x)
    out["logp"] = float(lp.mean())
    out["logp_q10"] = float(lp.quantile(0.1))
    k = (x[:, None] - MU[None]).pow(2).sum(-1).argmin(1)
    occ = torch.bincount(k, minlength=KMIX).float() / len(x)
    out["mode_kl"] = float((occ * torch.log(occ.clamp_min(1e-12) * KMIX)).sum())
    out["mode_tv"] = float(0.5 * (occ - 1 / KMIX).abs().sum())
    dist = (x - MU[k]).pow(2).sum(-1).sqrt()
    inl = dist < 1.0                                      # within ~6.7 s of a mean
    out["frac_out"] = float(1 - inl.float().mean())
    out["logp_med"] = float(lp.median())
    out["width"] = float((x - MU[k]).pow(2).sum(-1)[inl].mean() / (2 * S * S))
    out["width_all"] = float((x - MU[k]).pow(2).sum(-1).mean() / (2 * S * S))
    out["var_ratio"] = float(tr_var(x) / VAR_X0)
    # energy distance on 8k vs 8k
    a, b = x[inl][:8192], truth[:8192]
    ed = 2 * torch.cdist(a, b).mean() - torch.cdist(a, a).mean() - torch.cdist(b, b).mean()
    out["energy"] = float(ed)
    for bw in (0.05, 0.3):
        kxx = torch.exp(-torch.cdist(a, a).pow(2) / (2 * bw * bw)).mean()
        kyy = torch.exp(-torch.cdist(b, b).pow(2) / (2 * bw * bw)).mean()
        kxy = torch.exp(-torch.cdist(a, b).pow(2) / (2 * bw * bw)).mean()
        out[f"mmd_{bw}"] = float(kxx + kyy - 2 * kxy)
    if train is not None:
        out["nn_train"] = float(torch.cdist(x[:4096], train).min(1).values.median())
    return out


def sample(out, steps_list=(4, 8, 16, 64), ntraj=16384, reps=6, strong=2048, weak=128, Ks=(2, 4, 8, 16, 32),
           kind="indep", tag=""):
    ps, data_s = load_pool(out, kind, strong)
    pw, data_w = load_pool(out, "indep", weak)
    truth, _ = draw(ntraj, torch.Generator(device=DEV).manual_seed(99))
    res = {}

    def add(key, x, consts, train=None):
        m = metrics(x, truth, train)
        m["const_mean"] = float(np.mean(consts)) if consts and consts[0] is not None else None
        m["const_max"] = float(np.max(consts)) if consts and consts[0] is not None else None
        res.setdefault(key, []).append(m)

    for S_ in steps_list:
        grid = karras(S_)
        none = lambda xc, s: None
        for rep in range(reps):
            sd = 100 * S_ + rep
            gseed = 500 + rep                     # same initial noise across methods within a replicate
            # exact references (only one replicate needed, but cheap)
            add(f"S{S_}/exact-M1", *run_method("M1", lambda x, s, c, g: post(x, s)[0], none, grid, ntraj, gseed))

            def m2(x, s, c, g):
                mu, cov, *_ = post(x, s)
                L = torch.linalg.cholesky(cov + 1e-9 * torch.eye(2, device=DEV))
                return mu + (L @ torch.randn(len(x), 2, 1, generator=g, device=DEV))[..., 0]
            add(f"S{S_}/exact-M2", *run_method("M2", m2, none, grid, ntraj, gseed))
            add(f"S{S_}/exact-Minf", *run_method("Minf", lambda x, s, c, g: post_sample(x, s, g), none, grid,
                                                  ntraj, gseed))
            perm = torch.randperm(G, generator=torch.Generator().manual_seed(sd)).to(DEV)
            i0, i1 = perm[0:1], perm[1:2]
            ds = data_s[perm[0]]
            add(f"S{S_}/single", *run_method("single", lambda x, s, c, g: ps(x, s, i0)[0], none, grid, ntraj, gseed),
                ds)
            # autoguidance: strong <- weak, VM per level
            pair = lambda x, s, i=i0: (ps(x, s, i)[0], pw(x, s, i)[0])
            add(f"S{S_}/AG", *run_method(
                "AG", lambda x, s, c, g: (lambda d1, d0: d0 + c * (d1 - d0))(*pair(x, s)),
                lambda xc, s: vm_pair(*pair(xc, s)), grid, ntraj, gseed), ds)
            # sibling guidance: two strong nets of the same N, no weak model
            sib = lambda x, s: (ps(x, s, i0)[0], ps(x, s, i1)[0])
            add(f"S{S_}/sib", *run_method(
                "sib", lambda x, s, c, g: (lambda d1, d0: d0 + c * (d1 - d0))(*sib(x, s)),
                lambda xc, s: vm_pair(*sib(xc, s)), grid, ntraj, gseed), ds)
            for K in Ks:
                ik = perm[:K]
                add(f"S{S_}/ens{K}", *run_method("ens", lambda x, s, c, g: ps(x, s, ik).mean(0), none, grid, ntraj,
                                                 gseed))
                # AG with ensemble-averaged strong and weak models
                pk = lambda x, s: (ps(x, s, ik).mean(0), pw(x, s, ik).mean(0))
                add(f"S{S_}/AG-ens{K}", *run_method(
                    "AGe", lambda x, s, c, g: (lambda d1, d0: d0 + c * (d1 - d0))(*pk(x, s)),
                    lambda xc, s: vm_pair(*pk(xc, s)), grid, ntraj, gseed))
                # AG: ensemble-mean strong guided by a single weak model
                pk1 = lambda x, s: (ps(x, s, ik).mean(0), pw(x, s, i0)[0])
                add(f"S{S_}/AG-ens{K}-w1", *run_method(
                    "AGe1", lambda x, s, c, g: (lambda d1, d0: d0 + c * (d1 - d0))(*pk1(x, s)),
                    lambda xc, s: vm_pair(*pk1(xc, s)), grid, ntraj, gseed))
                # ensemble guidance, random coefficients: fresh per call / per trajectory / once
                zt = torch.randn(K, ntraj, 1, generator=torch.Generator(device=DEV).manual_seed(sd + 7), device=DEV)
                zg = torch.randn(K, 1, 1, generator=torch.Generator(device=DEV).manual_seed(sd + 8), device=DEV)

                def eg(mode, x, s, c, g):
                    D = ps(x, s, ik)
                    Db = D.mean(0)
                    z = {"fresh": lambda: torch.randn(K, len(x), 1, generator=g, device=DEV),
                         "traj": lambda: zt, "global": lambda: zg}[mode]()
                    return Db + c * (z * (D - Db)).sum(0)
                for mode in ("fresh", "traj", "global"):
                    add(f"S{S_}/EG{K}-{mode}", *run_method(
                        "EG", lambda x, s, c, g, mode=mode: eg(mode, x, s, c, g),
                        lambda xc, s: vm_ens(ps(xc, s, ik)), grid, ntraj, gseed))
                # ensemble guidance from weak nets only (wider spread, same rule)
                add(f"S{S_}/EGweak{K}-fresh", *run_method(
                    "EGw", lambda x, s, c, g: (lambda D: D.mean(0) + c * (torch.randn(K, len(x), 1, generator=g,
                                                                                    device=DEV) * (D - D.mean(0))
                                                                         ).sum(0))(pw(x, s, ik)),
                    lambda xc, s: vm_ens(pw(xc, s, ik)), grid, ntraj, gseed))
                # ensemble-mean with M(2)-style fresh Gaussian noise of the ensemble covariance but scaled by the
                # exact posterior trace ratio: an oracle for the scale (isolates the shape)
            print(f"S={S_} rep {rep} done", flush=True)
        summ = {k: {m: float(np.mean([r[m] for r in v if r[m] is not None])) if v[0][m] is not None else None
                    for m in v[0]} for k, v in res.items() if k.startswith(f"S{S_}/")}
        for k, v in summ.items():
            print(f"{k:28s} " + " ".join(f"{m}={x:.4g}" for m, x in v.items() if x is not None), flush=True)
    json.dump(res, open(out / f"sample{tag}.json", "w"), indent=1)


def sweep(out, steps_list=(8, 16), ntraj=16384, reps=4, strong=2048, weak=128, K=8):
    """Constant-scale sweeps (no VM): best-of-sweep comparison of guidance families."""
    ps, data_s = load_pool(out, "indep", strong)
    pw, _ = load_pool(out, "indep", weak)
    truth, _ = draw(ntraj, torch.Generator(device=DEV).manual_seed(99))
    none = lambda xc, s: None
    res = {}
    for S_ in steps_list:
        grid = karras(S_)
        for rep in range(reps):
            gseed = 500 + rep
            perm = torch.randperm(G, generator=torch.Generator().manual_seed(100 * S_ + rep)).to(DEV)
            i0, i1, ik = perm[0:1], perm[1:2], perm[:K]
            zt = torch.randn(K, ntraj, 1, generator=torch.Generator(device=DEV).manual_seed(rep + 7), device=DEV)
            zg = torch.randn(K, 1, 1, generator=torch.Generator(device=DEV).manual_seed(rep + 8), device=DEV)

            def run(key, fn):
                x, _ = run_method(key, fn, none, grid, ntraj, gseed)
                res.setdefault(f"S{S_}/{key}", []).append(metrics(x, truth))
            run("single", lambda x, s, c, g: ps(x, s, i0)[0])
            run("ens", lambda x, s, c, g: ps(x, s, ik).mean(0))
            for w in (1.25, 1.5, 2.0, 3.0, 5.0):
                run(f"AG-w{w}", lambda x, s, c, g, w=w: (lambda a, b: b + w * (a - b))(ps(x, s, i0)[0], pw(x, s, i0)[0]))
                run(f"sib-w{w}", lambda x, s, c, g, w=w: (lambda a, b: b + w * (a - b))(ps(x, s, i0)[0], ps(x, s, i1)[0]))
                run(f"AGens-w{w}", lambda x, s, c, g, w=w: (lambda a, b: b + w * (a - b))(ps(x, s, ik).mean(0),
                                                                                         pw(x, s, i0)[0]))
            for al in (0.5, 1.0, 2.0, 4.0, 8.0):
                for mode in ("fresh", "traj", "global"):
                    def eg(x, s, c, g, al=al, mode=mode):
                        D = ps(x, s, ik)
                        Db = D.mean(0)
                        z = {"fresh": lambda: torch.randn(K, len(x), 1, generator=g, device=DEV),
                             "traj": lambda: zt, "global": lambda: zg}[mode]()
                        return Db + al * (z * (D - Db)).sum(0)
                    run(f"EG-{mode}-a{al}", eg)
            print(f"S={S_} rep {rep}", flush=True)
        for k, v in res.items():
            if k.startswith(f"S{S_}/"):
                m = {q: float(np.mean([r[q] for r in v])) for q in ("mmd_0.3", "mmd_0.05", "mode_kl", "frac_out",
                                                                     "width", "energy", "logp_med")}
                print(f"{k:24s} " + " ".join(f"{q}={x:.4g}" for q, x in m.items()), flush=True)
    json.dump(res, open(out / "sweep.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "moments", "sample", "sweep"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--strong", type=int, default=2048)
    ap.add_argument("--weak", type=int, default=128)
    ap.add_argument("--kind", default="indep")
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--tag", default="")
    ap.add_argument("--wmode", default="level")
    a = ap.parse_args()
    global WMODE
    WMODE = a.wmode
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        for kind, n in POOLS:
            if not (out / f"pool-{kind}-{n}.pt").exists():
                train_pool(kind, n, out, steps=a.steps)
    elif a.stage == "moments":
        moments(out)
    elif a.stage == "sweep":
        sweep(out)
    else:
        sample(out, strong=a.strong, weak=a.weak, kind=a.kind, reps=a.reps, tag=a.tag)


if __name__ == "__main__":
    main()
