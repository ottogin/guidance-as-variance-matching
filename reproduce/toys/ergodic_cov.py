#!/usr/bin/env python
"""Test of eq. (16), first line: the covariance of ONE guided pair over inputs, Cov_{x_t}(D_w(x_t)), against the
covariance over inputs AND retrainings, Cov_{p, x_t}(D_w), on the 2-D mixture of fig:story with trained MLPs.

    python reproduce/toys/ergodic_cov.py train   --out DIR --n 128     # G nets on their own N points (one GPU)
    python reproduce/toys/ergodic_cov.py measure --out DIR             # every pool pair, every noise level
    python reproduce/toys/ergodic_cov.py matrices --out DIR            # per pool, the 2x2 matrices (fig:ergodic-gap)

Pools: G nets per training-set size N, each net on its own N points and seed (ensemble_toy.Pool, EDM
preconditioning, 40k Adam steps). A "pair" is weak net g of pool N0 with strong net g of pool N1 (independent).

At each noise level sigma the same B inputs x_t = x0 + sigma eps (x0 ~ p_data) go to every net. For each pair g the
2x2 blocks C00, C01, C11 of the covariance over x_t of (D0^g, D1^g) give Cov_x(D_w^g) for any w in closed form:
    Cov_x(D_w^g) = (1-w)^2 C00 + w(1-w)(C01 + C10) + w^2 C11.
Pooled over g and x_t (law of total covariance, exact for a shared x_t set):
    Cov_{p,x}(D_w) = mean_g Cov_x(D_w^g) + Cov_g(mean_x D_w^g),
so the gap between the two sides of the first line is the variance across retrainings of each net's input-averaged
prediction (a shared offset), plus the scatter of single pairs about their mean. Also recorded: the two terms of
the second line, Cov_x(E_p[D_w|x_t]) and E_x Cov_p(D_w|x_t), and the third line's Cov(E[x0|x_t]), E Cov(x0|x_t)
from the exact posterior. Weights solved per level (positive root):
    w_pair[g] : tr Cov_x(D_w^g)        = tr Cov(x0)        (VMG, one pair, eq. 17)
    w_pool    : tr Cov_{p,x}(D_w)      = tr Cov(x0)        (the right side of the first line)
    w_post    : E_x tr Cov_p(D_w|x_t)  = E tr Cov(x0|x_t)  (eq. 15's c(w) = 1: retraining spread = posterior's)
The target tr Cov(x0) is the in-sample one of the x0 that made x_t, which removes most finite-B noise (it is shared).
Output: DIR/ergodic.npz and DIR/ergodic.json (measure), DIR/ergodic-mat.npz (matrices).
"""
import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ensemble_toy as et  # noqa: E402

DEV = et.DEV
SIGMAS = [float(s) for s in np.geomspace(0.005, 20.0, 16)]


def load(out, n):
    ck = torch.load(out / f"pool-indep-{n}.pt", map_location=DEV)
    g = ck["data"].shape[0]
    p = et.Pool(g, torch.Generator(device=DEV).manual_seed(0))
    p.load_state_dict(ck["state"])
    p.eval()
    for q in p.parameters():
        q.requires_grad_(False)
    return p, g


@torch.no_grad()
def predict(pool, x, sig, chunk=8192):
    return torch.cat([pool(x[i:i + chunk], sig) for i in range(0, len(x), chunk)], 1).double()   # [G,B,2]


def roots(a2, a1, a0):
    """Positive root of a2 w^2 + a1 w + a0 = 0 (nan if none)."""
    disc = a1 * a1 - 4 * a2 * a0
    r = (-a1 + np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * a2)
    return r


def quad(t00, t01, t11):
    """tr of (1-w)^2 C00 + 2w(1-w) C01s + w^2 C11 as coefficients (a2, a1, a0) of w^2, w, 1."""
    return t00 - 2 * t01 + t11, -2 * t00 + 2 * t01, t00


def blocks(D0, D1):
    """Per-net covariance over x of (D0, D1): C00, C01 (symmetrized), C11 [G,2,2] and means [G,2]."""
    B = D0.shape[1]
    m0, m1 = D0.mean(1), D1.mean(1)
    c0, c1 = D0 - m0[:, None], D1 - m1[:, None]
    C00 = torch.einsum("gbi,gbj->gij", c0, c0) / (B - 1)
    C11 = torch.einsum("gbi,gbj->gij", c1, c1) / (B - 1)
    C01 = torch.einsum("gbi,gbj->gij", c0, c1) / (B - 1)
    return C00, 0.5 * (C01 + C01.transpose(1, 2)), C11, m0, m1


def measure(out, B, seed):
    ns = sorted(int(p.stem.split("-")[-1]) for p in out.glob("pool-indep-*.pt"))
    pools = {n: load(out, n) for n in ns}
    G = min(g for _, g in pools.values())
    pairs = list(itertools.combinations(ns, 2))
    print(f"pools {ns}, G = {G}, B = {B}, pairs {pairs}", flush=True)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    res = {f"{a}-{b}": {} for a, b in pairs}
    for sig in SIGMAS:
        t0 = time.time()
        x0, _ = et.draw(B, gen)
        x0 = x0.double()
        xt = (x0 + sig * torch.randn(B, 2, generator=gen, device=DEV, dtype=torch.float64))
        mu, cov, *_ = et.post(xt.float(), sig)
        mu, cov = mu.double(), cov.double()
        tr_x0 = float(torch.trace(torch.cov(x0.T)))
        tr_mu = float(torch.trace(torch.cov(mu.T)))
        tr_post = float(cov.diagonal(dim1=1, dim2=2).sum(-1).mean())
        preds = {n: predict(pools[n][0], xt.float(), sig)[:G] for n in ns}
        for a, b in pairs:
            D0, D1 = preds[a], preds[b]
            C00, C01, C11, m0, m1 = blocks(D0, D1)
            tr = lambda C: C.diagonal(dim1=1, dim2=2).sum(-1)                                 # noqa: E731
            t00, t01, t11 = (tr(C).cpu().numpy() for C in (C00, C01, C11))                  # per pair [G]
            # pooled: mean over g of per-pair covariances + covariance over g of per-pair means
            dm0, dm1 = m0 - m0.mean(0), m1 - m1.mean(0)
            o00 = float((dm0 * dm0).sum(-1).sum() / (G - 1))
            o11 = float((dm1 * dm1).sum(-1).sum() / (G - 1))
            o01 = float((dm0 * dm1).sum(-1).sum() / (G - 1))
            P = [t.mean() * (B - 1) / B + o * (G - 1) / G for t, o in ((t00, o00), (t01, o01), (t11, o11))]
            # second line: across-net covariance at each x (unbiased), and covariance over x of the net-mean
            E0, E1 = D0.mean(0), D1.mean(0)
            e0, e1 = D0 - E0, D1 - E1
            v00 = float((e0 * e0).sum(-1).sum(0).mean() / (G - 1))
            v11 = float((e1 * e1).sum(-1).sum(0).mean() / (G - 1))
            v01 = float((e0 * e1).sum(-1).sum(0).mean() / (G - 1))
            Ec = blocks(E0[None], E1[None])
            s00, s01, s11 = (float(tr(C)[0]) for C in Ec[:3])
            # bias of the net-mean against the exact posterior mean
            bias0 = float((E0 - mu).pow(2).sum(-1).mean())
            bias1 = float((E1 - mu).pow(2).sum(-1).mean())
            # weights
            a2, a1, a0 = quad(t00, t01, t11)
            w_pair = roots(a2, a1, a0 - tr_x0)
            a2p, a1p, a0p = quad(*P)
            w_pool = float(roots(a2p, a1p, a0p - tr_x0))
            a2v, a1v, a0v = quad(v00, v01, v11)
            w_post = float(roots(a2v, a1v, a0v - tr_post))
            # both sides of the first line, and full matrices, at w = w_pool (and at w_post)
            def both(w):
                Cg = (1 - w) ** 2 * C00 + 2 * w * (1 - w) * C01 + w * w * C11                    # [G,2,2]
                mg = (1 - w) * m0 + w * m1
                dmg = mg - mg.mean(0)
                Cp = Cg.mean(0) * (B - 1) / B + torch.einsum("gi,gj->ij", dmg, dmg) / G         # pop. norm.
                lhs = tr(Cg).cpu().numpy() * (B - 1) / B
                rhs = float(torch.trace(Cp))
                fro = ((Cg * (B - 1) / B - Cp).pow(2).sum((1, 2)).sqrt() / Cp.pow(2).sum().sqrt()).cpu().numpy()
                excess_l = lhs - tr_mu
                excess_r = rhs - tr_mu
                return dict(lhs=lhs, rhs=rhs, fro=fro, excess_l=excess_l, excess_r=excess_r,
                            offset=float(torch.einsum("gi,gi->", dmg, dmg) / G))
            at_pool = both(w_pool if np.isfinite(w_pool) else 1.0)
            at_post = both(w_post if np.isfinite(w_post) else 1.0)
            r = dict(sig=sig, G=G, B=B, tr_x0=tr_x0, tr_mu=tr_mu, tr_post=tr_post, t00=t00, t01=t01, t11=t11,
                     pooled=P, off=[o00, o01, o11], v=[v00, v01, v11], s=[s00, s01, s11],
                     bias=[bias0, bias1], w_pair=w_pair, w_pool=w_pool, w_post=w_post,
                     at_pool=at_pool, at_post=at_post)
            res[f"{a}-{b}"][sig] = r
            q = np.nanpercentile(w_pair, [5, 50, 95])
            print(f"  sig {sig:7.3f} {a:>5}-{b:<5} w_pool {w_pool:6.3f} w_pair p5/50/95 "
                  f"{q[0]:6.3f} {q[1]:6.3f} {q[2]:6.3f} w_post {w_post:6.3f} | lhs/rhs {at_pool['lhs'].mean() / at_pool['rhs']:.4f}"
                  f" excess l/r {at_pool['excess_l'].mean() / at_pool['excess_r']:.3f}", flush=True)
        print(f"sig {sig:.3f}: {time.time() - t0:.0f}s", flush=True)
    flat = {}
    for key, per in res.items():
        for sig, r in per.items():
            for k, v in r.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat.setdefault(f"{key}/{k}/{k2}", []).append(np.asarray(v2))
                else:
                    flat.setdefault(f"{key}/{k}", []).append(np.asarray(v))
    np.savez(out / "ergodic.npz", sigmas=np.array(SIGMAS), **{k: np.stack(v) for k, v in flat.items()})
    summ = {key: [dict(sig=r["sig"], w_pool=r["w_pool"], w_post=r["w_post"],
                       w_pair=list(np.nanpercentile(r["w_pair"], [5, 25, 50, 75, 95])),
                       lhs_over_rhs=float(r["at_pool"]["lhs"].mean() / r["at_pool"]["rhs"]),
                       excess_ratio=float(r["at_pool"]["excess_l"].mean() / r["at_pool"]["excess_r"]),
                       fro_median=float(np.median(r["at_pool"]["fro"])))
                  for r in per.values()] for key, per in res.items()}
    (out / "ergodic.json").write_text(json.dumps(summ, indent=1))


def matrices(out, B, seed):
    """Per pool (no guidance), per sigma, the full 2x2 matrices of eq. 16's first line and of its error part.

    C[g]    = Cov_x(D^g)                     one net over x_t                    [S,G,2,2]
    m[g]    = mean_x D^g                                                         [S,G,2]
    Cpool   = Cov_{p,x}(D) = mean_g C + Cov_g(m)  (population normalization)     [S,2,2]
    E[g]    = Cov_x(e^g), e^g = D^g - E_p[D|x]   one net's error field over x_t  [S,G,2,2]
    Vp      = E_x Cov_p(D|x)                 retraining spread at fixed x_t      [S,2,2]
    Cmu, Cpost, Cx0 = Cov(E[x0|x_t]), E Cov(x0|x_t), Cov(x0) (in-sample)         [S,2,2]
    -> DIR/ergodic-mat.npz with keys '<N>/<name>'.
    """
    ns = sorted(int(p.stem.split("-")[-1]) for p in out.glob("pool-indep-*.pt"))
    pools = {n: load(out, n)[0] for n in ns}
    gen = torch.Generator(device=DEV).manual_seed(seed)
    acc = {}

    def popcov(a, b=None):                       # a [..., B, 2] -> [..., 2, 2]
        b = a if b is None else b
        ca, cb = a - a.mean(-2, keepdim=True), b - b.mean(-2, keepdim=True)
        return torch.einsum("...bi,...bj->...ij", ca, cb) / a.shape[-2]

    for sig in SIGMAS:
        t0 = time.time()
        x0, _ = et.draw(B, gen)
        x0 = x0.double()
        xt = x0 + sig * torch.randn(B, 2, generator=gen, device=DEV, dtype=torch.float64)
        mu, cov, *_ = et.post(xt.float(), sig)
        mu, cov = mu.double(), cov.double()
        common = dict(Cmu=popcov(mu), Cpost=cov.mean(0), Cx0=popcov(x0))
        for n in ns:
            D = predict(pools[n], xt.float(), sig)                                        # [G,B,2]
            G = D.shape[0]
            C = popcov(D)
            m = D.mean(1)
            dm = m - m.mean(0)
            Cpool = C.mean(0) + torch.einsum("gi,gj->ij", dm, dm) / G
            Ep = D.mean(0)
            e = D - Ep
            E = popcov(e)
            Vp = torch.einsum("gbi,gbj->ij", e, e) / (G * B)                              # population over g
            row = dict(C=C, m=m, Cpool=Cpool, E=E, Vp=Vp, bias=((Ep - mu) ** 2).sum(-1).mean(), **common)
            for k, v in row.items():
                acc.setdefault(f"{n}/{k}", []).append(v.cpu().numpy())
            tr = lambda M: float(M.diagonal(dim1=-2, dim2=-1).sum(-1).mean())                   # noqa: E731
            print(f"  sig {sig:7.3f} N {n:5d}  tr: mean C {tr(C):.4e} Cpool {tr(Cpool):.4e} diff "
                  f"{tr(C) - tr(Cpool):+.3e} | mean E {tr(E):.3e} Vp {tr(Vp):.3e}", flush=True)
        print(f"sig {sig:.3f}: {time.time() - t0:.0f}s", flush=True)
    np.savez(out / "ergodic-mat.npz", sigmas=np.array(SIGMAS), **{k: np.stack(v) for k, v in acc.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "measure", "matrices"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--G", type=int, default=256)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--B", type=int, default=65536)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "train":
        et.G = a.G
        et.train_pool("indep", a.n, out, steps=a.steps, seed=a.seed)
    elif a.stage == "matrices":
        matrices(out, a.B, a.seed + 200)
    else:
        measure(out, a.B, a.seed + 100)


if __name__ == "__main__":
    main()
