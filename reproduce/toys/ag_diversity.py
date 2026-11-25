#!/usr/bin/env python
"""Autoguidance with a trained (big, small) MLP pair on the 2-D ten-mode mixture: does a large w drop modes?

    python reproduce/toys/ag_diversity.py train  --out DIR [--pairs 32] [--n 0]   # one GPU, minutes
    python reproduce/toys/ag_diversity.py sample --out DIR                        # DDIM, every pair and w

The trained-pair side of fig:diversity-ladder (Sec. 7; drawn by reproduce/figures/fig_ag_ladder.py): M(2) with one
fixed error field concentrates the samples on few modes, and so does guidance, whose field is D_1 - D_0 of a real pair.

Data: ten Gaussians N(mu_k, s^2 I), s = 0.15, means on a 2 x 1 ellipse (fig:moments a). --n 0: every batch is a
fresh draw from the mixture (the errors are capacity and training only); --n N: each pair trains on its own N points
(strong and weak share them); --weak-n N: only the weak nets train on their own N points per pair.
Pairs: P independent pairs, each a strong net (--strong, default STRONG) and a weak net (--weak width,depth,steps: smaller, and
trained for fewer steps if asked), with their own initialisation and data stream. One strong pool per --out, any
number of weak pools beside it (weak-TAG.pt, samples-TAG.*). EDM preconditioning and loss, as ensemble_toy.py.
Sampling: DDIM (Euler, VE) on a Karras grid, D_w = D_0 + w (D_1 - D_0), the same 5,000 starting noises for every
pair and w, so the pairs differ only through their networks. Arms: the exact posterior mean, constant w in WS, and
the variance-matching w of each pair (per level, and averaged over levels).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
GDEV = "cuda" if DEV == "cuda" else "cpu"               # where the generators live
KMIX, AX, AY, S = 10, 2.0, 1.0, 0.15
TH = 2 * math.pi * torch.arange(KMIX) / KMIX
MU = torch.stack([AX * torch.cos(TH), AY * torch.sin(TH)], 1)                 # [K, 2]
VAR_X0 = 2 * S * S + float((MU - MU.mean(0)).pow(2).sum(1).mean())            # tr Cov(x0)
SD = math.sqrt(VAR_X0 / 2)
FREQS = torch.tensor([0.5, 1, 2, 4, 8, 16.0])
STRONG = dict(width=256, depth=4, steps=40000)
BATCH, LR = 512, 2e-3
WS = [1, 1.5, 2, 3, 4, 6, 8, 12, 16]
STEPS, SMIN, SMAX, NSAMP = 256, 0.002, 20.0, 5000
DIAG_SIGMAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8]


def rn(shape, gen):
    return torch.randn(shape, generator=gen, device=GDEV).to(DEV)


def ru(shape, gen):
    return torch.rand(shape, generator=gen, device=GDEV).to(DEV)


def draw(shape, gen):
    """x0 ~ mixture, shape (..., 2)."""
    c = torch.randint(KMIX, shape, generator=gen, device=GDEV).to(DEV)
    return MU.to(DEV)[c] + S * rn((*shape, 2), gen)


def post_mean(x, sig):
    """Exact posterior mean of the mixture at VE level sig, x [..., 2]."""
    mu = MU.to(x.device)
    dl = x[..., None, :] - mu
    g = torch.softmax(-0.5 * dl.pow(2).sum(-1) / (S * S + sig * sig), -1)
    m = mu + (S * S / (S * S + sig * sig)) * dl
    return (g[..., None] * m).sum(-2)


def logp(x):
    d = x[..., None, :] - MU.to(x.device)
    return torch.logsumexp(-0.5 * d.pow(2).sum(-1) / (S * S), -1) - math.log(KMIX) - math.log(2 * math.pi * S * S)


# ------------------------------------------------------------------ batched MLP denoisers (EDM preconditioning)
class Pool(torch.nn.Module):
    """G independent MLPs evaluated together: x [G,B,2], sig [G,B] or float -> D [G,B,2]."""

    def __init__(self, g, width, depth, gen):
        super().__init__()
        dims = [2 + 2 * len(FREQS)] + [width] * depth + [2]
        self.W, self.b = torch.nn.ParameterList(), torch.nn.ParameterList()
        for i, (a, o) in enumerate(zip(dims[:-1], dims[1:])):
            w = rn((g, a, o), gen) * math.sqrt(1.0 / a)
            if i == len(dims) - 2:
                w = w * 0.1
            self.W.append(torch.nn.Parameter(w))
            self.b.append(torch.nn.Parameter(torch.zeros(g, 1, o, device=DEV)))
        self.register_buffer("freqs", FREQS.to(DEV))

    def forward(self, x, sig):
        if not torch.is_tensor(sig):
            sig = torch.full(x.shape[:2], float(sig), device=x.device)
        s = sig[..., None]
        cin, cskip = 1 / torch.sqrt(s * s + SD * SD), SD * SD / (s * s + SD * SD)
        cout = s * SD / torch.sqrt(s * s + SD * SD)
        cn = torch.log(s) / 4 * self.freqs
        h = torch.cat([x * cin, torch.cos(cn), torch.sin(cn)], -1)
        for i in range(len(self.W)):
            h = torch.baddbmm(self.b[i], h, self.W[i])
            if i < len(self.W) - 1:
                h = torch.nn.functional.silu(h)
        return cskip * x + cout * h


def train_pool(name, cfg, P, n, seed, out):
    gen = torch.Generator(device=GDEV).manual_seed(seed)
    dgen = torch.Generator(device=GDEV).manual_seed(10_000 + (seed if n == 0 else 0))
    data = draw((P, n), dgen) if n else None            # --n N: strong and weak see the same N points per pair
    pool = Pool(P, cfg["width"], cfg["depth"], gen)
    opt = torch.optim.Adam(pool.parameters(), lr=LR, betas=(0.9, 0.99))
    steps = cfg["steps"]
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / 500) * 0.5 * (1 + math.cos(math.pi * min(i / steps, 1.0))))
    t0 = time.time()
    for it in range(steps):
        if data is None:
            x0 = draw((P, BATCH), gen)
        else:
            idx = torch.randint(n, (P, BATCH), generator=gen, device=GDEV).to(DEV)
            x0 = torch.gather(data, 1, idx[..., None].expand(-1, -1, 2))
        # half log-normal around the data scale, half log-uniform on [0.002, 40]: the sampler's whole range
        sig = torch.exp(-0.6 + 1.6 * rn((P, BATCH), gen))
        lu = torch.exp(math.log(0.002) + math.log(40 / 0.002) * ru((P, BATCH), gen))
        sig = torch.where(ru((P, BATCH), gen) < 0.5, sig, lu)
        xt = x0 + sig[..., None] * rn((P, BATCH, 2), gen)
        lam = (sig * sig + SD * SD) / (sig * SD) ** 2
        loss = (lam[..., None] * (pool(xt, sig) - x0).pow(2)).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sch.step()
        if it % 5000 == 0 or it == steps - 1:
            print(f"  {name} it {it} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(state=pool.state_dict(), cfg=cfg, P=P, n=n, seed=seed, data=data), out / f"{name}.pt")
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)


def load(out, name):
    ck = torch.load(out / f"{name}.pt", map_location=DEV)
    p = Pool(ck["P"], ck["cfg"]["width"], ck["cfg"]["depth"], torch.Generator(device=GDEV).manual_seed(0))
    p.load_state_dict(ck["state"])
    p.eval()
    for q in p.parameters():
        q.requires_grad_(False)
    return p


def weak_cfg(spec, n=0):
    w, d, st = (int(v) for v in spec.split(","))
    return dict(width=w, depth=d, steps=st), f"w{w}d{d}s{st // 1000}k" + (f"-n{n}" if n else "")


def train(a):
    """The strong pool once per --out (one data setting), then the weak pool of --weak as weak-TAG.pt."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, tag = weak_cfg(a.weak, a.weak_n)
    if not (out / "strong.pt").exists():
        STRONG.update(weak_cfg(a.strong)[0])
        json.dump(dict(strong=STRONG, pairs=a.pairs, n=a.n, batch=BATCH, lr=LR, dev=DEV),
                  open(out / "train.json", "w"), indent=1)
        train_pool("strong", STRONG, a.pairs, a.n, 1, out)
    n = a.weak_n or torch.load(out / "strong.pt", map_location="cpu")["n"]
    train_pool(f"weak-{tag}", cfg, a.pairs, n, 2, out)


# ------------------------------------------------------------------ sampling
def karras(steps, smin=SMIN, smax=SMAX, rho=7.0):
    i = torch.arange(steps, dtype=torch.float64)
    s = (smax ** (1 / rho) + i / (steps - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    return [float(v) for v in s] + [0.0]


def tr_var(x):
    return (x - x.mean(-2, keepdim=True)).pow(2).sum(-1).mean(-1)


def vm_w(d1, d0):
    """Per pair: w solving tr Cov_{x_t}(D_0 + w (D_1 - D_0)) = tr Cov(x0), as (a, b, c) of a w^2 - 2 b w + c = 0."""
    c0, c1 = d0 - d0.mean(-2, keepdim=True), d1 - d1.mean(-2, keepdim=True)
    a = tr_var(d1 - d0)
    b = tr_var(d0) - (c0 * c1).sum(-1).mean(-1)
    c = tr_var(d0) - VAR_X0
    return torch.stack([a, b, c], -1)                                   # [P, 3]


def solve(abc):
    a, b, c = abc.unbind(-1)
    return (b + torch.sqrt((b * b - a * c).clamp_min(0))) / a


def mode_stats(x):
    """x [n,2] -> dict of mode weights, effective modes, log-likelihood, spread and offset within modes."""
    x, mu = x.cpu().double(), MU.double()
    ok = torch.isfinite(x).all(1) & (x.abs().max(1).values < 1e3)     # trajectories that diverged are left out
    diverged = float(1 - ok.double().mean())
    x = x[ok]
    d2 = (x[:, None] - mu).pow(2).sum(-1)
    k = d2.argmin(1)
    occ = torch.bincount(k, minlength=KMIX).double() / len(x)
    p = occ[occ > 0]
    off = torch.zeros(KMIX, 2, dtype=torch.float64).index_add_(0, k, x - mu[k])
    cnt = torch.bincount(k, minlength=KMIX).double().clamp_min(1)
    offset = float(((off / cnt[:, None]).norm(dim=1) * occ).sum() / S)  # mass-weighted |mean offset| per mode, in s
    return dict(diverged=diverged, weights=[float(v) for v in occ], eff_modes=float(torch.exp(-(p * p.log()).sum())),
                top=float(occ.max()), logp=float(logp(x).mean()), frac_out=float((d2.min(1).values > 1).float().mean()),
                spread=float(d2.min(1).values.mean() / (2 * S * S)), offset=offset)


def report(name, rs, t0):
    """Per-pair effective modes, and those of the pair-averaged weights: if each pair keeps few modes but the
    average keeps all ten, the pairs keep different modes; if the average keeps few too, they keep the same ones."""
    em = np.array([r["eff_modes"] for r in rs])
    avg = np.mean([r["weights"] for r in rs], 0)
    pa = avg[avg > 0]
    print(f"{name}: eff modes median {np.median(em):.2f} (p10 {np.quantile(em, .1):.2f}, p90 {np.quantile(em, .9):.2f});"
          f" of the pair average {np.exp(-(pa * np.log(pa)).sum()):.2f}; logp {np.median([r['logp'] for r in rs]):.2f};"
          f" out {np.median([r['frac_out'] for r in rs]):.3f}; diverged {np.mean([r['diverged'] for r in rs]):.4f}"
          f"  ({time.time() - t0:.0f}s)", flush=True)


@torch.no_grad()
def sample(a):
    out = Path(a.out)
    tag = weak_cfg(a.weak, a.weak_n)[1]
    otag = tag + (f"-i{a.interval:g}" if a.interval else "") + ("-ws" if a.ws else "") + ("-ddpm" if a.sampler == "ddpm" else "")
    ws = a.ws or WS
    strong, weak = load(out, "strong"), load(out, f"weak-{tag}")
    P = strong.W[0].shape[0]
    grid = karras(STEPS)

    # variance-matching w per pair and level, from calibration inputs x0 + sigma eps
    gc = torch.Generator(device=GDEV).manual_seed(11)
    x0c, ec = draw((8192,), gc), rn((8192, 2), gc)
    abc = []
    for s in grid[:-1]:
        xc = (x0c + s * ec)[None].expand(P, -1, -1)
        abc.append(vm_w(strong(xc, s), weak(xc, s)))
    abc = torch.stack(abc, 1)                                              # [P, L, 3]
    w_level = solve(abc)                                                   # [P, L]
    w_global = solve(abc.mean(1))                                          # [P]

    # error-field diagnostics: Delta = D_1 - D_0 over x_t ~ p_sigma
    gd = torch.Generator(device=GDEV).manual_seed(13)
    diag = {}
    for s in DIAG_SIGMAS:
        x = (draw((4096,), gd) + s * rn((4096, 2), gd))[None].expand(P, -1, -1)
        d1, d0, ds = strong(x, s), weak(x, s), post_mean(x[0], s)
        dl = d1 - d0
        diag[str(s)] = dict(
            rms_delta=[float(v) for v in dl.pow(2).sum(-1).mean(-1).sqrt()],
            shared=[float(v) for v in dl.mean(1).pow(2).sum(-1) / dl.pow(2).sum(-1).mean(-1)],
            rms_err_strong=[float(v) for v in (d1 - ds).pow(2).sum(-1).mean(-1).sqrt()],
            rms_err_weak=[float(v) for v in (d0 - ds).pow(2).sum(-1).mean(-1).sqrt()])

    xT = grid[0] * rn((NSAMP, 2), torch.Generator(device=GDEV).manual_seed(7))

    def step(x, d, s, sn, gen):
        """DDIM (deterministic Euler) or DDPM (ancestral: x_sn ~ p(x_sn | x_s, x0 = d)); the noise is shared by the pairs."""
        if a.sampler == "ddim" or sn == 0:
            return (sn / s) * x + (1 - sn / s) * d
        r = (sn / s) ** 2
        return d + r * (x - d) + sn * math.sqrt(1 - r) * rn((NSAMP, 2), gen)[None]

    def run(wfn):
        gen = torch.Generator(device=GDEV).manual_seed(17)
        x = xT[None].expand(P, -1, -1).clone()
        for i in range(len(grid) - 1):
            s, sn = grid[i], grid[i + 1]
            if s < a.interval:                                             # guidance interval: unguided below
                x = step(x, strong(x, s), s, sn, gen)
                continue
            d0 = weak(x, s)
            x = step(x, d0 + wfn(i)[:, None, None] * (strong(x, s) - d0), s, sn, gen)
        return x

    arms, finals = {}, {}
    x = xT.clone()
    for i in range(len(grid) - 1):                                         # exact posterior mean, same noise
        s, sn = grid[i], grid[i + 1]
        x = (sn / s) * x + (1 - sn / s) * post_mean(x, s)
    finals["exact"] = x[None].cpu().numpy()
    arms["exact"] = [mode_stats(x)]
    ones = torch.ones(P, device=DEV)
    for w in [0] + ws:
        t0 = time.time()
        x = run(lambda i, w=w: w * ones)
        finals[f"w{w:g}"] = x.cpu().numpy()
        arms[f"w{w:g}"] = [mode_stats(x[p]) for p in range(P)]
        report(f"w={w:g}", arms[f"w{w:g}"], t0)
    for name, wfn in (("vmg-level", lambda i: w_level[:, i]), ("vmg-global", lambda i: w_global)):
        t0 = time.time()
        x = run(wfn)
        finals[name] = x.cpu().numpy()
        arms[name] = [mode_stats(x[p]) for p in range(P)]
        report(name, arms[name], t0)
    print("w_global per pair:", np.round(w_global.cpu().numpy(), 2).tolist(), flush=True)
    np.savez_compressed(out / f"samples-{otag}.npz", grid=np.array(grid), w_level=w_level.cpu().numpy(),
                        w_global=w_global.cpu().numpy(), **finals)
    json.dump(dict(arms=arms, diag=diag, ws=ws, steps=STEPS, nsamp=NSAMP, interval=a.interval, w_global=w_global.tolist(),
                   w_level_median=w_level.median(0).values.tolist(), grid=grid),
              open(out / f"samples-{otag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "sample"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--weak", default="32,2,10000", help="width,depth,steps of the weak nets")
    ap.add_argument("--strong", default="256,4,40000", help="width,depth,steps of the strong nets (first train only)")
    ap.add_argument("--ws", type=float, nargs="+", help="constant w values to sample (default WS); outputs samples-TAG[-iX]-ws")
    ap.add_argument("--sampler", choices=["ddim", "ddpm"], default="ddim", help="ddpm: ancestral steps (outputs ...-ddpm)")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="guide only at sigma >= this (w = 1 below); outputs samples-TAG-iX")
    ap.add_argument("--weak-n", type=int, default=0,
                    help="the weak nets train on their own N points per pair (0: the strong pool's data setting)")
    ap.add_argument("--smoke", action="store_true", help="a few hundred training steps, for a quick check")
    a = ap.parse_args()
    if a.smoke:
        STRONG["steps"], STEPS, NSAMP = 300, 32, 500
    {"train": train, "sample": sample}[a.cmd](a)
