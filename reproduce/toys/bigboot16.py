#!/usr/bin/env python
"""fig:scalecopy: 16-D Gaussian data, 81,000 small denoisers per family, and how their disagreement is shaped.

    python reproduce/toys/bigboot16.py --stage train --shard k     # k = 0 .. 169, one GPU each (4 to 30 min)
    python reproduce/toys/bigboot16.py --stage reduce              # CPU, a few minutes: -> scalecopy.npz

Data: a flat Gaussian ellipsoid, 8 axes with sd log-spaced 1 -> 0.125 in a random 8-D plane of R^16 (the other 8
directions hold no data). Three families of small EDM-preconditioned residual MLPs (width 128, depth 2, 30k Adam
steps, batch 256), every net on its own training set: weak (its own 10,000 points), strong (its own 30,000) and
fresh (new points every batch: training noise alone). Each shard trains its nets side by side as one batched
ensemble, evaluates them at 69 fixed points x_t per noise level (9 levels) and keeps only the per-family sums of
D - D* and of its outer products (D* = the exact posterior mean), a few MB.

The shards are the paper's own, so a rebuild trains the same 81,000 training sets from the same seeds: shards 0-9
are the first run (100 nets per family each, sets 0-2999, seeds 1000 k), shards 10-169 the extension (500 per
family each, sets from 3000 on, seeds 500 + 1000 s). Training is on the GPU, so the nets agree with the paper's to
floating-point order of operations, not bitwise.

The reduce stage forms the across-net covariances C at every x_t, takes the data parts W = C_weak - C_fresh and
S = C_strong - C_fresh, and fits W ~ a S with a per point, per noise level or once overall, scoring each by
||W - a S||^2 / ||W||^2 (the means over x_t carry +-1 SE: a 10-block jackknife over nets and the spread over x_t, in
quadrature). It also projects W and S onto the data's longest and shortest axes for the ellipse panels.
Outputs under $VMG_RUNS/_toys/bigboot16/: shards/shard-<k>.npz and scalecopy.npz.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

SIGMAS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
ARMS = [("ind10k", 10000), ("ind30k", 30000), ("inf", 0)]       # n = 0: fresh data every batch
FAM = {"weak": "ind10k", "strong": "ind30k", "fresh": "inf"}
GOOD = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)                          # the noise levels drawn, and fitted by the one a
DATA_SEED, QUERIES = 1, 64
TRAIN = dict(steps=30000, batch=256, lr=1e-3, width=128, depth=2)


def out_dir():
    from vmg import runs
    return runs.root() / "_toys" / "bigboot16"


# --- the shard plan --------------------------------------------------------------------------------------------
def plan():
    """[(per family, first training set, family stride, shard offset, seed)] for the 170 shards of the paper."""
    first = [(100, 0, 1000, 100 * s, 1000 * s) for s in range(10)]                     # 1,000 nets per family
    ext = [(500, 3000, 100000, 500 * s, 500 + 1000 * s) for s in range(160)]           # 80,000 more
    return first + ext


def shard_models(k):
    """[(training-set index, family, n)] of shard k: every family gets an equal slice."""
    per, g0, stride, off, _ = plan()[k]
    return [(g0 + ai * stride + off + j, nm, n) for ai, (nm, n) in enumerate(ARMS) for j in range(per)]


# --- the data ----------------------------------------------------------------------------------------------------
class EmbeddedGaussian:
    """x0 = A z, z ~ N(0, I_k), A = U diag(sd): covariance U diag(sd^2) U^T of rank k in R^d."""

    def __init__(self, d=16, k=8, sd_max=1.0, sd_min=0.125, seed=0):
        import torch
        g = torch.Generator().manual_seed(seed)
        U, _ = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
        sd = torch.logspace(math.log10(sd_max), math.log10(sd_min), k, dtype=torch.float64)
        self.U, self.lam, self.sd = U[:, :k].contiguous(), sd ** 2, sd           # longest axis first
        self.d, self.k = d, k
        self.A = self.U * self.sd
        self.sigma_data = float((self.lam.sum() / d).sqrt())

    def sample(self, n, g):
        import torch
        return torch.randn(n, self.k, generator=g, dtype=torch.float64) @ self.A.T

    def fresh(self, device):
        """(m, batch, generator) -> [m, batch, d] float32 on device: the fresh family's batches."""
        import torch
        A = self.A.T.float().to(device)
        return lambda m, batch, gen: torch.randn(m, batch, self.k, generator=gen, device=device) @ A

    def denoise(self, x, sigma):
        """Exact E[x0 | x_t = x], x_t = x0 + sigma eps."""
        import torch
        x = torch.as_tensor(x, dtype=torch.float64)
        return ((x @ self.U) * (self.lam / (self.lam + sigma ** 2))) @ self.U.T

    def post_cov(self, sigma):
        return (self.U * (self.lam * sigma ** 2 / (self.lam + sigma ** 2))) @ self.U.T

    def places(self, g):
        """The five named clean points of the paper's run, in its order."""
        import torch
        return [torch.zeros(self.d, dtype=torch.float64), 2 * self.sd[0] * self.U[:, 0],
                2 * self.sd[-1] * self.U[:, -1], self.sample(1, g)[0],
                self.A @ torch.full((self.k,), 3 / self.k ** 0.5, dtype=torch.float64)]


def queries(dist):
    """x_t = y + sigma eps at every level: 5 named places and 64 data draws y, one fixed eps each. [S, 69, d]"""
    import torch
    g = torch.Generator().manual_seed(DATA_SEED + 777)
    y = torch.cat([torch.stack(dist.places(g)), dist.sample(QUERIES, g)])
    eps = torch.randn(len(y), dist.d, generator=g, dtype=torch.float64)
    return [y + s * eps for s in SIGMAS]


def training_sets(dist, models):
    """[M, nmax, d] float32: model g's set comes from its own generator, whatever the sharding."""
    import torch
    data = torch.zeros(len(models), max(n for *_, n in models), dist.d, dtype=torch.float32)
    for i, (gi, _, n) in enumerate(models):
        if n:
            data[i, :n] = dist.sample(n, torch.Generator().manual_seed(DATA_SEED * 1_000_003 + gi)).float()
    return data, torch.tensor([n for *_, n in models])


# --- the networks (unconditional) -------------------------------------------------------------------------------
def ensemble(M, sigma_data, width, depth, seed, dim, n_fourier=32):
    """M independent residual MLPs with EDM preconditioning, stacked so they train in one batched matmul."""
    import torch

    class Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.M, self.sigma_data = M, float(sigma_data)

            def lin(fan_in, fan_out, zero=False):
                b = 1 / math.sqrt(fan_in)
                if zero:
                    w, c = torch.zeros(M, fan_in, fan_out), torch.zeros(M, 1, fan_out)
                else:
                    w = (torch.rand(M, fan_in, fan_out, generator=g) * 2 - 1) * b
                    c = (torch.rand(M, 1, fan_out, generator=g) * 2 - 1) * b
                return torch.nn.Parameter(w), torch.nn.Parameter(c)

            self.register_buffer("freqs", torch.randn(M, 1, n_fourier, generator=g))
            self.emb_w, self.emb_b = lin(2 * n_fourier, width)
            self.inp_w, self.inp_b = lin(dim, width)
            mids = [lin(width, width) for _ in range(2 * depth)]
            self.mid_w = torch.nn.ParameterList([w for w, _ in mids])
            self.mid_b = torch.nn.ParameterList([b for _, b in mids])
            self.out_w, self.out_b = lin(width, dim, zero=True)

        def raw(self, x, c_noise):
            silu = torch.nn.functional.silu
            f = 2 * math.pi * c_noise[..., None] * self.freqs
            e = torch.baddbmm(self.emb_b, torch.cat([f.sin(), f.cos()], -1), self.emb_w)
            h = silu(torch.baddbmm(self.inp_b, x, self.inp_w) + e)
            ws, bs = list(self.mid_w), list(self.mid_b)
            for i in range(0, len(ws), 2):
                a = silu(torch.baddbmm(bs[i], h, ws[i]))
                h = h + silu(torch.baddbmm(bs[i + 1], a, ws[i + 1]))
            return torch.baddbmm(self.out_b, h, self.out_w)

        def forward(self, x, s):
            sd = self.sigma_data
            s = torch.as_tensor(s, dtype=x.dtype, device=x.device).expand(x.shape[:2])
            c_skip = sd ** 2 / (s ** 2 + sd ** 2)
            c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
            c_in = 1 / (s ** 2 + sd ** 2).sqrt()
            return c_skip[..., None] * x + c_out[..., None] * self.raw(c_in[..., None] * x, s.log() / 4)

    return Ensemble()


def train(data, n, sigma_data, fresh, device, steps, batch, lr, width, depth, seed, warmup=200, p_mean=-1.2,
          p_std=1.2, ema_frac=0.05, clip=1.0, lr_floor=0.05, log_every=1000):
    """EDM loss, Adam with warmup and a cosine schedule, gradients clipped per net, power EMA; returns the EMA net.
    Net k sees only its own data[k, :n[k]] (or fresh draws when n[k] = 0), its own init and its own noise."""
    import torch
    gd = torch.Generator(device=device).manual_seed(seed + 8)        # every per-step draw, on the device
    M, nmax, dim = data.shape
    fr = (n == 0).nonzero().flatten()
    n_dev = n.clamp(min=1).to(device=device, dtype=torch.float32)[:, None]
    data = data.to(device=device, dtype=torch.float32)
    net = ensemble(M, sigma_data, width, depth, seed, dim).to(device)
    params = list(net.parameters())
    ema = [p.detach().clone() for p in params]
    opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.99))
    run = torch.zeros(M, device=device)
    for step in range(1, steps + 1):
        ramp = min(1.0, step / warmup)
        cos = lr_floor + (1 - lr_floor) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for gp in opt.param_groups:
            gp["lr"] = lr * ramp * cos
        u = torch.rand(M, batch, generator=gd, device=device)
        idx = (u * n_dev).long().clamp(max=nmax - 1)
        y = torch.gather(data, 1, idx[..., None].expand(-1, -1, dim))
        if len(fr):
            y[fr] = fresh(len(fr), batch, gd)
        s = (p_mean + p_std * torch.randn(M, batch, generator=gd, device=device)).exp()
        x = y + s[..., None] * torch.randn(M, batch, dim, generator=gd, device=device)
        wgt = (s ** 2 + sigma_data ** 2) / (s * sigma_data) ** 2
        per = (wgt * (net(x, s) - y).pow(2).sum(-1)).mean(1)
        opt.zero_grad(set_to_none=True)
        per.sum().backward()
        with torch.no_grad():
            sq = sum(p.grad.pow(2).flatten(1).sum(1) for p in params)
            k = (clip / (sq.sqrt() + 1e-12)).clamp(max=1.0)
            for p in params:
                p.grad.mul_(k.view(-1, *([1] * (p.dim() - 1))))
        opt.step()
        beta = 0.5 ** (1 / max(1.0, ema_frac * step))
        with torch.no_grad():
            torch._foreach_lerp_(ema, [p.detach() for p in params], 1 - beta)
        run += per.detach()
        if step % log_every == 0:
            r = (run / log_every).cpu()
            print(f"    step {step:6d}  loss {float(r.mean()):.4f} (sd over nets {float(r.std()):.4f})", flush=True)
            run.zero_()
    with torch.no_grad():
        for e, p in zip(ema, params):
            p.copy_(e)
    return net


def stage_train(k, device, steps=None, models=None):
    """Train shard k, evaluate every net at the query points, write the per-family moments of D - D*."""
    import torch
    dist = EmbeddedGaussian()
    models = models or shard_models(k)
    data, n = training_sets(dist, models)
    cfg = dict(TRAIN, steps=steps or TRAIN["steps"])
    print(f"shard {k}: {len(models)} nets ({len(models) // len(ARMS)} per family), seed {plan()[k][4]}, "
          f"{cfg['steps']} steps on {device}", flush=True)
    t0 = time.time()
    net = train(data, n, dist.sigma_data, dist.fresh(device), device, seed=plan()[k][4], **cfg)
    secs = time.time() - t0
    qs = queries(dist)
    with torch.no_grad():                                         # D_k(q, sigma) for every net: [S, M, Q, d]
        pr = np.stack([net(q.to(device=device, dtype=torch.float32)[None].expand(net.M, -1, -1).contiguous(),
                           float(s)).float().cpu().double().float().numpy() for q, s in zip(qs, SIGMAS)])
    ref = np.stack([dist.denoise(q, s).numpy() for q, s in zip(qs, SIGMAS)])
    fam = np.array([m[1] for m in models])
    mom = dict(q=np.stack([q.numpy() for q in qs]), sigmas=np.array(SIGMAS))
    for nm, _ in ARMS:
        yv = pr[:, fam == nm].astype(np.float64) - ref[:, None]
        mom[f"{nm}_n"] = yv.shape[1]
        mom[f"{nm}_s1"] = yv.sum(1)
        mom[f"{nm}_s2"] = np.einsum("smqi,smqj->sqij", yv, yv)
    path = out_dir() / "shards" / f"shard-{k:03d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, meta=json.dumps(dict(shard=k, plan=plan()[k], train=cfg, device=_device(device), seconds=secs)),
             **mom)
    tmp.replace(path)
    print(f"trained in {secs / 60:.1f} min; wrote {path}")
    return path


def _device(device):
    import torch
    return torch.cuda.get_device_name() if str(device).startswith("cuda") else str(device)


# --- the reduction -----------------------------------------------------------------------------------------------
def shard_files(dirs):
    """Moments files in shard order: DIR/shard-<k>.npz (this script) or DIR/shard-<k>/moments.npz (the paper's)."""
    out = []
    for d in dirs:
        d = Path(d)
        fs = [p for p in d.glob("shard-*.npz") if ".tmp" not in p.name] or list(d.glob("shard-*/moments.npz"))
        key = lambda p: int((p.stem if p.suffix == ".npz" and p.name != "moments.npz" else p.parent.name).split("-")[1])
        out += sorted(fs, key=key)
    return out


def load_parts(files):
    """[{family: (n, s1 [S, Q, d], s2 [S, Q, d, d])}] in file order, and the query points."""
    parts, q0 = [], None
    for p in files:
        z = np.load(p)
        if q0 is None:
            q0 = z["q"]
        assert np.allclose(z["q"], q0, atol=1e-10), f"{p}: evaluated at other x_t than the first shard"
        parts.append({g: (int(z[f"{src}_n"]), z[f"{src}_s1"], z[f"{src}_s2"]) for g, src in FAM.items()})
    return parts, q0


def add(parts):
    return {g: tuple(sum(p[g][i] for p in parts) for i in range(3)) for g in FAM}


def sub(a, b):
    return {g: tuple(x - y for x, y in zip(a[g], b[g])) for g in FAM}


def cov(m):
    n, s1, s2 = m
    return (s2 - np.einsum("sqi,sqj->sqij", s1, s1) / n) / (n - 1)


def ip(A, B):
    return np.sum(A * B, (-2, -1))


def jse(t):
    """Delete-one jackknife standard error over axis 0."""
    K = t.shape[0]
    return np.sqrt((K - 1) / K * ((t - t.mean(0)) ** 2).sum(0))


def fits(mom, good):
    """W ~ a S with a per x_t (loc), per noise level (sig) or one for all good levels (uni); each a minimises the
    mean of ||W - a S||^2 / ||W||^2 over the points it covers."""
    C = {g: cov(mom[g]) for g in FAM}
    W, S = C["weak"] - C["fresh"], C["strong"] - C["fresh"]
    ww, ss, ws = ip(W, W), ip(S, S), ip(W, S)
    r1, r2 = ws / ww, ss / ww
    a_sig = r1.sum(1) / r2.sum(1)
    a_uni = r1[good].sum() / r2[good].sum()
    return {"a_loc": ws / ss, "L_loc": 1 - ws ** 2 / (ww * ss),
            "a_sig": a_sig, "L_sig": 1 - 2 * a_sig[:, None] * r1 + a_sig[:, None] ** 2 * r2,
            "a_uni": np.array(a_uni), "L_uni": 1 - 2 * a_uni * r1 + a_uni ** 2 * r2, "W": W, "S": S}


def reduce(dirs, out, blocks=10):
    parts, q = load_parts(shard_files(dirs))
    sig = np.array(SIGMAS)
    good = np.isin(np.round(sig, 3), GOOD)
    tot = add(parts)
    M, Q = tot["weak"][0], q.shape[1]
    K = min(blocks, len(parts))
    F = fits(tot, good)
    FJ = [fits(sub(tot, add(parts[j::K])), good) for j in range(K)] if K > 1 else []
    R = {"sigmas": sig, "good": good, "M": M, "parts": len(parts), "blocks": K,
         "a_loc": F["a_loc"], "a_sig": F["a_sig"], "a_uni": float(F["a_uni"]),
         "a_uni_se": float(jse(np.array([float(f["a_uni"]) for f in FJ]))) if FJ else np.nan}
    for key in ("L_loc", "L_sig", "L_uni"):
        L = F[key]
        R[key] = L
        R[key + "_mean"] = L.mean(1)
        R[key + "_se_nets"] = jse(np.stack([f[key].mean(1) for f in FJ])) if FJ else np.zeros(len(sig))
        R[key + "_se_xt"] = L.std(1, ddof=1) / np.sqrt(Q)
        R[key + "_se"] = np.hypot(R[key + "_se_nets"], R[key + "_se_xt"])
    # the ellipse panels: W and S on the data's longest and shortest axes, and the a that fits there
    U = EmbeddedGaussian().U.numpy()[:, [0, 7]]
    W2 = np.einsum("ia,sqij,jb->sqab", U, F["W"], U)
    S2 = np.einsum("ia,sqij,jb->sqab", U, F["S"], U)
    R.update(W2=W2, S2=S2, a2=ip(W2, S2) / ip(S2, S2), L2=1 - ip(W2, S2) ** 2 / (ip(W2, W2) * ip(S2, S2)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: np.asarray(v) for k, v in R.items()})
    gi = np.flatnonzero(good)
    print(f"{len(parts)} shards, {M} nets per family, {K}-block jackknife; one a for all: {R['a_uni']:.4f}")
    for key in ("L_loc", "L_sig", "L_uni"):
        print(f"  {key} mean over x_t: " + "  ".join(f"t {sig[i]:g}: {R[key + '_mean'][i]:.4f}" for i in gi))
    print(f"wrote {out}")
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=("train", "reduce"), required=True)
    ap.add_argument("--shard", type=int, help="train: shard 0 .. 169")
    ap.add_argument("--shards", type=int, default=len(plan()), help="train: must be 170, the paper's plan")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, help="train: fewer steps for a quick check (default 30,000)")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--from", dest="src", nargs="+", help="reduce: shard folders (default this script's own)")
    ap.add_argument("--partial", action="store_true", help="reduce: whatever shards exist (a smoke run)")
    ap.add_argument("--out", help="reduce: output file (default $VMG_RUNS/_toys/bigboot16/scalecopy.npz)")
    a = ap.parse_args()
    if a.stage == "train":
        assert a.shards == len(plan()), f"the plan has {len(plan())} shards"
        assert a.shard is not None and 0 <= a.shard < len(plan()), "--shard 0 .. 169"
        p = out_dir() / "shards" / f"shard-{a.shard:03d}.npz"
        if p.exists() and not a.redo:
            print(f"{p} exists (use --redo)")
            return
        stage_train(a.shard, a.device, a.steps)
    else:
        src = a.src or [out_dir() / "shards"]
        if not a.src and not a.partial:
            got = len(shard_files(src))
            assert got == len(plan()), f"{got} of {len(plan())} shards present (--partial for a smoke reduce)"
        reduce(src, Path(a.out) if a.out else out_dir() / "scalecopy.npz")


if __name__ == "__main__":
    main()
