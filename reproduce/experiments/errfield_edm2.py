#!/usr/bin/env python
"""How far an EDM2 pair's error difference is shared across inputs and noise levels (paper Sec. 7, App. C.5).

    python reproduce/experiments/errfield_edm2.py work --pair ag --out DIR [--n 1000]                       # GPU
    python reproduce/experiments/errfield_edm2.py work --ckpt A=... --ckpt B=... --d1 A --d0 B --w 2 --out DIR
    python reproduce/experiments/errfield_edm2.py pairs --pair ag --out DIR [--n 1000]                      # GPU
    python reproduce/experiments/errfield_edm2.py random --pair ag --out DIR [--cls 850 --per-image 80]     # GPU
    python reproduce/experiments/errfield_edm2.py spread --d1 S --d0 XS --out DIR [--cls 850]               # GPU
    python reproduce/experiments/errfield_edm2.py spreadt --d1 S --d0 XS --out DIR [--cls 850]              # GPU
    python reproduce/experiments/errfield_edm2.py report DIR [DIR ...]                                     # CPU

App. C.5 replaces the noise eps of M(2) by a field g(x, sigma) with
    Cov(g(x, s), g(x', s')) = exp(-|x - x'|^2 / 2 l_x^2 - (log s - log s')^2 / 2 l_t^2) I,
l_x in units of the median nearest-image distance of 1,000 ImageNet-64 images (one per class; fig:diversity-lx).
Here the field is a real pair's difference Delta(x, sigma) = D1(x, sigma) - D0(x, sigma), treated as a random
variable over inputs. At two inputs its correlation is the pooled one,
    rho = sum_i <Delta_i, Delta'_i> / sqrt(sum_i |Delta_i|^2 sum_i |Delta'_i|^2)        (over base images i),
raw (Delta as the sampler sees it) and centred (each evaluation's mean over the bases removed first: a mean part
of Delta is an error shared by every input, which one pair cannot tell apart from the field). l = the distance at
which rho falls to exp(-1/2), the kernel's value at |x - x'| = l, interpolated in log distance.

Bases: 1,000 ImageNet-64 val images, one per class (imagenet.calib_rows), each with its own class for every
evaluation, and one fixed noise n per base; x = x0 + sigma n at each of the 32 levels of EDM2's sampler. At each
level, against Delta(x, sigma):
  onshell  x' = x0 + sigma n', n' = cos(th) n + sin(th) m with m orthogonal to n and |m| = |n|, so x' stays on the
           noisy distribution and |x - x'| = 2 sigma |n| sin(th / 2): th is set per base to hit r / unit on the grid
           R_GRID where reachable (th <= pi / 2), plus th = pi / 2 itself ('indep', an independent noise)
  nnlerp   x0' = x0 + q (nearest other base - x0), same noise: the image direction, r = q |x0' - x0|
  nnimg    x0' = the nearest other base, same noise (r ~ 1 unit, reachable at low sigma)
  rndimg   x0' = a random other base, same noise
  dlogs    the same x, the networks told sigma' = sigma exp(delta): the kernel's fixed-x slice in log sigma
  repeat   the same call again (must give rho = 1)
and, separately, `traj`: 1,000 seeds (generate_images.py's noise and class) through the pair's guided Heun sampler,
Delta at the state each step starts from, the 32 x 32 correlation across steps: what M(2)'s eps would see from
one step to the next (x moves too, so this mixes l_x and l_t).

Outputs in DIR: meta.json, field.npz (per level, perturbation and base: dot, |Delta|^2, |Delta'|^2 raw and
centred, r in pixel units), traj.npz (per seed 32 x 32 dots raw and centred), and from `report`, lengths.json.
"""
import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

R_GRID = (0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0)       # on-shell distances, in units
LERPS = (0.01, 0.03, 0.1, 0.2, 0.3, 0.5)    # image direction: x0 moved this far toward its nearest base
DL_GRID = (0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)   # spreadt: |log sigma' - log sigma|
DLOGS = (-1.0, -0.5, -0.3, -0.2, -0.1, -0.03, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0)
PAIRS = {  # name: (d1, d0, w), members of ens_edm2.MODELS; ag = the App. B.4 pair, NVIDIA's FID preset
    "ag": ("S45", "XS", 1.70),
}
SEED_N, SEED_M, SEED_RND = 11, 12, 13
E_HALF = math.exp(-0.5)


def atomic_npz(path, **arrays):
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def names():
    """Perturbation names, in the order of field.npz's axis 1."""
    return (["repeat"] + [f"onshell{r:g}" for r in R_GRID] + ["indep", "nnimg", "rndimg"]
            + [f"nnlerp{q:g}" for q in LERPS] + [f"dlogs{d:+g}" for d in DLOGS])


def units(dev):
    """Median nearest-image distance (EDM2 pixels, [-1, 1]) of the toy's 1,000 train images (fig:diversity-lx's
    unit), of its 10,000, and of the 1,000 val bases (filled in by work)."""
    import torch
    from reproduce.toys import corr_diversity_in64 as CD
    z = np.load(CD.out_dir() / "in64-10k.npz")
    u8 = z["u8"]
    first = np.unique(z["labels"], return_index=True)[1]
    out = {}
    for key, ims in (("train1k", u8[first]), ("train10k", u8)):
        X = torch.from_numpy(ims.reshape(len(ims), -1)).to(dev, torch.float64) / 127.5 - 1
        out[key] = float(CD.kth_sqdist(X, 1).sqrt().median())
    return out


def work(a):
    import torch
    from reproduce.experiments import ens_edm2 as E
    from reproduce.toys import corr_diversity_in64 as CD
    from vmg.data import imagenet
    for c in a.ckpt:
        k, v = c.split("=", 1)
        E.MODELS[k] = v
    d1n, d0n, w = PAIRS[a.pair] if a.pair else (a.d1, a.d0, a.w)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    sig = E.t_steps()[:-1].double().numpy()
    S = len(sig)
    print(f"{socket.gethostname()} {torch.cuda.get_device_name()}: D1 = {d1n} ({E.MODELS[d1n]}), "
          f"D0 = {d0n} ({E.MODELS[d0n]}), w = {w}", flush=True)

    def delta(x, s, lab):
        t = torch.full((len(x),), float(s), device=dev)
        return E.call(d1n, x, t, lab, fp32=True) - E.call(d0n, x, t, lab, fp32=True)

    # ---- bases, unit, fixed noises
    rows = imagenet.calib_rows(a.n)
    x0 = imagenet.pixels(rows).to(dev)
    lab = torch.nn.functional.one_hot(imagenet.class_of(rows, 64), 1000).float().to(dev)
    N, d = len(x0), x0[0].numel()
    U = units(dev)
    X = x0.reshape(N, -1).double()
    U["val_bases"] = float(CD.kth_sqdist(X, 1).sqrt().median())
    unit = U["train1k"]
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None] - 2 * X @ X.T
    D2.fill_diagonal_(float("inf"))
    nn_idx = D2.argmin(1)
    g = torch.Generator(device=dev).manual_seed(SEED_RND)
    rnd_idx = (torch.arange(N, device=dev) + torch.randint(1, N, (N,), generator=g, device=dev)) % N
    n = torch.randn(x0.shape, generator=torch.Generator(device=dev).manual_seed(SEED_N), device=dev)
    m = torch.randn(x0.shape, generator=torch.Generator(device=dev).manual_seed(SEED_M), device=dev)
    nf, mf = n.reshape(N, -1), m.reshape(N, -1)
    mf = mf - (mf * nf).sum(1, keepdim=True) / (nf * nf).sum(1, keepdim=True) * nf
    mf = mf * (nf.norm(dim=1, keepdim=True) / mf.norm(dim=1, keepdim=True))
    m = mf.reshape(x0.shape)
    nnorm = nf.norm(dim=1)                                          # [N]
    print(f"{N} bases, d = {d}; median nearest-image distance: toy 1k train {U['train1k']:.2f}, 10k train "
          f"{U['train10k']:.2f}, val bases {U['val_bases']:.2f} (unit = {unit:.2f}); |n| ~ {float(nnorm.mean()):.1f}",
          flush=True)

    P = names()
    shp = (S, len(P), N)
    F = {k: np.full(shp, np.nan) for k in ("dot", "a2", "b2", "cdot", "ca2", "cb2", "r")}
    mse = np.zeros((S, 2))
    t0 = time.time()
    with torch.no_grad():
        for k, s in enumerate(sig):
            x = x0 + float(s) * n
            evals = {}                       # name -> (Delta' [N, d], r [N] or None, valid [N] bool)

            def run(name, xp, sp, r, valid):
                buf = torch.empty(N, d, device=dev)
                for b0 in range(0, N, a.batch):
                    sl = slice(b0, min(b0 + a.batch, N))
                    buf[sl] = delta(xp[sl], sp, lab[sl]).reshape(-1, d)
                evals[name] = (buf, r, valid)

            # the base call, with both MSEs as a sanity check (D1 should beat D0)
            base = torch.empty(N, d, device=dev)
            m1 = m0 = 0.0
            for b0 in range(0, N, a.batch):
                sl = slice(b0, min(b0 + a.batch, N))
                t = torch.full((sl.stop - sl.start,), float(s), device=dev)
                p1 = E.call(d1n, x[sl], t, lab[sl], fp32=True)
                p0 = E.call(d0n, x[sl], t, lab[sl], fp32=True)
                base[sl] = (p1 - p0).reshape(-1, d)
                m1 += float(((p1 - x0[sl]) ** 2).sum())
                m0 += float(((p0 - x0[sl]) ** 2).sum())
            mse[k] = m1 / (N * d), m0 / (N * d)
            ones = torch.ones(N, dtype=torch.bool, device=dev)
            run("repeat", x, s, torch.zeros(N, device=dev), ones)
            for rr in R_GRID:
                arg = rr * unit / (2 * float(s) * nnorm)
                valid = arg <= math.sin(math.pi / 4)
                th = 2 * torch.arcsin(arg.clamp(max=math.sin(math.pi / 4)))
                xp = x0 + float(s) * (torch.cos(th)[:, None, None, None] * n + torch.sin(th)[:, None, None, None] * m)
                if bool(valid.any()):
                    run(f"onshell{rr:g}", xp, s, (xp - x).reshape(N, -1).norm(dim=1), valid)
            xp = x0 + float(s) * m
            run("indep", xp, s, (xp - x).reshape(N, -1).norm(dim=1), ones)
            for name, idx in (("nnimg", nn_idx), ("rndimg", rnd_idx)):
                xp = x0[idx] + float(s) * n
                run(name, xp, s, (xp - x).reshape(N, -1).norm(dim=1), ones)
            for q in LERPS:
                xp = x0 + q * (x0[nn_idx] - x0) + float(s) * n
                run(f"nnlerp{q:g}", xp, s, (xp - x).reshape(N, -1).norm(dim=1), ones)
            for dl in DLOGS:
                sp = float(s) * math.exp(dl)
                if E.SMIN <= sp <= E.SMAX:             # outside the trained range: left NaN
                    run(f"dlogs{dl:+g}", x, sp, torch.zeros(N, device=dev), ones)
            # per-base scalars; centred = each evaluation's own mean over the bases removed
            bc = base - base.mean(0)
            for j, name in enumerate(P):
                if name not in evals:
                    continue
                v, r, valid = evals[name]
                vc = v - v.mean(0)
                sel = valid.cpu().numpy()
                vals = dict(dot=(base * v).sum(1), a2=(base * base).sum(1), b2=(v * v).sum(1),
                            cdot=(bc * vc).sum(1), ca2=(bc * bc).sum(1), cb2=(vc * vc).sum(1), r=r)
                for key, val in vals.items():
                    arr = val.double().cpu().numpy()
                    F[key][k, j, sel] = arr[sel]
            rep = F["dot"][k, 0].sum() / math.sqrt(F["a2"][k, 0].sum() * F["b2"][k, 0].sum())
            assert abs(rep - 1) < 1e-5, f"sigma {s:.3g}: repeated call gives rho {rep}"

            def rho(name):
                j = P.index(name)
                ok = ~np.isnan(F["dot"][k, j])
                if not ok.any():
                    return float("nan")
                return F["dot"][k, j, ok].sum() / math.sqrt(F["a2"][k, j, ok].sum() * F["b2"][k, j, ok].sum())
            print(f"  sigma {s:8.4f}: MSE D1 {mse[k, 0]:.3g} D0 {mse[k, 1]:.3g}, |Delta|^2/d "
                  f"{F['a2'][k, 0].mean() / d:.3g}; rho onshell0.01 {rho('onshell0.01'):.3f} 0.1 "
                  f"{rho('onshell0.1'):.3f} 1 {rho('onshell1'):.3f} indep {rho('indep'):.3f} nnimg "
                  f"{rho('nnimg'):.3f} rndimg {rho('rndimg'):.3f} dlogs+0.1 {rho('dlogs+0.1'):.3f}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            del evals
    atomic_npz(out / "field.npz", sigma=sig, names=np.array(P), mse=mse, **F)
    print(f"wrote {out / 'field.npz'}", flush=True)

    # ---- along the guided sampler
    ws = np.full(S, w)
    DT = torch.empty(S, a.n_traj, d, device=dev)

    def make_pred(rec):
        def pred(xx, t, labels, i):
            tt = torch.full((len(xx),), float(t), device=dev)
            p1, p0 = E.call(d1n, xx, tt, labels, fp32=True), E.call(d0n, xx, tt, labels, fp32=True)
            if rec is not None:
                rec(i, p1 - p0)
            return p0 + float(ws[i]) * (p1 - p0)
        return pred

    @torch.no_grad()
    def heun(noise, labels, rec):
        """E.heun (all calls in float32), recording Delta at the state each step starts from (the first call of each step)."""
        ts = E.t_steps(noise.device)
        pred = make_pred(None)
        x_next = noise * ts[0]
        for i, (t_cur, t_next) in enumerate(zip(ts[:-1], ts[1:])):
            x_hat, t_hat = x_next, t_cur
            d_cur = (x_hat - make_pred(lambda _, dd: rec(i, dd))(x_hat, t_hat, labels, i)) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < S - 1:
                d_prime = (x_next - pred(x_next, t_next, labels, i + 1)) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next

    noise, labels = E.initial(list(range(8)))
    E.STEPS = S
    ref = E.heun(make_pred(None), noise, labels)
    got = heun(noise, labels, lambda *_: None)
    rel = float((ref - got).norm() / ref.norm())
    print(f"check: recording loop against ens_edm2.heun at w = {w} (8 seeds): relative L2 {rel:.1e}", flush=True)
    assert rel < 1e-6, "the recording loop is not the sampler"
    tb = 125
    for b0 in range(0, a.n_traj, tb):
        seeds = list(range(b0, min(b0 + tb, a.n_traj)))
        noise, labels = E.initial(seeds)

        def rec(i, dd, b0=b0):
            DT[i, b0:b0 + len(dd)] = dd.float().reshape(len(dd), -1)
        heun(noise, labels, rec)
    Dc = DT - DT.mean(1, keepdim=True)
    G = torch.einsum("isd,jsd->sij", DT, DT).double().cpu().numpy()       # [seeds, S, S]
    Gc = torch.einsum("isd,jsd->sij", Dc, Dc).double().cpu().numpy()
    atomic_npz(out / "traj.npz", sigma=sig, gram=G, cgram=Gc, w=w)
    meta = dict(d1=d1n, d0=d0n, w=w, net=E.MODELS[d1n], guide=E.MODELS[d0n], n=N, n_traj=a.n_traj, d=d,
                units=U, unit=unit, r_grid=list(R_GRID), dlogs=list(DLOGS), names=P,
                device=torch.cuda.get_device_name(), host=socket.gethostname(), minutes=(time.time() - t0) / 60)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: field.npz, traj.npz, meta.json ({meta['minutes']:.1f} min)", flush=True)


# ------------------------------------------------------------------ pairs: Corr against distance, no kernel assumed
LOGR_BINS = np.arange(-3.5, 3.01, 0.1)          # log10 of the distance in units


def pairs(a):
    """Every pair of n guided trajectories at every step: the correlation of Delta at the two states against their
    distance, with no kernel assumed. Writes pairs.npz: per step, binned stats over all n(n-1)/2 pairs (count, mean,
    quantiles of the per-pair cosine, raw and centred) and a random subsample of pairs (distance, cosines, the
    distance between the two final samples, same class)."""
    import torch
    from reproduce.experiments import ens_edm2 as E
    for c in a.ckpt:
        k, v = c.split("=", 1)
        E.MODELS[k] = v
    d1n, d0n, w = PAIRS[a.pair] if a.pair else (a.d1, a.d0, a.w)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    sig = E.t_steps()[:-1].double().numpy()
    S, n, d = len(sig), a.n, 3 * 64 * 64
    U = units(dev)
    unit = U["train1k"]
    print(f"{socket.gethostname()} {torch.cuda.get_device_name()}: pairs, D1 = {d1n}, D0 = {d0n}, w = {w}, "
          f"{n} seeds, unit {unit:.2f} px", flush=True)
    E.STEPS = S

    def call2(xx, t, labels):
        tt = torch.full((len(xx),), float(t), device=dev)
        return E.call(d1n, xx, tt, labels, fp32=True), E.call(d0n, xx, tt, labels, fp32=True)

    XS = torch.empty(S, n, d, device=dev)
    DT = torch.empty(S, n, d, device=dev)
    D1 = torch.empty(S, n, d, device=dev) if a.denoised else None
    X0 = torch.empty(n, d, device=dev)
    cls = np.empty(n, np.int64)
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, n, 125):
            seeds = list(range(b0, min(b0 + 125, n)))
            noise, labels = E.initial(seeds)
            if a.classes:                          # K fixed classes, seed s gets class number s % K
                pick = np.random.default_rng(0).choice(1000, a.classes, replace=False)
                c = torch.as_tensor(pick[np.array(seeds) % a.classes], device=dev)
                labels = torch.nn.functional.one_hot(c, 1000).float()
            cls[b0:b0 + len(seeds)] = labels.argmax(1).cpu().numpy()
            ts = E.t_steps(dev)
            x_next = noise * ts[0]
            for i, (t_cur, t_next) in enumerate(zip(ts[:-1], ts[1:])):
                x_hat, t_hat = x_next, t_cur
                p1, p0 = call2(x_hat, t_hat, labels)
                XS[i, b0:b0 + len(seeds)] = x_hat.reshape(len(seeds), -1)
                DT[i, b0:b0 + len(seeds)] = (p1 - p0).reshape(len(seeds), -1)
                if D1 is not None:
                    D1[i, b0:b0 + len(seeds)] = p1.reshape(len(seeds), -1)
                d_cur = (x_hat - (p0 + w * (p1 - p0))) / t_hat
                x_next = x_hat + (t_next - t_hat) * d_cur
                if i < S - 1:
                    q1, q0 = call2(x_next, t_next, labels)
                    d_prime = (x_next - (q0 + w * (q1 - q0))) / t_next
                    x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            X0[b0:b0 + len(seeds)] = x_next.reshape(len(seeds), -1)
        print(f"  sampled {n} trajectories ({(time.time() - t0) / 60:.1f} min)", flush=True)

        def gram_dist(Z):
            Z = Z.double()
            G = Z @ Z.T
            q = torch.diagonal(G)
            return (q[:, None] + q[None] - 2 * G).clamp(min=0).sqrt()

        iu = np.triu_indices(n, 1)
        rng = np.random.default_rng(0)
        rf = (gram_dist(X0).cpu().numpy() / unit)[iu]
        same = (cls[:, None] == cls[None])[iu]
        # subsample: up to n_sub same-class and n_sub other-class pairs
        sub = np.concatenate([rng.choice(np.flatnonzero(m), size=min(a.n_sub, int(m.sum())), replace=False)
                              for m in (same, ~same) if m.any()])
        nb = len(LOGR_BINS) - 1
        Q = (0.1, 0.25, 0.5, 0.75, 0.9)
        GROUPS = ("all", "same", "diff")
        stats = {f"{g}_{kind}_{s}": np.full((S, nb), np.nan) for g in GROUPS for kind in ("raw", "cen")
                 for s in ("mean",) + tuple(f"q{int(q * 100)}" for q in Q)}
        count = {f"{g}_count": np.zeros((S, nb), np.int64) for g in GROUPS}
        sub_r, sub_raw, sub_cen = (np.empty((S, len(sub))) for _ in range(3))
        # --denoised: the same pair statistics for the denoisers' outputs themselves (x0 predictions)
        FIELDS = ("d1", "d0", "dw") if a.denoised else ()
        subD = {f"sub_{f}_{kind}": np.empty((S, len(sub)), np.float32) for f in FIELDS for kind in ("raw", "cen")}
        meanD = {f"{f}_{kind}_pairmean": np.empty(S) for f in FIELDS for kind in ("raw", "cen")}
        for k in range(S):
            r = (gram_dist(XS[k]).cpu().numpy() / unit)[iu]
            cos = {}
            for kind, Z in (("raw", DT[k]), ("cen", DT[k] - DT[k].mean(0))):
                Z = Z.double()
                Z = Z / Z.norm(dim=1, keepdim=True)
                cos[kind] = (Z @ Z.T).cpu().numpy()[iu]
            b = np.digitize(np.log10(np.maximum(r, 1e-12)), LOGR_BINS) - 1
            for g, gm in (("all", np.ones_like(same)), ("same", same), ("diff", ~same)):
                for j in np.unique(b[gm]):
                    if j < 0 or j >= nb:
                        continue
                    m = gm & (b == j)
                    count[f"{g}_count"][k, j] = m.sum()
                    for kind in ("raw", "cen"):
                        v = cos[kind][m]
                        stats[f"{g}_{kind}_mean"][k, j] = v.mean()
                        for q, val in zip(Q, np.quantile(v, Q)):
                            stats[f"{g}_{kind}_q{int(q * 100)}"][k, j] = val
            sub_r[k], sub_raw[k], sub_cen[k] = r[sub], cos["raw"][sub], cos["cen"][sub]
            for f in FIELDS:
                F = {"d1": D1[k], "d0": D1[k] - DT[k], "dw": D1[k] + (w - 1) * DT[k]}[f]
                for kind, Z in (("raw", F), ("cen", F - F.mean(0))):
                    Z = Z.double()
                    Z = Z / Z.norm(dim=1, keepdim=True)
                    cf = (Z @ Z.T).cpu().numpy()[iu]
                    subD[f"sub_{f}_{kind}"][k] = cf[sub]
                    meanD[f"{f}_{kind}_pairmean"][k] = cf.mean()
            if FIELDS:
                print("    denoised, mean corr raw/cen: " + "  ".join(
                    f"{f} {meanD[f + '_raw_pairmean'][k]:.3f}/{meanD[f + '_cen_pairmean'][k]:.3f}" for f in FIELDS),
                    flush=True)
            print(f"  sigma {sig[k]:8.4f}: pair distance median {np.median(r):.3g} units "
                  f"(10-90 % {np.quantile(r, .1):.3g}-{np.quantile(r, .9):.3g}), cos raw same class "
                  f"{cos['raw'][same].mean() if same.any() else float('nan'):.3f} other {cos['raw'][~same].mean():.3f}, "
                  f"centred same {cos['cen'][same].mean() if same.any() else float('nan'):.3f} other "
                  f"{cos['cen'][~same].mean():.3f}", flush=True)
    atomic_npz(out / (a.tag + ".npz"), sigma=sig, logr_bins=LOGR_BINS, cls=cls, **count, sub_r=sub_r, sub_raw=sub_raw,
               sub_cen=sub_cen, sub_rfinal=rf[sub], sub_same=same[sub], **stats, **subD, **meanD)
    meta = dict(d1=d1n, d0=d0n, w=w, net=E.MODELS[d1n], guide=E.MODELS[d0n], n=n, n_sub=len(sub), units=U,
                classes=a.classes,
                unit=unit, device=torch.cuda.get_device_name(), minutes=(time.time() - t0) / 60)
    (out / (a.tag + "-meta.json")).write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / (a.tag + '.npz')} ({meta['minutes']:.1f} min)", flush=True)


# ------------------------------------------------------------------ random
def random_x(a):
    """No sampler: x = x0 + sigma n from the forward process, x0 = the val images of one class (50), --per-image
    fresh noises each, the class label fixed. At every level, all pairs of these x: |x - x'| and the correlation
    (raw, and centred over the x) of D1, D0, D_w = D0 + w (D1 - D0) and Delta = D1 - D0. A subsample of pairs is kept,
    every same-image pair (up to n_sub / 2) and the rest from different-image pairs."""
    import torch
    from reproduce.experiments import ens_edm2 as E
    from vmg.data import imagenet
    for c in a.ckpt:
        k, v = c.split("=", 1)
        E.MODELS[k] = v
    d1n, d0n, w = PAIRS[a.pair] if a.pair else (a.d1, a.d0, a.w)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    sig = E.t_steps()[:-1].double().numpy()
    S = len(sig)
    U = units(dev)
    unit = U["train1k"]
    rows = np.flatnonzero(imagenet.labels(64) == a.cls)
    x0 = imagenet.pixels([int(r) for r in rows]).to(dev)
    M, P = len(rows), a.per_image
    n, d = M * P, x0[0].numel()
    img = np.repeat(np.arange(M), P)
    X0 = x0.reshape(M, d)[torch.as_tensor(img, device=dev)]
    lab = torch.nn.functional.one_hot(torch.full((n,), a.cls, device=dev), 1000).float()
    print(f"{socket.gethostname()} {torch.cuda.get_device_name()}: random x, D1 = {d1n}, D0 = {d0n}, w = {w}, "
          f"class {a.cls}: {M} val images x {P} noises = {n} x per level, unit {unit:.2f} px", flush=True)
    iu = np.triu_indices(n, 1)
    same = (img[:, None] == img[None])[iu]
    rng = np.random.default_rng(0)
    si, di = np.flatnonzero(same), np.flatnonzero(~same)
    ns = min(len(si), a.n_sub // 2)
    sub = np.sort(np.concatenate([rng.choice(si, ns, replace=False), rng.choice(di, a.n_sub - ns, replace=False)]))
    FIELDS = ("delta", "d1", "d0", "dw")
    keep = {f"sub_{f}_{k}": np.empty((S, len(sub)), np.float32) for f in FIELDS for k in ("raw", "cen")}
    pm = {f"{f}_{k}_{g}mean": np.empty(S) for f in FIELDS for k in ("raw", "cen") for g in ("same", "diff")}
    sub_r = np.empty((S, len(sub)), np.float32)
    g = torch.Generator(device=dev).manual_seed(SEED_N)
    t0 = time.time()
    with torch.no_grad():
        for k, s in enumerate(sig):
            x = X0 + float(s) * torch.randn(n, d, generator=g, device=dev)
            D1 = torch.empty(n, d, device=dev)
            D0 = torch.empty(n, d, device=dev)
            for b0 in range(0, n, a.batch):
                sl = slice(b0, min(b0 + a.batch, n))
                t = torch.full((sl.stop - sl.start,), float(s), device=dev)
                xx = x[sl].reshape(-1, 3, 64, 64)
                D1[sl] = E.call(d1n, xx, t, lab[sl], fp32=True).reshape(-1, d)
                D0[sl] = E.call(d0n, xx, t, lab[sl], fp32=True).reshape(-1, d)
            Z = x.double()
            G = Z @ Z.T
            q = torch.diagonal(G)
            r = ((q[:, None] + q[None] - 2 * G).clamp(min=0).sqrt() / unit).cpu().numpy()[iu]
            sub_r[k] = r[sub]
            del Z, G
            for f in FIELDS:
                F = {"delta": D1 - D0, "d1": D1, "d0": D0, "dw": D0 + w * (D1 - D0)}[f]
                for kind, V in (("raw", F), ("cen", F - F.mean(0))):
                    V = V.double()
                    V = V / V.norm(dim=1, keepdim=True)
                    cf = (V @ V.T).cpu().numpy()[iu]
                    keep[f"sub_{f}_{kind}"][k] = cf[sub]
                    pm[f"{f}_{kind}_samemean"][k] = cf[same].mean()
                    pm[f"{f}_{kind}_diffmean"][k] = cf[~same].mean()
            print(f"  sigma {s:8.4f}: |x - x'| same image {np.median(r[same]):.3g}, other image {np.median(r[~same]):.3g} "
                  f"units; corr raw same/other: D1 {pm['d1_raw_samemean'][k]:.3f}/{pm['d1_raw_diffmean'][k]:.3f}, "
                  f"Delta {pm['delta_raw_samemean'][k]:.3f}/{pm['delta_raw_diffmean'][k]:.3f}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    atomic_npz(out / f"{a.tag}.npz", sigma=sig, cls=np.full(n, a.cls), img=img, rows=rows, sub_r=sub_r,
               sub_sameimg=same[sub], sub_raw=keep["sub_delta_raw"], sub_cen=keep["sub_delta_cen"], **keep, **pm)
    meta = dict(d1=d1n, d0=d0n, w=w, net=E.MODELS[d1n], guide=E.MODELS[d0n], n=n, images=M, per_image=P, cls=a.cls,
                n_sub=len(sub), units=U, unit=unit, device=torch.cuda.get_device_name(),
                minutes=(time.time() - t0) / 60)
    (out / f"{a.tag}-meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / (a.tag + '.npz')} ({meta['minutes']:.1f} min)", flush=True)


# ------------------------------------------------------------------ spread
def spread(a):
    """Random x of one class with partners at every distance. Base x = x0 + sigma n (x0 a random val image of the
    class, n fresh); each base gets --per-base partners
        x' = x0 + t (x0j - x0) + sigma (cos(t pi/2) n + sin(t pi/2) m),   t ~ log-uniform [--tmin, 1],
    x0j another val image of the class, m a fresh noise orthogonal to n with |m| = |n|. t -> 0 gives x' ~ x, t = 1 an
    independent random x of the class (another image, an independent noise), so |x - x'| spans that whole range at
    every sigma. Per (base, partner): |x - x'| (units) and the cosine of D1, D0 and Delta = D1 - D0, raw and centred
    (the mean over all evaluations at that sigma removed). Class label fixed; same bases, t and noises at every level."""
    import torch
    from reproduce.experiments import ens_edm2 as E
    from vmg.data import imagenet
    for c in a.ckpt:
        k, v = c.split("=", 1)
        E.MODELS[k] = v
    d1n, d0n = a.d1, a.d0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ks = [int(k) for k in a.levels.split(",")]
    sig = E.t_steps()[:-1].double().numpy()[ks]
    unit = units(dev)["train1k"]
    rows = np.flatnonzero(imagenet.labels(64) == a.cls)
    M = len(rows)
    X0 = imagenet.pixels([int(r) for r in rows]).to(dev).reshape(M, -1)
    d = X0.shape[1]
    B, K = a.n_base, a.per_base
    rng = np.random.default_rng(0)
    ib = rng.integers(M, size=B)
    jb = (ib[:, None] + rng.integers(1, M, size=(B, K))) % M
    t = 10 ** rng.uniform(np.log10(a.tmin), 0, size=(B, K))
    g = torch.Generator(device=dev).manual_seed(SEED_N)
    n = torch.randn(B, d, generator=g, device=dev)
    m = torch.randn(B, K, d, generator=g, device=dev)
    nh = n / n.norm(dim=1, keepdim=True)
    m -= (m * nh[:, None]).sum(-1, keepdim=True) * nh[:, None]
    m *= n.norm(dim=1)[:, None, None] / m.norm(dim=-1, keepdim=True)
    tt = torch.as_tensor(t, device=dev, dtype=torch.float32)[..., None]
    th = tt * (math.pi / 2)
    A = X0[torch.as_tensor(ib, device=dev)]
    Ap = A[:, None] + tt * (X0[torch.as_tensor(jb, device=dev)] - A[:, None])
    Np = torch.cos(th) * n[:, None] + torch.sin(th) * m
    del m
    lab = torch.nn.functional.one_hot(torch.full((a.batch,), a.cls, device=dev), 1000).float()
    print(f"{socket.gethostname()} {torch.cuda.get_device_name()}: spread, D1 = {d1n}, D0 = {d0n}, class {a.cls} "
          f"({M} val images), {B} bases x {K} partners, t in [{a.tmin:g}, 1], sigma {np.round(sig, 4).tolist()}",
          flush=True)
    FIELDS = ("d1", "d0", "delta")
    res = {f"{f}_{k}": np.empty((len(ks), B, K), np.float32) for f in FIELDS for k in ("raw", "cen")}
    R = np.empty((len(ks), B, K), np.float32)
    t0 = time.time()

    def den(name, x, s):
        o = torch.empty_like(x)
        for b0 in range(0, len(x), a.batch):
            xx = x[b0:b0 + a.batch]
            tv = torch.full((len(xx),), float(s), device=dev)
            o[b0:b0 + a.batch] = E.call(name, xx.reshape(-1, 3, 64, 64), tv, lab[:len(xx)], fp32=True).reshape(len(xx), d)
        return o

    with torch.no_grad():
        for li, s in enumerate(sig):
            x = A + float(s) * n
            xp = (Ap + float(s) * Np).reshape(B * K, d)
            R[li] = ((x[:, None] - xp.reshape(B, K, d)).norm(dim=-1) / unit).cpu().numpy()
            D1 = den(d1n, torch.cat([x, xp]), s)
            D0 = den(d0n, torch.cat([x, xp]), s)
            for f in FIELDS:
                F = {"d1": D1, "d0": D0, "delta": D1 - D0}[f]
                for kind, V in (("raw", F), ("cen", F - F.mean(0))):
                    V = V / V.norm(dim=1, keepdim=True)
                    res[f"{f}_{kind}"][li] = (V[:B, None] * V[B:].reshape(B, K, d)).sum(-1).cpu().numpy()
            del D1, D0, F, V
            r = R[li]
            print(f"  sigma {s:8.4f}: |x - x'| 1/50/99 % {np.percentile(r, 1):.3g} {np.median(r):.3g} "
                  f"{np.percentile(r, 99):.3g} units; corr raw D1 near/far {res['d1_raw'][li][t < 1e-3].mean():.3f}/"
                  f"{res['d1_raw'][li][t > 0.5].mean():.3f}, Delta {res['delta_raw'][li][t < 1e-3].mean():.3f}/"
                  f"{res['delta_raw'][li][t > 0.5].mean():.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    atomic_npz(out / f"{a.tag}.npz", sigma=sig, levels=np.array(ks), t=t, ib=ib, jb=jb, rows=rows, r=R, **res)
    meta = dict(d1=d1n, d0=d0n, net=E.MODELS[d1n], guide=E.MODELS[d0n], cls=a.cls, images=M, n_base=B, per_base=K,
                tmin=a.tmin, unit=unit, device=torch.cuda.get_device_name(), minutes=(time.time() - t0) / 60)
    (out / f"{a.tag}-meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / (a.tag + '.npz')} ({meta['minutes']:.1f} min)", flush=True)


def spread_t(a):
    """The log-sigma side of the kernel for `spread`'s bases: the same x = x0 + sigma n (same seeds, so the same
    bases as `spread`), the networks told sigma' = sigma exp(delta) for delta in DL_GRID. Per base and delta: the
    cosine of Delta(x, sigma) and Delta(x, sigma') (and of D1 and D0), raw and centred (mean over the bases at each sigma')."""
    import torch
    from reproduce.experiments import ens_edm2 as E
    from vmg.data import imagenet
    for c in a.ckpt:
        k, v = c.split("=", 1)
        E.MODELS[k] = v
    d1n, d0n = a.d1, a.d0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    ks = [int(k) for k in a.levels.split(",")]
    sig = E.t_steps()[:-1].double().numpy()[ks]
    rows = np.flatnonzero(imagenet.labels(64) == a.cls)
    M = len(rows)
    X0 = imagenet.pixels([int(r) for r in rows]).to(dev).reshape(M, -1)
    d = X0.shape[1]
    B = a.n_base
    rng = np.random.default_rng(0)
    ib = rng.integers(M, size=B)                        # the first draw of `spread`, so the same bases
    g = torch.Generator(device=dev).manual_seed(SEED_N)
    n = torch.randn(B, d, generator=g, device=dev)
    A = X0[torch.as_tensor(ib, device=dev)]
    dl = np.array(sorted({0.0} | {s * v for v in DL_GRID for s in (-1, 1)}))
    lab = torch.nn.functional.one_hot(torch.full((a.batch,), a.cls, device=dev), 1000).float()
    print(f"{socket.gethostname()} {torch.cuda.get_device_name()}: spreadt, D1 = {d1n}, D0 = {d0n}, class {a.cls}, "
          f"{B} bases, sigma {np.round(sig, 4).tolist()}, delta {dl.tolist()}", flush=True)
    res = {f"{f}_{k}": np.empty((len(ks), len(dl), B), np.float32) for f in ("d1", "d0", "delta") for k in ("raw", "cen")}
    t0 = time.time()

    def den(name, x, s):
        o = torch.empty_like(x)
        for b0 in range(0, len(x), a.batch):
            xx = x[b0:b0 + a.batch]
            tv = torch.full((len(xx),), float(s), device=dev)
            o[b0:b0 + a.batch] = E.call(name, xx.reshape(-1, 3, 64, 64), tv, lab[:len(xx)], fp32=True).reshape(len(xx), d)
        return o

    def unit_rows(V):
        return V / V.norm(dim=1, keepdim=True)

    with torch.no_grad():
        for li, s in enumerate(sig):
            x = A + float(s) * n
            F = {}
            for di, dd in enumerate(dl):
                sp = float(s) * math.exp(dd)
                D1 = den(d1n, x, sp)
                D0 = den(d0n, x, sp)
                F[di] = {"d1": D1, "d0": D0, "delta": D1 - D0}
            i0 = int(np.flatnonzero(dl == 0)[0])
            for f in ("d1", "d0", "delta"):
                ref = {"raw": unit_rows(F[i0][f]), "cen": unit_rows(F[i0][f] - F[i0][f].mean(0))}
                for di in range(len(dl)):
                    V = F[di][f]
                    res[f"{f}_raw"][li, di] = (ref["raw"] * unit_rows(V)).sum(1).cpu().numpy()
                    res[f"{f}_cen"][li, di] = (ref["cen"] * unit_rows(V - V.mean(0))).sum(1).cpu().numpy()
            del F
            m = res["delta_raw"][li].mean(1)
            print(f"  sigma {s:8.4f}: corr Delta vs delta " + " ".join(f"{dd:+g}:{v:.3f}" for dd, v in zip(dl, m))
                  + f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    atomic_npz(out / f"{a.tag}.npz", sigma=sig, levels=np.array(ks), dlogs=dl, ib=ib, rows=rows, **res)
    meta = dict(d1=d1n, d0=d0n, net=E.MODELS[d1n], guide=E.MODELS[d0n], cls=a.cls, images=M, n_base=B,
                device=torch.cuda.get_device_name(), minutes=(time.time() - t0) / 60)
    (out / f"{a.tag}-meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / (a.tag + '.npz')} ({meta['minutes']:.1f} min)", flush=True)


# ------------------------------------------------------------------ report
def pooled(dot, a2, b2):
    return dot.sum(-1) / np.sqrt(a2.sum(-1) * b2.sum(-1))


def crossing(x, y, level=E_HALF):
    """First x (log-interpolated) where y falls below level, scanning x upward; ('<', x0) if y starts below it,
    ('>', x_max) if it never falls below it."""
    o = np.argsort(x)
    x, y = np.asarray(x)[o], np.asarray(y)[o]
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[keep], y[keep]
    if not len(x):
        return ("nan", float("nan"))
    if y[0] < level:
        return ("<", float(x[0]))
    for i in range(1, len(x)):
        if y[i] < level:
            f = (y[i - 1] - level) / (y[i - 1] - y[i])
            return ("=", float(math.exp(math.log(x[i - 1]) + f * (math.log(x[i]) - math.log(x[i - 1])))))
    return (">", float(x[-1]))


def report_dir(D, nboot, seed=0):
    D = Path(D)
    meta = json.loads((D / "meta.json").read_text())
    z = np.load(D / "field.npz")
    P, sig, unit = list(z["names"]), z["sigma"], meta["unit"]
    S, _, N = z["dot"].shape
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, N, size=(nboot, N))
    noise_dir = [p for p in P if p.startswith("onshell") or p == "indep"]
    image_dir = [p for p in P if p.startswith("nnlerp") or p in ("nnimg", "rndimg")]
    xs = noise_dir + image_dir
    ts = [p for p in P if p.startswith("dlogs")]
    res = dict(meta=meta, levels=[])
    for k in range(S):
        lev = dict(sigma=float(sig[k]), mse_d1=float(z["mse"][k, 0]), mse_d0=float(z["mse"][k, 1]),
                   delta2_per_dim=float(np.nanmean(z["a2"][k, 0]) / meta["d"]))
        for kind, pre in (("raw", ""), ("centred", "c")):
            pts, tps = [], []
            for name in xs + ts:
                j = P.index(name)
                ok = ~np.isnan(z["dot"][k, j])
                if ok.sum() < 0.5 * N:            # on-shell grid points reachable for under half the bases
                    continue
                dot, a2, b2 = (z[pre + q][k, j] for q in ("dot", "a2", "b2"))
                rho = float(pooled(dot[ok], a2[ok], b2[ok]))
                dot, a2, b2 = (np.nan_to_num(q) for q in (dot, a2, b2))        # unreachable bases add 0
                bt = pooled(dot[boots], a2[boots], b2[boots])
                lo, hi = np.percentile(bt, [5, 95])
                if name in ts:
                    tps.append(dict(name=name, dlogs=float(name[5:]), rho=rho, lo=float(lo), hi=float(hi)))
                else:
                    r = float(np.nanmedian(z["r"][k, j])) / unit
                    pts.append(dict(name=name, r_units=r, rho=rho, lo=float(lo), hi=float(hi)))
            # the field is not isotropic: separate lengths along the noise and along image content
            lx = {fam: crossing([p["r_units"] for p in pts if p["name"] in names_],
                                [p["rho"] for p in pts if p["name"] in names_])
                  for fam, names_ in (("noise", noise_dir), ("image", image_dir))}
            # l_t: rho averaged over +-delta where both exist, else the one that does
            byabs = {}
            for p in tps:
                byabs.setdefault(abs(p["dlogs"]), []).append(p["rho"])
            ad = sorted(byabs)
            lt = crossing(ad, [float(np.mean(byabs[q])) for q in ad])
            lev[kind] = dict(x=pts, t=tps, l_x=lx, l_t=lt)
        res["levels"].append(lev)
    # along the sampler
    tz = np.load(D / "traj.npz")
    res["traj"] = {}
    for kind, key in (("raw", "gram"), ("centred", "cgram")):
        G = tz[key].sum(0)
        dg = np.sqrt(np.diag(G))
        R = G / dg[:, None] / dg[None]
        ls = np.log(tz["sigma"])
        fwd = []
        for i in range(S - 1):
            j = np.arange(i + 1, S)
            fwd.append(crossing(np.abs(ls[i] - ls[j]), R[i, j]))
        res["traj"][kind] = dict(rho=R.tolist(), l_t_forward=fwd, next_step_rho=[float(R[i, i + 1]) for i in range(S - 1)])
    (D / "lengths.json").write_text(json.dumps(res, indent=1))
    return res


def fmt(c):
    op, v = c
    return f"{'' if op == '=' else op}{v:.3g}"


def report(a):
    for D in a.dirs:
        res = report_dir(D, a.nboot)
        m = res["meta"]
        print(f"\n{D}: D1 = {m['d1']}, D0 = {m['d0']}, w = {m['w']}, unit = {m['unit']:.2f} px "
              f"(val bases {m['units']['val_bases']:.2f}, 10k train {m['units']['train10k']:.2f})")
        print(f"{'sigma':>8} {'lx noise':>9} {'lx image':>9} {'lxi cen':>9} {'rho_nn':>7} {'rho_rnd':>7} {'l_t raw':>8} {'l_t cen':>8} "
              f"{'traj l_t':>9} {'traj next':>9}")
        for k, lev in enumerate(res["levels"]):
            R, C = lev["raw"], lev["centred"]
            nn = next((p["rho"] for p in R["x"] if p["name"] == "nnimg"), float("nan"))
            rd = next((p["rho"] for p in R["x"] if p["name"] == "rndimg"), float("nan"))
            tr = res["traj"]["raw"]
            tl = fmt(tr["l_t_forward"][k]) if k < len(tr["l_t_forward"]) else "-"
            tn = f"{tr['next_step_rho'][k]:.3f}" if k < len(tr["next_step_rho"]) else "-"
            print(f"{lev['sigma']:8.4f} {fmt(R['l_x']['noise']):>9} {fmt(R['l_x']['image']):>9} "
                  f"{fmt(C['l_x']['image']):>9} {nn:7.3f} {rd:7.3f} "
                  f"{fmt(R['l_t']):>8} {fmt(C['l_t']):>8} {tl:>9} {tn:>9}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    w = sub.add_parser("work")
    w.add_argument("--pair", choices=sorted(PAIRS))
    w.add_argument("--d1")
    w.add_argument("--d0")
    w.add_argument("--w", type=float, default=2.0)
    w.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    w.add_argument("--out", required=True)
    w.add_argument("--n", type=int, default=1000)
    w.add_argument("--n-traj", type=int, default=1000)
    w.add_argument("--batch", type=int, default=125)
    pp = sub.add_parser("pairs")
    pp.add_argument("--pair", choices=sorted(PAIRS))
    pp.add_argument("--d1")
    pp.add_argument("--d0")
    pp.add_argument("--w", type=float, default=2.0)
    pp.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    pp.add_argument("--out", required=True)
    pp.add_argument("--n", type=int, default=1000)
    pp.add_argument("--n-sub", type=int, default=5000)
    pp.add_argument("--denoised", action="store_true",
                    help="also the pair correlation of D1, D0 and the guided D0 + w (D1 - D0) themselves")
    pp.add_argument("--classes", type=int, default=0, help="K fixed classes (seed s -> class s %% K); 0: per seed")
    pp.add_argument("--tag", default="pairs", help="output name: TAG.npz, TAG-meta.json")
    rx = sub.add_parser("random")
    rx.add_argument("--pair", choices=sorted(PAIRS))
    rx.add_argument("--d1")
    rx.add_argument("--d0")
    rx.add_argument("--w", type=float, default=2.0)
    rx.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    rx.add_argument("--out", required=True)
    rx.add_argument("--cls", type=int, default=850)
    rx.add_argument("--per-image", type=int, default=80)
    rx.add_argument("--n-sub", type=int, default=400000)
    rx.add_argument("--batch", type=int, default=250)
    rx.add_argument("--tag", default="random-c850")
    r = sub.add_parser("report")
    sp = sub.add_parser("spread")
    sp.add_argument("--d1", default="S")
    sp.add_argument("--d0", default="XS")
    sp.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    sp.add_argument("--out", required=True)
    sp.add_argument("--cls", type=int, default=850)
    sp.add_argument("--levels", default="0,8,14,20,26", help="indices into EDM2's 32 sampler levels")
    sp.add_argument("--n-base", type=int, default=1000)
    sp.add_argument("--per-base", type=int, default=15)
    sp.add_argument("--tmin", type=float, default=1e-4)
    sp.add_argument("--batch", type=int, default=250)
    sp.add_argument("--tag", default="spread-c850")
    st = sub.add_parser("spreadt")
    st.add_argument("--d1", default="S")
    st.add_argument("--d0", default="XS")
    st.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    st.add_argument("--out", required=True)
    st.add_argument("--cls", type=int, default=850)
    st.add_argument("--levels", default="8,14,20,26")
    st.add_argument("--n-base", type=int, default=1000)
    st.add_argument("--batch", type=int, default=250)
    st.add_argument("--tag", default="spreadt-c850")
    r.add_argument("dirs", nargs="+")
    r.add_argument("--nboot", type=int, default=500)
    a = ap.parse_args()
    if a.stage in ("work", "pairs", "random", "spread", "spreadt"):
        assert getattr(a, "pair", None) or (a.d1 and a.d0), "--pair, or --d1 and --d0"
        dict(work=work, pairs=pairs, random=random_x, spread=spread, spreadt=spread_t)[a.stage](a)
    else:
        report(a)


if __name__ == "__main__":
    main()
