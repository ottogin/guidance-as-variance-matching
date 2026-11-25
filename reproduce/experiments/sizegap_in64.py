#!/usr/bin/env python
"""What the difference between an EDM2-S and an EDM2-XS prediction is made of (ImageNet-64, 32 Mi).

The paper's model of a guided pair (Sec. 5.2) gives each network the posterior mean plus an independent,
mean-zero error over retrainings, so any independent pair, two networks of one size included, reaches the
posterior's covariance at its own w. Sampling disagrees: our S guided by another S seed gains nothing for w in
[-1, 3], guided by an XS it gains 2.7 FID. This script splits D_S - D_XS into the part that changes with the
training seed and the part every seed shares, and measures both.

Members, all at 32 Mi and one EMA: S1 (ours) and Sn (NVIDIA) are EDM2-S; A, B (ours) and N (NVIDIA) are EDM2-XS.
On forward-noised validation images, with r_i = D_i - x0 and products per coordinate:

    seed part       V_S = 1/2 E|S1 - Sn|^2,  V_XS = 1/2 E|X - X'|^2  (X, X' two XS seeds)
    size part       E<S - X, S' - X'>      over pairs that share no network: |E_seed D_S - E_seed D_XS|^2
    cross-seed      Q_S = E<r_S, r_S'> = E tr Sigma + |bias_S|^2, and Q_XS, Q_SX alike

    python reproduce/experiments/sizegap_in64.py gram  --ema 0.100 --shard i --shards 4   # GPU: Grams, spectra
    python reproduce/experiments/sizegap_in64.py tcorr --ema 0.100                        # GPU: across noise levels
    python reproduce/experiments/sizegap_in64.py report                                   # CPU: tables, sizegap.json

Outputs: $VMG_RUNS/edm2-64-seedpair/sizegap/.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.experiments import seedpair_in64 as sp  # noqa: E402

S_NAMES, X_NAMES = ["S1", "Sn"], ["A", "B", "N"]
MEMBERS = S_NAMES + X_NAMES
STEPS, SMIN, SMAX, RHO = 32, 0.002, 80.0, 7.0


def out_dir():
    d = sp.out_root() / "sizegap"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sigmas():
    i = np.arange(STEPS, dtype=np.float32)
    a, b = np.float32(SMAX) ** (1 / RHO), np.float32(SMIN) ** (1 / RHO)
    return ((a + i / (STEPS - 1) * (b - a)) ** RHO).astype(np.float64)


def device():
    import torch
    if torch.cuda.is_available():
        return "cuda", "cuda"
    return ("mps" if torch.backends.mps.is_available() else "cpu"), "cpu"      # MPS has no float64: sums on the CPU


def load_nets(ema, dev):
    sp.edm2_on_path()
    nets = {}
    for m in MEMBERS:
        with open(sp.snapshot(m, ema), "rb") as f:
            nets[m] = pickle.load(f)["ema"].to(dev).eval().requires_grad_(False)
    return nets


def calib(n):
    import torch
    from vmg.data import imagenet
    rows = imagenet.calib_rows(n)
    return rows, imagenet.pixels(rows), torch.nn.functional.one_hot(imagenet.class_of(rows, 64), 1000).float()


def levels_arg(a):
    ks = list(range(STEPS)) if a.levels == "all" else [int(k) for k in a.levels.split(",")]
    return [k for j, k in enumerate(ks) if j % a.shards == a.shard]


# ------------------------------------------------------------------ GPU passes
def gram(a):
    """Per level: Gram of [D_1 .. D_K, x0] and their means (the format of ens_edm2.py moments), the Gram of
    [r_1 .. r_K, x0] per image, and its cross power spectrum per 2-D frequency (summed over colour channels)."""
    import torch
    dev, acc = device()
    nets = load_nets(a.ema, dev)
    rows, x0all, lab = calib(a.n)
    sig, ks, K, n = sigmas(), levels_arg(a), len(MEMBERS), len(rows)
    d = x0all[0].numel()
    G = np.zeros((len(ks), K + 1, K + 1))
    mean = np.zeros((len(ks), K + 1, d))
    Gz = np.zeros((len(ks), n, K + 1, K + 1), np.float32)
    spec = np.zeros((len(ks), K + 1, K + 1, 64, 64))
    t0 = time.time()
    for j, k in enumerate(ks):
        g = torch.Generator().manual_seed(1000 + k)                 # the noise of ens_edm2.py moments at this level
        for b0 in range(0, n, a.batch):
            x0 = x0all[b0:b0 + a.batch].to(dev)
            x = x0 + float(sig[k]) * torch.randn(x0.shape, generator=g).to(dev)
            t = torch.full((len(x),), float(sig[k]), device=dev)
            L = lab[b0:b0 + a.batch].to(dev)
            with torch.no_grad():
                D = [nets[m](x, t, L, force_fp32=True) for m in MEMBERS]
            Y = torch.stack([y.reshape(len(x), -1).to(acc, torch.float64) for y in D + [x0]])       # [K+1, B, d]
            G[j] += torch.einsum("ibd,jbd->ij", Y, Y).cpu().numpy()
            mean[j] += Y.sum(1).cpu().numpy()
            Z = torch.cat([Y[:K] - Y[K:], Y[K:]])                                                  # residuals, x0
            Gz[j, b0:b0 + len(x)] = torch.einsum("ibd,jbd->bij", Z, Z).float().cpu().numpy()
            F = torch.fft.fft2(Z.reshape(K + 1, len(x), 3, 64, 64), norm="ortho")
            spec[j] += torch.einsum("ibchw,jbchw->ijhw", F, F.conj()).real.cpu().numpy()
        print(f"level {k} sigma {sig[k]:.4g} done ({time.time() - t0:.0f}s)", flush=True)
    tag = f"{a.tag}e{a.ema[2:]}"
    np.savez(out_dir() / f"gram-{tag}-{a.shard:02d}of{a.shards:02d}.npz", names=np.array(MEMBERS), sigma=sig[ks],
             levels=np.array(ks), n=n, d=d, rows=np.array(rows), gram=G / n, mean=mean / n, gram_img=Gz, spec=spec / n)
    print("wrote", out_dir() / f"gram-{tag}-{a.shard:02d}of{a.shards:02d}.npz", flush=True)


def tcorr(a):
    """Gram of the residuals across noise levels, each image keeping one noise direction at every level:
    G[k, i, k', j] = E<r_i(x0 + sigma_k eps), r_j(x0 + sigma_k' eps)>."""
    import torch
    dev, acc = device()
    nets = load_nets(a.ema, dev)
    rows, x0all, lab = calib(a.n)
    sig, ks, K, n = sigmas(), levels_arg(a), len(MEMBERS), len(rows)
    G = np.zeros((len(ks) * K, len(ks) * K))
    g = torch.Generator().manual_seed(4242)
    t0 = time.time()
    for b0 in range(0, n, a.batch):
        x0 = x0all[b0:b0 + a.batch].to(dev)
        eps = torch.randn(x0.shape, generator=g).to(dev)
        L = lab[b0:b0 + a.batch].to(dev)
        Z = []
        for k in ks:
            t = torch.full((len(x0),), float(sig[k]), device=dev)
            with torch.no_grad():
                Z += [(nets[m](x0 + float(sig[k]) * eps, t, L, force_fp32=True) - x0).reshape(-1) for m in MEMBERS]
        Z = torch.stack(Z).to(acc, torch.float64)                                                  # [S K, B d]
        G += (Z @ Z.T).cpu().numpy()
        print(f"{b0 + len(x0)}/{n} ({time.time() - t0:.0f}s)", flush=True)
    tag = f"{a.tag}e{a.ema[2:]}"
    np.savez(out_dir() / f"tcorr-{tag}.npz", names=np.array(MEMBERS), sigma=sig[ks], levels=np.array(ks), n=n,
             d=x0all[0].numel(), gram=(G / n).reshape(len(ks), K, len(ks), K))
    print("wrote", out_dir() / f"tcorr-{tag}.npz", flush=True)


# ------------------------------------------------------------------ how far guidance moves the final images
def _arm_images(arg):
    arm, n = arg
    from PIL import Image
    d = sp.out_root() / "samples" / arm
    if not all((d / f"{i:05d}.png").exists() for i in (0, n - 1)):
        return None
    return np.stack([np.asarray(Image.open(d / f"{i:05d}.png").convert("RGB"), np.uint8).reshape(-1) for i in range(n)])


def shift(a):
    """The sampler's response to each part. Every arm shares its seeds and classes, so for seed r the guide G
    moves S1's image by d_G(w) = x[S1 <- G, w] - x[S1]. Two XS guides (A, N) share the size part and S1's own
    seed part; an XS and an S guide (Sn) share only S1's seed part. To first order in the denoiser change,

        <d_A, d_N> = size + seed_S1,   <d_A, d_Sn> = <d_N, d_Sn> = seed_S1,   |d_A|^2 = size + seed_S1 + seed_A.

    Per coordinate, pixels in [-1, 1]; the 8-bit rounding of the PNGs is subtracted (q per image)."""
    from multiprocessing import Pool
    n, q = a.n, (2 / 255) ** 2 / 12
    out = {}
    for ema in a.emas.split(","):
        e = ema[2:]
        arms = {"S1": f"sw-S1Sn-e{e}-w1", "Sn": f"sw-S1Sn-e{e}-w0", "SS": f"sw-S1Sn-e{e}-w0.5"}
        ws = a.w.split(",")
        for w in ws:
            arms.update({f"A{w}": f"sw-S1A-e{e}-w{w}", f"N{w}": f"sw-S1N-e{e}-w{w}", f"S{w}": f"sw-S1Sn-e{e}-w{w}",
                         f"nN{w}": f"sw-SnN-e{e}-w{w}", f"E{w}": f"sw-EsEx-e{e}-w{w}"})
        with Pool(a.workers) as pool:
            ims = dict(zip(arms, pool.map(_arm_images, [(v, n) for v in arms.values()])))
        ims = {k: v for k, v in ims.items() if v is not None}                    # uint8; differences are taken per w
        d = ims["S1"].shape[1]
        print(f"\n=== EMA {ema}: {n} seeds; arms missing: {sorted(set(arms) - set(ims)) or 'none'} ===")
        print("Squared shift of the final image per coordinate (x 1e4) and cosines between shifts, pooled over seeds.")
        print(f"{'w':>5} {'|d_A|^2':>8} {'|d_N|^2':>8} {'|d_Sn|^2':>8} {'|d_E|^2':>8} {'cos A,N':>8} {'cos A,Sn':>8} {'cos N,Sn':>8} "
              f"{'cos N,nN':>8} {'cos E,A':>8} | {'size':>7} {'seed S1':>7} {'seed XS':>7} {'size share':>10}")
        out[ema] = {}
        for w in ws:
            f = {}
            for key, base in ((f"A{w}", "S1"), (f"N{w}", "S1"), (f"S{w}", "S1"), (f"nN{w}", "Sn"), (f"E{w}", "SS")):
                if key in ims and base in ims:
                    f[key[:-len(w)]] = ((ims[key].astype(np.float32) - ims[base].astype(np.float32)) / 127.5, base)
            if not {"A", "N", "S"} <= set(f):
                continue

            def ip(u, v):       # E<d_u, d_v> / d, minus the rounding noise the two shifts share
                c = 2 * q if u == v else (q if f[u][1] == f[v][1] else 0.0)
                return float(np.einsum("nd,nd->", f[u][0], f[v][0], dtype=np.float64)) / (n * d) - c

            cos = lambda u, v: ip(u, v) / np.sqrt(ip(u, u) * ip(v, v)) if u in f and v in f else float("nan")
            seed_s = (ip("A", "S") + ip("N", "S")) / 2
            size = ip("A", "N") - seed_s
            tot = (ip("A", "A") + ip("N", "N")) / 2
            row = dict(dA=ip("A", "A"), dN=ip("N", "N"), dS=ip("S", "S"), dE=ip("E", "E") if "E" in f else float("nan"),
                       cos_AN=cos("A", "N"), cos_AS=cos("A", "S"), cos_NS=cos("N", "S"), cos_NnN=cos("N", "nN"),
                       cos_EA=cos("E", "A"), size=size, seed_s=seed_s, seed_x=tot - ip("A", "N"), size_share=size / tot)
            out[ema][w] = row
            print(f"{w:>5} {row['dA'] * 1e4:8.2f} {row['dN'] * 1e4:8.2f} {row['dS'] * 1e4:8.2f} {row['dE'] * 1e4:8.2f} "
                  f"{row['cos_AN']:8.3f} {row['cos_AS']:8.3f} {row['cos_NS']:8.3f} {row['cos_NnN']:8.3f} {row['cos_EA']:8.3f} | "
                  f"{size * 1e4:7.2f} {seed_s * 1e4:7.2f} {row['seed_x'] * 1e4:7.2f} {row['size_share']:10.3f}", flush=True)
    (out_dir() / "shift.json").write_text(json.dumps(out))
    print("\nwrote", out_dir() / "shift.json")


# ------------------------------------------------------------------ where guidance moves the sample distribution
def _arm_stats(arm):
    """Per detector: the arm's mean feature and the two terms of its Frechet distance to NVIDIA's training statistics."""
    from vmg.metrics import image as I
    p = sp.out_root() / "metrics" / f"{arm}.feats.npz"
    if not p.exists():
        return None
    z, ref, out = np.load(p), I.nvidia_stats(64), {}
    for det in I.DETECTORS:
        X = z[det].astype(np.float64)
        mu, cov = X.mean(0), np.cov(X, rowvar=False)
        mr, sr = ref[I.FD_NAME[det]]
        # tr sqrtm(cov sr) = sum of sqrt(eig(sr^1/2 cov sr^1/2)): two symmetric eigenproblems instead of a Schur form
        lam, U = np.linalg.eigh(sr)
        h = (U * np.sqrt(np.clip(lam, 0, None))) @ U.T
        cross = np.sqrt(np.clip(np.linalg.eigvalsh(h @ cov @ h), 0, None)).sum()
        out[det] = dict(mu=mu, mean_term=float(np.square(mu - mr).sum()),
                        cov_term=float(np.trace(cov) + np.trace(sr) - 2 * cross))
    return out


def feat(a):
    """In the feature space of each metric: the mean shift m_G = mean phi(x[base <- G, w]) - mean phi(x[base]) of
    every guide, the part of the per-image feature shift that this common shift accounts for, its direction
    against the XS -> S change (mean phi(S) - mean phi(XS)) and against the data (mean phi(data) - mean phi(S)),
    and the mean and covariance terms of the Frechet distance."""
    from multiprocessing import Pool
    from vmg.metrics import image as I
    ref = I.nvidia_stats(64)
    out = {}
    for ema in a.emas.split(","):
        e = ema[2:]
        arms = {"S1": f"sw-S1Sn-e{e}-w1", "Sn": f"sw-S1Sn-e{e}-w0", "SS": f"sw-S1Sn-e{e}-w0.5",
                "A": f"sw-AN-e{e}-w1", "N": f"sw-AN-e{e}-w0"}
        pairs = {"A": ("S1A", "S1"), "N": ("S1N", "S1"), "Sn": ("S1Sn", "S1"), "nN": ("SnN", "Sn"), "E": ("EsEx", "SS")}
        ws = a.w.split(",")
        for w in ws:
            arms.update({f"{g}@{w}": f"sw-{pre}-e{e}-w{w}" for g, (pre, _) in pairs.items()})
        with Pool(a.workers) as pool:
            st = dict(zip(arms, pool.map(_arm_stats, list(arms.values()))))
        st = {k: v for k, v in st.items() if v is not None}
        out[ema] = {}
        for det in I.DETECTORS:
            F = {k: np.load(sp.out_root() / "metrics" / f"{arms[k]}.feats.npz")[det].astype(np.float64) for k in st}
            n = len(F["S1"])
            mr = ref[I.FD_NAME[det]][0]
            mu = {k: st[k][det]["mu"] for k in st}
            g_size, g_data = mu["S1"] - (mu["A"] + mu["N"]) / 2, mr - mu["S1"]
            cos = lambda u, v: float(u @ v / np.sqrt((u @ u) * (v @ v)))
            print(f"\n=== EMA {ema}, {I.FD_NAME[det]} features ({F['S1'].shape[1]}-d, {n} seeds) ===")
            print(f"|mean(S) - mean(XS)|^2 = {g_size @ g_size:.4g}; |mean(S1) - mean(Sn)|^2 = {np.square(mu['S1'] - mu['Sn']).sum():.4g} "
                  f"(floor from {n} seeds {np.square(F['S1'] - F['Sn'] - (mu['S1'] - mu['Sn'])).sum(1).mean() / n:.4g}); "
                  f"|mean(data) - mean(S1)|^2 = {g_data @ g_data:.4g}")
            for k in ("S1", "Sn", "SS", "A", "N"):
                if k in st:
                    print(f"  {k:>3} alone: FD {st[k][det]['mean_term'] + st[k][det]['cov_term']:8.3f} = mean term "
                          f"{st[k][det]['mean_term']:7.3f} + covariance term {st[k][det]['cov_term']:8.3f}")
            print(f"{'guide':>6} {'w':>4} {'FD':>8} {'mean':>7} {'cov':>8} | {'|m|^2':>8} {'floor':>7} {'common share':>12} "
                  f"{'cos(m, S-XS)':>12} {'steps':>6} {'cos(m, data-S)':>14} | {'cos(m, m_A)':>11}")
            res = {}
            for w in ws:
                for g, (pre, base) in pairs.items():
                    k = f"{g}@{w}"
                    if k not in st or base not in st:
                        continue
                    dphi = F[k] - F[base]
                    m = dphi.mean(0)
                    en = float(np.square(dphi).sum(1).mean())
                    floor = float(np.square(dphi - m).sum(1).mean()) / n
                    mA = F[f"A@{w}"].mean(0) - mu["S1"] if f"A@{w}" in F else None
                    row = dict(fd=st[k][det]["mean_term"] + st[k][det]["cov_term"], mean_term=st[k][det]["mean_term"],
                               cov_term=st[k][det]["cov_term"], m2=float(m @ m), floor=floor, common=float(m @ m) / en,
                               cos_size=cos(m, g_size), steps=float(m @ g_size / (g_size @ g_size)), cos_data=cos(m, g_data),
                               cos_mA=cos(m, mA) if mA is not None else float("nan"))
                    res[k] = row
                    print(f"{g:>6} {w:>4} {row['fd']:8.3f} {row['mean_term']:7.3f} {row['cov_term']:8.3f} | {row['m2']:8.4f} "
                          f"{floor:7.4f} {row['common']:12.4f} {row['cos_size']:12.3f} {row['steps']:6.2f} {row['cos_data']:14.3f} | "
                          f"{row['cos_mA']:11.3f}", flush=True)
            out[ema][det] = dict(arms=res, alone={k: dict(mean_term=st[k][det]["mean_term"], cov_term=st[k][det]["cov_term"])
                                                  for k in ("S1", "Sn", "SS", "A", "N") if k in st},
                                 g_size2=float(g_size @ g_size), g_data2=float(g_data @ g_data))
        (out_dir() / "feat.json").write_text(json.dumps(out))          # after every EMA, so a time limit loses one at most
    print("\nwrote", out_dir() / "feat.json")


# ------------------------------------------------------------------ CPU report
def load_gram(tag):
    """The shards of one EMA, levels in sampler order."""
    ps = sorted(out_dir().glob(f"gram-{tag}-*of*.npz"))
    if not ps:
        return None
    zs = [np.load(p) for p in ps]
    assert len(zs) == int(ps[0].stem.split("of")[-1]), f"{len(zs)} shards of {ps[0].stem}"
    lev = np.concatenate([z["levels"] for z in zs])
    o = np.argsort(lev)
    cat = lambda key: np.concatenate([z[key] for z in zs])[o]
    return dict(names=[str(x) for x in zs[0]["names"]], levels=lev[o], sigma=cat("sigma"), n=int(zs[0]["n"]),
                d=int(zs[0]["d"]), gram=cat("gram"), mean=cat("mean"), gram_img=cat("gram_img"), spec=cat("spec"))


def split(Gz, names):
    """Seed and size parts from a residual Gram Gz[..., K+1, K+1] (any leading axes), per image, not yet / d."""
    s = [names.index(m) for m in S_NAMES]
    x = [names.index(m) for m in X_NAMES]
    x0 = len(names)
    g = lambda i, j: Gz[..., i, j]
    avg = lambda pairs: sum(g(i, j) for i, j in pairs) / len(pairs)
    ss, xx = [(s[0], s[1])], [(i, j) for i in x for j in x if i < j]
    sx = [(i, j) for i in s for j in x]
    mse_s, mse_x = avg([(i, i) for i in s]), avg([(i, i) for i in x])
    q_s, q_x, q_sx = avg(ss), avg(xx), avg(sx)                    # cross-seed products: E tr Sigma + <bias, bias'>
    return dict(mse_s=mse_s, mse_x=mse_x, q_s=q_s, q_x=q_x, q_sx=q_sx,
                v_s=mse_s - q_s, v_x=mse_x - q_x,                # seed variance of one network
                size=q_s + q_x - 2 * q_sx,                       # |bias_S - bias_XS|^2
                total=mse_s + mse_x - 2 * q_sx,                  # E|D_S - D_XS|^2 for one pair
                sig_s=avg([(i, x0) for i in s]), sig_x=avg([(i, x0) for i in x]))     # E<r, x0>


def radial(P):
    """[..., 64, 64] power per 2-D frequency -> [..., 33] summed over rings of integer radius (corners in the last)."""
    f = np.fft.fftfreq(64) * 64
    r = np.minimum(np.rint(np.hypot(f[:, None], f[None, :])).astype(int), 32)
    out = np.zeros(P.shape[:-2] + (33,))
    for q in range(33):
        out[..., q] = P[..., r == q].sum(-1)
    return out


def variance_curve(m, w):
    """tr Cov_{x_t}(D_w) / d at each level for the pair moments m (solve.Moments) and a weight w."""
    c00, c01, c11 = m.rr_centred.T
    return m.var_x0 + 2 * ((1 - w) * m.xr[:, 0] + w * m.xr[:, 1]) + (1 - w) ** 2 * c00 + 2 * w * (1 - w) * c01 + w * w * c11


def report(a):
    from vmg import solve
    res = {}
    for ema in a.emas.split(","):
        tag = f"{a.tag}e{ema[2:]}"
        g = load_gram(tag)
        if g is None:
            continue
        names, sig, d, n = g["names"], g["sigma"], g["d"], g["n"]
        K = len(names)
        Gz = g["gram_img"].astype(np.float64)                                   # [S, n, K+1, K+1]
        p = {k: v.mean(1) / d for k, v in split(Gz, names).items()}             # per level, per coordinate
        # the mean over images of each residual, for centred versions of the bias terms
        rb = g["mean"][:, :K] - g["mean"][:, K:]
        xbar = g["mean"][:, K]
        var_x0 = float(np.mean(g["gram"][:, K, K] - np.einsum("sd,sd->s", xbar, xbar))) / d
        # bootstrap over images (the same resamples for every quantity): [S, 200] means per coordinate
        per = split(Gz, names)
        W = np.random.default_rng(0).multinomial(n, np.full(n, 1 / n), size=200).T / n       # [n, 200]
        bt = {k: v @ W / d for k, v in per.items()}
        lo, hi = np.percentile(bt["size"] / bt["total"], [5, 95], axis=1)
        share = p["size"] / p["total"]
        w_mse_size = (p["q_x"] - p["q_sx"]) / p["size"]           # MSE-optimal w of the seed-averaged pair
        w_mse_pair = (p["mse_x"] - p["q_sx"]) / p["total"]        # ... of one pair (eq. wmse)
        # the same quantities with the levels of a noise band summed (numerators and denominators apart)
        bands_sigma = {"sigma >= 10": sig >= 10, "1 - 10": (sig >= 1) & (sig < 10), "0.1 - 1": (sig >= 0.1) & (sig < 1),
                       "< 0.1": sig < 0.1, "all levels": sig > 0}
        pooled = {}
        for name, sel in bands_sigma.items():
            if not sel.any():
                continue
            f = {k: (p[k][sel].sum(), bt[k][sel].sum(0)) for k in p}
            q = {"size x1e4": lambda u: u["size"] * 1e4 / sel.sum(),
                 "size share": lambda u: u["size"] / u["total"],
                 "V_XS / V_S": lambda u: u["v_x"] / u["v_s"],
                 "bias gap / size": lambda u: (u["q_x"] - u["q_s"]) / u["size"],
                 "wMSE size": lambda u: (u["q_x"] - u["q_sx"]) / u["size"],
                 "wMSE pair": lambda u: (u["mse_x"] - u["q_sx"]) / u["total"]}
            pooled[name] = {}
            for key, fn in q.items():
                v = float(fn({k: f[k][0] for k in f}))
                b = np.percentile(fn({k: f[k][1] for k in f}), [5, 95])
                pooled[name][key] = [v, float(b[0]), float(b[1])]
        # E<D_S - D_XS, x0 - xbar>: how much of the difference is signal (1 x size if XS is S with information removed)
        s_i, x_i = [names.index(m) for m in S_NAMES], [names.index(m) for m in X_NAMES]
        cen = lambda idx: np.mean([np.einsum("sd,sd->s", rb[:, i], xbar) for i in idx], 0) / d
        signal = (p["sig_s"] - cen(s_i)) - (p["sig_x"] - cen(x_i))
        print(f"\n=== EMA {ema}: n = {n} images, {len(sig)} levels; Var x0 = {var_x0:.4f} per coordinate ===")
        print("Per coordinate, x 1e4.  V = seed variance of one network; size = |E_seed D_S - E_seed D_XS|^2;\n"
              "total = E|D_S - D_XS|^2 of one pair = size + V_S + V_XS; MSE gap = MSE_XS - MSE_S.")
        print(f"{'sigma':>8} {'MSE_S':>8} {'MSEgap':>7} {'V_S':>7} {'V_XS':>7} {'size':>7} {'total':>7} {'size share (5-95%)':>20} "
              f"{'gap-(V_XS-V_S)':>14} {'wMSE size':>9} {'wMSE pair':>9} {'signal/size':>11}")
        for k in range(len(sig)):
            print(f"{sig[k]:8.3f} {p['mse_s'][k] * 1e4:8.1f} {(p['mse_x'][k] - p['mse_s'][k]) * 1e4:7.2f} {p['v_s'][k] * 1e4:7.3f} "
                  f"{p['v_x'][k] * 1e4:7.3f} {p['size'][k] * 1e4:7.3f} {p['total'][k] * 1e4:7.3f} "
                  f"{share[k]:8.3f} ({lo[k]:.3f}-{hi[k]:.3f}) {(p['q_x'][k] - p['q_s'][k]) * 1e4:14.3f} {w_mse_size[k]:9.2f} "
                  f"{w_mse_pair[k]:9.2f} {signal[k] / p['size'][k]:11.2f}")
        print("\nLevels of a noise band summed, with 5-95 % over images. bias gap = |bias_XS|^2 - |bias_S|^2;\n"
              "wMSE size = the weight that brings the seed-averaged guided prediction closest to x0.")
        keys = list(next(iter(pooled.values())))
        print(f"{'band':>12} " + " ".join(f"{k:>26}" for k in keys))
        for name, row in pooled.items():
            print(f"{name:>12} " + " ".join(f"{row[k][0]:9.3f} ({row[k][1]:7.3f},{row[k][2]:7.3f})" for k in keys))
        # spectra: seed part, size part and the posterior's (cross-seed residual product) by radial frequency
        sp_ = split(np.moveaxis(radial(g["spec"]), -1, 1), names)                # [S, 33] each, not / d
        spec = {k: sp_[k] / d for k in ("v_s", "v_x", "size", "q_s", "total")}
        bands = {"low (0-3)": slice(0, 4), "mid (4-12)": slice(4, 13), "high (13+)": slice(13, 33)}
        print("\nShare of each part's power in three frequency bands (cycles per image), and the part relative to the "
              "posterior's power\n(cross-seed residual product of S) in that band, x 1e3:")
        print(f"{'sigma':>8} " + " ".join(f"{part + ' ' + b:>22}" for part in ("seed S", "size", "posterior") for b in bands))
        for k in range(len(sig)):
            cells = []
            for part in ("v_s", "size", "q_s"):
                for b in bands.values():
                    frac = spec[part][k, b].sum() / spec[part][k].sum()
                    rel = spec[part][k, b].sum() / spec["q_s"][k, b].sum()
                    cells.append(f"{frac:10.3f} ({rel * 1e3:8.2f})")
            print(f"{sig[k]:8.3f} " + " ".join(cells))
        # variance-matched weights of the pairs, and the share of the missing variance each tested w restores
        gg = dict(names=names, sigma=sig, n=n, d=d, gram=g["gram"], mean=g["mean"])
        vm = {}
        print("\nVariance matching (eq. 18-20), and the share of the missing variance restored at w, summed over levels:")
        ws = [1.5, 2, 2.5, 3, 4, 6, 8]
        print(f"{'pair':>10} {'w_VM':>7} {'ok':>5} " + " ".join(f"{'w=' + str(w):>8}" for w in ws) + "   Var[D1-D0] / missing")
        for d1, d0 in (("S1", "Sn"), ("S1", "A"), ("S1", "N"), ("Sn", "N"), ("Sn", "A"), ("A", "B"), ("A", "N")):
            m = sp.pair_moments(gg, d1, d0)
            wg, ok = solve.fit_global(m)
            miss = (m.var_x0 - variance_curve(m, 1.0)).sum()
            fr = [float((variance_curve(m, w) - variance_curve(m, 1.0)).sum() / miss) for w in ws]
            c00, c01, c11 = m.rr_centred.T
            aa = float((c00 - 2 * c01 + c11).sum() / miss)
            vm[f"{d1}<-{d0}"] = dict(w_vm=float(wg), admissible=bool(ok), w=ws, restored=fr, a_over_missing=aa,
                                     w_schedule=solve.fit_schedule(m)["w"].tolist())
            print(f"{d1 + '<-' + d0:>10} {wg:7.2f} {str(bool(ok)):>5} " + " ".join(f"{f:8.3f}" for f in fr) + f"   {aa:.4f}")
        res[ema] = dict(sigma=sig.tolist(), n=n, var_x0=var_x0, pooled=pooled, **{k: v.tolist() for k, v in p.items()},
                        size_share=share.tolist(), size_share_lo=lo.tolist(), size_share_hi=hi.tolist(),
                        w_mse_size=w_mse_size.tolist(), w_mse_pair=w_mse_pair.tolist(), signal=signal.tolist(),
                        spec={k: v.tolist() for k, v in spec.items()}, vm=vm)
        tp = out_dir() / f"tcorr-{tag}.npz"
        if tp.exists():
            z = np.load(tp)
            T, tn, ts = z["gram"], [str(x) for x in z["names"]], z["sigma"]
            s, x = [tn.index(m) for m in S_NAMES], [tn.index(m) for m in X_NAMES]
            blk = lambda i, j: T[:, i, :, j]                                    # [S, S]
            avg = lambda pairs: sum(blk(i, j) for i, j in pairs) / len(pairs)
            q_s = (blk(s[0], s[1]) + blk(s[1], s[0])) / 2
            q_x = avg([(i, j) for i in x for j in x if i != j])
            q_sx = (avg([(i, j) for i in s for j in x]) + avg([(j, i) for i in s for j in x])) / 2
            seed = avg([(i, i) for i in s]) - q_s                               # E<e_S(k), e_S(k')>
            size = q_s + q_x - 2 * q_sx
            corr = lambda C: C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
            cs, cz, cp = corr(seed), corr(size), corr(q_s)
            print(f"\nCorrelation across noise levels for one image and one noise direction (n = {int(z['n'])}): "
                  "seed part of S, size part, posterior residual")
            print(f"{'sigma':>8} " + " ".join(f"{h:>22}" for h in ("next level", "2 levels on", "4 levels on")))
            for k in range(len(ts)):
                cells = [" ".join(f"{C[k, k + l]:6.2f}" if k + l < len(ts) else f"{'':>6}" for C in (cs, cz, cp)) for l in (1, 2, 4)]
                print(f"{ts[k]:8.3f} " + " ".join(f"{c:>22}" for c in cells))
            res[ema]["tcorr"] = dict(sigma=ts.tolist(), seed=cs.tolist(), size=cz.tolist(), posterior=cp.tolist())
    (out_dir() / f"sizegap{a.tag and '-' + a.tag}.json").write_text(json.dumps(res))
    print("\nwrote", out_dir() / f"sizegap{a.tag and '-' + a.tag}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["gram", "tcorr", "shift", "feat", "report"])
    ap.add_argument("--w", default="1.5,2,2.5,3", help="shift: the weights to compare")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ema", default="0.100")
    ap.add_argument("--emas", default="0.050,0.100", help="report: the EMAs to read")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--levels", default="all", help="sampler levels (0-31), comma-separated")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--tag", default="", help="prefix of the output names (e.g. a small local run)")
    a = ap.parse_args()
    if a.stage == "gram":
        a.n, a.batch = a.n or 10000, a.batch or 100
        gram(a)
    elif a.stage == "tcorr":
        a.n, a.batch = a.n or 1000, a.batch or 20
        tcorr(a)
    elif a.stage == "shift":
        a.n = a.n or 10000
        shift(a)
    elif a.stage == "feat":
        feat(a)
    else:
        report(a)


if __name__ == "__main__":
    main()
