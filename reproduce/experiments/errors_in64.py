#!/usr/bin/env python
"""Errors of EDM2 ImageNet-64 denoisers on the same noised validation images (fig:shared-error, fig:mean-worse).

    python reproduce/experiments/errors_in64.py fetch                                # CPU: checkpoints, img64.pkl
    python reproduce/experiments/errors_in64.py gram --shard i --shards 8            # GPU: per-image error Grams
    python reproduce/experiments/errors_in64.py analyze                              # CPU: analysis.json

The guidance sweep of fig:mean-worse runs through ens_edm2.py (S45 there is EDM2-S at EMA 0.045, the S here),
one arm per weight of SWEEP_W:

    python reproduce/experiments/ens_edm2.py sample --out $VMG_RUNS/errors-in64/sweep --arm w1.7 \
        --d1 S45 --d0 XS --w 1.7                                                     # GPU
    python reproduce/experiments/ens_edm2.py eval --out $VMG_RUNS/errors-in64/sweep --arm w1.7    # GPU

EDM2-S (EMA 0.045) guided by EDM2-XS trained on 2^27 images (EMA 0.110) is NVIDIA's ImageNet-64 FID preset;
EDM2-XL (EMA 0.040) stands in for the posterior mean D*. `gram` takes the 10,000 calibration rows of
vmg/data/imagenet.py (10 val images per class, seed 0, EDM2's center-crop-dhariwal at 64) and noises each once
per level of EDM2's 32-step sampler, x = x0 + sigma eps, with eps from a CPU generator seeded 10000 + level and
shared by all networks. With r_k = x0 - D_k(x) the residual of network k,
    gram[n, k, l] = <r_k, r_l> / d          (image n, d = 12,288)
for the six networks of NETS, run at the release's own precision with TF32 off. Inner products of residuals
give every quantity below, errors against any reference included: D_i - D_R = r_R - r_i.

`analyze`, for a guided pair (strong D1, weak D0) and a reference R taken as D* (E_i = D_i - D_R):
    w_mse  = (G00 - G01) / (G00 + G11 - 2 G01)    MSE-optimal weight against x0 itself: needs no D*
    kappa  = E<E0, E1> / E|E1|^2                   the part of D1's error that D0 repeats (E0 = kappa E1 + E0perp)
    q      = E|E0perp|^2 / E|E1|^2                 the rest of D0's error, relative to D1's
    w_ref  = (kappa (kappa - 1) + q) / ((kappa - 1)^2 + q)   the weight closest to D_R; > 1 iff kappa > 1
and E|D_w - D_R|^2 / E|E1|^2 = (1 - w)^2 (q + kappa^2) + 2 w (1 - w) kappa + w^2 (distance_ratio). Bands are
90% bootstrap intervals over images (300 resamples). The per-input spread is that of alpha_0 / alpha_1 =
<E0, E1> / |E1|^2 over single images. R is XL for every pair, and for the pairs with S as the strong model also
the average of M, L and XL, which shrinks the part of the reference's own error that is not shared.

The paper's pass used the same 10 images per class as here (the same draws over each class's images in the
official file order), but visited the rows in another order, so each image met a different noise draw.

Outputs in $VMG_RUNS/errors-in64/: gram/sigma_<level>.npz, analysis.json, sweep/{samples,metrics}/ (ens_edm2.py).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402

# (name here, ens_edm2.MODELS member); ens_edm2's "S" is EMA 0.075 and its "S45" EMA 0.045
NETS = (("XS", "XS"), ("S", "S45"), ("S-ema0.075", "S"), ("M", "M"), ("L", "L"), ("XL", "XL"))
PAIRS = (("S", "XS", "the autoguidance pair"), ("S", "S-ema0.075", "same run, other EMA"),
         ("L", "S", "independent runs"), ("L", "M", "independent runs"))
REF, ENSEMBLE = "XL", "M+L+XL"
MAIN = "S<-XS|XL"
SEED0 = 10000                      # noise generator of level k: SEED0 + k
BOOT, BOOT_SEED = 300, 0
SWEEP_W = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.7, 2.0, 2.25, 2.5, 3.0)
W_GRID = np.linspace(-0.5, 3.0, 141)     # fig:mean-worse (a): w* is the grid point closest to D*


def folder(out=None):
    return Path(out) if out else runs.root() / "errors-in64"


def refs(strong):
    return (REF, ENSEMBLE) if strong == "S" else (REF,)


# ------------------------------------------------------------------ GPU pass
def fetch(a):
    from reproduce.experiments import ens_edm2 as E
    from vmg.models import _edm
    for _, m in NETS:
        _edm.fetch(E.MODELS[m])
    _edm.fetch("edm2/dataset-refs/img64.pkl")       # FID / FD_DINOv2 reference of the sweep (ens_edm2.py eval)


def gram(a):
    import torch
    from reproduce.experiments import ens_edm2 as E
    from vmg.data import imagenet
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = folder(a.out) / "gram"
    out.mkdir(parents=True, exist_ok=True)
    rows = imagenet.calib_rows(a.n)
    x0all = imagenet.pixels(rows)
    lab = torch.nn.functional.one_hot(imagenet.class_of(rows, 64), 1000).float()
    sig = E.t_steps()[:-1].double().numpy()
    names = [n for n, _ in NETS]
    N, K, d = len(rows), len(NETS), x0all[0].numel()
    for k in range(a.shard, len(sig), a.shards):
        path = out / f"sigma_{k:02d}.npz"
        if path.exists():
            continue
        s, t0 = float(sig[k]), time.time()
        g = torch.Generator().manual_seed(SEED0 + k)
        G = np.zeros((N, K, K), np.float32)
        for b0 in range(0, N, a.batch):
            x0 = x0all[b0:b0 + a.batch].cuda()
            x = x0 + s * torch.randn(x0.shape, generator=g).cuda()
            t = torch.full((len(x),), s, device="cuda")
            lb = lab[b0:b0 + a.batch].cuda()
            with torch.no_grad():
                R = torch.stack([(x0.double() - E.call(m, x, t, lb).double()).flatten(1) for _, m in NETS], 1)
            G[b0:b0 + len(x)] = (torch.einsum("bkd,bld->bkl", R, R) / d).float().cpu().numpy()
        tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, sigma=s, gram=G, rows=np.array(rows), names=np.array(names))
        os.replace(tmp, path)
        print(f"sigma[{k}] = {s:.4g} ({time.time() - t0:.0f}s); MSE " +
              " ".join(f"{n} {m:.4e}" for n, m in zip(names, G.mean(0).diagonal())), flush=True)


# ------------------------------------------------------------------ analysis
def load(d):
    """sigma [S] (high noise first), gram [S, N, K, K] float64, names."""
    zs = [np.load(f) for f in sorted((d / "gram").glob("sigma_*.npz"))]
    assert zs, f"no sigma_*.npz in {d / 'gram'}"
    order = np.argsort([-float(z["sigma"]) for z in zs])
    return (np.array([float(zs[k]["sigma"]) for k in order]),
            np.stack([zs[k]["gram"].astype(np.float64) for k in order]), [str(n) for n in zs[0]["names"]])


def mix(names, spec):
    """Weights of the networks in a reference: one network, or 'A+B+C' for their average."""
    c = np.zeros(len(names))
    for m in spec.split("+"):
        c[names.index(m)] += 1 / len(spec.split("+"))
    return c


def errors(names, strong, weak, ref):
    """Coefficient vectors over the residuals: D1 - D0, E1 = D1 - D_R, E0 = D0 - D_R, and e0 (r0 itself)."""
    e = np.eye(len(names))
    e1, e0, c = e[names.index(strong)], e[names.index(weak)], mix(names, ref)
    return e0 - e1, c - e1, c - e0, e0


def pair_stats(G, names, strong, weak, ref):
    """The pair's statistics from mean Gram matrices G [..., K, K], over the leading axes."""
    ip = lambda a, b: np.einsum("k,...kl,l->...", a, G, b)  # noqa: E731
    dd, E1, E0, e0 = errors(names, strong, weak, ref)
    c11, c00, c10 = ip(E1, E1), ip(E0, E0), ip(E1, E0)
    return {"w_mse": ip(e0, dd) / ip(dd, dd), "w_ref": ip(-E0, dd) / ip(dd, dd), "kappa": c10 / c11,
            "q": c00 / c11 - (c10 / c11) ** 2, "ratio": np.sqrt(c11 / c00), "rho": c10 / np.sqrt(c11 * c00)}


def distance_ratio(kappa, q, w):
    """E|D_w - D_R|^2 / E|D_1 - D_R|^2 per level (rows) and weight (columns)."""
    k, q, w = np.asarray(kappa)[:, None], np.asarray(q)[:, None], np.asarray(w)[None]
    return (1 - w) ** 2 * (q + k ** 2) + 2 * w * (1 - w) * k + w ** 2


def resample_means(gram, B=BOOT, seed=BOOT_SEED):
    """Mean Gram matrices of B bootstrap resamples of the images: [B, S, K, K]."""
    N = gram.shape[1]
    rng = np.random.default_rng(seed)
    return np.stack([gram[:, rng.integers(0, N, N)].mean(1) for _ in range(B)])


def boot(mean, means, fn):
    """fn of the mean Gram and its 90% bootstrap interval, per key."""
    point, draws = fn(mean), fn(means)
    return {k: {"value": point[k].tolist(), "lo": np.percentile(draws[k], 5, axis=0).tolist(),
                "hi": np.percentile(draws[k], 95, axis=0).tolist()} for k in point}


def analyze(a):
    d = folder(a.dir)
    sig, gram, names = load(d)
    print(f"{len(sig)} noise levels, {gram.shape[1]} images, networks {names}")
    mean, means = gram.mean(1), resample_means(gram)
    r = names.index(REF)
    mse_rel = lambda G: {n: G[..., i, i] / G[..., r, r] - 1 for i, n in enumerate(names) if i != r}  # noqa: E731
    out = {"sigma": sig.tolist(), "names": names, "n": int(gram.shape[1]),
           "mse": mean.diagonal(axis1=1, axis2=2).tolist(), "mse_rel_" + REF: boot(mean, means, mse_rel), "pairs": {}}
    for strong, weak, what in PAIRS:
        for ref in refs(strong):
            P = boot(mean, means, lambda G: pair_stats(G, names, strong, weak, ref))
            _, E1, E0, _ = errors(names, strong, weak, ref)
            alpha = np.einsum("k,snkl,l->sn", E1, gram, E0) / np.einsum("k,snkl,l->sn", E1, gram, E1)
            P["alpha_ratio"] = dict(zip(("p5", "p25", "p50", "p75", "p95"),
                                        np.percentile(alpha, [5, 25, 50, 75, 95], axis=1).tolist()))
            out["pairs"][f"{strong}<-{weak}|{ref}"] = {"what": what, **P}
    path = d / "analysis.json"
    path.write_text(json.dumps(out))
    report(out)
    print(f"wrote {path}")


def report(res):
    sig = np.array(res["sigma"])
    for key, P in res["pairs"].items():
        rng = lambda k: f"{min(P[k]['value']):.2f}-{max(P[k]['value']):.2f}"  # noqa: E731
        print(f"  {key:22} kappa {rng('kappa')}  w_ref {rng('w_ref')}  w_mse {rng('w_mse')}  ({P['what']})")
    m = res["mse_rel_" + REF]["S"]
    print(f"  MSE of S relative to {REF} (90% interval):")
    for k, s in enumerate(sig):
        lo, hi = m["lo"][k], m["hi"][k]
        verdict = "S<XL" if hi < 0 else "S>XL" if lo > 0 else "tie"
        print(f"    {k:2d} sigma {s:7.3f}  {m['value'][k]:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    if MAIN in res["pairs"]:
        P = res["pairs"][MAIN]
        k = int(np.argmin(np.abs(sig - 1.2)))
        R = distance_ratio(P["kappa"]["value"], P["q"]["value"], W_GRID)
        med = np.median(R, 0)
        at = lambda w: R[:, int(np.argmin(np.abs(W_GRID - w)))]  # noqa: E731
        print(f"  {MAIN}: kappa {P['kappa']['value'][k]:.2f} at sigma {sig[k]:.3g}; w* (median distance) "
              f"{W_GRID[np.argmin(med)]:.3g}; distance relative to S at w = 0: {np.median(at(0)):.2f}, "
              f"w = 1.7: {at(1.7).min():.2f}-{at(1.7).max():.2f} over levels")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["fetch", "gram", "analyze"])
    ap.add_argument("--n", type=int, default=10000, help="gram: calibration rows (10 per class at 10,000)")
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--out", help="gram: output folder instead of $VMG_RUNS/errors-in64")
    ap.add_argument("--dir", help="analyze: the gram's output folder (default $VMG_RUNS/errors-in64)")
    a = ap.parse_args()
    {"fetch": fetch, "gram": gram, "analyze": analyze}[a.stage](a)


if __name__ == "__main__":
    main()
