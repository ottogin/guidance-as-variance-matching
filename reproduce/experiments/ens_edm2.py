#!/usr/bin/env python
"""Guidance with an ensemble of EDM2 models on ImageNet-64 (a side study).

    python reproduce/experiments/ens_edm2.py moments --n 2000                      # GPU, ~15 min
    python reproduce/experiments/ens_edm2.py sample --arm NAME --d1 S,M,L,XL [--d0 XS --w 1.7]
                     [--eg S,M,L,XL --alpha 2 --zmode fresh] --shard i --shards n  # GPU
    python reproduce/experiments/ens_edm2.py eval --arm NAME                       # GPU, a few min

Members are NVIDIA's released, independently trained class-conditional ImageNet-64 models (one run per size),
plus two controls: S45, a different post-hoc EMA of S's own run (an 'ensemble' member that shares S's training
noise), and XSu, the unconditional XS (CFG).

The guided prediction of one network call is
    Dtilde = D0 + w (D1 - D0) + alpha sum_{k in eg} z_k (D^k - mean_eg D),
with D1, D0 the plain means of their member lists (D0 empty: Dtilde = D1 + ...), and z ~ N(0, I_K) drawn
    fresh   at every network call (Heun's two calls too)       -> M(2) with a rank-(K-1) covariance estimate
    traj    once per trajectory                                -> a per-sample fixed combination of the members
    global  once for all trajectories                          -> one fixed linear combination (= plain guidance)
Sampler: EDM's deterministic Heun, 32 steps (63 calls), seeds 0..9,999, class from the seed (generate_images.py).
Outputs: $VMG_RUNS/edm2-64-ens/imagenet64/{moments-<tag>-<shard>of<shards>.npz, samples/<arm>/<row>.png,
metrics/<arm>.json}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import runs  # noqa: E402
from vmg.data import imagenet  # noqa: E402
from vmg.models import _edm  # noqa: E402
from vmg.models._edm import posthoc, raw  # noqa: E402

MODELS = {"XS": posthoc("edm2-img64-xs-0134217-0.110"), "S": posthoc("edm2-img64-s-1073741-0.075"),
          "M": posthoc("edm2-img64-m-2147483-0.060"), "L": posthoc("edm2-img64-l-1073741-0.040"),
          "XL": posthoc("edm2-img64-xl-0671088-0.040"), "S45": posthoc("edm2-img64-s-1073741-0.045"),
          "XSu": raw("edm2-img64-xs-uncond-2147483-0.050")}
STEPS, SMIN, SMAX, RHO = 32, 0.002, 80.0, 7.0
NEVAL = 10000
OUT = None          # --out: another output folder (reproduce/experiments/seedpair_in64.py uses it)


def root():
    d = Path(OUT) if OUT else runs.root() / "edm2-64-ens" / "imagenet64"
    d.mkdir(parents=True, exist_ok=True)
    return d


def t_steps(device="cpu"):
    i = torch.arange(STEPS, dtype=torch.float32, device=device)
    a, b = SMAX ** (1 / RHO), SMIN ** (1 / RHO)
    t = (a + i / (STEPS - 1) * (b - a)) ** RHO
    return torch.cat([t, torch.zeros_like(t[:1])])


_NETS = {}


def net(name):
    if name not in _NETS:
        if not Path(MODELS[name]).is_absolute():         # a released pickle: fetched on first use
            _edm.fetch(MODELS[name])
        _NETS[name] = _edm.edm_pickle(MODELS[name])[0]
    return _NETS[name]


def call(name, x, t, labels, fp32=False):
    n = net(name)
    return n(x, t, labels if n.label_dim else None, force_fp32=fp32).float()


# ------------------------------------------------------------------ moments
def moments(a):
    names = a.models.split(",")
    rows = imagenet.calib_rows(a.n)
    x0all = imagenet.pixels(rows)
    lab = torch.nn.functional.one_hot(imagenet.class_of(rows, 64), 1000).float()
    sig = t_steps()[:-1].double().numpy()
    K = len(names)
    d = x0all[0].numel()
    gram = np.zeros((STEPS, K + 1, K + 1))       # raw inner products E<Y_i, Y_j> over rows, Y = [D_1..D_K, x0]
    mean = np.zeros((STEPS, K + 1, d))
    pair_sq = np.zeros((STEPS, K, K))            # E|D_i - D_j|^2 (also from gram, kept as a float64 check)
    t0 = time.time()
    for k in [k for k in range(STEPS) if k % a.shards == a.shard]:
        g = torch.Generator().manual_seed(1000 + k)
        for b0 in range(0, len(rows), a.batch):
            x0 = x0all[b0:b0 + a.batch].cuda()
            eps = torch.randn(x0.shape, generator=g).cuda()
            x = x0 + float(sig[k]) * eps
            t = torch.full((len(x),), float(sig[k]), device="cuda")
            with torch.no_grad():
                Y = [call(m, x, t, lab[b0:b0 + a.batch].cuda(), fp32=True) for m in names] + [x0]
            Yf = torch.stack([y.reshape(len(x), -1).double() for y in Y])          # [K+1, B, d]
            gram[k] += torch.einsum("ibd,jbd->ij", Yf, Yf).cpu().numpy()
            mean[k] += Yf.sum(1).cpu().numpy()
            D = Yf[:K]
            pair_sq[k] += ((D[:, None] - D[None]) ** 2).sum((2, 3)).cpu().numpy()
        print(f"level {k} sigma {sig[k]:.4g} done ({time.time() - t0:.0f}s)", flush=True)
    n = len(rows)
    np.savez(root() / f"moments-{a.tag or 'n%d' % a.n}-{a.shard:02d}of{a.shards:02d}.npz", names=np.array(names), sigma=sig, n=n,
             gram=gram / n, mean=mean / n, pair_sq=pair_sq / n, d=d)
    print("wrote moments", flush=True)


# ------------------------------------------------------------------ sampling
def initial(seeds):
    rnd = _edm.StackedRandomGenerator("cuda", seeds)
    noise = rnd.randn([len(seeds), 3, 64, 64], device="cuda")
    return noise, torch.eye(1000, device="cuda")[rnd.randint(1000, size=[len(seeds)], device="cuda")]


def parse_w(s, off=1.0):
    """A constant 'w' or a per-level list 'w0,w1,...' (32 entries) or 'interval:w:lo:hi' (w where lo<sigma<=hi)."""
    sig = t_steps()[:-1].numpy()
    if s.startswith("interval:"):
        _, w, lo, hi = s.split(":")
        return np.where((sig > float(lo)) & (sig <= float(hi)), float(w), off)
    v = np.array([float(x) for x in s.split(",")])
    return np.full(STEPS, v[0]) if len(v) == 1 else v


def make_pred(a, B, seeds):
    d1 = a.d1.split(",")
    d0 = a.d0.split(",") if a.d0 else []
    eg = a.eg.split(",") if a.eg else []
    ws, als = parse_w(a.w), parse_w(a.alpha, 0.0)
    need = sorted(set(d1) | set(d0) | set(eg))
    gz = torch.Generator(device="cuda").manual_seed(int(a.zseed) * 1000003 + int(seeds[0]))
    ztraj = torch.randn(len(eg), B, 1, 1, 1, generator=gz, device="cuda") if eg else None
    zglob = torch.randn(len(eg), 1, 1, 1, 1, generator=torch.Generator(device="cuda").manual_seed(int(a.zseed)),
                        device="cuda") if eg else None

    def pred(x, t, labels, i):
        tt = torch.full((len(x),), float(t), device="cuda")
        out = {m: call(m, x, tt, labels) for m in need}
        D1 = torch.stack([out[m] for m in d1]).mean(0)
        if d0:
            D0 = torch.stack([out[m] for m in d0]).mean(0)
            D = D0 + float(ws[i]) * (D1 - D0)
        else:
            D = D1
        if eg and float(als[i]) != 0:
            E = torch.stack([out[m] for m in eg])
            Eb = E.mean(0)
            z = {"fresh": lambda: torch.randn(len(eg), len(x), 1, 1, 1, generator=gz, device="cuda"),
                 "traj": lambda: ztraj, "global": lambda: zglob}[a.zmode]()
            D = D + float(als[i]) * (z * (E - Eb)).sum(0)
        return D

    return pred


@torch.no_grad()
def heun(pred, noise, labels):
    ts = t_steps(noise.device)
    x_next = noise * ts[0]
    for i, (t_cur, t_next) in enumerate(zip(ts[:-1], ts[1:])):
        x_hat, t_hat = x_next, t_cur
        d_cur = (x_hat - pred(x_hat, t_hat, labels, i)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < STEPS - 1:
            d_prime = (x_next - pred(x_next, t_next, labels, i + 1)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def sample(a):
    out = root() / "samples" / a.arm
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / "args.json", "w"))
    rows = [a.seed0 + r for r in range(NEVAL) if r % a.shards == a.shard]
    t0 = time.time()
    for b0 in range(0, len(rows), a.batch):
        rb = [r for r in rows[b0:b0 + a.batch] if not (out / f"{r - a.seed0:05d}.png").exists()]
        if not rb:
            continue
        noise, labels = initial(rb)
        x = heun(make_pred(a, len(rb), rb), noise, labels)
        ims = (x * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for r, im in zip(rb, ims):
            runs.save_image(im, out / f"{r - a.seed0:05d}.png")
        print(f"{a.arm}: {b0 + len(rb)}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ evaluation
def evaluate(a):
    from PIL import Image
    from vmg.metrics import common as C, image as I
    I.IMAGENET["imagenet64"] = 64
    _edm.fetch("edm2/dataset-refs/img64.pkl")          # NVIDIA's training statistics, the Frechet reference
    ids = list(range(a.n))
    nref = min(a.n, 10000)          # the k-NN reference: held-out val images (10,000); FDs use NVIDIA's train stats
    refp = root() / f"ref-{nref}.npz"
    if refp.exists():
        real = dict(np.load(refp))
    else:
        real, _ = I._imagenet_reference("imagenet64", ids[:nref])
        tmp = refp.with_name(f"{refp.stem}.{os.getpid()}.npz")       # concurrent evals: write, then rename
        np.savez(tmp, **real)
        os.replace(tmp, refp)
    src = root() / "samples" / a.arm
    acc = I.Features(I._detectors())
    for b in range(0, len(ids), 100):
        u8 = torch.from_numpy(np.stack([np.asarray(Image.open(src / f"{i:05d}.png").convert("RGB"))
                                        for i in ids[b:b + 100]])).permute(0, 3, 1, 2).cuda()
        acc.update(u8)
    fake = acc.numpy()
    res = C.scores(fake, real, I.spaces(None, "imagenet64"))
    res = dict(arm=a.arm, n=a.n, args=json.load(open(src / "args.json")), **{k: float(v) for k, v in res.items()})
    (root() / "metrics").mkdir(exist_ok=True)
    json.dump(res, open(root() / "metrics" / f"{a.arm}.json", "w"), indent=1)
    np.savez(root() / "metrics" / f"{a.arm}.feats.npz", **fake)
    print(json.dumps({k: v for k, v in res.items() if k != "args"}))


def main():
    global STEPS, OUT, NEVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["moments", "sample", "eval"])
    ap.add_argument("--models", default="XS,S,M,L,XL,S45,XSu")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arm")
    ap.add_argument("--d1", default="S")
    ap.add_argument("--d0", default="")
    ap.add_argument("--w", default="1")
    ap.add_argument("--eg", default="")
    ap.add_argument("--alpha", default="0")
    ap.add_argument("--zmode", default="fresh")
    ap.add_argument("--zseed", default="0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH",
                    help="add a member from a pickle (an absolute path, or relative to VMG_WEIGHTS)")
    ap.add_argument("--out", help="output folder instead of $VMG_RUNS/edm2-64-ens/imagenet64")
    ap.add_argument("--neval", type=int, default=NEVAL, help="sample: seeds seed0 .. seed0 + neval - 1")
    a = ap.parse_args()
    STEPS, OUT, NEVAL = a.steps, a.out, a.neval
    for c in a.ckpt:
        k, v = c.split("=", 1)
        MODELS[k] = v
    if a.stage == "moments":
        a.n, a.batch = a.n or 2000, a.batch or 100
        moments(a)
    elif a.stage == "sample":
        a.batch = a.batch or 128
        sample(a)
    else:
        a.n = a.n or NEVAL
        evaluate(a)


if __name__ == "__main__":
    main()
