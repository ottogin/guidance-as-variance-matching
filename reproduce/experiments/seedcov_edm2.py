#!/usr/bin/env python
"""fig:cov-trajectory for an EDM2 pair on ImageNet-64: Cov(D_c) against Cov(D_u) along unguided trajectories of one class.

    python reproduce/experiments/seedcov_edm2.py work  --model edm2-64-cfg-s-xsu-fid --out DIR [--cls 779 --n 1000]  # GPU
    python reproduce/experiments/seedcov_edm2.py align --out DIR                                                   # CPU
    python reproduce/figures/fig_cov_trajectory.py --sd35m D --cfg DIR --ag DIR2 --out F                           # figure

The SD 3.5 Medium figure (reproduce/experiments/seedcov.py, one prompt, 1,000 seeds) for the EDM2 pairs of
vmg/models/edm2.py. One class for every seed (default 779, "school bus": the SD 3.5 run's prompt is a bus), seeds
0..n-1 for the initial noise (generate_images.py's per-seed noise; the class it would draw is replaced), sampled
with EDM's deterministic Heun sampler at w = 1, so every step uses D_c = the pair's net alone. At the state x_k each
step starts from, both networks are evaluated:

    D_c = net(x_k, sigma_k, class),   D_u = gnet(x_k, sigma_k, class or none),   v_c - v_u = (D_u - D_c) / s_k

with v = eps - x_0 the paper's (flow-matching) velocity and s_k = sigma_k / (1 + sigma_k)

(for the CFG pair gnet is unconditional; for an autoguidance pair it is the weak conditional net). The last Heun
step is an Euler step to sigma = 0, which lands on D_c of that step, so x_0 is D_c at the last step, as in SD 3.5.

`work` writes, in seedcov.py's layout, everything its `align` stage reads, from the fields held in memory:
    meta.json        model, class, n, batch (the seed-half unit of seedcov's halves: even / odd blocks of 25), and
                     t = sigma per step and s = sigma / (1 + sigma), its flow-matching time (t = s / (1 - s))
    grams.npz       full-dimensional Grams [S, N, N] of the per-coordinate-centred D_c and D_u (cc, uu, cu)
    grams_x0.npz     the same against the final samples x_0 (c0, u0), of D_c - D_u (dd, d0), and x_0's own (x00)
    vgap.npz         fig:vgap's quantity on these seeds: E|v_c - v_u|^2 / d per step (total = mean part + spread), in
                     the paper's velocity v = eps - x_0: v_c - v_u = (D_u - D_c) / s
    x_final.npy      the final samples [N, 3, 64, 64] (pixels / 127.5 - 1); thumbs.png, the first 64 of them
`align` runs seedcov.align_steps on those (D_c vs D_u, D_c vs x_0, D_u vs x_0, principal planes for K = 50, 100)
-> align.npz.
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from reproduce.experiments import seedcov as SC  # noqa: E402

HALF = 25            # seedcov's batch: its seed halves are the even and odd blocks of 25 seeds


def atomic_npz(path, **arrays):
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def trajectory(m, noise, labels, each=None):
    """The adapter's own Heun loop (EDMHeun.heun) at w = 1; each(i, x_i, D_c, D_u) sees the state every step starts
    from and both predictions there (D_u only computed when `each` is given). The samples do not depend on `each`."""
    import torch
    dt, ts, N = m.state_dtype, m.t_steps(noise.device), m.steps
    x_next = noise.to(dt) * ts[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(ts[:-1], ts[1:])):
            x_hat, t_hat = x_next, t_cur
            d1, d0 = m.pair(x_hat, t_hat, labels, released=True, ref=each is not None)
            d_hat = d1.to(dt)
            if each is not None:
                each(i, x_hat, d_hat, d0.to(dt))
            d_cur = (x_hat - d_hat) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < N - 1:
                d_prime = (x_next - m.pair(x_next, t_next, labels, released=True, ref=False)[0].to(dt)) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def work(a):
    import torch
    from vmg.models import base
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = base.get(a.model)
    m.load()
    S, N = m.steps, a.n
    sig = m.t_steps()[:-1].double().numpy()
    # the paper's velocity v = eps - x_0 (the flow-matching one), whatever the network predicts: with eps = (x - D) /
    # sigma, v_c - v_u = (D_u - D_c)(1 + sigma) / sigma = (D_u - D_c) / s at the flow-matching time s = sigma / (1 +
    # sigma), seedcov.py's formula; (x - D) / sigma alone, EDM's eps, would shrink the high-noise gap by (1 + sigma)^2
    s_fm = sig / (1 + sig)
    gpu = torch.cuda.get_device_name()
    print(f"{socket.gethostname()} {gpu}: {m.spec.name}, class {a.cls}, seeds 0..{N - 1}, {S} Heun steps at w = 1",
          flush=True)
    onehot = torch.nn.functional.one_hot(torch.tensor([a.cls]), m.label_dim).float().cuda()

    # the loop is the adapter's sampler: on the first seeds, bit for bit EDMHeun.heun at w = 1
    noise, _ = m.initial(list(range(8)))
    lab = onehot.expand(len(noise), -1)
    ref = m.heun(noise, lab, np.ones(S))
    got = trajectory(m, noise, lab, each=lambda *_: None)
    rel = float((ref - got).norm() / ref.norm())
    print(f"check: the loop against EDMHeun.heun at w = 1 (8 seeds): bitwise {torch.equal(ref, got)}, "
          f"relative L2 {rel:.1e}", flush=True)
    assert rel < 1e-5, "the loop does not reproduce the adapter's sampler"

    d = m.channels * m.res * m.res
    Dc = np.empty((S, N, d), np.float32)
    Du = np.empty_like(Dc)
    X0 = np.empty((N, d), np.float32)
    t0 = time.time()
    for b0 in range(0, N, a.batch):
        seeds = list(range(b0, min(b0 + a.batch, N)))
        noise, _ = m.initial(seeds)
        lab = onehot.expand(len(seeds), -1)

        def each(i, x, dc, du):
            Dc[i, b0:b0 + len(seeds)] = dc.reshape(len(seeds), -1).cpu().numpy()
            Du[i, b0:b0 + len(seeds)] = du.reshape(len(seeds), -1).cpu().numpy()

        X0[b0:b0 + len(seeds)] = trajectory(m, noise, lab, each).reshape(len(seeds), -1).cpu().numpy()
        print(f"  {seeds[-1] + 1}/{N} seeds  {(time.time() - t0) / 60:.1f} min", flush=True)
    sec = time.time() - t0
    dev = np.abs(Dc[-1] - X0).max() / np.abs(X0).max()
    print(f"last step: D_c against x_0, max relative difference {dev:.1e}", flush=True)
    assert dev < 1e-5, "x_0 is not D_c of the last step"

    # Grams of the per-coordinate-centred fields, step by step on the GPU in float64
    dev_ = "cuda"
    x0c = torch.from_numpy(X0).to(dev_, torch.float64)
    x0c -= x0c.mean(0)
    g = {k: np.zeros((S, N, N)) for k in ("cc", "uu", "cu", "c0", "u0", "dd", "d0")}
    tot, mean = np.zeros(S), np.zeros(S)
    for k in range(S):
        c = torch.from_numpy(Dc[k]).to(dev_, torch.float64)
        u = torch.from_numpy(Du[k]).to(dev_, torch.float64)
        diff = c - u
        tot[k] = float((diff ** 2).sum()) / (N * d) / s_fm[k] ** 2         # E|v_c - v_u|^2 / d, as in seedcov.py
        mean[k] = float((diff.mean(0) ** 2).sum()) / d / s_fm[k] ** 2
        c -= c.mean(0)
        u -= u.mean(0)
        dd = c - u
        for key, (p, q) in dict(cc=(c, c), uu=(u, u), cu=(c, u), c0=(c, x0c), u0=(u, x0c), dd=(dd, dd),
                                d0=(dd, x0c)).items():
            g[key][k] = (p @ q.T).cpu().numpy()
    x00 = (x0c @ x0c.T).cpu().numpy()
    atomic_npz(out / "grams.npz", cc=g["cc"], uu=g["uu"], cu=g["cu"])
    atomic_npz(out / "grams_x0.npz", x00=x00, c0=g["c0"], u0=g["u0"], dd=g["dd"], d0=g["d0"])
    atomic_npz(out / "vgap.npz", total=tot, mean=mean, spread=tot - mean, n=N, s=s_fm)
    np.save(out / "x_final.npy", X0.reshape(N, m.channels, m.res, m.res))
    from PIL import Image
    ims = m.decode(torch.from_numpy(X0[:64]).cuda().reshape(-1, m.channels, m.res, m.res)).permute(0, 2, 3, 1)
    ims = ims.cpu().numpy()
    Image.fromarray(ims.reshape(8, 8, m.res, m.res, 3).transpose(0, 2, 1, 3, 4).reshape(8 * m.res, 8 * m.res, 3)) \
        .save(out / "thumbs.png")
    meta = dict(model=a.model, name=m.spec.name, cls=a.cls, n=N, batch=HALF, seed=0, steps=S,
                s=s_fm.tolist(), t=sig.tolist(), sigma=sig.tolist(), d=d, device=gpu, seconds=sec,
                net=str(m.net_ckpt), guide=str(m.guide_ckpt))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}: Grams, vgap, x_final, thumbs ({sec / 60:.1f} min sampling)", flush=True)


def align(a):
    out = Path(a.out)
    res = SC.align_steps(out, SC.ALIGN_PAIRS, planes=SC.ELL_K)
    SC.atomic(out / "align.npz", lambda f: np.savez(f, **res))
    # the last step lands on D_c (see work), so there D_c against x_0 is a field against itself
    dev = [abs(res["Dc_paired_scov"][-1] - 1)] + [abs(res["Dc_paired_sub"][-1][:K] - 1).max() for K in (50, 500)]
    print(f"last step, D_c vs x_0: |S_cov - 1| {dev[0]:.1e}, max |S_k - 1| for k <= 50 {dev[1]:.1e}, "
          f"k <= 500 {dev[2]:.1e}")
    assert dev[0] < 1e-6 and dev[1] < 1e-4, "D_c at the last step is not x_0"
    for K in (10, 50, 100):
        print(f"S_{K}: D_c vs D_u " + " ".join(f"{v:.3f}" for v in res["DcDu_paired_sub"][:, K - 1]))
        print(f"S_{K}: D_c vs x_0 " + " ".join(f"{v:.3f}" for v in res["Dc_paired_sub"][:, K - 1]))
    print(f"wrote {out / 'align.npz'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("work", "align"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="work: the EDM2 pair (vmg/models/edm2.py)")
    ap.add_argument("--cls", type=int, default=779, help="work: the one class of every seed (779: school bus)")
    ap.add_argument("--n", type=int, default=1000, help="work: seeds 0..n-1")
    ap.add_argument("--batch", type=int, default=250, help="work: seeds per network call")
    a = ap.parse_args()
    {"work": work, "align": align}[a.stage](a)


if __name__ == "__main__":
    main()
