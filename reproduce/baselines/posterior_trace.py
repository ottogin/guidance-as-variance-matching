#!/usr/bin/env python
"""The posterior trace tr Cov[x0 | x_s] of both branches, from the network alone (no data in the estimate).

    python reproduce/baselines/posterior_trace.py --model sd35m --dataset coco --n 128      # one GPU, ~1 h
    python reproduce/baselines/posterior_trace.py --model sd35m --check-h                   # the step-size check

For x_s = (1 - s) x0 + s eps the second-order Tweedie identity reads Cov[x0 | x_s] = s^2 / (1 - s) dD/dx_s, with
D = E[x0 | x_s] (Xing et al. 2026, arXiv 2605.00941 write it for the velocity: (s^2 / (1 - s)) (I - s J_v)). Its
trace is s^2 / (1 - s) div D, estimated here with Rademacher probes z and central differences,
div D ~ mean_z <z, D(x + h z) - D(x - h z)> / (2 h), with the transformer in float32 (bf16 differences would
drown in rounding). The points x_s are the moments pass's own: calibration photos noised with noise.eps, so the
per-photo errors ||D_i - x0||^2 written beside the traces are the stored pass's (reproduce/baselines/analysis.py
compares them).

If D were the exact posterior mean, E tr Cov[x0 | x_s] = E||D - x0||^2 at every level, and for the unconditional
branch both equal the matching equation's gap c(t) (vmg/solve.py). Where they part, the network is not the
posterior mean there. Writes runs/<model>/<dataset>/posterior-trace.npz; all values are per coordinate (/ d).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import cli, noise, runs  # noqa: E402

PROBE_STREAM = 11


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int, default=128, help="calibration rows")
    ap.add_argument("--every", type=int, default=4, help="every k-th sampler level (levels k - 1, 2k - 1, ...)")
    ap.add_argument("--probes", type=int, default=4)
    ap.add_argument("--h", type=float, default=1e-2, help="finite-difference step")
    ap.add_argument("--batch", type=int, default=2, help="photos per batch (times 1 + 2 probes per branch)")
    ap.add_argument("--check-h", action="store_true", help="4 photos, 3 levels, h in {3e-3, 1e-2, 3e-2}; no output")
    ap.add_argument("--tag")
    a = ap.parse_args()
    import torch
    from vmg.models import get
    model = get(a.model)
    dataset = a.dataset or model.spec.default_dataset
    model.load()
    net = model.pipe.transformer.to(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    L = len(model.s)
    levels = [k for k in range(L) if k % a.every == a.every - 1 and model.s[k] < 1]
    rows = model.rows(dataset, "calib")[: (4 if a.check_h else a.n)]
    if a.tag == "smoke":
        rows, levels, a.probes = rows[:2], levels[-2:], 2
    hs = (3e-3, 1e-2, 3e-2) if a.check_h else (a.h,)
    if a.check_h:
        levels = [levels[0], levels[len(levels) // 2], levels[-1]]
    ne, npl = (v.float() for v in model.null)

    def D(x, k, emb, pool):
        """D = x - s v in float32, for a batch that may hold several points per photo."""
        t = torch.tensor(model.ts[k], device=x.device, dtype=torch.float32).expand(len(x))
        out = []
        for i in range(0, len(x), 8):
            v = net(hidden_states=x[i:i + 8], timestep=t[i:i + 8], encoder_hidden_states=emb[i:i + 8],
                    pooled_projections=pool[i:i + 8], return_dict=False)[0]
            out.append(x[i:i + 8] - float(model.s[k]) * v.float())
        return torch.cat(out)

    P, K = a.probes, len(levels)
    tr = np.zeros((len(hs), len(rows), K, 2, P))         # [h, row, level, (uncond, cond), probe]
    err = np.zeros((len(rows), K, 2))                    # ||D_i - x0||^2 / d
    t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, len(rows), a.batch):
            rb = rows[b0:b0 + a.batch]
            x0, (emb, pool) = model.data(dataset, rb)
            x0, B = x0.float(), len(rb)
            d = x0[0].numel()
            shape = tuple(x0.shape[1:])
            for j, k in enumerate(levels):
                s = float(model.s[k])
                x = model.noise(x0, k, noise.eps_batch(model.seed, rb, k, shape, device=x0.device))
                z = torch.stack([torch.randint(0, 2, (P, *shape), generator=noise.generator(model.seed, r, PROBE_STREAM))
                                 for r in rb]).to(x.device, torch.float32) * 2 - 1      # [B, P, ...] fixed per photo
                for bi, (e, p) in enumerate(((ne.expand(B, *ne.shape[1:]), npl.expand(B, *npl.shape[1:])),
                                             (emb.float(), pool.float()))):
                    d0 = D(x, k, e, p)
                    err[b0:b0 + B, j, bi] = ((d0 - x0) ** 2).flatten(1).sum(1).cpu().numpy() / d
                    rep = lambda u: u.repeat_interleave(2 * P, 0)                    # noqa: E731
                    for hi, h in enumerate(hs):
                        zz = z.reshape(B * P, *shape)
                        xs = torch.cat([x[:, None] + h * z, x[:, None] - h * z], 1).reshape(B * 2 * P, *shape)
                        dd = D(xs, k, rep(e), rep(p)).reshape(B, 2, P, *shape)
                        div = ((dd[:, 0] - dd[:, 1]) * zz.reshape(B, P, *shape)).flatten(2).sum(2) / (2 * h)
                        tr[hi, b0:b0 + B, j, bi] = (s * s / (1 - s) * div / d).cpu().numpy()
            el = time.time() - t0
            print(f"  {b0 + B}/{len(rows)} rows  {el / 60:.1f} min  eta {(len(rows) - b0 - B) * el / (b0 + B) / 60:.1f} min",
                  flush=True)
    if a.check_h:
        m = tr.mean(-1)                                                    # [h, row, level, branch]
        for j, k in enumerate(levels):
            for bi, br in enumerate(("uncond", "cond")):
                v = m[:, :, j, bi].mean(1)
                print(f"  level {k:2d} s {model.s[k]:.3f} {br:6s}: tr per coord at h = {hs}: {np.round(v, 5)}; "
                      f"spread {np.ptp(v) / abs(v[1]):.1%}; data ||D - x0||^2 {err[:, j, bi].mean():.5f}")
        return
    path = runs.run_dir(a.model, dataset, a.tag) / "posterior-trace.npz"
    np.savez(path, rows=np.asarray(rows), levels=np.asarray(levels), s=model.s[levels],
             sigma=model.levels()["sigma"][levels], tr=tr[0], err=err, h=a.h, probes=P, seed=model.seed,
             branches=np.array(["uncond", "cond"]), git=cli.git_sha())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
