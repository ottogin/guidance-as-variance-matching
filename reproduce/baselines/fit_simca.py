#!/usr/bin/env python
"""SIMCA's per-level least-squares guidance weights (Moroy et al. 2026, arXiv 2610.03503), ported to CFG.

    python reproduce/baselines/fit_simca.py --model sd35m --dataset coco --form sf            # CPU, from wmse.json
    python reproduce/baselines/fit_simca.py --model sd35m --dataset coco --form 1 --n 256     # one GPU, minutes
    python reproduce/baselines/fit_simca.py --model sd35m --dataset coco --form 2 --n 256

SIMCA calibrates the weights that mix a sampler's terms by least squares against the network's own regression
target, with ground-truth data, one level at a time. It was written for posterior sampling (a prior and a
measurement term); for CFG the two terms are the unconditional and conditional predictions. For flow matching
the target is psi = eps - x0, and with x_s = (1 - s) x0 + s eps that is psi = (x_s - x0) / s.

  sf  simulation-free, CFG form: at every level the w minimising E||v_u + w (v_c - v_u) - psi||^2 over
      forward-noised calibration photos. That is E||D_w - x0||^2 / s^2, so the weight is w_MSE (paper eq. wmse),
      read from compute_wmse.py's result.
  1   simulation-based (SIMCA Alg. 1), CFG form: N trajectories start from noise, each with the caption of one
      calibration photo; at each level the weight is fitted to psi_i = (x_i - x0_i) / s on the trajectories' own
      states, and every trajectory then takes the sampler's Euler step with it.
  2   the same with SIMCA's own two free weights, alpha v_c + beta v_u.

Writes runs/<model>/<dataset>/simca-{sf,sb1,sb2}.json, which sample.py --mix reads. The simulation keeps its
state in float32 (sample.py --fp32-state) and runs the network as the moments pass does (model.denoise_pair).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import baselines as B, cli, noise, runs  # noqa: E402

STREAM = 10         # noise.generator stream of the trajectories' initial noise (sampling uses 1, controls 8 / 9)


def from_wmse(a):
    j = json.load(open(runs.wmse_json(a.model, a.dataset, a.tag)))
    w = np.asarray(j["w_mse"], dtype=np.float64)
    return dict(method="simca-sf", form="sf", mode="simulation-free (= w_MSE, compute_wmse.py)", n=j["n"],
                levels=j["levels"], sigma=j["sigma"], t=j["t"], w=w,
                ci_lo=j.get("ci_lo"), ci_hi=j.get("ci_hi"), source=str(runs.wmse_json(a.model, a.dataset, a.tag)))


def simulate(a, model, dataset):
    import torch
    model.load()
    rows = model.rows(dataset, "calib")[a.start: a.start + a.n]
    L = len(model.s)
    s = np.append(model.s, float(model.pipe.scheduler.sigmas[-1]))      # the grid and its terminal sigma
    c, h = model.pipe.transformer.config.in_channels, model.size // model.pipe.vae_scale_factor
    X, X0, C = [], [], []
    for b0 in range(0, len(rows), a.batch):
        rb = rows[b0:b0 + a.batch]
        x0, cond = model.data(dataset, rb)
        X0.append(x0.to("cuda", torch.float32))
        C.append(cond)
        X.append(torch.stack([torch.randn((c, h, h), generator=noise.generator(model.seed, r, STREAM))
                              for r in rb]).to("cuda", torch.float32))
    print(f"{model.spec.name}: {len(rows)} trajectories (rows {rows[0]}..{rows[-1]}), {L} levels, form {a.form}",
          flush=True)
    rng = np.random.default_rng(0)
    sol, se, resid = [], [], []

    def ip(u, v):
        return (u.double() * v.double()).flatten(1).sum(1)

    t0 = time.time()
    for k in range(L):
        sk, sn = float(s[k]), float(s[k + 1])
        P, V = [], []
        for x, x0, cond in zip(X, X0, C):
            d1, d0 = model.denoise_pair(x, k, cond)
            vc, vu = (x - d1) / sk, (x - d0) / sk
            psi = (x - x0) / sk
            P.append(torch.stack([ip(vu, vu), ip(vu, vc), ip(vc, vc), ip(vu, psi), ip(vc, psi), ip(psi, psi)], 1).cpu())
            V.append((vu, vc))
        P = torch.cat(P).numpy()
        wk = B.simca_solve(P[:, :5], a.form)
        boot = np.array([B.simca_solve(P[rng.integers(0, len(P), len(P)), :5], a.form) for _ in range(a.boot)])
        # the relative least-squares error at the fitted weights: sum||v - psi||^2 / sum||psi||^2
        uu, uc, cc, up, cp, pp = P.sum(0)
        al, be = (wk[0], 1 - wk[0]) if a.form == 1 else (wk[0], wk[1])
        err = al * al * cc + be * be * uu + 2 * al * be * uc - 2 * al * cp - 2 * be * up + pp
        sol.append(wk)
        se.append(boot.std(0))
        resid.append(err / pp)
        for i, (vu, vc) in enumerate(V):
            v = vu + float(wk[0]) * (vc - vu) if a.form == 1 else float(wk[0]) * vc + float(wk[1]) * vu
            X[i] = X[i] + (sn - sk) * v
        print(f"  level {k:2d}  s {sk:.4f}  " + ("w {:8.4f}".format(wk[0]) if a.form == 1 else
                                                 "alpha {:8.4f} beta {:8.4f}".format(*wk)) +
              f"  +- {boot.std(0).max():.4f}  rel. error {err / pp:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    sol, se = np.array(sol), np.array(se)
    lv = model.levels()
    out = dict(method=f"simca-sb{a.form}", form=a.form, mode="simulation-based (SIMCA Alg. 1, greedy over levels)",
               n=len(rows), rows=[int(rows[0]), int(rows[-1])], noise_stream=STREAM, seed=model.seed, boot=a.boot,
               levels=list(range(L)), sigma=lv["sigma"], t=lv["t"], rel_error=np.array(resid))
    if a.form == 1:
        out.update(w=sol[:, 0], se=se[:, 0])
    else:
        out.update(alpha=sol[:, 0], beta=sol[:, 1], se_alpha=se[:, 0], se_beta=se[:, 1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--form", required=True, choices=("sf", "1", "2"))
    ap.add_argument("--n", type=int, default=256, help="trajectories (calibration rows), simulation-based forms")
    ap.add_argument("--start", type=int, default=0, help="first calibration row")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--boot", type=int, default=200, help="photo-bootstrap replicates of each level's solve")
    ap.add_argument("--tag")
    a = ap.parse_args()
    from vmg.models import get
    model = get(a.model)
    a.dataset = a.dataset or model.spec.default_dataset
    if a.tag == "smoke":
        a.n, a.boot = min(a.n, 4), min(a.boot, 10)
    if a.form == "sf":
        out = from_wmse(a)
    else:
        a.form = int(a.form)
        out = simulate(a, model, a.dataset)
    out.update(model=a.model, dataset=a.dataset, tag=a.tag, created=time.strftime("%Y-%m-%d %H:%M:%S"),
               git=cli.git_sha())
    path = runs.run_dir(a.model, a.dataset, a.tag) / f"{out['method']}.json"
    cli.dump(path, out)
    w = np.asarray(out["w"]) if "w" in out else np.asarray(out["alpha"]) / (np.asarray(out["alpha"]) + np.asarray(out["beta"]))
    print(f"{out['method']}: w per level {np.round(w[:3], 4)} ... {np.round(w[-3:], 4)}, mean {w.mean():.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
