#!/usr/bin/env python
"""Side study on the fig:story pools: guidance scales of trained 2-D pairs, centered against uncentered.

    python reproduce/toys/story2d_scales.py --src $VMG_RUNS/_toys/story2d            # pair 2,048 / 256
    python reproduce/toys/story2d_scales.py --src DIR --strong 512 --weak 64

Reads reproduce/toys/story2d.py's <optim>-n<N>.npz. Model r of the strong size is paired with model r of the weak
size. At each noise level, for Adam and SGD:
    centered  : the rule over the 256 inputs, tr Cov_x(D_w) = tr Cov(x0), per pair, median over the first 64 pairs
    2nd moment: E_x |D_w - xbar0|^2 = tr Cov(x0) with xbar0 the mixture's mean, the same pairs and median
    matches S : the coefficients of the rule with the spreads across the 1,024 retrainings at each input, averaged
                over the inputs, target the input average of tr Sigma(x_t), solved once (larger root throughout)
    shared    : the part of the weak model's error e0 = D0 - E[x0 | x_t] common to all inputs,
                |E_x e0|^2 / E_x |e0|^2, median over the same 64 weak models
Then, at fig:concept's input (41 at sigma = 0.4), the spread across retrainings of the Adam networks over that of
the exact posterior means of their training sets, per N.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.toys.story2d import MEANS, NS, SIGMAS, VAR_X0, posterior_mean_cov  # noqa: E402

PAIRS = 64


def root(a, b, c):
    """Larger root of b w^2 - 2 a w - c = 0, the form every rule below reduces to."""
    return (a + math.sqrt(max(a * a + b * c, 0))) / b


def centered(D0, D1):
    C0, C1 = np.cov(D0.T), np.cov(D1.T, D0.T)[:2, 2:]
    return root(np.trace(C0) - np.trace(C1), np.trace(np.cov((D1 - D0).T)), VAR_X0 - np.trace(C0))


def second_moment(D0, D1):
    e0, dd = D0 - MEANS.mean(0), D1 - D0
    return root(-(e0 * dd).sum(-1).mean(), (dd ** 2).sum(-1).mean(), VAR_X0 - (e0 ** 2).sum(-1).mean())


def matched(P0, P1, trS):
    """P0, P1 [R, B, 2]: spreads across retrainings at each input, coefficients averaged over inputs."""
    c0, c1 = P0 - P0.mean(0), P1 - P1.mean(0)
    v0 = (c0 ** 2).sum(-1).mean(0)
    return root((v0 - (c1 * c0).sum(-1).mean(0)).mean(), ((c1 - c0) ** 2).sum(-1).mean(0).mean(), (trS - v0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", help="reproduce/toys/story2d.py's folder (default $VMG_RUNS/_toys/story2d)")
    ap.add_argument("--strong", type=int, default=2048)
    ap.add_argument("--weak", type=int, default=256)
    a = ap.parse_args()
    src = Path(a.src or Path(os.environ.get("VMG_RUNS", REPO / "runs")) / "_toys" / "story2d")
    print(f"pair N1 = {a.strong} / N0 = {a.weak}; per pair: median over {PAIRS} pairs")
    print(f"{'optim':5s} {'sigma':>5s} {'centered':>9s} {'2nd mom.':>9s} {'matches S':>9s} {'shared':>7s}")
    for optim in ("adam", "sgd"):
        z1, z0 = np.load(src / f"{optim}-n{a.strong}.npz"), np.load(src / f"{optim}-n{a.weak}.npz")
        for s in SIGMAS:
            xt = z1[f"xt_{s}"]
            mu, Sig = posterior_mean_cov(xt, s)
            P1, P0 = z1[f"pred_{s}"].astype(np.float64), z0[f"pred_{s}"].astype(np.float64)
            wc = np.median([centered(P0[r], P1[r]) for r in range(PAIRS)])
            wr = np.median([second_moment(P0[r], P1[r]) for r in range(PAIRS)])
            e0 = P0[:PAIRS] - mu
            shared = np.median((e0.mean(1) ** 2).sum(-1) / (e0 ** 2).sum(-1).mean(1))
            wm = matched(P0, P1, np.trace(Sig, axis1=1, axis2=2))
            print(f"{optim:5s} {s:5.1f} {wc:9.2f} {wr:9.2f} {wm:9.2f} {shared:7.2f}")
    sizes = []
    for n in NS:
        z = np.load(src / f"adam-n{n}.npz")
        net, exact = (z[f"{k}_0.4"][:, 41].astype(np.float64) for k in ("pred", "ideal"))
        sizes.append(f"{n}: {np.trace(np.cov(net.T)) / np.trace(np.cov(exact.T)):.3f}")
    print("input 41, sigma 0.4, Adam: spread of the networks / spread of the exact posterior means:", ", ".join(sizes))


if __name__ == "__main__":
    main()
