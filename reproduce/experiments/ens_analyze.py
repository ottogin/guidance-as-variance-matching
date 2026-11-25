#!/usr/bin/env python
"""CPU analysis of ens_edm2.py / ens_sd35.py moments: what an ensemble of the members buys.

    python reproduce/experiments/ens_analyze.py SHARD.npz [more shards ...] [--pairs S:XS;L:M] [--ens S,M,L,XL;...]

From the raw Grams E<Y_i, Y_j> (Y = members + x0) and the means, at every level:
  * MSE_i = E|D_i - x0|^2 per member and of ensemble means (better posterior mean?)
  * M_ij = E<D_i - x0, D_j - x0> = C^e_ij + J, with J = E tr Cov(x0|x_t) (the MMSE) and C^e the members' error
    Gram against the posterior mean: independent errors leave every off-diagonal at J
  * the VM weight (eq. wstar, positive root) of any pair (mean of list A guided by mean of list B), and the
    ensemble-noise scale alpha of Dbar + alpha sum z_k (D_k - Dbar)
Global weights average the coefficients over the levels (eq. wconstant, uniform over the sampler's levels).
"""
import argparse
import itertools

import numpy as np


def load(paths):
    zs = [np.load(p, allow_pickle=True) for p in paths]
    names = [str(x) for x in zs[0]["names"]]
    if "pair_sq" in zs[0]:                     # ens_edm2: normalised, one level-shard per file
        return names, zs[0]["sigma"], sum(z["gram"] for z in zs), sum(z["mean"] for z in zs)
    n = sum(int(z["n"]) for z in zs)
    gram = sum(z["gram"] for z in zs) / n
    mean = sum(z["mean"] for z in zs) / n
    return names, zs[0]["sigma"], gram, mean


def vec(names, spec):
    v = np.zeros(len(names) + 1)
    for m in spec.split(","):
        v[names.index(m)] += 1.0 / len(spec.split(","))
    return v


def stats(names, gram, mean):
    """Per level: raw Gram G, centred covariance C (both over [members, x0])."""
    C = gram - np.einsum("kid,kjd->kij", mean, mean)
    return gram, C


def pair_coef(C, u, v, x):
    a = np.einsum("ki,kij,kj->k", u - v, C, u - v)
    b = np.einsum("ki,kij,kj->k", v, C, v) - np.einsum("ki,kij,kj->k", v, C, u)
    c = np.einsum("ki,kij,kj->k", v, C, v) - np.einsum("ki,kij,kj->k", x, C, x)
    return a, b, c


def root(a, b, c):
    return (b + np.sqrt(np.clip(b * b - a * c, 0, None))) / a, (b - np.sqrt(np.clip(b * b - a * c, 0, None))) / a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--ens", default="")
    a = ap.parse_args()
    names, sig, gram, mean = load(a.paths)
    S, K1 = gram.shape[:2]
    K = K1 - 1
    G, C = stats(names, gram, mean)
    tile = lambda v: np.broadcast_to(v, (S, K1))
    x = tile(np.eye(K1)[K])
    M = G - G[:, :, K:K + 1] - G[:, K:K + 1, :] + G[:, K:K + 1, K:K + 1]          # E<D_i - x0, D_j - x0>
    varx = C[:, K, K]
    print("levels:", " ".join(f"{s:.3g}" for s in sig))
    print("\n== per-level MSE / Var(x0), members")
    print("sigma   " + " ".join(f"{n:>7s}" for n in names))
    for k in range(S):
        print(f"{sig[k]:7.3g} " + " ".join(f"{M[k, i, i] / varx[k]:7.4f}" for i in range(K)))
    print("\n== error Gram M_ij / Var(x0) at selected levels (independent errors: equal off-diagonals = J)")
    for k in np.linspace(0, S - 1, 8).astype(int):
        print(f"sigma {sig[k]:.3g}")
        for i in range(K):
            print("   " + names[i].rjust(4) + " " + " ".join(f"{M[k, i, j] / varx[k]:7.4f}" for j in range(K)))
    # ensemble means
    ens_list = a.ens.split(";") if a.ens else []
    pair_list = [p.split(":") for p in a.pairs.split(";")] if a.pairs else []
    if ens_list:
        print("\n== ensemble-mean MSE / Var(x0) (excess over the best member in brackets)")
        for e in ens_list:
            u = tile(vec(names, e))
            mse = np.einsum("ki,kij,kj->k", u, M, u)
            best = np.min([M[:, names.index(m), names.index(m)] for m in e.split(",")], 0)
            print(f"{e:20s} " + " ".join(f"{m / v:.4f}" for m, v in zip(mse, varx)))
            print(f"{'  minus best member':20s} " + " ".join(f"{(m - b) / v:+.4f}" for m, b, v in zip(mse, best, varx)))
    if pair_list:
        print("\n== VM weights: per level (positive root) and global (averaged coefficients)")
        for d1, d0 in pair_list:
            u, v = tile(vec(names, d1)), tile(vec(names, d0))
            aa, bb, cc = pair_coef(C, u, v, x)
            wp, wm = root(aa, bb, cc)
            gp, gm = root(aa.mean(), bb.mean(), cc.mean())
            # MSE-optimal weight per level: E<x0 - D0, D1 - D0> / E|D1 - D0|^2 (raw, not centred)
            num = np.einsum("ki,kij,kj->k", x - v, G, u - v)
            den = np.einsum("ki,kij,kj->k", u - v, G, u - v)
            print(f"{d1} <- {d0}: global w+ {gp:.3f} w- {gm:.3f} | per level w+: " +
                  " ".join(f"{w:.2f}" for w in wp) + "\n      w_MSE: " + " ".join(f"{w:.2f}" for w in num / den))
    if ens_list:
        print("\n== ensemble noise scale alpha (Dbar + alpha sum z_k (D_k - Dbar))")
        for e in ens_list:
            mem = e.split(",")
            u = tile(vec(names, e))
            vb = np.einsum("ki,kij,kj->k", u, C, u)
            spread = sum(np.einsum("ki,kij,kj->k", tile(np.eye(K1)[names.index(m)]) - u, G,
                                   tile(np.eye(K1)[names.index(m)]) - u) for m in mem)
            num, den = np.clip(varx - vb, 0, None), spread
            print(f"{e}: global alpha {np.sqrt(num.mean() / den.mean()):.3f} | per level: " +
                  " ".join(f"{np.sqrt(n / d):.2f}" for n, d in zip(num, den)))
            print(f"      spread / Var(x0): " + " ".join(f"{d / v:.4f}" for d, v in zip(den, varx)))
            print(f"      Var(Dbar)/Var(x0): " + " ".join(f"{b / v:.3f}" for b, v in zip(vb, varx)))


if __name__ == "__main__":
    main()
