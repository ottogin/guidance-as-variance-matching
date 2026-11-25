"""The calibration pass: a model's (D1, D0) on forward-noised calibration rows, reduced to the sums the fit needs.

For every calibration row and calibrated level the pass keeps three per-row scalars, summed over the row's
coordinates (and over its samples, for rows that hold several):

    q_xx [R]        |x0|^2
    q_xr [R, S, 2]  <x0, r_i>             columns (D0, D1)
    q_rr [R, S, 3]  <r_i, r_j>            columns (00, 01, 11)

plus the first moments s_x [d] = sum x0 and s_r [S, 2, d] = sum r_i, which give the mean terms. That is
enough for both rules of vmg.solve and for w_MSE, keeps a per-row record (row bootstrap, per-row checks), and
is a few MB even for a video latent. Shards hold disjoint rows and merge by concatenation.
"""
import json
import os
import socket
import time

import numpy as np
import torch

from . import noise
from .solve import Moments

ROW_KEYS = ("rows", "cnt", "q_xx", "q_xr", "q_rr")
FIRST_KEYS = ("s_x", "s_r")
SAME = ("model", "dataset", "seed", "levels", "d", "shape")   # meta that every shard of one pass shares


def _flat(model, x, cond):
    X, mask = model.flat(x, cond)
    X = X.double()
    m = torch.ones(X.shape[:2], dtype=torch.float64, device=X.device) if mask is None else mask.double()
    return X, m


@torch.no_grad()
def accumulate(model, dataset, rows, levels=None, batch=None, log=print, extras=()):
    """Run the pass over `rows` and return (sums, meta). `levels` are indices into model.levels().

    `extras` (vmg/extras.py) see every batch's (x0, D0, D1) at every level and add their own sums, saved
    beside the standard ones as x_<name>_<key>; the standard sums do not depend on them."""
    lv = model.levels()
    levels = list(model.calib_levels() if levels is None else levels)
    S, B = len(levels), int(batch or model.batch)
    rows = [int(r) for r in rows]
    R = len(rows)
    cnt, q_xx = np.zeros(R), np.zeros(R)
    q_xr, q_rr = np.zeros((R, S, 2)), np.zeros((R, S, 3))
    s_x = s_r = shape = None
    t0 = time.time()
    for b0 in range(0, R, B):
        rb = rows[b0:b0 + B]
        sl = slice(b0, b0 + len(rb))
        x0, cond = model.data(dataset, rb)
        x0 = x0.float()
        X, m = _flat(model, x0, cond)
        if s_x is None:
            d, shape = X.shape[-1], tuple(x0.shape[1:])
            s_x = torch.zeros(d, dtype=torch.float64, device=X.device)
            s_r = torch.zeros(S, 2, d, dtype=torch.float64, device=X.device)
        dot = lambda U, V: ((U * V).sum(-1) * m).sum(1).cpu().numpy()      # [B] per-row sums
        cnt[sl] = m.sum(1).cpu().numpy()
        q_xx[sl] = dot(X, X)
        s_x += (X * m[..., None]).sum((0, 1))
        for j, k in enumerate(levels):
            e = noise.eps_batch(model.seed, rb, k, shape, device=x0.device)
            D1, D0 = model.denoise_pair(model.noise(x0, k, e), k, cond)
            R0 = _flat(model, D0.float() - x0, cond)[0]
            R1 = _flat(model, D1.float() - x0, cond)[0]
            s_r[j, 0] += (R0 * m[..., None]).sum((0, 1))
            s_r[j, 1] += (R1 * m[..., None]).sum((0, 1))
            q_xr[sl, j, 0], q_xr[sl, j, 1] = dot(X, R0), dot(X, R1)
            q_rr[sl, j, 0], q_rr[sl, j, 1], q_rr[sl, j, 2] = dot(R0, R0), dot(R0, R1), dot(R1, R1)
            for ex in extras:
                ex.update(model, j, S, x0, D0.float(), D1.float(), cond)
        done = b0 + len(rb)
        el = time.time() - t0
        log(f"  {done}/{R} rows  {el / 60:.1f} min  eta {(R - done) * el / done / 60:.1f} min")
    sums = dict(rows=np.asarray(rows), cnt=cnt, q_xx=q_xx, q_xr=q_xr, q_rr=q_rr,
                s_x=s_x.cpu().numpy(), s_r=s_r.cpu().numpy())
    for ex in extras:
        sums.update({f"x_{ex.name}_{k}": v for k, v in ex.result().items()})
    meta = dict(model=model.spec.key, dataset=dataset, seed=int(model.seed), levels=levels,
                sigma=[float(lv["sigma"][k]) for k in levels], t=[float(lv["t"][k]) for k in levels],
                d=int(sums["s_x"].shape[0]), shape=list(shape), host=socket.gethostname(),
                device=_device_name(), seconds=time.time() - t0)
    if extras:
        meta["extras"] = [ex.name for ex in extras]
    return sums, meta


def _device_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"


def save(path, sums, meta):
    """Write one shard atomically. Large first moments are stored in float32 (a video latent has ~2M coords)."""
    big = sums["s_r"].size > 5e7
    out = {k: sums[k] for k in ROW_KEYS + ("s_x",) + tuple(k for k in sums if k.startswith("x_"))}
    out["s_r"] = sums["s_r"].astype(np.float32 if big else np.float64)
    path = str(path)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, meta=json.dumps(meta), **out)
    os.replace(tmp, path)


def load(paths):
    """Merge shards into (sums, meta). Asserts that they come from one pass and hold disjoint rows."""
    paths = [str(p) for p in paths]
    assert paths, "no moments shards"
    sums, meta = None, None
    for p in paths:
        z = np.load(p)
        mt = json.loads(str(z["meta"]))
        extra = tuple(k for k in z.files if k.startswith("x_"))       # vmg/extras.py sums: additive over rows
        part = {k: np.asarray(z[k], dtype=np.float64) for k in ROW_KEYS + FIRST_KEYS + extra}
        part["rows"] = np.asarray(z["rows"], dtype=np.int64)
        if sums is None:
            sums, meta = part, mt
            continue
        for k in SAME + ("extras",):
            assert mt.get(k) == meta.get(k), f"{p}: {k} differs from the first shard ({mt.get(k)} vs {meta.get(k)})"
        for k in ROW_KEYS:
            sums[k] = np.concatenate([sums[k], part[k]])
        for k in FIRST_KEYS + extra:
            sums[k] = sums[k] + part[k]
    assert len(np.unique(sums["rows"])) == len(sums["rows"]), "shards overlap: a row appears twice"
    meta["shards"] = len(paths)
    return sums, meta


def reduce(sums, meta, weights=None):
    """solve.Moments from the sums. `weights` [R] reweights rows (a bootstrap draw); the mean terms always
    use every row, which changes the result only at O(1/n)."""
    wt = np.ones(len(sums["rows"])) if weights is None else np.asarray(weights, dtype=np.float64)
    n_all = sums["cnt"].sum()
    n = (wt * sums["cnt"]).sum()
    d = int(meta["d"])
    xbar = sums["s_x"] / n_all                      # [d]
    rbar = sums["s_r"] / n_all                      # [S, 2, d]
    ws = lambda q: np.tensordot(wt, q, axes=(0, 0)) / n
    var_x0 = ws(sums["q_xx"]) - xbar @ xbar
    xr = ws(sums["q_xr"]) - np.einsum("d,sjd->sj", xbar, rbar)
    rr = ws(sums["q_rr"])
    mean_rr = np.stack([np.einsum("sd,sd->s", rbar[:, i], rbar[:, j]) for i, j in ((0, 0), (0, 1), (1, 1))], 1)
    return Moments(sigma=meta["sigma"], t=meta["t"], var_x0=float(var_x0) / d, xr=xr / d, rr=rr / d,
                   rr_centred=(rr - mean_rr) / d, n=int(round(n)), d=d)


def bootstrap(sums, meta, reps=200, seed=0):
    """`reps` row-bootstrap replicates of reduce()."""
    rng = np.random.default_rng(seed)
    R = len(sums["rows"])
    for _ in range(reps):
        yield reduce(sums, meta, weights=np.bincount(rng.integers(0, R, R), minlength=R))


# --- residuals of the matching statistics (vmg.solve: R_t(w) = E0 + w E1 + w^2 E2 per level) -------------------
def _sym(G):
    return G + np.swapaxes(G, -1, -2)


def chan_matrices(sums, meta):
    """(E [S, 3, C, C], MM [S, 3, C, C]) from a pass run with the chancov extra (vmg/extras.py).

    E are the centred channel covariances of the guided prediction's excess over the data's, Cov(D_w) - Cov(x0) =
    E0 + w E1 + w^2 E2, summed over positions and divided by the samples and positions; MM is what moving the
    centre to the data mean adds, the outer product of the mean errors rbar0 + w (rbar1 - rbar0). With X = x0 - xbar,
    R_i = D_i - x0 centred on their means and G_ab the normalised Gram of a and b:
        E0 = sym(G_X0) + G_00,  E1 = sym(G_X1) - sym(G_X0) + sym(G_10) - 2 G_00,  E2 = G_11 - sym(G_10) + G_00.
    E2 is Cov(D1 - D0) and -E0 is Cov(x0) - Cov(D0)."""
    assert "x_chancov_G" in sums, "this moments pass has no chancov sums: rerun its moments stage with --match cov"
    G = sums["x_chancov_G"]                                   # [S, 3C, 3C] raw sums over samples and positions
    S, C = G.shape[0], G.shape[1] // 3
    N = float(sums["cnt"].sum())
    P = int(meta["d"]) // C
    xbar = (sums["s_x"] / N).reshape(C, P)
    rbar = (sums["s_r"] / N).reshape(S, 2, C, P)              # columns (r0, r1)
    blk = lambda g, a, b: g[a * C:(a + 1) * C, b * C:(b + 1) * C]     # noqa: E731  0 = x0, 1 = r0, 2 = r1
    E, MM = np.zeros((S, 3, C, C)), np.zeros((S, 3, C, C))
    for j in range(S):
        mu = [xbar, rbar[j, 0], rbar[j, 1]]
        c = lambda a, b: (blk(G[j], a, b) - N * mu[a] @ mu[b].T) / (N * P)     # noqa: E731  centred, normalised
        X0, X1, R00, R10, R11 = c(0, 1), c(0, 2), c(1, 1), c(2, 1), c(2, 2)
        E[j] = [_sym(X0) + R00, _sym(X1) - _sym(X0) + _sym(R10) - 2 * R00, R11 - _sym(R10) + R00]
        m0, dm = rbar[j, 0], rbar[j, 1] - rbar[j, 0]
        oo = lambda u, v: u @ v.T / P                         # noqa: E731
        MM[j] = [oo(m0, m0), _sym(oo(m0, dm)), oo(dm, dm)]
    return E, MM


def residual_cov(sums, meta, rule="centred"):
    """E [S, 3, C, C] of the covariance match (the second moment about the data mean for rule "datamean")."""
    E, MM = chan_matrices(sums, meta)
    return E + MM if rule == "datamean" else E


def residual_diag(sums, meta, rule="centred"):
    """E [S, 3, d] of the per-coordinate variance match, from a pass run with the diag extra (vmg/extras.py).

    Per coordinate k, Var(D_w)_k - Var(x0)_k = 2 Cov(X, r_w)_k + Var(r_w)_k, so with c_ij the coordinatewise
    covariances E0 = 2 c_x0 + c_00, E1 = 2 (c_x1 - c_x0) + 2 (c_01 - c_00), E2 = c_11 - 2 c_01 + c_00. Their mean
    over the coordinates is the trace match's residual. "datamean" adds the squared mean error rbar_w,k^2."""
    assert "x_diag_Q" in sums, "this moments pass has no diag sums: rerun its moments stage with --match diag"
    Q = sums["x_diag_Q"]                                      # [S, 5, d]: x r0, x r1, r0 r0, r0 r1, r1 r1
    N = float(sums["cnt"].sum())
    xbar, rbar = sums["s_x"] / N, sums["s_r"] / N             # [d], [S, 2, d]
    r0, r1 = rbar[:, 0], rbar[:, 1]
    cx0, cx1 = Q[:, 0] / N - xbar * r0, Q[:, 1] / N - xbar * r1
    c00, c01, c11 = Q[:, 2] / N - r0 * r0, Q[:, 3] / N - r0 * r1, Q[:, 4] / N - r1 * r1
    E = np.stack([2 * cx0 + c00, 2 * (cx1 - cx0) + 2 * (c01 - c00), c11 - 2 * c01 + c00], 1)
    if rule == "datamean":
        dm = r1 - r0
        E = E + np.stack([r0 * r0, 2 * r0 * dm, dm * dm], 1)
    return E


def residuals(sums, meta, rule="centred"):
    """{match: E} for every non-trace matching statistic whose sums this pass holds."""
    out = {}
    if "x_chancov_G" in sums:
        out["cov"] = residual_cov(sums, meta, rule)
    if "x_diag_Q" in sums:
        out["diag"] = residual_diag(sums, meta, rule)
    return out
