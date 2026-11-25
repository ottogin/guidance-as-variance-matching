#!/usr/bin/env python
"""tab:ncalib (App. E.2): how many calibration examples our constant scale needs. The constant is refitted on random
subsets of the calibration set, each subset from its own examples alone, its means included.

    python reproduce/experiments/ncalib.py blocks --model sd35m --dataset coco --shard 0 --shards 8   # GPU
    python reproduce/experiments/ncalib.py gram --model sd35m --dataset coco --shards 8              # CPU
    python reproduce/experiments/ncalib.py fit --model sd35m --dataset coco                          # CPU
    python reproduce/experiments/ncalib.py test                                                     # CPU, synthetic

blocks  compute_w.py's calibration pass (vmg.moments.accumulate: the same rows, levels and noise draws) over every
        calibration row, which also keeps the first moments of x0, r0 = D0 - x0 and r1 = D1 - x0 summed over each
        block of BLOCK consecutive rows, at every level. A shard is a run of whole blocks; the last
        len(rows) % BLOCK rows form no block (they stay in the full-set fit).
gram    the per-block sums of the per-row scalars (level-averaged) and five K x K Grams of the block first moments
        (level-averaged): a union of blocks then has its own means through quadratic forms in the Grams. The
        constant (vmg.solve.fit_global: the root of the matching equation with a, b, c averaged over the levels,
        every level weighted equally) needs nothing else, since a, b and c are linear in the level's moments.
fit     SUBSETS random unions of n / BLOCK distinct blocks for each size n, each fitted alone, and the full set;
        for the first 300 subsets of a size, the row bootstrap over the subset's own rows and whether its 95%
        interval covers the full-set w (the coverage App. E.2 quotes). One generator (seed 0) runs through the
        sizes in order, subsets then bootstraps; sizes above half the blocks run in a fresh one without bootstrap.
        Subsets drawn without replacement from the same K blocks overlap, which narrows their spread by the
        finite-population factor sqrt(1 - k / K) for k blocks; the SD and the central 95% range (each quantile's
        distance from the mean) are divided by it, i.e. reported for n independent examples.

Outputs under $VMG_RUNS/<model>/<dataset>/ncalib[-<tag>]/: shard-<k>-of-<K>.npz (the standard sums and meta, as
vmg.moments.save), bx-<k>-of-<K>.npy [blocks, d] and br-<k>-of-<K>.npy [levels, blocks, 2, d] (float32), then
grams.npz and ncalib.json (read by reproduce/tables/tab_ncalib.py).
"""
import argparse
import json
import sys
from math import gcd
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import moments as M, runs, solve  # noqa: E402

BLOCK = 8
SIZES = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048)   # the table prints the powers of 2
SUBSETS = 2000


def outdir(model, dataset, tag=None):
    d = runs.root() / model / dataset / ("ncalib" + (f"-{tag}" if tag else ""))
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ blocks (GPU)
class BlockSums:
    """A vmg/extras.py-style extra: sums of x0, r0, r1 over each BLOCK consecutive rows of the pass, per level.
    accumulate() feeds the rows in order, so a row's place in the shard is a running count. result() is empty: the
    blocks are saved apart from the standard sums (they are GB-sized)."""
    name = "blocks"

    def __init__(self, n_rows):
        self.K = n_rows // BLOCK
        self.pos = 0
        self.bx = self.br = None

    def update(self, model, j, S, x0, D0, D1, cond):
        import torch
        X, m = M._flat(model, x0, cond)
        w = m[..., None]
        blk = (self.pos + torch.arange(len(x0))) // BLOCK
        keep = (blk < self.K).cpu()
        idx = blk[keep]
        if self.bx is None:
            d = X.shape[-1]
            self.bx = torch.zeros(self.K, d, dtype=torch.float64)
            self.br = torch.zeros(S, self.K, 2, d, dtype=torch.float32)
        if j == 0:
            self.bx.index_add_(0, idx, (X * w).sum(1).cpu()[keep])
        for i, D in enumerate((D0, D1)):
            R = M._flat(model, D - x0, cond)[0]
            self.br[j, :, i].index_add_(0, idx, (R * w).sum(1).cpu()[keep].float())
        if j == S - 1:
            self.pos += len(x0)

    def result(self):
        return {}


def shard_rows(rows, shard, shards, align):
    """This shard's rows: a run of whole blocks, its ends multiples of `align` (the last shard takes the tail)."""
    al = BLOCK * max(int(align), 1) // gcd(BLOCK, max(int(align), 1))
    cut = lambda k: len(rows) if k == shards else (len(rows) * k // shards) // al * al     # noqa: E731
    return rows[cut(shard):cut(shard + 1)]


def blocks(a):
    from vmg.models import get
    model = get(a.model)
    rows = model.rows(a.dataset, "calib")
    rows = rows[:a.n] if a.n else rows
    mine = shard_rows(rows, a.shard, a.shards, model.row_align)
    model.load()
    ex = BlockSums(len(mine))
    print(f"{model.spec.name} on {a.dataset}: rows {len(mine)} of {len(rows)} in shard {a.shard}/{a.shards}, "
          f"{ex.K} blocks of {BLOCK}")
    sums, meta = M.accumulate(model, a.dataset, mine, batch=a.batch, extras=[ex])
    meta.update(n_rows=len(rows), block=BLOCK, blocks=ex.K)
    d = outdir(a.model, a.dataset, a.tag)
    stem = f"{a.shard}-of-{a.shards}"
    np.save(d / f"bx-{stem}.npy", ex.bx.numpy())
    np.save(d / f"br-{stem}.npy", ex.br.numpy())
    M.save(d / f"shard-{stem}.npz", sums, meta)       # last: its presence marks the shard done
    print(f"wrote {d}/*-{stem}.*")


# ------------------------------------------------------------------ gram (CPU)
def block_scalars(sums, n_blocks):
    """Per-block sums of the per-row scalars, levels averaged: cnt [K], q_xr [K, 2], q_rr [K, 3]."""
    per = BLOCK * n_blocks
    pick = lambda v: v[:per].reshape(n_blocks, BLOCK, *v.shape[1:]).sum(1)        # noqa: E731
    return pick(sums["cnt"]), pick(sums["q_xr"].mean(1)), pick(sums["q_rr"].mean(1))


def grams_from(bx, br_levels):
    """Level-averaged Grams of the block first moments: G [5, K, K] for (x r0, x r1, r0 r0, r0 r1, r1 r1).
    bx [K, d]; br_levels yields [K, 2, d] per level."""
    G, S = 0.0, 0
    for r in br_levels:
        r0, r1 = r[:, 0].astype(np.float64), r[:, 1].astype(np.float64)
        G = G + np.stack([bx @ r0.T, bx @ r1.T, r0 @ r0.T, r0 @ r1.T, r1 @ r1.T])
        S += 1
    return G / S


def gram(a):
    d = outdir(a.model, a.dataset, a.tag)
    stems = [f"{k}-of-{a.shards}" for k in range(a.shards)]
    sums, meta = M.load([d / f"shard-{s}.npz" for s in stems])        # shard order = row order
    bxs = [np.load(d / f"bx-{s}.npy") for s in stems]
    br = [np.load(d / f"br-{s}.npy", mmap_mode="r") for s in stems]   # [S, Kb, 2, d] each
    bx = np.concatenate(bxs)
    K, S = len(bx), br[0].shape[0]
    # shard k's blocks are its first BLOCK * K_k rows: drop each shard's tail (only the last has one)
    nrows = [len(np.load(d / f"shard-{s}.npz")["rows"]) for s in stems]
    keep = np.concatenate([o + np.arange(BLOCK * len(b)) for o, b in zip(np.cumsum([0] + nrows[:-1]), bxs)])
    blk = {k: sums[k][keep] for k in ("cnt", "q_xr", "q_rr")}
    cnt, q_xr, q_rr = block_scalars(blk, K)
    G = grams_from(bx, (np.concatenate([b[j] for b in br]) for j in range(S)))
    np.savez(d / "grams.npz", G=G, cnt=cnt, q_xr=q_xr, q_rr=q_rr, row_cnt=blk["cnt"], row_xr=blk["q_xr"].mean(1),
             row_rr=blk["q_rr"].mean(1), meta=json.dumps(meta))
    print(f"wrote {d / 'grams.npz'}: {K} blocks, {S} levels, d = {meta['d']}")


# ------------------------------------------------------------------ fit (CPU)
def const_w(xr, rr, rrc, d):
    """vmg.solve.fit_global for each row of level-averaged moments xr [T, 2], rr and rrc [T, 3] (sums over n):
    a, b and c are linear in a level's moments, so their level average is that of the averaged moments."""
    m = solve.Moments(sigma=np.zeros(len(xr)), t=np.zeros(len(xr)), var_x0=np.nan, xr=xr / d, rr=rr / d,
                      rr_centred=rrc / d, n=0, d=d)
    return np.array([solve.root(*abc)[0] for abc in zip(*solve.coefficients(m))])


def subset_means(I, G, n):
    """[T, 5] products of each subset's mean vectors (x r0, x r1, r0 r0, r0 r1, r1 r1), levels averaged:
    quadratic forms of the block indicators I [T, K] in the Grams, over the subsets' n [T] samples squared."""
    K = I.shape[1]
    q = ((I @ G.transpose(1, 0, 2).reshape(K, -1)).reshape(len(I), -1, K) * I[:, None]).sum(-1)
    return q / n[:, None] ** 2


def subset_w(I, G, cnt, q_xr, q_rr, d):
    """Constant w of each union of blocks I [T, K] (0/1), from its own sums and its own means."""
    n = I @ cnt
    mm = subset_means(I, G, n)
    rr = (I @ q_rr) / n[:, None]
    return const_w((I @ q_xr) / n[:, None] - mm[:, :2], rr, rr - mm[:, 2:], d)


def fit_run(z, d, sizes, rng, subsets, boot_subsets, boot, w_full):
    """One generator stream over `sizes`: per size n, `subsets` random unions of k = n / BLOCK distinct blocks
    (k = 1: every block alone), then for the first `boot_subsets` of them the row bootstrap a practitioner would
    report, over the subset's own rows with its means held, and whether its 95% interval covers the full-set w."""
    G, cnt, q_xr, q_rr = z["G"], z["cnt"], z["q_xr"], z["q_rr"]
    rc, rxr, rrr = z["row_cnt"], z["row_xr"], z["row_rr"]
    K, out = len(cnt), []
    for n in sizes:
        k = n // BLOCK
        assert k * BLOCK == n and k < K, f"size {n}: a multiple of {BLOCK} below {K * BLOCK}"
        subs = np.arange(K)[:, None] if k == 1 else np.stack([rng.choice(K, k, replace=False) for _ in range(subsets)])
        I = np.zeros((len(subs), K))
        np.put_along_axis(I, subs, 1.0, 1)
        nn = I @ cnt
        mm = subset_means(I, G, nn)
        rr = (I @ q_rr) / nn[:, None]
        w = const_w((I @ q_xr) / nn[:, None] - mm[:, :2], rr, rr - mm[:, 2:], d)
        cover = []
        for r in range(min(boot_subsets, len(subs)) if k > 1 else 0):
            rows = (subs[r][:, None] * BLOCK + np.arange(BLOCK)).ravel()
            W = rng.multinomial(n, np.full(n, 1 / n), size=boot).astype(np.float64)       # [boot, n]
            nb = W @ rc[rows]
            brr = (W @ rrr[rows]) / nb[:, None]
            wb = const_w((W @ rxr[rows]) / nb[:, None] - mm[r, :2], brr, brr - mm[r, 2:], d)
            cover.append(abs(w[r] - w_full) <= 1.96 * np.std(wb, ddof=1))
        f = 1 / np.sqrt(1 - k / K) if k > 1 else 1.0       # subsets overlap: the spread of an infinite pool
        c = w.mean()
        lo, hi = np.quantile(w, [0.025, 0.975])
        row = dict(n=n, blocks=k, subsets=len(subs), mean=float(c), sd=float(f * w.std(ddof=1)),
                   lo=float(c + f * (lo - c)), hi=float(c + f * (hi - c)), sd_raw=float(w.std(ddof=1)),
                   q025_raw=float(lo), q975_raw=float(hi), nan=int(np.isnan(w).sum()),
                   boot_coverage95=float(np.mean(cover)) if cover else None)
        out.append(row)
        cv = "" if not cover else f"  bootstrap covers {100 * row['boot_coverage95']:.0f}%"
        print(f"  n {n:5d}: mean {c:.3f}  SD {row['sd']:.3f}  95% {row['lo']:.2f}-{row['hi']:.2f}{cv}", flush=True)
    return out


def fit(a):
    d = outdir(a.model, a.dataset, a.tag)
    z = dict(np.load(d / "grams.npz"))
    meta = json.loads(str(z["meta"]))
    sums, _ = M.load(sorted(d.glob("shard-*.npz")))
    w_full = solve.fit_global(M.reduce(sums, meta))[0]         # every row, as compute_w.py --mode global
    K = len(z["cnt"])
    # sizes up to max_frac of the blocks in one stream with the bootstrap, larger ones in a fresh stream without
    small = [n for n in a.sizes if n // BLOCK <= int(a.max_frac * K) and n // BLOCK < K]
    big = [n for n in a.sizes if n // BLOCK < K and n not in small]
    print(f"{a.model} on {a.dataset}: {len(sums['rows'])} rows, {K} blocks, full-set w {w_full:.5f}")
    D = int(meta["d"])
    res = fit_run(z, D, small, np.random.default_rng(a.seed), a.subsets, a.boot_subsets, a.boot, w_full) + \
        fit_run(z, D, big, np.random.default_rng(a.seed), a.subsets, 0, a.boot, w_full)
    out = dict(model=a.model, dataset=a.dataset, rows=len(sums["rows"]), levels=len(meta["levels"]), block=BLOCK,
               blocks=K, subsets=a.subsets, boot_subsets=a.boot_subsets, boot=a.boot, max_frac=a.max_frac,
               seed=a.seed, w_full=float(w_full), sizes=res,
               correction="SD and the 2.5% / 97.5% quantiles' distance from the mean divided by sqrt(1 - k / K)")
    (d / "ncalib.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {d / 'ncalib.json'}")


# ------------------------------------------------------------------ test (synthetic, CPU)
def test(a):
    """The block machinery against a direct subset fit on synthetic fields (subset means included)."""
    rng = np.random.default_rng(0)
    R, S, d = 8 * 12 + 3, 4, 6
    x0 = rng.normal(size=(R, d)) + 0.3
    r = rng.normal(size=(R, S, 2, d)) * np.array([1.0, 0.6])[None, None, :, None] - 0.4 * x0[:, None, None]
    sums = dict(cnt=np.ones(R), q_xr=np.einsum("rd,rsid->rsi", x0, r),
                q_rr=np.stack([np.einsum("rsd,rsd->rs", r[:, :, i], r[:, :, j]) for i, j in ((0, 0), (0, 1), (1, 1))], 2))
    K = R // BLOCK
    bx = x0[:BLOCK * K].reshape(K, BLOCK, d).sum(1)
    br = r[:BLOCK * K].reshape(K, BLOCK, S, 2, d).sum(1)          # [K, S, 2, d]
    G = grams_from(bx, (br[:, j] for j in range(S)))
    cnt, q_xr, q_rr = block_scalars(sums, K)
    for blocks_ in ([0, 3, 7], [1, 2], list(range(K))):
        I = np.zeros((1, K))
        I[0, blocks_] = 1
        got = subset_w(I, G, cnt, q_xr, q_rr, d)[0]
        rows = np.concatenate([np.arange(BLOCK) + BLOCK * b for b in blocks_])
        X, Rr = x0[rows], r[rows]
        Xc, Rc = X - X.mean(0), Rr - Rr.mean(0)
        xr = np.einsum("rd,rsid->si", Xc, Rr) / len(rows)
        rrc = np.stack([np.einsum("rsd,rsd->s", Rc[:, :, i], Rc[:, :, j]) for i, j in ((0, 0), (0, 1), (1, 1))], 1)
        m = solve.Moments(sigma=np.zeros(S), t=np.zeros(S), var_x0=np.nan, xr=xr / d, rr=rrc / len(rows) / d,
                          rr_centred=rrc / len(rows) / d, n=len(rows), d=d)
        want = solve.fit_global(m)[0]
        assert np.isfinite(want) and np.isclose(got, want, rtol=1e-10), (blocks_, got, want)
    print("ncalib test OK: block-Gram subset fits equal direct fits with each subset's own means")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["blocks", "gram", "fit", "test"])
    ap.add_argument("--model")
    ap.add_argument("--dataset")
    ap.add_argument("--n", type=int, help="blocks: only the first N calibration rows (a smoke run)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--batch", type=int, help="blocks: rows per forward batch (default: the model's)")
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=list(SIZES))
    ap.add_argument("--subsets", type=int, default=SUBSETS, help="fit: random subsets per size")
    ap.add_argument("--boot-subsets", type=int, default=300, help="fit: subsets per size that get the bootstrap")
    ap.add_argument("--boot", type=int, default=200, help="fit: bootstrap replicates per subset")
    ap.add_argument("--max-frac", type=float, default=0.5,
                    help="fit: sizes above this fraction of the blocks run in a second stream, without bootstrap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag")
    a = ap.parse_args()
    if a.stage != "test":
        assert a.model and a.dataset, "--model and --dataset are required"
    {"blocks": blocks, "gram": gram, "fit": fit, "test": test}[a.stage](a)


if __name__ == "__main__":
    main()
