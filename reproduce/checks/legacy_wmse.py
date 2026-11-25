#!/usr/bin/env python
"""Write the stored w_MSE results behind fig:wmse-published as wmse.json files, to check the figure scripts.

    python reproduce/checks/legacy_wmse.py --runs /tmp/legacy-runs
    VMG_RUNS=/tmp/legacy-runs python reproduce/figures/fig_wmse_published.py --out /tmp/wmse-published

The figures normally read runs/<model>/<dataset>/wmse.json from fresh compute_wmse.py runs. This turns the
earlier pipeline's results/wmse/summary.json (one row per pair) into the same files, so the ported figure
scripts can be compared with the paper's copies before any pair is recomputed. Only this check reads the old
results/ tree, which is not part of this repository: results/wmse/summary.json and red/ (GRAM, FRAMES below).
Flow and SD-family rows stored the schedule coordinate s, mapped to sigma = s / (1 - s).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from reproduce.rosters.wmse_pairs import CROSS, EXTRA, PAIRS, T2I  # noqa: E402

RES = REPO / "results" / "wmse"
GRAM = RES / "red" / "cov__cov__coco-sd35m-n5705g-s40.npz"     # per-photo products behind SD3.5-M COCO
FRAMES = RES / "red" / "frames__video__svd.npz"                 # SVD per-frame products, 16 shards


def from_gram(path, pub, reps=200, seed=0):
    """w_MSE and the denoising errors, with a photo bootstrap, from per-photo (r1.r1, r1.r2, r2.r2) / d.
    r1 = D_cond - x0 = D1 - x0 and r2 = D_uncond - x0 = D0 - x0, so (m00, m01, m11) = columns (2, 1, 0)."""
    g = np.load(path, allow_pickle=True)["gram_p"].astype(np.float64)

    def fits(mean):
        m00, m01, m11 = mean[..., 2], mean[..., 1], mean[..., 0]
        b = m11 - 2 * m01 + m00
        w = (m00 - m01) / b
        e = lambda v: m00 + 2 * v * (m01 - m00) + v * v * b      # noqa: E731
        return w, e(1.0), e(w), e(pub)

    rng = np.random.default_rng(seed)
    n = len(g)
    pt = fits(g.mean(0))
    bt = [np.stack(x) for x in zip(*(fits(g[rng.integers(0, n, n)].mean(0)) for _ in range(reps)))]
    band = lambda k: np.percentile(bt[k], [2.5, 97.5], axis=0)    # noqa: E731
    return dict(w_mse=pt[0], ci_lo=band(0)[0], ci_hi=band(0)[1], mse_w1=pt[1], mse_wmse=pt[2], mse_published=pt[3],
                mse_w1_ci=band(1), mse_wmse_ci=band(2), mse_published_ci=band(3))


def svd_frame1(path):
    """w_MSE on frame 1's coordinates alone, with the delete-one-shard jackknife of the paper's figure."""
    z = np.load(path)
    P = z["s_rr"][..., 0]                     # [K, S, 3]: frame 1, columns (r1.r1, r1.r2, r2.r2)
    f = lambda S3: (S3[..., 2] - S3[..., 1]) / (S3[..., 0] + S3[..., 2] - 2 * S3[..., 1])   # noqa: E731
    K, tot = P.shape[0], P.sum(0)
    jw = np.stack([f(tot - P[k]) for k in range(K)])
    se = np.sqrt((K - 1) / K * ((jw - jw.mean(0)) ** 2).sum(0))
    w = f(tot)
    return dict(sig=z["sigmas"].tolist(), w=w, lo=w - 1.96 * se, hi=w + 1.96 * se)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", required=True, help="a scratch VMG_RUNS root to write into")
    a = ap.parse_args()
    rows = {r["name"]: r for r in json.load(open(RES / "summary.json"))}
    rows["SVD frame 1"] = svd_frame1(FRAMES)
    for p in PAIRS + EXTRA:
        if p.old not in rows:
            print(f"missing: {p.old}")
            continue
        r = rows[p.old]
        s = np.asarray(r["sig"], float)
        sigma = np.where(s >= 1 - 1e-9, np.inf, s / np.maximum(1 - s, 1e-12)) if p.group in (T2I, CROSS) else s
        w = np.asarray(r["w"], float)
        out = dict(model=p.model, dataset=p.dataset, legacy=p.old, sigma=sigma.tolist(), w_mse=w.tolist(),
                   ci_lo=None if r.get("lo") is None else list(r["lo"]),
                   ci_hi=None if r.get("hi") is None else list(r["hi"]), median=float(np.nanmedian(w)),
                   published_w=p.pub)
        if p == next(q for q in PAIRS if q.old == "SD3.5-M COCO (40 steps)"):
            g = from_gram(GRAM, p.pub)
            assert np.allclose(g["w_mse"], w, atol=2e-3), "the Gram file is not the summary row"
            out.update({k: v.tolist() for k, v in g.items() if k not in ("w_mse", "ci_lo", "ci_hi")})
        d = Path(a.runs) / p.model / p.dataset
        d.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(d / "wmse.json", "w"), indent=0, default=lambda o: None if o is None else float(o))
    print(f"wrote {len(PAIRS) + len(EXTRA)} pairs under {a.runs}")


if __name__ == "__main__":
    main()
