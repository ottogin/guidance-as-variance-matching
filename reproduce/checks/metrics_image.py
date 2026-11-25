#!/usr/bin/env python
"""Gate M: the image metrics reproduce the features and distances behind the paper's image tables.

    python reproduce/checks/metrics_image.py --old-ref <dir with the earlier -pr references> [--n 64] \
        [--datasets coco,mjhq,imagenet512]
    python reproduce/checks/metrics_image.py --rescore <stored arm dir> --table <pr-table json> --old-ref <dir>

1. Real-side features of the first N evaluation images from vmg.metrics.image against the per-image features
   stored in the earlier references (EDM2's detectors, the same crop and resize). DINOv2 agrees to ~1e-5 of the
   features' scale, which shows the images reach the detectors unchanged; Inception's TorchScript convolutions
   vary by ~2e-3 across GPU types, batch sizes and cuDNN algorithms (neither EDM2 nor the earlier code disables
   TF32), far below the printed digits of a 10,000-image FID. For ImageNet-512 the held-out rows themselves
   (which 10,000 val images, which classes) must equal the stored reference's.
2. frechet() on two halves of a stored reference against the earlier formula from feature sums.
3. --rescore (CPU, no images): one stored ImageNet-512 sweep arm's per-image features, scored by the new code
   (FID / FD_DINOv2 against NVIDIA's training statistics (each space's fd_stats), k = 3 P/R/D/C against the stored
   held-out reference) against the stored table row.
Exits non-zero on failure. Every input named --old-ref, --rescore or --table is a file of the earlier pipeline
(the *-pr.npz references in OLD, a sweep arm's shard-*.npz, a pr-table json), not part of this repository and made
by no job here, so this check runs only where those files are.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OLD = {"coco": "coco-val2014-10000-o5705-256-pr.npz", "mjhq": "mjhq-test-10000-o5705-256-pr.npz",
       "imagenet512": "img512-val-n10000-pr.npz"}


def old_frechet(fa, fb):
    """The earlier pipeline's Frechet distance: statistics from float64 sums, then scipy sqrtm."""
    return stats_frechet(*stats(fa), *stats(fb))


def stats(f):
    f = f.astype(np.float64)
    n, mu = len(f), f.sum(0) / len(f)
    return mu, (f.T @ f - np.outer(mu, mu) * n) / (n - 1)


def stats_frechet(ma, sa, mb, sb):
    import scipy.linalg
    s = scipy.linalg.sqrtm(sa @ sb)
    return float(np.real(np.square(ma - mb).sum() + np.trace(sa + sb - 2 * s)))


def real_side(name, n, old, dets):
    """(new features, stored features) of the first n evaluation rows' real images."""
    import torch
    from vmg.metrics import image as I
    if name in I.IMAGENET:
        from vmg.data import imagenet as D
        rows = D.heldout_rows(res=I.IMAGENET[name])
        assert np.array_equal(rows, old["idx"]), "the held-out val rows differ from the stored reference's"
        assert np.array_equal(D.labels(I.IMAGENET[name])[list(rows)], old["labels"]), "their classes differ"
        print(f"{name}: the {len(rows)} held-out rows and their classes equal the stored reference's")
        u8 = torch.from_numpy(D.real_u8(rows[:n], I.IMAGENET[name])).permute(0, 3, 1, 2).cuda()
    else:
        real, _ = I._dataset(name)
        ids = list(range(5705, 5705 + n))
        u8 = torch.from_numpy(np.stack([I.real_to_metric(real(i)) for i in ids])).permute(0, 3, 1, 2).cuda()
    acc = I.Features(dets)
    acc.update(u8)
    return acc.numpy()


def check_real(a):
    from vmg.metrics import image as I
    dets = I.detectors()
    ok = True
    for name in a.datasets.split(","):
        old = np.load(Path(a.old_ref) / OLD[name], allow_pickle=True)
        new = real_side(name, a.n, old, dets)
        for k, d in enumerate(I.DETECTORS):
            ref = old[f"feat{k}"][:a.n].astype(np.float64)
            err = np.abs(new[d] - ref).max() / np.abs(ref).max()
            print(f"{name:11s} {d:9s} max |new - old| / max |old| = {err:.2e}")
            ok &= err < (5e-3 if d == "inception" else 1e-4)
        for k, d in enumerate(I.DETECTORS):
            f = old[f"feat{k}"]
            h = len(f) // 2
            x, y = I.frechet(f[:h], f[h:]), old_frechet(f[:h], f[h:])
            print(f"{name:11s} {d:9s} FD of the two halves: {x:.6f} vs earlier formula {y:.6f}")
            ok &= abs(x - y) <= 1e-6 * max(1.0, abs(y))
    return ok


def rescore(a):
    """Gate 3: a stored sweep arm's features through the new scoring, against the stored table row."""
    from vmg.metrics import common as C, image as I
    tag = os.path.basename(os.path.normpath(a.rescore))
    shards = sorted(glob.glob(os.path.join(a.rescore, "shard-*.npz")))
    zs = [np.load(p) for p in shards]
    fake = {d: np.concatenate([z[f"feat{k}"] for z in zs]) for k, d in enumerate(I.DETECTORS)}
    old = np.load(Path(a.old_ref) / OLD["imagenet512"], allow_pickle=True)
    real = {d: old[f"feat{k}"] for k, d in enumerate(I.DETECTORS)}
    ok = True
    new = C.scores(fake, real, I.spaces(None, "imagenet512"))    # Frechet against NVIDIA's statistics (fd_stats)
    row = next(r for r in json.load(open(a.table))["rows"] if r["tag"] == tag)
    worst = 0.0
    for m, v in new.items():
        if m in row:
            rel = abs(v - row[m]) / max(abs(row[m]), 1e-12)
            worst = max(worst, rel)
            print(f"  {m:22s} new {v:12.6f}  stored {row[m]:12.6f}  rel {rel:.1e}")
    ok &= worst < 1e-6
    print(f"{tag}: {len(fake['inception'])} images, worst relative difference {worst:.1e}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old-ref", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--datasets", default="coco", help="comma-separated: coco, mjhq, imagenet512")
    ap.add_argument("--rescore", help="a stored sweep arm (dir of shard-*.npz with feat0 / feat1)")
    ap.add_argument("--table", help="--rescore: the stored pr-table json holding that arm's row")
    a = ap.parse_args()
    ok = rescore(a) if a.rescore else check_real(a)
    print("METRICS PASS" if ok else "METRICS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
