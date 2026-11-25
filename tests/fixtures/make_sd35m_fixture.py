#!/usr/bin/env python
"""Build tests/fixtures/sd35m-coco-s40.npz from the stored SD 3.5 Medium calibration pass behind the paper.

    python tests/fixtures/make_sd35m_fixture.py [--mom results/domains/mom/coco-sd35m-n5705g-s40.npz]

Run once, where the pass's results/ tree exists; the fixture (~70 KB) is committed and the tests need nothing else.
The source pass: COCO photos 0-5,704, 40 flow-matching Euler levels, seed 0. It stores, per level, the centred per-coordinate moments V_* (summed over photos) and,
per photo, the raw residual products gram_p = (<r1,r1>, <r1,r2>, <r2,r2>) / d, where r1 is the conditional and
r2 the unconditional residual. The fixture keeps the per-level reductions, computed in the same order of
operations as the fit that produced the reference schedules, and the first 64 photos' products.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mom", default=str(REPO / "results/domains/mom/coco-sd35m-n5705g-s40.npz"))
    ap.add_argument("--ref", default=str(REPO / "results/vm-datamean/w-s40-datamean.npz"),
                    help="the data-mean and centred schedules fitted from the same pass")
    ap.add_argument("--out", default=str(Path(__file__).with_name("sd35m-coco-s40.npz")))
    a = ap.parse_args()
    z = np.load(a.mom)
    n, d = int(z["n"]), int(z["dim"])
    t = np.asarray(z["sigmas"], dtype=np.float64)              # flow-matching t of the 40 sampler levels
    V = {k: np.asarray(z[f"V_{k}"], dtype=np.float64) / n for k in ("XX", "XR1", "XR2", "R11", "R12", "R22")}
    cx1, cx2 = V["XR1"].sum(1) / d, V["XR2"].sum(1) / d
    cen = np.stack([V["R11"].sum(1), V["R12"].sum(1), V["R22"].sum(1)], 1) / d
    unc = np.asarray(z["gram_p"], dtype=np.float64).mean(0)
    flip = [2, 1, 0]                                            # (11, 12, 22) of r1=D1, r2=D0 -> (00, 01, 11)
    ref = np.load(a.ref)
    assert np.allclose(ref["sigmas"], t)
    np.savez_compressed(
        a.out, t=t, sigma=np.where(t < 1, t / np.maximum(1 - t, 1e-12), 1e4), n=n, d=d,
        var_x0=V["XX"].sum(1)[0] / d, xr=np.stack([cx2, cx1], 1), rr=unc[:, flip], rr_centred=cen[:, flip],
        gram_p64=np.asarray(z["gram_p"][:64], dtype=np.float64)[:, :, flip],
        w_datamean=ref["w:datamean"], w_centred=ref["w:trace"],
        source=Path(a.mom).name, source_sha256=hashlib.sha256(open(a.mom, "rb").read()).hexdigest())
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
