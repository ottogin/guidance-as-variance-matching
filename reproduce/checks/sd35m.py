#!/usr/bin/env python
"""GPU checks of the SD 3.5 Medium adapter against the calibration pass behind the paper.

    python reproduce/checks/sd35m.py pipeline --legacy-captions coco-captions.json --legacy-latents coco-sd3-latents.npy
    python reproduce/checks/sd35m.py fit     # after compute_w.py / compute_wmse.py --model sd35m --dataset coco --n all

pipeline (one GPU, ~15 min):
  G1  the COCO photo order and captions equal the frozen list the paper used
  G2  fresh latents of photos 0-63 equal the stored ones
  G3  denoise_pair equals the released pipeline's own transformer call (model.smoke)
  G4  a 64-photo moments pass equals the stored per-photo residual products (tests/fixtures)
  G6  a constant schedule through the hook is bitwise the plain pipeline, and a varying schedule reaches
      every step at the right weight
fit (CPU): G5  the full pass gives the stored constant weight, schedule and w_MSE, to within the cross-GPU
          noise of the network (a fraction of the fit's own bootstrap SE).
The legacy files (the frozen caption list and latents behind the paper) are not part of this repository and are
optional; G1 / G2 are skipped without them. G4 / G5 compare with tests/fixtures/sd35m-coco-s40.npz, which is.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "tests/fixtures/sd35m-coco-s40.npz"
failed = []


def gate(name, ok, msg):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)
    if not ok:
        failed.append(name)


def pipeline(a):
    import torch
    from vmg import moments as M
    from vmg.data import coco
    from vmg.models import get
    fx = np.load(FIX)
    if a.legacy_captions:
        old = json.load(open(a.legacy_captions))["captions"]
        new = coco.select()
        same = len(old) == len(new) and all(o["image_id"] == n["image_id"] and o["full"] == n["caption"]
                                            and o["est"] == n["eligible"] for o, n in zip(old, new))
        gate("G1 photo order", same, f"{len(new)} photos, {sum(n['eligible'] for n in new)} eligible")
    m = get("sd35m")
    m.load()
    if a.legacy_latents:
        old = np.load(a.legacy_latents, mmap_mode="r")
        new = m.latents(coco.images(range(64))).cpu().numpy()
        d = float(np.abs(new - old[:64]).max())
        var = float(np.asarray(old[:coco.N_CALIB], dtype=np.float64).var(0).mean())
        med = float(np.median(np.abs(new - old[:64])))
        # fp32 convolutions are not bitwise across GPU types (TF32, kernel choice); the latents have std ~0.7
        gate("G2 latents", d <= 1e-2, f"max |diff| {d:.2e}, median {med:.1e} on photos 0-63; stored Var[x0] per "
             f"coord {var:.6f} (the pass: {float(fx['var_x0']):.6f})")
    try:
        m.smoke()
        gate("G3 pipeline parity", True, "see above")
    except AssertionError as e:
        gate("G3 pipeline parity", False, str(e))
    sums, meta = M.accumulate(m, "coco", list(range(64)), batch=8)
    got = sums["q_rr"] / meta["d"]                                   # [64, 40, 3] (00, 01, 11)
    ref = fx["gram_p64"]
    rel = np.abs(got - ref) / np.abs(ref)
    # the stored pass ran on other GPU types, so the comparison is not bitwise; the decisive check is G5
    gate("G4 per-photo products", np.median(rel) <= 1e-3 and np.percentile(rel, 99) <= 1e-2,
         f"relative diff median {np.median(rel):.1e}, 99% {np.percentile(rel, 99):.1e}, max {rel.max():.1e}")
    caps = coco.captions(coco.rows("eval")[:2])
    z = m.initial_noise(coco.rows("eval")[:2])
    plain = m.pipe(prompt=caps, height=m.size, width=m.size, num_inference_steps=m.spec.steps, guidance_scale=4.5,
                   max_sequence_length=m.max_tokens, latents=z.to("cuda", m.dtype), output_type="latent").images
    hooked = m.generate(caps, z, np.full(m.spec.steps, 4.5))
    gate("G6a constant schedule", torch.equal(plain, hooked), "hook + LiveScale at 4.5 vs guidance_scale=4.5")
    lv = m.levels()
    w = 1.5 + np.arange(m.spec.steps) / 10
    _, seen = m.generate(caps, z, w, trace=True)
    got_w = np.array([s[1] for s in seen])
    want = np.array([w[int(np.argmin(np.abs(lv["t"] - s[0])))] for s in seen])
    gate("G6b schedule reaches every step", len(seen) == m.spec.steps and np.allclose(got_w, want)
         and np.allclose(got_w, w), f"{len(seen)} network calls, weights {got_w[:3]} ... {got_w[-1]}")


def fit(a):
    from vmg import runs
    fx = np.load(FIX)
    w = json.load(open(runs.w_json("sd35m", "coco")))
    g, se = w["global"]["w"], w["global"]["se"]
    # Same photos and noise as the stored pass, but other GPU types: the bf16 network outputs differ by ~1e-3 per
    # photo (G4), which moves the fit by a fraction of its own row-bootstrap SE. The stored tr Cov fit is 4.13408,
    # the paper's printed 4.14.
    gate("G5 constant weight", abs(g - 4.13408) <= 0.5 * se and abs(g / 4.13408 - 1) <= 2e-3,
         f"{g:.5f} vs 4.13408 (stored pass, same noise): {abs(g - 4.13408) / se:.2f} bootstrap SE")
    sw, sse = np.asarray(w["schedule"]["w"]), np.asarray(w["schedule"]["se_raw"], dtype=float)
    rel = np.abs(sw / fx["w_centred"] - 1)[:-1]
    zz = (np.abs(sw - fx["w_centred"]) / np.where(sse > 0, sse, np.inf))[:-1]
    gate("G5 schedule", np.median(rel) <= 2e-3 and rel.max() <= 1e-2 and zz.max() <= 2.5,
         f"levels 0-38: relative diff median {np.median(rel):.1e}, max {rel.max():.1e}; "
         f"median {np.median(zz):.2f} SE, max {zz.max():.2f} SE")
    gate("G5 Var[x0]", abs(w["var_x0"] / float(fx["var_x0"]) - 1) <= 1e-3, f"{w['var_x0']:.6f} vs {float(fx['var_x0']):.6f}")
    p = runs.wmse_json("sd35m", "coco")
    if p.exists():
        e = np.asarray(json.load(open(p))["w_mse"])
        gate("G5 w_MSE", np.allclose(e[:3], [1.0743, 1.0339, 1.0153], atol=3e-3), f"first steps {np.round(e[:3], 4)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("pipeline", "fit"))
    ap.add_argument("--legacy-captions")
    ap.add_argument("--legacy-latents")
    a = ap.parse_args()
    (pipeline if a.what == "pipeline" else fit)(a)
    print("ALL PASS" if not failed else f"FAILED: {', '.join(failed)}")
    sys.exit(1 if failed else 0)
