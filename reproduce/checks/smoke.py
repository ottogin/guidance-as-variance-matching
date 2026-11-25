#!/usr/bin/env python
"""Gate S: does a model's adapter run end to end? One GPU. Measured on an L40S: SD 1.5 5 min, FLUX.2 [klein]
12 min, CogVideoX 27 min, Wan2.1 60 min.

    python reproduce/checks/smoke.py --model sd15 [--dataset coco]

1. model.smoke(): denoise_pair matches the released pipeline's own network call
2. compute_w.py / compute_wmse.py --smoke: moments on 2 calibration rows, both fits, finite output
3. the guidance hook (adapters with generate()): a constant schedule is bitwise the plain pipeline, and a
   varying schedule reaches every network call at the weight of its level
4. sample.py on 2 evaluation rows at the published weight and at the fitted schedule (skipped with
   --no-sample, for the moment-only pairs of fig:wmse-published that the paper never samples)
Exits non-zero on the first failure.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def run(*args):
    print("$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--no-sample", action="store_true", help="moments and fits only (no sample arm in the paper)")
    a = ap.parse_args()
    from vmg import runs
    from vmg.models import get
    m = get(a.model)
    ds = a.dataset or m.spec.default_dataset
    d = ["--model", a.model, "--dataset", ds]

    run("compute_w.py", *d, "--smoke", "--redo")            # includes model.smoke()
    run("compute_wmse.py", *d, "--smoke", "--stage", "fit")
    w = json.load(open(runs.w_json(a.model, ds, "smoke")))
    S = len(w["levels"])
    assert S == (m.spec.calib_levels or S), f"{S} levels, the paper calibrates {m.spec.calib_levels}"
    co = {k: np.asarray(v, float) for k, v in w["coefficients"].items()}
    assert all(np.isfinite(v).all() for v in co.values()) and (co["b"] > 0).all(), "non-finite moments"
    print(f"fit OK: {S} levels, finite moments, b > 0; global w {w['global']['w']} "
          f"(2 rows: the weights themselves are noise, possibly NaN)")

    if hasattr(m, "generate"):
        import torch
        m.load()
        rows = m.rows(ds, "eval")[:1]
        if hasattr(m, "captions"):                  # adapters whose prompts are not COCO's
            caps = m.captions(ds, rows)
        else:
            from vmg.data import coco
            caps = coco.captions(rows)
        z = m.initial_noise(rows)
        pub = m.spec.published_w
        net = getattr(m.pipe, "transformer", None) or m.pipe.unet
        plain = m.pipe(prompt=caps, height=m.size, width=m.size, num_inference_steps=m.spec.steps, guidance_scale=pub,
                       latents=z.to("cuda", m.dtype), output_type="latent",
                       **({"max_sequence_length": m.max_tokens} if hasattr(m, "max_tokens") else {})).images
        hooked = m.generate(caps, z, np.full(S, pub))
        assert torch.equal(plain, hooked), "a constant schedule through the hook is not the plain pipeline"
        ws = 1.5 + np.arange(S) / 10
        _, seen = m.generate(caps, z, ws, trace=True)
        lv = m.levels()
        want = [ws[int(np.argmin(np.abs(lv["t"] - t)))] for t, _ in seen]
        assert np.allclose([s for _, s in seen], want), "the schedule did not reach every network call"
        print(f"hook OK: constant {pub} bitwise; {len(seen)} network calls at their levels' weights ({type(net).__name__})")
        import gc
        m = net = plain = hooked = None             # sample.py loads its own copy; two SD3.5-L copies overflow an L40S
        gc.collect()
        torch.cuda.empty_cache()

    if a.no_sample:
        print(f"SMOKE PASS {a.model} (moments only)")
        return
    run("sample.py", *d, "--published", "--n", "2", "--tag", "smoke")
    run("sample.py", *d, "--weights", str(runs.w_json(a.model, ds, "smoke")), "--kind", "schedule", "--n", "2",
        "--tag", "smoke")
    out = runs.run_dir(a.model, ds, "smoke") / "samples"
    files = {arm.name: sorted(p.name for p in arm.iterdir()) for arm in out.iterdir()}
    assert all(len(v) >= 1 for v in files.values()), files
    print(f"samples OK: {files}")
    print(f"SMOKE PASS {a.model}")


if __name__ == "__main__":
    main()
