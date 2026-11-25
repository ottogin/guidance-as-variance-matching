#!/usr/bin/env python
"""Sample SD 3.5 Medium with the posterior-DDPM sampler of vmg/ddpm.py: x~0 ~ N(D1, Sigma from kappa (D1 - D0)),
then the exact DDPM reverse step given x~0.

    python reproduce/experiments/ddpm_posterior.py --gate                                  # one GPU, a few minutes
    python reproduce/experiments/ddpm_posterior.py --shape iso --kappa 3.5 --shard 0 --shards 8
    python reproduce/experiments/ddpm_posterior.py --shape none                           # plain DDPM at D1
    python reproduce/experiments/ddpm_posterior.py --shape none --mean dw --w 4.5         # DDPM with CFG at 4.5

Arms go to runs/sd35m/coco[-<tag>]/samples/<arm>/ (arm names from PosteriorDDPM.name, e.g. ddpm-iso-k3.5) with
the sampler's settings in ddpm.json; score them with reproduce/experiments/evaluate.py. The prompts and initial
noise are sample.py's, so every arm is paired row by row with sample.py's arms (pub-f32 is the released sampler
with the same float32 state), and the DDPM arms share their step noise too.

--gate checks, on two evaluation rows:
  G1  our loop with the Euler step and CFG at 4.5 equals the pipeline's generate(..., fp32_state=True)
  G2  every DDPM arm runs, ends finite, and differs from the Euler result
and writes a contact sheet of every arm (ddpm-gate.png in the run folder) to look at.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import cli, runs  # noqa: E402
from vmg.ddpm import MEANS, SHAPES, PosteriorDDPM  # noqa: E402

GATE_ARMS = [PosteriorDDPM("none", 0.0, "d1"), PosteriorDDPM("none", 0.0, "dw", 4.5),
             PosteriorDDPM("iso", 3.5), PosteriorDDPM("iso", 4.5), PosteriorDDPM("diag", 3.5),
             PosteriorDDPM("rank1", 3.5)]


def gate(model, dataset, tag):
    import torch
    from PIL import Image
    rows = model.rows(dataset, "eval")[:2]
    caps, z = model.captions(dataset, rows), model.initial_noise(rows)
    L = model.spec.steps
    rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm())     # noqa: E731
    failed = []

    def check(name, ok, msg):
        print(f"{'PASS' if ok else 'FAIL'} {name}: {msg}", flush=True)
        failed.append(name) if not ok else None

    ref = model.generate(caps, z, np.full(L, 4.5), fp32_state=True)
    ours = model.generate_loop(caps, z, w=4.5)
    check("G1 Euler copy = pipeline (fp32 state)", torch.equal(ours, ref) or rel(ours, ref) < 1e-3,
          f"bitwise {torch.equal(ours, ref)}, relative L2 {rel(ours, ref):.1e}")
    tiles = {"pub-f32 (pipeline)": model.decode(ref)}
    draws = model.ddpm_noise(rows)
    for s in GATE_ARMS:
        x = model.generate_loop(caps, z, s, draws=draws)
        ok = bool(torch.isfinite(x).all()) and not torch.equal(x, ref)
        check(f"G2 {s.name()}", ok, f"finite {bool(torch.isfinite(x).all())}, latent std {float(x.std()):.3f} "
                                    f"(pipeline {float(ref.std()):.3f}), relative L2 to it {rel(x, ref):.2f}")
        tiles[s.name()] = model.decode(x)
    sheet = np.concatenate([np.concatenate(list(t), axis=1) for t in tiles.values()], axis=0)
    out = runs.run_dir(model.spec.key, dataset, tag) / "ddpm-gate.png"
    Image.fromarray(sheet).resize((sheet.shape[1] // 2, sheet.shape[0] // 2)).save(out)
    print("contact sheet rows, top to bottom: " + " | ".join(tiles) + f" -> {out}")
    if failed:
        raise SystemExit(f"FAILED: {', '.join(failed)}")
    print("all gates pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="sd35m")
    ap.add_argument("--dataset")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--shape", choices=SHAPES, default="iso")
    ap.add_argument("--kappa", type=float, default=3.5, help="spread scale: w - 1 adds what guidance at w adds")
    ap.add_argument("--mean", choices=MEANS, default="d1", help="d1, or dw: the guided prediction at --w")
    ap.add_argument("--w", type=float, default=4.5, help="the weight of --mean dw")
    ap.add_argument("--arm", help="output folder name (default from the settings)")
    ap.add_argument("--n", type=int, help="only the first N evaluation rows")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--tag")
    a = ap.parse_args()
    from vmg.models import get
    model = get(a.model)
    assert hasattr(model, "generate_loop"), f"{a.model}: the posterior-DDPM sampler is written for the SD 3 family"
    dataset = a.dataset or model.spec.default_dataset
    model.load()
    if a.gate:
        return gate(model, dataset, a.tag)
    s = PosteriorDDPM(a.shape, a.kappa if a.shape != "none" else 0.0, a.mean, a.w)
    rows = model.rows(dataset, "eval")
    rows = rows[: a.n] if a.n else rows
    rows = rows[len(rows) * a.shard // a.shards: len(rows) * (a.shard + 1) // a.shards]
    arm = a.arm or s.name()
    out = runs.samples_dir(a.model, dataset, arm, a.tag)
    (out / "ddpm.json").write_text(json.dumps(dict(**s.info(), nfe=2 * model.spec.steps, git=cli.git_sha(),
                                                   streams=list(model.DDPM_STREAMS))))
    print(f"{model.spec.name}: arm {arm}, {len(rows)} rows -> {out}", flush=True)
    model.sample_ddpm(dataset, rows, s, out)
    print("done")


if __name__ == "__main__":
    main()
