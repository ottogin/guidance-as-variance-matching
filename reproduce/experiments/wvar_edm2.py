#!/usr/bin/env python
"""fig:wvar for an EDM2 pair: the variance of the guided prediction D_w(x_t, t) along EDM's own Heun trajectories.

    python reproduce/experiments/wvar_edm2.py run --model edm2-64-cfg-s-xsu-fid --arm vm --shard 0 --shards 4  # GPU
    python reproduce/experiments/wvar_edm2.py summary --model edm2-64-cfg-s-xsu-fid                         # CPU

The SD 3.5 Medium and AudioLDM 2 panels of fig:wvar, for the class-conditional EDM2 pairs of vmg/models/edm2.py.
Arms:
    w<value>     a constant weight, e.g. w1 (unguided), w2.5
    pub          the pair's published weight
    vm           the variance-matched schedule, compute_w.py's runs/<model>/<dataset>/w.json (run that first)
    vm-global    the variance-matched constant from the same file

Rows are generate_images.py seeds 0..n-1: seed r fixes both the initial noise and the class, so every arm of a
pair sees the same noise and the same classes. At every step i the guided prediction D_w(x_i, t_i) at the state
the step starts from is accumulated per coordinate (sum and sum of squares, float64), and so is the finished
sample, in the networks' own data space (ImageNet-64: pixels / 127.5 - 1). The figure's Var is the
across-sample variance per coordinate, averaged over coordinates, as in the SD 3.5 / AudioLDM 2 panels. The
sampler is the adapter's own Heun loop, which smoke() checks bitwise against edm2/generate_images.edm_sampler.

Outputs $VMG_RUNS/<model>/<dataset>/wvar/<arm>/shard-<k>-of-<K>.npz; `summary` merges them into
$VMG_RUNS/<model>/<dataset>/wvar/wvar.json (with the calibration data's own variance, var_x0, for reference).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

THUMBS = 64          # finished samples kept (uint8) from shard 0, for a look at each arm


TAG = None           # --tag: keep a smoke or variant run apart from the default one


def out_dir(model_key, dataset, arm=None):
    from vmg import runs
    d = runs.run_dir(model_key, dataset, TAG) / "wvar"
    return d if arm is None else d / arm


def arm_weights(model, model_key, dataset, arm):
    """One weight per sampler level for an arm name."""
    from vmg import guidance as G, runs
    lv = model.levels()
    if arm == "pub":
        pub = model.spec.published_w
        assert np.isfinite(pub), f"{model_key} has no published weight"
        return G.weights(pub, lv)
    if arm in ("vm", "vm-global"):
        path = runs.w_json(model_key, dataset)
        assert path.exists(), f"{path} missing: run compute_w.py --model {model_key} first"
        return G.from_json(path, "schedule" if arm == "vm" else "global", lv)
    assert arm.startswith("w"), f"unknown arm {arm!r}"
    return G.weights(float(arm[1:]), lv)


def run(a):
    import torch
    from vmg.models import base
    model = base.get(a.model)
    dataset = model.spec.default_dataset
    model.load()
    ws = arm_weights(model, a.model, dataset, a.arm)
    rows = list(range(a.n))
    lo, hi = len(rows) * a.shard // a.shards, len(rows) * (a.shard + 1) // a.shards
    dst = out_dir(a.model, dataset, a.arm) / f"shard-{a.shard:03d}-of-{a.shards:03d}.npz"
    if dst.exists() and not a.redo:
        print(f"{dst} exists")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    S = model.steps
    s1 = s2 = None
    thumbs = []
    t0 = time.time()
    print(f"{model.spec.name}, arm {a.arm}: seeds {lo}..{hi - 1}, w per step {np.round(ws, 3).tolist()}", flush=True)
    for b0 in range(lo, hi, a.batch):
        rb = rows[b0:min(b0 + a.batch, hi)]
        noise, labels = model.initial(rb)
        if s1 is None:
            s1 = torch.zeros((S + 1,) + tuple(noise.shape[1:]), dtype=torch.float64, device="cuda")
            s2 = torch.zeros_like(s1)

        def record(i, d):
            d = d.double()
            s1[i] += d.sum(0)
            s2[i] += (d * d).sum(0)

        x = model.heun(noise, labels, ws, record=record)
        record(S, x)
        if a.shard == 0 and len(thumbs) < THUMBS:
            thumbs.append(model.decode(x[:THUMBS - len(thumbs)]).permute(0, 2, 3, 1).cpu().numpy())
        done = rb[-1] + 1 - lo
        el = time.time() - t0
        print(f"  {done}/{hi - lo} seeds  {el / 60:.1f} min  eta {(hi - lo - done) * el / done / 60:.1f} min",
              flush=True)
    meta = dict(model=a.model, dataset=dataset, arm=a.arm, w=[float(v) for v in ws], seeds=[lo, hi], steps=S,
                sigma=[float(s) for s in model.t_steps()[:-1]], device=torch.cuda.get_device_name(),
                seconds=time.time() - t0)
    tmp = dst.with_name(f".{dst.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, s1=s1.cpu().numpy(), s2=s2.cpu().numpy(), n=np.asarray(hi - lo), meta=json.dumps(meta),
             thumbs=np.concatenate(thumbs) if thumbs else np.zeros((0,), np.uint8))
    os.replace(tmp, dst)
    print(f"wrote {dst} in {(time.time() - t0) / 60:.1f} min")


def load_arm(model_key, dataset, arm):
    """Merged shards of one arm: (var [S + 1] coordinate-averaged across-sample variance, mean_sq [S + 1] average
    squared coordinate, meta, thumbs) or None when no shard is there. Asserts the shards tile the seeds."""
    paths = sorted(out_dir(model_key, dataset, arm).glob("shard-*-of-*.npz"))
    if not paths:
        return None
    K = int(paths[0].stem.rsplit("-of-", 1)[1])
    assert len(paths) == K, f"{arm}: {len(paths)} of {K} shards"
    s1 = s2 = 0
    n, spans, thumbs, meta = 0, [], None, None
    for p in paths:
        z = np.load(p)
        s1, s2, n = s1 + z["s1"], s2 + z["s2"], n + int(z["n"])
        meta = json.loads(str(z["meta"]))
        spans.append(tuple(meta["seeds"]))
        if z["thumbs"].size:
            thumbs = z["thumbs"]
    spans.sort()
    assert spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:])), spans
    S1 = s1.reshape(len(s1), -1)
    S2 = s2.reshape(len(s2), -1)
    var = ((S2 - S1 ** 2 / n) / (n - 1)).mean(1)
    meta.update(n=n, seeds=[0, spans[-1][1]])
    return var, (S2 / n).mean(1), meta, thumbs


def var_x0(model_key, dataset):
    """Coordinate-averaged variance of the calibration data (the target the matching rule aims at)."""
    from vmg import moments as M, runs
    paths = runs.moments_shards(model_key, dataset)
    if not paths:
        return None
    sums, meta = M.load(paths)
    return float(M.reduce(sums, meta).var_x0)


def summary(a):
    from vmg.models import base
    model = base.get(a.model)
    dataset = model.spec.default_dataset
    root = out_dir(a.model, dataset)
    res = dict(model=a.model, name=model.spec.name, published_w=model.spec.published_w, var_x0=var_x0(a.model, dataset),
               arms={})
    strips = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        got = load_arm(a.model, dataset, d.name)
        if got is None:
            continue
        var, msq, meta, th = got
        if th is not None and len(th) >= a.thumbs:
            strips.append((d.name, np.concatenate(list(th[:a.thumbs]), 1)))
        res["arms"][d.name] = dict(var=var.tolist(), mean_sq=msq.tolist(), w=meta["w"], n=meta["n"],
                                   sigma=meta["sigma"])
        print(f"{d.name:>10} (n={meta['n']}): var step 1 {var[0]:.4f}, max {var.max():.4f} at step "
              f"{int(var.argmax()) + 1}, final {var[-1]:.4f}")
    if res["var_x0"] is not None:
        print(f"{'data':>10}: var_x0 {res['var_x0']:.4f}")
    (root / "wvar.json").write_text(json.dumps(res, indent=1))
    print(f"wrote {root / 'wvar.json'}")
    if strips:
        from PIL import Image, ImageDraw
        lab = 70
        h, w = strips[0][1].shape[:2]
        im = Image.new("RGB", (lab + w, h * len(strips)), "white")
        dr = ImageDraw.Draw(im)
        for r, (name, s) in enumerate(strips):
            im.paste(Image.fromarray(s), (lab, r * h))
            dr.text((4, r * h + h // 2 - 6), name, fill="black")
        im.save(root / "thumbs.png")
        print(f"wrote {root / 'thumbs.png'} (seeds 0..{a.thumbs - 1}, one row per arm)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["run", "summary"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", help="run: w<value>, pub, vm or vm-global")
    ap.add_argument("--n", type=int, default=10000, help="seeds 0..n-1")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--thumbs", type=int, default=12, help="summary: samples per arm in thumbs.png")
    ap.add_argument("--tag", help="write under runs/<model>/<dataset>-<tag>/wvar (the vm arm still reads the "
                                  "default w.json)")
    a = ap.parse_args()
    global TAG
    TAG = a.tag
    {"run": run, "summary": summary}[a.stage](a)


if __name__ == "__main__":
    main()
