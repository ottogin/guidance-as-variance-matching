#!/usr/bin/env python
"""Generate a model's evaluation set with its released sampler, at the published or a variance-matched weight.

    python sample.py --model toy --weights runs/toy/mixture/w.json --kind schedule
    python sample.py --model sd35m --published --n 16                         # the released w = 4.5
    python sample.py --model sd35m --w 3.99 --shard 0 --shards 32             # a constant
    python sample.py --model sd35m --weights runs/sd35m/coco/w.json --kind schedule
    python sample.py --model sd35m --vm schedule                              # the same, from this pair's w.json
    python sample.py --model sd35m --vm global --match cov --pool l2          # another fit of that w.json
    python sample.py --guidance cfg --model stabilityai/stable-diffusion-3.5-large --vm selected   # compute_w's pick
    python sample.py --model sd35m --sched interval --base vm                 # a published schedule at our weight
    python sample.py --model sd35m --isonoise          # no guidance, isotropic noise of our schedule's variance
    python sample.py --model sd35m --shapenoise        # the same variance, shaped like Cov(x0)

Other papers' rules (vmg/baselines.py; SD 3 family only), here with the float32 latent state the paper's SD 3.5
runs used:

    python sample.py --model sd35m --mix simca-sf --fp32-state             # a fit of reproduce/baselines/fit_simca.py
    python sample.py --model sd35m --stepag 0.5 --base vm-refit --fp32-state   # Step AG, our weight refit
    python sample.py --model sd35m --oec 0.9 --base vm-schedule --fp32-state   # CFG-OEC on our schedule

Every arm uses the same prompts (or conditioning) and the same initial noise per row, so arms are paired
row by row. Outputs go to runs/<model>/<dataset>/samples/<arm>/, and rows already written are skipped, so a
preempted job resumes where it stopped.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from vmg import baselines as B
from vmg import guidance as G
from vmg import schedules as S
from vmg import runs
from vmg import solve, sources
from vmg.cli import DEFAULT_N
from vmg.models import REGISTRY

BASE_SHORT = {"pub": "pub", "vm": "vm", "vm-schedule": "vms", "vm-refit": "vmr"}


def base_weights(a, model, wjson, levels):
    """The per-level weights a baseline rule runs on (--base), and the arm name's short label for them."""
    L = len(levels["sigma"])
    if a.base == "pub":
        return np.full(L, float(model.spec.published_w)), "pub"
    if a.base == "vm":
        return G.from_json(wjson, "global", levels), "vm"
    if a.base == "vm-schedule":
        return G.from_json(wjson, "schedule", levels), "vms"
    assert a.base == "vm-refit" and a.stepag is not None, "--base vm-refit is our constant refit for --stepag"
    j = json.load(open(wjson))
    assert list(j["levels"]) == list(range(L)), "vm-refit needs a fit on every sampler level"
    keep = np.arange(L) < B.k_stop(a.stepag, L)
    w, ok = B.refit_pruned(*(j["coefficients"][k] for k in "abc"), keep, tuple(j["clip"]))
    assert ok or a.tag == "smoke", f"no admissible refit weight with guidance on {keep.sum()} of {L} levels"
    print(f"vm-refit: constant {w:.4f} on the first {keep.sum()} levels (all levels: {j['global']['w']:.4f})")
    return np.full(L, w if np.isfinite(w) else 2.0), "vmr"


def fit_suffix(a, kind):
    """The arm name's part for a fit other than the paper's (trace match; a global weight pooled by its mean)."""
    if kind == "selected":
        return ""
    parts = ([] if a.match == "trace" else [a.match]) + ([a.pool] if kind == "global" and a.pool != "mean" else [])
    return "".join(f"-{p}" for p in parts)


def mix_fit(name, model_key, wfrom, wtag):
    """A baseline fit by path, or by name in this pair's run folder (e.g. simca-sf -> <run>/simca-sf.json)."""
    p = Path(name)
    return p if p.suffix == ".json" and p.exists() else runs.run_dir(model_key, wfrom, wtag) / f"{name}.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True,
                    help=f"a preset (one of: {', '.join(REGISTRY)}), or a checkpoint with --guidance as in compute_w.py")
    ap.add_argument("--guidance", choices=sources.KINDS, default="auto",
                    help="as compute_w.py: how D0 is made, for a non-preset model (default: %(default)s)")
    ap.add_argument("--reference", help="as compute_w.py: D0 of a cfg-pair / autoguidance pair")
    ap.add_argument("--steps", type=int, help="as compute_w.py")
    ap.add_argument("--published-w", type=float, help="as compute_w.py")
    ap.add_argument("--size", type=int, help="as compute_w.py")
    ap.add_argument("--dataset")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--published", action="store_true", help="the model's released guidance")
    g.add_argument("--w", type=float, help="a constant weight")
    g.add_argument("--weights", help="a compute_w.py result (w.json)")
    g.add_argument("--vm", choices=("global", "schedule", "selected"),
                   help="this pair's own compute_w.py result: its schedule, a global weight (--match, --pool pick "
                        "which), or the fit compute_w.py selected")
    g.add_argument("--sched", choices=S.KINDS, help="a published schedule (vmg/schedules.py), base weight --base")
    g.add_argument("--isonoise", nargs="?", const="vm", metavar="W_JSON",
                   help="the noise control: no guidance (w = 1), plus isotropic Gaussian noise on the clean prediction "
                        "at every level, sized so that it meets the second-moment target our weights meet "
                        "(vmg.guidance.matched_noise_std; this pair's w.json, or the compute_w.py result given)")
    g.add_argument("--shapenoise", nargs="?", const="vm", metavar="W_JSON",
                   help="as --isonoise, with the noise shaped like the data: covariance proportional to Cov(x0) of "
                        "the calibration rows, the same variance per level")
    g.add_argument("--mix", metavar="FIT",
                   help="a baseline fit (reproduce/baselines/fit_simca.py): per-level w (any value, below 1 too) or "
                        "free alpha / beta; a path, or a name in this pair's run folder, e.g. simca-sf")
    g.add_argument("--stepag", type=float, metavar="P",
                   help="Step AG: guidance at the --base weights on the first fraction P of the levels, the "
                        "conditional prediction alone after")
    g.add_argument("--oec", type=float, metavar="TAU", help="CFG-OEC (Dynamic) at threshold TAU on the --base weights")
    ap.add_argument("--base", choices=tuple(BASE_SHORT), default="pub",
                    help="with --sched: the published weight, or this pair's global variance-matched weight (vm); with "
                         "--stepag / --oec also our schedule (vm-schedule) or, for --stepag, our constant refit with "
                         "guidance off on the pruned levels (vm-refit)")
    ap.add_argument("--datamean", action="store_true",
                    help="our weights from the --datamean fit (w-datamean.json), the second-moment variant about the "
                         "data mean, instead of the default tr Cov fit (w.json)")
    ap.add_argument("--fp32-state", action="store_true",
                    help="keep the latent in float32 between steps, as the paper's SD 3.5 runs did (arm suffix -f32)")
    ap.add_argument("--kind", choices=("global", "schedule", "selected"), default="schedule", help="with --weights")
    ap.add_argument("--match", choices=solve.MATCHES, default="trace", help="with --vm / --weights: which matching "
                                                                          "statistic's fit (default trace)")
    ap.add_argument("--pool", choices=solve.POOLS, default="mean", help="with a global weight: its pooling (default mean)")
    ap.add_argument("--arm", help="output folder name (default from the weight)")
    ap.add_argument("--n", type=int, help="only the first N evaluation rows")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--tag")
    ap.add_argument("--weights-from", help="--vm / --sched --base vm: the dataset whose fitted w.json to use "
                                           "(default: --dataset; e.g. kitml for MDM's MultiModality rows kitml-mm)")
    ap.add_argument("--weights-tag", help="the tag of that fit (default: smoke with --tag smoke, where the whole "
                                          "pipeline runs under one tag; otherwise none, e.g. for an evaluation subset)")
    a = ap.parse_args()

    model = sources.resolve(a)
    a.model = model.spec.key                     # the run folder's name (a repo id or a pickle path is not one)
    dataset = a.dataset or model.spec.default_dataset
    wfrom = a.weights_from or dataset
    wtag = a.weights_tag if a.weights_tag is not None else ("smoke" if a.tag == "smoke" else None)
    if a.weights_tag is None and wtag is None and not runs.w_json(a.model, wfrom, None).exists() and \
            runs.w_json(a.model, wfrom, f"n{DEFAULT_N}").exists():
        wtag = f"n{DEFAULT_N}"                   # compute_w.py's default: the first 1,024 calibration rows
    rows = model.rows(dataset, "eval")
    rows = rows[: a.n] if a.n else rows
    rows = rows[len(rows) * a.shard // a.shards: len(rows) * (a.shard + 1) // a.shards]
    model.load()
    noise_std = None
    noise_src = a.isonoise or a.shapenoise
    shape = "data" if a.shapenoise else "iso"
    rule = "datamean" if a.datamean else solve.DEFAULT
    wjson = runs.w_json(a.model, wfrom, wtag, rule)
    ours = bool(a.vm) or (a.base != "pub" and (a.sched or a.stepag is not None or a.oec is not None))
    if (ours or noise_src == "vm") and wjson.exists():
        # a w.json fitted with the other rule (data-mean or tr Cov) is refused rather than sampled under this
        # rule's name
        got = json.load(open(wjson)).get("rule")
        assert got == rule, (f"{wjson} was fitted with rule {got!r}, not {rule!r}: refit it with compute_w.py "
                             f"--stage fit{' --datamean' if a.datamean else ''}")
    guide, info = None, {}
    assert a.base in ("pub", "vm") or a.stepag is not None or a.oec is not None, f"--base {a.base} needs --stepag/--oec"
    if (a.mix or a.stepag is not None or a.oec is not None or a.fp32_state) and not hasattr(model, "guide_hook"):
        raise SystemExit(f"{a.model}: --mix / --stepag / --oec / --fp32-state are implemented for the SD 3 family only")
    if a.published:
        w, arm = None, "pub"
    elif a.mix:
        path = mix_fit(a.mix, a.model, wfrom, wtag)
        j = json.load(open(path))
        if "w" in j:
            w = np.asarray(j["w"], dtype=np.float64)
            guide = B.Mix(w)
        else:
            al, be = (np.asarray(j[k], dtype=np.float64) for k in ("alpha", "beta"))
            assert np.isfinite(al).all() and np.isfinite(be).all(), f"{path}: non-finite weights"
            w = al / (al + be)                   # alpha v_c + beta v_u = (alpha + beta) [v_u + w (v_c - v_u)]
            guide = B.Free2(al, be)
            info.update(alpha=[float(x) for x in al], beta=[float(x) for x in be],
                        note="w = alpha / (alpha + beta), the CFG weight of the same direction; scale alpha + beta")
        arm, info["method"], info["fit"] = j["method"], j["method"], str(path)
    elif a.stepag is not None:
        bw, short = base_weights(a, model, wjson, model.levels())
        ks = B.k_stop(a.stepag, len(bw))
        guide = B.StepAG(bw, ks)
        w = np.where(np.arange(len(bw)) < ks, bw, 1.0)
        arm = f"stepag{round(100 * a.stepag)}-{short}"
        info.update(method="stepag", p=a.stepag, k_stop=ks, base=a.base)
    elif a.oec is not None:
        w, short = base_weights(a, model, wjson, model.levels())
        guide = B.OEC(w, a.oec)
        arm = f"oec{round(100 * a.oec)}-{short}"
        info.update(method="oec", tau=a.oec, base=a.base)
    elif noise_src:
        path = wjson if noise_src == "vm" else noise_src
        noise_std = G.matched_noise_std(path, model.levels())
        arm = "shapenoise" if a.shapenoise else "isonoise"
        w, arm = 1.0, arm + ("-datamean" if str(path).endswith("-datamean.json") else "")
    elif a.w is not None:
        w, arm = a.w, f"w{a.w:g}"
    elif a.sched:
        assert a.base in ("pub", "vm"), "--sched takes --base pub or vm"
        base = model.spec.published_w if a.base == "pub" else G.from_json(wjson, "global", model.levels())[0]
        w, arm = S.published(a.sched, model.levels(), base), f"{'pub' if a.base == 'pub' else 'gw'}-{a.sched}"
    elif a.vm:
        w, arm = G.from_json(wjson, a.vm, model.levels(), a.match, a.pool), f"vm-{a.vm}" + fit_suffix(a, a.vm)
    else:
        w = G.from_json(a.weights, a.kind, model.levels(), a.match, a.pool)
        arm = f"vm-{a.kind}" + fit_suffix(a, a.kind) + ("-datamean" if a.weights.endswith("-datamean.json") else "")
    arm += ("-datamean" if a.datamean and ours else "") + ("-f32" if a.fp32_state else "")
    if w is not None and not np.isfinite(np.asarray(w, dtype=float)).all():
        # a fit with no admissible root has no weight: an error in a real run; a smoke fit (a few rows) often has
        # none, and smoke checks the plumbing, so it guides at 2 there instead
        assert a.tag == "smoke", f"{a.model}/{wfrom}: the fitted weight is not finite here; refit on more rows"
        print(f"smoke: the {wtag or ''} fit of {a.model}/{wfrom} has no admissible weight at some levels; using w = 2")
        w = np.where(np.isfinite(np.asarray(w, dtype=float)), w, 2.0)
        w = float(w) if np.ndim(w) == 0 else w
    if guide is not None and hasattr(guide, "w"):        # the same fallback for the weights a guide runs on
        guide.w = np.where(np.isfinite(guide.w), guide.w, 2.0)
    out = runs.samples_dir(a.model, dataset, a.arm or arm, a.tag)
    ws = G.weights(model.spec.published_w if w is None else w, model.levels()) if w is not None or \
        np.isfinite(model.spec.published_w) else None
    L = len(model.levels()["sigma"])
    info.update(nfe=guide.nfe(L) if guide is not None else 2 * L, fp32_state=bool(a.fp32_state))
    if ws is not None:                       # the weight each sampler level used, for evaluate.py's metrics
        (out / "weights.json").write_text(json.dumps(dict(w=[float(x) for x in ws], w_mean=float(np.mean(ws)),
                                                          **info)))
    if noise_std is not None:
        (out / "noise.json").write_text(json.dumps(dict(source=str(path), shape=shape,
                                                        std=[float(x) for x in noise_std])))
        print(f"{shape} noise: std {noise_std.min():.3g} to {noise_std.max():.3g} per coordinate, "
              f"{int((noise_std == 0).sum())} of {len(noise_std)} levels without noise")
    print(f"{model.spec.name}: arm {a.arm or arm}, {len(rows)} rows -> {out}")
    kw = {} if noise_std is None else dict(noise_std=noise_std, noise_shape=shape, noise_dataset=wfrom)
    if guide is not None or a.fp32_state:
        kw.update(guide=guide, fp32_state=bool(a.fp32_state))
    model.sample(dataset, rows, w, out, **kw)
    if isinstance(guide, B.OEC) and guide.calls:     # how often the correction fired, per shard (merged on resume)
        p = out / f"oec-{a.shard:03d}-of-{a.shards:03d}.json"
        st = guide.stats()
        if p.exists():
            old = json.load(open(p))
            st["fired"] += old["fired"]
            st["calls"] += old["calls"]
            st["rate"] = st["fired"] / st["calls"]
        p.write_text(json.dumps(st))
        print(f"OEC: the correction fired on {st['fired']} of {st['calls']} (row, level) pairs ({st['rate']:.1%})")
    print("done")


if __name__ == "__main__":
    main()
