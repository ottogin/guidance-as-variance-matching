#!/usr/bin/env python
"""The variance-matched guidance weight w* of a guided diffusion model (paper Sec. 4), without sampling.

The guided prediction is D_w = D0 + w (D1 - D0). The script evaluates D1 and D0 on forward-noised calibration
examples at every noise level of the model's sampler and chooses w so that the guided predictions spread over
the inputs as much as the data do (eq. varmatch_t).

Three kinds of guided pair:
    python compute_w.py --model sd35m                                   # a preset of the paper (vmg.models.REGISTRY)
    python compute_w.py --guidance cfg --model stabilityai/stable-diffusion-3.5-medium --dataset coco
    python compute_w.py --guidance cfg-pair --model edm2-img64-s-1073741-0.045 \\
        --reference cond_free_uncond.pkl --dataset imagenet64           # CFG with two networks (EDM2)
    python compute_w.py --guidance autoguidance --model edm2-img64-s-1073741-0.045 \\
        --reference edm2-img64-xs-0134217-0.110 --dataset imagenet64    # autoguidance
    python compute_w.py --model mypkg.mymodule:MyAdapter --dataset folder:/data/my_images   # your own adapter

What to match (--match) and the form of w (--mode, --pool):
    trace   tr Cov(D_w) = tr Cov(x0) at each level, the paper's rule (default)
    cov     the channel covariance matrix of D_w to the data's, in Frobenius norm (App. D.1)
    diag    every coordinate's variance of D_w to the data's, in least squares
    --mode schedule   one w per sampler level, w(t) (default; eq. wstar)
    --mode global     one constant w, pooled over the levels (App. D.2):
        --pool mean   the level-averaged residual vanishes (default; the paper's eq. wconstant)
        --pool l1     minimise the mean |residual| over levels
        --pool l2     minimise the mean squared residual over levels

Every fit the moments allow (each match the pass collected, the schedule and all three pools) is written to
runs/<model>/<dataset>/w.json; the flags pick the one printed, recorded as "selected" (sample.py --vm selected).
--datamean matches the second moment about the data mean instead of the covariance (writes w-datamean.json).
--n sets the number of calibration examples (default 1024; 'all' for every one). A large pass splits into
shards: --stage moments --shard k --shards K on K GPUs, then --stage fit on a CPU.
"""
import numpy as np

from vmg import cli, extras as X, moments as M, runs, solve


def main():
    ap = cli.parser(__doc__.split("\n")[0])
    f = ap.add_argument_group("the fit")
    f.add_argument("--match", choices=solve.MATCHES + ("all",), default="trace",
                   help="what D_w must match: trace, cov or diag; 'all' collects the sums of every one and reports "
                        "the trace (default: %(default)s)")
    f.add_argument("--mode", choices=solve.MODES, default="schedule",
                   help="one weight per level or one constant (default: %(default)s)")
    f.add_argument("--pool", choices=solve.POOLS, default="mean",
                   help="how a global weight pools the levels (default: %(default)s)")
    f.add_argument("--global", dest="mode", action="store_const", const="global", help="same as --mode global")
    f.add_argument("--schedule", dest="mode", action="store_const", const="schedule", help="same as --mode schedule")
    f.add_argument("--datamean", action="store_true",
                   help="the second moment about the data mean instead of the covariance (writes w-datamean.json; "
                        "default: the covariance)")
    a = cli.finish_args(ap.parse_args())
    model, dataset = cli.model_and_dataset(a)
    need = [e for mt in (solve.MATCHES if a.match == "all" else (a.match,)) for e in X.FOR_MATCH[mt]]
    if a.stage in ("all", "moments"):
        cli.moments_stage(a, model, dataset, extras=need)
    if a.stage == "moments":
        return
    rule = "datamean" if a.datamean else solve.DEFAULT
    sums, meta = cli.load_moments(a, dataset)
    m = M.reduce(sums, meta)
    res = M.residuals(sums, meta, rule)
    match = "trace" if a.match == "all" else a.match
    assert match == "trace" or match in res, f"the moments hold no {match} sums: rerun the moments stage with --redo"
    fits = solve.fit_all(m, res, rule)
    sch = solve.fit_schedule(m, rule)
    boot = list(M.bootstrap(sums, meta, a.boot)) if a.boot else []
    if boot:                                     # standard errors of the trace fits (cov / diag sums are pooled)
        bf = [solve.fit_all(b, None, rule)["trace"] for b in boot]
        fits["trace"]["schedule"]["se"] = np.nanstd([b["schedule"]["w"] for b in bf], 0)
        for p in solve.POOLS:
            fits["trace"]["global"][p]["se"] = float(np.std([b["global"][p]["w"] for b in bf]))
    se_raw = np.nanstd([solve.fit_schedule(b, rule)["raw"] for b in boot], 0) if boot else None
    tg = fits["trace"]["global"]["mean"]

    out = cli.header(model, dataset, meta)
    sel = fits[match]["schedule"] if a.mode == "schedule" else fits[match]["global"][a.pool]
    out.update(n=m.n, rule=rule, clip=list(solve.CLIP), domain=list(solve.DOMAIN), var_x0=m.var_x0, boot=a.boot,
               guidance=getattr(a, "guidance", None) or "preset",
               # the paper's two fits (trace match): read by vmg.guidance.from_json and the figures
               **{"global": dict(w=tg["w"], admissible=tg["admissible"], se=tg.get("se"))},
               schedule=dict(w=sch["w"], raw=sch["raw"], admissible=sch["admissible"], se_raw=se_raw),
               coefficients=dict(a=sch["a"], b=sch["b"], c=sch["c"]),
               fits=fits, se_note="standard errors are row bootstraps of the trace fits; cov and diag have none",
               selected=dict(match=match, mode=a.mode, pool=a.pool if a.mode == "global" else None, w=sel["w"]))
    path = runs.w_json(a.key, dataset, a.tag, rule)
    cli.dump(path, out)

    pub = model.spec.published_w
    print(f"{model.spec.name} on {dataset}: n={m.n}, d={m.d}, {len(m.sigma)} levels, rule {rule}")
    print(f"fit: --match {match} --mode {a.mode}" + (f" --pool {a.pool}" if a.mode == "global" else "")
          + "   (defaults: --match trace --mode schedule --pool mean)")
    if not sch["admissible"].any():
        print("WARNING: no level has a trace root in [1, 40]; too few calibration rows?")
    if a.mode == "schedule":
        w = np.asarray(sel["w"])
        se = fits[match]["schedule"].get("se")
        for i, s in enumerate(m.sigma):
            note = ""
            if match == "trace":
                note = (f"root {sch['raw'][i]:.3f}" + ("" if se is None else f" +- {se[i]:.3f}")
                        if sch["admissible"][i] else "no admissible root: held")
            print(f"  level {meta['levels'][i]:4d}  sigma {s:10.4f}  w {w[i]:7.3f}  {note}")
    for p in solve.POOLS:
        g = fits[match]["global"][p]
        se = "" if g.get("se") is None else f" +- {g['se']:.3f}"
        star = "  <- selected" if a.mode == "global" and p == a.pool else ""
        flag = "" if g["admissible"] else " (no admissible root: clipped)"
        print(f"global w ({match}, pool {p}) = {g['w']:.4f}{se}{flag}{star}")
    print(f"published {pub}" if np.isfinite(pub) else f"published: {model.spec.published_note or 'none'}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
