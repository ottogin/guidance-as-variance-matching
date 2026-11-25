#!/usr/bin/env python
"""The MSE-optimal guidance weight w_MSE(t) of a model on a calibration dataset (paper eq. wmse).

    python compute_wmse.py --model toy
    python compute_wmse.py --model sd35m --dataset coco      # reuses compute_w's moments if they exist

w_MSE(t) = E<x0 - D0, D1 - D0> / E||D1 - D0||^2 minimises the guided prediction's denoising error at each
noise level. It comes from the same moments pass as compute_w.py, so running both costs one pass. Writes
runs/<model>/<dataset>/wmse.json: w_MSE per level with row-bootstrap intervals, its median over the levels
(the dots of fig:wmse-published), and the denoising error E||D_w - x0||^2 / d at w = 1, w_MSE and the published
weight, each with its bootstrap interval.
"""
import numpy as np

from vmg import cli, moments as M, runs, solve


def main():
    ap = cli.parser(__doc__.split("\n")[0])
    a = cli.finish_args(ap.parse_args())
    model, dataset = cli.model_and_dataset(a)
    if a.stage in ("all", "moments"):
        cli.moments_stage(a, model, dataset)
    if a.stage == "moments":
        return
    sums, meta = cli.load_moments(a, dataset)
    m = M.reduce(sums, meta)
    w = solve.w_mse(m)
    pub = model.spec.published_w
    reps = list(M.bootstrap(sums, meta, a.boot)) if a.boot else []
    boot = np.array([solve.w_mse(b) for b in reps]) if reps else None
    band = lambda v: None if v is None else np.nanpercentile(v, [2.5, 97.5], axis=0)   # noqa: E731  [2, S]
    med_boot = np.nanmedian(boot, 1) if boot is not None else None
    # the denoising error at w = 1, at w_MSE (refit inside every replicate) and at the published weight
    err = lambda f: (f(m), band(np.array([f(b) for b in reps])) if reps else None)     # noqa: E731
    e1, e1_ci = err(lambda x: solve.mse(x, 1.0))
    ew, ew_ci = err(lambda x: solve.mse(x, solve.w_mse(x)))
    ep, ep_ci = err(lambda x: solve.mse(x, pub)) if np.isfinite(pub) else (None, None)
    out = cli.header(model, dataset, meta)
    out.update(n=m.n, boot=a.boot, w_mse=w, ci_lo=None if boot is None else band(boot)[0],
               ci_hi=None if boot is None else band(boot)[1], median=float(np.median(w)),
               median_ci=None if boot is None else [float(np.percentile(med_boot, 2.5)),
                                                     float(np.percentile(med_boot, 97.5))],
               mse_w1=e1, mse_wmse=ew, mse_published=ep, mse_w1_ci=e1_ci, mse_wmse_ci=ew_ci, mse_published_ci=ep_ci)
    path = runs.wmse_json(a.key, dataset, a.tag)
    cli.dump(path, out)

    print(f"{model.spec.name} on {dataset}: n={m.n}, d={m.d}, {len(m.sigma)} levels")
    for i, s in enumerate(m.sigma):
        ci = "" if boot is None else f"  [{out['ci_lo'][i]:.4f}, {out['ci_hi'][i]:.4f}]"
        print(f"  level {meta['levels'][i]:4d}  sigma {s:10.4f}  w_MSE {w[i]:.4f}{ci}")
    print(f"median w_MSE = {out['median']:.4f}; published {pub}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
