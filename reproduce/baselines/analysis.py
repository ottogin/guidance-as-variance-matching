#!/usr/bin/env python
"""What the SD 3.5 Medium calibration data says about the four baselines (CPU, seconds).

    python reproduce/baselines/analysis.py                          # the calibration pass in $VMG_RUNS
    python reproduce/baselines/analysis.py --run $VMG_RUNS/sd35m/coco    # also the GPU fits, when they exist

Inputs: the SD 3.5 Medium calibration pass, $VMG_RUNS/sd35m/coco/moments/ (reproduce/plan.calibrate: its per-level
moments and per-photo products <r_i, r_j> / d), checked against tests/fixtures/sd35m-coco-s40.npz (the stored pass
behind the paper's printed weights, n = 5,705). Without the pass: the fixture and the stored per-photo products
(GRAM below, not shipped). With --run, also simca-sb{1,2}.json (fit_simca.py) and posterior-trace.npz
(posterior_trace.py).

  (a) SIMCA, simulation-free = w_MSE per level (paper eq. wmse), against our schedule, and the simulated fits.
  (b) Step AG: our constant refit with guidance off on the pruned levels, the share of our schedule's guidance
      (sum of w - 1) on those levels, the equivalent SNR threshold, and every arm's second-moment excess.
  (c) Error geometry (CFG-EC/OEC): the cosine between the two branches' errors per level (pooled and per photo),
      and w_MSE when the unconditional error is orthogonalised against the conditional one with the true x0 (the
      ideal CFG-EC removes exactly the cross term): it lies in (0, 1) at every level by construction.
  (d) Posterior trace: E||r_u||^2, E||r_c||^2 and the matching gap c(t) per level, beside the network-only
      Hutchinson estimate of tr Cov[x0 | x_s] when posterior_trace.py has run.

Writes analysis-sd35m.json and three figures (pdf + png) into --out (default $VMG_RUNS/baselines).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from vmg import baselines as B, cli, solve  # noqa: E402

FIX = REPO / "tests/fixtures/sd35m-coco-s40.npz"
GRAM = REPO / "results/domains/mom/coco-sd35m-n5705g-s40.npz"      # the stored pass's per-photo products
PUB = 4.5
WMSE_FIRST = (1.0743, 1.0339, 1.0153)       # the paper's w_MSE at the first three steps (G5)
SB2_COL = "#9b59d0"                          # fourth line colour


def moments(fx):
    return solve.Moments(sigma=fx["sigma"], t=fx["t"], var_x0=float(fx["var_x0"]), xr=fx["xr"], rr=fx["rr"],
                         rr_centred=fx["rr_centred"], n=int(fx["n"]), d=int(fx["d"]))


def inputs():
    """(Moments, per-photo Grams [n, L, 3] in the order (cc, cu, uu), stored): the SD 3.5 Medium calibration pass in
    $VMG_RUNS when it is there, else the stored pass (FIX and GRAM; stored = True)."""
    from vmg import moments as M, runs
    paths = runs.moments_shards("sd35m", "coco")
    if paths:
        sums, meta = M.load(paths)
        G = sums["q_rr"][..., ::-1] / (sums["cnt"][:, None, None] * int(meta["d"]))       # (00, 01, 11) = (uu, uc, cc)
        return M.reduce(sums, meta), G[np.argsort(sums["rows"])], False
    assert GRAM.exists(), f"no SD 3.5 Medium calibration pass in {runs.root()} (reproduce/plan.calibrate), no {GRAM}"
    return moments(np.load(FIX)), np.load(GRAM)["gram_p"], True


def excess(w, a, b, c, var):
    """E||D_w - xbar0||^2 - E||x0 - xbar0||^2 per level, relative to the target (0 = matched)."""
    w = np.asarray(w, dtype=np.float64)
    return (b * w * w - 2 * a * w - c) / var


def load_fit(run, name):
    p = None if run is None else Path(run) / f"{name}.json"
    if p is None or not p.exists():
        return None
    j = json.load(open(p))
    if "w" in j:
        return dict(w=np.asarray(j["w"]), se=np.asarray(j.get("se", np.zeros(len(j["w"])))), raw=j)
    al, be = np.asarray(j["alpha"]), np.asarray(j["beta"])
    return dict(w=al / (al + be), alpha=al, beta=be, scale=al + be, raw=j)


def default_run():
    """$VMG_RUNS/sd35m/coco when it exists, else None (the stored data only)."""
    from vmg import runs
    p = runs.root() / "sd35m" / "coco"
    return p if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=default_run(),
                    help="the GPU run folder of sd35m/coco (fit_simca.py, posterior_trace.py outputs); default "
                         "$VMG_RUNS/sd35m/coco when it exists")
    ap.add_argument("--out", help="default $VMG_RUNS/baselines")
    a = ap.parse_args()
    res = run(a.run, a.out)
    sys.exit(0 if all(res["checks"].values()) else 1)


def run(run_dir=None, out=None):
    """Everything above, written to `out`; returns the results dict (res["checks"]: name -> passed)."""
    from vmg import runs
    a = argparse.Namespace(run=run_dir)
    out = Path(out) if out else runs.root() / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    fx = np.load(FIX)
    m, G, stored = inputs()
    tol = 1e-9 if stored else 1e-2               # a rebuilt pass: other GPUs, the stored pass's rows and noise
    L = len(m.sigma)
    with np.errstate(invalid="ignore"):      # the flow time s of each level: the fixture's "t"; a pass's t is the
        sg = np.asarray(m.sigma, dtype=np.float64)                                          # timestep (x 1000)
        s = m.t if stored else np.where(np.isinf(sg), 1.0, sg / (1 + sg))
    step = np.arange(1, L + 1)
    res, checks = dict(n=m.n, levels=L, s=s, sigma=m.sigma), {}

    # (a) ---------------------------------------------------------------------------------------------------
    sch = solve.fit_schedule(m, "centred")
    ours = sch["w"]
    ours_g, _ = solve.fit_global(m, "centred")
    wmse = solve.w_mse(m)
    checks["schedule = fixture w_centred"] = bool(np.allclose(ours, fx["w_centred"], rtol=tol))
    checks["constant = 4.13408, the fit behind the printed 4.14"] = bool(abs(ours_g - 4.13408) < max(1e-5, tol))
    checks["w_MSE first steps"] = bool(np.allclose(wmse[:3], WMSE_FIRST, atol=max(3e-3, tol)))
    sb1, sb2 = load_fit(a.run, "simca-sb1"), load_fit(a.run, "simca-sb2")
    res["a"] = dict(ours_schedule=ours, ours_constant=ours_g, ours_schedule_mean=float(ours.mean()), w_mse=wmse,
                    w_mse_median=float(np.median(wmse)), w_mse_levels_above_1=int((wmse > 1).sum()),
                    simca_sb1=None if sb1 is None else sb1["w"], simca_sb2_w=None if sb2 is None else sb2["w"],
                    simca_sb2_scale=None if sb2 is None else sb2["scale"])
    if sb1 is not None:
        checks["SIMCA-SB1 level 0 ~ w_MSE level 0 (both see pure noise)"] = bool(
            abs(sb1["w"][0] - wmse[0]) <= max(3 * float(sb1["se"][0]), 0.02))

    # (b) ---------------------------------------------------------------------------------------------------
    ac, bc, cc_ = solve.coefficients(m, "centred")
    var = m.var_x0
    w_all, _ = B.refit_pruned(ac, bc, cc_, np.ones(L, bool))
    checks["refit with every level kept = fit_global"] = bool(abs(w_all - ours_g) < 1e-12)
    step_ag = {}
    for p in (0.3, 0.5, 0.7):
        ks = B.k_stop(p, L)
        keep = np.arange(L) < ks
        wr, ok = B.refit_pruned(ac, bc, cc_, keep)
        step_ag[f"p{round(100 * p)}"] = dict(
            p=p, k_stop=ks, nfe=2 * ks + (L - ks), s_cut=float(s[ks]), snr_lambda=float((1 - s[ks]) / s[ks]),
            refit_w=wr, refit_admissible=ok,
            share_of_our_guidance_pruned=float((ours[~keep] - 1).sum() / (ours - 1).sum()),
            share_of_pub_guidance_pruned=float((~keep).sum() / L))
    res["b"] = step_ag
    arms = {"pub": np.full(L, PUB), "vm-global": np.full(L, ours_g), "vm-schedule": ours, "simca-sf": wmse,
            "w=1": np.ones(L)}
    for p in (0.3, 0.5):
        ks = B.k_stop(p, L)
        cut = np.arange(L) < ks
        arms[f"stepag{round(100 * p)}-pub"] = np.where(cut, PUB, 1.0)
        arms[f"stepag{round(100 * p)}-vms"] = np.where(cut, ours, 1.0)
        arms[f"stepag{round(100 * p)}-vmr"] = np.where(cut, step_ag[f"p{round(100 * p)}"]["refit_w"], 1.0)
    if sb1 is not None:
        arms["simca-sb1"] = sb1["w"]
    res["excess"] = {k: dict(mean=float(excess(w, ac, bc, cc_, var).mean()), per_level=excess(w, ac, bc, cc_, var),
                             w_mean=float(np.mean(w))) for k, w in arms.items()}

    # (c) ---------------------------------------------------------------------------------------------------
    # G [n, L, 3]: (cc, cu, uu), the dump's naming
    first = (lambda g: g) if stored else (lambda g: g.mean(0))        # a rebuilt pass: the 64 photos' means
    checks["gram_p order (cc, cu, uu) vs fixture gram_p64 (uu, uc, cc)"] = bool(
        np.allclose(first(G[:64, :, ::-1]), first(fx["gram_p64"]), rtol=max(1e-6, tol)))
    gcc, guc, guu = G[..., 0], G[..., 1], G[..., 2]
    pooled_wmse = (guu - guc).sum(0) / (gcc - 2 * guc + guu).sum(0)
    checks["w_MSE from per-photo Grams = fixture w_MSE"] = bool(np.allclose(pooled_wmse, wmse, rtol=1e-6))
    cos = guc / np.sqrt(guu * gcc)
    uperp = guu - guc ** 2 / gcc                                # ||r_u - proj_{r_c} r_u||^2 per photo
    w_orth = uperp.sum(0) / (gcc + uperp).sum(0)

    def mse(w, orth):
        return ((w * w * gcc + (1 - w) ** 2 * uperp) if orth else
                (w * w * gcc + 2 * w * (1 - w) * guc + (1 - w) ** 2 * guu)).mean(0)
    res["c"] = dict(cos_pooled=guc.sum(0) / np.sqrt(guu.sum(0) * gcc.sum(0)),
                    cos_photo_q=np.quantile(cos, [0.1, 0.25, 0.5, 0.75, 0.9], axis=0),
                    w_mse=wmse, w_mse_oracle_orth=w_orth,
                    mse_ratio_pub_vs_w1=mse(PUB, False) / mse(1.0, False),
                    mse_ratio_pub_orth_vs_w1=mse(PUB, True) / mse(1.0, False))
    checks["oracle-orthogonalised w_MSE in (0, 1)"] = bool(((w_orth > 0) & (w_orth < 1)).all())

    # (d) ---------------------------------------------------------------------------------------------------
    ad, bd, cd = solve.coefficients(m, "datamean")
    # c - E||r_u||^2 = -2 E<D_u - xbar0, r_u>: zero when D_u is the exact posterior mean (orthogonality principle),
    # so then the gap is the expected posterior trace
    res["d"] = dict(err_u=m.rr[:, 0], err_c=m.rr[:, 2], c_datamean=cd, c_centred=cc_,
                    gap_minus_err_u=cd - m.rr[:, 0])
    pt = None if a.run is None else Path(a.run) / "posterior-trace.npz"
    if pt is not None and pt.exists():
        z = np.load(pt)
        tr = z["tr"].mean(-1)                                  # [rows, levels, branch]: mean over probes
        err = z["err"]
        from scipy.stats import spearmanr
        res["d"]["hutchinson"] = dict(levels=z["levels"], sigma=z["sigma"], n=len(z["rows"]), probes=int(z["probes"]),
                                      h=float(z["h"]), tr_mean=tr.mean(0), tr_se=tr.std(0) / np.sqrt(len(tr)),
                                      err_mean=err.mean(0),
                                      spearman_photo=np.array([[spearmanr(tr[:, j, b], err[:, j, b])[0]
                                                                for b in range(2)] for j in range(tr.shape[1])]))
    res["checks"] = checks
    res["git"] = cli.git_sha()
    cli.dump(out / "analysis-sd35m.json", res)
    figures(out, res, step, sb1, sb2)
    report(res)
    print(f"wrote {out / 'analysis-sd35m.json'} and the figures")
    return res


def report(res):
    a, b, c = res["a"], res["b"], res["c"]
    print("checks:")
    for k, v in res["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'} {k}")
    print(f"(a) ours: constant {a['ours_constant']:.4f}, schedule mean {a['ours_schedule_mean']:.2f} "
          f"({a['ours_schedule'][0]:.2f} at step 1 -> {a['ours_schedule'][-1]:.2f} at step 40)")
    print(f"    SIMCA simulation-free = w_MSE: median {a['w_mse_median']:.3f}, above 1 at "
          f"{a['w_mse_levels_above_1']} of {len(a['w_mse'])} levels, last {a['w_mse'][-1]:.3f}")
    if a["simca_sb1"] is not None:
        w = np.asarray(a["simca_sb1"])
        print(f"    SIMCA simulation-based, CFG form: mean {w.mean():.3f}, range {w.min():.3f}..{w.max():.3f}")
    if a["simca_sb2_w"] is not None:
        w, sc = np.asarray(a["simca_sb2_w"]), np.asarray(a["simca_sb2_scale"])
        print(f"    SIMCA simulation-based, free: equivalent w mean {w.mean():.3f}, scale alpha+beta "
              f"{sc.min():.3f}..{sc.max():.3f}")
    print("(b) Step AG (guidance on the first p of the 40 steps):")
    for k, v in b.items():
        print(f"    p={v['p']}: steps 1-{v['k_stop']} guided (NFE {v['nfe']}), cut at s={v['s_cut']:.3f} "
              f"(SNR alpha/sigma {v['snr_lambda']:.3f}); refit constant {v['refit_w']:.3f}; it removes "
              f"{v['share_of_our_guidance_pruned']:.0%} of our schedule's guidance (sum of w-1)")
    print("    second-moment excess (mean over levels, relative to tr Cov x0; 0 = matched, centred rule):")
    for k, v in res["excess"].items():
        print(f"      {k:14s} mean w {v['w_mean']:6.2f}   excess {v['mean']:+.4f}")
    print(f"(c) error cosine cond vs uncond, pooled: {c['cos_pooled'].min():.3f}..{c['cos_pooled'].max():.3f}; "
          f"per-photo median {np.median(c['cos_photo_q'][2]):.3f}")
    print(f"    w_MSE {np.median(c['w_mse']):.3f} (median) -> with the ideal orthogonalisation "
          f"{np.median(c['w_mse_oracle_orth']):.3f} (median), max {c['w_mse_oracle_orth'].max():.3f}")
    print(f"    denoising error at w=4.5 relative to w=1: plain {np.median(c['mse_ratio_pub_vs_w1']):.2f}x, "
          f"ideal orthogonalisation {np.median(c['mse_ratio_pub_orth_vs_w1']):.2f}x (medians over levels)")
    d = res["d"]
    if "hutchinson" in d:
        h = d["hutchinson"]
        print(f"(d) posterior trace on {h['n']} photos x {h['probes']} probes (h={h['h']}):")
        for j, k in enumerate(h["levels"]):
            print(f"    level {k:2d} sigma {h['sigma'][j]:8.3f}: uncond tr {h['tr_mean'][j, 0]:.4f} vs ||r_u||^2 "
                  f"{h['err_mean'][j, 0]:.4f} (c {d['c_datamean'][k]:.4f}); cond tr {h['tr_mean'][j, 1]:.4f} vs "
                  f"||r_c||^2 {h['err_mean'][j, 1]:.4f}")


def figures(out, res, step, sb1, sb2):
    from reproduce import style as S
    plt = S.rc()
    a, b, c, d = res["a"], res["b"], res["c"], res["d"]

    # weights per level -------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(S.W_COL * 1.6, 2.2))
    S.style(ax)
    lines = [("ours w(σ)", a["ours_schedule"], S.ACC), ("SIMCA sim.-free = w_MSE", a["w_mse"], S.COND)]
    if sb1 is not None:
        lines.append(("SIMCA sim.-based", sb1["w"], S.UNCOND))
    if sb2 is not None:
        lines.append(("SIMCA sim.-based, free (equiv. w)", sb2["w"], SB2_COL))
    ax.axhline(1, color=S.MUTED, lw=0.6, ls="--")
    ax.axhline(PUB, color=S.MUTED, lw=0.6, ls=":")
    ax.text(40.6, PUB * 1.04, "Pub. 4.5", color=S.MUTED, va="bottom", fontsize=6)
    ax.text(40.6, 1.04, "w = 1", color=S.MUTED, va="bottom", fontsize=6)
    for name, w, col in lines:
        ax.plot(step, w, color=col, lw=1.4, label=name)
    for p in ("p50", "p30"):
        k = b[p]["k_stop"]
        ax.axvline(k + 0.5, color=S.GRID, lw=1.0, zorder=0)
        ax.text(k + 0.7, 26, f"Step AG p={b[p]['p']}", color=S.MUTED, fontsize=6, va="top")
    ax.set_yscale("log")
    ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16])
    ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8", "16"])
    ax.set_xlim(0.5, 43)
    ax.set_xlabel("sampler step (1 = pure noise)")
    ax.set_ylabel("guidance weight w")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    S.title(ax, "SD 3.5 Medium: per-step weights of the least-squares fits and ours")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig-baseline-weights.{ext}", dpi=200)
    plt.close(fig)

    # error geometry ----------------------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(S.W_COL * 2, 2.1))
    for ax in (a1, a2):
        S.style(ax)
    q = c["cos_photo_q"]
    a1.fill_between(step, q[1], q[3], color=S.COND, alpha=0.18, lw=0, label="per photo, middle 50%")
    a1.plot(step, q[2], color=S.COND, lw=0.8, ls="--", label="per photo, median")
    a1.plot(step, c["cos_pooled"], color=S.COND, lw=1.4, label="pooled")
    a1.set_xlabel("sampler step")
    a1.set_ylabel("cos(r_c, r_u)")
    a1.legend(frameon=False, loc="lower right")
    S.title(a1, "cosine of the two branches' errors")
    a2.axhline(1, color=S.MUTED, lw=0.6, ls="--")
    a2.plot(step, c["w_mse"], color=S.COND, lw=1.4, label="w_MSE")
    a2.plot(step, c["w_mse_oracle_orth"], color=S.ACC, lw=1.4, label="w_MSE, ideal orthogonalisation")
    a2.set_xlabel("sampler step")
    a2.set_ylabel("MSE-optimal weight")
    a2.legend(frameon=False, loc="center right")
    S.title(a2, "MSE-optimal weight with and without CFG-EC")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig-error-geometry.{ext}", dpi=200)
    plt.close(fig)

    # posterior trace ---------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(S.W_COL * 1.3, 2.2))
    S.style(ax)
    sig = np.asarray(res["sigma"], dtype=float)
    fin = np.isfinite(sig) & (sig < 1e3)
    ax.plot(sig[fin], d["err_u"][fin], color=S.UNCOND, lw=1.4, label="E||D_u - x0||² (data)")
    ax.plot(sig[fin], d["err_c"][fin], color=S.COND, lw=1.4, label="E||D_c - x0||² (data)")
    ax.plot(sig[fin], d["c_datamean"][fin], color=S.ACC, lw=1.2, ls="--", label="matching gap c(t)")
    if "hutchinson" in d:
        h = d["hutchinson"]
        for bi, col, lab in ((0, S.UNCOND, "tr Cov, uncond (network only)"), (1, S.COND, "tr Cov, cond (network only)")):
            ax.errorbar(h["sigma"], h["tr_mean"][:, bi], yerr=2 * h["tr_se"][:, bi], fmt="o", ms=3.5, color=col,
                        mec="white", mew=0.6, elinewidth=0.8, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("noise level σ = s / (1 - s)")
    ax.set_ylabel("per coordinate")
    ax.legend(frameon=False, loc="lower right")
    S.title(ax, "posterior trace and the matching gap")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig-posterior-trace.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
